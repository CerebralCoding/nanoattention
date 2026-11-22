/*
 * Adapted from mlx-lm's Inkling 2 relative attention kernel
 * (mlx_lm/models/inkling2/kernels.py, _RELATIVE_ATTENTION_SOURCE, at
 * 0c4c99192230 of jundot/mlx-lm). Copyright © 2026 Apple Inc. (MIT license).
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

// ---- batched relative attention in 16-key tiles, off the 16-bit mirrors --------
// Append after inkling_long.metal. The body is the reference kernel's: 32 queries
// share each staged 16-key tile, FP16 fragments, FP32 online normalization in
// log2 units. Changed here, and only here: the queries, the distance bias and the
// sink are read from the arena and the result is written to it (float32 words
// holding bfloat16 values, or with MK_BIAS16 the bias as 16-bit words); K and V are read from this engine's position-major
// 16-bit mirrors (inkling_long.metal), a key's slot being its position modulo
// the cache's capacity; heads, relative extent and window are run-time values.
// On an unwindowed cache it computes mk_relative_attention_prefill's bits; on a
// sliding window its tiles are aligned to position 0 instead of the window's
// first key, so the sums are associated differently.
#define T bfloat
kernel void mk_relative_prefill_tiles(device float *arena [[buffer(0)]], device const bfloat *k [[buffer(1)]],
                                      device const bfloat *v [[buffer(2)]], constant MK &p [[buffer(8)]],
                                      uint tid_ [[thread_index_in_threadgroup]],
                                      uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
                                      uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
                                      uint group_ [[threadgroup_position_in_grid]]) {
    constexpr int D = 128, BQ = 32, TG = 128;
    const int HQ = p.heads, HKV = p.kv_heads, R = p.m, WINDOW = p.window;
    // The reference's launch geometry and inputs, in this engine's terms.
    const uint3 thread_position_in_threadgroup = uint3(tid_, 0, 0);
    const uint3 threadgroup_position_in_grid = uint3(group_ / HQ, group_ % HQ, 0);
    const int sizes[2] = { int(p.rows), int(p.pos + p.row0 + p.rows) }; // query rows, keys so far
    const int key_start[1] = { 0 };
    const float scale = p.scale;
    const size_t k_strides[3] = { 0, size_t(D), size_t(HKV) * D };
    const size_t v_strides[3] = { 0, size_t(D), size_t(HKV) * D };

    constexpr float LOG2E = 1.4426950408889634f;
    constexpr int BK = 16;
    constexpr int LDK = BK + 8;
    constexpr int LDV = D + 8;
    const int L = sizes[0];
    const int S = sizes[1];
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const int batch = threadgroup_position_in_grid.z;
    const int head = threadgroup_position_in_grid.y;
    const int kv_head = head / (HQ / HKV);
    const int q0 = threadgroup_position_in_grid.x * BQ;
    const int start = key_start[batch];
    const float qscale = scale * LOG2E;

    // The lane holds row fm and columns fn, fn + 1 of each 8x8 fragment.
    const short fm = ((lane / 4) & 4) + (lane / 2) % 4;
    const short fn = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
    const int row = q0 + int(sg) * 8 + fm;
    const bool row_ok = row < L;
    const int diag = S - L + row;

    // q and k are clipped and v has unit RMS, so half holds them.
    threadgroup half Ks[D * LDK];
    threadgroup half Vs[BK * LDV];

    simdgroup_half8x8 q_frag[D / 8];
    for (int db = 0; db < D / 8; ++db) {
        half2 qv = half2(0.0h);
        if (row_ok) {
            const device float* qp = arena + p.x + (p.row0 + row) * p.xs + head * D + db * 8 + fn;
            qv = half2(qp[0], qp[1]);
        }
        q_frag[db].thread_elements()[0] = qv.x;
        q_frag[db].thread_elements()[1] = qv.y;
    }
    simdgroup_float8x8 acc[D / 8];
    for (int db = 0; db < D / 8; ++db) {
        acc[db] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    float row_max = row_ok ? arena[p.s + (p.row0 + row) * p.every + head] * LOG2E : 0.0f;
    float row_sum = 1.0f;
    const int bias_row = batch * L + (row_ok ? row : 0);
    const device float* bias = arena + p.z + size_t(p.row0 + bias_row) * p.zs + head * R;
    // MK_BIAS16: the row's bias as 16-bit words (mk_grouped_project_bf16x4_pack).
    const bool bias_packed = (p.flags & MK_BIAS16) != 0;
    const device bfloat* bias16 = reinterpret_cast<const device bfloat*>(arena + p.z + size_t(p.row0 + bias_row) * p.zs) + head * R;

    const int first_diag = S - L + q0;
    const int last_diag = S - L + min(q0 + BQ, L) - 1;
    int first_key = WINDOW > 0 ? max(0, first_diag - WINDOW + 1) : 0;
    first_key = max(first_key, start);
    const device T* k_head = k + batch * k_strides[0] + kv_head * k_strides[1];
    const device T* v_head = v + batch * v_strides[0] + kv_head * v_strides[1];

    // The slot of a tile's first key, carried from tile to tile: a division only
    // where the ring wraps instead of one for every staged vector.
    const uint capacity = p.capacity;
    uint slot0 = uint(first_key / BK * BK) % capacity;
    for (int k0 = first_key / BK * BK; k0 <= last_diag; k0 += BK, slot0 += BK) {
        if (slot0 >= capacity) slot0 %= capacity;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int c = tid; c < BK * D / 8; c += TG) {
            const int r = c / (D / 8);
            const int col = (c % (D / 8)) * 8;
            const int key = k0 + r;
            vec<T, 4> kv[2] = {vec<T, 4>(0), vec<T, 4>(0)};
            vec<T, 4> vv[2] = {vec<T, 4>(0), vec<T, 4>(0)};
            if (key < S) {
                uint slot = slot0 + uint(r);
                if (slot >= capacity) slot %= capacity;
                auto kp = (const device vec<T, 4>*)(k_head + size_t(slot) * k_strides[2] + col);
                auto vp = (const device vec<T, 4>*)(v_head + size_t(slot) * v_strides[2] + col);
                kv[0] = kp[0];
                kv[1] = kp[1];
                vv[0] = vp[0];
                vv[1] = vp[1];
            }
            for (int j = 0; j < 8; ++j) {
                Ks[(col + j) * LDK + r] = half(float(kv[j / 4][j % 4]));
                Vs[r * LDV + col + j] = half(float(vv[j / 4][j % 4]));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (q0 + int(sg) * 8 >= L) {
            continue;
        }

        simdgroup_float8x8 scores[BK / 8];
        for (int kb = 0; kb < BK / 8; ++kb) {
            scores[kb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int db = 0; db < D / 8; ++db) {
                simdgroup_half8x8 k_frag;
                simdgroup_load(k_frag, Ks + db * 8 * LDK + kb * 8, LDK);
                simdgroup_multiply_accumulate(scores[kb], q_frag[db], k_frag, scores[kb]);
            }
        }

        float tile_max = -INFINITY;
        for (int kb = 0; kb < BK / 8; ++kb) {
            for (int e = 0; e < 2; ++e) {
                const int key = k0 + kb * 8 + fn + e;
                const int d = diag - key;
                const bool visible = row_ok && d >= 0 && key >= start
                    && (WINDOW == 0 || d < WINDOW);
                float s = -INFINITY;
                if (visible) {
                    s = scores[kb].thread_elements()[e] * qscale;
                    if (d < R) {
                        float b;
                        if (bias_packed) b = float(bias16[d]); else b = bias[d];
                        s += b * LOG2E;
                    }
                }
                scores[kb].thread_elements()[e] = s;
                tile_max = max(tile_max, s);
            }
        }
        // The 4 lanes of a row are lane ^ 1 and lane ^ 8 apart.
        tile_max = max(tile_max, simd_shuffle_xor(tile_max, 1));
        tile_max = max(tile_max, simd_shuffle_xor(tile_max, 8));
        const float new_max = max(row_max, tile_max);
        const float factor = fast::exp2(row_max - new_max);
        float tile_sum = 0.0f;
        simdgroup_half8x8 probs[BK / 8];
        for (int kb = 0; kb < BK / 8; ++kb) {
            for (int e = 0; e < 2; ++e) {
                const float p = fast::exp2(scores[kb].thread_elements()[e] - new_max);
                probs[kb].thread_elements()[e] = half(p);
                tile_sum += p;
            }
        }
        tile_sum += simd_shuffle_xor(tile_sum, 1);
        tile_sum += simd_shuffle_xor(tile_sum, 8);
        row_sum = row_sum * factor + tile_sum;
        row_max = new_max;
        for (int db = 0; db < D / 8; ++db) {
            acc[db].thread_elements()[0] *= factor;
            acc[db].thread_elements()[1] *= factor;
            for (int kb = 0; kb < BK / 8; ++kb) {
                simdgroup_half8x8 v_frag;
                simdgroup_load(v_frag, Vs + kb * 8 * LDV + db * 8, LDV);
                simdgroup_multiply_accumulate(acc[db], probs[kb], v_frag, acc[db]);
            }
        }
    }

    if (row_ok) {
        device float* op = arena + p.y + (p.row0 + row) * p.ys + head * D;
        for (int db = 0; db < D / 8; ++db) {
            op[db * 8 + fn] = float(acc[db].thread_elements()[0] / row_sum);
            op[db * 8 + fn + 1] = float(acc[db].thread_elements()[1] / row_sum);
        }
    }
}
#undef T
