// Inkling 2 Mini optimized decode and prefill kernels; embedded after inkling_tiles.metal.
//
// Upstream attribution: six optimization designs (constant-cache FP8/FP4 LUTs,
// compact single-TG MoE tiling and BF16 MoE intermediates, fused QKVR prep,
// inline single-row relative bias, and fused controller merge+norm) are adapted
// from qwegnai/qwegn by @Andrei-cloud / QwegnAI (Apache-2.0; see
// THIRD_PARTY_NOTICES.md and licenses/qwegn-APACHE-2.0.txt) and implemented
// natively for nano's Zig/Metal IR and Lean-checked integer helpers.

// Decode controller up-projections + message merge:
// 1 SIMD computes 4 output channels of forget_weight (buffer 0, p.w), gain_weight (buffer 1, p.s),
// and merges hidden (p.y) + message (p.z) into out (p.every, stride p.pad_k).
kernel void mk_bf16_controller_merge(device const uchar *wf [[buffer(0)]], device const uchar *wg [[buffer(1)]],
                                     device float *arena [[buffer(2)]], constant MK &p [[buffer(8)]],
                                     uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    const uint simd_id = gid / 32u;
    const uint tiles = (p.m + 3u) / 4u;
    const uint r = simd_id / tiles;
    const uint row = (simd_id % tiles) * 4u;
    if (r >= p.rows || row >= p.m) return;
    const uint row_idx = p.row0 + r;
    const device float *x = arena + p.x + row_idx * p.xs;
    const device ushort *f_w = (const device ushort *)(wf + p.w) + ulong(row) * p.n;
    const device ushort *g_w = (const device ushort *)(wg + p.s) + ulong(row) * p.n;
    float acc_f[4] = {};
    float acc_g[4] = {};
    if (p.n == 64u) {
        const float x0 = x[lane];
        const float x1 = x[lane + 32u];
        #pragma unroll
        for (uint j = 0u; j < 4u; ++j) {
            const float fw0 = as_type<float>(uint(f_w[ulong(j) * 64u + lane]) << 16);
            const float fw1 = as_type<float>(uint(f_w[ulong(j) * 64u + lane + 32u]) << 16);
            const float gw0 = as_type<float>(uint(g_w[ulong(j) * 64u + lane]) << 16);
            const float gw1 = as_type<float>(uint(g_w[ulong(j) * 64u + lane + 32u]) << 16);
            acc_f[j] = fw0 * x0 + fw1 * x1;
            acc_g[j] = gw0 * x0 + gw1 * x1;
        }
    } else {
        for (uint k = lane; k < p.n; k += 32u) {
            const float xv = x[k];
            #pragma unroll
            for (uint j = 0u; j < 4u; ++j) {
                if (row + j < p.m) {
                    const float fw = as_type<float>(uint(f_w[ulong(j) * p.n + k]) << 16);
                    const float gw = as_type<float>(uint(g_w[ulong(j) * p.n + k]) << 16);
                    acc_f[j] += fw * xv;
                    acc_g[j] += gw * xv;
                }
            }
        }
    }
    float sum_f[4], sum_g[4];
    #pragma unroll
    for (uint j = 0u; j < 4u; ++j) {
        sum_f[j] = simd_sum(acc_f[j]);
        sum_g[j] = simd_sum(acc_g[j]);
    }
    if (lane < 4u && row + lane < p.m) {
        const float forget = tanh(sum_f[lane]);
        const float gain = tanh(sum_g[lane]);
        const float hid = arena[p.y + row_idx * p.ys + row + lane];
        const float msg = arena[p.z + row_idx * p.zs + row + lane];
        float out = (1.0f - forget) * hid + gain * msg;
        if (p.flags & MK_ROUND_BF16) out = mk_bf16(out);
        arena[p.every + row_idx * p.pad_k + row + lane] = out;
    }
}

// Multi-projection BF16 decode using Lean-proved MultiProjExpr tile mapping (up to 6 outputs).
// x.kind bits 0..5 select MK_ROUND_OUT per projection.
template <bool Wide>
inline void mk_multi_bf16_impl(device const uchar *w, device float *arena, constant MKX &x, uint gid, uint lane) {
    constant MK &p = x.p;
    const uint c0 = x.mat[0].m >> 2;
    const uint p1 = c0;
    const uint p2 = p1 + (x.count > 1u ? (x.mat[1].m >> 2) : 0u);
    const uint p3 = p2 + (x.count > 2u ? (x.mat[2].m >> 2) : 0u);
    const uint p4 = p3 + (x.count > 3u ? (x.mat[3].m >> 2) : 0u);
    const uint p5 = p4 + (x.count > 4u ? (x.mat[4].m >> 2) : 0u);
    const uint total = p5 + (x.count > 5u ? (x.mat[5].m >> 2) : 0u);
    const uint simd = gid / 32u;
    if (simd >= total) return;
    const uint which = inkling_multi_proj_index(simd, p1, p2, p3, p4, p5);
    const uint loc = inkling_multi_proj_local_tile(simd, p1, p2, p3, p4, p5);
    constant MKMat &mat = x.mat[which];
    const uint row = loc * 4u;
    const uint stride = p.n / 4u;
    device const ushort4 *weights = (device const ushort4 *)(w + mat.w) + ulong(row) * stride;
    device const float4 *xp = (device const float4 *)(arena + p.x + p.row0 * p.xs);
    float partial[4] = {};
    if constexpr (Wide) {
        const uint wstride = p.n / 8u;
        device const uint4 *w8 = (device const uint4 *)(w + mat.w) + ulong(row) * wstride;
        for (uint k = lane; k < wstride; k += 32u) {
            const float4 a0 = xp[2u * k];
            const float4 a1 = xp[2u * k + 1u];
            #pragma unroll
            for (uint j = 0u; j < 4u; ++j) {
                const uint4 raw = w8[ulong(j) * wstride + k];
                const ushort4 u0(ushort(raw.x), ushort(raw.x >> 16), ushort(raw.y), ushort(raw.y >> 16));
                const ushort4 u1(ushort(raw.z), ushort(raw.z >> 16), ushort(raw.w), ushort(raw.w >> 16));
                const float4 b0 = as_type<float4>(uint4(u0) << 16);
                const float4 b1 = as_type<float4>(uint4(u1) << 16);
                partial[j] = fma(a0.x, b0.x, fma(a0.y, b0.y, fma(a0.z, b0.z, fma(a0.w, b0.w, partial[j]))));
                partial[j] = fma(a1.x, b1.x, fma(a1.y, b1.y, fma(a1.z, b1.z, fma(a1.w, b1.w, partial[j]))));
            }
        }
    } else {
        #pragma unroll 4
        for (uint k = lane; k < stride; k += 32u) {
            const float4 a = xp[k];
            #pragma unroll
            for (uint j = 0u; j < 4u; ++j) {
                const float4 b = as_type<float4>(uint4(weights[ulong(j) * stride + k]) << 16);
                for (uint i = 0u; i < 4u; ++i) partial[j] = fma(a[i], b[i], partial[j]);
            }
        }
    }
    const bool round_out = ((x.kind >> which) & 1u) != 0u;
    #pragma unroll
    for (uint j = 0u; j < 4u; ++j) {
        const float sum = simd_sum(partial[j]);
        if (lane == 0u) arena[mat.y + p.row0 * mat.ys + row + j] = round_out ? mk_bf16(sum) : sum;
    }
}

kernel void mk_multi_bf16_4x1_vec(device const uchar *w [[buffer(0)]], device float *arena [[buffer(1)]],
                                  constant MKX &x [[buffer(8)]], uint gid [[thread_position_in_grid]],
                                  uint lane [[thread_index_in_simdgroup]]) {
    mk_multi_bf16_impl<false>(w, arena, x, gid, lane);
}

kernel void mk_multi_bf16_4x1_vec8(device const uchar *w [[buffer(0)]], device float *arena [[buffer(1)]],
                                   constant MKX &x [[buffer(8)]], uint gid [[thread_position_in_grid]],
                                   uint lane [[thread_index_in_simdgroup]]) {
    mk_multi_bf16_impl<true>(w, arena, x, gid, lane);
}

// Wide 128-bit single BF16 decode matvec (`n % 256 == 0`, `w.offset % 16 == 0`, `p.x % 4 == 0`).
kernel void mk_bf16_linear_4x1_vec8(device const uchar *w [[buffer(0)]], device float *arena [[buffer(1)]],
                                    constant MK &p [[buffer(8)]], uint gid [[thread_position_in_grid]],
                                    uint lane [[thread_index_in_simdgroup]]) {
    const uint row = (gid / 32u) * 4u;
    if (row >= p.m) return;
    const uint wstride = p.n / 8u;
    device const uint4 *weights = (device const uint4 *)(w + p.w) + ulong(row) * wstride;
    device const float4 *x = (device const float4 *)(arena + p.x + p.row0 * p.xs);
    float partial[4] = {};
    for (uint k = lane; k < wstride; k += 32u) {
        const float4 a0 = x[2u * k];
        const float4 a1 = x[2u * k + 1u];
        #pragma unroll
        for (uint j = 0u; j < 4u; ++j) {
            const uint4 raw = weights[ulong(j) * wstride + k];
            const ushort4 u0(ushort(raw.x), ushort(raw.x >> 16), ushort(raw.y), ushort(raw.y >> 16));
            const ushort4 u1(ushort(raw.z), ushort(raw.z >> 16), ushort(raw.w), ushort(raw.w >> 16));
            const float4 b0 = as_type<float4>(uint4(u0) << 16);
            const float4 b1 = as_type<float4>(uint4(u1) << 16);
            partial[j] = fma(a0.x, b0.x, fma(a0.y, b0.y, fma(a0.z, b0.z, fma(a0.w, b0.w, partial[j]))));
            partial[j] = fma(a1.x, b1.x, fma(a1.y, b1.y, fma(a1.z, b1.z, fma(a1.w, b1.w, partial[j]))));
        }
    }
    #pragma unroll
    for (uint j = 0u; j < 4u; ++j) {
        const float sum = simd_sum(partial[j]);
        if (lane == 0u) arena[p.y + p.row0 * p.ys + row + j] = (p.flags & MK_ROUND_OUT) ? mk_bf16(sum) : sum;
    }
}

// Split-K small-M BF16 decode: INKLING_SPLITK_SIMDS (8) SIMDs = 256 threads per output row.
kernel void mk_bf16_linear_splitk(device const uchar *w [[buffer(0)]], device float *arena [[buffer(1)]],
                                  constant MK &p [[buffer(8)]], uint row [[threadgroup_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]],
                                  uint simd [[simdgroup_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]]) {
    if (row >= p.m) return;
    const uint stride = p.n / 4u;
    device const ushort4 *weights = (device const ushort4 *)(w + p.w) + ulong(row) * stride;
    device const float4 *x = (device const float4 *)(arena + p.x + p.row0 * p.xs);
    float acc = 0.0f;
    for (uint k = tid; k < stride; k += INKLING_SPLITK_THREADS) {
        const float4 a = x[k];
        const float4 b = as_type<float4>(uint4(weights[k]) << 16);
        acc = fma(a.w, b.w, fma(a.z, b.z, fma(a.y, b.y, fma(a.x, b.x, acc))));
    }
    acc = simd_sum(acc);
    threadgroup float partials[INKLING_SPLITK_SIMDS];
    if (lane == 0u) partials[simd] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0u) {
        const float total = simd_sum(lane < INKLING_SPLITK_SIMDS ? partials[lane] : 0.0f);
        if (lane == 0u) arena[p.y + p.row0 * p.ys + row] = (p.flags & MK_ROUND_OUT) ? mk_bf16(total) : total;
    }
}

// Fused head linear + BF16 round + tanh softcap for unpadded vocab `p.m`.
template <bool Wide>
inline void mk_bf16_softcap_impl(device const uchar *w, device float *arena, constant MK &p, uint gid, uint lane) {
    const uint row = (gid / 32u) * 4u;
    if (row >= p.m) return;
    const uint valid = min(4u, p.m - row);
    const uint stride = p.n / 4u;
    device const ushort4 *weights = (device const ushort4 *)(w + p.w) + ulong(row) * stride;
    device const float4 *x = (device const float4 *)(arena + p.x + p.row0 * p.xs);
    float partial[4] = {};
    if constexpr (Wide) {
        const uint wstride = p.n / 8u;
        device const uint4 *w8 = (device const uint4 *)(w + p.w) + ulong(row) * wstride;
        for (uint k = lane; k < wstride; k += 32u) {
            const float4 a0 = x[2u * k];
            const float4 a1 = x[2u * k + 1u];
            #pragma unroll
            for (uint j = 0u; j < 4u; ++j) {
                if (j < valid) {
                    const uint4 raw = w8[ulong(j) * wstride + k];
                    const ushort4 u0(ushort(raw.x), ushort(raw.x >> 16), ushort(raw.y), ushort(raw.y >> 16));
                    const ushort4 u1(ushort(raw.z), ushort(raw.z >> 16), ushort(raw.w), ushort(raw.w >> 16));
                    const float4 b0 = as_type<float4>(uint4(u0) << 16);
                    const float4 b1 = as_type<float4>(uint4(u1) << 16);
                    partial[j] = fma(a0.x, b0.x, fma(a0.y, b0.y, fma(a0.z, b0.z, fma(a0.w, b0.w, partial[j]))));
                    partial[j] = fma(a1.x, b1.x, fma(a1.y, b1.y, fma(a1.z, b1.z, fma(a1.w, b1.w, partial[j]))));
                }
            }
        }
    } else {
        #pragma unroll 4
        for (uint k = lane; k < stride; k += 32u) {
            const float4 a = x[k];
            #pragma unroll
            for (uint j = 0u; j < 4u; ++j) {
                if (j < valid) {
                    const float4 b = as_type<float4>(uint4(weights[ulong(j) * stride + k]) << 16);
                    for (uint i = 0u; i < 4u; ++i) partial[j] = fma(a[i], b[i], partial[j]);
                }
            }
        }
    }
    #pragma unroll
    for (uint j = 0u; j < 4u; ++j) {
        const float sum = simd_sum(partial[j]);
        if (lane == 0u && j < valid) {
            const float rounded = mk_bf16(sum);
            arena[p.y + p.row0 * p.ys + row + j] = tanh(rounded / p.scale) * p.scale;
        }
    }
}

kernel void mk_bf16_linear_4x1_softcap(device const uchar *w [[buffer(0)]], device float *arena [[buffer(1)]],
                                       constant MK &p [[buffer(8)]], uint gid [[thread_position_in_grid]],
                                       uint lane [[thread_index_in_simdgroup]]) {
    mk_bf16_softcap_impl<false>(w, arena, p, gid, lane);
}

kernel void mk_bf16_linear_4x1_softcap_vec8(device const uchar *w [[buffer(0)]], device float *arena [[buffer(1)]],
                                            constant MK &p [[buffer(8)]], uint gid [[thread_position_in_grid]],
                                            uint lane [[thread_index_in_simdgroup]]) {
    mk_bf16_softcap_impl<true>(w, arena, p, gid, lane);
}

// Single-weight-read dual head linear + BF16 round + tanh softcap for LoopCD logits/logits-adaptive.
// Reads BF16 unembed weights once across strong (`p.x`/`p.xs`) and weak (`p.z`/`p.zs`) hidden vectors.
template <bool Wide, bool Fixed>
inline void mk_bf16_softcap_loopcd_impl(device const uchar *w, device float *arena, constant MK &p, uint gid, uint lane) {
    const uint row = (gid / 32u) * 4u;
    if (row >= p.m) return;
    const uint valid = min(4u, p.m - row);
    const uint stride = p.n / 4u;
    device const float4 *xs = (device const float4 *)(arena + p.x + p.row0 * p.xs);
    device const float4 *xw = (device const float4 *)(arena + p.z + p.row0 * p.zs);
    float ps[4] = {};
    float pw[4] = {};
    if constexpr (Wide) {
        const uint wstride = p.n / 8u;
        device const uint4 *w8 = (device const uint4 *)(w + p.w) + ulong(row) * wstride;
        for (uint k = lane; k < wstride; k += 32u) {
            const float4 as0 = xs[2u * k], as1 = xs[2u * k + 1u];
            const float4 aw0 = xw[2u * k], aw1 = xw[2u * k + 1u];
            #pragma unroll
            for (uint j = 0u; j < 4u; ++j) {
                if (j < valid) {
                    const uint4 raw = w8[ulong(j) * wstride + k];
                    const ushort4 u0(ushort(raw.x), ushort(raw.x >> 16), ushort(raw.y), ushort(raw.y >> 16));
                    const ushort4 u1(ushort(raw.z), ushort(raw.z >> 16), ushort(raw.w), ushort(raw.w >> 16));
                    const float4 b0 = as_type<float4>(uint4(u0) << 16);
                    const float4 b1 = as_type<float4>(uint4(u1) << 16);
                    ps[j] = fma(as0.x, b0.x, fma(as0.y, b0.y, fma(as0.z, b0.z, fma(as0.w, b0.w, ps[j]))));
                    ps[j] = fma(as1.x, b1.x, fma(as1.y, b1.y, fma(as1.z, b1.z, fma(as1.w, b1.w, ps[j]))));
                    pw[j] = fma(aw0.x, b0.x, fma(aw0.y, b0.y, fma(aw0.z, b0.z, fma(aw0.w, b0.w, pw[j]))));
                    pw[j] = fma(aw1.x, b1.x, fma(aw1.y, b1.y, fma(aw1.z, b1.z, fma(aw1.w, b1.w, pw[j]))));
                }
            }
        }
    } else {
        device const ushort4 *weights = (device const ushort4 *)(w + p.w) + ulong(row) * stride;
        #pragma unroll 4
        for (uint k = lane; k < stride; k += 32u) {
            const float4 as = xs[k];
            const float4 aw = xw[k];
            #pragma unroll
            for (uint j = 0u; j < 4u; ++j) {
                if (j < valid) {
                    const float4 b = as_type<float4>(uint4(weights[ulong(j) * stride + k]) << 16);
                    for (uint i = 0u; i < 4u; ++i) {
                        ps[j] = fma(as[i], b[i], ps[j]);
                        pw[j] = fma(aw[i], b[i], pw[j]);
                    }
                }
            }
        }
    }
    #pragma unroll
    for (uint j = 0u; j < 4u; ++j) {
        const float sum_s = simd_sum(ps[j]);
        const float sum_w = simd_sum(pw[j]);
        if (lane == 0u && j < valid) {
            const float ss = tanh(mk_bf16(sum_s) / p.scale) * p.scale;
            const float sw = tanh(mk_bf16(sum_w) / p.scale) * p.scale;
            if constexpr (Fixed) {
                arena[p.y + p.row0 * p.ys + row + j] = (p.eps == 0.0f) ? ss : (ss + p.eps * (ss - sw));
            } else {
                arena[p.y + p.row0 * p.ys + row + j] = ss;
                arena[p.every + p.row0 * p.pad_k + row + j] = sw;
            }
        }
    }
}

kernel void mk_bf16_linear_4x2_softcap_loopcd(device const uchar *w [[buffer(0)]], device float *arena [[buffer(1)]],
                                              constant MK &p [[buffer(8)]], uint gid [[thread_position_in_grid]],
                                              uint lane [[thread_index_in_simdgroup]]) {
    mk_bf16_softcap_loopcd_impl<false, true>(w, arena, p, gid, lane);
}

kernel void mk_bf16_linear_4x2_softcap_loopcd_vec8(device const uchar *w [[buffer(0)]], device float *arena [[buffer(1)]],
                                                   constant MK &p [[buffer(8)]], uint gid [[thread_position_in_grid]],
                                                   uint lane [[thread_index_in_simdgroup]]) {
    mk_bf16_softcap_loopcd_impl<true, true>(w, arena, p, gid, lane);
}

kernel void mk_bf16_linear_4x2_softcap_dual(device const uchar *w [[buffer(0)]], device float *arena [[buffer(1)]],
                                            constant MK &p [[buffer(8)]], uint gid [[thread_position_in_grid]],
                                            uint lane [[thread_index_in_simdgroup]]) {
    mk_bf16_softcap_loopcd_impl<false, false>(w, arena, p, gid, lane);
}

kernel void mk_bf16_linear_4x2_softcap_dual_vec8(device const uchar *w [[buffer(0)]], device float *arena [[buffer(1)]],
                                                 constant MK &p [[buffer(8)]], uint gid [[thread_position_in_grid]],
                                                 uint lane [[thread_index_in_simdgroup]]) {
    mk_bf16_softcap_loopcd_impl<true, false>(w, arena, p, gid, lane);
}

// Coalesced 2-output-row-per-SIMD BF16 linear core for small batches (rows = 2..8).
// Exact FMA order matches 4x1_vec8 / 4x1_vec / 4x1_softcap bit-for-bit.
template <bool Wide, bool Softcap>
inline void mk_bf16_2xR_core(device const uchar *w, ulong w_off, device float *arena,
                             uint x_off, uint xs, uint y_off, uint ys,
                             uint row0, uint rows, uint m, uint n, uint row,
                             bool round_out, float softcap, uint lane) {
    float acc0[8] = {};
    float acc1[8] = {};
    const bool has1 = (row + 1u) < m;
    const uint xs4 = xs / 4u;
    device const float4 *xbase = (device const float4 *)(arena + x_off + row0 * xs);
    if constexpr (Wide) {
        const uint wstride = n / 8u;
        device const uint4 *w8 = (device const uint4 *)(w + w_off) + ulong(row) * wstride;
        for (uint k = lane; k < wstride; k += 32u) {
            const uint4 raw0 = w8[k];
            const ushort4 u00(ushort(raw0.x), ushort(raw0.x >> 16), ushort(raw0.y), ushort(raw0.y >> 16));
            const ushort4 u01(ushort(raw0.z), ushort(raw0.z >> 16), ushort(raw0.w), ushort(raw0.w >> 16));
            const float4 b00 = as_type<float4>(uint4(u00) << 16);
            const float4 b01 = as_type<float4>(uint4(u01) << 16);
            float4 b10 = float4(0.0f), b11 = float4(0.0f);
            if (has1) {
                const uint4 raw1 = w8[wstride + k];
                const ushort4 u10(ushort(raw1.x), ushort(raw1.x >> 16), ushort(raw1.y), ushort(raw1.y >> 16));
                const ushort4 u11(ushort(raw1.z), ushort(raw1.z >> 16), ushort(raw1.w), ushort(raw1.w >> 16));
                b10 = as_type<float4>(uint4(u10) << 16);
                b11 = as_type<float4>(uint4(u11) << 16);
            }
            #pragma unroll
            for (uint r = 0u; r < 8u; ++r) {
                if (r < rows) {
                    const float4 a0 = xbase[ulong(r) * xs4 + 2u * k];
                    const float4 a1 = xbase[ulong(r) * xs4 + 2u * k + 1u];
                    acc0[r] = fma(a0.x, b00.x, fma(a0.y, b00.y, fma(a0.z, b00.z, fma(a0.w, b00.w, acc0[r]))));
                    acc0[r] = fma(a1.x, b01.x, fma(a1.y, b01.y, fma(a1.z, b01.z, fma(a1.w, b01.w, acc0[r]))));
                    if (has1) {
                        acc1[r] = fma(a0.x, b10.x, fma(a0.y, b10.y, fma(a0.z, b10.z, fma(a0.w, b10.w, acc1[r]))));
                        acc1[r] = fma(a1.x, b11.x, fma(a1.y, b11.y, fma(a1.z, b11.z, fma(a1.w, b11.w, acc1[r]))));
                    }
                }
            }
        }
    } else {
        const uint stride = n / 4u;
        device const ushort4 *w4 = (device const ushort4 *)(w + w_off) + ulong(row) * stride;
        for (uint k = lane; k < stride; k += 32u) {
            const float4 b0 = as_type<float4>(uint4(w4[k]) << 16);
            const float4 b1 = has1 ? as_type<float4>(uint4(w4[stride + k]) << 16) : float4(0.0f);
            #pragma unroll
            for (uint r = 0u; r < 8u; ++r) {
                if (r < rows) {
                    const float4 a = xbase[ulong(r) * xs4 + k];
                    #pragma unroll
                    for (uint i = 0u; i < 4u; ++i) {
                        acc0[r] = fma(a[i], b0[i], acc0[r]);
                        if (has1) acc1[r] = fma(a[i], b1[i], acc1[r]);
                    }
                }
            }
        }
    }
    #pragma unroll
    for (uint r = 0u; r < 8u; ++r) {
        if (r < rows) {
            const float s0 = simd_sum(acc0[r]);
            const float s1 = has1 ? simd_sum(acc1[r]) : 0.0f;
            if (lane == 0u) {
                device float *dst = arena + y_off + ulong(row0 + r) * ys + row;
                if constexpr (Softcap) {
                    const float v0 = mk_bf16(s0);
                    dst[0] = tanh(v0 / softcap) * softcap;
                    if (has1) {
                        const float v1 = mk_bf16(s1);
                        dst[1] = tanh(v1 / softcap) * softcap;
                    }
                } else {
                    dst[0] = round_out ? mk_bf16(s0) : s0;
                    if (has1) dst[1] = round_out ? mk_bf16(s1) : s1;
                }
            }
        }
    }
}

kernel void mk_bf16_linear_2xR_vec(device const uchar *w [[buffer(0)]], device float *arena [[buffer(1)]],
                                   constant MK &p [[buffer(8)]], uint gid [[thread_position_in_grid]],
                                   uint lane [[thread_index_in_simdgroup]]) {
    const uint row = (gid / 32u) * 2u;
    if (row >= p.m) return;
    mk_bf16_2xR_core<false, false>(w, p.w, arena, p.x, p.xs, p.y, p.ys, p.row0, p.rows, p.m, p.n, row, (p.flags & MK_ROUND_OUT) != 0u, 0.0f, lane);
}

kernel void mk_bf16_linear_2xR_vec8(device const uchar *w [[buffer(0)]], device float *arena [[buffer(1)]],
                                    constant MK &p [[buffer(8)]], uint gid [[thread_position_in_grid]],
                                    uint lane [[thread_index_in_simdgroup]]) {
    const uint row = (gid / 32u) * 2u;
    if (row >= p.m) return;
    mk_bf16_2xR_core<true, false>(w, p.w, arena, p.x, p.xs, p.y, p.ys, p.row0, p.rows, p.m, p.n, row, (p.flags & MK_ROUND_OUT) != 0u, 0.0f, lane);
}

kernel void mk_bf16_linear_2xR_softcap(device const uchar *w [[buffer(0)]], device float *arena [[buffer(1)]],
                                       constant MK &p [[buffer(8)]], uint gid [[thread_position_in_grid]],
                                       uint lane [[thread_index_in_simdgroup]]) {
    const uint row = (gid / 32u) * 2u;
    if (row >= p.m) return;
    mk_bf16_2xR_core<false, true>(w, p.w, arena, p.x, p.xs, p.y, p.ys, p.row0, p.rows, p.m, p.n, row, false, p.scale, lane);
}

kernel void mk_bf16_linear_2xR_softcap_vec8(device const uchar *w [[buffer(0)]], device float *arena [[buffer(1)]],
                                            constant MK &p [[buffer(8)]], uint gid [[thread_position_in_grid]],
                                            uint lane [[thread_index_in_simdgroup]]) {
    const uint row = (gid / 32u) * 2u;
    if (row >= p.m) return;
    mk_bf16_2xR_core<true, true>(w, p.w, arena, p.x, p.xs, p.y, p.ys, p.row0, p.rows, p.m, p.n, row, false, p.scale, lane);
}

template <bool Wide>
inline void mk_multi_bf16_2xR_impl(device const uchar *w, device float *arena, constant MKX &x, uint gid, uint lane) {
    constant MK &p = x.p;
    const uint p1 = x.mat[0].m >> 1;
    const uint p2 = p1 + (x.count > 1u ? (x.mat[1].m >> 1) : 0u);
    const uint p3 = p2 + (x.count > 2u ? (x.mat[2].m >> 1) : 0u);
    const uint p4 = p3 + (x.count > 3u ? (x.mat[3].m >> 1) : 0u);
    const uint p5 = p4 + (x.count > 4u ? (x.mat[4].m >> 1) : 0u);
    const uint total = p5 + (x.count > 5u ? (x.mat[5].m >> 1) : 0u);
    const uint simd = gid / 32u;
    if (simd >= total) return;
    const uint which = inkling_multi_proj_index(simd, p1, p2, p3, p4, p5);
    const uint loc = inkling_multi_proj_local_tile(simd, p1, p2, p3, p4, p5);
    constant MKMat &mat = x.mat[which];
    const uint row = loc * 2u;
    const bool round_out = ((x.kind >> which) & 1u) != 0u;
    mk_bf16_2xR_core<Wide, false>(w, mat.w, arena, p.x, p.xs, mat.y, mat.ys, p.row0, p.rows, mat.m, p.n, row, round_out, 0.0f, lane);
}

kernel void mk_multi_bf16_2xR_vec(device const uchar *w [[buffer(0)]], device float *arena [[buffer(1)]],
                                  constant MKX &x [[buffer(8)]], uint gid [[thread_position_in_grid]],
                                  uint lane [[thread_index_in_simdgroup]]) {
    mk_multi_bf16_2xR_impl<false>(w, arena, x, gid, lane);
}

kernel void mk_multi_bf16_2xR_vec8(device const uchar *w [[buffer(0)]], device float *arena [[buffer(1)]],
                                   constant MKX &x [[buffer(8)]], uint gid [[thread_position_in_grid]],
                                   uint lane [[thread_index_in_simdgroup]]) {
    mk_multi_bf16_2xR_impl<true>(w, arena, x, gid, lane);
}

// Low-register 1-output-row-per-SIMD BF16 linear core for small batches (rows = 2..8).
// Preserves exact 4x1_vec8 FMA order bit-for-bit with 8x fewer accumulators/row than 4xR_tg and 4x higher grid parallelism.
template <bool Full8>
inline void mk_bf16_1xR_vec8_core(device const uchar *w, ulong w_off, device float *arena,
                                  uint x_off, uint xs, uint y_off, uint ys,
                                  uint row0, uint rows, uint n, uint row,
                                  bool round_out, uint lane) {
    float acc[8] = {};
    const uint xs4 = xs / 4u;
    const uint wstride = n / 8u;
    device const uint4 *w8 = (device const uint4 *)(w + w_off) + ulong(row) * wstride;
    device const float4 *xbase = (device const float4 *)(arena + x_off + ulong(row0) * xs);
    for (uint k = lane; k < wstride; k += 32u) {
        const uint4 raw = w8[k];
        const ushort4 u0(ushort(raw.x), ushort(raw.x >> 16), ushort(raw.y), ushort(raw.y >> 16));
        const ushort4 u1(ushort(raw.z), ushort(raw.z >> 16), ushort(raw.w), ushort(raw.w >> 16));
        const float4 b0 = as_type<float4>(uint4(u0) << 16);
        const float4 b1 = as_type<float4>(uint4(u1) << 16);
        #pragma unroll
        for (uint r = 0u; r < 8u; ++r) {
            if (Full8 || r < rows) {
                const float4 a0 = xbase[ulong(r) * xs4 + 2u * k];
                const float4 a1 = xbase[ulong(r) * xs4 + 2u * k + 1u];
                acc[r] = fma(a0.x, b0.x, fma(a0.y, b0.y, fma(a0.z, b0.z, fma(a0.w, b0.w, acc[r]))));
                acc[r] = fma(a1.x, b1.x, fma(a1.y, b1.y, fma(a1.z, b1.z, fma(a1.w, b1.w, acc[r]))));
            }
        }
    }
    #pragma unroll
    for (uint r = 0u; r < 8u; ++r) {
        if (Full8 || r < rows) {
            const float s = simd_sum(acc[r]);
            if (lane == 0u) arena[y_off + ulong(row0 + r) * ys + row] = round_out ? mk_bf16(s) : s;
        }
    }
}

kernel void mk_bf16_linear_1xR_vec8(device const uchar *w [[buffer(0)]], device float *arena [[buffer(1)]],
                                    constant MK &p [[buffer(8)]], uint gid [[thread_position_in_grid]],
                                    uint lane [[thread_index_in_simdgroup]]) {
    const uint row = gid / 32u;
    if (row >= p.m) return;
    if (p.rows == 8u) {
        mk_bf16_1xR_vec8_core<true>(w, p.w, arena, p.x, p.xs, p.y, p.ys, p.row0, 8u, p.n, row, (p.flags & MK_ROUND_OUT) != 0u, lane);
    } else {
        mk_bf16_1xR_vec8_core<false>(w, p.w, arena, p.x, p.xs, p.y, p.ys, p.row0, p.rows, p.n, row, (p.flags & MK_ROUND_OUT) != 0u, lane);
    }
}

kernel void mk_multi_bf16_1xR_vec8(device const uchar *w [[buffer(0)]], device float *arena [[buffer(1)]],
                                   constant MKX &x [[buffer(8)]], uint gid [[thread_position_in_grid]],
                                   uint lane [[thread_index_in_simdgroup]]) {
    constant MK &p = x.p;
    const uint p1 = x.mat[0].m;
    const uint p2 = p1 + (x.count > 1u ? x.mat[1].m : 0u);
    const uint p3 = p2 + (x.count > 2u ? x.mat[2].m : 0u);
    const uint p4 = p3 + (x.count > 3u ? x.mat[3].m : 0u);
    const uint p5 = p4 + (x.count > 4u ? x.mat[4].m : 0u);
    const uint total = p5 + (x.count > 5u ? x.mat[5].m : 0u);
    const uint simd_id = gid / 32u;
    if (simd_id >= total) return;
    const uint which = inkling_multi_proj_index(simd_id, p1, p2, p3, p4, p5);
    const uint row = inkling_multi_proj_local_tile(simd_id, p1, p2, p3, p4, p5);
    constant MKMat &mat = x.mat[which];
    const bool round_out = ((x.kind >> which) & 1u) != 0u;
    if (p.rows == 8u) {
        mk_bf16_1xR_vec8_core<true>(w, mat.w, arena, p.x, p.xs, mat.y, mat.ys, p.row0, 8u, p.n, row, round_out, lane);
    } else {
        mk_bf16_1xR_vec8_core<false>(w, mat.w, arena, p.x, p.xs, mat.y, mat.ys, p.row0, p.rows, p.n, row, round_out, lane);
    }
}

// 1-output-row-per-TG Split-K small-M BF16 linear (8 SIMDs = 256 threads per output row, rows <= 8).
kernel void mk_bf16_linear_splitk_rows(device const uchar *w [[buffer(0)]], device float *arena [[buffer(1)]],
                                       constant MK &p [[buffer(8)]], uint row [[threadgroup_position_in_grid]],
                                       uint tid [[thread_index_in_threadgroup]],
                                       uint simd [[simdgroup_index_in_threadgroup]],
                                       uint lane [[thread_index_in_simdgroup]]) {
    if (row >= p.m) return;
    const uint stride = p.n / 4u;
    const uint xs4 = p.xs / 4u;
    device const ushort4 *weights = (device const ushort4 *)(w + p.w) + ulong(row) * stride;
    device const float4 *xbase = (device const float4 *)(arena + p.x + ulong(p.row0) * p.xs);
    float acc[8] = {};
    if (p.rows == 8u) {
        for (uint k = tid; k < stride; k += INKLING_SPLITK_THREADS) {
            const float4 b = as_type<float4>(uint4(weights[k]) << 16);
            #pragma unroll
            for (uint r = 0u; r < 8u; ++r) {
                const float4 a = xbase[ulong(r) * xs4 + k];
                acc[r] = fma(a.w, b.w, fma(a.z, b.z, fma(a.y, b.y, fma(a.x, b.x, acc[r]))));
            }
        }
    } else {
        for (uint k = tid; k < stride; k += INKLING_SPLITK_THREADS) {
            const float4 b = as_type<float4>(uint4(weights[k]) << 16);
            #pragma unroll
            for (uint r = 0u; r < 8u; ++r) {
                if (r < p.rows) {
                    const float4 a = xbase[ulong(r) * xs4 + k];
                    acc[r] = fma(a.w, b.w, fma(a.z, b.z, fma(a.y, b.y, fma(a.x, b.x, acc[r]))));
                }
            }
        }
    }
    threadgroup float partials[8][INKLING_SPLITK_SIMDS];
    #pragma unroll
    for (uint r = 0u; r < 8u; ++r) {
        if (r < p.rows) {
            const float s = simd_sum(acc[r]);
            if (lane == 0u) partials[r][simd] = s;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd < p.rows) {
        const uint r = simd;
        const float total = simd_sum(lane < INKLING_SPLITK_SIMDS ? partials[r][lane] : 0.0f);
        if (lane == 0u) arena[p.y + ulong(p.row0 + r) * p.ys + row] = (p.flags & MK_ROUND_OUT) ? mk_bf16(total) : total;
    }
}

// ---- Step 3C kernels: depth attention, depth+norm fusions, rmsnorm_fast, add_norm, store_pack, conv, l2_clip, relbias ----
constant uint MK_INKLING_ROUND_RES = 64u;

inline float4 mk_fast_load4(device const uchar *base, ulong offset, uint dtype, uint idx4) {
    if (dtype == 0u) return ((device const float4 *)(base + offset))[idx4];
    const ushort4 h = ((device const ushort4 *)(base + offset))[idx4];
    if (dtype == 1u) return float4(as_type<half4>(h));
    return as_type<float4>(uint4(h) << 16);
}

// Register-resident depth attention (Mode 0: plain depth_attention, Mode 1: depth + rmsnorm,
// Mode 2: controller depth + input norm(bf16) + msg norm(bf16)). Supports p.n <= 4 * TG and p.heads <= 8.
template <uint TG, uint SIMDS, uint MODE>
inline void mk_depth_reg_impl(device float *arena, device const uchar *query, device const uchar *norm_w,
                              constant MK &p, uint tid, uint group, uint lane, uint simd,
                              threadgroup float (*partials)[8], threadgroup float *weights) {
    const uint row = p.row0 + group;
    const uint d4 = tid;
    const bool active = d4 * 4u < p.n;
    const float4 q = active ? *((const device float4 *)(query + p.w) + d4) : float4(0.0f);
    float4 vs[8];
    for (uint c = 0u; c < p.heads; ++c) {
        const float4 vc = active
            ? (c < p.m
                ? ((const device float4 *)(arena + p.x + row * p.xs + c * p.n))[d4]
                : ((const device float4 *)(arena + p.y + row * p.ys))[d4])
            : float4(0.0f);
        vs[c] = vc;
        const float s = simd_sum(dot(q, vc));
        if (lane == 0u) partials[simd][c] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0u) {
        float dot_c = 0.0f;
        if (lane < p.heads) {
            for (uint s = 0u; s < SIMDS; ++s) dot_c += partials[s][lane];
        }
        const float logit = lane < p.heads ? dot_c * p.scale : -INFINITY;
        const float m = simd_max(logit);
        const float ew = lane < p.heads ? metal::precise::exp(logit - m) : 0.0f;
        const float denom = simd_sum(ew);
        if (lane < p.heads) weights[lane] = ew / denom;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float4 val = float4(0.0f);
    for (uint c = 0u; c < p.heads; ++c) val = fma(float4(weights[c]), vs[c], val);

    if constexpr (MODE == 0u) {
        if (active) ((device float4 *)(arena + p.z + row * p.zs))[d4] = val;
        return;
    }
    if constexpr (MODE == 1u) {
        const float s = simd_sum(active ? dot(val, val) : 0.0f);
        if (lane == 0u) partials[simd][0] = s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd == 0u) {
            const float tot = simd_sum(lane < SIMDS ? partials[lane][0] : 0.0f);
            if (lane == 0u) weights[0] = metal::precise::rsqrt(tot / float(p.n) + p.eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {
            float4 outv = val * weights[0];
            if (p.flags & MK_ROUND_BF16) {
                for (uint j = 0u; j < 4u; ++j) outv[j] = mk_bf16(outv[j]);
            }
            if (p.flags & MK_HAS_WEIGHT) outv *= mk_fast_load4(norm_w, p.s, p.sdtype, d4);
            if (p.flags & MK_ROUND_OUT) {
                for (uint j = 0u; j < 4u; ++j) outv[j] = mk_bf16(outv[j]);
            }
            ((device float4 *)(arena + p.z + row * p.zs))[d4] = outv;
        }
        return;
    }
    if constexpr (MODE == 2u) {
        if (active) ((device float4 *)(arena + p.z + row * p.zs))[d4] = val;
        const float4 inp = active ? ((const device float4 *)(arena + p.every + row * p.pad_k))[d4] : float4(0.0f);
        const float s0 = simd_sum(active ? dot(inp, inp) : 0.0f);
        const float s1 = simd_sum(active ? dot(val, val) : 0.0f);
        if (lane == 0u) {
            partials[simd][0] = s0;
            partials[simd][1] = s1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd == 0u) {
            const float tot0 = simd_sum(lane < SIMDS ? partials[lane][0] : 0.0f);
            const float tot1 = simd_sum(lane < SIMDS ? partials[lane][1] : 0.0f);
            if (lane == 0u) {
                weights[0] = metal::precise::rsqrt(tot0 / float(p.n) + p.eps);
                weights[1] = metal::precise::rsqrt(tot1 / float(p.n) + p.eps);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {
            float4 out0 = inp * weights[0];
            float4 out1 = val * weights[1];
            if (p.flags & (MK_ROUND_BF16 | MK_ROUND_OUT)) {
                for (uint j = 0u; j < 4u; ++j) {
                    out0[j] = mk_bf16(out0[j]);
                    out1[j] = mk_bf16(out1[j]);
                }
            }
            ((device float4 *)(arena + p.rot + row * p.kind))[d4] = out0;
            ((device float4 *)(arena + p.rot + row * p.kind + p.n))[d4] = out1;
        }
    }
}

kernel void mk_depth_attention_reg(device float *arena [[buffer(0)]], device const uchar *query [[buffer(1)]],
                                   constant MK &p [[buffer(8)]], uint tid [[thread_index_in_threadgroup]],
                                   uint group [[threadgroup_position_in_grid]],
                                   uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partials[INKLING_DEPTH_REG_SIMDS][8];
    threadgroup float weights[8];
    mk_depth_reg_impl<INKLING_DEPTH_REG_THREADS, INKLING_DEPTH_REG_SIMDS, 0u>(arena, query, query, p, tid, group, lane, simd, partials, weights);
}

kernel void mk_depth_attention_vec(device float *arena [[buffer(0)]], device const uchar *query [[buffer(1)]],
                                   constant MK &p [[buffer(8)]], uint tid [[thread_index_in_threadgroup]],
                                   uint group [[threadgroup_position_in_grid]],
                                   uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partials[8][8];
    threadgroup float weights[8];
    mk_depth_reg_impl<256u, 8u, 0u>(arena, query, query, p, tid, group, lane, simd, partials, weights);
}

kernel void mk_depth_norm_2304(device float *arena [[buffer(0)]], device const uchar *query [[buffer(1)]],
                               device const uchar *norm_w [[buffer(2)]], constant MK &p [[buffer(8)]],
                               uint tid [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]],
                               uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partials[INKLING_DEPTH_REG_SIMDS][8];
    threadgroup float weights[8];
    mk_depth_reg_impl<INKLING_DEPTH_REG_THREADS, INKLING_DEPTH_REG_SIMDS, 1u>(arena, query, norm_w, p, tid, group, lane, simd, partials, weights);
}

kernel void mk_depth_norm_vec(device float *arena [[buffer(0)]], device const uchar *query [[buffer(1)]],
                              device const uchar *norm_w [[buffer(2)]], constant MK &p [[buffer(8)]],
                              uint tid [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]],
                              uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partials[8][8];
    threadgroup float weights[8];
    mk_depth_reg_impl<256u, 8u, 1u>(arena, query, norm_w, p, tid, group, lane, simd, partials, weights);
}

kernel void mk_controller_depth_norms_2304(device float *arena [[buffer(0)]], device const uchar *query [[buffer(1)]],
                                           constant MK &p [[buffer(8)]], uint tid [[thread_index_in_threadgroup]],
                                           uint group [[threadgroup_position_in_grid]],
                                           uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partials[INKLING_DEPTH_REG_SIMDS][8];
    threadgroup float weights[8];
    mk_depth_reg_impl<INKLING_DEPTH_REG_THREADS, INKLING_DEPTH_REG_SIMDS, 2u>(arena, query, query, p, tid, group, lane, simd, partials, weights);
}

kernel void mk_controller_depth_norms_vec(device float *arena [[buffer(0)]], device const uchar *query [[buffer(1)]],
                                          constant MK &p [[buffer(8)]], uint tid [[thread_index_in_threadgroup]],
                                          uint group [[threadgroup_position_in_grid]],
                                          uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partials[8][8];
    threadgroup float weights[8];
    mk_depth_reg_impl<256u, 8u, 2u>(arena, query, query, p, tid, group, lane, simd, partials, weights);
}

// Register-resident vectorized RMSNorm for n <= 4 * tg (single read, 2 barriers).
kernel void mk_rmsnorm_vec(device const uchar *w [[buffer(0)]], device float *arena [[buffer(1)]],
                           constant MK &p [[buffer(8)]], uint tgid [[threadgroup_position_in_grid]],
                           uint lid [[thread_position_in_threadgroup]], uint tg [[threads_per_threadgroup]],
                           uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[33];
    const uint r = tgid / p.heads, g = tgid % p.heads;
    if (r >= p.rows) return;
    const uint row = p.row0 + r;
    const bool active = lid * 4u < p.n;
    const device float4 *xp = (const device float4 *)(arena + p.x + row * p.xs + g * p.n);
    device float4 *yp = (device float4 *)(arena + p.y + row * p.ys + g * p.n);
    const float4 xv = active ? xp[lid] : float4(0.0f);
    const uint nsimds = tg / 32u;
    const float acc = simd_sum(active ? dot(xv, xv) : 0.0f);
    if (lane == 0u) partial[simd] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0u) {
        const float tot = simd_sum(lane < nsimds ? partial[lane] : 0.0f);
        if (lane == 0u) partial[32] = metal::precise::rsqrt(tot / float(p.n) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!active) return;
    float4 v = xv * partial[32];
    if (p.flags & MK_ROUND_BF16) {
        for (uint j = 0u; j < 4u; ++j) v[j] = mk_bf16(v[j]);
    }
    float4 res = (p.flags & MK_HAS_WEIGHT) ? mk_fast_load4(w, p.w, p.dtype, lid) * v : v;
    if (p.flags & MK_ROUND_OUT) {
        for (uint j = 0u; j < 4u; ++j) res[j] = mk_bf16(res[j]);
    }
    yp[lid] = res;
}

// Generic vectorized add + (optional bf16 round) + rmsnorm for both decode and prefill.
kernel void mk_add_norm_vec(device const uchar *w [[buffer(0)]], device float *arena [[buffer(1)]],
                            constant MK &p [[buffer(8)]], uint tgid [[threadgroup_position_in_grid]],
                            uint lid [[thread_position_in_threadgroup]], uint tg [[threads_per_threadgroup]],
                            uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[33];
    if (tgid >= p.rows) return;
    const uint row = p.row0 + tgid;
    const bool active = lid * 4u < p.n;
    const device float4 *ap = (const device float4 *)(arena + p.x + row * p.xs);
    const device float4 *bp = (const device float4 *)(arena + p.y + row * p.ys);
    device float4 *rp = (device float4 *)(arena + p.z + row * p.zs);
    device float4 *op = (device float4 *)(arena + p.every + row * p.pad_k);
    float4 rv = active ? (ap[lid] + bp[lid]) : float4(0.0f);
    if (active) {
        if (p.flags & MK_INKLING_ROUND_RES) {
            for (uint j = 0u; j < 4u; ++j) rv[j] = mk_bf16(rv[j]);
        }
        rp[lid] = rv;
    }
    const uint nsimds = tg / 32u;
    const float acc = simd_sum(active ? dot(rv, rv) : 0.0f);
    if (lane == 0u) partial[simd] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0u) {
        const float tot = simd_sum(lane < nsimds ? partial[lane] : 0.0f);
        if (lane == 0u) partial[32] = metal::precise::rsqrt(tot / float(p.n) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!active) return;
    float4 v = rv * partial[32];
    if (p.flags & MK_ROUND_BF16) {
        for (uint j = 0u; j < 4u; ++j) v[j] = mk_bf16(v[j]);
    }
    float4 res = (p.flags & MK_HAS_WEIGHT) ? mk_fast_load4(w, p.w, p.dtype, lid) * v : v;
    if (p.flags & MK_ROUND_OUT) {
        for (uint j = 0u; j < 4u; ++j) res[j] = mk_bf16(res[j]);
    }
    op[lid] = res;
}

// Fused KV float cache store + BF16 mirror pack (+ MK_MIRROR_PAD wrap copy).
kernel void mk_store_pack_bf16(device const float *arena [[buffer(0)]], device float *dst_f32 [[buffer(1)]],
                               device ushort *dst_u16 [[buffer(2)]], constant MK &p [[buffer(8)]],
                               uint i [[thread_position_in_grid]]) {
    const uint r = i / p.n, c = i % p.n;
    if (r >= p.rows) return;
    const uint row = p.row0 + r;
    const uint slot = (p.pos + row) % p.capacity;
    const float val = arena[p.x + row * p.xs + c];
    const ulong idx = ulong(slot) * p.n + c;
    dst_f32[idx] = val;
    const ushort h = ushort(as_type<uint>(val) >> 16);
    dst_u16[idx] = h;
    if (slot < 16u) dst_u16[ulong(p.capacity + slot) * p.n + c] = h;
}

// Fused single-row store + Inkling short conv using Lean-emitted inkling_ring_back.
kernel void mk_store_inkling_short_conv_ring(device float *ring [[buffer(0)]], device const uchar *w [[buffer(1)]],
                                             device float *arena [[buffer(2)]], constant MK &p [[buffer(8)]],
                                             uint c [[thread_position_in_grid]]) {
    if (c >= p.n) return;
    const uint row = p.row0;
    const uint position = p.pos + row;
    const uint slot = position % p.capacity;
    const float xval = arena[p.x + row * p.xs + c];
    ring[ulong(slot) * p.n + c] = xval;
    float acc = 0.0f;
    for (uint k = 0u; k < p.taps; ++k) {
        const uint back = p.taps - 1u - k;
        float value = back > position ? 0.0f
            : (back == 0u ? xval : ring[ulong(inkling_ring_back(slot, back, p.capacity)) * p.n + c]);
        if (p.dtype == 1u) value = float(half(value));
        else if (p.dtype == 2u) value = mk_bf16(value);
        acc += mk_load(w, p.w, p.dtype, ulong(c) * p.taps + k) * value;
    }
    if (p.dtype == 1u) acc = float(half(acc));
    else if (p.dtype == 2u) acc = mk_bf16(acc);
    float output = (acc + xval) * p.scale;
    if (p.flags & MK_ROUND_BF16) output = mk_bf16(output);
    arena[p.z + row * p.zs + c] = output;
}

// Single-row relative bias grouped projection: 1 query row x 8 distances per thread.
kernel void mk_grouped_project_bf16d8_row1(device float *arena [[buffer(0)]], device const uchar *w [[buffer(1)]],
                                           constant MK &p [[buffer(8)]], uint i [[thread_position_in_grid]]) {
    const uint blocks = (p.taps + 7u) / 8u;
    const uint groups = p.n / p.m;
    if (i >= groups * blocks) return;
    const uint row = p.row0;
    const uint h = i / blocks, d0 = (i % blocks) * 8u;
    const device float *x = arena + p.x + row * p.xs + h * 16u;
    const device ushort *weight = reinterpret_cast<const device ushort *>(w + p.w) + d0;
    float4 sum0 = 0.0f, sum1 = 0.0f;
    #pragma unroll
    for (uint k = 0u; k < 16u; ++k) {
        const float xi = x[k];
        const device packed_ushort4 *wp = reinterpret_cast<const device packed_ushort4 *>(weight + k * p.m);
        const float4 w0 = as_type<float4>(uint4(ushort4(wp[0])) << 16);
        const float4 w1 = as_type<float4>(uint4(ushort4(wp[1])) << 16);
        sum0 = fma(float4(xi), w0, sum0);
        sum1 = fma(float4(xi), w1, sum1);
    }
    device float *dst = arena + p.y + row * p.ys + h * p.m + d0;
    const bool round_bf = (p.flags & MK_ROUND_BF16) != 0u;
    for (uint j = 0u; j < 4u; ++j) {
        if (d0 + j < p.taps) dst[j] = round_bf ? mk_bf16(sum0[j]) : sum0[j];
    }
    for (uint j = 0u; j < 4u; ++j) {
        if (d0 + 4u + j < p.taps) dst[4u + j] = round_bf ? mk_bf16(sum1[j]) : sum1[j];
    }
}

// ---- Step 3D & Qwegn LUT kernels: fast NVFP4 & MXFP8 decode experts ----
static constant float mk_e4m3_f32_lut[256] = {
    0.0f, 0x1.0000000000000p-9f, 0x1.0000000000000p-8f, 0x1.8000000000000p-8f, 0x1.0000000000000p-7f, 0x1.4000000000000p-7f, 0x1.8000000000000p-7f, 0x1.c000000000000p-7f,
    0x1.0000000000000p-6f, 0x1.2000000000000p-6f, 0x1.4000000000000p-6f, 0x1.6000000000000p-6f, 0x1.8000000000000p-6f, 0x1.a000000000000p-6f, 0x1.c000000000000p-6f, 0x1.e000000000000p-6f,
    0x1.0000000000000p-5f, 0x1.2000000000000p-5f, 0x1.4000000000000p-5f, 0x1.6000000000000p-5f, 0x1.8000000000000p-5f, 0x1.a000000000000p-5f, 0x1.c000000000000p-5f, 0x1.e000000000000p-5f,
    0x1.0000000000000p-4f, 0x1.2000000000000p-4f, 0x1.4000000000000p-4f, 0x1.6000000000000p-4f, 0x1.8000000000000p-4f, 0x1.a000000000000p-4f, 0x1.c000000000000p-4f, 0x1.e000000000000p-4f,
    0x1.0000000000000p-3f, 0x1.2000000000000p-3f, 0x1.4000000000000p-3f, 0x1.6000000000000p-3f, 0x1.8000000000000p-3f, 0x1.a000000000000p-3f, 0x1.c000000000000p-3f, 0x1.e000000000000p-3f,
    0x1.0000000000000p-2f, 0x1.2000000000000p-2f, 0x1.4000000000000p-2f, 0x1.6000000000000p-2f, 0x1.8000000000000p-2f, 0x1.a000000000000p-2f, 0x1.c000000000000p-2f, 0x1.e000000000000p-2f,
    0x1.0000000000000p-1f, 0x1.2000000000000p-1f, 0x1.4000000000000p-1f, 0x1.6000000000000p-1f, 0x1.8000000000000p-1f, 0x1.a000000000000p-1f, 0x1.c000000000000p-1f, 0x1.e000000000000p-1f,
    0x1.0000000000000p+0f, 0x1.2000000000000p+0f, 0x1.4000000000000p+0f, 0x1.6000000000000p+0f, 0x1.8000000000000p+0f, 0x1.a000000000000p+0f, 0x1.c000000000000p+0f, 0x1.e000000000000p+0f,
    0x1.0000000000000p+1f, 0x1.2000000000000p+1f, 0x1.4000000000000p+1f, 0x1.6000000000000p+1f, 0x1.8000000000000p+1f, 0x1.a000000000000p+1f, 0x1.c000000000000p+1f, 0x1.e000000000000p+1f,
    0x1.0000000000000p+2f, 0x1.2000000000000p+2f, 0x1.4000000000000p+2f, 0x1.6000000000000p+2f, 0x1.8000000000000p+2f, 0x1.a000000000000p+2f, 0x1.c000000000000p+2f, 0x1.e000000000000p+2f,
    0x1.0000000000000p+3f, 0x1.2000000000000p+3f, 0x1.4000000000000p+3f, 0x1.6000000000000p+3f, 0x1.8000000000000p+3f, 0x1.a000000000000p+3f, 0x1.c000000000000p+3f, 0x1.e000000000000p+3f,
    0x1.0000000000000p+4f, 0x1.2000000000000p+4f, 0x1.4000000000000p+4f, 0x1.6000000000000p+4f, 0x1.8000000000000p+4f, 0x1.a000000000000p+4f, 0x1.c000000000000p+4f, 0x1.e000000000000p+4f,
    0x1.0000000000000p+5f, 0x1.2000000000000p+5f, 0x1.4000000000000p+5f, 0x1.6000000000000p+5f, 0x1.8000000000000p+5f, 0x1.a000000000000p+5f, 0x1.c000000000000p+5f, 0x1.e000000000000p+5f,
    0x1.0000000000000p+6f, 0x1.2000000000000p+6f, 0x1.4000000000000p+6f, 0x1.6000000000000p+6f, 0x1.8000000000000p+6f, 0x1.a000000000000p+6f, 0x1.c000000000000p+6f, 0x1.e000000000000p+6f,
    0x1.0000000000000p+7f, 0x1.2000000000000p+7f, 0x1.4000000000000p+7f, 0x1.6000000000000p+7f, 0x1.8000000000000p+7f, 0x1.a000000000000p+7f, 0x1.c000000000000p+7f, 0x1.e000000000000p+7f,
    0x1.0000000000000p+8f, 0x1.2000000000000p+8f, 0x1.4000000000000p+8f, 0x1.6000000000000p+8f, 0x1.8000000000000p+8f, 0x1.a000000000000p+8f, 0x1.c000000000000p+8f, NAN,
    -0.0f, -0x1.0000000000000p-9f, -0x1.0000000000000p-8f, -0x1.8000000000000p-8f, -0x1.0000000000000p-7f, -0x1.4000000000000p-7f, -0x1.8000000000000p-7f, -0x1.c000000000000p-7f,
    -0x1.0000000000000p-6f, -0x1.2000000000000p-6f, -0x1.4000000000000p-6f, -0x1.6000000000000p-6f, -0x1.8000000000000p-6f, -0x1.a000000000000p-6f, -0x1.c000000000000p-6f, -0x1.e000000000000p-6f,
    -0x1.0000000000000p-5f, -0x1.2000000000000p-5f, -0x1.4000000000000p-5f, -0x1.6000000000000p-5f, -0x1.8000000000000p-5f, -0x1.a000000000000p-5f, -0x1.c000000000000p-5f, -0x1.e000000000000p-5f,
    -0x1.0000000000000p-4f, -0x1.2000000000000p-4f, -0x1.4000000000000p-4f, -0x1.6000000000000p-4f, -0x1.8000000000000p-4f, -0x1.a000000000000p-4f, -0x1.c000000000000p-4f, -0x1.e000000000000p-4f,
    -0x1.0000000000000p-3f, -0x1.2000000000000p-3f, -0x1.4000000000000p-3f, -0x1.6000000000000p-3f, -0x1.8000000000000p-3f, -0x1.a000000000000p-3f, -0x1.c000000000000p-3f, -0x1.e000000000000p-3f,
    -0x1.0000000000000p-2f, -0x1.2000000000000p-2f, -0x1.4000000000000p-2f, -0x1.6000000000000p-2f, -0x1.8000000000000p-2f, -0x1.a000000000000p-2f, -0x1.c000000000000p-2f, -0x1.e000000000000p-2f,
    -0x1.0000000000000p-1f, -0x1.2000000000000p-1f, -0x1.4000000000000p-1f, -0x1.6000000000000p-1f, -0x1.8000000000000p-1f, -0x1.a000000000000p-1f, -0x1.c000000000000p-1f, -0x1.e000000000000p-1f,
    -0x1.0000000000000p+0f, -0x1.2000000000000p+0f, -0x1.4000000000000p+0f, -0x1.6000000000000p+0f, -0x1.8000000000000p+0f, -0x1.a000000000000p+0f, -0x1.c000000000000p+0f, -0x1.e000000000000p+0f,
    -0x1.0000000000000p+1f, -0x1.2000000000000p+1f, -0x1.4000000000000p+1f, -0x1.6000000000000p+1f, -0x1.8000000000000p+1f, -0x1.a000000000000p+1f, -0x1.c000000000000p+1f, -0x1.e000000000000p+1f,
    -0x1.0000000000000p+2f, -0x1.2000000000000p+2f, -0x1.4000000000000p+2f, -0x1.6000000000000p+2f, -0x1.8000000000000p+2f, -0x1.a000000000000p+2f, -0x1.c000000000000p+2f, -0x1.e000000000000p+2f,
    -0x1.0000000000000p+3f, -0x1.2000000000000p+3f, -0x1.4000000000000p+3f, -0x1.6000000000000p+3f, -0x1.8000000000000p+3f, -0x1.a000000000000p+3f, -0x1.c000000000000p+3f, -0x1.e000000000000p+3f,
    -0x1.0000000000000p+4f, -0x1.2000000000000p+4f, -0x1.4000000000000p+4f, -0x1.6000000000000p+4f, -0x1.8000000000000p+4f, -0x1.a000000000000p+4f, -0x1.c000000000000p+4f, -0x1.e000000000000p+4f,
    -0x1.0000000000000p+5f, -0x1.2000000000000p+5f, -0x1.4000000000000p+5f, -0x1.6000000000000p+5f, -0x1.8000000000000p+5f, -0x1.a000000000000p+5f, -0x1.c000000000000p+5f, -0x1.e000000000000p+5f,
    -0x1.0000000000000p+6f, -0x1.2000000000000p+6f, -0x1.4000000000000p+6f, -0x1.6000000000000p+6f, -0x1.8000000000000p+6f, -0x1.a000000000000p+6f, -0x1.c000000000000p+6f, -0x1.e000000000000p+6f,
    -0x1.0000000000000p+7f, -0x1.2000000000000p+7f, -0x1.4000000000000p+7f, -0x1.6000000000000p+7f, -0x1.8000000000000p+7f, -0x1.a000000000000p+7f, -0x1.c000000000000p+7f, -0x1.e000000000000p+7f,
    -0x1.0000000000000p+8f, -0x1.2000000000000p+8f, -0x1.4000000000000p+8f, -0x1.6000000000000p+8f, -0x1.8000000000000p+8f, -0x1.a000000000000p+8f, -0x1.c000000000000p+8f, NAN,
};

static constant half mk_e4m3_h_lut[256] = {
    0.0h, half(0x1.0000000000000p-9f), half(0x1.0000000000000p-8f), half(0x1.8000000000000p-8f), half(0x1.0000000000000p-7f), half(0x1.4000000000000p-7f), half(0x1.8000000000000p-7f), half(0x1.c000000000000p-7f),
    half(0x1.0000000000000p-6f), half(0x1.2000000000000p-6f), half(0x1.4000000000000p-6f), half(0x1.6000000000000p-6f), half(0x1.8000000000000p-6f), half(0x1.a000000000000p-6f), half(0x1.c000000000000p-6f), half(0x1.e000000000000p-6f),
    half(0x1.0000000000000p-5f), half(0x1.2000000000000p-5f), half(0x1.4000000000000p-5f), half(0x1.6000000000000p-5f), half(0x1.8000000000000p-5f), half(0x1.a000000000000p-5f), half(0x1.c000000000000p-5f), half(0x1.e000000000000p-5f),
    half(0x1.0000000000000p-4f), half(0x1.2000000000000p-4f), half(0x1.4000000000000p-4f), half(0x1.6000000000000p-4f), half(0x1.8000000000000p-4f), half(0x1.a000000000000p-4f), half(0x1.c000000000000p-4f), half(0x1.e000000000000p-4f),
    half(0x1.0000000000000p-3f), half(0x1.2000000000000p-3f), half(0x1.4000000000000p-3f), half(0x1.6000000000000p-3f), half(0x1.8000000000000p-3f), half(0x1.a000000000000p-3f), half(0x1.c000000000000p-3f), half(0x1.e000000000000p-3f),
    half(0x1.0000000000000p-2f), half(0x1.2000000000000p-2f), half(0x1.4000000000000p-2f), half(0x1.6000000000000p-2f), half(0x1.8000000000000p-2f), half(0x1.a000000000000p-2f), half(0x1.c000000000000p-2f), half(0x1.e000000000000p-2f),
    half(0x1.0000000000000p-1f), half(0x1.2000000000000p-1f), half(0x1.4000000000000p-1f), half(0x1.6000000000000p-1f), half(0x1.8000000000000p-1f), half(0x1.a000000000000p-1f), half(0x1.c000000000000p-1f), half(0x1.e000000000000p-1f),
    half(0x1.0000000000000p+0f), half(0x1.2000000000000p+0f), half(0x1.4000000000000p+0f), half(0x1.6000000000000p+0f), half(0x1.8000000000000p+0f), half(0x1.a000000000000p+0f), half(0x1.c000000000000p+0f), half(0x1.e000000000000p+0f),
    half(0x1.0000000000000p+1f), half(0x1.2000000000000p+1f), half(0x1.4000000000000p+1f), half(0x1.6000000000000p+1f), half(0x1.8000000000000p+1f), half(0x1.a000000000000p+1f), half(0x1.c000000000000p+1f), half(0x1.e000000000000p+1f),
    half(0x1.0000000000000p+2f), half(0x1.2000000000000p+2f), half(0x1.4000000000000p+2f), half(0x1.6000000000000p+2f), half(0x1.8000000000000p+2f), half(0x1.a000000000000p+2f), half(0x1.c000000000000p+2f), half(0x1.e000000000000p+2f),
    half(0x1.0000000000000p+3f), half(0x1.2000000000000p+3f), half(0x1.4000000000000p+3f), half(0x1.6000000000000p+3f), half(0x1.8000000000000p+3f), half(0x1.a000000000000p+3f), half(0x1.c000000000000p+3f), half(0x1.e000000000000p+3f),
    half(0x1.0000000000000p+4f), half(0x1.2000000000000p+4f), half(0x1.4000000000000p+4f), half(0x1.6000000000000p+4f), half(0x1.8000000000000p+4f), half(0x1.a000000000000p+4f), half(0x1.c000000000000p+4f), half(0x1.e000000000000p+4f),
    half(0x1.0000000000000p+5f), half(0x1.2000000000000p+5f), half(0x1.4000000000000p+5f), half(0x1.6000000000000p+5f), half(0x1.8000000000000p+5f), half(0x1.a000000000000p+5f), half(0x1.c000000000000p+5f), half(0x1.e000000000000p+5f),
    half(0x1.0000000000000p+6f), half(0x1.2000000000000p+6f), half(0x1.4000000000000p+6f), half(0x1.6000000000000p+6f), half(0x1.8000000000000p+6f), half(0x1.a000000000000p+6f), half(0x1.c000000000000p+6f), half(0x1.e000000000000p+6f),
    half(0x1.0000000000000p+7f), half(0x1.2000000000000p+7f), half(0x1.4000000000000p+7f), half(0x1.6000000000000p+7f), half(0x1.8000000000000p+7f), half(0x1.a000000000000p+7f), half(0x1.c000000000000p+7f), half(0x1.e000000000000p+7f),
    half(0x1.0000000000000p+8f), half(0x1.2000000000000p+8f), half(0x1.4000000000000p+8f), half(0x1.6000000000000p+8f), half(0x1.8000000000000p+8f), half(0x1.a000000000000p+8f), half(0x1.c000000000000p+8f), half(NAN),
    -0.0h, half(-0x1.0000000000000p-9f), half(-0x1.0000000000000p-8f), half(-0x1.8000000000000p-8f), half(-0x1.0000000000000p-7f), half(-0x1.4000000000000p-7f), half(-0x1.8000000000000p-7f), half(-0x1.c000000000000p-7f),
    half(-0x1.0000000000000p-6f), half(-0x1.2000000000000p-6f), half(-0x1.4000000000000p-6f), half(-0x1.6000000000000p-6f), half(-0x1.8000000000000p-6f), half(-0x1.a000000000000p-6f), half(-0x1.c000000000000p-6f), half(-0x1.e000000000000p-6f),
    half(-0x1.0000000000000p-5f), half(-0x1.2000000000000p-5f), half(-0x1.4000000000000p-5f), half(-0x1.6000000000000p-5f), half(-0x1.8000000000000p-5f), half(-0x1.a000000000000p-5f), half(-0x1.c000000000000p-5f), half(-0x1.e000000000000p-5f),
    half(-0x1.0000000000000p-4f), half(-0x1.2000000000000p-4f), half(-0x1.4000000000000p-4f), half(-0x1.6000000000000p-4f), half(-0x1.8000000000000p-4f), half(-0x1.a000000000000p-4f), half(-0x1.c000000000000p-4f), half(-0x1.e000000000000p-4f),
    half(-0x1.0000000000000p-3f), half(-0x1.2000000000000p-3f), half(-0x1.4000000000000p-3f), half(-0x1.6000000000000p-3f), half(-0x1.8000000000000p-3f), half(-0x1.a000000000000p-3f), half(-0x1.c000000000000p-3f), half(-0x1.e000000000000p-3f),
    half(-0x1.0000000000000p-2f), half(-0x1.2000000000000p-2f), half(-0x1.4000000000000p-2f), half(-0x1.6000000000000p-2f), half(-0x1.8000000000000p-2f), half(-0x1.a000000000000p-2f), half(-0x1.c000000000000p-2f), half(-0x1.e000000000000p-2f),
    half(-0x1.0000000000000p-1f), half(-0x1.2000000000000p-1f), half(-0x1.4000000000000p-1f), half(-0x1.6000000000000p-1f), half(-0x1.8000000000000p-1f), half(-0x1.a000000000000p-1f), half(-0x1.c000000000000p-1f), half(-0x1.e000000000000p-1f),
    half(-0x1.0000000000000p+0f), half(-0x1.2000000000000p+0f), half(-0x1.4000000000000p+0f), half(-0x1.6000000000000p+0f), half(-0x1.8000000000000p+0f), half(-0x1.a000000000000p+0f), half(-0x1.c000000000000p+0f), half(-0x1.e000000000000p+0f),
    half(-0x1.0000000000000p+1f), half(-0x1.2000000000000p+1f), half(-0x1.4000000000000p+1f), half(-0x1.6000000000000p+1f), half(-0x1.8000000000000p+1f), half(-0x1.a000000000000p+1f), half(-0x1.c000000000000p+1f), half(-0x1.e000000000000p+1f),
    half(-0x1.0000000000000p+2f), half(-0x1.2000000000000p+2f), half(-0x1.4000000000000p+2f), half(-0x1.6000000000000p+2f), half(-0x1.8000000000000p+2f), half(-0x1.a000000000000p+2f), half(-0x1.c000000000000p+2f), half(-0x1.e000000000000p+2f),
    half(-0x1.0000000000000p+3f), half(-0x1.2000000000000p+3f), half(-0x1.4000000000000p+3f), half(-0x1.6000000000000p+3f), half(-0x1.8000000000000p+3f), half(-0x1.a000000000000p+3f), half(-0x1.c000000000000p+3f), half(-0x1.e000000000000p+3f),
    half(-0x1.0000000000000p+4f), half(-0x1.2000000000000p+4f), half(-0x1.4000000000000p+4f), half(-0x1.6000000000000p+4f), half(-0x1.8000000000000p+4f), half(-0x1.a000000000000p+4f), half(-0x1.c000000000000p+4f), half(-0x1.e000000000000p+4f),
    half(-0x1.0000000000000p+5f), half(-0x1.2000000000000p+5f), half(-0x1.4000000000000p+5f), half(-0x1.6000000000000p+5f), half(-0x1.8000000000000p+5f), half(-0x1.a000000000000p+5f), half(-0x1.c000000000000p+5f), half(-0x1.e000000000000p+5f),
    half(-0x1.0000000000000p+6f), half(-0x1.2000000000000p+6f), half(-0x1.4000000000000p+6f), half(-0x1.6000000000000p+6f), half(-0x1.8000000000000p+6f), half(-0x1.a000000000000p+6f), half(-0x1.c000000000000p+6f), half(-0x1.e000000000000p+6f),
    half(-0x1.0000000000000p+7f), half(-0x1.2000000000000p+7f), half(-0x1.4000000000000p+7f), half(-0x1.6000000000000p+7f), half(-0x1.8000000000000p+7f), half(-0x1.a000000000000p+7f), half(-0x1.c000000000000p+7f), half(-0x1.e000000000000p+7f),
    half(-0x1.0000000000000p+8f), half(-0x1.2000000000000p+8f), half(-0x1.4000000000000p+8f), half(-0x1.6000000000000p+8f), half(-0x1.8000000000000p+8f), half(-0x1.a000000000000p+8f), half(-0x1.c000000000000p+8f), half(NAN),
};

static constant half2 mk_fp4_pair_h2_lut[256] = {
    half2(0.0h, 0.0h), half2(0.5h, 0.0h), half2(1.0h, 0.0h), half2(1.5h, 0.0h),
    half2(2.0h, 0.0h), half2(3.0h, 0.0h), half2(4.0h, 0.0h), half2(6.0h, 0.0h),
    half2(-0.0h, 0.0h), half2(-0.5h, 0.0h), half2(-1.0h, 0.0h), half2(-1.5h, 0.0h),
    half2(-2.0h, 0.0h), half2(-3.0h, 0.0h), half2(-4.0h, 0.0h), half2(-6.0h, 0.0h),
    half2(0.0h, 0.5h), half2(0.5h, 0.5h), half2(1.0h, 0.5h), half2(1.5h, 0.5h),
    half2(2.0h, 0.5h), half2(3.0h, 0.5h), half2(4.0h, 0.5h), half2(6.0h, 0.5h),
    half2(-0.0h, 0.5h), half2(-0.5h, 0.5h), half2(-1.0h, 0.5h), half2(-1.5h, 0.5h),
    half2(-2.0h, 0.5h), half2(-3.0h, 0.5h), half2(-4.0h, 0.5h), half2(-6.0h, 0.5h),
    half2(0.0h, 1.0h), half2(0.5h, 1.0h), half2(1.0h, 1.0h), half2(1.5h, 1.0h),
    half2(2.0h, 1.0h), half2(3.0h, 1.0h), half2(4.0h, 1.0h), half2(6.0h, 1.0h),
    half2(-0.0h, 1.0h), half2(-0.5h, 1.0h), half2(-1.0h, 1.0h), half2(-1.5h, 1.0h),
    half2(-2.0h, 1.0h), half2(-3.0h, 1.0h), half2(-4.0h, 1.0h), half2(-6.0h, 1.0h),
    half2(0.0h, 1.5h), half2(0.5h, 1.5h), half2(1.0h, 1.5h), half2(1.5h, 1.5h),
    half2(2.0h, 1.5h), half2(3.0h, 1.5h), half2(4.0h, 1.5h), half2(6.0h, 1.5h),
    half2(-0.0h, 1.5h), half2(-0.5h, 1.5h), half2(-1.0h, 1.5h), half2(-1.5h, 1.5h),
    half2(-2.0h, 1.5h), half2(-3.0h, 1.5h), half2(-4.0h, 1.5h), half2(-6.0h, 1.5h),
    half2(0.0h, 2.0h), half2(0.5h, 2.0h), half2(1.0h, 2.0h), half2(1.5h, 2.0h),
    half2(2.0h, 2.0h), half2(3.0h, 2.0h), half2(4.0h, 2.0h), half2(6.0h, 2.0h),
    half2(-0.0h, 2.0h), half2(-0.5h, 2.0h), half2(-1.0h, 2.0h), half2(-1.5h, 2.0h),
    half2(-2.0h, 2.0h), half2(-3.0h, 2.0h), half2(-4.0h, 2.0h), half2(-6.0h, 2.0h),
    half2(0.0h, 3.0h), half2(0.5h, 3.0h), half2(1.0h, 3.0h), half2(1.5h, 3.0h),
    half2(2.0h, 3.0h), half2(3.0h, 3.0h), half2(4.0h, 3.0h), half2(6.0h, 3.0h),
    half2(-0.0h, 3.0h), half2(-0.5h, 3.0h), half2(-1.0h, 3.0h), half2(-1.5h, 3.0h),
    half2(-2.0h, 3.0h), half2(-3.0h, 3.0h), half2(-4.0h, 3.0h), half2(-6.0h, 3.0h),
    half2(0.0h, 4.0h), half2(0.5h, 4.0h), half2(1.0h, 4.0h), half2(1.5h, 4.0h),
    half2(2.0h, 4.0h), half2(3.0h, 4.0h), half2(4.0h, 4.0h), half2(6.0h, 4.0h),
    half2(-0.0h, 4.0h), half2(-0.5h, 4.0h), half2(-1.0h, 4.0h), half2(-1.5h, 4.0h),
    half2(-2.0h, 4.0h), half2(-3.0h, 4.0h), half2(-4.0h, 4.0h), half2(-6.0h, 4.0h),
    half2(0.0h, 6.0h), half2(0.5h, 6.0h), half2(1.0h, 6.0h), half2(1.5h, 6.0h),
    half2(2.0h, 6.0h), half2(3.0h, 6.0h), half2(4.0h, 6.0h), half2(6.0h, 6.0h),
    half2(-0.0h, 6.0h), half2(-0.5h, 6.0h), half2(-1.0h, 6.0h), half2(-1.5h, 6.0h),
    half2(-2.0h, 6.0h), half2(-3.0h, 6.0h), half2(-4.0h, 6.0h), half2(-6.0h, 6.0h),
    half2(0.0h, -0.0h), half2(0.5h, -0.0h), half2(1.0h, -0.0h), half2(1.5h, -0.0h),
    half2(2.0h, -0.0h), half2(3.0h, -0.0h), half2(4.0h, -0.0h), half2(6.0h, -0.0h),
    half2(-0.0h, -0.0h), half2(-0.5h, -0.0h), half2(-1.0h, -0.0h), half2(-1.5h, -0.0h),
    half2(-2.0h, -0.0h), half2(-3.0h, -0.0h), half2(-4.0h, -0.0h), half2(-6.0h, -0.0h),
    half2(0.0h, -0.5h), half2(0.5h, -0.5h), half2(1.0h, -0.5h), half2(1.5h, -0.5h),
    half2(2.0h, -0.5h), half2(3.0h, -0.5h), half2(4.0h, -0.5h), half2(6.0h, -0.5h),
    half2(-0.0h, -0.5h), half2(-0.5h, -0.5h), half2(-1.0h, -0.5h), half2(-1.5h, -0.5h),
    half2(-2.0h, -0.5h), half2(-3.0h, -0.5h), half2(-4.0h, -0.5h), half2(-6.0h, -0.5h),
    half2(0.0h, -1.0h), half2(0.5h, -1.0h), half2(1.0h, -1.0h), half2(1.5h, -1.0h),
    half2(2.0h, -1.0h), half2(3.0h, -1.0h), half2(4.0h, -1.0h), half2(6.0h, -1.0h),
    half2(-0.0h, -1.0h), half2(-0.5h, -1.0h), half2(-1.0h, -1.0h), half2(-1.5h, -1.0h),
    half2(-2.0h, -1.0h), half2(-3.0h, -1.0h), half2(-4.0h, -1.0h), half2(-6.0h, -1.0h),
    half2(0.0h, -1.5h), half2(0.5h, -1.5h), half2(1.0h, -1.5h), half2(1.5h, -1.5h),
    half2(2.0h, -1.5h), half2(3.0h, -1.5h), half2(4.0h, -1.5h), half2(6.0h, -1.5h),
    half2(-0.0h, -1.5h), half2(-0.5h, -1.5h), half2(-1.0h, -1.5h), half2(-1.5h, -1.5h),
    half2(-2.0h, -1.5h), half2(-3.0h, -1.5h), half2(-4.0h, -1.5h), half2(-6.0h, -1.5h),
    half2(0.0h, -2.0h), half2(0.5h, -2.0h), half2(1.0h, -2.0h), half2(1.5h, -2.0h),
    half2(2.0h, -2.0h), half2(3.0h, -2.0h), half2(4.0h, -2.0h), half2(6.0h, -2.0h),
    half2(-0.0h, -2.0h), half2(-0.5h, -2.0h), half2(-1.0h, -2.0h), half2(-1.5h, -2.0h),
    half2(-2.0h, -2.0h), half2(-3.0h, -2.0h), half2(-4.0h, -2.0h), half2(-6.0h, -2.0h),
    half2(0.0h, -3.0h), half2(0.5h, -3.0h), half2(1.0h, -3.0h), half2(1.5h, -3.0h),
    half2(2.0h, -3.0h), half2(3.0h, -3.0h), half2(4.0h, -3.0h), half2(6.0h, -3.0h),
    half2(-0.0h, -3.0h), half2(-0.5h, -3.0h), half2(-1.0h, -3.0h), half2(-1.5h, -3.0h),
    half2(-2.0h, -3.0h), half2(-3.0h, -3.0h), half2(-4.0h, -3.0h), half2(-6.0h, -3.0h),
    half2(0.0h, -4.0h), half2(0.5h, -4.0h), half2(1.0h, -4.0h), half2(1.5h, -4.0h),
    half2(2.0h, -4.0h), half2(3.0h, -4.0h), half2(4.0h, -4.0h), half2(6.0h, -4.0h),
    half2(-0.0h, -4.0h), half2(-0.5h, -4.0h), half2(-1.0h, -4.0h), half2(-1.5h, -4.0h),
    half2(-2.0h, -4.0h), half2(-3.0h, -4.0h), half2(-4.0h, -4.0h), half2(-6.0h, -4.0h),
    half2(0.0h, -6.0h), half2(0.5h, -6.0h), half2(1.0h, -6.0h), half2(1.5h, -6.0h),
    half2(2.0h, -6.0h), half2(3.0h, -6.0h), half2(4.0h, -6.0h), half2(6.0h, -6.0h),
    half2(-0.0h, -6.0h), half2(-0.5h, -6.0h), half2(-1.0h, -6.0h), half2(-1.5h, -6.0h),
    half2(-2.0h, -6.0h), half2(-3.0h, -6.0h), half2(-4.0h, -6.0h), half2(-6.0h, -6.0h),
};

template <bool Lut>
inline void mk_nvfp4_decode16(uint2 raw, thread float4 &d0, thread float4 &d1, thread float4 &d2, thread float4 &d3) {
    if constexpr (Lut) {
        const uchar4 a0 = as_type<uchar4>(raw.x), a1 = as_type<uchar4>(raw.y);
        d0 = float4(half4(mk_fp4_pair_h2_lut[a0[0]], mk_fp4_pair_h2_lut[a0[1]]));
        d1 = float4(half4(mk_fp4_pair_h2_lut[a0[2]], mk_fp4_pair_h2_lut[a0[3]]));
        d2 = float4(half4(mk_fp4_pair_h2_lut[a1[0]], mk_fp4_pair_h2_lut[a1[1]]));
        d3 = float4(half4(mk_fp4_pair_h2_lut[a1[2]], mk_fp4_pair_h2_lut[a1[3]]));
    } else {
        d0 = inkling_nvfp4_decode4(raw.x);
        d1 = inkling_nvfp4_decode4(raw.x >> 16);
        d2 = inkling_nvfp4_decode4(raw.y);
        d3 = inkling_nvfp4_decode4(raw.y >> 16);
    }
}

template <bool Lut>
inline float mk_nvfp4_dot16(uint2 raw, float4 x0, float4 x1, float4 x2, float4 x3) {
    float4 d0, d1, d2, d3;
    mk_nvfp4_decode16<Lut>(raw, d0, d1, d2, d3);
    return dot(d0, x0) + dot(d1, x1) + dot(d2, x2) + dot(d3, x3);
}

template <bool Lut>
inline float4 mk_e4m3x4_decode(uint packed) {
    if constexpr (Lut) {
        const uchar4 b = as_type<uchar4>(packed);
        return float4(mk_e4m3_f32_lut[b[0]], mk_e4m3_f32_lut[b[1]], mk_e4m3_f32_lut[b[2]], mk_e4m3_f32_lut[b[3]]);
    } else {
        return float4(mk_expert_e4m3(uchar(packed)),
                      mk_expert_e4m3(uchar(packed >> 8)),
                      mk_expert_e4m3(uchar(packed >> 16)),
                      mk_expert_e4m3(uchar(packed >> 24)));
    }
}

template <bool Lut>
inline float mk_e4m3x4_dot(uint packed, float4 xv) {
    return dot(mk_e4m3x4_decode<Lut>(packed), xv);
}

// 16-lane sub-SIMD NVFP4 gate+up decode: 2 pairs (4 rows) per sub-SIMD, 8 rows per SIMD.
template <bool Lut, bool Bf16Out>
inline void mk_expert_gate_up_nv16_impl(device float *arena, device const uchar *w, device const uchar *s,
                                        device const uchar *amax, device const uchar *denom, constant MKExpert &p,
                                        uint gid, uint lane) {
    const uint simd_id = gid / 32u;
    const uint cols4 = p.n / 4u;
    const uint slot = simd_id / cols4;
    if (slot >= p.top_k) return;
    const uint sub = lane >> 4;
    const uint l16 = lane & 15u;
    const uint j = (simd_id % cols4) * 4u + sub * 2u;
    const uint row = p.row0;
    const uint expert = uint(arena[p.routes + row * p.rs + slot]);
    float4 acc = 0.0f;
    if (expert < p.experts) {
        const ulong base = (ulong(expert) * p.n + j) * 2u * p.k;
        device const float4 *xp = (device const float4 *)(arena + p.x + row * p.xs);
        device const uint2 *wp = (device const uint2 *)(w + p.w + base / 2u);
        device const uchar *sp = s + p.s + base / 16u;
        const uint wstep = p.k / 16u;
        const uint sstep = p.k / 16u;
        for (uint k4 = l16 * 4u; k4 < p.k / 4u; k4 += INKLING_NVFP4_SUBSIMD_LANES * 4u) {
            const float4 x0 = xp[k4];
            const float4 x1 = xp[k4 + 1u];
            const float4 x2 = xp[k4 + 2u];
            const float4 x3 = xp[k4 + 3u];
            const uint b = k4 / 4u;
            #pragma unroll
            for (uint c = 0u; c < 4u; ++c) {
                const uchar sc_code = sp[b + c * sstep];
                const float sc = Lut ? mk_e4m3_f32_lut[sc_code] : mk_expert_e4m3(sc_code);
                const float d = mk_nvfp4_dot16<Lut>(wp[b + c * wstep], x0, x1, x2, x3);
                acc[c] = fma(d, sc, acc[c]);
            }
        }
        acc.x += simd_shuffle_xor(acc.x, ushort(8));
        acc.y += simd_shuffle_xor(acc.y, ushort(8));
        acc.z += simd_shuffle_xor(acc.z, ushort(8));
        acc.w += simd_shuffle_xor(acc.w, ushort(8));
        acc.x += simd_shuffle_xor(acc.x, ushort(4));
        acc.y += simd_shuffle_xor(acc.y, ushort(4));
        acc.z += simd_shuffle_xor(acc.z, ushort(4));
        acc.w += simd_shuffle_xor(acc.w, ushort(4));
        acc.x += simd_shuffle_xor(acc.x, ushort(2));
        acc.y += simd_shuffle_xor(acc.y, ushort(2));
        acc.z += simd_shuffle_xor(acc.z, ushort(2));
        acc.w += simd_shuffle_xor(acc.w, ushort(2));
        acc.x += simd_shuffle_xor(acc.x, ushort(1));
        acc.y += simd_shuffle_xor(acc.y, ushort(1));
        acc.z += simd_shuffle_xor(acc.z, ushort(1));
        acc.w += simd_shuffle_xor(acc.w, ushort(1));
    }
    if (l16 == 0u) {
        if (expert < p.experts) acc *= mk_load(amax, p.amax, 0u, expert) / mk_load(denom, p.denominator, 0u, 0u);
        device float *out = arena + p.y + row * p.ys + slot * p.n + j;
        device bfloat *out16 = reinterpret_cast<device bfloat *>(arena + p.y + row * p.ys) + slot * p.n + j;
        for (uint c = 0u; c < 2u; ++c) {
            float gate = acc[c * 2u], up = acc[c * 2u + 1u];
            if (p.round_bf16) { gate = mk_bf16(gate); up = mk_bf16(up); }
            const float value = ((tanh(gate / p.gate_cap) * p.gate_cap) * mk_sigmoid(gate)) * (tanh(up / p.up_cap) * p.up_cap);
            if constexpr (Bf16Out) out16[c] = bfloat(p.round_bf16 ? mk_bf16(value) : value);
            else out[c] = p.round_bf16 ? mk_bf16(value) : value;
        }
    }
}

#define MK_EXPERT_NV16(NAME, LUT, BF16) \
    kernel void NAME(device float *arena [[buffer(0)]], device const uchar *w [[buffer(1)]], \
                     device const uchar *s [[buffer(2)]], device const uchar *amax [[buffer(3)]], \
                     device const uchar *denom [[buffer(4)]], constant MKExpert &p [[buffer(8)]], \
                     uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) { \
        mk_expert_gate_up_nv16_impl<LUT, BF16>(arena, w, s, amax, denom, p, gid, lane); \
    }
MK_EXPERT_NV16(mk_expert_gate_up_nv16, false, false)
MK_EXPERT_NV16(mk_expert_gate_up_nv16_lut, true, false)
MK_EXPERT_NV16(mk_expert_gate_up_nv16_bf16, false, true)
MK_EXPERT_NV16(mk_expert_gate_up_nv16_lut_bf16, true, true)
#undef MK_EXPERT_NV16

// 8-lane sub-SIMD NVFP4 expert down decode: 2 SIMDs (64 threads) per TG cover all 8 expert slots x 4 columns.
template <bool Lut, bool Bf16In>
inline void mk_expert_down_nv8_impl(device float *arena, device const uchar *w, device const uchar *s,
                                    device const uchar *amax, device const uchar *denom, constant MKExpert &p,
                                    uint i, uint tid, threadgroup float4 *dots) {
    const uint columns = p.n / 4u;
    if (i >= columns) return;
    const uint j = i * 4u;
    const uint row = p.row0;
    const uint slot = tid >> 3;
    const uint l8 = tid & 7u;
    device const float *routes = arena + p.routes + row * p.rs;
    const uint expert = uint(routes[slot]);
    float4 acc = 0.0f;
    if (expert < p.experts) {
        const ulong base = (ulong(expert) * p.n + j) * p.k;
        device const float4 *xp = Bf16In ? nullptr : (device const float4 *)(arena + p.x + row * p.xs + slot * p.k);
        device const bfloat4 *xp16 = Bf16In ? (reinterpret_cast<device const bfloat4 *>(arena + p.x + row * p.xs) + (slot * p.k) / 4u) : nullptr;
        device const uint2 *wp = (device const uint2 *)(w + p.w + base / 2u);
        device const uchar *sp = s + p.s + base / 16u;
        const uint wstep = p.k / 16u;
        const uint sstep = p.k / 16u;
        for (uint k4 = l8 * 4u; k4 < p.k / 4u; k4 += INKLING_NVFP4_SUBSIMD8_LANES * 4u) {
            const float4 x0 = Bf16In ? float4(xp16[k4]) : xp[k4];
            const float4 x1 = Bf16In ? float4(xp16[k4 + 1u]) : xp[k4 + 1u];
            const float4 x2 = Bf16In ? float4(xp16[k4 + 2u]) : xp[k4 + 2u];
            const float4 x3 = Bf16In ? float4(xp16[k4 + 3u]) : xp[k4 + 3u];
            const uint b = k4 / 4u;
            #pragma unroll
            for (uint c = 0u; c < 4u; ++c) {
                const uchar sc_code = sp[b + c * sstep];
                const float sc = Lut ? mk_e4m3_f32_lut[sc_code] : mk_expert_e4m3(sc_code);
                const float d = mk_nvfp4_dot16<Lut>(wp[b + c * wstep], x0, x1, x2, x3);
                acc[c] = fma(d, sc, acc[c]);
            }
        }
        acc.x += simd_shuffle_xor(acc.x, ushort(4));
        acc.y += simd_shuffle_xor(acc.y, ushort(4));
        acc.z += simd_shuffle_xor(acc.z, ushort(4));
        acc.w += simd_shuffle_xor(acc.w, ushort(4));
        acc.x += simd_shuffle_xor(acc.x, ushort(2));
        acc.y += simd_shuffle_xor(acc.y, ushort(2));
        acc.z += simd_shuffle_xor(acc.z, ushort(2));
        acc.w += simd_shuffle_xor(acc.w, ushort(2));
        acc.x += simd_shuffle_xor(acc.x, ushort(1));
        acc.y += simd_shuffle_xor(acc.y, ushort(1));
        acc.z += simd_shuffle_xor(acc.z, ushort(1));
        acc.w += simd_shuffle_xor(acc.w, ushort(1));
    }
    if (l8 == 0u) {
        if (expert < p.experts) {
            acc *= mk_load(amax, p.amax, 0u, expert) / mk_load(denom, p.denominator, 0u, 0u);
            if (p.round_bf16) {
                for (uint c = 0u; c < 4u; ++c) acc[c] = mk_bf16(acc[c]);
            }
            dots[slot] = acc * routes[p.top_k + slot];
        } else {
            dots[slot] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 4u) {
        float sum = 0.0f;
        for (uint k = 0u; k < p.top_k; ++k) sum += dots[k][tid];
        arena[p.y + row * p.ys + j + tid] = p.round_bf16 ? mk_bf16(sum) : sum;
    }
}

#define MK_EXPERT_NV8(NAME, LUT, BF16) \
    kernel void NAME(device float *arena [[buffer(0)]], device const uchar *w [[buffer(1)]], \
                     device const uchar *s [[buffer(2)]], device const uchar *amax [[buffer(3)]], \
                     device const uchar *denom [[buffer(4)]], constant MKExpert &p [[buffer(8)]], \
                     uint i [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) { \
        threadgroup float4 dots[8]; \
        mk_expert_down_nv8_impl<LUT, BF16>(arena, w, s, amax, denom, p, i, tid, dots); \
    }
MK_EXPERT_NV8(mk_expert_down_nv8, false, false)
MK_EXPERT_NV8(mk_expert_down_nv8_lut, true, false)
MK_EXPERT_NV8(mk_expert_down_nv8_bf16, false, true)
MK_EXPERT_NV8(mk_expert_down_nv8_lut_bf16, true, true)
#undef MK_EXPERT_NV8

// Vectorized 8-byte MXFP8 gate+up (k % 256 == 0, w % 8 == 0).
template <bool Lut, bool Bf16Out>
inline void mk_expert_gate_up_mx_vec_impl(device float *arena, device const uchar *w, device const uchar *s,
                                          constant MKExpert &p, uint gid, uint lane) {
    const uint simd_id = gid / 32u;
    const uint columns = p.n / 2u;
    const uint slot = simd_id / columns, j = (simd_id % columns) * 2u;
    if (slot >= p.top_k) return;
    const uint row = p.row0;
    const uint expert = uint(arena[p.routes + row * p.rs + slot]);
    float4 acc = 0.0f;
    if (expert < p.experts) {
        const ulong base = (ulong(expert) * p.n + j) * 2u * p.k;
        device const float4 *xp = (device const float4 *)(arena + p.x + row * p.xs);
        device const uint2 *wp = (device const uint2 *)(w + p.w + base);
        device const uchar *sp = s + p.s + base / 32u;
        const uint wstep = p.k / 8u;
        const uint sstep = p.k / 32u;
        for (uint k8 = lane; k8 < p.k / 8u; k8 += 32u) {
            const float4 x0 = xp[2u * k8];
            const float4 x1 = xp[2u * k8 + 1u];
            const uint sb = k8 / 4u;
            #pragma unroll
            for (uint c = 0u; c < 4u; ++c) {
                const uint e = sp[sb + c * sstep];
                const float scale = as_type<float>(e == 0u ? 0x00400000u : e == 255u ? 0x7fc00000u : e << 23);
                const uint2 packed = wp[k8 + c * wstep];
                const float part = mk_e4m3x4_dot<Lut>(packed.x, x0) + mk_e4m3x4_dot<Lut>(packed.y, x1);
                acc[c] = fma(part, scale, acc[c]);
            }
        }
        acc = float4(simd_sum(acc.x), simd_sum(acc.y), simd_sum(acc.z), simd_sum(acc.w));
    }
    if (lane == 0u) {
        device float *out = arena + p.y + row * p.ys + slot * p.n + j;
        device bfloat *out16 = reinterpret_cast<device bfloat *>(arena + p.y + row * p.ys) + slot * p.n + j;
        for (uint c = 0u; c < 2u; ++c) {
            float gate = acc[c * 2u], up = acc[c * 2u + 1u];
            if (p.round_bf16) { gate = mk_bf16(gate); up = mk_bf16(up); }
            const float value = ((tanh(gate / p.gate_cap) * p.gate_cap) * mk_sigmoid(gate)) * (tanh(up / p.up_cap) * p.up_cap);
            if constexpr (Bf16Out) out16[c] = bfloat(p.round_bf16 ? mk_bf16(value) : value);
            else out[c] = p.round_bf16 ? mk_bf16(value) : value;
        }
    }
}

#define MK_EXPERT_MX_UP(NAME, LUT, BF16) \
    kernel void NAME(device float *arena [[buffer(0)]], device const uchar *w [[buffer(1)]], \
                     device const uchar *s [[buffer(2)]], device const uchar *amax [[buffer(3)]], \
                     device const uchar *denom [[buffer(4)]], constant MKExpert &p [[buffer(8)]], \
                     uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) { \
        (void)amax; (void)denom; \
        mk_expert_gate_up_mx_vec_impl<LUT, BF16>(arena, w, s, p, gid, lane); \
    }
MK_EXPERT_MX_UP(mk_expert_gate_up_mx_vec, false, false)
MK_EXPERT_MX_UP(mk_expert_gate_up_mx_lut, true, false)
MK_EXPERT_MX_UP(mk_expert_gate_up_mx_bf16, false, true)
MK_EXPERT_MX_UP(mk_expert_gate_up_mx_lut_bf16, true, true)
#undef MK_EXPERT_MX_UP

// Vectorized 4-byte MXFP8 down (k % 128 == 0, w % 4 == 0).
template <bool Lut, bool Bf16In>
inline void mk_expert_down_mx_vec_impl(device float *arena, device const uchar *w, device const uchar *s,
                                       constant MKExpert &p, uint i, uint slot, uint lane, threadgroup float4 *dots) {
    const uint columns = p.n / 4u;
    if (i >= columns) return;
    const uint j = i * 4u;
    const uint row = p.row0;
    if (slot < p.top_k) {
        device const float *routes = arena + p.routes + row * p.rs;
        const uint expert = uint(routes[slot]);
        float4 acc = 0.0f;
        if (expert < p.experts) {
            const ulong base = (ulong(expert) * p.n + j) * p.k;
            device const float4 *xp = Bf16In ? nullptr : (device const float4 *)(arena + p.x + row * p.xs + slot * p.k);
            device const bfloat4 *xp16 = Bf16In ? (reinterpret_cast<device const bfloat4 *>(arena + p.x + row * p.xs) + (slot * p.k) / 4u) : nullptr;
            device const uint *wp = (device const uint *)(w + p.w + base);
            device const uchar *sp = s + p.s + base / 32u;
            const uint wstep = p.k / 4u;
            const uint sstep = p.k / 32u;
            for (uint k4 = lane; k4 < p.k / 4u; k4 += 32u) {
                const float4 xv = Bf16In ? float4(xp16[k4]) : xp[k4];
                const uint sb = k4 / 8u;
                #pragma unroll
                for (uint c = 0u; c < 4u; ++c) {
                    const uint e = sp[sb + c * sstep];
                    const float scale = as_type<float>(e == 0u ? 0x00400000u : e == 255u ? 0x7fc00000u : e << 23);
                    const float part = mk_e4m3x4_dot<Lut>(wp[k4 + c * wstep], xv);
                    acc[c] = fma(part, scale, acc[c]);
                }
            }
            acc = float4(simd_sum(acc.x), simd_sum(acc.y), simd_sum(acc.z), simd_sum(acc.w));
        }
        if (lane == 0u) {
            if (p.round_bf16) for (uint c = 0u; c < 4u; ++c) acc[c] = mk_bf16(acc[c]);
            dots[slot] = acc * routes[p.top_k + slot];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (slot == 0u && lane < 4u) {
        float sum = 0.0f;
        for (uint k = 0u; k < p.top_k; ++k) sum += dots[k][lane];
        arena[p.y + row * p.ys + j + lane] = p.round_bf16 ? mk_bf16(sum) : sum;
    }
}

#define MK_EXPERT_MX_DOWN(NAME, LUT, BF16) \
    kernel void NAME(device float *arena [[buffer(0)]], device const uchar *w [[buffer(1)]], \
                     device const uchar *s [[buffer(2)]], device const uchar *amax [[buffer(3)]], \
                     device const uchar *denom [[buffer(4)]], constant MKExpert &p [[buffer(8)]], \
                     uint i [[threadgroup_position_in_grid]], uint slot [[simdgroup_index_in_threadgroup]], \
                     uint lane [[thread_index_in_simdgroup]]) { \
        (void)amax; (void)denom; \
        threadgroup float4 dots[8]; \
        mk_expert_down_mx_vec_impl<LUT, BF16>(arena, w, s, p, i, slot, lane, dots); \
    }
MK_EXPERT_MX_DOWN(mk_expert_down_mx_vec, false, false)
MK_EXPERT_MX_DOWN(mk_expert_down_mx_lut, true, false)
MK_EXPERT_MX_DOWN(mk_expert_down_mx_bf16, false, true)
MK_EXPERT_MX_DOWN(mk_expert_down_mx_lut_bf16, true, true)
#undef MK_EXPERT_MX_DOWN

// ---- Exact 0-ULP grouped MoE kernels for rows = 2..8 (consumes mk_expert_group_tiles) ----
template <bool Lut, bool Bf16Out>
inline void mk_expert_grouped_up_nv16_impl(device float *arena, device const uchar *w, device const uchar *s,
                                           device const uchar *amax, device const uchar *denom,
                                           device const uint *lists, constant MKExpert &p,
                                           uint gid, uint lane) {
    const uint cols4 = p.n / 4u;
    const uint simd_id = gid / 32u;
    const uint tile_idx = simd_id / cols4;
    const uint tile_count = lists[p.experts + 1u];
    if (tile_idx >= tile_count) return;
    const uint sub = lane >> 4;
    const uint l16 = lane & 15u;
    const uint j = (simd_id % cols4) * 4u + sub * 2u;

    const uint total = p.rows * p.top_k;
    device const uint *expert_off = lists;
    device const uint *sorted_routes = lists + p.experts + 2u;
    device const uint *tile_e = sorted_routes + total;
    const uint expert = tile_e[tile_idx];
    const uint off = expert_off[expert];
    const uint count = expert_off[expert + 1u] - off;
    device const uint *entries = sorted_routes + off;

    const ulong base = (ulong(expert) * p.n + j) * 2u * p.k;
    device const uint2 *wp = (device const uint2 *)(w + p.w + base / 2u);
    device const uchar *sp = s + p.s + base / 16u;
    const uint wstep = p.k / 16u;
    const uint sstep = p.k / 16u;
    const float outer = mk_load(amax, p.amax, 0u, expert) / mk_load(denom, p.denominator, 0u, 0u);

    if (count == 1u) {
        const uint route = entries[0];
        const uint tok_row = p.row0 + route / p.top_k;
        const uint slot = route % p.top_k;
        device const float4 *xp = (device const float4 *)(arena + p.x + tok_row * p.xs);
        float4 acc = 0.0f;
        for (uint k4 = l16 * 4u; k4 < p.k / 4u; k4 += INKLING_NVFP4_SUBSIMD_LANES * 4u) {
            const float4 x0 = xp[k4], x1 = xp[k4 + 1u], x2 = xp[k4 + 2u], x3 = xp[k4 + 3u];
            const uint b = k4 / 4u;
            #pragma unroll
            for (uint c = 0u; c < 4u; ++c) {
                const uchar sc_code = sp[b + c * sstep];
                const float sc = Lut ? mk_e4m3_f32_lut[sc_code] : mk_expert_e4m3(sc_code);
                const float d = mk_nvfp4_dot16<Lut>(wp[b + c * wstep], x0, x1, x2, x3);
                acc[c] = fma(d, sc, acc[c]);
            }
        }
        #pragma unroll
        for (uint c = 0u; c < 4u; ++c) {
            acc[c] += simd_shuffle_xor(acc[c], ushort(8));
            acc[c] += simd_shuffle_xor(acc[c], ushort(4));
            acc[c] += simd_shuffle_xor(acc[c], ushort(2));
            acc[c] += simd_shuffle_xor(acc[c], ushort(1));
        }
        if (l16 == 0u) {
            acc *= outer;
            device float *out = arena + p.y + tok_row * p.ys + slot * p.n + j;
            device bfloat *out16 = reinterpret_cast<device bfloat *>(arena + p.y + tok_row * p.ys) + slot * p.n + j;
            for (uint c = 0u; c < 2u; ++c) {
                float gate = acc[c * 2u], up = acc[c * 2u + 1u];
                if (p.round_bf16) { gate = mk_bf16(gate); up = mk_bf16(up); }
                const float value = ((tanh(gate / p.gate_cap) * p.gate_cap) * mk_sigmoid(gate)) * (tanh(up / p.up_cap) * p.up_cap);
                if constexpr (Bf16Out) out16[c] = bfloat(p.round_bf16 ? mk_bf16(value) : value);
                else out[c] = p.round_bf16 ? mk_bf16(value) : value;
            }
        }
        return;
    }

    if (count == 2u) {
        const uint r0 = entries[0], r1 = entries[1];
        const uint tr0 = p.row0 + r0 / p.top_k, sl0 = r0 % p.top_k;
        const uint tr1 = p.row0 + r1 / p.top_k, sl1 = r1 % p.top_k;
        device const float4 *xp0 = (device const float4 *)(arena + p.x + tr0 * p.xs);
        device const float4 *xp1 = (device const float4 *)(arena + p.x + tr1 * p.xs);
        float4 acc0 = 0.0f, acc1 = 0.0f;
        for (uint k4 = l16 * 4u; k4 < p.k / 4u; k4 += INKLING_NVFP4_SUBSIMD_LANES * 4u) {
            const float4 x00 = xp0[k4], x01 = xp0[k4 + 1u], x02 = xp0[k4 + 2u], x03 = xp0[k4 + 3u];
            const float4 x10 = xp1[k4], x11 = xp1[k4 + 1u], x12 = xp1[k4 + 2u], x13 = xp1[k4 + 3u];
            const uint b = k4 / 4u;
            #pragma unroll
            for (uint c = 0u; c < 4u; ++c) {
                const uchar sc_code = sp[b + c * sstep];
                const float sc = Lut ? mk_e4m3_f32_lut[sc_code] : mk_expert_e4m3(sc_code);
                float4 d0, d1, d2, d3;
                mk_nvfp4_decode16<Lut>(wp[b + c * wstep], d0, d1, d2, d3);
                acc0[c] = fma(dot(d0, x00) + dot(d1, x01) + dot(d2, x02) + dot(d3, x03), sc, acc0[c]);
                acc1[c] = fma(dot(d0, x10) + dot(d1, x11) + dot(d2, x12) + dot(d3, x13), sc, acc1[c]);
            }
        }
        #pragma unroll
        for (uint c = 0u; c < 4u; ++c) {
            acc0[c] += simd_shuffle_xor(acc0[c], ushort(8));
            acc0[c] += simd_shuffle_xor(acc0[c], ushort(4));
            acc0[c] += simd_shuffle_xor(acc0[c], ushort(2));
            acc0[c] += simd_shuffle_xor(acc0[c], ushort(1));
            acc1[c] += simd_shuffle_xor(acc1[c], ushort(8));
            acc1[c] += simd_shuffle_xor(acc1[c], ushort(4));
            acc1[c] += simd_shuffle_xor(acc1[c], ushort(2));
            acc1[c] += simd_shuffle_xor(acc1[c], ushort(1));
        }
        if (l16 == 0u) {
            acc0 *= outer;
            acc1 *= outer;
            const uint trs[2] = { tr0, tr1 }, sls[2] = { sl0, sl1 };
            const float4 vals[2] = { acc0, acc1 };
            for (uint ti = 0u; ti < 2u; ++ti) {
                device float *out = arena + p.y + trs[ti] * p.ys + sls[ti] * p.n + j;
                device bfloat *out16 = reinterpret_cast<device bfloat *>(arena + p.y + trs[ti] * p.ys) + sls[ti] * p.n + j;
                for (uint c = 0u; c < 2u; ++c) {
                    float gate = vals[ti][c * 2u], up = vals[ti][c * 2u + 1u];
                    if (p.round_bf16) { gate = mk_bf16(gate); up = mk_bf16(up); }
                    const float value = ((tanh(gate / p.gate_cap) * p.gate_cap) * mk_sigmoid(gate)) * (tanh(up / p.up_cap) * p.up_cap);
                    if constexpr (Bf16Out) out16[c] = bfloat(p.round_bf16 ? mk_bf16(value) : value);
                    else out[c] = p.round_bf16 ? mk_bf16(value) : value;
                }
            }
        }
        return;
    }

    for (uint t0 = 0u; t0 < count; t0 += 4u) {
        const uint n_cur = min(4u, count - t0);
        device const float4 *xp_t[4];
        #pragma unroll
        for (uint ti = 0u; ti < 4u; ++ti) {
            const uint route = entries[t0 + min(ti, n_cur - 1u)];
            xp_t[ti] = (device const float4 *)(arena + p.x + (p.row0 + route / p.top_k) * p.xs);
        }
        float4 acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        for (uint k4 = l16 * 4u; k4 < p.k / 4u; k4 += INKLING_NVFP4_SUBSIMD_LANES * 4u) {
            const uint b = k4 / 4u;
            uint2 raw[4];
            float sc[4];
            #pragma unroll
            for (uint c = 0u; c < 4u; ++c) {
                const uchar sc_code = sp[b + c * sstep];
                sc[c] = Lut ? mk_e4m3_f32_lut[sc_code] : mk_expert_e4m3(sc_code);
                raw[c] = wp[b + c * wstep];
            }
            #pragma unroll
            for (uint ti = 0u; ti < 4u; ++ti) {
                if (ti < n_cur) {
                    const float4 x0 = xp_t[ti][k4], x1 = xp_t[ti][k4 + 1u], x2 = xp_t[ti][k4 + 2u], x3 = xp_t[ti][k4 + 3u];
                    #pragma unroll
                    for (uint c = 0u; c < 4u; ++c) {
                        acc[ti][c] = fma(mk_nvfp4_dot16<Lut>(raw[c], x0, x1, x2, x3), sc[c], acc[ti][c]);
                    }
                }
            }
        }
        #pragma unroll
        for (uint ti = 0u; ti < 4u; ++ti) {
            if (ti < n_cur) {
                #pragma unroll
                for (uint c = 0u; c < 4u; ++c) {
                    acc[ti][c] += simd_shuffle_xor(acc[ti][c], ushort(8));
                    acc[ti][c] += simd_shuffle_xor(acc[ti][c], ushort(4));
                    acc[ti][c] += simd_shuffle_xor(acc[ti][c], ushort(2));
                    acc[ti][c] += simd_shuffle_xor(acc[ti][c], ushort(1));
                }
            }
        }
        if (l16 == 0u) {
            #pragma unroll
            for (uint ti = 0u; ti < 4u; ++ti) {
                if (ti < n_cur) {
                    const uint route = entries[t0 + ti];
                    const uint tok_row = p.row0 + route / p.top_k;
                    const uint slot = route % p.top_k;
                    const float4 v = acc[ti] * outer;
                    device float *out = arena + p.y + tok_row * p.ys + slot * p.n + j;
                    device bfloat *out16 = reinterpret_cast<device bfloat *>(arena + p.y + tok_row * p.ys) + slot * p.n + j;
                    for (uint c = 0u; c < 2u; ++c) {
                        float gate = v[c * 2u], up = v[c * 2u + 1u];
                        if (p.round_bf16) { gate = mk_bf16(gate); up = mk_bf16(up); }
                        const float value = ((tanh(gate / p.gate_cap) * p.gate_cap) * mk_sigmoid(gate)) * (tanh(up / p.up_cap) * p.up_cap);
                        if constexpr (Bf16Out) out16[c] = bfloat(p.round_bf16 ? mk_bf16(value) : value);
                        else out[c] = p.round_bf16 ? mk_bf16(value) : value;
                    }
                }
            }
        }
    }
}

inline void mk_expert_grouped_down_store_weighted(device const float *arena, device float *partial,
                                                  constant MKExpert &p, uint route, uint j, float4 acc) {
    const uint tok_row = p.row0 + route / p.top_k;
    const uint slot = route % p.top_k;
    const float rw = arena[p.routes + tok_row * p.rs + p.top_k + slot];
    acc *= rw;
    reinterpret_cast<device float4 *>(partial)[ulong(route) * (p.n / 4u) + (j / 4u)] = acc;
}

template <bool Lut, bool Bf16Io>
inline void mk_expert_grouped_down_nv8_impl(device float *arena, device const uchar *w, device const uchar *s,
                                            device const uchar *amax, device const uchar *denom,
                                            device const uint *lists, device float *partial,
                                            constant MKExpert &p, uint gid) {
    const uint cols4 = p.n / 4u;
    const uint sub8_id = gid >> 3;
    const uint l8 = gid & 7u;
    const uint tile_idx = sub8_id / cols4;
    const uint tile_count = lists[p.experts + 1u];
    if (tile_idx >= tile_count) return;
    const uint j = (sub8_id % cols4) * 4u;

    const uint total = p.rows * p.top_k;
    device const uint *expert_off = lists;
    device const uint *sorted_routes = lists + p.experts + 2u;
    device const uint *tile_e = sorted_routes + total;
    const uint expert = tile_e[tile_idx];
    const uint off = expert_off[expert];
    const uint count = expert_off[expert + 1u] - off;
    device const uint *entries = sorted_routes + off;

    const ulong base = (ulong(expert) * p.n + j) * p.k;
    device const uint2 *wp = (device const uint2 *)(w + p.w + base / 2u);
    device const uchar *sp = s + p.s + base / 16u;
    const uint wstep = p.k / 16u;
    const uint sstep = p.k / 16u;
    const float outer = mk_load(amax, p.amax, 0u, expert) / mk_load(denom, p.denominator, 0u, 0u);

    if (count == 1u) {
        const uint route = entries[0];
        const uint tok_row = p.row0 + route / p.top_k, slot = route % p.top_k;
        device const float4 *xp = Bf16Io ? nullptr : (device const float4 *)(arena + p.x + tok_row * p.xs + slot * p.k);
        device const bfloat4 *xp16 = Bf16Io ? (reinterpret_cast<device const bfloat4 *>(arena + p.x + tok_row * p.xs) + (slot * p.k) / 4u) : nullptr;
        float4 acc = 0.0f;
        for (uint k4 = l8 * 4u; k4 < p.k / 4u; k4 += INKLING_NVFP4_SUBSIMD8_LANES * 4u) {
            const float4 x0 = Bf16Io ? float4(xp16[k4]) : xp[k4];
            const float4 x1 = Bf16Io ? float4(xp16[k4 + 1u]) : xp[k4 + 1u];
            const float4 x2 = Bf16Io ? float4(xp16[k4 + 2u]) : xp[k4 + 2u];
            const float4 x3 = Bf16Io ? float4(xp16[k4 + 3u]) : xp[k4 + 3u];
            const uint b = k4 / 4u;
            #pragma unroll
            for (uint c = 0u; c < 4u; ++c) {
                const uchar sc_code = sp[b + c * sstep];
                const float sc = Lut ? mk_e4m3_f32_lut[sc_code] : mk_expert_e4m3(sc_code);
                acc[c] = fma(mk_nvfp4_dot16<Lut>(wp[b + c * wstep], x0, x1, x2, x3), sc, acc[c]);
            }
        }
        #pragma unroll
        for (uint c = 0u; c < 4u; ++c) {
            acc[c] += simd_shuffle_xor(acc[c], ushort(4));
            acc[c] += simd_shuffle_xor(acc[c], ushort(2));
            acc[c] += simd_shuffle_xor(acc[c], ushort(1));
        }
        if (l8 == 0u) {
            acc *= outer;
            if (p.round_bf16) for (uint c = 0u; c < 4u; ++c) acc[c] = mk_bf16(acc[c]);
            mk_expert_grouped_down_store_weighted(arena, partial, p, route, j, acc);
        }
        return;
    }

    if (count == 2u) {
        const uint r0 = entries[0], r1 = entries[1];
        const uint tr0 = p.row0 + r0 / p.top_k, sl0 = r0 % p.top_k;
        const uint tr1 = p.row0 + r1 / p.top_k, sl1 = r1 % p.top_k;
        device const float4 *xp0 = Bf16Io ? nullptr : (device const float4 *)(arena + p.x + tr0 * p.xs + sl0 * p.k);
        device const float4 *xp1 = Bf16Io ? nullptr : (device const float4 *)(arena + p.x + tr1 * p.xs + sl1 * p.k);
        device const bfloat4 *xp16_0 = Bf16Io ? (reinterpret_cast<device const bfloat4 *>(arena + p.x + tr0 * p.xs) + (sl0 * p.k) / 4u) : nullptr;
        device const bfloat4 *xp16_1 = Bf16Io ? (reinterpret_cast<device const bfloat4 *>(arena + p.x + tr1 * p.xs) + (sl1 * p.k) / 4u) : nullptr;
        float4 acc0 = 0.0f, acc1 = 0.0f;
        for (uint k4 = l8 * 4u; k4 < p.k / 4u; k4 += INKLING_NVFP4_SUBSIMD8_LANES * 4u) {
            const float4 x00 = Bf16Io ? float4(xp16_0[k4]) : xp0[k4];
            const float4 x01 = Bf16Io ? float4(xp16_0[k4 + 1u]) : xp0[k4 + 1u];
            const float4 x02 = Bf16Io ? float4(xp16_0[k4 + 2u]) : xp0[k4 + 2u];
            const float4 x03 = Bf16Io ? float4(xp16_0[k4 + 3u]) : xp0[k4 + 3u];
            const float4 x10 = Bf16Io ? float4(xp16_1[k4]) : xp1[k4];
            const float4 x11 = Bf16Io ? float4(xp16_1[k4 + 1u]) : xp1[k4 + 1u];
            const float4 x12 = Bf16Io ? float4(xp16_1[k4 + 2u]) : xp1[k4 + 2u];
            const float4 x13 = Bf16Io ? float4(xp16_1[k4 + 3u]) : xp1[k4 + 3u];
            const uint b = k4 / 4u;
            #pragma unroll
            for (uint c = 0u; c < 4u; ++c) {
                const uchar sc_code = sp[b + c * sstep];
                const float sc = Lut ? mk_e4m3_f32_lut[sc_code] : mk_expert_e4m3(sc_code);
                float4 d0, d1, d2, d3;
                mk_nvfp4_decode16<Lut>(wp[b + c * wstep], d0, d1, d2, d3);
                acc0[c] = fma(dot(d0, x00) + dot(d1, x01) + dot(d2, x02) + dot(d3, x03), sc, acc0[c]);
                acc1[c] = fma(dot(d0, x10) + dot(d1, x11) + dot(d2, x12) + dot(d3, x13), sc, acc1[c]);
            }
        }
        #pragma unroll
        for (uint c = 0u; c < 4u; ++c) {
            acc0[c] += simd_shuffle_xor(acc0[c], ushort(4));
            acc0[c] += simd_shuffle_xor(acc0[c], ushort(2));
            acc0[c] += simd_shuffle_xor(acc0[c], ushort(1));
            acc1[c] += simd_shuffle_xor(acc1[c], ushort(4));
            acc1[c] += simd_shuffle_xor(acc1[c], ushort(2));
            acc1[c] += simd_shuffle_xor(acc1[c], ushort(1));
        }
        if (l8 == 0u) {
            acc0 *= outer;
            acc1 *= outer;
            if (p.round_bf16) {
                for (uint c = 0u; c < 4u; ++c) { acc0[c] = mk_bf16(acc0[c]); acc1[c] = mk_bf16(acc1[c]); }
            }
            mk_expert_grouped_down_store_weighted(arena, partial, p, r0, j, acc0);
            mk_expert_grouped_down_store_weighted(arena, partial, p, r1, j, acc1);
        }
        return;
    }

    for (uint t0 = 0u; t0 < count; t0 += 4u) {
        const uint n_cur = min(4u, count - t0);
        device const float4 *xp_t[4];
        device const bfloat4 *xp16_t[4];
        #pragma unroll
        for (uint ti = 0u; ti < 4u; ++ti) {
            const uint route = entries[t0 + min(ti, n_cur - 1u)];
            const uint tok_row = p.row0 + route / p.top_k, slot = route % p.top_k;
            xp_t[ti] = Bf16Io ? nullptr : (device const float4 *)(arena + p.x + tok_row * p.xs + slot * p.k);
            xp16_t[ti] = Bf16Io ? (reinterpret_cast<device const bfloat4 *>(arena + p.x + tok_row * p.xs) + (slot * p.k) / 4u) : nullptr;
        }
        float4 acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        for (uint k4 = l8 * 4u; k4 < p.k / 4u; k4 += INKLING_NVFP4_SUBSIMD8_LANES * 4u) {
            const uint b = k4 / 4u;
            #pragma unroll
            for (uint c = 0u; c < 4u; ++c) {
                const uchar sc_code = sp[b + c * sstep];
                const float sc = Lut ? mk_e4m3_f32_lut[sc_code] : mk_expert_e4m3(sc_code);
                float4 d0, d1, d2, d3;
                mk_nvfp4_decode16<Lut>(wp[b + c * wstep], d0, d1, d2, d3);
                #pragma unroll
                for (uint ti = 0u; ti < 4u; ++ti) {
                    if (ti < n_cur) {
                        const float4 x0 = Bf16Io ? float4(xp16_t[ti][k4]) : xp_t[ti][k4];
                        const float4 x1 = Bf16Io ? float4(xp16_t[ti][k4 + 1u]) : xp_t[ti][k4 + 1u];
                        const float4 x2 = Bf16Io ? float4(xp16_t[ti][k4 + 2u]) : xp_t[ti][k4 + 2u];
                        const float4 x3 = Bf16Io ? float4(xp16_t[ti][k4 + 3u]) : xp_t[ti][k4 + 3u];
                        acc[ti][c] = fma(dot(d0, x0) + dot(d1, x1) + dot(d2, x2) + dot(d3, x3), sc, acc[ti][c]);
                    }
                }
            }
        }
        #pragma unroll
        for (uint ti = 0u; ti < 4u; ++ti) {
            if (ti < n_cur) {
                #pragma unroll
                for (uint c = 0u; c < 4u; ++c) {
                    acc[ti][c] += simd_shuffle_xor(acc[ti][c], ushort(4));
                    acc[ti][c] += simd_shuffle_xor(acc[ti][c], ushort(2));
                    acc[ti][c] += simd_shuffle_xor(acc[ti][c], ushort(1));
                }
            }
        }
        if (l8 == 0u) {
            #pragma unroll
            for (uint ti = 0u; ti < 4u; ++ti) {
                if (ti < n_cur) {
                    float4 v = acc[ti] * outer;
                    if (p.round_bf16) for (uint c = 0u; c < 4u; ++c) v[c] = mk_bf16(v[c]);
                    mk_expert_grouped_down_store_weighted(arena, partial, p, entries[t0 + ti], j, v);
                }
            }
        }
    }
}

template <bool Lut, bool Bf16Out>
inline void mk_expert_grouped_up_mx_impl(device float *arena, device const uchar *w, device const uchar *s,
                                         device const uint *lists, constant MKExpert &p,
                                         uint gid, uint lane) {
    const uint columns = p.n / 2u;
    const uint simd_id = gid / 32u;
    const uint tile_idx = simd_id / columns;
    const uint tile_count = lists[p.experts + 1u];
    if (tile_idx >= tile_count) return;
    const uint j = (simd_id % columns) * 2u;

    const uint total = p.rows * p.top_k;
    device const uint *expert_off = lists;
    device const uint *sorted_routes = lists + p.experts + 2u;
    device const uint *tile_e = sorted_routes + total;
    const uint expert = tile_e[tile_idx];
    const uint off = expert_off[expert];
    const uint count = expert_off[expert + 1u] - off;
    device const uint *entries = sorted_routes + off;

    const ulong base = (ulong(expert) * p.n + j) * 2u * p.k;
    device const uint2 *wp = (device const uint2 *)(w + p.w + base);
    device const uchar *sp = s + p.s + base / 32u;
    const uint wstep = p.k / 8u;
    const uint sstep = p.k / 32u;

    for (uint t0 = 0u; t0 < count; t0 += 4u) {
        const uint n_cur = min(4u, count - t0);
        device const float4 *xp_t[4];
        #pragma unroll
        for (uint ti = 0u; ti < 4u; ++ti) {
            const uint route = entries[t0 + min(ti, n_cur - 1u)];
            xp_t[ti] = (device const float4 *)(arena + p.x + (p.row0 + route / p.top_k) * p.xs);
        }
        float4 acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        for (uint k8 = lane; k8 < p.k / 8u; k8 += 32u) {
            const uint sb = k8 / 4u;
            uint2 packed[4];
            float scale[4];
            #pragma unroll
            for (uint c = 0u; c < 4u; ++c) {
                const uint e = sp[sb + c * sstep];
                scale[c] = as_type<float>(e == 0u ? 0x00400000u : e == 255u ? 0x7fc00000u : e << 23);
                packed[c] = wp[k8 + c * wstep];
            }
            #pragma unroll
            for (uint ti = 0u; ti < 4u; ++ti) {
                if (ti < n_cur) {
                    const float4 x0 = xp_t[ti][2u * k8];
                    const float4 x1 = xp_t[ti][2u * k8 + 1u];
                    #pragma unroll
                    for (uint c = 0u; c < 4u; ++c) {
                        const float part = mk_e4m3x4_dot<Lut>(packed[c].x, x0) + mk_e4m3x4_dot<Lut>(packed[c].y, x1);
                        acc[ti][c] = fma(part, scale[c], acc[ti][c]);
                    }
                }
            }
        }
        #pragma unroll
        for (uint ti = 0u; ti < 4u; ++ti) {
            if (ti < n_cur) {
                acc[ti] = float4(simd_sum(acc[ti].x), simd_sum(acc[ti].y), simd_sum(acc[ti].z), simd_sum(acc[ti].w));
            }
        }
        if (lane == 0u) {
            #pragma unroll
            for (uint ti = 0u; ti < 4u; ++ti) {
                if (ti < n_cur) {
                    const uint route = entries[t0 + ti];
                    const uint tok_row = p.row0 + route / p.top_k;
                    const uint slot = route % p.top_k;
                    device float *out = arena + p.y + tok_row * p.ys + slot * p.n + j;
                    device bfloat *out16 = reinterpret_cast<device bfloat *>(arena + p.y + tok_row * p.ys) + slot * p.n + j;
                    for (uint c = 0u; c < 2u; ++c) {
                        float gate = acc[ti][c * 2u], up = acc[ti][c * 2u + 1u];
                        if (p.round_bf16) { gate = mk_bf16(gate); up = mk_bf16(up); }
                        const float value = ((tanh(gate / p.gate_cap) * p.gate_cap) * mk_sigmoid(gate)) * (tanh(up / p.up_cap) * p.up_cap);
                        if constexpr (Bf16Out) out16[c] = bfloat(p.round_bf16 ? mk_bf16(value) : value);
                        else out[c] = p.round_bf16 ? mk_bf16(value) : value;
                    }
                }
            }
        }
    }
}

template <bool Lut, bool Bf16Io>
inline void mk_expert_grouped_down_mx_impl(device float *arena, device const uchar *w, device const uchar *s,
                                           device const uint *lists, device float *partial,
                                           constant MKExpert &p, uint gid, uint lane) {
    const uint columns = p.n / 4u;
    const uint simd_id = gid / 32u;
    const uint tile_idx = simd_id / columns;
    const uint tile_count = lists[p.experts + 1u];
    if (tile_idx >= tile_count) return;
    const uint j = (simd_id % columns) * 4u;

    const uint total = p.rows * p.top_k;
    device const uint *expert_off = lists;
    device const uint *sorted_routes = lists + p.experts + 2u;
    device const uint *tile_e = sorted_routes + total;
    const uint expert = tile_e[tile_idx];
    const uint off = expert_off[expert];
    const uint count = expert_off[expert + 1u] - off;
    device const uint *entries = sorted_routes + off;

    const ulong base = (ulong(expert) * p.n + j) * p.k;
    device const uint *wp = (device const uint *)(w + p.w + base);
    device const uchar *sp = s + p.s + base / 32u;
    const uint wstep = p.k / 4u;
    const uint sstep = p.k / 32u;

    for (uint t0 = 0u; t0 < count; t0 += 4u) {
        const uint n_cur = min(4u, count - t0);
        device const float4 *xp_t[4];
        device const bfloat4 *xp16_t[4];
        #pragma unroll
        for (uint ti = 0u; ti < 4u; ++ti) {
            const uint route = entries[t0 + min(ti, n_cur - 1u)];
            const uint tok_row = p.row0 + route / p.top_k, slot = route % p.top_k;
            xp_t[ti] = Bf16Io ? nullptr : (device const float4 *)(arena + p.x + tok_row * p.xs + slot * p.k);
            xp16_t[ti] = Bf16Io ? (reinterpret_cast<device const bfloat4 *>(arena + p.x + tok_row * p.xs) + (slot * p.k) / 4u) : nullptr;
        }
        float4 acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        for (uint k4 = lane; k4 < p.k / 4u; k4 += 32u) {
            const uint sb = k4 / 8u;
            uint packed[4];
            float scale[4];
            #pragma unroll
            for (uint c = 0u; c < 4u; ++c) {
                const uint e = sp[sb + c * sstep];
                scale[c] = as_type<float>(e == 0u ? 0x00400000u : e == 255u ? 0x7fc00000u : e << 23);
                packed[c] = wp[k4 + c * wstep];
            }
            #pragma unroll
            for (uint ti = 0u; ti < 4u; ++ti) {
                if (ti < n_cur) {
                    const float4 xv = Bf16Io ? float4(xp16_t[ti][k4]) : xp_t[ti][k4];
                    #pragma unroll
                    for (uint c = 0u; c < 4u; ++c) {
                        acc[ti][c] = fma(mk_e4m3x4_dot<Lut>(packed[c], xv), scale[c], acc[ti][c]);
                    }
                }
            }
        }
        #pragma unroll
        for (uint ti = 0u; ti < 4u; ++ti) {
            if (ti < n_cur) {
                acc[ti] = float4(simd_sum(acc[ti].x), simd_sum(acc[ti].y), simd_sum(acc[ti].z), simd_sum(acc[ti].w));
            }
        }
        if (lane == 0u) {
            #pragma unroll
            for (uint ti = 0u; ti < 4u; ++ti) {
                if (ti < n_cur) {
                    float4 v = acc[ti];
                    if (p.round_bf16) for (uint c = 0u; c < 4u; ++c) v[c] = mk_bf16(v[c]);
                    mk_expert_grouped_down_store_weighted(arena, partial, p, entries[t0 + ti], j, v);
                }
            }
        }
    }
}

#define MK_EXPERT_GROUPED_SIMD(NAME, FN, LUT, BF16IO) \
    kernel void NAME(device float *arena [[buffer(0)]], device const uchar *w [[buffer(1)]], \
                     device const uchar *s [[buffer(2)]], device const uchar *amax [[buffer(3)]], \
                     device const uchar *denom [[buffer(4)]], device const uint *lists [[buffer(5)]], \
                     device float *partial [[buffer(6)]], constant MKExpert &p [[buffer(8)]], \
                     uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) { \
        (void)partial; \
        FN<LUT, BF16IO>(arena, w, s, amax, denom, lists, p, gid, lane); \
    }
#define MK_EXPERT_GROUPED_MX_UP(NAME, LUT, BF16IO) \
    kernel void NAME(device float *arena [[buffer(0)]], device const uchar *w [[buffer(1)]], \
                     device const uchar *s [[buffer(2)]], device const uchar *amax [[buffer(3)]], \
                     device const uchar *denom [[buffer(4)]], device const uint *lists [[buffer(5)]], \
                     device float *partial [[buffer(6)]], constant MKExpert &p [[buffer(8)]], \
                     uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) { \
        (void)amax; (void)denom; (void)partial; \
        mk_expert_grouped_up_mx_impl<LUT, BF16IO>(arena, w, s, lists, p, gid, lane); \
    }
#define MK_EXPERT_GROUPED_DOWN_NV8(NAME, LUT, BF16IO) \
    kernel void NAME(device float *arena [[buffer(0)]], device const uchar *w [[buffer(1)]], \
                     device const uchar *s [[buffer(2)]], device const uchar *amax [[buffer(3)]], \
                     device const uchar *denom [[buffer(4)]], device const uint *lists [[buffer(5)]], \
                     device float *partial [[buffer(6)]], constant MKExpert &p [[buffer(8)]], \
                     uint gid [[thread_position_in_grid]]) { \
        mk_expert_grouped_down_nv8_impl<LUT, BF16IO>(arena, w, s, amax, denom, lists, partial, p, gid); \
    }
#define MK_EXPERT_GROUPED_MX_DOWN(NAME, LUT, BF16IO) \
    kernel void NAME(device float *arena [[buffer(0)]], device const uchar *w [[buffer(1)]], \
                     device const uchar *s [[buffer(2)]], device const uchar *amax [[buffer(3)]], \
                     device const uchar *denom [[buffer(4)]], device const uint *lists [[buffer(5)]], \
                     device float *partial [[buffer(6)]], constant MKExpert &p [[buffer(8)]], \
                     uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) { \
        (void)amax; (void)denom; \
        mk_expert_grouped_down_mx_impl<LUT, BF16IO>(arena, w, s, lists, partial, p, gid, lane); \
    }

MK_EXPERT_GROUPED_SIMD(mk_expert_grouped_up_nv16, mk_expert_grouped_up_nv16_impl, false, false)
MK_EXPERT_GROUPED_SIMD(mk_expert_grouped_up_nv16_bf16, mk_expert_grouped_up_nv16_impl, false, true)
MK_EXPERT_GROUPED_SIMD(mk_expert_grouped_up_nv16_lut, mk_expert_grouped_up_nv16_impl, true, false)
MK_EXPERT_GROUPED_SIMD(mk_expert_grouped_up_nv16_lut_bf16, mk_expert_grouped_up_nv16_impl, true, true)
MK_EXPERT_GROUPED_DOWN_NV8(mk_expert_grouped_down_nv8, false, false)
MK_EXPERT_GROUPED_DOWN_NV8(mk_expert_grouped_down_nv8_bf16, false, true)
MK_EXPERT_GROUPED_DOWN_NV8(mk_expert_grouped_down_nv8_lut, true, false)
MK_EXPERT_GROUPED_DOWN_NV8(mk_expert_grouped_down_nv8_lut_bf16, true, true)
MK_EXPERT_GROUPED_MX_UP(mk_expert_grouped_up_mx, false, false)
MK_EXPERT_GROUPED_MX_UP(mk_expert_grouped_up_mx_bf16, false, true)
MK_EXPERT_GROUPED_MX_UP(mk_expert_grouped_up_mx_lut, true, false)
MK_EXPERT_GROUPED_MX_UP(mk_expert_grouped_up_mx_lut_bf16, true, true)
MK_EXPERT_GROUPED_MX_DOWN(mk_expert_grouped_down_mx, false, false)
MK_EXPERT_GROUPED_MX_DOWN(mk_expert_grouped_down_mx_bf16, false, true)
MK_EXPERT_GROUPED_MX_DOWN(mk_expert_grouped_down_mx_lut, true, false)
MK_EXPERT_GROUPED_MX_DOWN(mk_expert_grouped_down_mx_lut_bf16, true, true)
#undef MK_EXPERT_GROUPED_SIMD
#undef MK_EXPERT_GROUPED_MX_UP
#undef MK_EXPERT_GROUPED_DOWN_NV8
#undef MK_EXPERT_GROUPED_MX_DOWN

// ---- Qwegn Wave 2 kernels: fused sliding/dynamic QKVR prep + KV f32/u16 store ----
inline float4 mk_bf16_vec4(float4 v) {
    return float4(mk_bf16(v.x), mk_bf16(v.y), mk_bf16(v.z), mk_bf16(v.w));
}

template <bool Sconv>
inline void mk_inkling_qkvr_prep_impl(device float *arena, device float *k_ring, device float *v_ring,
                                      device float *keys_f32, device ushort *keys_u16,
                                      device float *vals_f32, device ushort *vals_u16,
                                      device const uchar *wq_scale, device const uchar *wr_scale,
                                      device const uchar *wk_w, device const uchar *wv_w,
                                      constant MKX &x, uint gid, uint lane) {
    constant MK &p = x.p;
    const uint units = (p.rot > 0u ? 2u : 1u) * p.heads + 2u * p.kv_heads;
    const uint simd_id = gid / 32u;
    const uint r = simd_id / units;
    if (r >= p.rows) return;
    const uint u = simd_id % units;
    const uint row = p.row0 + r;
    const uint position = p.pos + row;

    if (u < p.heads) {
        const uint h = u;
        const device float *qp_in = arena + p.x + row * p.xs + h * 128u + lane;
        device float *qp_out = arena + p.x + row * p.xs + h * 128u + lane;
        const ulong s_base = ulong(h) * 128u + lane;
        float val[4];
        #pragma unroll
        for (uint j = 0u; j < 4u; ++j) {
            val[j] = qp_in[j * 32u] * mk_load(wq_scale, x.mat[0].w, x.mat[0].sdtype, s_base + j * 32u);
        }
        const float total = (simd_sum(val[0] * val[0]) + simd_sum(val[1] * val[1])) +
                            (simd_sum(val[2] * val[2]) + simd_sum(val[3] * val[3]));
        const float factor = min(as_type<float>(x.mat[0].m) / (sqrt(total) + 1e-6f), 1.0f);
        const float lscale = Sconv ? 1.0f : (1.0f + p.scale * metal::precise::log(max(float(position + 1u) / float(p.m), 1.0f)));
        #pragma unroll
        for (uint j = 0u; j < 4u; ++j) {
            float res = mk_bf16(val[j] * factor);
            if constexpr (!Sconv) res = mk_bf16(res * lscale);
            qp_out[j * 32u] = res;
        }
        return;
    }

    if (u < p.heads + p.kv_heads) {
        const uint kvh = u - p.heads;
        const uint kv_width = p.kv_heads * 128u;
        const uint kv_base = kvh * 128u + lane;
        float val[4];
        if constexpr (Sconv) {
            const uint ring_cap = x.mat[2].m;
            const device float *in_base = arena + x.mat[2].y;
            const uint slot = position % ring_cap;
            const uint dtype = x.mat[2].sdtype;
            #pragma unroll
            for (uint j = 0u; j < 4u; ++j) {
                const uint c = kv_base + j * 32u;
                const float xcur = in_base[row * x.mat[2].ys + c];
                k_ring[ulong(slot) * kv_width + c] = xcur;
                float acc = 0.0f;
                for (uint k = 0u; k < p.taps; ++k) {
                    const uint back = p.taps - 1u - k;
                    float tap = 0.0f;
                    if (back == 0u) {
                        tap = xcur;
                    } else if (back <= r) {
                        tap = in_base[(row - back) * x.mat[2].ys + c];
                    } else if (back <= position) {
                        tap = k_ring[ulong(inkling_ring_back(slot, back, ring_cap)) * kv_width + c];
                    }
                    if (dtype == 1u) tap = float(half(tap));
                    else if (dtype == 2u) tap = mk_bf16(tap);
                    acc += mk_load(wk_w, x.mat[2].w, dtype, ulong(c) * p.taps + k) * tap;
                }
                if (dtype == 1u) acc = float(half(acc));
                else if (dtype == 2u) acc = mk_bf16(acc);
                val[j] = mk_bf16((acc + xcur) * p.scale);
            }
        } else {
            const device float *ck_in = arena + x.mat[2].y + row * x.mat[2].ys + kv_base;
            #pragma unroll
            for (uint j = 0u; j < 4u; ++j) val[j] = ck_in[j * 32u];
        }
        const float total = (simd_sum(val[0] * val[0]) + simd_sum(val[1] * val[1])) +
                            (simd_sum(val[2] * val[2]) + simd_sum(val[3] * val[3]));
        const float factor = min(as_type<float>(x.mat[1].pad) / (sqrt(total) + 1e-6f), 1.0f);
        const uint kv_slot = position % p.every;
        #pragma unroll
        for (uint j = 0u; j < 4u; ++j) {
            const float res = mk_bf16(val[j] * factor);
            const ulong dst_idx = ulong(kv_slot) * kv_width + kv_base + j * 32u;
            keys_f32[dst_idx] = res;
            const ushort h = ushort(as_type<uint>(res) >> 16u);
            keys_u16[dst_idx] = h;
            if (kv_slot < 16u) keys_u16[ulong(p.every + kv_slot) * kv_width + kv_base + j * 32u] = h;
        }
        return;
    }

    if (u < p.heads + 2u * p.kv_heads) {
        const uint kvh = u - (p.heads + p.kv_heads);
        const uint kv_width = p.kv_heads * 128u;
        const uint kv_elem_off = kvh * 128u + lane * 4u;
        float4 val;
        if constexpr (Sconv) {
            const uint ring_cap = x.mat[3].m;
            const device float *in_base = arena + x.mat[3].y;
            const float4 xcur = *reinterpret_cast<const device float4 *>(in_base + row * x.mat[3].ys + kv_elem_off);
            *reinterpret_cast<device float4 *>(v_ring + ulong(position % ring_cap) * kv_width + kv_elem_off) = xcur;
            const uint slot = position % ring_cap;
            const uint dtype = x.mat[3].sdtype;
            float4 acc = float4(0.0f);
            for (uint k = 0u; k < p.taps; ++k) {
                const uint back = p.taps - 1u - k;
                float4 tap = float4(0.0f);
                if (back == 0u) {
                    tap = xcur;
                } else if (back <= r) {
                    tap = *reinterpret_cast<const device float4 *>(in_base + (row - back) * x.mat[3].ys + kv_elem_off);
                } else if (back <= position) {
                    tap = *reinterpret_cast<const device float4 *>(v_ring + ulong(inkling_ring_back(slot, back, ring_cap)) * kv_width + kv_elem_off);
                }
                if (dtype == 1u) tap = float4(half4(tap));
                else if (dtype == 2u) tap = mk_bf16_vec4(tap);
                #pragma unroll
                for (uint j = 0u; j < 4u; ++j) {
                    acc[j] += mk_load(wv_w, x.mat[3].w, dtype, ulong(kv_elem_off + j) * p.taps + k) * tap[j];
                }
            }
            if (dtype == 1u) acc = float4(half4(acc));
            else if (dtype == 2u) acc = mk_bf16_vec4(acc);
            val = mk_bf16_vec4((acc + xcur) * p.scale);
        } else {
            val = *reinterpret_cast<const device float4 *>(arena + x.mat[3].y + row * x.mat[3].ys + kv_elem_off);
        }
        const float tot = simd_sum(dot(val, val));
        const float inv = metal::precise::rsqrt(tot / 128.0f + p.eps);
        val = mk_bf16_vec4(val * inv);
        const uint kv_slot = position % p.pad_k;
        const ulong dst_idx = ulong(kv_slot) * kv_width + kv_elem_off;
        *reinterpret_cast<device float4 *>(vals_f32 + dst_idx) = val;
        const ushort4 h4 = ushort4(as_type<uint4>(val) >> 16u);
        *reinterpret_cast<device ushort4 *>(vals_u16 + dst_idx) = h4;
        if (kv_slot < 16u) *reinterpret_cast<device ushort4 *>(vals_u16 + ulong(p.pad_k + kv_slot) * kv_width + kv_elem_off) = h4;
        return;
    }

    const uint h = u - (p.heads + 2u * p.kv_heads);
    const uint r_base = p.y + row * p.ys + h * 16u;
    const bool active = lane < 16u;
    float val = active ? arena[r_base + lane] * mk_load(wr_scale, x.mat[1].w, x.mat[1].sdtype, ulong(h) * 16u + lane) : 0.0f;
    const float total = simd_sum(active ? val * val : 0.0f);
    if (active) {
        const float factor = min(as_type<float>(x.mat[1].m) / (sqrt(total) + 1e-6f), 1.0f);
        val = mk_bf16(val * factor);
        if constexpr (!Sconv) {
            const float lscale = 1.0f + p.scale * metal::precise::log(max(float(position + 1u) / float(p.m), 1.0f));
            val = mk_bf16(val * lscale);
        }
        arena[r_base + lane] = val;
    }
}

kernel void mk_inkling_qkvr_sconv_fused(device float *arena [[buffer(0)]], device float *k_ring [[buffer(1)]],
                                        device float *v_ring [[buffer(2)]], device float *keys_f32 [[buffer(3)]],
                                        device ushort *keys_u16 [[buffer(4)]], device float *vals_f32 [[buffer(5)]],
                                        device ushort *vals_u16 [[buffer(6)]], device const uchar *weights [[buffer(7)]],
                                        constant MKX &x [[buffer(8)]], uint gid [[thread_position_in_grid]],
                                        uint lane [[thread_index_in_simdgroup]]) {
    mk_inkling_qkvr_prep_impl<true>(arena, k_ring, v_ring, keys_f32, keys_u16, vals_f32, vals_u16,
                                    weights, weights, weights, weights, x, gid, lane);
}

kernel void mk_inkling_qkvr_dyn_prep_fused(device float *arena [[buffer(0)]], device float *keys_f32 [[buffer(1)]],
                                           device ushort *keys_u16 [[buffer(2)]], device float *vals_f32 [[buffer(3)]],
                                           device ushort *vals_u16 [[buffer(4)]], device const uchar *wq_scale [[buffer(5)]],
                                           device const uchar *wr_scale [[buffer(6)]], constant MKX &x [[buffer(8)]],
                                           uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    mk_inkling_qkvr_prep_impl<false>(arena, nullptr, nullptr, keys_f32, keys_u16, vals_f32, vals_u16,
                                     wq_scale, wr_scale, nullptr, nullptr, x, gid, lane);
}

// ---- Step 3E kernels: fast single-pass log2 striped decode/verify attention (rows = 1..8) + gated merge ----
// Single-pass online softmax per SIMD in registers (~8.3 KB TG reduction instead of 30.9 KB);
// maxima stored in parts[*][128] are in log2 units and pair with mk_relative_long_merge_fast / _gated.
#define MK_LONG_FAST_SIMDS 8u
inline uint mk_long_fast_taps(uint visible) {
    const uint stripe = (visible <= 2048u) ? 64u : (visible <= 16384u) ? 256u : 512u;
    return (visible + stripe - 1u) / stripe;
}

template <bool InlineBias>
inline void mk_relative_long_keys_fast_impl(device const float *arena, device const ushort *key,
                                            device const ushort *val, device float *parts,
                                            device const uchar *rel_w_bytes, constant MK &p,
                                            uint lane, uint simd, uint simds, uint g,
                                            threadgroup float4 *red_acc, threadgroup float *red_m,
                                            threadgroup float *red_d) {
    const uint kv_parts = p.kv_heads * p.taps;
    const uint r = g / kv_parts, rem = g % kv_parts;
    if (r >= p.rows) return;
    const uint kvh = rem / p.taps, part = rem % p.taps;
    const uint row = p.row0 + r, pos = p.pos + row;
    const uint visible = p.window ? min(p.window, pos + 1u) : pos + 1u;
    const uint row_taps = (p.rows == 1u) ? p.taps : mk_long_fast_taps(visible);
    if (part >= row_taps) return;
    const uint first = pos + 1u - visible;
    const uint s_off = inkling_stripe_start(visible, row_taps, part);
    const uint e_off = inkling_stripe_end(visible, row_taps, part);
    const uint start = first + s_off;
    const uint n = e_off - s_off;
    const uint stride = p.kv_heads * 128u;
    const uint h0 = kvh * 3u;
    const float qscale = p.scale * M_LOG2E_F;
    float4 q[3];
    #pragma unroll
    for (uint h = 0u; h < 3u; ++h) {
        q[h] = float4(*(device const packed_float4 *)(arena + p.x + row * p.xs + (h0 + h) * 128u + lane * 4u)) * qscale;
    }

    const bool stripe_has_rel = (n > 0u) && ((pos - (start + n - 1u)) < p.m);
    float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f;
    const device ushort *rel_w = InlineBias ? reinterpret_cast<const device ushort *>(rel_w_bytes + p.w) : nullptr;
    if constexpr (InlineBias) {
        if (stripe_has_rel && lane < 16u) {
            const device float *rp = arena + p.z + row * p.zs;
            r0 = rp[(h0 + 0u) * 16u + lane];
            r1 = rp[(h0 + 1u) * 16u + lane];
            r2 = rp[(h0 + 2u) * 16u + lane];
        }
    }

    float m[3] = { -INFINITY, -INFINITY, -INFINITY };
    float d[3] = { 0.0f, 0.0f, 0.0f };
    float4 acc[3] = { float4(0.0f), float4(0.0f), float4(0.0f) };
    device const float *bias_base = InlineBias ? nullptr : (arena + p.z + row * p.zs + h0 * p.m);
    const uint k_off = kvh * 128u + lane * 4u;
    if (simd < n) {
        const bool linear_slot = (p.window == 0u);
        uint slot = linear_slot ? (start + simd) : ((start + simd) % p.capacity);
        const uint step = linear_slot ? simds : (simds % p.capacity);
        for (uint i = simd; i < n; i += simds) {
            const uint t = start + i;
            const ulong kv_idx = ulong(slot) * stride + k_off;
            const float4 k = mk_bf16x4(key, kv_idx);
            const float4 v = mk_bf16x4(val, kv_idx);
            slot += step;
            if (!linear_slot && slot >= p.capacity) slot -= p.capacity;
            const float s0 = simd_sum(dot(q[0], k));
            const float s1 = simd_sum(dot(q[1], k));
            const float s2 = simd_sum(dot(q[2], k));
            const uint distance = pos - t;
            float sc[3];
            if (stripe_has_rel && distance < p.m) {
                if constexpr (InlineBias) {
                    const float wk = (lane < 16u) ? as_type<float>(uint(rel_w[ulong(lane) * p.m + distance]) << 16u) : 0.0f;
                    const float b0 = mk_bf16(simd_sum(r0 * wk));
                    const float b1 = mk_bf16(simd_sum(r1 * wk));
                    const float b2 = mk_bf16(simd_sum(r2 * wk));
                    sc[0] = s0 + b0 * M_LOG2E_F;
                    sc[1] = s1 + b1 * M_LOG2E_F;
                    sc[2] = s2 + b2 * M_LOG2E_F;
                } else {
                    sc[0] = s0 + bias_base[distance] * M_LOG2E_F;
                    sc[1] = s1 + bias_base[p.m + distance] * M_LOG2E_F;
                    sc[2] = s2 + bias_base[2u * p.m + distance] * M_LOG2E_F;
                }
            } else {
                sc[0] = s0;
                sc[1] = s1;
                sc[2] = s2;
            }
            #pragma unroll
            for (uint h = 0u; h < 3u; ++h) {
                if (d[h] == 0.0f) {
                    m[h] = sc[h];
                    d[h] = 1.0f;
                    acc[h] = v;
                } else {
                    const float nm = max(m[h], sc[h]);
                    const float prev = fast::exp2(m[h] - nm);
                    const float w = fast::exp2(sc[h] - nm);
                    d[h] = d[h] * prev + w;
                    acc[h] = acc[h] * prev + w * v;
                    m[h] = nm;
                }
            }
        }
    }

    #pragma unroll
    for (uint h = 0u; h < 3u; ++h) {
        red_acc[(h * MK_LONG_FAST_SIMDS + simd) * 32u + lane] = acc[h];
        if (lane == 0u) {
            red_m[h * MK_LONG_FAST_SIMDS + simd] = m[h];
            red_d[h * MK_LONG_FAST_SIMDS + simd] = d[h];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd < 3u) {
        const uint h = simd;
        const uint off = h * MK_LONG_FAST_SIMDS;
        const float sd = (lane < simds) ? red_d[off + lane] : 0.0f;
        const float sm = (sd > 0.0f) ? red_m[off + lane] : -INFINITY;
        const float gm = simd_max(sm);
        const float scale_my = (sd > 0.0f) ? fast::exp2(sm - gm) : 0.0f;
        const float gd = simd_sum(sd * scale_my);
        float4 gout = float4(0.0f);
        for (uint s = 0u; s < simds; ++s) {
            const float ws = simd_shuffle(scale_my, ushort(s));
            gout = fma(red_acc[(off + s) * 32u + lane], ws, gout);
        }
        device float *dst = parts + ((ulong(r) * p.heads + (h0 + h)) * p.taps + part) * 130u;
        *(device packed_float4 *)(dst + lane * 4u) = packed_float4(gout);
        if (lane == 0u) {
            dst[128] = gm;
            dst[129] = gd;
        }
    }
}

kernel void mk_relative_long_keys_fast(device const float *arena [[buffer(0)]], device const ushort *key [[buffer(1)]],
                                       device const ushort *val [[buffer(2)]], device float *parts [[buffer(3)]],
                                       constant MK &p [[buffer(8)]], uint lane [[thread_index_in_simdgroup]],
                                       uint simd [[simdgroup_index_in_threadgroup]], uint simds [[simdgroups_per_threadgroup]],
                                       uint g [[threadgroup_position_in_grid]]) {
    threadgroup float4 red_acc[MK_LONG_FAST_SIMDS * 3u * 32u];
    threadgroup float red_m[MK_LONG_FAST_SIMDS * 3u];
    threadgroup float red_d[MK_LONG_FAST_SIMDS * 3u];
    mk_relative_long_keys_fast_impl<false>(arena, key, val, parts, nullptr, p, lane, simd, simds, g, red_acc, red_m, red_d);
}

kernel void mk_relative_long_keys_inline_fast(device const float *arena [[buffer(0)]], device const ushort *key [[buffer(1)]],
                                              device const ushort *val [[buffer(2)]], device float *parts [[buffer(3)]],
                                              device const uchar *rel_w [[buffer(4)]], constant MK &p [[buffer(8)]],
                                              uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]],
                                              uint simds [[simdgroups_per_threadgroup]], uint g [[threadgroup_position_in_grid]]) {
    threadgroup float4 red_acc[MK_LONG_FAST_SIMDS * 3u * 32u];
    threadgroup float red_m[MK_LONG_FAST_SIMDS * 3u];
    threadgroup float red_d[MK_LONG_FAST_SIMDS * 3u];
    mk_relative_long_keys_fast_impl<true>(arena, key, val, parts, rel_w, p, lane, simd, simds, g, red_acc, red_m, red_d);
}
#undef MK_LONG_FAST_SIMDS

template <bool Gated>
inline void mk_relative_long_merge_impl(device float *arena, device const float *parts, constant MK &p,
                                        uint tid, uint lane, uint simd, uint gid,
                                        threadgroup float *group, threadgroup float *factor) {
    const uint r = gid / p.heads, h = gid % p.heads;
    if (r >= p.rows) return;
    const uint row = p.row0 + r, pos = p.pos + row;
    const uint visible = p.window ? min(p.window, pos + 1u) : pos + 1u;
    const uint row_taps = (p.rows == 1u || (p.flags & MK_FIXED_TAPS) != 0u) ? p.taps : mk_long_fast_taps(visible);
    device const float *src = parts + (ulong(r) * p.heads + h) * p.taps * 130u;
    const float sink = arena[p.s + row * p.every + h] * M_LOG2E_F;
    float m = -INFINITY;
    for (uint k = tid; k < row_taps; k += 128u) m = max(m, src[ulong(k) * 130u + 128u]);
    m = simd_max(m);
    if (lane == 0u) group[simd] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = max(max(max(group[0], group[1]), max(group[2], group[3])), sink);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float d = 0.0f;
    for (uint k = tid; k < row_taps; k += 128u) {
        const float dk = src[ulong(k) * 130u + 129u];
        const float f = (dk > 0.0f) ? fast::exp2(src[ulong(k) * 130u + 128u] - m) : 0.0f;
        factor[k] = f;
        d += f * dk;
    }
    d = simd_sum(d);
    if (lane == 0u) group[simd] = d;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float denominator = fast::exp2(sink - m) + ((group[0] + group[1]) + (group[2] + group[3]));
    float acc = 0.0f;
    for (uint k = 0u; k < row_taps; ++k) acc += factor[k] * src[ulong(k) * 130u + tid];
    float val = acc / denominator;
    if constexpr (Gated) {
        if (p.flags & MK_ROUND_BF16) val = mk_bf16(val);
        const float res = val * mk_sigmoid(arena[p.z + row * p.zs + h * 128u + tid]);
        arena[p.y + row * p.ys + h * 128u + tid] = (p.flags & MK_ROUND_OUT) ? mk_bf16(res) : res;
    } else {
        arena[p.y + row * p.ys + h * 128u + tid] = (p.flags & MK_ROUND_OUT) ? mk_bf16(val) : val;
    }
}

kernel void mk_relative_long_merge_fast(device float *arena [[buffer(0)]], device const float *parts [[buffer(1)]],
                                        constant MK &p [[buffer(8)]], uint tid [[thread_index_in_threadgroup]],
                                        uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]],
                                        uint h [[threadgroup_position_in_grid]]) {
    threadgroup float group[4];
    threadgroup float factor[MK_LONG_PARTS];
    mk_relative_long_merge_impl<false>(arena, parts, p, tid, lane, simd, h, group, factor);
}

kernel void mk_relative_long_merge_gated(device float *arena [[buffer(0)]], device const float *parts [[buffer(1)]],
                                         constant MK &p [[buffer(8)]], uint tid [[thread_index_in_threadgroup]],
                                         uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]],
                                         uint h [[threadgroup_position_in_grid]]) {
    threadgroup float group[4];
    threadgroup float factor[MK_LONG_PARTS];
    mk_relative_long_merge_impl<true>(arena, parts, p, tid, lane, simd, h, group, factor);
}

// Shared N=64 expert block output/activation writer.
template <bool Down, bool Bf16Io>
inline void mk_expert_block_store(device float *arena, device float *partial, constant MKExpert &p,
                                  uint r, uint c, float lo, float hi) {
    if (p.round_bf16) {
        lo = mk_bf16(lo);
        hi = mk_bf16(hi);
    }
    if constexpr (Down) {
        if constexpr (Bf16Io) {
            device bfloat *out16 = reinterpret_cast<device bfloat *>(partial) + ulong(r) * p.n + c;
            out16[0] = bfloat(lo);
            out16[1] = bfloat(hi);
        } else {
            partial[ulong(r) * p.n + c] = lo;
            partial[ulong(r) * p.n + c + 1u] = hi;
        }
    } else {
        const float out = (tanh(lo / p.gate_cap) * p.gate_cap * mk_sigmoid(lo)) * (tanh(hi / p.up_cap) * p.up_cap);
        const float val = p.round_bf16 ? mk_bf16(out) : out;
        if constexpr (Bf16Io) {
            device bfloat *out16 = reinterpret_cast<device bfloat *>(arena + p.y + (p.row0 + r / p.top_k) * p.ys) + (r % p.top_k) * p.n + c / 2u;
            out16[0] = bfloat(val);
        } else {
            arena[p.y + (p.row0 + r / p.top_k) * p.ys + (r % p.top_k) * p.n + c / 2u] = val;
        }
    }
}

// Single-TG compact MoE tile builder (expert_off[E+1], tile_count[1], sorted_routes[total], tile_e[max_tiles], tile_r[max_tiles]).
kernel void mk_expert_group_tiles(device const float *arena [[buffer(0)]], device uint *lists [[buffer(1)]],
                                  constant MKExpert &p [[buffer(8)]], uint tid [[thread_index_in_threadgroup]],
                                  uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint E = p.experts;
    const uint total = p.rows * p.top_k;
    const uint max_tiles = inkling_moe_max_tiles(E, total);
    device uint *expert_off = lists;
    device uint *tile_count_ptr = lists + E + 1u;
    device uint *sorted_routes = lists + E + 2u;
    device uint *tile_e = sorted_routes + total;
    device uint *tile_r = tile_e + max_tiles;

    threadgroup atomic_uint tg_count[512];
    threadgroup uint tg_off[513];
    threadgroup uint tg_tile_base[513];
    for (uint e = tid; e < E; e += 256u) atomic_store_explicit(&tg_count[e], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < total; i += 256u) {
        const uint row = p.row0 + i / p.top_k;
        const uint slot = i % p.top_k;
        const uint e = uint(arena[p.routes + row * p.rs + slot]);
        if (e < E) atomic_fetch_add_explicit(&tg_count[e], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd == 0u) {
        const uint chunk = (E + 31u) / 32u;
        const uint s_e = lane * chunk;
        const uint e_e = min(E, s_e + chunk);
        uint my_cnt = 0u, my_tiles = 0u;
        for (uint e = s_e; e < e_e; ++e) {
            const uint c = atomic_load_explicit(&tg_count[e], memory_order_relaxed);
            my_cnt += c;
            if (c > 0u) my_tiles += (c >> 6u) + ((c & 63u) != 0u ? 1u : 0u);
        }
        const uint base_cnt = simd_prefix_exclusive_sum(my_cnt);
        const uint base_tile = simd_prefix_exclusive_sum(my_tiles);
        uint cur_c = base_cnt, cur_t = base_tile;
        for (uint e = s_e; e < e_e; ++e) {
            tg_off[e] = cur_c;
            tg_tile_base[e] = cur_t;
            const uint c = atomic_load_explicit(&tg_count[e], memory_order_relaxed);
            cur_c += c;
            if (c > 0u) cur_t += (c >> 6u) + ((c & 63u) != 0u ? 1u : 0u);
        }
        if (lane == 31u) {
            tg_off[E] = cur_c;
            tg_tile_base[E] = cur_t;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = tid; e <= E; e += 256u) expert_off[e] = tg_off[e];
    for (uint e = tid; e < E; e += 256u) {
        atomic_store_explicit(&tg_count[e], tg_off[e], memory_order_relaxed);
        const uint c = tg_off[e + 1u] - tg_off[e];
        if (c > 0u) {
            const uint t0 = tg_tile_base[e];
            const uint q = c >> 6u, rem = c & 63u;
            for (uint t = 0u; t < q; ++t) {
                tile_e[t0 + t] = e;
                tile_r[t0 + t] = inkling_moe_pack_tile_r(t, 64u);
            }
            if (rem > 0u) {
                const uint tbm = uint(inkling_moe_tile_block(int(rem)));
                tile_e[t0 + q] = e;
                tile_r[t0 + q] = inkling_moe_pack_tile_r((q * 64u) / tbm, tbm);
            }
        }
    }
    if (tid == 0u) *tile_count_ptr = tg_tile_base[E];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < total; i += 256u) {
        const uint row = p.row0 + i / p.top_k;
        const uint slot = i % p.top_k;
        const uint e = uint(arena[p.routes + row * p.rs + slot]);
        if (e < E) {
            const uint at = atomic_fetch_add_explicit(&tg_count[e], 1u, memory_order_relaxed);
            sorted_routes[at] = i;
        }
    }
}

kernel void mk_expert_combine_mm_bf16(device float *arena [[buffer(0)]], device const bfloat *partial [[buffer(1)]],
                                      constant MKExpert &p [[buffer(8)]], uint i [[thread_position_in_grid]]) {
    const uint cols4 = p.n / 4u;
    const uint r = i / cols4, j4 = i % cols4;
    if (r >= p.rows) return;
    device const float *routes = arena + p.routes + (p.row0 + r) * p.rs;
    device const bfloat4 *bp = reinterpret_cast<const device bfloat4 *>(partial);
    float4 sum = 0.0f;
    for (uint s = 0u; s < p.top_k; ++s) {
        if (uint(routes[s]) < p.experts) {
            const float rw = routes[p.top_k + s];
            const ulong idx4 = (ulong(r) * p.top_k + s) * cols4 + j4;
            sum += float4(bp[idx4]) * rw;
        }
    }
    device float4 *dst = reinterpret_cast<device float4 *>(arena + p.y + (p.row0 + r) * p.ys);
    dst[j4] = p.round_bf16 ? float4(mk_bf16(sum.x), mk_bf16(sum.y), mk_bf16(sum.z), mk_bf16(sum.w)) : sum;
}

kernel void mk_expert_combine_sum4(device float *arena [[buffer(0)]], device const float *partial [[buffer(1)]],
                                   constant MKExpert &p [[buffer(8)]], uint i [[thread_position_in_grid]]) {
    const uint cols4 = p.n / 4u;
    const uint r = i / cols4, j4 = i % cols4;
    if (r >= p.rows) return;
    device const float *routes = arena + p.routes + (p.row0 + r) * p.rs;
    device const float4 *p4 = reinterpret_cast<const device float4 *>(partial);
    float4 sum = 0.0f;
    for (uint s = 0u; s < p.top_k; ++s) {
        if (uint(routes[s]) < p.experts) sum += p4[(ulong(r) * p.top_k + s) * cols4 + j4];
    }
    device float4 *dst = reinterpret_cast<device float4 *>(arena + p.y + (p.row0 + r) * p.ys);
    dst[j4] = p.round_bf16 ? float4(mk_bf16(sum.x), mk_bf16(sum.y), mk_bf16(sum.z), mk_bf16(sum.w)) : sum;
}

// ---- Step 2 prefill kernel: half fragments, non-transposed staging + Lean far-band split ----
template <int BK>
inline void mk_relative_prefill_fast_impl(device float *arena, device const bfloat *k, device const bfloat *v,
                                          constant MK &p, uint tid, uint sg, uint lane, uint group_,
                                          threadgroup half *Ks, threadgroup half *Vs) {
    constexpr int D = 128, BQ = 32, TG = 128, LDV = D + 8;
    constexpr float LOG2E = 1.4426950408889634f;
    const int HQ = int(p.heads), HKV = int(p.kv_heads), R = int(p.m), WINDOW = int(p.window);
    const int L = int(p.rows), S = int(p.pos + p.row0 + p.rows);
    const int head = int(group_ % uint(HQ)), kv_head = head / (HQ / HKV);
    const int q_tiles = (L + BQ - 1) / BQ;
    const int q0 = (q_tiles - 1 - int(group_ / uint(HQ))) * BQ;
    const float qscale = p.scale * LOG2E;

    const short fm = ((lane / 4u) & 4u) + (lane / 2u) % 4u;
    const short fn = ((lane / 4u) & 2u) * 2u + (lane % 2u) * 2u;
    const int row = q0 + int(sg) * 8 + fm;
    const bool row_ok = row < L;
    const int diag = S - L + row;

    simdgroup_half8x8 q_frag[D / 8];
    #pragma unroll
    for (int db = 0; db < D / 8; ++db) {
        half2 qv = half2(0.0h);
        if (row_ok) {
            const device float *qp = arena + p.x + (p.row0 + row) * p.xs + head * D + db * 8 + fn;
            qv = half2(qp[0], qp[1]);
        }
        q_frag[db].thread_elements()[0] = qv.x;
        q_frag[db].thread_elements()[1] = qv.y;
    }
    simdgroup_float8x8 acc[D / 8];
    #pragma unroll
    for (int db = 0; db < D / 8; ++db) acc[db] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    float row_max = row_ok ? arena[p.s + (p.row0 + row) * p.every + head] * LOG2E : 0.0f;
    float row_sum = 1.0f;
    const int bias_row = row_ok ? row : 0;
    const device float *bias = arena + p.z + size_t(p.row0 + bias_row) * p.zs + head * R;
    const bool bias_packed = (p.flags & MK_BIAS16) != 0u;
    const device bfloat *bias16 = reinterpret_cast<const device bfloat *>(arena + p.z + size_t(p.row0 + bias_row) * p.zs) + head * R;

    const int first_diag = S - L + q0;
    const int last_diag = S - L + min(q0 + BQ, L) - 1;
    const int first_key = WINDOW > 0 ? max(0, first_diag - WINDOW + 1) : 0;
    const device bfloat *k_head = k + size_t(kv_head) * D;
    const device bfloat *v_head = v + size_t(kv_head) * D;
    const size_t kv_stride = size_t(HKV) * D;
    const uint capacity = p.capacity;

    const int far_end = (WINDOW == 0) ? int(inkling_far_band_end(uint(first_diag), uint(R), uint(BK))) : 0;
    const bool use_far = (far_end > 0) && (uint(far_end) <= capacity);
    const int near_start = use_far ? far_end : (first_key / BK * BK);

    // Phase 1: branchless, modulo-free, bias-free far band when WINDOW == 0.
    if (use_far) {
        for (int k0 = 0; k0 < far_end; k0 += BK) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int c = int(tid); c < BK * D / 8; c += TG) {
                const int r = c / (D / 8);
                const int col = (c % (D / 8)) * 8;
                const size_t base = size_t(uint(k0 + r)) * kv_stride + col;
                const auto kp = reinterpret_cast<const device bfloat4 *>(k_head + base);
                const auto vp = reinterpret_cast<const device bfloat4 *>(v_head + base);
                threadgroup half4 *kd = reinterpret_cast<threadgroup half4 *>(Ks + r * LDV + col);
                threadgroup half4 *vd = reinterpret_cast<threadgroup half4 *>(Vs + r * LDV + col);
                kd[0] = half4(float4(kp[0]));
                kd[1] = half4(float4(kp[1]));
                vd[0] = half4(float4(vp[0]));
                vd[1] = half4(float4(vp[1]));
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (q0 + int(sg) * 8 >= L) continue;

            simdgroup_float8x8 scores[BK / 8];
            #pragma unroll
            for (int kb = 0; kb < BK / 8; ++kb) {
                scores[kb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                #pragma unroll
                for (int db = 0; db < D / 8; ++db) {
                    simdgroup_half8x8 k_frag;
                    simdgroup_load(k_frag, Ks + kb * 8 * LDV + db * 8, LDV, ulong2(0), true);
                    simdgroup_multiply_accumulate(scores[kb], q_frag[db], k_frag, scores[kb]);
                }
            }
            float tile_max = -INFINITY;
            #pragma unroll
            for (int kb = 0; kb < BK / 8; ++kb) {
                #pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const float s = scores[kb].thread_elements()[e] * qscale;
                    scores[kb].thread_elements()[e] = s;
                    tile_max = max(tile_max, s);
                }
            }
            tile_max = max(tile_max, simd_shuffle_xor(tile_max, 1));
            tile_max = max(tile_max, simd_shuffle_xor(tile_max, 8));
            const float new_max = max(row_max, tile_max);
            const float factor = fast::exp2(row_max - new_max);
            float tile_sum = 0.0f;
            simdgroup_half8x8 probs[BK / 8];
            #pragma unroll
            for (int kb = 0; kb < BK / 8; ++kb) {
                #pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const float pr = fast::exp2(scores[kb].thread_elements()[e] - new_max);
                    probs[kb].thread_elements()[e] = half(pr);
                    tile_sum += pr;
                }
            }
            tile_sum += simd_shuffle_xor(tile_sum, 1);
            tile_sum += simd_shuffle_xor(tile_sum, 8);
            row_sum = row_sum * factor + tile_sum;
            row_max = new_max;
            #pragma unroll
            for (int db = 0; db < D / 8; ++db) {
                acc[db].thread_elements()[0] *= factor;
                acc[db].thread_elements()[1] *= factor;
                #pragma unroll
                for (int kb = 0; kb < BK / 8; ++kb) {
                    simdgroup_half8x8 v_frag;
                    simdgroup_load(v_frag, Vs + kb * 8 * LDV + db * 8, LDV);
                    simdgroup_multiply_accumulate(acc[db], probs[kb], v_frag, acc[db]);
                }
            }
        }
    }

    // Phase 2: causal / relative-bias / windowed tiles.
    uint slot0 = uint(near_start) % capacity;
    for (int k0 = near_start; k0 <= last_diag; k0 += BK, slot0 += uint(BK)) {
        if (slot0 >= capacity) slot0 %= capacity;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int c = int(tid); c < BK * D / 8; c += TG) {
            const int r = c / (D / 8);
            const int col = (c % (D / 8)) * 8;
            const int key = k0 + r;
            half4 kv0 = half4(0.0h), kv1 = half4(0.0h);
            half4 vv0 = half4(0.0h), vv1 = half4(0.0h);
            if (key < S) {
                uint slot = slot0 + uint(r);
                if (slot >= capacity) slot %= capacity;
                const size_t base = size_t(slot) * kv_stride + col;
                const auto kp = reinterpret_cast<const device bfloat4 *>(k_head + base);
                const auto vp = reinterpret_cast<const device bfloat4 *>(v_head + base);
                kv0 = half4(float4(kp[0]));
                kv1 = half4(float4(kp[1]));
                vv0 = half4(float4(vp[0]));
                vv1 = half4(float4(vp[1]));
            }
            threadgroup half4 *kd = reinterpret_cast<threadgroup half4 *>(Ks + r * LDV + col);
            threadgroup half4 *vd = reinterpret_cast<threadgroup half4 *>(Vs + r * LDV + col);
            kd[0] = kv0;
            kd[1] = kv1;
            vd[0] = vv0;
            vd[1] = vv1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (q0 + int(sg) * 8 >= L) continue;

        simdgroup_float8x8 scores[BK / 8];
        #pragma unroll
        for (int kb = 0; kb < BK / 8; ++kb) {
            scores[kb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            #pragma unroll
            for (int db = 0; db < D / 8; ++db) {
                simdgroup_half8x8 k_frag;
                simdgroup_load(k_frag, Ks + kb * 8 * LDV + db * 8, LDV, ulong2(0), true);
                simdgroup_multiply_accumulate(scores[kb], q_frag[db], k_frag, scores[kb]);
            }
        }
        float tile_max = -INFINITY;
        #pragma unroll
        for (int kb = 0; kb < BK / 8; ++kb) {
            #pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int key = k0 + kb * 8 + fn + e;
                const int d = diag - key;
                const bool visible = row_ok && d >= 0 && (WINDOW == 0 || d < WINDOW);
                float s = -INFINITY;
                if (visible) {
                    s = scores[kb].thread_elements()[e] * qscale;
                    if (d < R) {
                        const float b = bias_packed ? float(bias16[d]) : bias[d];
                        s += b * LOG2E;
                    }
                }
                scores[kb].thread_elements()[e] = s;
                tile_max = max(tile_max, s);
            }
        }
        tile_max = max(tile_max, simd_shuffle_xor(tile_max, 1));
        tile_max = max(tile_max, simd_shuffle_xor(tile_max, 8));
        const float new_max = max(row_max, tile_max);
        const float factor = fast::exp2(row_max - new_max);
        float tile_sum = 0.0f;
        simdgroup_half8x8 probs[BK / 8];
        #pragma unroll
        for (int kb = 0; kb < BK / 8; ++kb) {
            #pragma unroll
            for (int e = 0; e < 2; ++e) {
                const float pr = fast::exp2(scores[kb].thread_elements()[e] - new_max);
                probs[kb].thread_elements()[e] = half(pr);
                tile_sum += pr;
            }
        }
        tile_sum += simd_shuffle_xor(tile_sum, 1);
        tile_sum += simd_shuffle_xor(tile_sum, 8);
        row_sum = row_sum * factor + tile_sum;
        row_max = new_max;
        #pragma unroll
        for (int db = 0; db < D / 8; ++db) {
            acc[db].thread_elements()[0] *= factor;
            acc[db].thread_elements()[1] *= factor;
            #pragma unroll
            for (int kb = 0; kb < BK / 8; ++kb) {
                simdgroup_half8x8 v_frag;
                simdgroup_load(v_frag, Vs + kb * 8 * LDV + db * 8, LDV);
                simdgroup_multiply_accumulate(acc[db], probs[kb], v_frag, acc[db]);
            }
        }
    }

    if (row_ok) {
        device float *op = arena + p.y + (p.row0 + row) * p.ys + head * D;
        #pragma unroll
        for (int db = 0; db < D / 8; ++db) {
            op[db * 8 + fn] = float(acc[db].thread_elements()[0] / row_sum);
            op[db * 8 + fn + 1] = float(acc[db].thread_elements()[1] / row_sum);
        }
    }
}

kernel void mk_relative_prefill_fast(device float *arena [[buffer(0)]], device const bfloat *k [[buffer(1)]],
                                     device const bfloat *v [[buffer(2)]], constant MK &p [[buffer(8)]],
                                     uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                                     uint lane [[thread_index_in_simdgroup]], uint group_ [[threadgroup_position_in_grid]]) {
    alignas(8) threadgroup half Ks[16 * 136], Vs[16 * 136];
    mk_relative_prefill_fast_impl<16>(arena, k, v, p, tid, sg, lane, group_, Ks, Vs);
}

// ---- Final Step 1 expert optimization: M=64, N=64, B=32 (10.24 KB TG) + SLC-resident BF16 input packing ----
kernel void mk_pack_rows_bf16(device const float *arena [[buffer(0)]], device bfloat *dst [[buffer(1)]],
                              constant MKExpert &p [[buffer(8)]], uint gid [[thread_position_in_grid]]) {
    const uint k4 = p.k / 4u;
    const uint total = p.rows * k4;
    if (gid >= total) return;
    const uint r = gid / k4, c4 = gid % k4;
    const float4 v = reinterpret_cast<const device float4 *>(arena + p.x + (p.row0 + r) * p.xs)[c4];
    reinterpret_cast<device bfloat4 *>(dst)[gid] = bfloat4(v);
}

template <bool Down, bool Nv, uint M, bool X16, bool Lut, bool Bf16Io>
inline void mk_expert_block_m64n64_core(device float *arena, device const uchar *w, device const uchar *s,
                                          device const uchar *amax, device const uchar *denom, device const uint *entries,
                                          uint count, device float *partial, constant MKExpert &p, uint expert, uint c0,
                                          uint tid, uint simd, uint lane, uint start, uint stop, uint step,
                                          threadgroup bfloat *a, threadgroup bfloat *b,
                                          threadgroup const half *tg8, threadgroup const half2 *tg4) {
    constexpr uint B = 32u, L = B + 8u, TG = 128u;
    constexpr uint ALoads = (M * (B / 4u) + TG - 1u) / TG;
    constexpr uint RM = M >= 16u ? M / 16u : 1u;
    constexpr uint RN = (M == 8u) ? 2u : 4u;
    const uint n = Down ? p.n : 2u * p.n;
    const float outer = Nv ? mk_load(amax, p.amax, 0, expert) / mk_load(denom, p.denominator, 0, 0) : 1.0f;
    const uint sr = simd / 2u, sc = simd % 2u;
    const uint row_base = M >= 16u ? sr * (M / 2u) : 0u;
    const uint col_simd_base = (M == 8u) ? (simd * 16u) : (sc * 32u);
    const uint fr = ((lane / 4u) & 4u) + (lane / 2u) % 4u;
    const uint fc = (((lane / 4u) & 2u) + lane % 2u) * 2u;
    device const bfloat *x16_base = reinterpret_cast<device const bfloat *>(partial);
    for (uint t0 = start; t0 < stop; t0 += step) {
        device const float *xs[ALoads];
        device const bfloat *xs16[ALoads];
        bool active[ALoads];
        #pragma unroll
        for (uint i = 0u; i < ALoads; ++i) {
            const uint id = tid + i * TG, xr = t0 + id / (B / 4u), xk = (id % (B / 4u)) * 4u;
            active[i] = (id < M * (B / 4u)) && (xr < count);
            const uint route = active[i] ? entries[xr] : 0u;
            if constexpr (X16) {
                if constexpr (Down) {
                    xs16[i] = reinterpret_cast<device const bfloat *>(arena + p.x + (p.row0 + route / p.top_k) * p.xs) + (route % p.top_k) * p.k + xk;
                } else {
                    xs16[i] = x16_base + ulong(route / p.top_k) * p.k + xk;
                }
                xs[i] = nullptr;
            } else {
                xs[i] = arena + p.x + (p.row0 + route / p.top_k) * p.xs + (Down ? (route % p.top_k) * p.k : 0u) + xk;
                xs16[i] = nullptr;
            }
        }
        const uint wc = tid / 2u, wk = (tid % 2u) * 16u;
        const ulong w_off = (ulong(expert) * n + c0 + wc) * p.k + wk;
        simdgroup_float8x8 acc[RM][RN];
        #pragma unroll
        for (uint i = 0u; i < RM; ++i) {
            #pragma unroll
            for (uint j = 0u; j < RN; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
        for (uint k = 0u; k < p.k; k += B) {
            #pragma unroll
            for (uint i = 0u; i < ALoads; ++i) {
                const uint id = tid + i * TG;
                if (id < M * (B / 4u)) {
                    const uint ar = id / (B / 4u), ak = (id % (B / 4u)) * 4u;
                    if constexpr (X16) {
                        const device bfloat4 *src16 = reinterpret_cast<const device bfloat4 *>(xs16[i] + k);
                        reinterpret_cast<threadgroup bfloat4 *>(a + ar * L + ak)[0] = active[i] ? src16[0] : bfloat4(float4(0));
                    } else {
                        const device float4 *src = reinterpret_cast<const device float4 *>(xs[i] + k);
                        reinterpret_cast<threadgroup bfloat4 *>(a + ar * L + ak)[0] = bfloat4(active[i] ? src[0] : float4(0));
                    }
                }
            }
            const ulong windex = w_off + k;
            threadgroup bfloat4 *dst = reinterpret_cast<threadgroup bfloat4 *>(b + wc * L + wk);
            if constexpr (Nv) {
                const uint2 raw = uint2(*reinterpret_cast<const device packed_uint2 *>(w + p.w + windex / 2u));
                const uchar sc_code = s[p.s + windex / 16u];
                if constexpr (Lut) {
                    const float sc_val = float(tg8[sc_code]) * outer;
                    const uchar4 rx0 = as_type<uchar4>(raw.x), rx1 = as_type<uchar4>(raw.y);
                    dst[0] = bfloat4(float4(half4(tg4[rx0[0]], tg4[rx0[1]])) * sc_val);
                    dst[1] = bfloat4(float4(half4(tg4[rx0[2]], tg4[rx0[3]])) * sc_val);
                    dst[2] = bfloat4(float4(half4(tg4[rx1[0]], tg4[rx1[1]])) * sc_val);
                    dst[3] = bfloat4(float4(half4(tg4[rx1[2]], tg4[rx1[3]])) * sc_val);
                } else {
                    const float sc_val = mk_expert_e4m3(sc_code) * outer;
                    dst[0] = bfloat4(inkling_nvfp4_decode4(raw.x) * sc_val);
                    dst[1] = bfloat4(inkling_nvfp4_decode4(raw.x >> 16) * sc_val);
                    dst[2] = bfloat4(inkling_nvfp4_decode4(raw.y) * sc_val);
                    dst[3] = bfloat4(inkling_nvfp4_decode4(raw.y >> 16) * sc_val);
                }
            } else {
                const uint e = s[p.s + windex / 32u];
                const float factor = as_type<float>(e == 0u ? 0x00400000u : e == 255u ? 0x7fc00000u : e << 23);
                const uint4 q = *reinterpret_cast<const device uint4 *>(w + p.w + windex);
                #pragma unroll
                for (uint u = 0u; u < 4u; ++u) {
                    if constexpr (Lut) {
                        const uchar4 qv = as_type<uchar4>(q[u]);
                        dst[u] = bfloat4(float4(half4(tg8[qv[0]], tg8[qv[1]], tg8[qv[2]], tg8[qv[3]])) * factor);
                    } else {
                        const uint qu = q[u];
                        dst[u] = bfloat4(float4(mk_expert_e4m3(uchar(qu)), mk_expert_e4m3(uchar(qu >> 8)),
                                                mk_expert_e4m3(uchar(qu >> 16)), mk_expert_e4m3(uchar(qu >> 24))) * factor);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            #pragma unroll
            for (uint kf = 0u; kf < B; kf += 8u) {
                simdgroup_bfloat8x8 bf[RN];
                #pragma unroll
                for (uint j = 0u; j < RN; ++j) simdgroup_load(bf[j], b + (col_simd_base + j * 8u) * L + kf, L, ulong2(0), true);
                #pragma unroll
                for (uint i = 0u; i < RM; ++i) {
                    simdgroup_bfloat8x8 af;
                    simdgroup_load(af, a + (row_base + i * 8u) * L + kf, L);
                    #pragma unroll
                    for (uint j = 0u; j < RN; ++j) simdgroup_multiply_accumulate(acc[i][j], af, bf[j], acc[i][j]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const uint col0 = c0 + col_simd_base + fc;
        #pragma unroll
        for (uint i = 0u; i < RM; ++i) {
            const uint t = t0 + row_base + i * 8u + fr;
            if (t >= count) continue;
            const uint r = entries[t];
            #pragma unroll
            for (uint j = 0u; j < RN; ++j) {
                mk_expert_block_store<Down, Bf16Io>(arena, partial, p, r, col0 + j * 8u, acc[i][j].thread_elements()[0], acc[i][j].thread_elements()[1]);
            }
        }
    }
}

#define MK_EXPERT_BLOCK_M64N64_STRIPE(NAME, DOWN, NV, LUT, BF16IO) \
    kernel void NAME(device float *arena [[buffer(0)]], device const uchar *w [[buffer(1)]], device const uchar *s [[buffer(2)]], \
                     device const uchar *amax [[buffer(3)]], device const uchar *denom [[buffer(4)]], device const uint *lists [[buffer(5)]], \
                     device float *partial [[buffer(6)]], constant MKExpert &p [[buffer(8)]], uint g [[threadgroup_position_in_grid]], \
                     uint tid [[thread_index_in_threadgroup]], uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
        alignas(8) threadgroup bfloat a[64u * 40u], b[64u * 40u]; \
        threadgroup half tg8[LUT ? 256u : 1u]; \
        threadgroup half2 tg4[LUT ? 256u : 1u]; \
        const uint n = DOWN ? p.n : 2u * p.n; \
        const uint cols = n / INKLING_EXPERT_BLOCK_N64, parts = p.rows >= 2048u ? 8u : p.rows >= 256u ? 4u : p.rows >= 64u ? 2u : 1u; \
        const uint expert = g / (cols * parts), part = (g / cols) % parts; \
        device const uint *list = lists + expert * (p.rows * p.top_k + 1u); \
        const uint count = list[0]; \
        if (count == 0u) return; \
        if constexpr (LUT) { \
            for (uint i = tid; i < 256u; i += 128u) { \
                tg8[i] = mk_e4m3_h_lut[i]; \
                if constexpr (NV) tg4[i] = mk_fp4_pair_h2_lut[i]; \
            } \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
        } \
        device const uint *entries = list + 1u; \
        const uint c0 = (g % cols) * INKLING_EXPERT_BLOCK_N64; \
        const uint full = (count / 64u) * 64u, tail = count - full; \
        constexpr bool X16 = (!DOWN) || BF16IO; \
        if (part * 64u < full) { \
            mk_expert_block_m64n64_core<DOWN, NV, 64u, X16, LUT, BF16IO>(arena, w, s, amax, denom, entries, count, partial, p, expert, c0, tid, simd, lane, part * 64u, full, 64u * parts, a, b, tg8, tg4); \
        } \
        if ((full / 64u) % parts != part) return; \
        if (tail > 32u) { \
            mk_expert_block_m64n64_core<DOWN, NV, 64u, X16, LUT, BF16IO>(arena, w, s, amax, denom, entries, count, partial, p, expert, c0, tid, simd, lane, full, count, 64u, a, b, tg8, tg4); \
        } else if (tail > 16u) { \
            mk_expert_block_m64n64_core<DOWN, NV, 32u, X16, LUT, BF16IO>(arena, w, s, amax, denom, entries, count, partial, p, expert, c0, tid, simd, lane, full, count, 32u, a, b, tg8, tg4); \
        } else if (tail > 8u) { \
            mk_expert_block_m64n64_core<DOWN, NV, 16u, X16, LUT, BF16IO>(arena, w, s, amax, denom, entries, count, partial, p, expert, c0, tid, simd, lane, full, count, 16u, a, b, tg8, tg4); \
        } else if (tail != 0u) { \
            mk_expert_block_m64n64_core<DOWN, NV, 8u, X16, LUT, BF16IO>(arena, w, s, amax, denom, entries, count, partial, p, expert, c0, tid, simd, lane, full, count, 8u, a, b, tg8, tg4); \
        } \
    }

#define MK_EXPERT_BLOCK_M64N64_CT(NAME, DOWN, NV, LUT, BF16IO) \
    kernel void NAME(device float *arena [[buffer(0)]], device const uchar *w [[buffer(1)]], device const uchar *s [[buffer(2)]], \
                     device const uchar *amax [[buffer(3)]], device const uchar *denom [[buffer(4)]], device const uint *lists [[buffer(5)]], \
                     device float *partial [[buffer(6)]], constant MKExpert &p [[buffer(8)]], uint g [[threadgroup_position_in_grid]], \
                     uint tid [[thread_index_in_threadgroup]], uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
        const uint n = DOWN ? p.n : 2u * p.n; \
        const uint cols = n / INKLING_EXPERT_BLOCK_N64; \
        const uint tile_idx = g / cols; \
        const uint tile_count = lists[p.experts + 1u]; \
        if (tile_idx >= tile_count) return; \
        alignas(8) threadgroup bfloat a[64u * 40u], b[64u * 40u]; \
        threadgroup half tg8[LUT ? 256u : 1u]; \
        threadgroup half2 tg4[LUT ? 256u : 1u]; \
        if constexpr (LUT) { \
            for (uint i = tid; i < 256u; i += 128u) { \
                tg8[i] = mk_e4m3_h_lut[i]; \
                if constexpr (NV) tg4[i] = mk_fp4_pair_h2_lut[i]; \
            } \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
        } \
        const uint total = p.rows * p.top_k; \
        const uint max_tiles = inkling_moe_max_tiles(p.experts, total); \
        device const uint *expert_off = lists; \
        device const uint *sorted_routes = lists + p.experts + 2u; \
        device const uint *tile_e = sorted_routes + total; \
        device const uint *tile_r = tile_e + max_tiles; \
        const uint expert = tile_e[tile_idx]; \
        const uint packed_r = tile_r[tile_idx]; \
        const uint off = expert_off[expert]; \
        const uint count = expert_off[expert + 1u] - off; \
        const uint bm = uint(inkling_moe_unpack_bm(packed_r)); \
        const uint t0 = inkling_moe_unpack_tid_y(packed_r) * bm; \
        const uint c0 = (g % cols) * INKLING_EXPERT_BLOCK_N64; \
        device const uint *entries = sorted_routes + off; \
        constexpr bool X16 = (!DOWN) || BF16IO; \
        if (bm == 64u) { \
            mk_expert_block_m64n64_core<DOWN, NV, 64u, X16, LUT, BF16IO>(arena, w, s, amax, denom, entries, count, partial, p, expert, c0, tid, simd, lane, t0, min(count, t0 + 64u), 64u, a, b, tg8, tg4); \
        } else if (bm == 32u) { \
            mk_expert_block_m64n64_core<DOWN, NV, 32u, X16, LUT, BF16IO>(arena, w, s, amax, denom, entries, count, partial, p, expert, c0, tid, simd, lane, t0, min(count, t0 + 32u), 32u, a, b, tg8, tg4); \
        } else { \
            mk_expert_block_m64n64_core<DOWN, NV, 16u, X16, LUT, BF16IO>(arena, w, s, amax, denom, entries, count, partial, p, expert, c0, tid, simd, lane, t0, min(count, t0 + 16u), 16u, a, b, tg8, tg4); \
        } \
    }

MK_EXPERT_BLOCK_M64N64_STRIPE(mk_expert_block_up_nv64_m64, false, true, false, false)
MK_EXPERT_BLOCK_M64N64_STRIPE(mk_expert_block_down_nv64_m64, true, true, false, false)
MK_EXPERT_BLOCK_M64N64_STRIPE(mk_expert_block_up_mx64_m64, false, false, false, false)
MK_EXPERT_BLOCK_M64N64_STRIPE(mk_expert_block_down_mx64_m64, true, false, false, false)
MK_EXPERT_BLOCK_M64N64_STRIPE(mk_expert_block_up_nv64_m64_lut, false, true, true, false)
MK_EXPERT_BLOCK_M64N64_STRIPE(mk_expert_block_down_nv64_m64_lut, true, true, true, false)
MK_EXPERT_BLOCK_M64N64_STRIPE(mk_expert_block_up_mx64_m64_lut, false, false, true, false)
MK_EXPERT_BLOCK_M64N64_STRIPE(mk_expert_block_down_mx64_m64_lut, true, false, true, false)
MK_EXPERT_BLOCK_M64N64_STRIPE(mk_expert_block_up_nv64_m64_bf16, false, true, false, true)
MK_EXPERT_BLOCK_M64N64_STRIPE(mk_expert_block_down_nv64_m64_bf16, true, true, false, true)
MK_EXPERT_BLOCK_M64N64_STRIPE(mk_expert_block_up_mx64_m64_bf16, false, false, false, true)
MK_EXPERT_BLOCK_M64N64_STRIPE(mk_expert_block_down_mx64_m64_bf16, true, false, false, true)
MK_EXPERT_BLOCK_M64N64_STRIPE(mk_expert_block_up_nv64_m64_lut_bf16, false, true, true, true)
MK_EXPERT_BLOCK_M64N64_STRIPE(mk_expert_block_down_nv64_m64_lut_bf16, true, true, true, true)
MK_EXPERT_BLOCK_M64N64_STRIPE(mk_expert_block_up_mx64_m64_lut_bf16, false, false, true, true)
MK_EXPERT_BLOCK_M64N64_STRIPE(mk_expert_block_down_mx64_m64_lut_bf16, true, false, true, true)

MK_EXPERT_BLOCK_M64N64_CT(mk_expert_block_up_nv64_m64_ct, false, true, false, false)
MK_EXPERT_BLOCK_M64N64_CT(mk_expert_block_down_nv64_m64_ct, true, true, false, false)
MK_EXPERT_BLOCK_M64N64_CT(mk_expert_block_up_mx64_m64_ct, false, false, false, false)
MK_EXPERT_BLOCK_M64N64_CT(mk_expert_block_down_mx64_m64_ct, true, false, false, false)
MK_EXPERT_BLOCK_M64N64_CT(mk_expert_block_up_nv64_m64_ct_lut, false, true, true, false)
MK_EXPERT_BLOCK_M64N64_CT(mk_expert_block_down_nv64_m64_ct_lut, true, true, true, false)
MK_EXPERT_BLOCK_M64N64_CT(mk_expert_block_up_mx64_m64_ct_lut, false, false, true, false)
MK_EXPERT_BLOCK_M64N64_CT(mk_expert_block_down_mx64_m64_ct_lut, true, false, true, false)
MK_EXPERT_BLOCK_M64N64_CT(mk_expert_block_up_nv64_m64_ct_bf16, false, true, false, true)
MK_EXPERT_BLOCK_M64N64_CT(mk_expert_block_down_nv64_m64_ct_bf16, true, true, false, true)
MK_EXPERT_BLOCK_M64N64_CT(mk_expert_block_up_mx64_m64_ct_bf16, false, false, false, true)
MK_EXPERT_BLOCK_M64N64_CT(mk_expert_block_down_mx64_m64_ct_bf16, true, false, false, true)
MK_EXPERT_BLOCK_M64N64_CT(mk_expert_block_up_nv64_m64_ct_lut_bf16, false, true, true, true)
MK_EXPERT_BLOCK_M64N64_CT(mk_expert_block_down_nv64_m64_ct_lut_bf16, true, true, true, true)
MK_EXPERT_BLOCK_M64N64_CT(mk_expert_block_up_mx64_m64_ct_lut_bf16, false, false, true, true)
MK_EXPERT_BLOCK_M64N64_CT(mk_expert_block_down_mx64_m64_ct_lut_bf16, true, false, true, true)
#undef MK_EXPERT_BLOCK_M64N64_STRIPE
#undef MK_EXPERT_BLOCK_M64N64_CT

#define MK_EXPERT_BLOCK_M8N64_CT(NAME, DOWN, NV, LUT, BF16IO) \
    kernel void NAME(device float *arena [[buffer(0)]], device const uchar *w [[buffer(1)]], device const uchar *s [[buffer(2)]], \
                     device const uchar *amax [[buffer(3)]], device const uchar *denom [[buffer(4)]], device const uint *lists [[buffer(5)]], \
                     device float *partial [[buffer(6)]], constant MKExpert &p [[buffer(8)]], uint g [[threadgroup_position_in_grid]], \
                     uint tid [[thread_index_in_threadgroup]], uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
        const uint n = DOWN ? p.n : 2u * p.n; \
        const uint cols = n / INKLING_EXPERT_BLOCK_N64; \
        const uint tile_idx = g / cols; \
        const uint tile_count = lists[p.experts + 1u]; \
        if (tile_idx >= tile_count) return; \
        alignas(8) threadgroup bfloat a[8u * 40u], b[64u * 40u]; \
        threadgroup half tg8[LUT ? 256u : 1u]; \
        threadgroup half2 tg4[LUT ? 256u : 1u]; \
        if constexpr (LUT) { \
            for (uint i = tid; i < 256u; i += 128u) { \
                tg8[i] = mk_e4m3_h_lut[i]; \
                if constexpr (NV) tg4[i] = mk_fp4_pair_h2_lut[i]; \
            } \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
        } \
        const uint total = p.rows * p.top_k; \
        const uint max_tiles = inkling_moe_max_tiles(p.experts, total); \
        device const uint *expert_off = lists; \
        device const uint *sorted_routes = lists + p.experts + 2u; \
        device const uint *tile_e = sorted_routes + total; \
        const uint expert = tile_e[tile_idx]; \
        const uint off = expert_off[expert]; \
        const uint count = expert_off[expert + 1u] - off; \
        const uint c0 = (g % cols) * INKLING_EXPERT_BLOCK_N64; \
        device const uint *entries = sorted_routes + off; \
        constexpr bool X16 = (!DOWN) || BF16IO; \
        mk_expert_block_m64n64_core<DOWN, NV, 8u, X16, LUT, BF16IO>(arena, w, s, amax, denom, entries, count, partial, p, expert, c0, tid, simd, lane, 0u, count, 8u, a, b, tg8, tg4); \
    }

MK_EXPERT_BLOCK_M8N64_CT(mk_expert_block_up_nv64_m8_ct, false, true, false, false)
MK_EXPERT_BLOCK_M8N64_CT(mk_expert_block_down_nv64_m8_ct, true, true, false, false)
MK_EXPERT_BLOCK_M8N64_CT(mk_expert_block_up_mx64_m8_ct, false, false, false, false)
MK_EXPERT_BLOCK_M8N64_CT(mk_expert_block_down_mx64_m8_ct, true, false, false, false)
MK_EXPERT_BLOCK_M8N64_CT(mk_expert_block_up_nv64_m8_ct_lut, false, true, true, false)
MK_EXPERT_BLOCK_M8N64_CT(mk_expert_block_down_nv64_m8_ct_lut, true, true, true, false)
MK_EXPERT_BLOCK_M8N64_CT(mk_expert_block_up_mx64_m8_ct_lut, false, false, true, false)
MK_EXPERT_BLOCK_M8N64_CT(mk_expert_block_down_mx64_m8_ct_lut, true, false, true, false)
MK_EXPERT_BLOCK_M8N64_CT(mk_expert_block_up_nv64_m8_ct_bf16, false, true, false, true)
MK_EXPERT_BLOCK_M8N64_CT(mk_expert_block_down_nv64_m8_ct_bf16, true, true, false, true)
MK_EXPERT_BLOCK_M8N64_CT(mk_expert_block_up_mx64_m8_ct_bf16, false, false, false, true)
MK_EXPERT_BLOCK_M8N64_CT(mk_expert_block_down_mx64_m8_ct_bf16, true, false, false, true)
MK_EXPERT_BLOCK_M8N64_CT(mk_expert_block_up_nv64_m8_ct_lut_bf16, false, true, true, true)
MK_EXPERT_BLOCK_M8N64_CT(mk_expert_block_down_nv64_m8_ct_lut_bf16, true, true, true, true)
MK_EXPERT_BLOCK_M8N64_CT(mk_expert_block_up_mx64_m8_ct_lut_bf16, false, false, true, true)
MK_EXPERT_BLOCK_M8N64_CT(mk_expert_block_down_mx64_m8_ct_lut_bf16, true, false, true, true)
#undef MK_EXPERT_BLOCK_M8N64_CT

// Wave 4: Fused controller message merge + RMSNorm(+BF16 cast) for batched forwards.
kernel void mk_inkling_merge_norm_vec(device const uchar *w [[buffer(0)]], device float *arena [[buffer(1)]],
                                      constant MK &p [[buffer(8)]], uint tgid [[threadgroup_position_in_grid]],
                                      uint lid [[thread_position_in_threadgroup]], uint tg [[threads_per_threadgroup]],
                                      uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[33];
    if (tgid >= p.rows) return;
    const uint row = p.row0 + tgid;
    const bool active = lid * 4u < p.n;
    const device float4 *hp = (const device float4 *)(arena + p.x + row * p.xs);
    const device float4 *mp = (const device float4 *)(arena + p.y + row * p.ys);
    const device float4 *gp0 = (const device float4 *)(arena + p.z + row * p.zs);
    const device float4 *gp1 = (const device float4 *)(arena + p.z + row * p.zs + p.n);
    device float4 *op = (device float4 *)(arena + p.every + row * p.pad_k);
    float4 mv = float4(0.0f);
    if (active) {
        const float4 hid = hp[lid];
        const float4 msg = mp[lid];
        const float4 fg = tanh(gp0[lid]);
        const float4 gn = tanh(gp1[lid]);
        mv = (float4(1.0f) - fg) * hid + gn * msg;
        if (p.flags & MK_INKLING_ROUND_RES) {
            for (uint j = 0u; j < 4u; ++j) mv[j] = mk_bf16(mv[j]);
        }
    }
    const uint nsimds = tg / 32u;
    const float acc = simd_sum(active ? dot(mv, mv) : 0.0f);
    if (lane == 0u) partial[simd] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0u) {
        const float tot = simd_sum(lane < nsimds ? partial[lane] : 0.0f);
        if (lane == 0u) partial[32] = metal::precise::rsqrt(tot / float(p.n) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!active) return;
    float4 v = mv * partial[32];
    if (p.flags & MK_ROUND_BF16) {
        for (uint j = 0u; j < 4u; ++j) v[j] = mk_bf16(v[j]);
    }
    float4 res = (p.flags & MK_HAS_WEIGHT) ? mk_fast_load4(w, p.w, p.dtype, lid) * v : v;
    if (p.flags & MK_ROUND_OUT) {
        for (uint j = 0u; j < 4u; ++j) res[j] = mk_bf16(res[j]);
    }
    op[lid] = res;
}




