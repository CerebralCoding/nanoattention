// Inkling 2 Mini optimized decode and prefill kernels; embedded after inkling_tiles.metal.

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

// ---- Step 3D kernels: fast NVFP4 & MXFP8 decode experts ----
inline float mk_nvfp4_dot16(uint2 raw, float4 x0, float4 x1, float4 x2, float4 x3) {
    const float4 d0 = inkling_nvfp4_decode4(raw.x);
    const float4 d1 = inkling_nvfp4_decode4(raw.x >> 16);
    const float4 d2 = inkling_nvfp4_decode4(raw.y);
    const float4 d3 = inkling_nvfp4_decode4(raw.y >> 16);
    float acc = dot(d0, x0) + dot(d1, x1) + dot(d2, x2) + dot(d3, x3);
    return acc;
}

inline float mk_e4m3x4_dot(uint packed, float4 xv) {
    const float4 wv(mk_expert_e4m3(uchar(packed)),
                    mk_expert_e4m3(uchar(packed >> 8)),
                    mk_expert_e4m3(uchar(packed >> 16)),
                    mk_expert_e4m3(uchar(packed >> 24)));
    return dot(wv, xv);
}

// 16-lane sub-SIMD NVFP4 gate+up decode: 2 pairs (4 rows) per sub-SIMD, 8 rows per SIMD.
kernel void mk_expert_gate_up_nv16(device float *arena [[buffer(0)]], device const uchar *w [[buffer(1)]],
                                   device const uchar *s [[buffer(2)]], device const uchar *amax [[buffer(3)]],
                                   device const uchar *denom [[buffer(4)]], constant MKExpert &p [[buffer(8)]],
                                   uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
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
                const float sc = mk_expert_e4m3(sp[b + c * sstep]);
                const float d = mk_nvfp4_dot16(wp[b + c * wstep], x0, x1, x2, x3);
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
        for (uint c = 0u; c < 2u; ++c) {
            float gate = acc[c * 2u], up = acc[c * 2u + 1u];
            if (p.round_bf16) { gate = mk_bf16(gate); up = mk_bf16(up); }
            const float value = ((tanh(gate / p.gate_cap) * p.gate_cap) * mk_sigmoid(gate)) * (tanh(up / p.up_cap) * p.up_cap);
            out[c] = p.round_bf16 ? mk_bf16(value) : value;
        }
    }
}

// 8-lane sub-SIMD NVFP4 expert down decode: 2 SIMDs (64 threads) per TG cover all 8 expert slots x 4 columns.
kernel void mk_expert_down_nv8(device float *arena [[buffer(0)]], device const uchar *w [[buffer(1)]],
                               device const uchar *s [[buffer(2)]], device const uchar *amax [[buffer(3)]],
                               device const uchar *denom [[buffer(4)]], constant MKExpert &p [[buffer(8)]],
                               uint i [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint columns = p.n / 4u;
    if (i >= columns) return;
    const uint j = i * 4u;
    const uint row = p.row0;
    const uint slot = tid >> 3;
    const uint l8 = tid & 7u;
    threadgroup float4 dots[8];
    device const float *routes = arena + p.routes + row * p.rs;
    const uint expert = uint(routes[slot]);
    float4 acc = 0.0f;
    if (expert < p.experts) {
        const ulong base = (ulong(expert) * p.n + j) * p.k;
        device const float4 *xp = (device const float4 *)(arena + p.x + row * p.xs + slot * p.k);
        device const uint2 *wp = (device const uint2 *)(w + p.w + base / 2u);
        device const uchar *sp = s + p.s + base / 16u;
        const uint wstep = p.k / 16u;
        const uint sstep = p.k / 16u;
        for (uint k4 = l8 * 4u; k4 < p.k / 4u; k4 += INKLING_NVFP4_SUBSIMD8_LANES * 4u) {
            const float4 x0 = xp[k4];
            const float4 x1 = xp[k4 + 1u];
            const float4 x2 = xp[k4 + 2u];
            const float4 x3 = xp[k4 + 3u];
            const uint b = k4 / 4u;
            #pragma unroll
            for (uint c = 0u; c < 4u; ++c) {
                const float sc = mk_expert_e4m3(sp[b + c * sstep]);
                const float d = mk_nvfp4_dot16(wp[b + c * wstep], x0, x1, x2, x3);
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

// Vectorized 8-byte MXFP8 gate+up (k % 256 == 0, w % 8 == 0).
kernel void mk_expert_gate_up_mx_vec(device float *arena [[buffer(0)]], device const uchar *w [[buffer(1)]],
                                     device const uchar *s [[buffer(2)]], device const uchar *amax [[buffer(3)]],
                                     device const uchar *denom [[buffer(4)]], constant MKExpert &p [[buffer(8)]],
                                     uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
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
                const float part = mk_e4m3x4_dot(packed.x, x0) + mk_e4m3x4_dot(packed.y, x1);
                acc[c] = fma(part, scale, acc[c]);
            }
        }
        acc = float4(simd_sum(acc.x), simd_sum(acc.y), simd_sum(acc.z), simd_sum(acc.w));
    }
    if (lane == 0u) {
        device float *out = arena + p.y + row * p.ys + slot * p.n + j;
        for (uint c = 0u; c < 2u; ++c) {
            float gate = acc[c * 2u], up = acc[c * 2u + 1u];
            if (p.round_bf16) { gate = mk_bf16(gate); up = mk_bf16(up); }
            const float value = ((tanh(gate / p.gate_cap) * p.gate_cap) * mk_sigmoid(gate)) * (tanh(up / p.up_cap) * p.up_cap);
            out[c] = p.round_bf16 ? mk_bf16(value) : value;
        }
    }
}

// Vectorized 4-byte MXFP8 down (k % 128 == 0, w % 4 == 0).
kernel void mk_expert_down_mx_vec(device float *arena [[buffer(0)]], device const uchar *w [[buffer(1)]],
                                  device const uchar *s [[buffer(2)]], device const uchar *amax [[buffer(3)]],
                                  device const uchar *denom [[buffer(4)]], constant MKExpert &p [[buffer(8)]],
                                  uint i [[threadgroup_position_in_grid]], uint slot [[simdgroup_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]]) {
    const uint columns = p.n / 4u;
    if (i >= columns) return;
    const uint j = i * 4u;
    const uint row = p.row0;
    threadgroup float4 dots[8];
    if (slot < p.top_k) {
        device const float *routes = arena + p.routes + row * p.rs;
        const uint expert = uint(routes[slot]);
        float4 acc = 0.0f;
        if (expert < p.experts) {
            const ulong base = (ulong(expert) * p.n + j) * p.k;
            device const float4 *xp = (device const float4 *)(arena + p.x + row * p.xs + slot * p.k);
            device const uint *wp = (device const uint *)(w + p.w + base);
            device const uchar *sp = s + p.s + base / 32u;
            const uint wstep = p.k / 4u;
            const uint sstep = p.k / 32u;
            for (uint k4 = lane; k4 < p.k / 4u; k4 += 32u) {
                const float4 xv = xp[k4];
                const uint sb = k4 / 8u;
                #pragma unroll
                for (uint c = 0u; c < 4u; ++c) {
                    const uint e = sp[sb + c * sstep];
                    const float scale = as_type<float>(e == 0u ? 0x00400000u : e == 255u ? 0x7fc00000u : e << 23);
                    const float part = mk_e4m3x4_dot(wp[k4 + c * wstep], xv);
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

// ---- Step 3E kernels: fast single-pass log2 striped decode attention + gated merge ----
// Single-pass online softmax per SIMD in registers (~8.3 KB TG reduction instead of 30.9 KB);
// maxima stored in parts[*][128] are in log2 units and pair with mk_relative_long_merge_fast / _gated.
kernel void mk_relative_long_keys_fast(device const float *arena [[buffer(0)]], device const ushort *key [[buffer(1)]],
                                       device const ushort *val [[buffer(2)]], device float *parts [[buffer(3)]],
                                       constant MK &p [[buffer(8)]], uint lane [[thread_index_in_simdgroup]],
                                       uint simd [[simdgroup_index_in_threadgroup]], uint simds [[simdgroups_per_threadgroup]],
                                       uint g [[threadgroup_position_in_grid]]) {
    const uint kvh = g / p.taps, part = g % p.taps;
    const uint row = p.row0, pos = p.pos + row;
    const uint visible = p.window ? min(p.window, pos + 1u) : pos + 1u;
    const uint first = pos + 1u - visible;
    const uint s_off = inkling_stripe_start(visible, p.taps, part);
    const uint e_off = inkling_stripe_end(visible, p.taps, part);
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

    float m[3] = { -INFINITY, -INFINITY, -INFINITY };
    float d[3] = { 0.0f, 0.0f, 0.0f };
    float4 acc[3] = { float4(0.0f), float4(0.0f), float4(0.0f) };
    device const float *bias_base = arena + p.z + row * p.zs + h0 * p.m;
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
            if (distance < p.m) {
                sc[0] = s0 + bias_base[distance] * M_LOG2E_F;
                sc[1] = s1 + bias_base[p.m + distance] * M_LOG2E_F;
                sc[2] = s2 + bias_base[2u * p.m + distance] * M_LOG2E_F;
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

    // 8-SIMD max reduces TG memory from 24.6 KB to 12.3 KB for higher core occupancy.
    #define MK_LONG_FAST_SIMDS 8u
    threadgroup float4 red_acc[MK_LONG_FAST_SIMDS * 3u * 32u];
    threadgroup float red_m[MK_LONG_FAST_SIMDS * 3u];
    threadgroup float red_d[MK_LONG_FAST_SIMDS * 3u];
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
        device float *dst = parts + (ulong(h0 + h) * p.taps + part) * 130u;
        *(device packed_float4 *)(dst + lane * 4u) = packed_float4(gout);
        if (lane == 0u) {
            dst[128] = gm;
            dst[129] = gd;
        }
    }
    #undef MK_LONG_FAST_SIMDS
}

template <bool Gated>
inline void mk_relative_long_merge_impl(device float *arena, device const float *parts, constant MK &p,
                                        uint tid, uint lane, uint simd, uint h,
                                        threadgroup float *group, threadgroup float *factor) {
    const uint row = p.row0;
    device const float *src = parts + ulong(h) * p.taps * 130u;
    const float sink = arena[p.s + row * p.every + h] * M_LOG2E_F;
    float m = -INFINITY;
    for (uint k = tid; k < p.taps; k += 128u) m = max(m, src[ulong(k) * 130u + 128u]);
    m = simd_max(m);
    if (lane == 0u) group[simd] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = max(max(max(group[0], group[1]), max(group[2], group[3])), sink);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float d = 0.0f;
    for (uint k = tid; k < p.taps; k += 128u) {
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
    for (uint k = 0u; k < p.taps; ++k) acc += factor[k] * src[ulong(k) * 130u + tid];
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
template <bool Down>
inline void mk_expert_block_store(device float *arena, device float *partial, constant MKExpert &p,
                                  uint r, uint c, float lo, float hi) {
    if (p.round_bf16) {
        lo = mk_bf16(lo);
        hi = mk_bf16(hi);
    }
    if constexpr (Down) {
        partial[ulong(r) * p.n + c] = lo;
        partial[ulong(r) * p.n + c + 1u] = hi;
    } else {
        const float out = (tanh(lo / p.gate_cap) * p.gate_cap * mk_sigmoid(lo)) * (tanh(hi / p.up_cap) * p.up_cap);
        arena[p.y + (p.row0 + r / p.top_k) * p.ys + (r % p.top_k) * p.n + c / 2u] = p.round_bf16 ? mk_bf16(out) : out;
    }
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

template <bool Down, bool Nv, uint M, bool X16>
inline void mk_expert_block_m64n64(device float *arena, device const uchar *w, device const uchar *s,
                                    device const uchar *amax, device const uchar *denom, device const uint *lists,
                                    device float *partial, constant MKExpert &p, uint g, uint tid, uint simd, uint lane,
                                    uint start, uint stop, uint step, threadgroup bfloat *a, threadgroup bfloat *b) {
    constexpr uint N = INKLING_EXPERT_BLOCK_N64, B = 32u, L = B + 8u, TG = 128u;
    constexpr uint ALoads = (M * (B / 4u) + TG - 1u) / TG;
    constexpr uint RM = M >= 16u ? M / 16u : 1u;
    const uint n = Down ? p.n : 2u * p.n;
    const uint expert = g / (n / N), c0 = (g % (n / N)) * N;
    device const uint *list = lists + expert * (p.rows * p.top_k + 1u);
    const uint count = list[0];
    if (count == 0u) return;
    const float outer = Nv ? mk_load(amax, p.amax, 0, expert) / mk_load(denom, p.denominator, 0, 0) : 1.0f;
    const uint sr = simd / 2u, sc = simd % 2u;
    const uint row_base = M >= 16u ? sr * (M / 2u) : 0u;
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
            const uint route = active[i] ? list[xr + 1u] : 0u;
            if constexpr (X16) {
                xs16[i] = x16_base + ulong(route / p.top_k) * p.k + xk;
                xs[i] = nullptr;
            } else {
                xs[i] = arena + p.x + (p.row0 + route / p.top_k) * p.xs + (Down ? (route % p.top_k) * p.k : 0u) + xk;
                xs16[i] = nullptr;
            }
        }
        const uint wc = tid / 2u, wk = (tid % 2u) * 16u;
        const ulong w_off = (ulong(expert) * n + c0 + wc) * p.k + wk;
        simdgroup_float8x8 acc[RM][4];
        #pragma unroll
        for (uint i = 0u; i < RM; ++i) {
            #pragma unroll
            for (uint j = 0u; j < 4u; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
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
                const float sc_val = mk_expert_e4m3(s[p.s + windex / 16u]) * outer;
                dst[0] = bfloat4(inkling_nvfp4_decode4(raw.x) * sc_val);
                dst[1] = bfloat4(inkling_nvfp4_decode4(raw.x >> 16) * sc_val);
                dst[2] = bfloat4(inkling_nvfp4_decode4(raw.y) * sc_val);
                dst[3] = bfloat4(inkling_nvfp4_decode4(raw.y >> 16) * sc_val);
            } else {
                const uint e = s[p.s + windex / 32u];
                const float factor = as_type<float>(e == 0u ? 0x00400000u : e == 255u ? 0x7fc00000u : e << 23);
                const uint4 q = *reinterpret_cast<const device uint4 *>(w + p.w + windex);
                #pragma unroll
                for (uint u = 0u; u < 4u; ++u) {
                    const uint qu = q[u];
                    dst[u] = bfloat4(float4(mk_expert_e4m3(uchar(qu)), mk_expert_e4m3(uchar(qu >> 8)),
                                            mk_expert_e4m3(uchar(qu >> 16)), mk_expert_e4m3(uchar(qu >> 24))) * factor);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (M >= 16u || sr == 0u) {
                #pragma unroll
                for (uint kf = 0u; kf < B; kf += 8u) {
                    simdgroup_bfloat8x8 bf[4];
                    #pragma unroll
                    for (uint j = 0u; j < 4u; ++j) simdgroup_load(bf[j], b + (sc * 32u + j * 8u) * L + kf, L, ulong2(0), true);
                    #pragma unroll
                    for (uint i = 0u; i < RM; ++i) {
                        simdgroup_bfloat8x8 af;
                        simdgroup_load(af, a + (row_base + i * 8u) * L + kf, L);
                        #pragma unroll
                        for (uint j = 0u; j < 4u; ++j) simdgroup_multiply_accumulate(acc[i][j], af, bf[j], acc[i][j]);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (M >= 16u || sr == 0u) {
            const uint col0 = c0 + sc * 32u + fc;
            #pragma unroll
            for (uint i = 0u; i < RM; ++i) {
                const uint t = t0 + row_base + i * 8u + fr;
                if (t >= count) continue;
                const uint r = list[t + 1u];
                #pragma unroll
                for (uint j = 0u; j < 4u; ++j) {
                    mk_expert_block_store<Down>(arena, partial, p, r, col0 + j * 8u, acc[i][j].thread_elements()[0], acc[i][j].thread_elements()[1]);
                }
            }
        }
    }
}

#define MK_EXPERT_BLOCK_M64N64(NAME, DOWN, NV, X16) \
    kernel void NAME(device float *arena [[buffer(0)]], device const uchar *w [[buffer(1)]], device const uchar *s [[buffer(2)]], \
                     device const uchar *amax [[buffer(3)]], device const uchar *denom [[buffer(4)]], device const uint *lists [[buffer(5)]], \
                     device float *partial [[buffer(6)]], constant MKExpert &p [[buffer(8)]], uint g [[threadgroup_position_in_grid]], \
                     uint tid [[thread_index_in_threadgroup]], uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
        alignas(8) threadgroup bfloat a[64u * 40u], b[64u * 40u]; \
        const uint n = DOWN ? p.n : 2u * p.n; \
        const uint cols = n / INKLING_EXPERT_BLOCK_N64, parts = p.rows >= 2048u ? 8u : p.rows >= 256u ? 4u : p.rows >= 64u ? 2u : 1u; \
        const uint expert = g / (cols * parts), part = (g / cols) % parts; \
        const uint count = lists[expert * (p.rows * p.top_k + 1u)]; \
        const uint full = (count / 64u) * 64u, tail = count - full; \
        g = expert * cols + g % cols; \
        if (part * 64u < full) { \
            mk_expert_block_m64n64<DOWN, NV, 64u, X16>(arena, w, s, amax, denom, lists, partial, p, g, tid, simd, lane, part * 64u, full, 64u * parts, a, b); \
        } \
        if ((full / 64u) % parts != part) return; \
        if (tail > 32u) { \
            mk_expert_block_m64n64<DOWN, NV, 64u, X16>(arena, w, s, amax, denom, lists, partial, p, g, tid, simd, lane, full, count, 64u, a, b); \
        } else if (tail > 16u) { \
            mk_expert_block_m64n64<DOWN, NV, 32u, X16>(arena, w, s, amax, denom, lists, partial, p, g, tid, simd, lane, full, count, 32u, a, b); \
        } else if (tail > 8u) { \
            mk_expert_block_m64n64<DOWN, NV, 16u, X16>(arena, w, s, amax, denom, lists, partial, p, g, tid, simd, lane, full, count, 16u, a, b); \
        } else if (tail != 0u) { \
            mk_expert_block_m64n64<DOWN, NV, 8u, X16>(arena, w, s, amax, denom, lists, partial, p, g, tid, simd, lane, full, count, 8u, a, b); \
        } \
    }
MK_EXPERT_BLOCK_M64N64(mk_expert_block_up_nv64_m64, false, true, true)
MK_EXPERT_BLOCK_M64N64(mk_expert_block_down_nv64_m64, true, true, false)
MK_EXPERT_BLOCK_M64N64(mk_expert_block_up_mx64_m64, false, false, true)
MK_EXPERT_BLOCK_M64N64(mk_expert_block_down_mx64_m64, true, false, false)
#undef MK_EXPERT_BLOCK_M64N64



