// DFlash 2 block-diffusion draft kernels (companion kernels for DFlash2DraftModel).
// Shared types (MK, mk_bf16, mk_load, MK_ROUND_BF16, MK_ROUND_OUT) come from kernels.metal.
#include <metal_simdgroup_matrix>

// Fast hardware MMA BF16 linear projection (N=64, K=32, 128 threads = 4 SIMDgroups per 64 output cols).
// Caller guarantees w_rows >= ceil(p.m / 64) * 64, p.n >= 32, p.n % 32 == 0, and 8-byte weight alignment.
template <uint M>
inline void df_bf16_matrix_tile(device const float *src, device const uchar *weights,
                                device float *dst, constant MK &p,
                                uint group, uint tid, uint simd, uint lane,
                                threadgroup bfloat *a) {
    constexpr uint K = 32u, L = K + 8u, N = 64u;
    const uint tiles_n = (p.m + N - 1u) / N;
    const uint r0 = (group / tiles_n) * M;
    const uint c0 = (group % tiles_n) * N;
    const uint fr = ((lane / 4u) & 4u) + (lane / 2u) % 4u;
    const uint fc = (((lane / 4u) & 2u) + lane % 2u) * 2u;
    device const bfloat *w = reinterpret_cast<device const bfloat *>(weights + p.w) + ulong(c0 + simd * 16u) * p.n;
    simdgroup_float8x8 acc[M / 8u][2];
    #pragma unroll
    for (uint i = 0u; i < M / 8u; ++i)
        for (uint j = 0u; j < 2u; ++j)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    const uint rr = tid / (K / 4u), ac = tid % (K / 4u);
    const bool lo = r0 + rr < p.rows, hi = r0 + rr + 16u < p.rows;
    const device float4 *xlo = reinterpret_cast<const device float4 *>(src + p.x + ulong(p.row0 + (lo ? r0 + rr : 0u)) * p.xs);
    const device float4 *xhi = reinterpret_cast<const device float4 *>(src + p.x + ulong(p.row0 + (hi ? r0 + rr + 16u : 0u)) * p.xs);
    for (uint k = 0u; k < p.n; k += K) {
        if (rr < M) {
            reinterpret_cast<threadgroup bfloat4 *>(a + rr * L)[ac] = bfloat4(lo ? xlo[k / 4u + ac] : float4(0.0f));
        }
        if constexpr (M > 16u) {
            reinterpret_cast<threadgroup bfloat4 *>(a + (rr + 16u) * L)[ac] = bfloat4(hi ? xhi[k / 4u + ac] : float4(0.0f));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (uint kf = 0u; kf < K; kf += 8u) {
            simdgroup_bfloat8x8 af[M / 8u], bf[2];
            #pragma unroll
            for (uint i = 0u; i < M / 8u; ++i) simdgroup_load(af[i], a + i * 8u * L + kf, L);
            #pragma unroll
            for (uint i = 0u; i < 2u; ++i) simdgroup_load(bf[i], w + ulong(i * 8u) * p.n + k + kf, p.n, ulong2(0), true);
            #pragma unroll
            for (uint i = 0u; i < M / 8u; ++i)
                for (uint j = 0u; j < 2u; ++j)
                    simdgroup_multiply_accumulate(acc[i][j], af[i], bf[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint cbase = c0 + simd * 16u + fc;
    #pragma unroll
    for (uint i = 0u; i < M / 8u; ++i) {
        const uint row = r0 + i * 8u + fr;
        if (row >= p.rows) continue;
        device float *out = dst + p.y + ulong(p.row0 + row) * p.ys + cbase;
        #pragma unroll
        for (uint j = 0u; j < 2u; ++j)
            for (uint e = 0u; e < 2u; ++e) {
                const uint col = cbase + j * 8u + e;
                if (col < p.m) {
                    const float value = acc[i][j].thread_elements()[e];
                    out[j * 8u + e] = (p.flags & MK_ROUND_OUT) ? mk_bf16(value) : value;
                }
            }
    }
}

kernel void df_bf16_matrix8(device const float *src [[buffer(0)]], device const uchar *weights [[buffer(1)]],
                            device float *dst [[buffer(2)]], constant MK &p [[buffer(8)]],
                            uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                            uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    alignas(8) threadgroup bfloat a[8u * 40u];
    df_bf16_matrix_tile<8u>(src, weights, dst, p, group, tid, simd, lane, a);
}

kernel void df_bf16_matrix32(device const float *src [[buffer(0)]], device const uchar *weights [[buffer(1)]],
                             device float *dst [[buffer(2)]], constant MK &p [[buffer(8)]],
                             uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                             uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    alignas(8) threadgroup bfloat a[32u * 40u];
    df_bf16_matrix_tile<32u>(src, weights, dst, p, group, tid, simd, lane, a);
}

// Coalesced 2-output-row-per-SIMD BF16 linear core for DFlash 2 (rows = 1..8, arbitrary p.m).
template <bool Wide>
inline void df_bf16_2xR_core(device const float *src, device const uchar *weights, device float *dst,
                             constant MK &p, uint row, uint lane) {
    float acc0[8] = {};
    float acc1[8] = {};
    const uint rows = min(p.rows, 8u);
    const bool has1 = (row + 1u) < p.m;
    const bool round_in = (p.flags & MK_ROUND_BF16) != 0u;
    const bool round_out = (p.flags & MK_ROUND_OUT) != 0u;
    const uint xs4 = p.xs / 4u;
    device const float4 *xbase = (device const float4 *)(src + p.x + ulong(p.row0) * p.xs);
    if constexpr (Wide) {
        const uint wstride = p.n / 8u;
        device const uint4 *w8 = (device const uint4 *)(weights + p.w) + ulong(row) * wstride;
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
                    float4 a0 = xbase[ulong(r) * xs4 + 2u * k];
                    float4 a1 = xbase[ulong(r) * xs4 + 2u * k + 1u];
                    if (round_in) {
                        a0 = float4(mk_bf16(a0.x), mk_bf16(a0.y), mk_bf16(a0.z), mk_bf16(a0.w));
                        a1 = float4(mk_bf16(a1.x), mk_bf16(a1.y), mk_bf16(a1.z), mk_bf16(a1.w));
                    }
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
        const uint stride = p.n / 4u;
        device const ushort4 *w4 = (device const ushort4 *)(weights + p.w) + ulong(row) * stride;
        for (uint k = lane; k < stride; k += 32u) {
            const float4 b0 = as_type<float4>(uint4(w4[k]) << 16);
            const float4 b1 = has1 ? as_type<float4>(uint4(w4[stride + k]) << 16) : float4(0.0f);
            #pragma unroll
            for (uint r = 0u; r < 8u; ++r) {
                if (r < rows) {
                    float4 a = xbase[ulong(r) * xs4 + k];
                    if (round_in) a = float4(mk_bf16(a.x), mk_bf16(a.y), mk_bf16(a.z), mk_bf16(a.w));
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
                device float *out = dst + p.y + ulong(p.row0 + r) * p.ys + row;
                out[0] = round_out ? mk_bf16(s0) : s0;
                if (has1) out[1] = round_out ? mk_bf16(s1) : s1;
            }
        }
    }
}

kernel void df_bf16_linear_2xR_vec(device const float *src [[buffer(0)]], device const uchar *weights [[buffer(1)]],
                                   device float *dst [[buffer(2)]], constant MK &p [[buffer(8)]],
                                   uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    const uint row = (gid / 32u) * 2u;
    if (row >= p.m) return;
    df_bf16_2xR_core<false>(src, weights, dst, p, row, lane);
}

kernel void df_bf16_linear_2xR_vec8(device const float *src [[buffer(0)]], device const uchar *weights [[buffer(1)]],
                                    device float *dst [[buffer(2)]], constant MK &p [[buffer(8)]],
                                    uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    const uint row = (gid / 32u) * 2u;
    if (row >= p.m) return;
    df_bf16_2xR_core<true>(src, weights, dst, p, row, lane);
}

// Multi-row Split-K small-M BF16 linear for DFlash 2 (8 SIMDs = 256 threads per output row, rows <= 8).
kernel void df_bf16_linear_splitk_rows(device const float *src [[buffer(0)]], device const uchar *weights [[buffer(1)]],
                                       device float *dst [[buffer(2)]], constant MK &p [[buffer(8)]],
                                       uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                       uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    if (row >= p.m) return;
    const uint rows = min(p.rows, 8u);
    const bool round_in = (p.flags & MK_ROUND_BF16) != 0u;
    const bool round_out = (p.flags & MK_ROUND_OUT) != 0u;
    const uint stride = p.n / 4u;
    const uint xs4 = p.xs / 4u;
    device const ushort4 *w4 = (device const ushort4 *)(weights + p.w) + ulong(row) * stride;
    device const float4 *xbase = (device const float4 *)(src + p.x + ulong(p.row0) * p.xs);
    float acc[8] = {};
    for (uint k = tid; k < stride; k += 256u) {
        const float4 b = as_type<float4>(uint4(w4[k]) << 16);
        #pragma unroll
        for (uint r = 0u; r < 8u; ++r) {
            if (r < rows) {
                float4 a = xbase[ulong(r) * xs4 + k];
                if (round_in) a = float4(mk_bf16(a.x), mk_bf16(a.y), mk_bf16(a.z), mk_bf16(a.w));
                acc[r] = fma(a.w, b.w, fma(a.z, b.z, fma(a.y, b.y, fma(a.x, b.x, acc[r]))));
            }
        }
    }
    threadgroup float partials[8][8];
    #pragma unroll
    for (uint r = 0u; r < 8u; ++r) {
        if (r < rows) {
            const float s = simd_sum(acc[r]);
            if (lane == 0u) partials[r][simd] = s;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0u) {
        #pragma unroll
        for (uint r = 0u; r < 8u; ++r) {
            if (r < rows) {
                const float total = simd_sum(lane < 8u ? partials[r][lane] : 0.0f);
                if (lane == 0u) dst[ulong(p.y) + ulong(p.row0 + r) * p.ys + row] = round_out ? mk_bf16(total) : total;
            }
        }
    }
}

// Fast small-batch hardware MMA BF16 linear projection (M<=8, N=16, K=64, 64 threads = 2 SIMDs, 3.4 KB TG).
// Raw coalesced uint4 weight loads -> threadgroup bfloat4 -> simdgroup_bfloat8x8 MMA.
// Safe for arbitrary p.m (including unpadded vocab 200058). Requires p.rows <= 8, p.n % 64 == 0, p.w % 16 == 0.
kernel void df_bf16_mma_m8n16(device const float *src [[buffer(0)]], device const uchar *weights [[buffer(1)]],
                              device float *dst [[buffer(2)]], constant MK &p [[buffer(8)]],
                              uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                              uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint K = 64u, L = 72u;
    alignas(16) threadgroup bfloat a[8u * L];
    alignas(16) threadgroup bfloat b[16u * L];
    const uint c0 = group * 16u;
    const uint rows = min(p.rows, 8u);
    const uint ar = tid / 8u, ac = (tid % 8u) * 8u;
    const uint wc = tid / 4u, wk = (tid % 4u) * 16u;
    const bool a_ok = ar < rows;
    const bool w_ok = (c0 + wc) < p.m;
    const device float4 *xp = reinterpret_cast<const device float4 *>(src + p.x + ulong(p.row0 + (a_ok ? ar : 0u)) * p.xs);
    const uint wstride8 = p.n / 8u;
    const device uint4 *wp = reinterpret_cast<const device uint4 *>(weights + p.w) + ulong(w_ok ? (c0 + wc) : 0u) * wstride8;
    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k = 0u; k < p.n; k += K) {
        const float4 v0 = a_ok ? xp[(k + ac) / 4u] : float4(0.0f);
        const float4 v1 = a_ok ? xp[(k + ac) / 4u + 1u] : float4(0.0f);
        reinterpret_cast<threadgroup bfloat4 *>(a + ar * L + ac)[0] = bfloat4(v0);
        reinterpret_cast<threadgroup bfloat4 *>(a + ar * L + ac)[1] = bfloat4(v1);
        threadgroup uint4 *bd = reinterpret_cast<threadgroup uint4 *>(b + wc * L + wk);
        bd[0] = w_ok ? wp[(k + wk) / 8u] : uint4(0u);
        bd[1] = w_ok ? wp[(k + wk) / 8u + 1u] : uint4(0u);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (uint kf = 0u; kf < K; kf += 8u) {
            simdgroup_bfloat8x8 af, bf;
            simdgroup_load(af, a + kf, L);
            simdgroup_load(bf, b + simd * 8u * L + kf, L, ulong2(0), true);
            simdgroup_multiply_accumulate(acc, af, bf, acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint fr = ((lane / 4u) & 4u) + (lane / 2u) % 4u;
    const uint fc = (((lane / 4u) & 2u) + lane % 2u) * 2u;
    if (fr < rows) {
        const uint col = c0 + simd * 8u + fc;
        const bool round_out = (p.flags & MK_ROUND_OUT) != 0u;
        device float *out = dst + p.y + ulong(p.row0 + fr) * p.ys + col;
        if (col < p.m) {
            const float v0 = acc.thread_elements()[0];
            out[0] = round_out ? mk_bf16(v0) : v0;
        }
        if (col + 1u < p.m) {
            const float v1 = acc.thread_elements()[1];
            out[1] = round_out ? mk_bf16(v1) : v1;
        }
    }
}

// Split-K Direct-read hardware MMA BF16 linear projection (M<=8, N=16, K=64, 64 threads = 2 SIMDs, 1.15 KB TG).
// Loads weight 8x8 fragments straight from L1/L2 via simdgroup_load without TG staging, and shards K across
// `p.taps` splits (`splits == 1` writes directly to dst; `splits > 1` writes FP32 partials to `dst`).
// Requires p.rows <= 8, (p.m % 16 == 0 or w_rows >= ceil(p.m/16)*16 for splits==1), p.n % 64 == 0, p.w % 16 == 0, p.x % 4 == 0, p.xs % 4 == 0.
kernel void df_bf16_mma_m8n16_direct(device const float *src [[buffer(0)]], device const uchar *weights [[buffer(1)]],
                                     device float *dst [[buffer(2)]], constant MK &p [[buffer(8)]],
                                     uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                     uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint K = 64u, L = 72u;
    alignas(16) threadgroup bfloat a[8u * L];
    const uint n_tiles = (p.m + 15u) / 16u;
    const uint splits = max(p.taps, 1u);
    const uint split = group / n_tiles;
    if (split >= splits) return;
    const uint c0 = (group % n_tiles) * 16u;
    if (c0 >= p.m) return;
    const uint rows = min(p.rows, 8u);
    const uint k_blocks = p.n / 64u;
    const uint kb0 = (k_blocks * split) / splits;
    const uint kb1 = (k_blocks * (split + 1u)) / splits;
    const uint ar = tid / 8u, ac = (tid % 8u) * 8u;
    const bool a_ok = ar < rows;
    const device float4 *xp = reinterpret_cast<const device float4 *>(src + p.x + ulong(p.row0 + (a_ok ? ar : 0u)) * p.xs);
    const device bfloat *w_base = reinterpret_cast<const device bfloat *>(weights + p.w) + ulong(c0 + simd * 8u) * p.n;
    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint kb = kb0; kb < kb1; ++kb) {
        const uint k = kb * 64u;
        const float4 v0 = a_ok ? xp[(k + ac) / 4u] : float4(0.0f);
        const float4 v1 = a_ok ? xp[(k + ac) / 4u + 1u] : float4(0.0f);
        reinterpret_cast<threadgroup bfloat4 *>(a + ar * L + ac)[0] = bfloat4(v0);
        reinterpret_cast<threadgroup bfloat4 *>(a + ar * L + ac)[1] = bfloat4(v1);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (uint kf = 0u; kf < K; kf += 8u) {
            simdgroup_bfloat8x8 af, bf;
            simdgroup_load(af, a + kf, L);
            simdgroup_load(bf, w_base + k + kf, p.n, ulong2(0), true);
            simdgroup_multiply_accumulate(acc, af, bf, acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint fr = ((lane / 4u) & 4u) + (lane / 2u) % 4u;
    const uint fc = (((lane / 4u) & 2u) + lane % 2u) * 2u;
    if (fr < rows) {
        const uint col = c0 + simd * 8u + fc;
        const float v0 = acc.thread_elements()[0];
        const float v1 = acc.thread_elements()[1];
        if (splits == 1u) {
            const bool round_out = (p.flags & MK_ROUND_OUT) != 0u;
            device float *out = dst + p.y + ulong(p.row0 + fr) * p.ys + col;
            if (col < p.m) out[0] = round_out ? mk_bf16(v0) : v0;
            if (col + 1u < p.m) out[1] = round_out ? mk_bf16(v1) : v1;
        } else {
            device float *out = dst + (ulong(split) * rows + fr) * p.m + col;
            out[0] = v0;
            out[1] = v1;
        }
    }
}

// Direct-read hardware MMA BF16 linear projection (M<=8, N=32, K=64, 128 threads = 4 SIMDs).
// Caller guarantees w_rows >= ((p.m + 31) / 32) * 32, p.n % 64 == 0, p.w % 16 == 0, p.x % 4 == 0, p.xs % 4 == 0.
kernel void df_bf16_mma_m8n32_direct(device const float *src [[buffer(0)]], device const uchar *weights [[buffer(1)]],
                                     device float *dst [[buffer(2)]], constant MK &p [[buffer(8)]],
                                     uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                     uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint K = 64u, L = 72u;
    alignas(16) threadgroup bfloat a[8u * L];
    const uint c0 = group * 32u;
    if (c0 >= p.m) return;
    const uint rows = min(p.rows, 8u);
    const uint ar = tid / 16u, ac = (tid % 16u) * 4u;
    const bool a_ok = ar < rows;
    const device float4 *xp = reinterpret_cast<const device float4 *>(src + p.x + ulong(p.row0 + (a_ok ? ar : 0u)) * p.xs);
    const device bfloat *w_base = reinterpret_cast<const device bfloat *>(weights + p.w) + ulong(c0 + simd * 8u) * p.n;
    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k = 0u; k < p.n; k += K) {
        const float4 v = a_ok ? xp[(k + ac) / 4u] : float4(0.0f);
        reinterpret_cast<threadgroup bfloat4 *>(a + ar * L + ac)[0] = bfloat4(v);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (uint kf = 0u; kf < K; kf += 8u) {
            simdgroup_bfloat8x8 af, bf;
            simdgroup_load(af, a + kf, L);
            simdgroup_load(bf, w_base + k + kf, p.n, ulong2(0), true);
            simdgroup_multiply_accumulate(acc, af, bf, acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint fr = ((lane / 4u) & 4u) + (lane / 2u) % 4u;
    const uint fc = (((lane / 4u) & 2u) + lane % 2u) * 2u;
    if (fr < rows) {
        const uint col = c0 + simd * 8u + fc;
        const bool round_out = (p.flags & MK_ROUND_OUT) != 0u;
        device float *out = dst + p.y + ulong(p.row0 + fr) * p.ys + col;
        if (col < p.m) {
            const float v0 = acc.thread_elements()[0];
            out[0] = round_out ? mk_bf16(v0) : v0;
        }
        if (col + 1u < p.m) {
            const float v1 = acc.thread_elements()[1];
            out[1] = round_out ? mk_bf16(v1) : v1;
        }
    }
}

// Fused multi-matrix direct-read MMA BF16 projection (M<=8, N=16, 64 threads = 2 SIMDs per tile).
// Used for Q+K+V (propose) and K+V (commitContext) sharing the same input and weight buffer.
kernel void df_multi_bf16_mma_m8n16_direct(device const float *src [[buffer(0)]], device const uchar *weights [[buffer(1)]],
                                           device float *dst [[buffer(2)]], constant MKX &x [[buffer(8)]],
                                           uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                           uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    constant MK &p = x.p;
    uint which = 0u;
    uint local_tile = group;
    for (; which < x.count; ++which) {
        const uint n_tiles = (x.mat[which].m + 15u) / 16u;
        if (local_tile < n_tiles) break;
        local_tile -= n_tiles;
    }
    if (which >= x.count) return;

    constexpr uint K = 64u, L = 72u;
    alignas(16) threadgroup bfloat a[8u * L];
    const uint c0 = local_tile * 16u;
    const uint m_mat = x.mat[which].m;
    const uint rows = min(p.rows, 8u);
    const uint ar = tid / 8u, ac = (tid % 8u) * 8u;
    const bool a_ok = ar < rows;
    const device float4 *xp = reinterpret_cast<const device float4 *>(src + p.x + ulong(p.row0 + (a_ok ? ar : 0u)) * p.xs);
    const device bfloat *w_base = reinterpret_cast<const device bfloat *>(weights + x.mat[which].w) + ulong(c0 + simd * 8u) * p.n;
    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k = 0u; k < p.n; k += K) {
        const float4 v0 = a_ok ? xp[(k + ac) / 4u] : float4(0.0f);
        const float4 v1 = a_ok ? xp[(k + ac) / 4u + 1u] : float4(0.0f);
        reinterpret_cast<threadgroup bfloat4 *>(a + ar * L + ac)[0] = bfloat4(v0);
        reinterpret_cast<threadgroup bfloat4 *>(a + ar * L + ac)[1] = bfloat4(v1);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (uint kf = 0u; kf < K; kf += 8u) {
            simdgroup_bfloat8x8 af, bf;
            simdgroup_load(af, a + kf, L);
            simdgroup_load(bf, w_base + k + kf, p.n, ulong2(0), true);
            simdgroup_multiply_accumulate(acc, af, bf, acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint fr = ((lane / 4u) & 4u) + (lane / 2u) % 4u;
    const uint fc = (((lane / 4u) & 2u) + lane % 2u) * 2u;
    if (fr < rows) {
        const uint col = c0 + simd * 8u + fc;
        const bool round_out = (p.flags & MK_ROUND_OUT) != 0u;
        device float *out = dst + x.mat[which].y + ulong(p.row0 + fr) * x.mat[which].ys + col;
        if (col < m_mat) {
            const float v0 = acc.thread_elements()[0];
            out[0] = round_out ? mk_bf16(v0) : v0;
        }
        if (col + 1u < m_mat) {
            const float v1 = acc.thread_elements()[1];
            out[1] = round_out ? mk_bf16(v1) : v1;
        }
    }
}

// Fused gate + up MMA BF16 projection + inline SwiGLU (SiLU(gate) * up) for M <= 8.
// Shares threadgroup activation tile `a[8 * 72]` across gate (p.w) and up (p.s) weights.
kernel void df_bf16_gate_up_swiglu_mma(device const float *src [[buffer(0)]], device const uchar *weights [[buffer(1)]],
                                       device float *dst [[buffer(2)]], constant MK &p [[buffer(8)]],
                                       uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                       uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint K = 64u, L = 72u;
    alignas(16) threadgroup bfloat a[8u * L];
    const uint c0 = group * 16u;
    if (c0 >= p.m) return;
    const uint rows = min(p.rows, 8u);
    const uint ar = tid / 8u, ac = (tid % 8u) * 8u;
    const bool a_ok = ar < rows;
    const device float4 *xp = reinterpret_cast<const device float4 *>(src + p.x + ulong(p.row0 + (a_ok ? ar : 0u)) * p.xs);
    const ulong col_off = ulong(c0 + simd * 8u) * p.n;
    const device bfloat *wg_base = reinterpret_cast<const device bfloat *>(weights + p.w) + col_off;
    const device bfloat *wu_base = reinterpret_cast<const device bfloat *>(weights + p.s) + col_off;
    simdgroup_float8x8 acc_g = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc_u = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k = 0u; k < p.n; k += K) {
        const float4 v0 = a_ok ? xp[(k + ac) / 4u] : float4(0.0f);
        const float4 v1 = a_ok ? xp[(k + ac) / 4u + 1u] : float4(0.0f);
        reinterpret_cast<threadgroup bfloat4 *>(a + ar * L + ac)[0] = bfloat4(v0);
        reinterpret_cast<threadgroup bfloat4 *>(a + ar * L + ac)[1] = bfloat4(v1);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (uint kf = 0u; kf < K; kf += 8u) {
            simdgroup_bfloat8x8 af, bg, bu;
            simdgroup_load(af, a + kf, L);
            simdgroup_load(bg, wg_base + k + kf, p.n, ulong2(0), true);
            simdgroup_load(bu, wu_base + k + kf, p.n, ulong2(0), true);
            simdgroup_multiply_accumulate(acc_g, af, bg, acc_g);
            simdgroup_multiply_accumulate(acc_u, af, bu, acc_u);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint fr = ((lane / 4u) & 4u) + (lane / 2u) % 4u;
    const uint fc = (((lane / 4u) & 2u) + lane % 2u) * 2u;
    if (fr < rows) {
        const uint col = c0 + simd * 8u + fc;
        const bool round_out = (p.flags & MK_ROUND_OUT) != 0u;
        device float *out = dst + p.y + ulong(p.row0 + fr) * p.ys + col;
        if (col < p.m) {
            const float g0 = acc_g.thread_elements()[0];
            const float u0 = acc_u.thread_elements()[0];
            const float res0 = (g0 * mk_sigmoid(g0)) * u0;
            out[0] = round_out ? mk_bf16(res0) : res0;
        }
        if (col + 1u < p.m) {
            const float g1 = acc_g.thread_elements()[1];
            const float u1 = acc_u.thread_elements()[1];
            const float res1 = (g1 * mk_sigmoid(g1)) * u1;
            out[1] = round_out ? mk_bf16(res1) : res1;
        }
    }
}

// Wide uint4 (8 bf16) Multi-row Split-K small-M BF16 linear for DFlash 2 (rows <= 8, n % 256 == 0, w % 16 == 0).
kernel void df_bf16_linear_splitk_rows_vec8(device const float *src [[buffer(0)]], device const uchar *weights [[buffer(1)]],
                                            device float *dst [[buffer(2)]], constant MK &p [[buffer(8)]],
                                            uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                            uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    if (row >= p.m) return;
    const uint rows = min(p.rows, 8u);
    const bool round_in = (p.flags & MK_ROUND_BF16) != 0u;
    const bool round_out = (p.flags & MK_ROUND_OUT) != 0u;
    const uint wstride = p.n / 8u;
    const uint xs4 = p.xs / 4u;
    device const uint4 *w8 = (device const uint4 *)(weights + p.w) + ulong(row) * wstride;
    device const float4 *xbase = (device const float4 *)(src + p.x + ulong(p.row0) * p.xs);
    float acc[8] = {};
    for (uint k = tid; k < wstride; k += 256u) {
        const uint4 raw = w8[k];
        const ushort4 u0(ushort(raw.x), ushort(raw.x >> 16), ushort(raw.y), ushort(raw.y >> 16));
        const ushort4 u1(ushort(raw.z), ushort(raw.z >> 16), ushort(raw.w), ushort(raw.w >> 16));
        const float4 b0 = as_type<float4>(uint4(u0) << 16);
        const float4 b1 = as_type<float4>(uint4(u1) << 16);
        #pragma unroll
        for (uint r = 0u; r < 8u; ++r) {
            if (r < rows) {
                float4 a0 = xbase[ulong(r) * xs4 + 2u * k];
                float4 a1 = xbase[ulong(r) * xs4 + 2u * k + 1u];
                if (round_in) {
                    a0 = float4(mk_bf16(a0.x), mk_bf16(a0.y), mk_bf16(a0.z), mk_bf16(a0.w));
                    a1 = float4(mk_bf16(a1.x), mk_bf16(a1.y), mk_bf16(a1.z), mk_bf16(a1.w));
                }
                acc[r] = fma(a0.w, b0.w, fma(a0.z, b0.z, fma(a0.y, b0.y, fma(a0.x, b0.x, acc[r]))));
                acc[r] = fma(a1.w, b1.w, fma(a1.z, b1.z, fma(a1.y, b1.y, fma(a1.x, b1.x, acc[r]))));
            }
        }
    }
    threadgroup float partials[8][8];
    #pragma unroll
    for (uint r = 0u; r < 8u; ++r) {
        if (r < rows) {
            const float s = simd_sum(acc[r]);
            if (lane == 0u) partials[r][simd] = s;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd < rows) {
        const uint r = simd;
        const float total = simd_sum(lane < 8u ? partials[r][lane] : 0.0f);
        if (lane == 0u) dst[ulong(p.y) + ulong(p.row0 + r) * p.ys + row] = round_out ? mk_bf16(total) : total;
    }
}

kernel void df_bf16_splitk_reduce(device const float *partials [[buffer(0)]], device float *dst [[buffer(1)]],
                                  constant MK &p [[buffer(8)]], uint gid [[thread_position_in_grid]]) {
    const uint cols4 = p.m / 4u;
    const uint rows = min(p.rows, 8u);
    const uint r = gid / cols4, c4 = gid % cols4;
    if (r >= rows) return;
    const uint splits = max(p.taps, 1u);
    device const float4 *p4 = reinterpret_cast<const device float4 *>(partials);
    float4 sum = 0.0f;
    for (uint s = 0u; s < splits; ++s) sum += p4[(ulong(s) * rows + r) * cols4 + c4];
    if ((p.flags & MK_ROUND_OUT) != 0u) sum = float4(mk_bf16(sum.x), mk_bf16(sum.y), mk_bf16(sum.z), mk_bf16(sum.w));
    reinterpret_cast<device float4 *>(dst + p.y + ulong(p.row0 + r) * p.ys)[c4] = sum;
}

inline float4 df_load_bf16_4(device const uchar *base, ulong addr) {
    if ((addr & 7ul) == 0ul) {
        ushort4 h = *(device const ushort4 *)(base + addr);
        return float4(as_type<float>(uint(h.x) << 16),
                      as_type<float>(uint(h.y) << 16),
                      as_type<float>(uint(h.z) << 16),
                      as_type<float>(uint(h.w) << 16));
    }
    if ((addr & 1ul) == 0ul) {
        device const ushort *hp = (device const ushort *)(base + addr);
        return float4(as_type<float>(uint(hp[0]) << 16),
                      as_type<float>(uint(hp[1]) << 16),
                      as_type<float>(uint(hp[2]) << 16),
                      as_type<float>(uint(hp[3]) << 16));
    }
    device const uchar *q = base + addr;
    uint h0 = uint(q[0]) | (uint(q[1]) << 8);
    uint h1 = uint(q[2]) | (uint(q[3]) << 8);
    uint h2 = uint(q[4]) | (uint(q[5]) << 8);
    uint h3 = uint(q[6]) | (uint(q[7]) << 8);
    return float4(as_type<float>(h0 << 16), as_type<float>(h1 << 16),
                  as_type<float>(h2 << 16), as_type<float>(h3 << 16));
}

inline ushort df_to_bf16_bits(float v) {
    uint bits = as_type<uint>(v);
    if ((bits & 0x7f800000u) == 0x7f800000u) return ushort(bits >> 16);
    bits = bits + 0x7fffu + ((bits >> 16) & 1u);
    return ushort(bits >> 16);
}

// Fast multi-row BF16 linear projection for small batches (p.rows <= 8, p.n % 4 == 0).
// Grid: ceil(m / 4) * 32 threads (1 SIMDgroup of 32 lanes per 4 output rows), group = 128.
// Buffers: 0 = x (float), 1 = w (bf16), 2 = y (float).
kernel void df_bf16_linear(
    device const float *src [[buffer(0)]],
    device const uchar *w_base [[buffer(1)]],
    device float *dst [[buffer(2)]],
    constant MK &p [[buffer(8)]],
    uint simd_id [[threadgroup_position_in_grid]],
    uint sidx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    // 4 SIMDgroups per threadgroup -> reconstruct global SIMDgroup index cleanly.
    uint sg = simd_id * 4u + sidx;
    uint o0 = sg * 4u;
    if (o0 >= p.m) return;
    uint rows = min(p.rows, 8u);
    bool round_in = (p.flags & MK_ROUND_BF16) != 0u;
    bool round_out = (p.flags & MK_ROUND_OUT) != 0u;

    float acc[8][4];
    for (uint r = 0; r < rows; ++r)
        for (uint j = 0; j < 4u; ++j)
            acc[r][j] = 0.0f;

    ulong row_stride_b = ulong(p.n) * 2ul;
    ulong w0 = p.w + ulong(o0) * row_stride_b;

    for (uint k = lane * 4u; k < p.n; k += 128u) {
        float4 xv[8];
        for (uint r = 0; r < rows; ++r) {
            device const float *xp = src + ulong(p.x) + ulong(p.row0 + r) * p.xs + k;
            float4 v = float4(xp[0], xp[1], xp[2], xp[3]);
            if (round_in) v = float4(mk_bf16(v.x), mk_bf16(v.y), mk_bf16(v.z), mk_bf16(v.w));
            xv[r] = v;
        }
        ulong k_off = ulong(k) * 2ul;
        for (uint j = 0; j < 4u; ++j) {
            if (o0 + j >= p.m) break;
            float4 wv = df_load_bf16_4(w_base, w0 + ulong(j) * row_stride_b + k_off);
            for (uint r = 0; r < rows; ++r) {
                acc[r][j] += dot(xv[r], wv);
            }
        }
    }

    for (uint r = 0; r < rows; ++r) {
        for (uint j = 0; j < 4u; ++j) {
            if (o0 + j >= p.m) continue;
            float s = simd_sum(acc[r][j]);
            if (lane == 0u) {
                dst[ulong(p.y) + ulong(p.row0 + r) * p.ys + o0 + j] = round_out ? mk_bf16(s) : s;
            }
        }
    }
}

// 32x32 tiled simdgroup_float8x8 BF16 matmul for larger batches (p.rows > 8, e.g. prefill context).
// Dispatch grid: ceil(p.m / 32) * ceil(p.rows / 32) * 128 threads, group = 128 (or 2D grid -> use 1D TG).
// Using 1D threadgroups of 128 threads: tg_id decomposes into tile_m = tg_id % tiles_m, tile_r = tg_id / tiles_m.
kernel void df_bf16_linear_tile(
    device const float *src [[buffer(0)]],
    device const uchar *w_base [[buffer(1)]],
    device float *dst [[buffer(2)]],
    constant MK &p [[buffer(8)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint sidx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float xs[32][32];
    threadgroup float ws[32][32];
    threadgroup float cs[32][32];

    uint tiles_m = (p.m + 31u) / 32u;
    uint tm = tg_id % tiles_m;
    uint tr = tg_id / tiles_m;
    uint r_base = tr * 32u;
    uint m_base = tm * 32u;
    if (r_base >= p.rows || m_base >= p.m) return;

    bool round_in = (p.flags & MK_ROUND_BF16) != 0u;
    bool round_out = (p.flags & MK_ROUND_OUT) != 0u;

    uint sr = (sidx >> 1) * 16u;
    uint sm = (sidx & 1u) * 16u;

    simdgroup_float8x8 acc00 = simdgroup_float8x8(0.0f);
    simdgroup_float8x8 acc01 = simdgroup_float8x8(0.0f);
    simdgroup_float8x8 acc10 = simdgroup_float8x8(0.0f);
    simdgroup_float8x8 acc11 = simdgroup_float8x8(0.0f);

    ulong row_stride_b = ulong(p.n) * 2ul;

    // 128 threads load 32x32 = 1024 elements -> 8 elements per thread (two float4 / 4-elem chunks).
    uint lr = tid >> 3;        // 0..15
    uint lk = (tid & 7u) * 4u; // 0,4,8,...,28

    for (uint k0 = 0; k0 < p.n; k0 += 32u) {
        // Load two rows of xs and two rows of ws per thread.
        for (uint pass = 0; pass < 2u; ++pass) {
            uint tr_i = lr + pass * 16u;
            uint gr = r_base + tr_i;
            float4 xv = float4(0.0f);
            if (gr < p.rows && k0 + lk < p.n) {
                device const float *xp = src + ulong(p.x) + ulong(p.row0 + gr) * p.xs + k0 + lk;
                xv = float4(xp[0], xp[1], xp[2], xp[3]);
                if (round_in) xv = float4(mk_bf16(xv.x), mk_bf16(xv.y), mk_bf16(xv.z), mk_bf16(xv.w));
            }
            xs[tr_i][lk + 0u] = xv.x;
            xs[tr_i][lk + 1u] = xv.y;
            xs[tr_i][lk + 2u] = xv.z;
            xs[tr_i][lk + 3u] = xv.w;

            uint gm = m_base + tr_i;
            float4 wv = float4(0.0f);
            if (gm < p.m && k0 + lk < p.n) {
                wv = df_load_bf16_4(w_base, p.w + ulong(gm) * row_stride_b + ulong(k0 + lk) * 2ul);
            }
            // Store transposed into ws[k][m] so B in simdgroup_multiply_accumulate is KxN contiguous.
            ws[lk + 0u][tr_i] = wv.x;
            ws[lk + 1u][tr_i] = wv.y;
            ws[lk + 2u][tr_i] = wv.z;
            ws[lk + 3u][tr_i] = wv.w;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint ks = 0; ks < 32u; ks += 8u) {
            simdgroup_float8x8 a0, a1, b0, b1;
            simdgroup_load(a0, &xs[sr + 0u][ks], 32);
            simdgroup_load(a1, &xs[sr + 8u][ks], 32);
            simdgroup_load(b0, &ws[ks][sm + 0u], 32);
            simdgroup_load(b1, &ws[ks][sm + 8u], 32);
            simdgroup_multiply_accumulate(acc00, a0, b0, acc00);
            simdgroup_multiply_accumulate(acc01, a0, b1, acc01);
            simdgroup_multiply_accumulate(acc10, a1, b0, acc10);
            simdgroup_multiply_accumulate(acc11, a1, b1, acc11);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(acc00, &cs[sr + 0u][sm + 0u], 32);
    simdgroup_store(acc01, &cs[sr + 0u][sm + 8u], 32);
    simdgroup_store(acc10, &cs[sr + 8u][sm + 0u], 32);
    simdgroup_store(acc11, &cs[sr + 8u][sm + 8u], 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint pass = 0; pass < 2u; ++pass) {
        uint tr_i = lr + pass * 16u;
        uint gr = r_base + tr_i;
        if (gr >= p.rows) continue;
        for (uint j = 0; j < 4u; ++j) {
            uint gm = m_base + lk + j;
            if (gm < p.m) {
                float v = cs[tr_i][lk + j];
                dst[ulong(p.y) + ulong(p.row0 + gr) * p.ys + gm] = round_out ? mk_bf16(v) : v;
            }
        }
    }
}

// Grouped 2-tap intra-block causal dynamic convolution (DFlash 2).
// p.kind = stage (0 = prepare, 1 = finish + residual); p.heads = groups; p.dim = group_size; p.taps = kernel_size.
// Buffers: 0 = arena, 1 = base_kernel (bf16 [2, taps, n]).
kernel void df_grouped_dyn_conv(
    device float *arena [[buffer(0)]],
    device const uchar *base_w [[buffer(1)]],
    constant MK &p [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) {
    uint r = gid / p.n;
    uint c = gid % p.n;
    if (r >= p.rows) return;
    uint stage = p.kind;
    uint g = c / p.dim;
    float acc = 0.0f;
    for (uint off = 0; off < p.taps; ++off) {
        if (r >= off) {
            float v = arena[ulong(p.x) + ulong(p.row0 + r - off) * p.xs + c];
            float bw = mk_load(base_w, p.w, p.dtype, (ulong(stage) * p.taps + off) * p.n + c);
            uint dyn_idx = (stage * p.taps + off) * p.heads + g;
            float dw = arena[ulong(p.z) + ulong(p.row0 + r) * p.zs + dyn_idx];
            acc += bw * v;
            acc += dw * v;
        }
    }
    if (stage == 1u) {
        acc += arena[ulong(p.pad_r) + ulong(p.row0 + r) * p.pad_k + c];
    }
    arena[ulong(p.y) + ulong(p.row0 + r) * p.ys + c] = acc;
}

// Vectorized 4-channel grouped 2-tap dynamic convolution (p.n % 4 == 0, p.dim % 4 == 0, 4-aligned arena & aligned base_w).
kernel void df_grouped_dyn_conv_vec4(
    device float *arena [[buffer(0)]],
    device const uchar *base_w [[buffer(1)]],
    constant MK &p [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) {
    const uint n4 = p.n / 4u;
    const uint r = gid / n4;
    const uint c4 = gid % n4;
    if (r >= p.rows) return;
    const uint stage = p.kind;
    const uint g = (c4 * 4u) / p.dim;
    const uint xs4 = p.xs / 4u;
    const uint ys4 = p.ys / 4u;
    float4 acc = float4(0.0f);
    for (uint off = 0u; off < p.taps; ++off) {
        if (r >= off) {
            const float4 v = reinterpret_cast<device const float4 *>(arena + ulong(p.x) + ulong(p.row0 + r - off) * p.xs)[c4];
            const uint base_idx4 = ((stage * p.taps + off) * p.n) / 4u + c4;
            const float4 bw = mk_fast_load4(base_w, p.w, p.dtype, base_idx4);
            const uint dyn_idx = (stage * p.taps + off) * p.heads + g;
            const float dw = arena[ulong(p.z) + ulong(p.row0 + r) * p.zs + dyn_idx];
            acc += bw * v;
            acc += dw * v;
        }
    }
    if (stage == 1u) {
        acc += reinterpret_cast<device const float4 *>(arena + ulong(p.pad_r) + ulong(p.row0 + r) * p.pad_k)[c4];
    }
    reinterpret_cast<device float4 *>(arena + ulong(p.y) + ulong(p.row0 + r) * p.ys)[c4] = acc;
    (void)xs4; (void)ys4;
}

// SiLU(gate) * up -> out inside arena.
kernel void df_swiglu_mul(
    device float *arena [[buffer(0)]],
    constant MK &p [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) {
    uint r = gid / p.n;
    uint c = gid % p.n;
    if (r >= p.rows) return;
    uint row = p.row0 + r;
    float g = arena[ulong(p.x) + ulong(row) * p.xs + c];
    float u = arena[ulong(p.y) + ulong(row) * p.ys + c];
    arena[ulong(p.z) + ulong(row) * p.zs + c] = (g * mk_sigmoid(g)) * u;
}

// Store S context rows of [n] floats into BF16 ring buffer at (pos + r) % capacity.
kernel void df_kv_ring_store(
    device const float *arena [[buffer(0)]],
    device ushort *ring [[buffer(1)]],
    constant MK &p [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) {
    uint r = gid / p.n;
    uint c = gid % p.n;
    if (r >= p.rows) return;
    uint slot = (p.pos + r) % p.capacity;
    float v = arena[ulong(p.x) + ulong(p.row0 + r) * p.xs + c];
    ring[ulong(slot) * p.n + c] = df_to_bf16_bits(v);
}

// Vectorized 4-word BF16 ring store (requires p.n % 4 == 0, p.x % 4 == 0, p.xs % 4 == 0).
kernel void df_kv_ring_store_vec4(
    device const float *arena [[buffer(0)]],
    device ushort *ring [[buffer(1)]],
    constant MK &p [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) {
    const uint n4 = p.n / 4u;
    const uint r = gid / n4;
    const uint c4 = gid % n4;
    if (r >= p.rows) return;
    const uint slot = (p.pos + r) % p.capacity;
    const float4 v = reinterpret_cast<device const float4 *>(arena + ulong(p.x) + ulong(p.row0 + r) * p.xs)[c4];
    reinterpret_cast<device ushort4 *>(ring + ulong(slot) * p.n)[c4] = ushort4(
        df_to_bf16_bits(v.x), df_to_bf16_bits(v.y), df_to_bf16_bits(v.z), df_to_bf16_bits(v.w));
}

#if DFLASH_HEAD_DIM != 128u || DFLASH_GQA4_RATIO != 4u
#error dflash2 geometry mismatch with verified inkling_plan.metal
#endif
// Fused per-head RMSNorm + RoPE in-place for head_dim == 128, rot == 128 (zero TG barriers).
// 1 SIMDgroup (32 lanes) per (row, head), 4 SIMDgroups per threadgroup (128 threads).
kernel void df_rmsnorm_rope(
    device const uchar *w [[buffer(0)]],
    device float *arena [[buffer(1)]],
    device const float *tables [[buffer(2)]],
    constant MK &p [[buffer(8)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint head_idx = tgid * 4u + simd;
    if (head_idx >= p.rows * p.heads) return;
    const uint r = head_idx / p.heads;
    const uint h = head_idx % p.heads;
    const uint row = p.row0 + r;
    device float4 *xp = reinterpret_cast<device float4 *>(arena + ulong(p.x) + ulong(row) * p.xs + ulong(h) * 128u);
    const float4 xv = xp[lane];
    const float tot = simd_sum(dot(xv, xv));
    const float inv = metal::precise::rsqrt(tot / 128.0f + p.eps);
    const float4 nv = mk_fast_load4(w, p.w, p.dtype, lane) * (xv * inv);

    const float4 peer = float4(
        simd_shuffle_xor(nv.x, 16u),
        simd_shuffle_xor(nv.y, 16u),
        simd_shuffle_xor(nv.z, 16u),
        simd_shuffle_xor(nv.w, 16u));
    const bool lo = lane < 16u;
    const float4 a = lo ? nv : peer;
    const float4 b = lo ? peer : nv;
    const float4 th = float(p.pos + row) * reinterpret_cast<device const float4 *>(tables + p.z)[lane & 15u];
    const float4 c = (p.flags & MK_FAST_TRIG)
        ? float4(metal::fast::cos(th.x), metal::fast::cos(th.y), metal::fast::cos(th.z), metal::fast::cos(th.w))
        : float4(metal::precise::cos(th.x), metal::precise::cos(th.y), metal::precise::cos(th.z), metal::precise::cos(th.w));
    const float4 s = (p.flags & MK_FAST_TRIG)
        ? float4(metal::fast::sin(th.x), metal::fast::sin(th.y), metal::fast::sin(th.z), metal::fast::sin(th.w))
        : float4(metal::precise::sin(th.x), metal::precise::sin(th.y), metal::precise::sin(th.z), metal::precise::sin(th.w));
    xp[lane] = lo ? (a * c - b * s) : (a * s + b * c);
}

// Fused per-head RMSNorm + RoPE + BF16 ring store for K in commitContext (head_dim == 128, rot == 128).
kernel void df_rmsnorm_rope_ring_store(
    device const uchar *w [[buffer(0)]],
    device const float *arena [[buffer(1)]],
    device const float *tables [[buffer(2)]],
    device ushort *ring [[buffer(3)]],
    constant MK &p [[buffer(8)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint head_idx = tgid * 4u + simd;
    if (head_idx >= p.rows * p.heads) return;
    const uint r = head_idx / p.heads;
    const uint h = head_idx % p.heads;
    const uint row = p.row0 + r;
    const device float4 *xp = reinterpret_cast<const device float4 *>(arena + ulong(p.x) + ulong(row) * p.xs + ulong(h) * 128u);
    const float4 xv = xp[lane];
    const float tot = simd_sum(dot(xv, xv));
    const float inv = metal::precise::rsqrt(tot / 128.0f + p.eps);
    const float4 nv = mk_fast_load4(w, p.w, p.dtype, lane) * (xv * inv);

    const float4 peer = float4(
        simd_shuffle_xor(nv.x, 16u),
        simd_shuffle_xor(nv.y, 16u),
        simd_shuffle_xor(nv.z, 16u),
        simd_shuffle_xor(nv.w, 16u));
    const bool lo = lane < 16u;
    const float4 a = lo ? nv : peer;
    const float4 b = lo ? peer : nv;
    const float4 th = float(p.pos + row) * reinterpret_cast<device const float4 *>(tables + p.z)[lane & 15u];
    const float4 c = (p.flags & MK_FAST_TRIG)
        ? float4(metal::fast::cos(th.x), metal::fast::cos(th.y), metal::fast::cos(th.z), metal::fast::cos(th.w))
        : float4(metal::precise::cos(th.x), metal::precise::cos(th.y), metal::precise::cos(th.z), metal::precise::cos(th.w));
    const float4 s = (p.flags & MK_FAST_TRIG)
        ? float4(metal::fast::sin(th.x), metal::fast::sin(th.y), metal::fast::sin(th.z), metal::fast::sin(th.w))
        : float4(metal::precise::sin(th.x), metal::precise::sin(th.y), metal::precise::sin(th.z), metal::precise::sin(th.w));
    const float4 out = lo ? (a * c - b * s) : (a * s + b * c);

    const uint slot = (p.pos + row) % p.capacity;
    const uint kv_width = p.heads * 128u;
    device ushort4 *kp = reinterpret_cast<device ushort4 *>(ring + ulong(slot) * kv_width + ulong(h) * 128u);
    kp[lane] = ushort4(
        df_to_bf16_bits(out.x), df_to_bf16_bits(out.y), df_to_bf16_bits(out.z), df_to_bf16_bits(out.w));
}

// GQA-4 shared bidirectional sliding-window attention (p.heads == 4 * p.kv_heads, p.dim == 128).
// 1 TG of 256 threads (8 SIMDs) per (row, kv_head) loads K/V once for all 4 query heads.
kernel void df_attention_gqa4(
    device float *arena [[buffer(0)]],
    device const ushort *k_ring [[buffer(1)]],
    device const ushort *v_ring [[buffer(2)]],
    constant MK &p [[buffer(8)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float maxes[4][8], sums[4][8];
    threadgroup float4 outs[4][8][32];
    const uint r = tgid / p.kv_heads;
    const uint kvh = tgid % p.kv_heads;
    if (r >= p.rows) return;

    const uint ctx_pos = p.pos;
    const uint q_pos = ctx_pos + r;
    const uint kv_width = p.kv_heads * 128u;
    uint ctx_start = (p.window != 0u && q_pos + 1u > p.window) ? (q_pos + 1u - p.window) : 0u;
    if (ctx_start > ctx_pos) ctx_start = ctx_pos;
    const uint total_keys = (ctx_pos - ctx_start) + p.rows;
    const uint h0 = kvh * 4u;

    float4 qv[4], ov[4];
    float m_s[4], l_s[4];
    #pragma unroll
    for (uint g = 0u; g < 4u; ++g) {
        const device float4 *qp = reinterpret_cast<const device float4 *>(
            arena + ulong(p.x) + ulong(p.row0 + r) * p.xs + ulong(h0 + g) * 128u);
        qv[g] = qp[lane];
        ov[g] = float4(0.0f);
        m_s[g] = -INFINITY;
        l_s[g] = 0.0f;
    }

    constexpr float LOG2E_F = 1.4426950408889634f;
    for (uint idx = simd; idx < total_keys; idx += 8u) {
        const uint k_pos = ctx_start + idx;
        float4 kv, vv;
        if (k_pos < ctx_pos) {
            const uint slot = k_pos % p.capacity;
            const ulong base = ulong(slot) * kv_width + ulong(kvh) * 128u;
            const ushort4 kr = reinterpret_cast<device const ushort4 *>(k_ring + base)[lane];
            const ushort4 vr = reinterpret_cast<device const ushort4 *>(v_ring + base)[lane];
            kv = as_type<float4>(uint4(kr) << 16);
            vv = as_type<float4>(uint4(vr) << 16);
        } else {
            const uint nr = k_pos - ctx_pos;
            const device float4 *kp = reinterpret_cast<const device float4 *>(
                arena + ulong(p.y) + ulong(p.row0 + nr) * p.ys + ulong(kvh) * 128u);
            const device float4 *vp = reinterpret_cast<const device float4 *>(
                arena + ulong(p.z) + ulong(p.row0 + nr) * p.zs + ulong(kvh) * 128u);
            kv = kp[lane];
            vv = vp[lane];
        }
        #pragma unroll
        for (uint g = 0u; g < 4u; ++g) {
            const float score = simd_sum(dot(qv[g], kv)) * p.scale;
            const float m_next = max(m_s[g], score);
            const float alpha = (m_s[g] == -INFINITY) ? 0.0f : metal::fast::exp2((m_s[g] - m_next) * LOG2E_F);
            const float beta = metal::fast::exp2((score - m_next) * LOG2E_F);
            l_s[g] = l_s[g] * alpha + beta;
            ov[g] = ov[g] * alpha + beta * vv;
            m_s[g] = m_next;
        }
    }

    #pragma unroll
    for (uint g = 0u; g < 4u; ++g) {
        if (lane == 0u) {
            maxes[g][simd] = m_s[g];
            sums[g][simd] = l_s[g];
        }
        outs[g][simd][lane] = ov[g];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd < 4u) {
        const uint g = simd;
        float m_all = -INFINITY;
        #pragma unroll
        for (uint s = 0u; s < 8u; ++s) m_all = max(m_all, maxes[g][s]);
        float denom = 0.0f;
        float4 acc = float4(0.0f);
        #pragma unroll
        for (uint s = 0u; s < 8u; ++s) {
            if (sums[g][s] == 0.0f) continue;
            const float scale_o = metal::fast::exp2((maxes[g][s] - m_all) * LOG2E_F);
            denom += sums[g][s] * scale_o;
            acc += scale_o * outs[g][s][lane];
        }
        const float inv_d = denom > 0.0f ? (1.0f / denom) : 0.0f;
        device float4 *op = reinterpret_cast<device float4 *>(
            arena + ulong(p.every) + ulong(p.row0 + r) * p.pad_k + ulong(h0 + g) * 128u);
        op[lane] = acc * inv_d;
    }
}

// Bidirectional sliding-window GQA for DFlash 2 (head_dim <= 256, 8 SIMDgroups per (row, head)).
// p.pos = ctx_pos (number of committed context tokens prior to noise block).
// Context keys [max(0, q_pos + 1 - p.window) .. p.pos - 1] live in k_ring/v_ring (bf16 ushort, capacity p.capacity).
// Noise keys [0 .. p.rows - 1] live in arena at k_off (p.y, p.ys) and v_off (p.z, p.zs).
// Q lives at p.x (p.xs), out lives at p.every (p.pad_k).
constant uint DF_ATT_SIMDS = 8u;
kernel void df_attention(
    device float *arena [[buffer(0)]],
    device const ushort *k_ring [[buffer(1)]],
    device const ushort *v_ring [[buffer(2)]],
    constant MK &p [[buffer(8)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float maxes[DF_ATT_SIMDS], sums[DF_ATT_SIMDS], outs[DF_ATT_SIMDS][256];
    uint r = tgid / p.heads;
    uint h = tgid % p.heads;
    if (r >= p.rows) return;

    uint ctx_pos = p.pos;
    uint q_pos = ctx_pos + r;
    uint kvh = h / (p.heads / p.kv_heads);
    uint kv_width = p.kv_heads * p.dim;
    uint ctx_start = (p.window != 0u && q_pos + 1u > p.window) ? (q_pos + 1u - p.window) : 0u;
    if (ctx_start > ctx_pos) ctx_start = ctx_pos;
    uint total_keys = (ctx_pos - ctx_start) + p.rows;

    uint E = (p.dim + 31u) / 32u;
    float qv[8], ov[8];
    device const float *qp = arena + ulong(p.x) + ulong(p.row0 + r) * p.xs + ulong(h) * p.dim;
    for (uint e = 0; e < E; ++e) {
        uint c = E * lane + e;
        qv[e] = (c < p.dim) ? qp[c] : 0.0f;
        ov[e] = 0.0f;
    }

    float m_s = -INFINITY;
    float l_s = 0.0f;
    for (uint idx = simd; idx < total_keys; idx += DF_ATT_SIMDS) {
        uint k_pos = ctx_start + idx;
        float dot_acc = 0.0f;
        float kv[8], vv[8];
        if (k_pos < ctx_pos) {
            uint slot = k_pos % p.capacity;
            ulong base = ulong(slot) * kv_width + ulong(kvh) * p.dim;
            for (uint e = 0; e < E; ++e) {
                uint c = E * lane + e;
                kv[e] = (c < p.dim) ? as_type<float>(uint(k_ring[base + c]) << 16) : 0.0f;
                vv[e] = (c < p.dim) ? as_type<float>(uint(v_ring[base + c]) << 16) : 0.0f;
                dot_acc += qv[e] * kv[e];
            }
        } else {
            uint nr = k_pos - ctx_pos;
            device const float *kp = arena + ulong(p.y) + ulong(p.row0 + nr) * p.ys + ulong(kvh) * p.dim;
            device const float *vp = arena + ulong(p.z) + ulong(p.row0 + nr) * p.zs + ulong(kvh) * p.dim;
            for (uint e = 0; e < E; ++e) {
                uint c = E * lane + e;
                kv[e] = (c < p.dim) ? kp[c] : 0.0f;
                vv[e] = (c < p.dim) ? vp[c] : 0.0f;
                dot_acc += qv[e] * kv[e];
            }
        }
        float score = simd_sum(dot_acc) * p.scale;
        float m_next = max(m_s, score);
        float alpha = (m_s == -INFINITY) ? 0.0f : metal::precise::exp(m_s - m_next);
        float beta = metal::precise::exp(score - m_next);
        l_s = l_s * alpha + beta;
        for (uint e = 0; e < E; ++e) ov[e] = ov[e] * alpha + beta * vv[e];
        m_s = m_next;
    }

    if (lane == 0u) {
        maxes[simd] = m_s;
        sums[simd] = l_s;
    }
    for (uint e = 0; e < E; ++e) {
        uint c = E * lane + e;
        if (c < p.dim) outs[simd][c] = ov[e];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd == 0u) {
        float m_all = -INFINITY;
        for (uint s = 0; s < DF_ATT_SIMDS; ++s) m_all = max(m_all, maxes[s]);
        float denom = 0.0f;
        float acc[8];
        for (uint e = 0; e < E; ++e) acc[e] = 0.0f;
        for (uint s = 0; s < DF_ATT_SIMDS; ++s) {
            if (sums[s] == 0.0f) continue;
            float w = sums[s] * metal::precise::exp(maxes[s] - m_all);
            float scale_o = metal::precise::exp(maxes[s] - m_all);
            denom += w;
            for (uint e = 0; e < E; ++e) {
                uint c = E * lane + e;
                if (c < p.dim) acc[e] += scale_o * outs[s][c];
            }
        }
        float inv_d = denom > 0.0f ? (1.0f / denom) : 0.0f;
        device float *op = arena + ulong(p.every) + ulong(p.row0 + r) * p.pad_k + ulong(h) * p.dim;
        for (uint e = 0; e < E; ++e) {
            uint c = E * lane + e;
            if (c < p.dim) op[c] = acc[e] * inv_d;
        }
    }
}

// Stage 1 of exact GPU top-k (k <= 16) over vocab [0 .. p.m).
// Dispatch grid: (p.rows * shards) threadgroups of 256 threads (shards = p.heads, e.g. 64).
// Outputs: part_vals [rows, shards, k], part_ids [rows, shards, k].
kernel void df_top16_part(
    device const float *logits [[buffer(0)]],
    device float *part_vals [[buffer(1)]],
    device uint *part_ids [[buffer(2)]],
    constant MK &p [[buffer(8)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float tg_v[8];
    threadgroup uint tg_i[8];
    threadgroup uint chosen_id;
    threadgroup float tg_s[8];

    uint shards = p.heads;
    uint top_k = p.n;
    uint r = tgid / shards;
    uint s = tgid % shards;
    if (r >= p.rows) return;

    uint chunk = (p.m + shards - 1u) / shards;
    uint lo = s * chunk;
    uint hi = min(p.m, lo + chunk);
    device const float *row_p = logits + ulong(p.x) + ulong(p.row0 + r) * p.xs;

    float lv[24];
    uint li[24];
    uint count = 0u;
    for (uint idx = lo + tid; idx < hi; idx += 256u) {
        if (count < 24u) {
            lv[count] = row_p[idx];
            li[count] = idx;
            ++count;
        }
    }

    if (p.kind != 0u) {
        float lmax = -INFINITY;
        for (uint j = 0u; j < count; ++j) {
            float v = (p.eps > 0.0f) ? (p.eps * tanh(lv[j] / p.eps)) : lv[j];
            lmax = max(lmax, v);
        }
        float smax = simd_max(lmax);
        if (lane == 0u) tg_v[simd] = smax;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd == 0u) {
            float rmax = simd_max(lane < 8u ? tg_v[lane] : -INFINITY);
            if (lane == 0u) tg_s[0] = rmax;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float shard_max = tg_s[0];
        float lsum = 0.0f;
        if (shard_max > -1e30f) {
            for (uint j = 0u; j < count; ++j) {
                float v = (p.eps > 0.0f) ? (p.eps * tanh(lv[j] / p.eps)) : lv[j];
                lsum += metal::precise::exp(v - shard_max);
            }
        }
        float ssum = simd_sum(lsum);
        if (lane == 0u) tg_v[simd] = ssum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            float tot = 0.0f;
            for (uint w = 0u; w < 8u; ++w) tot += tg_v[w];
            ulong stat_base = ulong(p.rows) * shards * top_k + (ulong(r) * shards + s) * 2u;
            part_vals[stat_base] = shard_max;
            part_vals[stat_base + 1u] = tot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint k = 0u; k < top_k; ++k) {
        float bv = -INFINITY;
        uint bi = 0xffffffffu;
        for (uint j = 0u; j < count; ++j) {
            if (lv[j] > bv || (lv[j] == bv && li[j] < bi)) {
                bv = lv[j];
                bi = li[j];
            }
        }
        // SIMD reduce using shuffle.
        for (uint off = 16u; off > 0u; off >>= 1) {
            float ov = simd_shuffle_down(bv, off);
            uint oi = simd_shuffle_down(bi, off);
            if (ov > bv || (ov == bv && oi < bi)) {
                bv = ov;
                bi = oi;
            }
        }
        if (lane == 0u) {
            tg_v[simd] = bv;
            tg_i[simd] = bi;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0u) {
            float mv = -INFINITY;
            uint mi = 0xffffffffu;
            for (uint w = 0u; w < 8u; ++w) {
                if (tg_v[w] > mv || (tg_v[w] == mv && tg_i[w] < mi)) {
                    mv = tg_v[w];
                    mi = tg_i[w];
                }
            }
            ulong out_idx = (ulong(r) * shards + s) * top_k + k;
            part_vals[out_idx] = mv;
            part_ids[out_idx] = mi;
            chosen_id = mi;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint mask_id = chosen_id;
        for (uint j = 0u; j < count; ++j) {
            if (li[j] == mask_id) lv[j] = -INFINITY;
        }
    }
}

// Stage 2 of exact GPU top-k (k <= 16): merges (shards * top_k) <= 1024 candidates per row.
// Dispatch grid: p.rows threadgroups of 256 threads.
kernel void df_top16_merge(
    device const float *part_vals [[buffer(0)]],
    device const uint *part_ids [[buffer(1)]],
    device float *out_vals [[buffer(2)]],
    device uint *out_ids [[buffer(3)]],
    constant MK &p [[buffer(8)]],
    uint r [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float tg_v[8];
    threadgroup uint tg_i[8];
    threadgroup uint chosen_id;

    if (r >= p.rows) return;
    uint shards = p.heads;
    uint top_k = p.n;
    uint total = shards * top_k;
    ulong row_base = ulong(r) * total;

    float lv[8];
    uint li[8];
    uint count = 0u;
    for (uint idx = tid; idx < total; idx += 256u) {
        if (count < 8u) {
            lv[count] = part_vals[row_base + idx];
            li[count] = part_ids[row_base + idx];
            ++count;
        }
    }

    for (uint k = 0u; k < top_k; ++k) {
        float bv = -INFINITY;
        uint bi = 0xffffffffu;
        for (uint j = 0u; j < count; ++j) {
            if (lv[j] > bv || (lv[j] == bv && li[j] < bi)) {
                bv = lv[j];
                bi = li[j];
            }
        }
        for (uint off = 16u; off > 0u; off >>= 1) {
            float ov = simd_shuffle_down(bv, off);
            uint oi = simd_shuffle_down(bi, off);
            if (ov > bv || (ov == bv && oi < bi)) {
                bv = ov;
                bi = oi;
            }
        }
        if (lane == 0u) {
            tg_v[simd] = bv;
            tg_i[simd] = bi;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0u) {
            float mv = -INFINITY;
            uint mi = 0xffffffffu;
            for (uint w = 0u; w < 8u; ++w) {
                if (tg_v[w] > mv || (tg_v[w] == mv && tg_i[w] < mi)) {
                    mv = tg_v[w];
                    mi = tg_i[w];
                }
            }
            ulong out_idx = ulong(r) * top_k + k;
            out_vals[out_idx] = mv;
            out_ids[out_idx] = mi;
            chosen_id = mi;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint mask_id = chosen_id;
        for (uint j = 0u; j < count; ++j) {
            if (li[j] == mask_id) lv[j] = -INFINITY;
        }
    }

    if (p.kind != 0u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float smax_local = -INFINITY;
        float ssum_local = 0.0f;
        const ulong part_stat_base = ulong(p.rows) * shards * top_k + ulong(r) * shards * 2u;
        for (uint s = tid; s < shards; s += 256u) {
            float m = part_vals[part_stat_base + ulong(s) * 2u];
            float l = part_vals[part_stat_base + ulong(s) * 2u + 1u];
            if (l > 0.0f) {
                if (m > smax_local) {
                    ssum_local = ssum_local * metal::precise::exp(smax_local - m) + l;
                    smax_local = m;
                } else {
                    ssum_local += l * metal::precise::exp(m - smax_local);
                }
            }
        }
        float sm_s = simd_max(smax_local);
        if (lane == 0u) tg_v[simd] = sm_s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd == 0u) {
            float gmax = simd_max(lane < 8u ? tg_v[lane] : -INFINITY);
            if (lane == 0u) tg_v[0] = gmax;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float row_max = tg_v[0];
        float scaled_sum = (ssum_local > 0.0f && row_max > -1e30f) ? (ssum_local * metal::precise::exp(smax_local - row_max)) : 0.0f;
        float s_tot = simd_sum(scaled_sum);
        if (lane == 0u) tg_v[simd] = s_tot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            float z_total = 0.0f;
            for (uint w = 0u; w < 8u; ++w) z_total += tg_v[w];
            float z_top = 0.0f;
            if (row_max > -1e30f) {
                for (uint k = 0u; k < top_k; ++k) {
                    float raw_v = out_vals[ulong(r) * top_k + k];
                    float cap_v = (p.eps > 0.0f) ? (p.eps * tanh(raw_v / p.eps)) : raw_v;
                    z_top += metal::precise::exp(cap_v - row_max);
                }
            }
            float z_tail = max(z_total - z_top, 0.0f);
            out_vals[ulong(p.rows) * top_k + r] = (z_tail > 1e-30f && row_max > -1e30f)
                ? (row_max + metal::precise::log(z_tail))
                : -1e30f;
        }
    }
}

// DFlash 2 CandidateSelector sequential greedy path decoding.
// One threadgroup of 512 threads (16 SIMDgroups, k = simd < top_k).
// p.rows = draft positions (B - 1, e.g. 7), p.n = selector_rank (e.g. 256),
// p.m = top_k (e.g. 16), p.pos = anchor_token, p.eps = softcap (30.0 for after_correction, 0 = none).
// p.w = pred_codebook byte offset, p.s = succ_codebook byte offset, p.dtype = codebook dtype (2 = bf16).
// When p.kind != 0, writes post-selector top-1 confidence as float bit-pattern to out_tokens[p.rows + 1 + pos].
kernel void df_select_path(
    device const float *h_sel [[buffer(0)]],
    device const uchar *pred_cb [[buffer(1)]],
    device const uchar *succ_cb [[buffer(2)]],
    device const float *top_vals [[buffer(3)]],
    device const uint *top_ids [[buffer(4)]],
    device uint *out_tokens [[buffer(5)]],
    constant MK &p [[buffer(8)]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float tg_scores[32];
    threadgroup uint tg_cands[32];
    threadgroup uint tg_prev;

    uint top_k = p.m;
    uint rank = p.n;
    if (tid == 0u) {
        tg_prev = p.pos;
        out_tokens[0] = p.pos;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint pos = 0u; pos < p.rows; ++pos) {
        uint prev = tg_prev;
        if (simd < top_k) {
            uint cand = top_ids[ulong(pos) * top_k + simd];
            float dot_acc = 0.0f;
            for (uint d = lane; d < rank; d += 32u) {
                float pr = mk_load(pred_cb, p.w, p.dtype, ulong(prev) * rank + d);
                float hs = h_sel[ulong(pos) * p.xs + p.x + d];
                float sc = mk_load(succ_cb, p.s, p.dtype, ulong(cand) * rank + d);
                dot_acc += pr * hs * sc;
            }
            float edge = simd_sum(dot_acc);
            if (lane == 0u) {
                float raw = top_vals[ulong(pos) * top_k + simd] + edge;
                tg_scores[simd] = (p.eps > 0.0f) ? (p.eps * tanh(raw / p.eps)) : raw;
                tg_cands[simd] = cand;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0u) {
            float best_s = -INFINITY;
            uint best_c = 0u;
            for (uint k = 0u; k < top_k; ++k) {
                float s = tg_scores[k];
                uint c = tg_cands[k];
                if (s > best_s || (s == best_s && c < best_c)) {
                    best_s = s;
                    best_c = c;
                }
            }
            if (p.kind != 0u) {
                float tail_lse = top_vals[ulong(p.rows) * top_k + pos];
                float m = best_s;
                if (tail_lse > -1e29f && tail_lse > m) m = tail_lse;
                float z = (tail_lse > -1e29f) ? metal::precise::exp(tail_lse - m) : 0.0f;
                for (uint k = 0u; k < top_k; ++k) {
                    z += metal::precise::exp(tg_scores[k] - m);
                }
                float conf = (z > 0.0f) ? clamp(metal::precise::exp(best_s - m) / z, 0.0f, 1.0f) : 0.0f;
                out_tokens[p.rows + 1u + pos] = as_type<uint>(conf);
            }
            tg_prev = best_c;
            out_tokens[1u + pos] = best_c;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
