// Full-Slice Multi-Head Q4 Latent Sparse Attention (LSA) for Inkling-2 global layers.
// Adapted from qwegn (https://github.com/qwegnai/qwegn, Apache-2.0):
//   - qwlsa_mh_common.metalh, qwlsa_mh.metal, qwlsa_mh_select.metal
// Adapted for nano's arena/mirror layout, learned attention sinks, relative bias,
// and single-axis fm_dispatch threadgroup grid (<= 7 buffers per kernel).

// Host mirror: LSAParams in src/mega/gpu.zig (240 bytes).
struct MKLSA {
    ulong off_head_scores;
    ulong off_tg_stats;
    ulong off_head_lse;
    ulong off_scores;
    ulong off_hist;
    ulong off_cand_count;
    ulong off_cand_m;
    ulong off_cand_u;
    ulong off_fine_cand;
    ulong off_fine_count;
    ulong rel_w_off;
    uint units, rows, row0, pos0;
    uint row_step, unit_offset, dim, heads;
    uint kv_heads, capacity, q_off, q_stride;
    uint sink_off, sink_stride, bias_off, bias_stride;
    uint out_off, out_stride, m, budget_blocks;
    uint window_blocks, coarse_cap, pooled_meta_bytes, pooled_row_bytes;
    uint start_blk, num_blks, list_stride, score_stride;
    uint head_score_stride, cand_cap, score_tiles_x, stage_mode;
    uint bq_tile, taps, flags, has_sink;
    float scale;
    uint tile_items;
};

struct MKLSAGeom {
    uint qRow;
    uint totalBlocks;
    uint middleEnd;
    uint numMiddle;
    uint kEff;
    uint isSparse;
    uint totalCoarse;
    uint numCoarseMid;
    uint coarseKEff;
};

inline MKLSAGeom mk_lsa_geom(constant MKLSA &p, uint unit) {
    MKLSAGeom g;
    uint global_unit = p.unit_offset + unit;
    g.qRow = (p.row_step > 1u) ? min(global_unit * p.row_step + (p.row_step - 1u), p.rows - 1u) : global_unit;
    g.totalBlocks = (p.pos0 + g.qRow + 1u) >> 2u;
    g.isSparse = (g.totalBlocks > 1u + p.budget_blocks + p.window_blocks) ? 1u : 0u;
    g.middleEnd = (g.isSparse != 0u) ? (g.totalBlocks - p.window_blocks) : 0u;
    g.numMiddle = (g.middleEnd > 1u) ? (g.middleEnd - 1u) : 0u;
    g.kEff = min(p.budget_blocks, g.numMiddle);
    g.totalCoarse = g.totalBlocks >> 2u;
    g.numCoarseMid = min(g.totalCoarse, (g.middleEnd + 3u) >> 2u);
    g.coarseKEff = min(p.coarse_cap, g.numCoarseMid);
    return g;
}

inline void mk_lsa_pack_q4_head(device half2 *metaOut, device uchar *payOut, uint head, uint sPairs,
                                uint tid, uint sg, uint lane, bool active, float v0, float v1,
                                threadgroup float *tgMin, threadgroup float *tgMax) {
    float lMin = active ? min(v0, v1) : 1e30f;
    float lMax = active ? max(v0, v1) : -1e30f;
    float sgMin = simd_min(lMin);
    float sgMax = simd_max(lMax);
    if (lane == 0u) {
        tgMin[sg] = sgMin;
        tgMax[sg] = sgMax;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float mnF = min(min(tgMin[0], tgMin[1]), min(tgMin[2], tgMin[3]));
    float mxF = max(max(tgMax[0], tgMax[1]), max(tgMax[2], tgMax[3]));
    half mnH = half(mnF);
    half scH = half(max(0.0f, mxF - float(mnH)) / 15.0f);
    if (tid == 0u) metaOut[head] = half2(scH, mnH);
    if (active) {
        float scVal = float(scH);
        float mnVal = float(mnH);
        float invSc = (scVal > 0.0f) ? (1.0f / scVal) : 0.0f;
        uint q0 = uint(clamp(rint((v0 - mnVal) * invSc), 0.0f, 15.0f));
        uint q1 = uint(clamp(rint((v1 - mnVal) * invSc), 0.0f, 15.0f));
        payOut[head * sPairs + tid] = uchar(q0 | (q1 << 4u));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void mk_lsa_pick_radix_bin(threadgroup uint *hist, threadgroup uint *scanSum,
                                  threadgroup uint &greaterAccum, threadgroup uint &chosenBin,
                                  threadgroup uint &sgTotal, uint kEff, bool addGreater,
                                  uint tid, uint sg, uint lane) {
    uint c = (tid < 256u) ? hist[255u - tid] : 0u;
    uint pSum = simd_prefix_inclusive_sum(c);
    if (lane == 31u) scanSum[sg] = pSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint sOff = 0u;
    for (uint s = 0u; s < sg; ++s) sOff += scanSum[s];
    uint incl = sOff + pSum;
    uint excl = incl - c;
    uint base = addGreater ? greaterAccum : 0u;
    if (base + excl < kEff && base + incl >= kEff) {
        chosenBin = 255u - tid;
        sgTotal = base + incl;
        greaterAccum = base + excl;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void mk_lsa_fast_merge(threadgroup uint *keep, threadgroup uint &keepCount,
                              device const uint *cm, device const uint *cu,
                              uint rawCount, uint total, uint greater, uint needTies,
                              uint threshU, device uint *outBuf, uint outOffset, uint tid) {
    constexpr uint TIE_FLAG = 0x80000000u;
    uint n = 256u;
    while (n < total) n <<= 1u;
    if (tid == 0u) keepCount = 0u;
    for (uint i = tid; i < n; i += 256u) keep[i] = 0xFFFFFFFFu;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < rawCount; i += 256u) {
        uint u = cu[i];
        if (u >= threshU) {
            uint slot = atomic_fetch_add_explicit((threadgroup atomic_uint *)&keepCount, 1u, memory_order_relaxed);
            keep[slot] = cm[i] | (u == threshU ? TIE_FLAG : 0u);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 2u; k <= n; k <<= 1u) {
        for (uint j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint i = tid; i < n; i += 256u) {
                uint ixj = i ^ j;
                if (ixj > i) {
                    uint a = keep[i], c = keep[ixj];
                    if ((a > c) == ((i & k) == 0u)) {
                        keep[i] = c;
                        keep[ixj] = a;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    uint bBase = total - needTies;
    for (uint i = tid; i < greater; i += 256u) {
        uint v = keep[i];
        uint lo = 0u, hi = needTies;
        while (lo < hi) {
            uint mid = (lo + hi) >> 1u;
            if ((keep[bBase + mid] & ~TIE_FLAG) < v) lo = mid + 1u;
            else hi = mid;
        }
        outBuf[outOffset + i + lo] = v;
    }
    for (uint j = tid; j < needTies; j += 256u) {
        uint w = keep[bBase + j] & ~TIE_FLAG;
        uint lo = 0u, hi = greater;
        while (lo < hi) {
            uint mid = (lo + hi) >> 1u;
            if (keep[mid] < w) lo = mid + 1u;
            else hi = mid;
        }
        outBuf[outOffset + j + lo] = w;
    }
}

// ---- Q4 key block pooling ---------------------------------------------------

inline float mk_lsa_load_key_f32(device const float *k_f32, ulong idx) { return k_f32[idx]; }
inline float mk_lsa_load_key_bf16(device const ushort *k_bf16, ulong idx) { return as_type<float>(uint(k_bf16[idx]) << 16u); }

// Pools 4 consecutive key tokens into per-(block, kvHead) asymmetric uint4.
kernel void mk_lsa_pool_fine_f32(device const float *K [[buffer(0)]], device uchar *pooledKeys [[buffer(1)]],
                                 constant MKLSA &p [[buffer(8)]], uint tg [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]]) {
    if (tg >= p.num_blks) return;
    uint blk = p.start_blk + tg;
    uint p0 = blk << 2u;
    uint r0 = (p0 + 0u) % p.capacity, r1 = (p0 + 1u) % p.capacity;
    uint r2 = (p0 + 2u) % p.capacity, r3 = (p0 + 3u) % p.capacity;
    uint rowStrideK = p.kv_heads * p.dim;
    uint sPairs = p.dim >> 1u;

    device uchar *blkPtr = pooledKeys + ulong(blk) * p.pooled_row_bytes;
    device half2 *metaOut = (device half2 *)blkPtr;
    device uchar *payOut = blkPtr + p.pooled_meta_bytes;
    threadgroup float tgMin[4], tgMax[4];
    uint sg = tid >> 5u, lane = tid & 31u;
    bool active = tid < sPairs;

    for (uint h = 0u; h < p.kv_heads; ++h) {
        float v0 = 0.0f, v1 = 0.0f;
        if (active) {
            uint hOff = h * p.dim + (tid << 1u);
            v0 = 0.25f * (K[ulong(r0) * rowStrideK + hOff] + K[ulong(r1) * rowStrideK + hOff] +
                          K[ulong(r2) * rowStrideK + hOff] + K[ulong(r3) * rowStrideK + hOff]);
            v1 = 0.25f * (K[ulong(r0) * rowStrideK + hOff + 1u] + K[ulong(r1) * rowStrideK + hOff + 1u] +
                          K[ulong(r2) * rowStrideK + hOff + 1u] + K[ulong(r3) * rowStrideK + hOff + 1u]);
        }
        mk_lsa_pack_q4_head(metaOut, payOut, h, sPairs, tid, sg, lane, active, v0, v1, tgMin, tgMax);
    }
}

kernel void mk_lsa_pool_fine_bf16(device const ushort *K [[buffer(0)]], device uchar *pooledKeys [[buffer(1)]],
                                  constant MKLSA &p [[buffer(8)]], uint tg [[threadgroup_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]]) {
    if (tg >= p.num_blks) return;
    uint blk = p.start_blk + tg;
    uint p0 = blk << 2u;
    uint r0 = (p0 + 0u) % p.capacity, r1 = (p0 + 1u) % p.capacity;
    uint r2 = (p0 + 2u) % p.capacity, r3 = (p0 + 3u) % p.capacity;
    uint rowStrideK = p.kv_heads * p.dim;
    uint sPairs = p.dim >> 1u;

    device uchar *blkPtr = pooledKeys + ulong(blk) * p.pooled_row_bytes;
    device half2 *metaOut = (device half2 *)blkPtr;
    device uchar *payOut = blkPtr + p.pooled_meta_bytes;
    threadgroup float tgMin[4], tgMax[4];
    uint sg = tid >> 5u, lane = tid & 31u;
    bool active = tid < sPairs;

    for (uint h = 0u; h < p.kv_heads; ++h) {
        float v0 = 0.0f, v1 = 0.0f;
        if (active) {
            uint hOff = h * p.dim + (tid << 1u);
            v0 = 0.25f * (mk_lsa_load_key_bf16(K, ulong(r0) * rowStrideK + hOff) +
                          mk_lsa_load_key_bf16(K, ulong(r1) * rowStrideK + hOff) +
                          mk_lsa_load_key_bf16(K, ulong(r2) * rowStrideK + hOff) +
                          mk_lsa_load_key_bf16(K, ulong(r3) * rowStrideK + hOff));
            v1 = 0.25f * (mk_lsa_load_key_bf16(K, ulong(r0) * rowStrideK + hOff + 1u) +
                          mk_lsa_load_key_bf16(K, ulong(r1) * rowStrideK + hOff + 1u) +
                          mk_lsa_load_key_bf16(K, ulong(r2) * rowStrideK + hOff + 1u) +
                          mk_lsa_load_key_bf16(K, ulong(r3) * rowStrideK + hOff + 1u));
        }
        mk_lsa_pack_q4_head(metaOut, payOut, h, sPairs, tid, sg, lane, active, v0, v1, tgMin, tgMax);
    }
}

// Pools 4 consecutive Q4 fine blocks into a 16-token Q4 coarse block per KV head.
kernel void mk_lsa_pool_coarse(device const uchar *pooledKeys [[buffer(0)]], device uchar *pooledCoarseKeys [[buffer(1)]],
                               constant MKLSA &p [[buffer(8)]], uint tg [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]]) {
    if (tg >= p.num_blks) return;
    uint cBlk = p.start_blk + tg;
    uint f0 = cBlk << 2u;
    uint sPairs = p.dim >> 1u;
    device uchar *outPtr = pooledCoarseKeys + ulong(cBlk) * p.pooled_row_bytes;
    device half2 *metaOut = (device half2 *)outPtr;
    device uchar *payOut = outPtr + p.pooled_meta_bytes;
    threadgroup float tgMin[4], tgMax[4];
    uint sg = tid >> 5u, lane = tid & 31u;
    bool active = tid < sPairs;

    for (uint h = 0u; h < p.kv_heads; ++h) {
        float v0 = 0.0f, v1 = 0.0f;
        for (uint b = 0u; b < 4u; ++b) {
            device const uchar *fPtr = pooledKeys + ulong(f0 + b) * p.pooled_row_bytes;
            half2 hm = ((device const half2 *)fPtr)[h];
            if (active) {
                uchar pk = (fPtr + p.pooled_meta_bytes)[h * sPairs + tid];
                float sc = float(hm.x), mn = float(hm.y);
                v0 += mn + sc * float(pk & 0xFu);
                v1 += mn + sc * float(pk >> 4u);
            }
        }
        mk_lsa_pack_q4_head(metaOut, payOut, h, sPairs, tid, sg, lane, active, v0 * 0.25f, v1 * 0.25f, tgMin, tgMax);
    }
}

// ---- Multi-head Q4 scoring & radix block selection ---------------------------

template <uint U>
inline void mk_lsa_mh_score_body(device const float *arena, device const uchar *keyPool,
                                 device const uint *fineCandIdx, device half *mhHeadScores,
                                 device float2 *mhTGStats, constant MKLSA &p,
                                 threadgroup const MKLSAGeom *gArr, uint uBase, uint uCnt,
                                 uint shFineCnt, const thread uint *itemLim, uint maxItemLim,
                                 threadgroup float2 (*sgStat)[64], uint tg_x, uint tid) {
    uint sg = tid >> 5u, lane = tid & 31u;
    uint gPerKV = p.heads / p.kv_heads;
    uint sPairs = p.dim >> 1u;
    uint headVec4 = p.dim >> 2u;
    uint fineStride = p.coarse_cap * 4u;

    float myQSum0[U], myMax0[U], mySum0[U];
    float myQSum1[U], myMax1[U], mySum1[U];
    half myScore0[U], myScore1[U];
    #pragma unroll
    for (uint qb = 0u; qb < U; ++qb) {
        float qs0 = 0.0f, qs1 = 0.0f;
        if (qb < uCnt && gArr[qb].isSparse != 0u) {
            device const float *qRowBase = arena + p.q_off + ulong(p.row0 + gArr[qb].qRow) * p.q_stride;
            if (lane < p.heads) {
                device const packed_float4 *qPtr = (device const packed_float4 *)(qRowBase + lane * p.dim);
                float4 s4 = float4(0.0f);
                for (uint k = 0u; k < headVec4; ++k) s4 += float4(qPtr[k]);
                qs0 = (s4.x + s4.y) + (s4.z + s4.w);
            }
            if ((lane + 32u) < p.heads) {
                device const packed_float4 *qPtr = (device const packed_float4 *)(qRowBase + (lane + 32u) * p.dim);
                float4 s4 = float4(0.0f);
                for (uint k = 0u; k < headVec4; ++k) s4 += float4(qPtr[k]);
                qs1 = (s4.x + s4.y) + (s4.z + s4.w);
            }
        }
        myQSum0[qb] = qs0;
        myQSum1[qb] = qs1;
        myMax0[qb] = -1e30f;
        mySum0[qb] = 0.0f;
        myScore0[qb] = half(0.0h);
        myMax1[qb] = -1e30f;
        mySum1[qb] = 0.0f;
        myScore1[qb] = half(0.0h);
    }

    for (uint w = 0u; w < 32u; ++w) {
        uint item = (tg_x << 8u) + (w << 3u) + sg;
        if (item < maxItemLim) {
            uint blk = item;
            if (p.stage_mode == 2u) {
                if (item < shFineCnt) blk = fineCandIdx[ulong(uBase) * fineStride + item];
                else if (item == shFineCnt) blk = 0u;
                else blk = gArr[0].middleEnd + (item - shFineCnt - 1u);
            }
            device const uchar *blkPtr = keyPool + ulong(blk) * p.pooled_row_bytes;
            device const half2 *metaPtr = (device const half2 *)blkPtr;
            device const uchar *payPtr = blkPtr + p.pooled_meta_bytes;
            for (uint g = 0u; g < p.kv_heads; ++g) {
                half2 hm = metaPtr[g];
                float sc = float(hm.x), mn = float(hm.y);
                device const ushort *pay2 = (device const ushort *)(payPtr + g * sPairs);
                ushort pk0 = (lane < headVec4) ? pay2[lane] : 0u;
                ushort pk1 = ((lane + 32u) < headVec4) ? pay2[lane + 32u] : 0u;
                float4 n0 = float4(float(pk0 & 0xFu), float((pk0 >> 4u) & 0xFu),
                                   float((pk0 >> 8u) & 0xFu), float((pk0 >> 12u) & 0xFu));
                float4 n1 = float4(float(pk1 & 0xFu), float((pk1 >> 4u) & 0xFu),
                                   float((pk1 >> 8u) & 0xFu), float((pk1 >> 12u) & 0xFu));
                #pragma unroll
                for (uint qb = 0u; qb < U; ++qb) {
                    if (qb >= uCnt || gArr[qb].isSparse == 0u || item >= itemLim[qb]) continue;
                    device const float *qRowBase = arena + p.q_off + ulong(p.row0 + gArr[qb].qRow) * p.q_stride;
                    for (uint sub = 0u; sub < gPerKV; ++sub) {
                        uint qh = g * gPerKV + sub;
                        device const packed_float4 *qPtr = (device const packed_float4 *)(qRowBase + qh * p.dim);
                        float acc = (lane < headVec4) ? dot(float4(qPtr[lane]), n0) : 0.0f;
                        if ((lane + 32u) < headVec4) acc += dot(float4(qPtr[lane + 32u]), n1);
                        float dotN = simd_sum(acc);
                        if (lane == qh) {
                            myScore0[qb] = half(clamp((sc * dotN + mn * myQSum0[qb]) * p.scale, -60000.0f, 60000.0f));
                        } else if ((lane + 32u) == qh) {
                            myScore1[qb] = half(clamp((sc * dotN + mn * myQSum1[qb]) * p.scale, -60000.0f, 60000.0f));
                        }
                    }
                }
            }
            #pragma unroll
            for (uint qb = 0u; qb < U; ++qb) {
                if (qb < uCnt && gArr[qb].isSparse != 0u && item < itemLim[qb]) {
                    if (lane < p.heads) {
                        float sf = float(myScore0[qb]);
                        float oldM = myMax0[qb];
                        float newM = max(oldM, sf);
                        mySum0[qb] = (oldM > -1e20f ? mySum0[qb] * metal::precise::exp(oldM - newM) : 0.0f) +
                                     metal::precise::exp(sf - newM);
                        myMax0[qb] = newM;
                    }
                    if ((lane + 32u) < p.heads) {
                        float sf = float(myScore1[qb]);
                        float oldM = myMax1[qb];
                        float newM = max(oldM, sf);
                        mySum1[qb] = (oldM > -1e20f ? mySum1[qb] * metal::precise::exp(oldM - newM) : 0.0f) +
                                     metal::precise::exp(sf - newM);
                        myMax1[qb] = newM;
                    }
                    uint writeIdx = 0u;
                    bool doWrite = false;
                    if (p.stage_mode == 0u) {
                        if (item >= 1u && item < gArr[qb].middleEnd) {
                            writeIdx = item - 1u;
                            doWrite = true;
                        }
                    } else if (p.stage_mode == 1u) {
                        if (item < gArr[qb].numCoarseMid) {
                            writeIdx = item;
                            doWrite = true;
                        }
                    } else {
                        if (item < shFineCnt) {
                            writeIdx = item;
                            doWrite = true;
                        }
                    }
                    if (doWrite) {
                        ulong rowOff = (ulong(uBase + qb) * p.head_score_stride + writeIdx) * p.heads;
                        if (lane < p.heads) mhHeadScores[rowOff + lane] = myScore0[qb];
                        if ((lane + 32u) < p.heads) mhHeadScores[rowOff + lane + 32u] = myScore1[qb];
                    }
                }
            }
        }
    }

    #pragma unroll
    for (uint qb = 0u; qb < U; ++qb) {
        if (qb < uCnt && gArr[qb].isSparse != 0u) {
            if (lane < p.heads) sgStat[sg][lane] = float2(myMax0[qb], mySum0[qb]);
            if ((lane + 32u) < p.heads) sgStat[sg][lane + 32u] = float2(myMax1[qb], mySum1[qb]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (qb < uCnt && gArr[qb].isSparse != 0u && tid < p.heads) {
            float tileM = -1e30f;
            for (uint s = 0u; s < 8u; ++s) {
                float2 st = sgStat[s][tid];
                if (st.y > 0.0f) tileM = max(tileM, st.x);
            }
            float tileS = 0.0f;
            if (tileM > -1e20f) {
                for (uint s = 0u; s < 8u; ++s) {
                    float2 st = sgStat[s][tid];
                    if (st.y > 0.0f) tileS += st.y * metal::precise::exp(st.x - tileM);
                }
            }
            mhTGStats[(ulong(uBase + qb) * p.heads + tid) * p.score_tiles_x + tg_x] = float2(tileM, tileS);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void mk_lsa_mh_score(device const float *arena [[buffer(0)]], device const uchar *keyPool [[buffer(1)]],
                            device uchar *lsa_scratch [[buffer(2)]], constant MKLSA &p [[buffer(8)]],
                            uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    uint tg_x = group % p.score_tiles_x;
    uint tg_y = group / p.score_tiles_x;
    uint uBase = tg_y * p.bq_tile;
    if (uBase >= p.units) return;
    uint uCnt = min(p.bq_tile, p.units - uBase);

    device const uint *fineCandIdx = (device const uint *)(lsa_scratch + p.off_fine_cand);
    device const uint *fineCandCount = (device const uint *)(lsa_scratch + p.off_fine_count);
    device half *mhHeadScores = (device half *)(lsa_scratch + p.off_head_scores);
    device float2 *mhTGStats = (device float2 *)(lsa_scratch + p.off_tg_stats);

    threadgroup float2 sgStat[8][64];
    threadgroup MKLSAGeom gArr[4];
    threadgroup uint shFineCnt;

    if (tid < uCnt) gArr[tid] = mk_lsa_geom(p, uBase + tid);
    if (p.stage_mode == 2u && tid == 0u) shFineCnt = fineCandCount[uBase];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint itemLim[4] = { 0u, 0u, 0u, 0u };
    uint maxItemLim = 0u;
    uint fCnt = (p.stage_mode == 2u) ? shFineCnt : 0u;
    for (uint qb = 0u; qb < uCnt; ++qb) {
        if (gArr[qb].isSparse == 0u) continue;
        itemLim[qb] = (p.stage_mode == 0u) ? gArr[qb].totalBlocks
                    : ((p.stage_mode == 1u) ? gArr[qb].totalCoarse : (fCnt + 1u + p.window_blocks));
        maxItemLim = max(maxItemLim, itemLim[qb]);
    }
    if (tg_x * 256u >= maxItemLim) return;

    if (uCnt == 1u) {
        mk_lsa_mh_score_body<1>(arena, keyPool, fineCandIdx, mhHeadScores, mhTGStats, p, gArr,
                                uBase, uCnt, fCnt, itemLim, maxItemLim, sgStat, tg_x, tid);
    } else if (uCnt == 2u) {
        mk_lsa_mh_score_body<2>(arena, keyPool, fineCandIdx, mhHeadScores, mhTGStats, p, gArr,
                                uBase, uCnt, fCnt, itemLim, maxItemLim, sgStat, tg_x, tid);
    } else if (uCnt == 3u) {
        mk_lsa_mh_score_body<3>(arena, keyPool, fineCandIdx, mhHeadScores, mhTGStats, p, gArr,
                                uBase, uCnt, fCnt, itemLim, maxItemLim, sgStat, tg_x, tid);
    } else {
        mk_lsa_mh_score_body<4>(arena, keyPool, fineCandIdx, mhHeadScores, mhTGStats, p, gArr,
                                uBase, uCnt, fCnt, itemLim, maxItemLim, sgStat, tg_x, tid);
    }
}

// Single-row decode specialization for dim == 128, heads <= 32, and dynamic tile_items.
// Caches Q across heads in threadgroup memory and spreads candidate tiles across the GPU.
kernel void mk_lsa_mh_score_dec(device const float *arena [[buffer(0)]], device const uchar *keyPool [[buffer(1)]],
                                device uchar *lsa_scratch [[buffer(2)]], constant MKLSA &p [[buffer(8)]],
                                uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    MKLSAGeom g0 = mk_lsa_geom(p, 0u);
    if (g0.isSparse == 0u) return;

    device const uint *fineCandIdx = (device const uint *)(lsa_scratch + p.off_fine_cand);
    device const uint *fineCandCount = (device const uint *)(lsa_scratch + p.off_fine_count);
    device half *mhHeadScores = (device half *)(lsa_scratch + p.off_head_scores);
    device float2 *mhTGStats = (device float2 *)(lsa_scratch + p.off_tg_stats);

    uint tileItems = (p.tile_items != 0u) ? p.tile_items : 256u;
    uint fCnt = (p.stage_mode == 2u) ? fineCandCount[0] : 0u;
    uint itemLim = (p.stage_mode == 0u) ? g0.totalBlocks
                 : ((p.stage_mode == 1u) ? g0.totalCoarse : (fCnt + 1u + p.window_blocks));
    if (group * tileItems >= itemLim) return;

    threadgroup float4 shQ[32][32];
    threadgroup float shQSum[32];
    threadgroup float2 sgStat[8][32];
    uint sg = tid >> 5u, lane = tid & 31u;

    device const float *qRowBase = arena + p.q_off + ulong(p.row0 + g0.qRow) * p.q_stride;
    for (uint h = sg; h < p.heads; h += 8u) {
        device const packed_float4 *qPtr = (device const packed_float4 *)(qRowBase + h * 128u);
        float4 qv = float4(qPtr[lane]);
        shQ[h][lane] = qv;
        float s = simd_sum((qv.x + qv.y) + (qv.z + qv.w));
        if (lane == 0u) shQSum[h] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float myQSum = (lane < p.heads) ? shQSum[lane] : 0.0f;
    uint gPerKV = p.heads / p.kv_heads;
    uint sPairs = 64u;
    uint wLim = tileItems >> 3u;

    float myMax = -1e30f;
    float mySum = 0.0f;
    half myScore = half(0.0h);

    for (uint w = 0u; w < wLim; ++w) {
        uint item = group * tileItems + (w << 3u) + sg;
        if (item < itemLim) {
            uint blk = item;
            if (p.stage_mode == 2u) {
                if (item < fCnt) blk = fineCandIdx[item];
                else if (item == fCnt) blk = 0u;
                else blk = g0.middleEnd + (item - fCnt - 1u);
            }
            device const uchar *blkPtr = keyPool + ulong(blk) * p.pooled_row_bytes;
            device const half2 *metaPtr = (device const half2 *)blkPtr;
            device const uchar *payPtr = blkPtr + p.pooled_meta_bytes;
            for (uint g = 0u; g < p.kv_heads; ++g) {
                half2 hm = metaPtr[g];
                float sc = float(hm.x), mn = float(hm.y);
                device const ushort *pay2 = (device const ushort *)(payPtr + g * sPairs);
                ushort pk0 = pay2[lane];
                float4 n0 = float4(float(pk0 & 0xFu), float((pk0 >> 4u) & 0xFu),
                                   float((pk0 >> 8u) & 0xFu), float((pk0 >> 12u) & 0xFu));
                for (uint sub = 0u; sub < gPerKV; ++sub) {
                    uint qh = g * gPerKV + sub;
                    float dotN = simd_sum(dot(shQ[qh][lane], n0));
                    if (lane == qh) {
                        myScore = half(clamp((sc * dotN + mn * myQSum) * p.scale, -60000.0f, 60000.0f));
                    }
                }
            }
            if (lane < p.heads) {
                float sf = float(myScore);
                if (mySum == 0.0f) {
                    myMax = sf;
                    mySum = 1.0f;
                } else {
                    float newM = max(myMax, sf);
                    mySum = mySum * metal::precise::exp(myMax - newM) + metal::precise::exp(sf - newM);
                    myMax = newM;
                }
            }
            uint writeIdx = 0u;
            bool doWrite = false;
            if (p.stage_mode == 0u) {
                if (item >= 1u && item < g0.middleEnd) {
                    writeIdx = item - 1u;
                    doWrite = true;
                }
            } else if (p.stage_mode == 1u) {
                if (item < g0.numCoarseMid) {
                    writeIdx = item;
                    doWrite = true;
                }
            } else {
                if (item < fCnt) {
                    writeIdx = item;
                    doWrite = true;
                }
            }
            if (doWrite && lane < p.heads) {
                mhHeadScores[ulong(writeIdx) * p.heads + lane] = myScore;
            }
        }
    }

    if (lane < p.heads) sgStat[sg][lane] = float2(myMax, mySum);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < p.heads) {
        float tileM = -1e30f;
        for (uint s = 0u; s < 8u; ++s) {
            float2 st = sgStat[s][tid];
            if (st.y > 0.0f) tileM = max(tileM, st.x);
        }
        float tileS = 0.0f;
        if (tileM > -1e20f) {
            for (uint s = 0u; s < 8u; ++s) {
                float2 st = sgStat[s][tid];
                if (st.y > 0.0f) tileS += st.y * metal::precise::exp(st.x - tileM);
            }
        }
        mhTGStats[ulong(tid) * p.score_tiles_x + group] = float2(tileM, tileS);
    }
}

// Reduces per-tile (max, sum_exp) statistics and optional learned sink into per-head LSE.
// One SIMD group (32 lanes) per (unit, head).
kernel void mk_lsa_mh_reduce_lse(device const float *arena [[buffer(0)]], device uchar *lsa_scratch [[buffer(1)]],
                                 constant MKLSA &p [[buffer(8)]], uint group [[threadgroup_position_in_grid]],
                                 uint lane [[thread_index_in_simdgroup]]) {
    uint unit = group / p.heads;
    uint qh = group % p.heads;
    if (unit >= p.units) return;
    MKLSAGeom geom = mk_lsa_geom(p, unit);
    if (geom.isSparse == 0u) return;

    device const uint *fineCandCount = (device const uint *)(lsa_scratch + p.off_fine_count);
    device const float2 *mhTGStats = (device const float2 *)(lsa_scratch + p.off_tg_stats);
    device float *mhHeadLSE = (device float *)(lsa_scratch + p.off_head_lse);

    uint itemLim = (p.stage_mode == 0u) ? geom.totalBlocks
                 : ((p.stage_mode == 1u) ? geom.totalCoarse : (fineCandCount[unit] + 1u + p.window_blocks));
    uint tileItems = (p.tile_items != 0u) ? p.tile_items : 256u;
    uint nTiles = (itemLim + tileItems - 1u) / tileItems;
    ulong tileBase = (ulong(unit) * p.heads + qh) * p.score_tiles_x;

    float localM = -1e30f, localS = 0.0f;
    for (uint tx = lane; tx < nTiles; tx += 32u) {
        float2 st = mhTGStats[tileBase + tx];
        if (st.y > 0.0f) {
            float newM = max(localM, st.x);
            localS = (localS > 0.0f ? localS * metal::precise::exp(localM - newM) : 0.0f) +
                     st.y * metal::precise::exp(st.x - newM);
            localM = newM;
        }
    }
    float maxVal = simd_max(localM);
    float sinkVal = (p.has_sink != 0u) ? arena[p.sink_off + ulong(p.row0 + geom.qRow) * p.sink_stride + qh] : -INFINITY;
    if (p.has_sink != 0u) maxVal = max(maxVal, sinkVal);

    float term = (localS > 0.0f && maxVal > -1e20f) ? (localS * metal::precise::exp(localM - maxVal)) : 0.0f;
    float totalS = simd_sum(term);
    if (p.has_sink != 0u && maxVal > -1e20f) totalS += metal::precise::exp(sinkVal - maxVal);

    if (lane == 0u) {
        mhHeadLSE[ulong(unit) * p.heads + qh] = (maxVal > -1e20f && totalS > 0.0f) ? (maxVal + log(totalS)) : 0.0f;
    }
}

// Aggregates normalized attention probability mass across heads and fills the MSB radix histogram.
kernel void mk_lsa_mh_agg_hist(device uchar *lsa_scratch [[buffer(0)]], constant MKLSA &p [[buffer(8)]],
                               uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    uint tg_x = group % p.score_tiles_x;
    uint unit = group / p.score_tiles_x;
    if (unit >= p.units) return;
    MKLSAGeom geom = mk_lsa_geom(p, unit);
    if (geom.isSparse == 0u) return;

    device const half *mhHeadScores = (device const half *)(lsa_scratch + p.off_head_scores);
    device const float *mhHeadLSE = (device const float *)(lsa_scratch + p.off_head_lse);
    device const uint *fineCandCount = (device const uint *)(lsa_scratch + p.off_fine_count);
    device uint *scoreScratch = (device uint *)(lsa_scratch + p.off_scores);
    device atomic_uint *uHist = (device atomic_uint *)(lsa_scratch + p.off_hist);

    uint tileItems = (p.tile_items != 0u) ? p.tile_items : 256u;
    uint wLim = tileItems >> 3u;
    uint nItems = (p.stage_mode == 0u) ? geom.numMiddle
                : ((p.stage_mode == 1u) ? geom.numCoarseMid : fineCandCount[unit]);
    if (tg_x * tileItems >= nItems) return;

    threadgroup uint localHist[256];
    localHist[tid] = 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint sg = tid >> 5u, lane = tid & 31u;
    ulong lseBase = ulong(unit) * p.heads;
    float myLSE0 = (lane < p.heads) ? mhHeadLSE[lseBase + lane] : 0.0f;
    float myLSE1 = ((lane + 32u) < p.heads) ? mhHeadLSE[lseBase + lane + 32u] : 0.0f;
    ulong scoreUnitBase = ulong(unit) * p.head_score_stride;

    for (uint w = 0u; w < wLim; ++w) {
        uint idx = tg_x * tileItems + (w << 3u) + sg;
        if (idx < nItems) {
            ulong rowOff = (scoreUnitBase + idx) * p.heads;
            float pVal = (lane < p.heads) ? metal::precise::exp(float(mhHeadScores[rowOff + lane]) - myLSE0) : 0.0f;
            if ((lane + 32u) < p.heads) pVal += metal::precise::exp(float(mhHeadScores[rowOff + lane + 32u]) - myLSE1);
            float probSum = simd_sum(pVal);
            if (lane == 0u) {
                float probClamped = isfinite(probSum) ? clamp(probSum, 0.0f, float(p.heads)) : 0.0f;
                uint bits = as_type<uint>(probClamped);
                uint u16 = (bits <= 0x33800000u) ? 0u : min(65535u, (bits - 0x33800000u) >> 12u);
                scoreScratch[ulong(unit) * p.score_stride + idx] = u16;
                atomic_fetch_add_explicit((threadgroup atomic_uint *)&localHist[(u16 >> 8u) & 0xFFu], 1u, memory_order_relaxed);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint hc = localHist[tid];
    if (hc != 0u) atomic_fetch_add_explicit(&uHist[unit * 512u + tid], hc, memory_order_relaxed);
}

// Filters items in the top MSB bins into candidate scratch and populates the LSB histogram.
kernel void mk_lsa_mh_collect(device uchar *lsa_scratch [[buffer(0)]], constant MKLSA &p [[buffer(8)]],
                              uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    uint tg_x = group % p.score_tiles_x;
    uint unit = group / p.score_tiles_x;
    if (unit >= p.units) return;
    MKLSAGeom g = mk_lsa_geom(p, unit);
    if (g.isSparse == 0u) return;

    device const uint *scoreScratch = (device const uint *)(lsa_scratch + p.off_scores);
    device const uint *fineCandIdx = (device const uint *)(lsa_scratch + p.off_fine_cand);
    device const uint *fineCandCount = (device const uint *)(lsa_scratch + p.off_fine_count);
    device atomic_uint *uHist = (device atomic_uint *)(lsa_scratch + p.off_hist);
    device atomic_uint *candCount = (device atomic_uint *)(lsa_scratch + p.off_cand_count);
    device uint *candM = (device uint *)(lsa_scratch + p.off_cand_m);
    device uint *candU = (device uint *)(lsa_scratch + p.off_cand_u);

    uint nItems = (p.stage_mode == 0u) ? g.numMiddle
                : ((p.stage_mode == 1u) ? g.numCoarseMid : fineCandCount[unit]);
    if (tg_x * 256u >= nItems) return;

    uint kTarget = (p.stage_mode == 1u) ? g.coarseKEff : g.kEff;
    threadgroup uint hist[256];
    threadgroup uint scanSum[8];
    threadgroup uint sgTotal, chosenBin, greaterAccum;
    uint sg = tid >> 5u, lane = tid & 31u;
    hist[tid] = atomic_load_explicit(&uHist[unit * 512u + tid], memory_order_relaxed);
    if (tid == 0u) {
        greaterAccum = 0u;
        chosenBin = 0u;
        sgTotal = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mk_lsa_pick_radix_bin(hist, scanSum, greaterAccum, chosenBin, sgTotal, kTarget, false, tid, sg, lane);
    uint topBin = chosenBin;

    hist[tid] = 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint idx = tg_x * 256u + tid;
    if (idx < nItems) {
        uint u16 = scoreScratch[ulong(unit) * p.score_stride + idx];
        uint bin = (u16 >> 8u) & 0xFFu;
        if (bin >= topBin) {
            uint mId = (p.stage_mode == 0u) ? (idx + 1u)
                     : ((p.stage_mode == 1u) ? idx : fineCandIdx[ulong(unit) * (p.coarse_cap * 4u) + idx]);
            uint slot = atomic_fetch_add_explicit(&candCount[unit], 1u, memory_order_relaxed);
            if (slot < p.cand_cap) {
                candM[ulong(unit) * p.cand_cap + slot] = mId;
                candU[ulong(unit) * p.cand_cap + slot] = u16;
            }
            if (bin == topBin) {
                atomic_fetch_add_explicit((threadgroup atomic_uint *)&hist[u16 & 0xFFu], 1u, memory_order_relaxed);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (hist[tid] != 0u) {
        atomic_fetch_add_explicit(&uHist[unit * 512u + 256u + tid], hist[tid], memory_order_relaxed);
    }
}

// Finalizes Stage 1 top-coarse blocks and expands them into sorted fine candidate block IDs.
kernel void mk_lsa_mh_coarse_expand(device uchar *lsa_scratch [[buffer(0)]], constant MKLSA &p [[buffer(8)]],
                                    uint unit [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    if (unit >= p.units) return;
    constexpr uint SORT_N = 4096u;
    MKLSAGeom g = mk_lsa_geom(p, unit);

    device const uint *scoreScratch = (device const uint *)(lsa_scratch + p.off_scores);
    device uint *fineCandIdx = (device uint *)(lsa_scratch + p.off_fine_cand);
    device uint *fineCandCount = (device uint *)(lsa_scratch + p.off_fine_count);
    device atomic_uint *uHist = (device atomic_uint *)(lsa_scratch + p.off_hist);
    device atomic_uint *candCount = (device atomic_uint *)(lsa_scratch + p.off_cand_count);
    device uint *candM = (device uint *)(lsa_scratch + p.off_cand_m);
    device const uint *candU = (device const uint *)(lsa_scratch + p.off_cand_u);

    if (g.isSparse == 0u || g.coarseKEff == 0u) {
        if (tid == 0u) fineCandCount[unit] = 0u;
        atomic_store_explicit(&uHist[unit * 512u + tid], 0u, memory_order_relaxed);
        atomic_store_explicit(&uHist[unit * 512u + 256u + tid], 0u, memory_order_relaxed);
        if (tid == 0u) atomic_store_explicit(&candCount[unit], 0u, memory_order_relaxed);
        return;
    }

    threadgroup uint hist[256];
    threadgroup uint scanSum[8];
    threadgroup uint sgTotal, chosenBin, greaterAccum;
    threadgroup uint keep[SORT_N];
    threadgroup uint keepCount;
    uint sg = tid >> 5u, lane = tid & 31u;

    hist[tid] = atomic_load_explicit(&uHist[unit * 512u + tid], memory_order_relaxed);
    if (tid == 0u) {
        greaterAccum = 0u;
        chosenBin = 0u;
        sgTotal = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mk_lsa_pick_radix_bin(hist, scanSum, greaterAccum, chosenBin, sgTotal, g.coarseKEff, false, tid, sg, lane);
    uint topBin = chosenBin;

    hist[tid] = atomic_load_explicit(&uHist[unit * 512u + 256u + tid], memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mk_lsa_pick_radix_bin(hist, scanSum, greaterAccum, chosenBin, sgTotal, g.coarseKEff, true, tid, sg, lane);
    uint threshU = (topBin << 8u) | chosenBin;
    uint rawCount = atomic_load_explicit(&candCount[unit], memory_order_relaxed);
    uint fineStride = p.coarse_cap * 4u;
    device uint *fineOut = fineCandIdx + ulong(unit) * fineStride;
    bool fast = (rawCount <= p.cand_cap) && (sgTotal <= SORT_N);

    if (fast) {
        uint total = sgTotal;
        uint greater = greaterAccum;
        uint needTies = g.coarseKEff - greater;
        device uint *cm = candM + ulong(unit) * p.cand_cap;
        device const uint *cu = candU + ulong(unit) * p.cand_cap;
        mk_lsa_fast_merge(keep, keepCount, cm, cu, rawCount, total, greater, needTies, threshU, cm, 0u, tid);
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

        if (tid == 0u) chosenBin = 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint cChunks = (g.coarseKEff + 255u) >> 8u;
        for (uint ch = 0u; ch < cChunks; ++ch) {
            uint i = (ch << 8u) + tid;
            uint fb0 = 0u, cnt = 0u;
            if (i < g.coarseKEff) {
                uint c = cm[i];
                fb0 = max(1u, c << 2u);
                uint fb1 = min(g.middleEnd, (c << 2u) + 4u);
                cnt = (fb1 > fb0) ? (fb1 - fb0) : 0u;
            }
            uint wRank = simd_prefix_exclusive_sum(cnt);
            uint wSum = simd_sum(cnt);
            if (lane == 0u) scanSum[sg] = wSum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint wBase = chosenBin;
            for (uint s = 0u; s < sg; ++s) wBase += scanSum[s];
            for (uint k = 0u; k < cnt; ++k) {
                uint dst = wBase + wRank + k;
                if (dst < fineStride) fineOut[dst] = fb0 + k;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0u) {
                uint chunkFine = 0u;
                for (uint s = 0u; s < 8u; ++s) chunkFine += scanSum[s];
                chosenBin += chunkFine;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0u) fineCandCount[unit] = min(chosenBin, fineStride);
    } else {
        uint skipTies = sgTotal - g.coarseKEff;
        if (tid == 0u) {
            chosenBin = 0u;
            greaterAccum = 0u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint numChunks = (g.numCoarseMid + 255u) >> 8u;
        for (uint ch = 0u; ch < numChunks; ++ch) {
            uint idx = (ch << 8u) + tid;
            bool valid = idx < g.numCoarseMid;
            uint u = valid ? scoreScratch[ulong(unit) * p.score_stride + idx] : 0u;
            bool isGt = valid && (u > threshU);
            bool isTie = valid && (u == threshU);
            uint tieRank = simd_prefix_exclusive_sum(isTie ? 1u : 0u);
            uint tieSum = simd_sum(isTie ? 1u : 0u);
            if (lane == 0u) scanSum[sg] = tieSum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint tieBase = greaterAccum;
            for (uint s = 0u; s < sg; ++s) tieBase += scanSum[s];
            bool keepCoarse = isGt || (isTie && ((tieBase + tieRank) >= skipTies));
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0u) {
                uint chunkTies = 0u;
                for (uint s = 0u; s < 8u; ++s) chunkTies += scanSum[s];
                greaterAccum += chunkTies;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint fb0 = 0u, cnt = 0u;
            if (keepCoarse) {
                fb0 = max(1u, idx << 2u);
                uint fb1 = min(g.middleEnd, (idx << 2u) + 4u);
                cnt = (fb1 > fb0) ? (fb1 - fb0) : 0u;
            }
            uint wRank = simd_prefix_exclusive_sum(cnt);
            uint wSum = simd_sum(cnt);
            if (lane == 0u) scanSum[sg] = wSum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint wBase = chosenBin;
            for (uint s = 0u; s < sg; ++s) wBase += scanSum[s];
            for (uint k = 0u; k < cnt; ++k) {
                uint dst = wBase + wRank + k;
                if (dst < fineStride) fineOut[dst] = fb0 + k;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0u) {
                uint chunkFine = 0u;
                for (uint s = 0u; s < 8u; ++s) chunkFine += scanSum[s];
                chosenBin += chunkFine;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0u) fineCandCount[unit] = min(chosenBin, fineStride);
    }

    atomic_store_explicit(&uHist[unit * 512u + tid], 0u, memory_order_relaxed);
    atomic_store_explicit(&uHist[unit * 512u + 256u + tid], 0u, memory_order_relaxed);
    if (tid == 0u) atomic_store_explicit(&candCount[unit], 0u, memory_order_relaxed);
}

// Finalizes sorted [0, middle top-K, trailing window] fine block IDs into blockList/blockCount
// and clears histogram + candidate counters for reuse.
kernel void mk_lsa_mh_finalize(device uchar *lsa_scratch [[buffer(0)]], device uint *blockList [[buffer(1)]],
                               device uint *blockCount [[buffer(2)]], constant MKLSA &p [[buffer(8)]],
                               uint unit [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    if (unit >= p.units) return;
    constexpr uint SORT_N = 4096u;
    MKLSAGeom g = mk_lsa_geom(p, unit);

    device const uint *scoreScratch = (device const uint *)(lsa_scratch + p.off_scores);
    device const uint *fineCandIdx = (device const uint *)(lsa_scratch + p.off_fine_cand);
    device const uint *fineCandCount = (device const uint *)(lsa_scratch + p.off_fine_count);
    device atomic_uint *uHist = (device atomic_uint *)(lsa_scratch + p.off_hist);
    device atomic_uint *candCount = (device atomic_uint *)(lsa_scratch + p.off_cand_count);
    device const uint *candM = (device const uint *)(lsa_scratch + p.off_cand_m);
    device const uint *candU = (device const uint *)(lsa_scratch + p.off_cand_u);

    uint global_unit = p.unit_offset + unit;
    if (g.isSparse == 0u) {
        if (tid == 0u) blockCount[global_unit] = 0u;
        atomic_store_explicit(&uHist[unit * 512u + tid], 0u, memory_order_relaxed);
        atomic_store_explicit(&uHist[unit * 512u + 256u + tid], 0u, memory_order_relaxed);
        if (tid == 0u) atomic_store_explicit(&candCount[unit], 0u, memory_order_relaxed);
        return;
    }

    device uint *outList = blockList + ulong(global_unit) * p.list_stride;
    if (tid == 0u) outList[0] = 0u;

    uint nItems = (p.stage_mode == 0u) ? g.numMiddle : fineCandCount[unit];
    uint kEff = min(g.kEff, nItems);
    if (kEff > 0u) {
        threadgroup uint hist[256];
        threadgroup uint scanSum[8];
        threadgroup uint sgTotal, chosenBin, greaterAccum;
        threadgroup uint keep[SORT_N];
        threadgroup uint keepCount;
        uint sg = tid >> 5u, lane = tid & 31u;

        hist[tid] = atomic_load_explicit(&uHist[unit * 512u + tid], memory_order_relaxed);
        if (tid == 0u) {
            greaterAccum = 0u;
            chosenBin = 0u;
            sgTotal = 0u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mk_lsa_pick_radix_bin(hist, scanSum, greaterAccum, chosenBin, sgTotal, kEff, false, tid, sg, lane);
        uint topBin = chosenBin;

        hist[tid] = atomic_load_explicit(&uHist[unit * 512u + 256u + tid], memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mk_lsa_pick_radix_bin(hist, scanSum, greaterAccum, chosenBin, sgTotal, kEff, true, tid, sg, lane);
        uint threshU = (topBin << 8u) | chosenBin;
        uint rawCount = atomic_load_explicit(&candCount[unit], memory_order_relaxed);
        bool fast = (rawCount <= p.cand_cap) && (sgTotal <= SORT_N);

        if (fast) {
            uint total = sgTotal;
            uint greater = greaterAccum;
            uint needTies = kEff - greater;
            device const uint *cm = candM + ulong(unit) * p.cand_cap;
            device const uint *cu = candU + ulong(unit) * p.cand_cap;
            mk_lsa_fast_merge(keep, keepCount, cm, cu, rawCount, total, greater, needTies, threshU, outList, 1u, tid);
        } else {
            uint skipTies = sgTotal - kEff;
            if (tid == 0u) {
                chosenBin = 0u;
                greaterAccum = 0u;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint numChunks = (nItems + 255u) >> 8u;
            uint fineStride = p.coarse_cap * 4u;
            for (uint ch = 0u; ch < numChunks; ++ch) {
                uint idx = (ch << 8u) + tid;
                bool valid = idx < nItems;
                uint u = valid ? scoreScratch[ulong(unit) * p.score_stride + idx] : 0u;
                bool isGt = valid && (u > threshU);
                bool isTie = valid && (u == threshU);
                uint tieRank = simd_prefix_exclusive_sum(isTie ? 1u : 0u);
                uint tieSum = simd_sum(isTie ? 1u : 0u);
                if (lane == 0u) scanSum[sg] = tieSum;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                uint tieBase = greaterAccum;
                for (uint s = 0u; s < sg; ++s) tieBase += scanSum[s];
                bool take = isGt || (isTie && ((tieBase + tieRank) >= skipTies));
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid == 0u) {
                    uint chunkTies = 0u;
                    for (uint s = 0u; s < 8u; ++s) chunkTies += scanSum[s];
                    greaterAccum += chunkTies;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                uint wRank = simd_prefix_exclusive_sum(take ? 1u : 0u);
                uint wSum = simd_sum(take ? 1u : 0u);
                if (lane == 0u) scanSum[sg] = wSum;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                uint wBase = chosenBin;
                for (uint s = 0u; s < sg; ++s) wBase += scanSum[s];
                if (take) {
                    uint dst = wBase + wRank;
                    if (dst < kEff) {
                        uint blk = (p.stage_mode == 0u) ? (idx + 1u) : fineCandIdx[ulong(unit) * fineStride + idx];
                        outList[1u + dst] = blk;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid == 0u) {
                    uint chunkOut = 0u;
                    for (uint s = 0u; s < 8u; ++s) chunkOut += scanSum[s];
                    chosenBin += chunkOut;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }

    for (uint w = tid; w < p.window_blocks; w += 256u) {
        outList[1u + kEff + w] = g.middleEnd + w;
    }
    if (tid == 0u) blockCount[global_unit] = 1u + kEff + p.window_blocks;

    atomic_store_explicit(&uHist[unit * 512u + tid], 0u, memory_order_relaxed);
    atomic_store_explicit(&uHist[unit * 512u + 256u + tid], 0u, memory_order_relaxed);
    if (tid == 0u) atomic_store_explicit(&candCount[unit], 0u, memory_order_relaxed);
}

// Multi-threadgroup decode LSE reduction + probability aggregation + MSB radix histogram.
// Replaces mk_lsa_mh_reduce_lse + mk_lsa_mh_agg_hist for decode (units == 1, heads <= 32).
// Expects score tile config in p.tile_items / p.score_tiles_x and agg tile size in p.bq_tile.
kernel void mk_lsa_mh_agg_hist_dec(device const float *arena [[buffer(0)]], device uchar *lsa_scratch [[buffer(1)]],
                                   constant MKLSA &p [[buffer(8)]], uint group [[threadgroup_position_in_grid]],
                                   uint tid [[thread_index_in_threadgroup]]) {
    MKLSAGeom geom = mk_lsa_geom(p, 0u);
    if (geom.isSparse == 0u) return;

    device const uint *fineCandCount = (device const uint *)(lsa_scratch + p.off_fine_count);
    device const half *mhHeadScores = (device const half *)(lsa_scratch + p.off_head_scores);
    device const float2 *mhTGStats = (device const float2 *)(lsa_scratch + p.off_tg_stats);
    device uint *scoreScratch = (device uint *)(lsa_scratch + p.off_scores);
    device atomic_uint *uHist = (device atomic_uint *)(lsa_scratch + p.off_hist);

    uint fCnt = (p.stage_mode == 2u) ? fineCandCount[0] : 0u;
    uint nItems = (p.stage_mode == 0u) ? geom.numMiddle
                : ((p.stage_mode == 1u) ? geom.numCoarseMid : fCnt);
    uint aggTile = (p.bq_tile >= 8u) ? p.bq_tile : 32u;
    if (group * aggTile >= nItems) return;

    threadgroup float shLSE[32];
    threadgroup uint localHist[256];
    uint sg = tid >> 5u, lane = tid & 31u;

    uint scoreTile = (p.tile_items != 0u) ? p.tile_items : 256u;
    uint itemLim = (p.stage_mode == 0u) ? geom.totalBlocks
                 : ((p.stage_mode == 1u) ? geom.totalCoarse : (fCnt + 1u + p.window_blocks));
    uint nScoreTiles = (itemLim + scoreTile - 1u) / scoreTile;

    for (uint qh = sg; qh < p.heads; qh += 8u) {
        ulong tileBase = ulong(qh) * p.score_tiles_x;
        float localM = -1e30f, localS = 0.0f;
        for (uint tx = lane; tx < nScoreTiles; tx += 32u) {
            float2 st = mhTGStats[tileBase + tx];
            if (st.y > 0.0f) {
                float newM = max(localM, st.x);
                localS = (localS > 0.0f ? localS * metal::precise::exp(localM - newM) : 0.0f) +
                         st.y * metal::precise::exp(st.x - newM);
                localM = newM;
            }
        }
        float maxVal = simd_max(localM);
        float sinkVal = (p.has_sink != 0u) ? arena[p.sink_off + ulong(p.row0 + geom.qRow) * p.sink_stride + qh] : -INFINITY;
        if (p.has_sink != 0u) maxVal = max(maxVal, sinkVal);
        float term = (localS > 0.0f && maxVal > -1e20f) ? (localS * metal::precise::exp(localM - maxVal)) : 0.0f;
        float totalS = simd_sum(term);
        if (p.has_sink != 0u && maxVal > -1e20f) totalS += metal::precise::exp(sinkVal - maxVal);
        if (lane == 0u) {
            shLSE[qh] = (maxVal > -1e20f && totalS > 0.0f) ? (maxVal + log(totalS)) : 0.0f;
        }
    }
    localHist[tid] = 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float myLSE = (lane < p.heads) ? shLSE[lane] : 0.0f;
    uint wLim = aggTile >> 3u;
    for (uint w = 0u; w < wLim; ++w) {
        uint idx = group * aggTile + (w << 3u) + sg;
        if (idx < nItems) {
            ulong rowOff = ulong(idx) * p.heads;
            float pVal = (lane < p.heads) ? metal::precise::exp(float(mhHeadScores[rowOff + lane]) - myLSE) : 0.0f;
            float probSum = simd_sum(pVal);
            if (lane == 0u) {
                float probClamped = isfinite(probSum) ? clamp(probSum, 0.0f, float(p.heads)) : 0.0f;
                uint bits = as_type<uint>(probClamped);
                uint u16 = (bits <= 0x33800000u) ? 0u : min(65535u, (bits - 0x33800000u) >> 12u);
                scoreScratch[idx] = u16;
                atomic_fetch_add_explicit((threadgroup atomic_uint *)&localHist[(u16 >> 8u) & 0xFFu], 1u, memory_order_relaxed);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint hc = localHist[tid];
    if (hc != 0u) atomic_fetch_add_explicit(&uHist[tid], hc, memory_order_relaxed);
}

// Single-threadgroup radix finalize/coarse-expand for decode (units == 1).
// Clears uHist[0..255] upfront and streams selected IDs in index order without sorting.
kernel void mk_lsa_mh_finalize_dec(device uchar *lsa_scratch [[buffer(0)]], device uint *blockList [[buffer(1)]],
                                   device uint *blockCount [[buffer(2)]], constant MKLSA &p [[buffer(8)]],
                                   uint tid [[thread_index_in_threadgroup]]) {
    device atomic_uint *uHist = (device atomic_uint *)(lsa_scratch + p.off_hist);
    uint hVal = atomic_load_explicit(&uHist[tid], memory_order_relaxed);
    atomic_store_explicit(&uHist[tid], 0u, memory_order_relaxed);

    MKLSAGeom g = mk_lsa_geom(p, 0u);
    device const uint *scoreScratch = (device const uint *)(lsa_scratch + p.off_scores);
    device uint *fineCandIdx = (device uint *)(lsa_scratch + p.off_fine_cand);
    device uint *fineCandCount = (device uint *)(lsa_scratch + p.off_fine_count);

    if (g.isSparse == 0u) {
        if (tid == 0u) {
            if (p.stage_mode == 1u) fineCandCount[0] = 0u;
            else blockCount[p.unit_offset] = 0u;
        }
        return;
    }

    uint fCnt = (p.stage_mode == 2u) ? fineCandCount[0] : 0u;
    uint nItems = (p.stage_mode == 0u) ? g.numMiddle
                : ((p.stage_mode == 1u) ? g.numCoarseMid : fCnt);
    uint kTarget = (p.stage_mode == 1u) ? g.coarseKEff : min(g.kEff, nItems);
    if (p.stage_mode == 1u && kTarget == 0u) {
        if (tid == 0u) fineCandCount[0] = 0u;
        return;
    }

    device uint *outList = blockList + ulong(p.unit_offset) * p.list_stride;
    if (p.stage_mode != 1u && tid == 0u) outList[0] = 0u;

    if (kTarget > 0u) {
        threadgroup uint hist[256];
        threadgroup uint scanTie[8];
        threadgroup uint scanOut[8];
        threadgroup uint sgTotal, chosenBin, greaterAccum;
        uint sg = tid >> 5u, lane = tid & 31u;

        hist[tid] = hVal;
        if (tid == 0u) {
            greaterAccum = 0u;
            chosenBin = 0u;
            sgTotal = 0u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mk_lsa_pick_radix_bin(hist, scanTie, greaterAccum, chosenBin, sgTotal, kTarget, false, tid, sg, lane);
        uint topBin = chosenBin;

        hist[tid] = 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint uReg[4] = {0u, 0u, 0u, 0u};
        uint numChunks = (nItems + 255u) >> 8u;
        for (uint ch = 0u; ch < numChunks; ++ch) {
            uint idx = (ch << 8u) + tid;
            uint u16 = (idx < nItems) ? scoreScratch[idx] : 0u;
            if (ch < 4u) uReg[ch] = u16;
            if (idx < nItems && ((u16 >> 8u) & 0xFFu) == topBin) {
                atomic_fetch_add_explicit((threadgroup atomic_uint *)&hist[u16 & 0xFFu], 1u, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mk_lsa_pick_radix_bin(hist, scanTie, greaterAccum, chosenBin, sgTotal, kTarget, true, tid, sg, lane);
        uint threshU = (topBin << 8u) | chosenBin;
        uint skipTies = sgTotal - kTarget;

        uint tieAccum = 0u;
        uint outAccum = 0u;
        uint fineStride = p.coarse_cap * 4u;
        for (uint ch = 0u; ch < numChunks; ++ch) {
            uint idx = (ch << 8u) + tid;
            bool valid = idx < nItems;
            uint u = valid ? ((ch < 4u) ? uReg[ch] : scoreScratch[idx]) : 0u;
            bool isGt = valid && (u > threshU);
            bool isTie = valid && (u == threshU);
            uint tieRank = simd_prefix_exclusive_sum(isTie ? 1u : 0u);
            uint tieSum = simd_sum(isTie ? 1u : 0u);
            if (lane == 0u) scanTie[sg] = tieSum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint tieBase = tieAccum;
            uint chunkTies = 0u;
            for (uint s = 0u; s < sg; ++s) tieBase += scanTie[s];
            for (uint s = 0u; s < 8u; ++s) chunkTies += scanTie[s];
            tieAccum += chunkTies;
            bool take = isGt || (isTie && ((tieBase + tieRank) >= skipTies));

            if (p.stage_mode == 1u) {
                uint fb0 = 0u, cnt = 0u;
                if (take) {
                    fb0 = max(1u, idx << 2u);
                    uint fb1 = min(g.middleEnd, (idx << 2u) + 4u);
                    cnt = (fb1 > fb0) ? (fb1 - fb0) : 0u;
                }
                uint wRank = simd_prefix_exclusive_sum(cnt);
                uint wSum = simd_sum(cnt);
                if (lane == 0u) scanOut[sg] = wSum;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                uint wBase = outAccum;
                uint chunkFine = 0u;
                for (uint s = 0u; s < sg; ++s) wBase += scanOut[s];
                for (uint s = 0u; s < 8u; ++s) chunkFine += scanOut[s];
                outAccum += chunkFine;
                for (uint k = 0u; k < cnt; ++k) {
                    uint dst = wBase + wRank + k;
                    if (dst < fineStride) fineCandIdx[dst] = fb0 + k;
                }
            } else {
                uint wRank = simd_prefix_exclusive_sum(take ? 1u : 0u);
                uint wSum = simd_sum(take ? 1u : 0u);
                if (lane == 0u) scanOut[sg] = wSum;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                uint wBase = outAccum;
                uint chunkOut = 0u;
                for (uint s = 0u; s < sg; ++s) wBase += scanOut[s];
                for (uint s = 0u; s < 8u; ++s) chunkOut += scanOut[s];
                outAccum += chunkOut;
                if (take) {
                    uint dst = wBase + wRank;
                    if (dst < kTarget) {
                        uint blk = (p.stage_mode == 0u) ? (idx + 1u) : fineCandIdx[idx];
                        outList[1u + dst] = blk;
                    }
                }
            }
        }
        if (p.stage_mode == 1u) {
            if (tid == 0u) fineCandCount[0] = min(outAccum, fineStride);
            return;
        }
    }

    for (uint w = tid; w < p.window_blocks; w += 256u) {
        outList[1u + kTarget + w] = g.middleEnd + w;
    }
    if (tid == 0u) blockCount[p.unit_offset] = 1u + kTarget + p.window_blocks;
}

// ---- Sparse attention kernels -----------------------------------------------

// General FP32 sparse relative attention reference over blockList/blockCount.
// One SIMD group per (row, head), supports dim <= 256, any GQA ratio, row_step >= 1,
// and per-row dense bypass (blockCount[unit] == 0).
kernel void mk_relative_lsa_ref(device float *arena [[buffer(0)]], device const float *K [[buffer(1)]],
                                device const float *V [[buffer(2)]], device const uint *blockList [[buffer(3)]],
                                device const uint *blockCount [[buffer(4)]], constant MKLSA &p [[buffer(8)]],
                                uint lane [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]]) {
    uint r = group / p.heads, h = group % p.heads;
    if (r >= p.rows) return;
    uint row = p.row0 + r;
    uint position = p.pos0 + r;
    uint kvh = h / (p.heads / p.kv_heads);
    uint unit = (p.row_step > 1u) ? (r / p.row_step) : r;
    uint cnt = blockCount[unit];

    uint E = (p.dim + 31u) / 32u;
    float q[8], acc[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    device const float *qp = arena + p.q_off + row * p.q_stride + h * p.dim;
    for (uint e = 0u; e < E; ++e) {
        uint d = lane + 32u * e;
        q[e] = d < p.dim ? qp[d] : 0.0f;
    }
    float maximum = (p.has_sink != 0u) ? arena[p.sink_off + row * p.sink_stride + h] : -INFINITY;
    float denominator = (p.has_sink != 0u) ? 1.0f : 0.0f;

    uint tail0 = 0u;
    if (cnt > 0u) {
        uint ref_r = (p.row_step > 1u) ? min(unit * p.row_step + (p.row_step - 1u), p.rows - 1u) : unit;
        tail0 = (p.pos0 + ref_r + 1u) & ~3u;
    }
    uint tail_end = position + 1u;
    uint tail_cnt = (tail_end > tail0) ? (tail_end - tail0) : 0u;
    uint total_keys = cnt * 4u + tail_cnt;
    device const uint *rowBlocks = blockList + ulong(unit) * p.list_stride;
    uint stride = p.kv_heads * p.dim;

    for (uint idx = 0u; idx < total_keys; ++idx) {
        uint t;
        if (cnt > 0u && idx < cnt * 4u) {
            t = (rowBlocks[idx >> 2u] << 2u) | (idx & 3u);
            if (t > position) continue;
        } else {
            uint k_off = (cnt > 0u) ? (idx - cnt * 4u) : idx;
            t = tail0 + k_off;
        }
        uint slot = (t < p.capacity) ? t : (t % p.capacity);
        ulong base = ulong(slot) * stride + kvh * p.dim;
        float dot = 0.0f;
        for (uint e = 0u; e < E; ++e) {
            uint d = lane + 32u * e;
            if (d < p.dim) dot += q[e] * K[base + d];
        }
        dot = simd_sum(dot);
        uint distance = position - t;
        float relative = distance < p.m ? arena[p.bias_off + row * p.bias_stride + h * p.m + distance] : 0.0f;
        float score = dot * p.scale + relative;
        float next_maximum = max(maximum, score);
        float old_weight = (denominator > 0.0f || maximum > -INFINITY) ? metal::precise::exp(maximum - next_maximum) : 0.0f;
        float key_weight = metal::precise::exp(score - next_maximum);
        denominator = denominator * old_weight + key_weight;
        for (uint e = 0u; e < E; ++e) {
            uint d = lane + 32u * e;
            if (d < p.dim) acc[e] = acc[e] * old_weight + key_weight * V[base + d];
        }
        maximum = next_maximum;
    }
    device float *out = arena + p.out_off + row * p.out_stride + h * p.dim;
    for (uint e = 0u; e < E; ++e) {
        uint d = lane + 32u * e;
        if (d < p.dim) out[d] = (denominator > 0.0f) ? (acc[e] / denominator) : 0.0f;
    }
}

// Striped single-row sparse relative decode (dim == 128, heads == 3 * kv_heads) reading BF16 mirrors.
// Emits log2 softmax partials into parts[h * taps * 130 + part * 130], matching mk_relative_long_merge_fast/_gated.
#define MK_LSA_DEC_SIMDS 8u
template <bool InlineBias>
inline void mk_relative_lsa_decode_keys_fast_impl(device const float *arena, device const ushort *key,
                                                  device const ushort *val, device float *parts,
                                                  device const uint *blockList, device const uint *blockCount,
                                                  device const uchar *rel_w_bytes, constant MKLSA &p,
                                                  uint lane, uint simd, uint simds, uint g,
                                                  threadgroup float4 *red_acc, threadgroup float *red_m,
                                                  threadgroup float *red_d) {
    uint kvh = g / p.taps, part = g % p.taps;
    uint row = p.row0, pos = p.pos0;
    uint cnt = blockCount[0];
    uint tail0 = (cnt > 0u) ? ((pos + 1u) & ~3u) : 0u;
    uint total_keys = cnt * 4u + (pos + 1u - tail0);
    uint s_off = uint(ulong(total_keys) * part / p.taps);
    uint e_off = uint(ulong(total_keys) * (part + 1u) / p.taps);
    uint n = e_off - s_off;

    uint stride = p.kv_heads * 128u;
    uint h0 = kvh * 3u;
    float qscale = p.scale * M_LOG2E_F;
    float4 q[3];
    #pragma unroll
    for (uint h = 0u; h < 3u; ++h) {
        q[h] = float4(*(device const packed_float4 *)(arena + p.q_off + row * p.q_stride + (h0 + h) * 128u + lane * 4u)) * qscale;
    }

    uint last_t = 0u;
    if (n > 0u) {
        uint last_idx = e_off - 1u;
        last_t = (cnt > 0u && last_idx < cnt * 4u)
               ? ((blockList[last_idx >> 2u] << 2u) | (last_idx & 3u))
               : (tail0 + (cnt > 0u ? (last_idx - cnt * 4u) : last_idx));
    }
    bool stripe_has_rel = (n > 0u) && ((pos - last_t) < p.m);
    float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f;
    device const ushort *rel_w = InlineBias ? reinterpret_cast<device const ushort *>(rel_w_bytes + p.rel_w_off) : nullptr;
    if constexpr (InlineBias) {
        if (stripe_has_rel && lane < 16u) {
            device const float *rp = arena + p.bias_off + row * p.bias_stride;
            r0 = rp[(h0 + 0u) * 16u + lane];
            r1 = rp[(h0 + 1u) * 16u + lane];
            r2 = rp[(h0 + 2u) * 16u + lane];
        }
    }

    float m[3] = { -INFINITY, -INFINITY, -INFINITY };
    float d[3] = { 0.0f, 0.0f, 0.0f };
    float4 acc[3] = { float4(0.0f), float4(0.0f), float4(0.0f) };
    device const float *bias_base = InlineBias ? nullptr : (arena + p.bias_off + row * p.bias_stride + h0 * p.m);
    uint k_off = kvh * 128u + lane * 4u;

    for (uint i = simd; i < n; i += simds) {
        uint idx = s_off + i;
        uint t = (cnt > 0u && idx < cnt * 4u)
               ? ((blockList[idx >> 2u] << 2u) | (idx & 3u))
               : (tail0 + (cnt > 0u ? (idx - cnt * 4u) : idx));
        uint slot = (t < p.capacity) ? t : (t % p.capacity);
        ulong kv_idx = ulong(slot) * stride + k_off;
        float4 k = as_type<float4>(uint4(*(device const ushort4 *)(key + kv_idx)) << 16u);
        float4 v = as_type<float4>(uint4(*(device const ushort4 *)(val + kv_idx)) << 16u);
        float s0 = simd_sum(dot(q[0], k));
        float s1 = simd_sum(dot(q[1], k));
        float s2 = simd_sum(dot(q[2], k));
        uint distance = pos - t;
        float sc[3];
        if (stripe_has_rel && distance < p.m) {
            if constexpr (InlineBias) {
                float wk = (lane < 16u) ? as_type<float>(uint(rel_w[ulong(lane) * p.m + distance]) << 16u) : 0.0f;
                float b0 = mk_bf16(simd_sum(r0 * wk));
                float b1 = mk_bf16(simd_sum(r1 * wk));
                float b2 = mk_bf16(simd_sum(r2 * wk));
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
                float nm = max(m[h], sc[h]);
                float prev = fast::exp2(m[h] - nm);
                float w = fast::exp2(sc[h] - nm);
                d[h] = d[h] * prev + w;
                acc[h] = acc[h] * prev + w * v;
                m[h] = nm;
            }
        }
    }

    #pragma unroll
    for (uint h = 0u; h < 3u; ++h) {
        red_acc[(h * MK_LSA_DEC_SIMDS + simd) * 32u + lane] = acc[h];
        if (lane == 0u) {
            red_m[h * MK_LSA_DEC_SIMDS + simd] = m[h];
            red_d[h * MK_LSA_DEC_SIMDS + simd] = d[h];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd < 3u) {
        uint h = simd;
        float m_max = -INFINITY;
        for (uint s = 0u; s < simds; ++s) {
            if (red_d[h * MK_LSA_DEC_SIMDS + s] > 0.0f) m_max = max(m_max, red_m[h * MK_LSA_DEC_SIMDS + s]);
        }
        float4 tot_acc = float4(0.0f);
        float tot_d = 0.0f;
        if (m_max > -INFINITY) {
            for (uint s = 0u; s < simds; ++s) {
                float sd = red_d[h * MK_LSA_DEC_SIMDS + s];
                if (sd > 0.0f) {
                    float w = fast::exp2(red_m[h * MK_LSA_DEC_SIMDS + s] - m_max);
                    tot_d += sd * w;
                    tot_acc += red_acc[(h * MK_LSA_DEC_SIMDS + s) * 32u + lane] * w;
                }
            }
        }
        device float *dst = parts + (ulong(h0 + h) * p.taps + part) * 130u;
        *(device packed_float4 *)(dst + lane * 4u) = packed_float4(tot_acc);
        if (lane == 0u) {
            dst[128] = m_max;
            dst[129] = tot_d;
        }
    }
}

kernel void mk_relative_lsa_decode_keys_fast(device const float *arena [[buffer(0)]], device const ushort *key [[buffer(1)]],
                                           device const ushort *val [[buffer(2)]], device float *parts [[buffer(3)]],
                                           device const uint *blockList [[buffer(4)]], device const uint *blockCount [[buffer(5)]],
                                           constant MKLSA &p [[buffer(8)]], uint lane [[thread_index_in_simdgroup]],
                                           uint simd [[simdgroup_index_in_threadgroup]], uint simds [[simdgroups_per_threadgroup]],
                                           uint g [[threadgroup_position_in_grid]]) {
    threadgroup float4 red_acc[3u * MK_LSA_DEC_SIMDS * 32u];
    threadgroup float red_m[3u * MK_LSA_DEC_SIMDS];
    threadgroup float red_d[3u * MK_LSA_DEC_SIMDS];
    mk_relative_lsa_decode_keys_fast_impl<false>(arena, key, val, parts, blockList, blockCount, nullptr,
                                                 p, lane, simd, simds, g, red_acc, red_m, red_d);
}

kernel void mk_relative_lsa_decode_keys_inline_fast(device const float *arena [[buffer(0)]], device const ushort *key [[buffer(1)]],
                                                  device const ushort *val [[buffer(2)]], device float *parts [[buffer(3)]],
                                                  device const uint *blockList [[buffer(4)]], device const uint *blockCount [[buffer(5)]],
                                                  device const uchar *rel_w [[buffer(6)]], constant MKLSA &p [[buffer(8)]],
                                                  uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]],
                                                  uint simds [[simdgroups_per_threadgroup]], uint g [[threadgroup_position_in_grid]]) {
    threadgroup float4 red_acc[3u * MK_LSA_DEC_SIMDS * 32u];
    threadgroup float red_m[3u * MK_LSA_DEC_SIMDS];
    threadgroup float red_d[3u * MK_LSA_DEC_SIMDS];
    mk_relative_lsa_decode_keys_fast_impl<true>(arena, key, val, parts, blockList, blockCount, rel_w,
                                                p, lane, simd, simds, g, red_acc, red_m, red_d);
}
#undef MK_LSA_DEC_SIMDS

// Tiled sparse BF16 prefill (dim == 128, 32 query rows / tile, 16-key gathered half MMA chunks).
// Supports row_step = 32 (per-tile selection) or row_step = 1 (row unit).
kernel void mk_relative_lsa_prefill(device float *arena [[buffer(0)]], device const bfloat *keys [[buffer(1)]],
                                    device const bfloat *values [[buffer(2)]], device const uint *blockList [[buffer(3)]],
                                    device const uint *blockCount [[buffer(4)]], constant MKLSA &p [[buffer(8)]],
                                    uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]) {
    constexpr uint D = 128u, Q = 32u, BK = 16u, LDK = BK + 8u, LDV = D + 8u;
    uint h = group % p.heads, kv = h / (p.heads / p.kv_heads);
    uint n_tiles = (p.rows + Q - 1u) / Q;
    uint tile = n_tiles - 1u - (group / p.heads);
    uint q0 = tile * Q;
    uint ref_r = min(q0 + Q - 1u, p.rows - 1u);
    uint unit = (p.row_step > 1u) ? tile : ref_r;
    uint ref_pos = p.pos0 + ref_r;

    uint cnt = blockCount[unit];
    uint tail0 = (cnt > 0u) ? ((ref_pos + 1u) & ~3u) : 0u;
    uint total_keys = cnt * 4u + (ref_pos + 1u - tail0);
    device const uint *rowBlocks = blockList + ulong(unit) * p.list_stride;

    uint fr = ((lane / 4u) & 4u) + (lane / 2u) % 4u;
    uint fc = (((lane / 4u) & 2u) + lane % 2u) * 2u;
    uint local = q0 + 8u * simd + fr;
    uint row = p.row0 + local;
    uint pos = p.pos0 + local;
    bool active = local < p.rows;

    simdgroup_half8x8 q_frag[D / 8u];
    simdgroup_float8x8 out[D / 8u];
    #pragma unroll
    for (uint j = 0u; j < D / 8u; ++j) {
        if (active) {
            const device float *qp = arena + p.q_off + ulong(row) * p.q_stride + h * D + j * 8u + fc;
            q_frag[j].thread_elements()[0] = half(qp[0]);
            q_frag[j].thread_elements()[1] = half(qp[1]);
        } else {
            q_frag[j].thread_elements()[0] = half(0.0h);
            q_frag[j].thread_elements()[1] = half(0.0h);
        }
        out[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    threadgroup half Ks[D * LDK], Vs[BK * LDV];
    threadgroup uint sh_kp[BK];
    threadgroup uchar sh_valid[BK];
    float maximum = active ? (p.has_sink != 0u ? (arena[p.sink_off + ulong(row) * p.sink_stride + h] * M_LOG2E_F) : -1e30f) : -1e30f;
    float denominator = (active && p.has_sink != 0u) ? 1.0f : 0.0f;
    float scale = p.scale * M_LOG2E_F;
    ulong stride = ulong(p.kv_heads) * D;

    for (uint base = 0u; base < total_keys; base += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint c = tid; c < BK * (D / 8u); c += 128u) {
            uint r = c / (D / 8u);
            uint col = (c % (D / 8u)) * 8u;
            uint k_idx = base + r;
            bool kvalid = (k_idx < total_keys);
            uint kp = 0u;
            vec<bfloat, 4> kv0 = vec<bfloat, 4>(0), kv1 = vec<bfloat, 4>(0);
            vec<bfloat, 4> vv0 = vec<bfloat, 4>(0), vv1 = vec<bfloat, 4>(0);
            if (kvalid) {
                if (cnt > 0u && k_idx < cnt * 4u) {
                    kp = (rowBlocks[k_idx >> 2u] << 2u) | (k_idx & 3u);
                } else {
                    kp = tail0 + (cnt > 0u ? (k_idx - cnt * 4u) : k_idx);
                }
                ulong kv_base = ulong((kp < p.capacity) ? kp : (kp % p.capacity)) * stride + kv * D + col;
                kv0 = *(const device vec<bfloat, 4> *)(keys + kv_base);
                kv1 = *(const device vec<bfloat, 4> *)(keys + kv_base + 4u);
                vv0 = *(const device vec<bfloat, 4> *)(values + kv_base);
                vv1 = *(const device vec<bfloat, 4> *)(values + kv_base + 4u);
            }
            if (col == 0u) {
                sh_kp[r] = kp;
                sh_valid[r] = kvalid ? 1u : 0u;
            }
            #pragma unroll
            for (uint j = 0u; j < 4u; ++j) {
                Ks[(col + j) * LDK + r] = half(float(kv0[j]));
                Ks[(col + 4u + j) * LDK + r] = half(float(kv1[j]));
                Vs[r * LDV + col + j] = half(float(vv0[j]));
                Vs[r * LDV + col + 4u + j] = half(float(vv1[j]));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (q0 + simd * 8u >= p.rows) continue;

        simdgroup_float8x8 scores[2];
        #pragma unroll
        for (uint kb = 0u; kb < 2u; ++kb) {
            scores[kb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            #pragma unroll
            for (uint k = 0u; k < D / 8u; ++k) {
                simdgroup_half8x8 wk;
                simdgroup_load(wk, Ks + k * 8u * LDK + kb * 8u, LDK);
                simdgroup_multiply_accumulate(scores[kb], q_frag[k], wk, scores[kb]);
            }
        }
        float bound = -INFINITY;
        #pragma unroll
        for (uint kb = 0u; kb < 2u; ++kb) {
            #pragma unroll
            for (uint e = 0u; e < 2u; ++e) {
                uint km = kb * 8u + fc + e;
                uint key_pos = sh_kp[km];
                float score = -INFINITY;
                if (active && sh_valid[km] != 0u && key_pos <= pos) {
                    uint distance = pos - key_pos;
                    score = scores[kb].thread_elements()[e] * scale;
                    if (distance < p.m) score += arena[p.bias_off + ulong(row) * p.bias_stride + h * p.m + distance] * M_LOG2E_F;
                }
                scores[kb].thread_elements()[e] = score;
                bound = max(bound, score);
            }
        }
        bound = max(bound, simd_shuffle_xor(bound, ushort(1)));
        bound = max(bound, simd_shuffle_xor(bound, ushort(8)));
        bound = max(bound, maximum);
        float rescale = fast::exp2(maximum - bound);
        maximum = bound;
        float weight_sum = 0.0f;
        simdgroup_half8x8 probs[2];
        #pragma unroll
        for (uint kb = 0u; kb < 2u; ++kb) {
            #pragma unroll
            for (uint e = 0u; e < 2u; ++e) {
                float w = fast::exp2(scores[kb].thread_elements()[e] - bound);
                weight_sum += w;
                probs[kb].thread_elements()[e] = half(w);
            }
        }
        weight_sum += simd_shuffle_xor(weight_sum, ushort(1));
        weight_sum += simd_shuffle_xor(weight_sum, ushort(8));
        denominator = rescale * denominator + weight_sum;
        #pragma unroll
        for (uint j = 0u; j < D / 8u; ++j) {
            out[j].thread_elements()[0] *= rescale;
            out[j].thread_elements()[1] *= rescale;
            #pragma unroll
            for (uint kb = 0u; kb < 2u; ++kb) {
                simdgroup_half8x8 vv;
                simdgroup_load(vv, Vs + kb * 8u * LDV + j * 8u, LDV);
                simdgroup_multiply_accumulate(out[j], probs[kb], vv, out[j]);
            }
        }
    }
    if (active) {
        const float inv_den = (denominator > 0.0f) ? (1.0f / denominator) : 0.0f;
        for (uint j = 0u; j < D / 8u; ++j) {
            for (uint e = 0u; e < 2u; ++e)
                arena[p.out_off + ulong(row) * p.out_stride + h * D + j * 8u + fc + e] = out[j].thread_elements()[e] * inv_den;
        }
    }
}
