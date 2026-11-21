// ---- striped single-row relative attention (decode) --------------------------
// One query row, head size 128. K and V are read from 16-bit mirrors of the
// float caches: every stored row is a bfloat16 value, so its upper half is the
// whole number and the arithmetic sees the same floats at half the memory
// traffic. One threadgroup per (KV head, stripe of keys); the heads / kv_heads
// query heads that share a KV head (exactly 3) share each key and value load,
// and SIMD group h of the threadgroup writes query head h, so a threadgroup
// needs at least three SIMD groups. Each stripe is reduced against its own
// maximum and the stripes are merged against the global one, so nothing in a
// stripe waits on another: the sum is the reference's with a different
// association.

#define MK_LONG_KEYS 512u   // most keys in one stripe
#define MK_LONG_SIMDS 16u   // most SIMD groups in one stripe's threadgroup
#define MK_LONG_PARTS 1024u // most stripes

inline float4 mk_bf16x4(device const ushort *base, ulong i) {
    ushort4 h = *(device const ushort4 *)(base + i);
    return as_type<float4>(uint4(h) << 16);
}

#define MK_MIRROR_PAD 16u // rows after the mirror's last that repeat its first

// The 16-bit mirror of a stored row (mk_store's parameters): capacity rows, then
// MK_MIRROR_PAD rows repeating the first ones, for readers of 16 consecutive rows.
kernel void mk_pack_bf16(device const float *arena [[buffer(0)]], device ushort *dst [[buffer(1)]],
                         constant MK &p [[buffer(8)]], uint i [[thread_position_in_grid]]) {
    uint r = i / p.n, c = i % p.n;
    if (r >= p.rows) return;
    uint row = p.row0 + r;
    uint slot = (p.pos + row) % p.capacity;
    const ushort value = ushort(as_type<uint>(arena[p.x + row * p.xs + c]) >> 16);
    dst[ulong(slot) * p.n + c] = value;
    if (slot < MK_MIRROR_PAD) dst[ulong(p.capacity + slot) * p.n + c] = value;
}

// parts: [heads][taps] records of {sum of w * value [128], stripe maximum, sum of w}.
kernel void mk_relative_long_keys(device const float *arena [[buffer(0)]], device const ushort *key [[buffer(1)]],
                                  device const ushort *val [[buffer(2)]], device float *parts [[buffer(3)]],
                                  constant MK &p [[buffer(8)]], uint lane [[thread_index_in_simdgroup]],
                                  uint simd [[simdgroup_index_in_threadgroup]], uint simds [[simdgroups_per_threadgroup]],
                                  uint g [[threadgroup_position_in_grid]]) {
    const uint kvh = g / p.taps, part = g % p.taps;
    const uint row = p.row0, pos = p.pos + row;
    const uint visible = p.window ? min(p.window, pos + 1) : pos + 1;
    const uint start = pos + 1 - visible + uint(ulong(visible) * part / p.taps);
    const uint n = pos + 1 - visible + uint(ulong(visible) * (part + 1) / p.taps) - start;
    const uint stride = p.kv_heads * 128;
    const uint h0 = kvh * 3;
    float4 q[3];
    #pragma unroll
    for (uint h = 0; h < 3; ++h) q[h] = float4(*(device const packed_float4 *)(arena + p.x + row * p.xs + (h0 + h) * 128 + lane * 4));

    threadgroup float score[3 * MK_LONG_KEYS];
    for (uint i = simd; i < n; i += simds) {
        const uint t = start + i;
        const float4 k = mk_bf16x4(key, ulong(t % p.capacity) * stride + kvh * 128 + lane * 4);
        const float s0 = simd_sum(dot(q[0], k)), s1 = simd_sum(dot(q[1], k)), s2 = simd_sum(dot(q[2], k));
        if (lane < 3) {
            const uint distance = pos - t;
            const float bias = distance < p.m ? arena[p.z + row * p.zs + (h0 + lane) * p.m + distance] : 0.0f;
            score[lane * MK_LONG_KEYS + i] = (lane == 0 ? s0 : lane == 1 ? s1 : s2) * p.scale + bias;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float maximum[3];
    #pragma unroll
    for (uint h = 0; h < 3; ++h) {
        float m = -INFINITY;
        for (uint i = lane; i < n; i += 32) m = max(m, score[h * MK_LONG_KEYS + i]);
        maximum[h] = simd_max(m);
    }
    float4 acc[3] = { float4(0.0f), float4(0.0f), float4(0.0f) };
    float denominator[3] = { 0.0f, 0.0f, 0.0f };
    for (uint i = simd; i < n; i += simds) {
        const uint t = start + i;
        const float4 v = mk_bf16x4(val, ulong(t % p.capacity) * stride + kvh * 128 + lane * 4);
        #pragma unroll
        for (uint h = 0; h < 3; ++h) {
            const float w = metal::precise::exp(score[h * MK_LONG_KEYS + i] - maximum[h]);
            denominator[h] += w;
            acc[h] += w * v;
        }
    }
    // The SIMD groups of a stripe share its maximum, so their sums add.
    threadgroup float4 sum[MK_LONG_SIMDS * 3 * 32];
    threadgroup float total[MK_LONG_SIMDS * 3];
    #pragma unroll
    for (uint h = 0; h < 3; ++h) sum[(simd * 3 + h) * 32 + lane] = acc[h];
    if (lane < 3) total[simd * 3 + lane] = lane == 0 ? denominator[0] : lane == 1 ? denominator[1] : denominator[2];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd < 3) {
        const uint h = simd;
        float4 out = float4(0.0f);
        float d = 0.0f;
        for (uint s = 0; s < simds; ++s) {
            out += sum[(s * 3 + h) * 32 + lane];
            d += total[s * 3 + h];
        }
        device float *dst = parts + (ulong(h0 + h) * p.taps + part) * 130;
        *(device packed_float4 *)(dst + lane * 4) = packed_float4(out);
        if (lane == 0) {
            dst[128] = maximum[h];
            dst[129] = d;
        }
    }
}

// 128 threads per head, one per output dimension.
kernel void mk_relative_long_merge(device float *arena [[buffer(0)]], device const float *parts [[buffer(1)]],
                                   constant MK &p [[buffer(8)]], uint tid [[thread_index_in_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]],
                                   uint h [[threadgroup_position_in_grid]]) {
    const uint row = p.row0;
    device const float *src = parts + ulong(h) * p.taps * 130;
    const float sink = arena[p.s + row * p.every + h];
    threadgroup float group[4];
    threadgroup float factor[MK_LONG_PARTS];
    float m = -INFINITY;
    for (uint k = tid; k < p.taps; k += 128) m = max(m, src[ulong(k) * 130 + 128]);
    m = simd_max(m);
    if (lane == 0) group[simd] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = max(max(max(group[0], group[1]), max(group[2], group[3])), sink);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float d = 0.0f;
    for (uint k = tid; k < p.taps; k += 128) {
        const float f = metal::precise::exp(src[ulong(k) * 130 + 128] - m);
        factor[k] = f;
        d += f * src[ulong(k) * 130 + 129];
    }
    d = simd_sum(d);
    if (lane == 0) group[simd] = d;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float denominator = metal::precise::exp(sink - m) + ((group[0] + group[1]) + (group[2] + group[3]));
    float acc = 0.0f;
    for (uint k = 0; k < p.taps; ++k) acc += factor[k] * src[ulong(k) * 130 + tid];
    arena[p.y + row * p.ys + h * 128 + tid] = acc / denominator;
}

// ---- batched relative attention in key stripes (prefill) -----------------------
// mk_relative_attention_prefill's tiles (32 queries share each 16-key tile, FP32
// online normalization in log2 units), with two changes. The tiles a row tile
// walks are cut into p.taps consecutive stripes, one threadgroup each, so a
// chunk occupies the GPU instead of 96 threadgroups walking every key in turn:
// stripe 0 starts from the sink as the tiled kernel does, the others start
// empty, each writes {sum of w * value [128], maximum, sum of w} per query and
// mk_relative_prefill_merge combines them (the sum is associated per stripe).
// And the K and V fragments are loaded straight from the 16-bit mirrors as
// bfloat16 matrices, so a tile needs no staging and no barrier; the products see
// the caches' exact values and the queries' (bfloat16 too), where the tiled
// kernel rounds all three to FP16. A tile is 16 consecutive mirror rows from
// its first key's slot: the mirror repeats its first 16 rows after its last
// (mk_pack_bf16), so a tile that crosses a ring's end, or reaches past the last
// stored position (those keys are masked), stays inside it.
#define MK_STRIPE_EMPTY -1e30f // a maximum below every score, finite so empty tiles stay finite

kernel void mk_relative_prefill_stripes(device const float *arena [[buffer(0)]], device const bfloat *keys [[buffer(1)]],
                                       device const bfloat *values [[buffer(2)]], device float *parts [[buffer(3)]],
                                       constant MK &p [[buffer(8)]], uint group [[threadgroup_position_in_grid]],
                                       uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]) {
    constexpr uint D = 128, Q = 32, B = 16;
    const uint part = group % p.taps, qh = group / p.taps;
    const uint h = qh % p.heads, kv = h / (p.heads / p.kv_heads);
    const uint q0 = ((p.rows + Q - 1) / Q - 1 - qh / p.heads) * Q;
    if (q0 + simd * 8 >= p.rows) return;
    const uint start = p.pos + p.row0;
    const uint earliest = p.window ? start + 1 - min(start + 1, p.window) : 0;
    const uint qfirst = start + q0, qlast = start + min(q0 + Q, p.rows) - 1;
    const uint lower = p.window ? qfirst + 1 - min(qfirst + 1, p.window) : 0;
    const uint block0 = earliest + (lower - earliest) / B * B;
    const uint tiles = (qlast - block0) / B + 1;
    const uint first = block0 + B * uint(ulong(tiles) * part / p.taps);
    const uint last = block0 + B * uint(ulong(tiles) * (part + 1) / p.taps);
    const uint fr = ((lane / 4) & 4) + (lane / 2) % 4;
    const uint fc = (((lane / 4) & 2) + lane % 2) * 2;
    const uint local = q0 + 8 * simd + fr, row = p.row0 + local, pos = start + local;
    const bool active = local < p.rows;
    simdgroup_bfloat8x8 q_frag[D/8];
    simdgroup_float8x8 out[D/8];
    #pragma unroll
    for (uint j = 0; j < D/8; ++j) {
        for (uint e = 0; e < 2; ++e)
            q_frag[j].thread_elements()[e] = active ? bfloat(arena[p.x + row * p.xs + h * D + j * 8 + fc + e]) : bfloat(0.0f);
        out[j] = make_filled_simdgroup_matrix<float,8,8>(0);
    }
    float maximum = part == 0 && active ? arena[p.s + row * p.every + h] * M_LOG2E_F : MK_STRIPE_EMPTY;
    float denominator = part == 0 ? 1 : 0;
    const float scale = p.scale * M_LOG2E_F;
    const ulong stride = ulong(p.kv_heads) * D;
    for (uint block = first; block < last; block += B) {
        const ulong slot = block % p.capacity;
        device const bfloat *kb = keys + slot * stride + kv * D;
        device const bfloat *vb = values + slot * stride + kv * D;
        simdgroup_float8x8 scores[B/8];
        simdgroup_half8x8 probs[B/8];
        float bound = -INFINITY;
        #pragma unroll
        for (uint j = 0; j < B/8; ++j) {
            scores[j] = make_filled_simdgroup_matrix<float,8,8>(0);
            #pragma unroll
            for (uint k = 0; k < D/8; ++k) {
                simdgroup_bfloat8x8 wk;
                simdgroup_load(wk, kb + ulong(j * 8) * stride + k * 8, stride, ulong2(0), true);
                simdgroup_multiply_accumulate(scores[j], q_frag[k], wk, scores[j]);
            }
            #pragma unroll
            for (uint e = 0; e < 2; ++e) {
                const uint key = block + j * 8 + fc + e;
                const uint distance = pos - key;
                float score = -INFINITY;
                if (active && key <= pos && (p.window == 0 || distance < p.window)) {
                    score = scores[j].thread_elements()[e] * scale;
                    if (distance < p.m) score += arena[p.z + row * p.zs + h * p.m + distance] * M_LOG2E_F;
                }
                scores[j].thread_elements()[e] = score;
                bound = max(bound, score);
            }
        }
        bound = max(bound, simd_shuffle_xor(bound, ushort(1)));
        bound = max(bound, simd_shuffle_xor(bound, ushort(8)));
        bound = max(bound, maximum);
        const float rescale = fast::exp2(maximum - bound);
        maximum = bound;
        float weight_sum = 0;
        #pragma unroll
        for (uint j = 0; j < B/8; ++j) for (uint e = 0; e < 2; ++e) {
            float w = fast::exp2(scores[j].thread_elements()[e] - bound);
            weight_sum += w;
            probs[j].thread_elements()[e] = half(w);
        }
        weight_sum += simd_shuffle_xor(weight_sum, ushort(1));
        weight_sum += simd_shuffle_xor(weight_sum, ushort(8));
        denominator = rescale * denominator + weight_sum;
        #pragma unroll
        for (uint j = 0; j < D/8; ++j) {
            out[j].thread_elements()[0] *= rescale;
            out[j].thread_elements()[1] *= rescale;
            #pragma unroll
            for (uint k = 0; k < B/8; ++k) {
                simdgroup_bfloat8x8 vv;
                simdgroup_load(vv, vb + ulong(k * 8) * stride + j * 8, stride);
                simdgroup_multiply_accumulate(out[j], probs[k], vv, out[j]);
            }
        }
    }
    if (active) {
        device float *dst = parts + ((ulong(local) * p.heads + h) * p.taps + part) * (D + 2);
        for (uint j = 0; j < D/8; ++j) for (uint e = 0; e < 2; ++e) dst[j * 8 + fc + e] = out[j].thread_elements()[e];
        if (fc == 0) {
            dst[D] = maximum;
            dst[D + 1] = denominator;
        }
    }
}

// One thread per (row, head, dimension).
kernel void mk_relative_prefill_merge(device float *arena [[buffer(0)]], device const float *parts [[buffer(1)]],
                                      constant MK &p [[buffer(8)]], uint i [[thread_position_in_grid]]) {
    const uint d = i % 128, qh = i / 128, h = qh % p.heads, local = qh / p.heads;
    if (local >= p.rows) return;
    device const float *src = parts + ulong(qh) * p.taps * 130;
    float maximum = src[128];
    for (uint k = 1; k < p.taps; ++k) maximum = max(maximum, src[ulong(k) * 130 + 128]);
    float acc = 0.0f, denominator = 0.0f;
    for (uint k = 0; k < p.taps; ++k) {
        const float f = fast::exp2(src[ulong(k) * 130 + 128] - maximum);
        denominator += f * src[ulong(k) * 130 + 129];
        acc += f * src[ulong(k) * 130 + d];
    }
    arena[p.y + (p.row0 + local) * p.ys + h * 128 + d] = acc / denominator;
}
