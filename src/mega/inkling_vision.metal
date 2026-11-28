// Native Metal kernels for Inkling 2 Vision (23-layer Perception-Encoder ViT +
// 2x2 spatial merge + GELU MLP projector) and multimodal embedding scatter.
// Uses the shared MK parameter block and mk_load / mk_bf16 helpers from kernels.metal.

inline float mk_erf_approx(float x) {
    float sign = x < 0.0f ? -1.0f : 1.0f;
    float ax = metal::abs(x);
    float t = 1.0f / (1.0f + 0.3275911f * ax);
    float poly = (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t;
    return sign * (1.0f - poly * metal::precise::exp(-ax * ax));
}

inline float mk_gelu(float x) {
    return 0.5f * x * (1.0f + mk_erf_approx(x * 0.70710678118654752440f));
}

// ---- Prepend CLS + add bilinearly interpolated 2D position embeddings -------
// Grid: (1 + gh * gw) * D threads.
// p.w = pos_emb offset, p.s = class_emb offset, p.b = cls_pos_emb offset,
// p.n = D, p.heads = gh, p.kv_heads = gw, p.dim = posemb_grid_size (32), p.dtype = weight dtype.
kernel void mk_vis_embed_prep(device const float *patch_proj [[buffer(0)]],
                              device const uchar *pos_buf [[buffer(1)]],
                              device const uchar *cls_buf [[buffer(2)]],
                              device const uchar *cls_pos_buf [[buffer(3)]],
                              device float *seq_out [[buffer(4)]],
                              constant MK &p [[buffer(8)]],
                              uint gid [[thread_position_in_grid]]) {
    uint D = p.n;
    uint gh = p.heads;
    uint gw = p.kv_heads;
    uint total_rows = 1u + gh * gw;
    uint r = gid / D;
    uint c = gid % D;
    if (r >= total_rows) return;
    if (r == 0u) {
        float cls_v = mk_load(cls_buf, p.s, p.dtype, c);
        float cls_p = mk_load(cls_pos_buf, p.b, p.dtype, c);
        seq_out[p.y + c] = cls_v + cls_p;
        return;
    }
    uint idx = r - 1u;
    uint py = idx / gw;
    uint px = idx % gw;
    uint G = p.dim;
    float sy = (float(py) + 0.5f) * (float(G) / float(gh)) - 0.5f;
    float sx = (float(px) + 0.5f) * (float(G) / float(gw)) - 0.5f;
    sy = clamp(sy, 0.0f, float(G - 1u));
    sx = clamp(sx, 0.0f, float(G - 1u));
    uint y0 = uint(floor(sy));
    uint x0 = uint(floor(sx));
    uint y1 = min(y0 + 1u, G - 1u);
    uint x1 = min(x0 + 1u, G - 1u);
    float wy = sy - float(y0);
    float wx = sx - float(x0);
    float p00 = mk_load(pos_buf, p.w, p.dtype, ulong(y0 * G + x0) * D + c);
    float p01 = mk_load(pos_buf, p.w, p.dtype, ulong(y0 * G + x1) * D + c);
    float p10 = mk_load(pos_buf, p.w, p.dtype, ulong(y1 * G + x0) * D + c);
    float p11 = mk_load(pos_buf, p.w, p.dtype, ulong(y1 * G + x1) * D + c);
    float pe = (1.0f - wy) * ((1.0f - wx) * p00 + wx * p01) + wy * ((1.0f - wx) * p10 + wx * p11);
    seq_out[p.y + r * p.ys + c] = patch_proj[p.x + idx * p.xs + c] + pe;
}

// ---- LayerNorm with weight and bias -----------------------------------------
// One threadgroup (256 threads) per row r in [0, p.rows).
kernel void mk_vis_layernorm(device const float *x_buf [[buffer(0)]],
                             device const uchar *w_buf [[buffer(1)]],
                             device const uchar *b_buf [[buffer(2)]],
                             device float *y_buf [[buffer(3)]],
                             constant MK &p [[buffer(8)]],
                             uint tgid [[threadgroup_position_in_grid]],
                             uint lid [[thread_position_in_threadgroup]],
                             uint tg [[threads_per_threadgroup]],
                             uint simd [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
    if (tgid >= p.rows) return;
    uint n = p.n;
    device const float *xr = x_buf + p.x + ulong(tgid) * p.xs;
    device float *yr = y_buf + p.y + ulong(tgid) * p.ys;

    float sum = 0.0f;
    for (uint c = lid; c < n; c += tg) sum += xr[c];
    sum = simd_sum(sum);
    threadgroup float partial[32];
    if (lane == 0u) partial[simd] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint n_simds = (tg + 31u) / 32u;
    float mean = 0.0f;
    if (simd == 0u) {
        float v = lane < n_simds ? partial[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0u) partial[0] = v / float(n);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mean = partial[0];

    float var_sum = 0.0f;
    for (uint c = lid; c < n; c += tg) {
        float d = xr[c] - mean;
        var_sum += d * d;
    }
    var_sum = simd_sum(var_sum);
    if (lane == 0u) partial[simd] = var_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0u) {
        float v = lane < n_simds ? partial[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0u) partial[0] = metal::precise::rsqrt(v / float(n) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = partial[0];

    bool round_out = (p.flags & MK_ROUND_OUT) != 0u;
    for (uint c = lid; c < n; c += tg) {
        float g = mk_load(w_buf, p.w, p.dtype, c);
        float b = mk_load(b_buf, p.b, p.bdtype, c);
        float v = (xr[c] - mean) * inv * g + b;
        yr[c] = round_out ? mk_bf16(v) : v;
    }
}

// ---- Tiled simdgroup_matrix dense linear: Y[rows, M] = X[rows, N] @ W[M, N]^T + bias
// Optional exact GELU epilogue when p.act_a == 4, optional BF16 output round via MK_ROUND_OUT.
// Threadgroup: 128 threads (4 simdgroups, 2x2 layout of 16x16 tiles = 32x32 output tile).
// Grid: tgs_m * tgs_r * 128 threads, where tgs_m = ceil(M/32), tgs_r = ceil(rows/32).
kernel void mk_vis_linear_mma(device const float *x_buf [[buffer(0)]],
                              device const uchar *w_buf [[buffer(1)]],
                              device const uchar *b_buf [[buffer(2)]],
                              device float *y_buf [[buffer(3)]],
                              constant MK &p [[buffer(8)]],
                              uint tgid [[threadgroup_position_in_grid]],
                              uint lid [[thread_position_in_threadgroup]],
                              uint simd [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
    uint M = p.m;
    uint N = p.n;
    uint R = p.rows;
    uint tiles_m = (M + 31u) / 32u;
    uint tm = tgid % tiles_m;
    uint tr = tgid / tiles_m;
    uint row_base = tr * 32u;
    uint col_base = tm * 32u;
    if (row_base >= R || col_base >= M) return;

    // 4 simdgroups arranged 2 (rows) x 2 (cols), each computing a 16x16 sub-tile (2x2 of 8x8).
    uint sg_r = (simd / 2u) * 16u;
    uint sg_c = (simd % 2u) * 16u;

    simdgroup_float8x8 acc[2][2];
    for (uint i = 0u; i < 2u; i++) {
        for (uint j = 0u; j < 2u; j++) {
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    threadgroup float As[32][32];
    threadgroup float Bs[32][32]; // Stored as [k][m] for direct simdgroup_multiply_accumulate

    bool round_in = (p.flags & MK_ROUND_BF16) != 0u;

    for (uint k0 = 0u; k0 < N; k0 += 32u) {
        // Cooperative load: 128 threads load 32x32 = 1024 floats each for As and Bs (8 elements/thread).
        for (uint e = lid; e < 1024u; e += 128u) {
            uint lr = e / 32u;
            uint lk = e % 32u;
            uint gr = row_base + lr;
            uint gk = k0 + lk;
            float xv = (gr < R && gk < N) ? x_buf[p.x + ulong(gr) * p.xs + gk] : 0.0f;
            As[lr][lk] = round_in ? mk_bf16(xv) : xv;

            uint gm = col_base + lr;
            float wv = (gm < M && gk < N) ? mk_load(w_buf, p.w, p.dtype, ulong(gm) * N + gk) : 0.0f;
            Bs[lk][lr] = wv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0u; kk < 32u; kk += 8u) {
            simdgroup_float8x8 a0, a1, b0, b1;
            simdgroup_load(a0, &As[sg_r + 0u][kk], 32);
            simdgroup_load(a1, &As[sg_r + 8u][kk], 32);
            simdgroup_load(b0, &Bs[kk][sg_c + 0u], 32);
            simdgroup_load(b1, &Bs[kk][sg_c + 8u], 32);
            simdgroup_multiply_accumulate(acc[0][0], a0, b0, acc[0][0]);
            simdgroup_multiply_accumulate(acc[0][1], a0, b1, acc[0][1]);
            simdgroup_multiply_accumulate(acc[1][0], a1, b0, acc[1][0]);
            simdgroup_multiply_accumulate(acc[1][1], a1, b1, acc[1][1]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store tile back to shared memory to apply bias / GELU / BF16 rounding and bounds check cleanly.
    for (uint i = 0u; i < 2u; i++) {
        for (uint j = 0u; j < 2u; j++) {
            simdgroup_store(acc[i][j], &As[sg_r + i * 8u][sg_c + j * 8u], 32);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    bool has_bias = (p.flags & MK_HAS_WEIGHT) != 0u;
    bool apply_gelu = (p.act_a == 4u);
    bool round_out = (p.flags & MK_ROUND_OUT) != 0u;

    for (uint e = lid; e < 1024u; e += 128u) {
        uint lr = e / 32u;
        uint lc = e % 32u;
        uint gr = row_base + lr;
        uint gc = col_base + lc;
        if (gr < R && gc < M) {
            float v = As[lr][lc];
            if (has_bias) v += mk_load(b_buf, p.b, p.bdtype, gc);
            if (apply_gelu) v = mk_gelu(v);
            if (round_out) v = mk_bf16(v);
            y_buf[p.y + ulong(gr) * p.ys + gc] = v;
        }
    }
}

// ---- In-place interleaved 2D RoPE on Q or K ---------------------------------
// Grid: rows * heads * (dim / 2) threads.
// Row 0 (CLS) is unrotated. Row 1 + py*gw + px uses 1-based coordinates:
// first dim/2 channels rotated by column (px + 1), second dim/2 channels
// rotated by row (py + 1), in interleaved pairs (2k, 2k+1).
// p.rows = S, p.heads = heads, p.dim = head_dim, p.window = gh, p.capacity = gw, p.scale = theta (10000).
kernel void mk_vis_rope2d(device float *buf [[buffer(0)]],
                          constant MK &p [[buffer(8)]],
                          uint gid [[thread_position_in_grid]]) {
    uint pairs_per_head = p.dim / 2u;
    uint pairs_per_row = p.heads * pairs_per_head;
    uint r = gid / pairs_per_row;
    if (r == 0u || r >= p.rows) return;
    uint rem = gid % pairs_per_row;
    uint h = rem / pairs_per_head;
    uint pair_idx = rem % pairs_per_head;

    uint patch_idx = r - 1u;
    uint gw = p.capacity;
    uint py = patch_idx / gw;
    uint px = patch_idx % gw;

    uint axis_pairs = pairs_per_head / 2u; // dim / 4
    bool is_row = pair_idx >= axis_pairs;
    uint k = is_row ? (pair_idx - axis_pairs) : pair_idx;
    float pos = is_row ? float(py + 1u) : float(px + 1u);
    float inv_freq = 1.0f / metal::precise::pow(p.scale, float(2u * k) / float(p.dim / 2u));
    float angle = pos * inv_freq;
    float cs = metal::precise::cos(angle);
    float sn = metal::precise::sin(angle);

    ulong off = p.x + ulong(r) * p.xs + ulong(h) * p.dim + ulong(2u * pair_idx);
    float a = buf[off];
    float b = buf[off + 1u];
    buf[off] = a * cs - b * sn;
    buf[off + 1u] = a * sn + b * cs;
}

// ---- Bidirectional multi-head self-attention --------------------------------
// One SIMD group (32 threads) per (row, head). Supports head_dim <= 128.
// Grid: rows * heads * 32 threads, threadgroup: 128 threads (4 simdgroups) or 32.
kernel void mk_vis_attention(device const float *q_buf [[buffer(0)]],
                             device const float *k_buf [[buffer(1)]],
                             device const float *v_buf [[buffer(2)]],
                             device float *out_buf [[buffer(3)]],
                             constant MK &p [[buffer(8)]],
                             uint gid [[thread_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]]) {
    uint task = gid / 32u;
    uint total_tasks = p.rows * p.heads;
    if (task >= total_tasks) return;
    uint r = task / p.heads;
    uint h = task % p.heads;
    uint dim = p.dim;
    uint head_off = h * dim;

    float q_reg[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float o_reg[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    device const float *qr = q_buf + p.x + ulong(r) * p.xs + head_off;
    uint iters = (dim + 31u) / 32u;
    for (uint u = 0u; u < iters; u++) {
        uint d = lane + u * 32u;
        if (d < dim) q_reg[u] = qr[d];
    }

    float m = -INFINITY;
    float l_sum = 0.0f;
    for (uint j = 0u; j < p.rows; j++) {
        device const float *kj = k_buf + p.y + ulong(j) * p.ys + head_off;
        float dot = 0.0f;
        for (uint u = 0u; u < iters; u++) {
            uint d = lane + u * 32u;
            if (d < dim) dot += q_reg[u] * kj[d];
        }
        dot = simd_sum(dot) * p.scale;

        float m_new = max(m, dot);
        float alpha = m == -INFINITY ? 0.0f : metal::precise::exp(m - m_new);
        float w = metal::precise::exp(dot - m_new);
        l_sum = l_sum * alpha + w;
        m = m_new;

        device const float *vj = v_buf + p.z + ulong(j) * p.zs + head_off;
        for (uint u = 0u; u < iters; u++) {
            uint d = lane + u * 32u;
            if (d < dim) o_reg[u] = o_reg[u] * alpha + w * vj[d];
        }
    }

    float inv_l = l_sum > 0.0f ? (1.0f / l_sum) : 0.0f;
    device float *orow = out_buf + p.pos + ulong(r) * p.window + head_off;
    for (uint u = 0u; u < iters; u++) {
        uint d = lane + u * 32u;
        if (d < dim) orow[d] = o_reg[u] * inv_l;
    }
}

// ---- LayerScale residual add: h[r, c] += gamma[c] * x[r, c] -----------------
kernel void mk_vis_layerscale_add(device float *h_buf [[buffer(0)]],
                                  device const float *x_buf [[buffer(1)]],
                                  device const uchar *g_buf [[buffer(2)]],
                                  constant MK &p [[buffer(8)]],
                                  uint gid [[thread_position_in_grid]]) {
    uint r = gid / p.n;
    uint c = gid % p.n;
    if (r >= p.rows) return;
    float g = mk_load(g_buf, p.w, p.dtype, c);
    ulong h_off = p.y + ulong(r) * p.ys + c;
    ulong x_off = p.x + ulong(r) * p.xs + c;
    h_buf[h_off] = h_buf[h_off] + g * x_buf[x_off];
}

// ---- 2x2 Spatial Merge (drops CLS at row 0) ---------------------------------
// Grid: (mh * mw) * (merge * merge * D) threads.
// p.heads = gh, p.kv_heads = gw, p.dim = merge (2), p.n = D (1024).
kernel void mk_vis_spatial_merge(device const float *seq_buf [[buffer(0)]],
                                 device float *merged_buf [[buffer(1)]],
                                 constant MK &p [[buffer(8)]],
                                 uint gid [[thread_position_in_grid]]) {
    uint merge = p.dim;
    uint D = p.n;
    uint out_w = merge * merge * D;
    uint mh = p.heads / merge;
    uint mw = p.kv_heads / merge;
    uint n_tok = mh * mw;
    uint t = gid / out_w;
    uint col = gid % out_w;
    if (t >= n_tok) return;

    uint my = t / mw;
    uint mx = t % mw;
    uint sub = col / D;
    uint c = col % D;
    uint dy = sub / merge;
    uint dx = sub % merge;
    uint sy = my * merge + dy;
    uint sx = mx * merge + dx;
    uint src_r = 1u + sy * p.kv_heads + sx;
    merged_buf[p.y + ulong(t) * p.ys + col] = seq_buf[p.x + ulong(src_r) * p.xs + c];
}

// ---- Scatter visual embeddings into LLM arena after embed_norm --------------
// Grid: rows * n threads.
// row_map[r] < 0 => leave arena untouched; row_map[r] >= 0 => copy vis_buf[row_map[r]].
kernel void mk_scatter_embeds(device float *arena [[buffer(0)]],
                              device const float *vis_buf [[buffer(1)]],
                              device const int *row_map [[buffer(2)]],
                              constant MK &p [[buffer(8)]],
                              uint gid [[thread_position_in_grid]]) {
    uint r = gid / p.n;
    uint c = gid % p.n;
    if (r >= p.rows) return;
    int src = row_map[r];
    if (src < 0) return;
    float v = vis_buf[p.x + ulong(uint(src)) * p.xs + c];
    if (p.flags & MK_ROUND_OUT) v = mk_bf16(v);
    arena[p.y + ulong(p.row0 + r) * p.ys + c] = v;
}
