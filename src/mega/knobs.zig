//! Run-time performance knobs. Each knob selects among compiled-in variants;
//! no value changes any token row's arithmetic (docs/QD.md admission checks that).
//! Defaults are the reference executor: one dispatch per op, no fusion.
const std = @import("std");

pub const Knobs = struct {
    /// Maximum token rows per forward (prefill chunk). Decode uses 1.
    batch: u32 = 64,
    /// Opt-in packed expert reduction in SIMD lanes (different summation order).
    expert_simd: bool = false,
    /// Share routed input loads across the packed gate and up projections.
    expert_gate_pair: bool = false,
    /// Reuse a packed input lane tile across four outputs (NVFP4 or MXFP8).
    expert_wide_dot: bool = false,
    /// Opt-in expert-major batched matmul (different summation order).
    expert_prefill: bool = false,
    /// Stage NVFP4 inputs and decoded blocks in BF16 (requires expert_prefill).
    expert_bf16_tiles: bool = false,
    /// Tile both packed formats, rounding dequantized weights to BF16 as in batched MLX.
    expert_mlx_scale: bool = false,
    /// Token rows whose routes are listed and multiplied together (expert_prefill):
    /// 0 keeps 512 with expert_mlx_scale and 128 without. Each of the 128 experts
    /// sees about a sixteenth of the rows, so 512 leaves most of its 32-row tiles
    /// partly empty; a few thousand rows fill them. Per-row arithmetic is unchanged.
    expert_tile_rows: u32 = 0,
    /// Wider packed-expert K tile for prefill. Requires expert_prefill.
    expert_wide_k: bool = false,
    /// With expert_mlx_scale: each expert's route list from 32 segments of the
    /// routes scanned at once (mk_expert_row_counts, mk_expert_row_fill) instead
    /// of one scan end to end. The same lists in the same order.
    expert_lists_split: bool = false,
    /// Parallel softmax over relative-bias decode partitions.
    relative_split: bool = false,
    /// Four simultaneous score dots per decode partition; requires relative_split.
    relative_split_4: bool = false,
    /// Split long relative-bias attention in prefill (requires relative_split).
    relative_split_prefill: bool = false,
    /// Single-row relative attention (head size 128, three query heads a KV head)
    /// in stripes of keys over 16-bit mirrors of the K and V caches, the query
    /// heads of a KV head sharing every load (inkling_long.metal). The mirrors
    /// hold the caches' bfloat16 values exactly; the sum is associated per
    /// stripe, so it is not bit-identical to the reference. Takes precedence
    /// over relative_split for single rows. With relative_tile_prefill, batched
    /// rows read the same mirrors: under 512 rows that kernel's tiles in stripes
    /// of keys, from 512 rows the staged tiles of inkling_tiles.metal.
    relative_stripes: bool = false,
    /// FP16 matrix fragments in batched relative attention (different arithmetic).
    relative_tile_prefill: bool = false,
    /// With relative_stripes and relative_tile_prefill, from 512 rows: the distance
    /// bias of a layer, which is rounded to bfloat16 and read only by that layer's
    /// staged-tile attention, is written and read as 16-bit words
    /// (mk_grouped_project_bf16x4_pack). The same values in half the bytes.
    relative_bias16: bool = false,
    /// With bf16_prefill_tile: a layer's distance bias through
    /// mk_grouped_project_bf16d8, two queries and eight distances a thread, each
    /// input loaded once instead of once per distance. The same FMA chains.
    relative_bias_d8: bool = false,
    /// Reuse BF16 weights and input lanes across a small dense tile.
    bf16_linear_tile: bool = false,
    /// Matrix tiles for large BF16 projections with BF16-valued inputs.
    bf16_prefill_matrix: bool = false,
    /// With bf16_prefill_matrix: the matrix tiles also take projections with as few
    /// as 64 outputs or 32 inputs (the relative, router and controller projections),
    /// which otherwise run one small tile kernel launch each. The sums are
    /// associated as the matrix tiles associate them, so the bits differ.
    bf16_matrix_small: bool = false,
    /// Fully unrolled BF16 prefill tile; zero selects the original tail-safe kernel.
    bf16_prefill_tile: u32 = 0,
    /// Vector loads for the BF16 decode tile (different summation order).
    bf16_decode_vec: bool = false,
    /// Fold an in-place BF16 cast into its producer without changing arithmetic.
    fuse_bf16_cast: bool = false,
    /// Inkling's short convolutions step the ring slot from tap to tap
    /// (mk_inkling_short_conv_ring, mk_inkling_dynamic_conv_ring) instead of taking
    /// a 64-bit remainder per tap. No arithmetic changes.
    conv_ring: bool = false,
    /// Pack scratch states whose lifetimes do not meet into the same arena words
    /// (exec.packScratch) instead of giving every state its own. No arithmetic
    /// changes; the arena shrinks by an order of magnitude on programs with many
    /// short-lived states, which is what lets Inkling take batches above 512.
    scratch_reuse: bool = false,
    /// Allow adjacent independent scratch kernels to overlap.
    concurrent_scratch: bool = false,
    /// Affine 4-bit linear: output rows and token rows per SIMD group,
    /// separately for single-row (decode) and multi-row (prefill) forwards.
    decode_rows: u32 = 2,
    decode_tokens: u32 = 1,
    /// Specialize the two-row decode tile for fixed K, group-64 and BF16 affine metadata.
    decode_q4_static: bool = false,
    prefill_rows: u32 = 2,
    prefill_tokens: u32 = 4,
    /// Threads per threadgroup for the 4-bit linear kernels.
    linear_group: u32 = 128,
    /// Fusions (src/mega/plan.zig). All off = the reference executor.
    fuse_projections: bool = false,
    fuse_glu: bool = false,
    fuse_norms: bool = false,
    fuse_add_norm: bool = false,
    fuse_conv: bool = false,
    fuse_gate: bool = false,
    // ---- decode search (decode objective) ----
    /// Single-row attention: 0 = mk_attention (reference); 2 = split into a
    /// parallel scores launch and the online-softmax chain with its max/exp
    /// computed 32 steps at a time (decode.metal). 1 (a per-step chain, the
    /// stepping stone) was removed; see benchmarks/lfm3-decode-search.md.
    /// 3 = the same split with specialized append-only 64/128-wide GQA.
    decode_attention: u32 = 0,
    /// q/k per-head norms, both ropes and the K/V stores of an attention layer in one launch.
    fuse_attn_prep: bool = false,
    /// Argmax: 0 = mk_argmax (one threadgroup per row), 1 = two-stage (decode.metal).
    decode_argmax: u32 = 0,

    // ---- prefill objective (prefill search) ----
    /// Multi-row affine 4-bit linears (plain, GLU, projections) through
    /// mk_q4_prep plus the matrix-matrix kernels of src/mega/prefill.metal.
    /// Single-row forwards (decode, the head) are unchanged.
    prefill_mm: bool = false,
    /// Output rows and token rows per SIMD group of the matrix-matrix kernels,
    /// and whether unpacked codes are held as half (exact) or float. Only the
    /// combinations in `mm_tiles` are compiled (the prefill-mm elite's; the
    /// climb's other tiles were removed, benchmarks/lfm3-prefill-search.md).
    mm_rows: u32 = 2,
    mm_tokens: u32 = 8,
    mm_half: bool = true,
    /// Threads per threadgroup of the matrix-matrix kernels.
    mm_group: u32 = 128,

    // ---- prefill-mlx (exact) ----
    /// Multi-row forwards run the fused gated conv as two launches
    /// (mk_gated_conv_store, mk_gated_conv_rows: every row's product stored,
    /// then every (row, channel) convolved in parallel) instead of one thread
    /// per channel walking the rows. Same arithmetic; needs fuse_conv.
    conv_rows: bool = false,
    /// Rows without logits (every row of a prefill forward but the last) stop after
    /// the program's last store to a persistent state (exec.trimmedHead): for LFM3,
    /// the last layer's conv output, out_proj and feed-forward run for the logits
    /// rows only, as in stock MLX's chunked prefill. Same arithmetic per row run.
    head_trim: bool = false,
    /// Row liveness (exec.rowNeeds). Nothing reads most rows of the last layer's
    /// attention output and feed-forward: the caches after it keep only a short
    /// convolution's last few rows, and the head reads the rows with logits. With a
    /// nonzero value every op runs on the rows something still reads, and an op that
    /// is trimmed keeps at least this many of the forward's last rows. Rows nothing
    /// reads are not computed. With the reference kernels a stored row holds the
    /// bits it would without the knob for any value. Knobs that pick a kernel by
    /// row count (matrix tiles, staged attention tiles) pick another one for a
    /// short tail, so with them the bits are kept only from 512, where a trimmed
    /// op still runs the kernels a full chunk does; 1 moved real logits by 0.1.
    tail_trim: u32 = 0,
    // ---- prefill-mlx (mlx-arith tier) ----
    /// NOT the reference arithmetic: multi-row affine 4-bit linears (group a
    /// multiple of 32, at least MLX's vector limit of rows) through the port of
    /// MLX's affine_qmm_t / qmm_t_splitk (mlx.metal), bit-identical to stock
    /// MLX's float32 mx.quantized_matmul at the same batch (tools/nano_mlx_ops.py).
    /// Admitted only in the mlx-arith tier (docs/QD.md); decode is unchanged.
    mlx_linear: bool = false,
    /// NOT the reference arithmetic: multi-row attention (more than 8 rows, head
    /// dim 128) through the port of MLX's steel attention (mlx.metal),
    /// bit-identical to stock MLX's float32 mx.fast.scaled_dot_product_attention
    /// over the same keys (tools/nano_mlx_ops.py). mlx-arith tier only.
    mlx_attention: bool = false,
    /// With mlx_linear and fuse_projections / fuse_glu: unsplit MLX-arithmetic
    /// products that share an input (q/k/v projections; w1 and w3) in one launch
    /// (mk_mlx_qmm_multi): every tile is mk_mlx_qmm_t's, so the bits are the same.
    mlx_multi: bool = false,
    // ---- c2-prefill-mlx (prefill, mlx-arith tier) ----
    /// NOT the reference arithmetic: with mlx_linear, the products stock MLX runs
    /// unsplit (plain and, with mlx_multi, multi-matrix) through mk_mlx2_qmm /
    /// mk_mlx2_multi (mlx2.metal): 64 x 16 tiles, x read straight from device memory,
    /// double-buffered dequantized weights. Each output is the same ascending-k fma
    /// chain as stock's affine_qmm_t, so the bits are stock MLX's
    /// (tools/nano_mlx_ops.py check --variants mlx,mlx2). mlx-arith tier only.
    mlx_qmm2: bool = false,
    /// With mlx_linear and fuse_glu: the gate and up products (both unsplit in stock
    /// MLX) and the GLU product in one launch (mk_mlx2_glu): equal bit for bit to
    /// mk_mlx_qmm_t twice then mk_binary. mlx-arith tier only.
    mlx_glu2: bool = false,
    /// NOT the reference arithmetic: with mlx_attention, multi-row attention through
    /// mk_mlx2_attention[_gated] (mlx2.metal): mk_mlx_attention's arithmetic step for
    /// step, scheduled heaviest query blocks first, queries held in registers, separate K
    /// and V tiles (two barriers per key block instead of four). Bits are stock MLX's
    /// (tools/nano_mlx_ops.py check --variants ..mlx2). mlx-arith tier only.
    mlx_attention2: bool = false,
    /// With conv_rows: mk_gated_conv_rows2 (mlx2.metal), the same arithmetic with the
    /// ring slot advanced per tap instead of a 64-bit modulo per tap. Exact tier.
    conv_rows2: bool = false,

    // ---- decode climb 2 (c2-decode, decode objective) ----
    /// Gated pre-encoded decode (exec.d2GatedForward): while token k runs, the
    /// forward for position k + 1 is encoded and committed behind a GPU gate
    /// (mk_d2_gate_packed) that the host opens only after it has read token k
    /// back and written token k + 1; the host still reads every token before
    /// the next forward starts. Same kernels and arithmetic (the embedding reads
    /// its token from the gate word, mk_d2_embed_gated). Single-row forwards.
    d2_gate: bool = false,
    /// Gate spin bound (iterations of one atomic load). A gate that gives up
    /// before the host opens it costs a recomputation, never a wrong token.
    d2_gate_spins: u32 = 200000,
    /// Single-row forwards: the fused sandwich norm with its operands held in
    /// registers (mk_d2_norm_add_norm) where n <= 4 x its threadgroup.
    d2_norm_regs: bool = false,

    // ---- climb 2, exact prefill (exact tier) ----
    /// Multi-row affine 4-bit linears (plain; projections and GLU run as their
    /// separate linears and the reference binary op) through mk_x2_prep and the
    /// matrix-unit kernel mk_x2mm_q4 (exact2.metal): each 4-input quad chain of
    /// the reference tree is one zero-padded simdgroup 8 x 8 x 8 step, the rest is
    /// the reference's element-wise arithmetic. Bit-identical to the reference.
    x2_linear: bool = false,
    /// Multi-row attention (head dim 97..128, 4 query heads per KV head) as three
    /// launches (exact2.metal): every visible score with the reference's dot tree,
    /// the online-softmax scalars per (row, head, SIMD class) in step order, then
    /// the value recurrences and the class merge. Bit-identical to mk_attention.
    x2_attention: bool = false,
    /// With x2_linear: the unpacked codes staged as half (exact for 4-bit masked
    /// codes), halving the threadgroup traffic of the weight operand.
    x2_half: bool = false,

    // ---- Inkling optimizations ----
    /// Fuse controller forget/gain projections and message merge in single-row decode.
    fuse_controller_merge: bool = false,
    /// Fuse up to 6 consecutive dense BF16 linears sharing one input into a single launch.
    fuse_bf16_projections: bool = false,
    /// Split-K BF16 decode matvec (8 SIMDs/row) for small-M dense projections.
    bf16_decode_splitk: bool = false,
    /// 128-bit weight loads (`ushort4` x 2) for single-row BF16 decode linears when aligned.
    bf16_decode_wide: bool = false,
    /// Fuse final vocabulary projection, BF16 cast and tanh softcap into one launch.
    fuse_head_softcap: bool = false,
    /// Vectorized register-resident depth attention (decode and prefill).
    depth_attention_fast: bool = false,
    /// Fuse depth attention + following RMSNorm(+BF16 cast) in controllers and visit blocks.
    fuse_depth_norm: bool = false,
    /// Single-read, two-barrier vectorized RMSNorm for n <= 4 * threadgroup.
    rmsnorm_fast: bool = false,
    /// Combine KV f32 cache store + BF16 mirror pack when relative_stripes is active.
    store_pack_bf16: bool = false,
    /// Fuse short-conv ring store + single-row mk_inkling_short_conv_ring.
    fuse_inkling_conv: bool = false,
    /// 8-distance-per-thread BF16 grouped projection for single-row relative bias.
    relative_bias_decode: bool = false,
    /// Sub-SIMD branchless NVFP4 and vectorized MXFP8 single-row decode experts.
    expert_nvfp4_fast: bool = false,
    /// Single-pass online softmax striped decode attention (log2 units, small TG memory).
    relative_long_fast: bool = false,
    /// Fuse relative attention + BF16 cast + sigmoid gate + BF16 cast in single-row decode.
    fuse_relative_gate: bool = false,
    /// Half-precision staged relative prefill tiles with reversed query schedule and Lean far-band split.
    relative_prefill_fast: bool = false,
    /// M=64, N=64, B=32 expert prefill tiles with SLC-resident BF16 input packing.
    expert_block_m64n64: bool = false,
    /// Constant-cache FP8 E4M3FN and FP4 pair LUTs for expert dequantization.
    expert_fp_lut: bool = false,
    /// Single-TG compact MoE tile builder and M=16/32/64 sparse expert tiles.
    expert_compact_tiles: bool = false,
    /// Packed BF16 intermediates for expert_gate_up -> expert_down -> combine.
    expert_bf16_io: bool = false,
    /// Fuse sliding sconv / global QKV + relative prep chains (sconv, rmsnorm, l2_clip, log_scale, kv store+pack).
    fuse_qkvr_prep: bool = false,
    /// Fuse single-row relative bias projection directly into striped attention key/value pass.
    relative_inline_bias: bool = false,
    /// Fuse controller message merge + following RMSNorm(+BF16 cast) on batched forwards.
    fuse_merge_norm: bool = false,

    /// (rows, tokens, half) of every compiled mk_mm_q4 / mk_mm_glu_q4 variant.
    pub const mm_tiles = [_]struct { u32, u32, bool }{.{ 2, 8, true }};

    pub fn fusion(self: Knobs) @import("plan.zig").Fusion {
        return .{
            .projections = self.fuse_projections,
            .glu = self.fuse_glu,
            .norms = self.fuse_norms,
            .add_norm = self.fuse_add_norm,
            .conv = self.fuse_conv,
            .gate = self.fuse_gate,
            .attn_prep = self.fuse_attn_prep,
            .bf16_cast = self.fuse_bf16_cast,
            .controller_merge = self.fuse_controller_merge,
            .bf16_projections = self.fuse_bf16_projections,
            .head_softcap = self.fuse_head_softcap,
            .depth_norm = self.fuse_depth_norm,
            .inkling_conv = self.fuse_inkling_conv,
            .relative_gate = self.fuse_relative_gate,
            .qkvr_prep = self.fuse_qkvr_prep,
            .merge_norm = self.fuse_merge_norm,
        };
    }

    pub fn validate(self: Knobs) !void {
        if (self.bf16_prefill_tile != 0 and self.bf16_prefill_tile != 8 and self.bf16_prefill_tile != 16 and self.bf16_prefill_tile != 32) return error.InvalidBf16PrefillTile;
        if (self.batch == 0 or self.batch > 4096) return error.InvalidBatch;
        if (self.expert_tile_rows != 0 and (self.expert_tile_rows < 8 or self.expert_tile_rows > 4096)) return error.InvalidExpertTileRows;
        for ([_]u32{ self.decode_rows, self.prefill_rows }) |r| if (r != 1 and r != 2 and r != 4 and r != 8) return error.InvalidLinearRows;
        for ([_]u32{ self.decode_tokens, self.prefill_tokens }) |t| if (t != 1 and t != 2 and t != 4) return error.InvalidLinearTokens;
        if (self.linear_group == 0 or self.linear_group % 32 != 0 or self.linear_group > 1024) return error.InvalidLinearGroup;
        // prefill objective (prefill search)
        var tile_ok = false;
        for (mm_tiles) |t| tile_ok = tile_ok or (t[0] == self.mm_rows and t[1] == self.mm_tokens and t[2] == self.mm_half);
        if (!tile_ok) return error.InvalidMmTile;
        if (self.mm_group == 0 or self.mm_group % 32 != 0 or self.mm_group > 1024) return error.InvalidMmGroup;
        if (self.decode_attention != 0 and self.decode_attention != 2 and self.decode_attention != 3) return error.InvalidDecodeAttention;
        if (self.decode_argmax > 1) return error.InvalidDecodeArgmax;
    }

    /// Apply `name=value` (flag form) to the knob of that name.
    pub fn set(self: *Knobs, name: []const u8, value: []const u8) !void {
        const info = @typeInfo(Knobs).@"struct";
        inline for (info.field_names, info.field_types) |field, T| {
            if (std.mem.eql(u8, field, name)) {
                switch (@typeInfo(T)) {
                    .int => @field(self, field) = try std.fmt.parseInt(T, value, 10),
                    .bool => @field(self, field) = std.mem.eql(u8, value, "1") or std.mem.eql(u8, value, "true"),
                    .@"enum" => @field(self, field) = std.meta.stringToEnum(T, value) orelse return error.InvalidKnobValue,
                    else => @compileError("unsupported knob type"),
                }
                return;
            }
        }
        return error.UnknownKnob;
    }

    /// A JSON object of knob values; unknown names fail closed.
    pub fn parseJson(a: std.mem.Allocator, bytes: []const u8) !Knobs {
        const parsed = try std.json.parseFromSliceLeaky(Knobs, a, bytes, .{});
        try parsed.validate();
        return parsed;
    }
};

test "knob flags and JSON configuration select variants and fail closed" {
    var k: Knobs = .{};
    try k.set("decode_rows", "4");
    try k.set("fuse_glu", "true");
    try std.testing.expect(k.fusion().glu and !k.fusion().norms);
    try std.testing.expectEqual(@as(u32, 4), k.decode_rows);
    try std.testing.expectError(error.UnknownKnob, k.set("typo", "1"));
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const j = try Knobs.parseJson(arena.allocator(), "{\"batch\":16,\"prefill_tokens\":2}");
    try std.testing.expectEqual(@as(u32, 16), j.batch);
    try std.testing.expectEqual(@as(u32, 2), j.prefill_tokens);
    try std.testing.expectError(error.UnknownField, Knobs.parseJson(arena.allocator(), "{\"nope\":1}"));
    try std.testing.expectError(error.InvalidLinearRows, Knobs.parseJson(arena.allocator(), "{\"decode_rows\":3}"));
}
