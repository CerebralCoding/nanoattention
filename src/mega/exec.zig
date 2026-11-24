//! Binds a Program to a checkpoint and Metal buffers, and executes forwards of
//! T consecutive tokens as one command buffer. Model-agnostic: everything comes
//! from the program's weight table, state table and op list.
const std = @import("std");
const ir = @import("ir.zig");
const gpu = @import("gpu.zig");
const knobs_mod = @import("knobs.zig");
const plan = @import("plan.zig");
const weights = @import("weights");
const metal = weights.metal;
const Knobs = knobs_mod.Knobs;
const io = @import("io.zig"); // decode climb 2
// climb 2, exact prefill
const exact2 = @import("exact2.zig");
const inkling_plan = @import("inkling_plan.zig");

pub fn bindWeight(store: *const weights.Store, spec: ir.Weight) !weights.Weight {
    if (spec.raw_dtype) |dtype| {
        const raw_dtype: weights.st.DType = switch (dtype) {
            .u8 => .u8,
            .f8_e4m3 => .f8_e4m3,
            .f32 => .f32,
        };
        return store.getRaw(spec.name, spec.shape, raw_dtype);
    }
    return store.get(spec.name, spec.shape);
}

pub const Trace = struct {
    /// [traces][rows][width] float32, filled by forwardTraced.
    data: []f32,
    width: u32,
};

/// prefill objective (prefill search): which input rows `Engine.prep` holds.
pub const PrepKey = struct { state: u32, col: u32, width: u32, row0: u32, rows: u32 };

pub const Stats = struct {
    dispatches: u64 = 0,
    submissions: u64 = 0,
    gpu_seconds: f64 = 0,
    // decode climb 2: host seconds spent encoding, and from finish() to completion.
    encode_seconds: f64 = 0,
    wait_seconds: f64 = 0,
};

/// What `Engine.tap` collects, in the order the forwards ran: per forward, per
/// attention visit, per row, `heads` floats (the L2 norm of each head's attention
/// output over its `dim` values, before the output gate); and per forward, per
/// router (one a visit), per row, `top_k` expert ids and then their `top_k` route
/// weights. `windows` has each visit's sliding window (0 = global), from the first forward.
pub const Tap = struct {
    gpa: std.mem.Allocator,
    heads: u32 = 0,
    top_k: u32 = 0,
    forwards: u32 = 0,
    norms: std.ArrayList(f32) = .empty,
    routes: std.ArrayList(f32) = .empty,
    windows: std.ArrayList(u32) = .empty,
    /// `nano eval --attention-out`: also record, per forward, attention visit and row,
    /// `context` floats (the weight each key position gets, averaged over the heads; zero
    /// where a key is not visible) in `weights`, and the average share the sink takes in
    /// `sinks`. Recomputed on the host from the engine's own queries, bias, sink and
    /// cached keys; `worst` is the largest difference between the host's weighted sum of
    /// values and the engine's output over everything recorded.
    attention: bool = false,
    context: u32 = 0,
    weights: std.ArrayList(f32) = .empty,
    sinks: std.ArrayList(f32) = .empty,
    scratch: std.ArrayList(f64) = .empty,
    worst: f32 = 0,

    pub fn deinit(self: *Tap) void {
        self.weights.deinit(self.gpa);
        self.sinks.deinit(self.gpa);
        self.scratch.deinit(self.gpa);
        self.norms.deinit(self.gpa);
        self.routes.deinit(self.gpa);
        self.windows.deinit(self.gpa);
    }
};

pub const Engine = struct {
    gpa: std.mem.Allocator,
    program: *const ir.Program,
    ctx: metal.Context,
    store: *const weights.Store,
    knobs: Knobs,
    batch: u32,
    context: u32,
    /// All tokens/scratch states share one buffer; offsets in words.
    arena: metal.Buffer,
    offsets: []u32,
    /// Persistent states (sequence/ring), one buffer each.
    caches: []?metal.Buffer,
    tables: metal.Buffer,
    table_offsets: []u32,
    bound: []weights.Weight,
    /// Launch schedule from the fusion knobs; one step per op for the reference.
    steps: []plan.Step,
    independent_launches: ?[]bool = null,
    /// Kept only across adjacent expert operations, with no intervening writes.
    expert_routes: ?struct { off: u32, stride: u32, row0: u32, rows: u32, experts: u32, top_k: u32 } = null,
    /// Ops at index >= head_start run only on rows with logits: the program's
    /// head_start, or earlier with knobs.head_trim (traced forwards use the program's).
    head_start: u32,
    /// knobs.tail_trim: per step, the rows of a forward something still reads.
    row_needs: ?[]RowNeed = null,
    /// Diagnostics (`nano eval --heads-out`): while set, every forward also records what
    /// each attention visit and each router did. Null in normal runs; no arithmetic changes.
    tap: ?*Tap = null,
    /// knobs.relative_bias16: the distance-bias states one grouped projection writes
    /// and one staged-tile attention reads, and the rows of the one written 16-bit
    /// in this forward until its attention has read them.
    bias16_ok: ?[]bool = null,
    bias16: ?struct { state: u32, row0: u32, rows: u32 } = null,
    position: u32 = 0,
    failed: bool = false,
    last: Stats = .{},
    /// Rows [out_first, T) of the last forward have logits.
    out_first: u32 = 0,
    out_end: u32 = 0,
    // prefill objective (prefill search): prepared inputs of the matrix-matrix
    // linears (mk_q4_prep layout, prefill.metal); null unless knobs.prefill_mm.
    prep: ?metal.Buffer = null,
    /// Word offset of the xsum region in `prep`.
    prep_xsum: u32 = 0,
    /// The input rows `prep` holds now; cleared by any write that could change them.
    prep_key: ?PrepKey = null,
    // ---- decode search ----
    /// Split decode attention: per-head scores of one row, `scores_stride` words per head.
    scores: ?metal.Buffer = null,
    scores_stride: u32 = 0,
    /// Two-stage argmax partials: batch x argmax_groups x 3 words.
    partials: ?metal.Buffer = null,
    // ---- prefill-mlx ----
    /// Split-K partial sums of the MLX-arithmetic linears; null unless knobs.mlx_linear.
    splitk: ?metal.Buffer = null,
    // ---- decode climb 2 ----
    /// Knob d2_gate: gate words (0: gen << 20 | token, 1: stale-generation flag,
    /// 2: gate spins), the next generation, and the position whose forward is
    /// committed and waiting at the gate (with its generation).
    d2_gate_buf: ?metal.Buffer = null,
    d2_gen: u32 = 0,
    d2_ahead: ?u32 = null,
    d2_ahead_gen: u32 = 0,
    d2_ahead_dispatches: u64 = 0,
    /// Gated forwards recomputed synchronously because the gate gave up first.
    d2_redos: u64 = 0,
    /// The argmax word of the last gated forward, read once when it was written
    /// (a later forward may reuse the arena word); null after any other forward.
    d2_last: ?u32 = null,
    // ---- climb 2, exact prefill ----
    /// Prepared inputs of the matrix-unit linears (mk_x2_prep layout); null unless knobs.x2_linear.
    x2prep: ?metal.Buffer = null,
    /// The input rows `x2prep` holds now; cleared by any write that could change them.
    x2_key: ?PrepKey = null,
    /// Batched exact attention scratch (knob x2_attention): scores/es and f, rows x
    /// heads x x2att_stride words each, and (m, l) per (row, head, class), for
    /// x2att_rows rows at a time.
    x2att: ?[3]metal.Buffer = null,
    x2att_stride: u32 = 0,
    x2att_rows: u32 = 0,
    expert_tiles: ?[2]metal.Buffer = null,
    relative_parts: ?metal.Buffer = null,
    /// knobs.relative_stripes: per state, the 16-bit mirror of a K or V cache the
    /// striped attention reads (written with every store), and its stripe records.
    relative_half: ?[]?metal.Buffer = null,
    stripe_parts: ?metal.Buffer = null,
    /// Stripe records of the batched striped attention: batch x heads x max_prefill_stripes.
    prefill_parts: ?metal.Buffer = null,
    const max_prefill_stripes: u32 = 4;
    const list_segments: u32 = 32; // MK_LIST_SEGS
    /// Rows from which a chunk runs mk_relative_prefill_tiles instead of stripes.
    const min_tile_rows: u32 = 512;
    const mirror_pad: u32 = 16; // MK_MIRROR_PAD
    const max_stripe_parts: u32 = 1024; // MK_LONG_PARTS
    const stripe_threads: usize = 512; // 16 SIMD groups (MK_LONG_SIMDS)
    const max_relative_parts: u32 = 64;
    const min_relative_split: u32 = 64;
    const relative_row_tile: u32 = 16;

    pub fn init(gpa: std.mem.Allocator, program: *const ir.Program, ctx: metal.Context, store: *const weights.Store, knobs: Knobs, context: u32) !Engine {
        try knobs.validate();
        if (context == 0 or context > program.meta.max_context) return error.InvalidContextLength;
        const batch = @min(knobs.batch, context);
        const n = program.states.len;
        const offsets = try gpa.alloc(u32, n);
        errdefer gpa.free(offsets);
        const caches = try gpa.alloc(?metal.Buffer, n);
        @memset(caches, null);
        errdefer {
            for (caches) |c| if (c) |b| b.deinit();
            gpa.free(caches);
        }
        var total: usize = 0;
        const packed_scratch = knobs.scratch_reuse;
        if (packed_scratch) total = try packScratch(gpa, program, batch, context, offsets);
        for (program.states, 0..) |s, i| {
            if (!packed_scratch) offsets[i] = 0;
            switch (s.kind) {
                .tokens, .scratch => if (!packed_scratch) {
                    offsets[i] = std.math.cast(u32, total) orelse return error.ArenaTooLarge;
                    total += @as(usize, s.capacity(batch, context)) * s.width;
                    total = (total + 63) / 64 * 64;
                },
                .sequence, .ring => {
                    const buf = try ctx.alloc(@as(usize, s.capacity(batch, context)) * s.width);
                    @memset(buf.floats(), 0);
                    caches[i] = buf;
                },
                .position => {},
            }
        }
        if (total > std.math.maxInt(u32)) return error.ArenaTooLarge;
        const arena = try ctx.alloc(@max(total, 1));
        errdefer arena.deinit();
        @memset(arena.floats(), 0);
        const table_offsets = try gpa.alloc(u32, program.tables.len);
        errdefer gpa.free(table_offsets);
        var table_words: usize = 0;
        for (program.tables, 0..) |t, i| {
            table_offsets[i] = @intCast(table_words);
            table_words += t.len;
        }
        const tables = try ctx.alloc(@max(table_words, 1));
        errdefer tables.deinit();
        for (program.tables, 0..) |t, i| @memcpy(tables.floats()[table_offsets[i]..][0..t.len], t);
        const bound = try gpa.alloc(weights.Weight, program.weights.len);
        errdefer gpa.free(bound);
        for (program.weights, 0..) |w, i| bound[i] = try bindWeight(store, w);
        const info = try gpa.alloc(plan.WeightInfo, bound.len);
        defer gpa.free(info);
        for (bound, info) |w, *f| {
            const same = if (w.quant) |q| q.scales.handle == w.buffer.handle and q.biases.handle == w.buffer.handle else true;
            f.* = .{
                .q4 = w.tensor.shape.len == 2 and q4Fast(w, @intCast(w.tensor.shape[1])),
                .buffer = if (same) @intFromPtr(w.buffer.handle) else 0,
                .group = if (w.quant) |q| q.group_size else 0,
                .dense = w.quant == null and w.tensor.dtype.isFloat(),
            };
        }
        // prefill-mlx (knob head_trim): rows without logits stop after the last op that
        // writes a persistent state; the plan respects both boundaries.
        const head_start = if (knobs.head_trim) trimmedHead(program) else program.head_start;
        var planned = program.*;
        planned.head_start = head_start;
        var steps = try plan.build(gpa, &planned, knobs.fusion(), info);
        errdefer gpa.free(steps);
        if (head_start != program.head_start) steps = try splitAcross(gpa, steps, program.head_start);
        var row_needs: ?[]RowNeed = null;
        errdefer if (row_needs) |v| gpa.free(v);
        if (knobs.tail_trim != 0) {
            const per_op = try rowNeeds(gpa, program);
            defer gpa.free(per_op);
            const per_step = try gpa.alloc(RowNeed, steps.len);
            for (steps, per_step) |st, *need| {
                need.* = .{};
                for (per_op[st.first()..][0..st.len()]) |op_need| need.* = need.merge(op_need);
            }
            row_needs = per_step;
        }
        const independent_launches = if (knobs.concurrent_scratch) try plan.independentLaunches(gpa, steps, program, info) else null;
        errdefer if (independent_launches) |v| gpa.free(v);
        // prefill objective (prefill search): room for the widest q4 linear input.
        var prep: ?metal.Buffer = null;
        var prep_xsum: u32 = 0;
        if (knobs.prefill_mm) {
            var max_segs: usize = 0;
            for (program.ops) |op| switch (op) {
                .linear => |l| if (q4Fast(bound[l.w], l.x.width)) {
                    max_segs = @max(max_segs, l.x.width / 16);
                },
                else => {},
            };
            const padded = (max_segs + 31) / 32 * 512;
            prep_xsum = std.math.cast(u32, @as(usize, batch) * padded) orelse return error.ArenaTooLarge;
            prep = try ctx.alloc(@max(@as(usize, batch) * (padded + max_segs), 1));
        }
        errdefer if (prep) |b| b.deinit();
        // decode search: scores scratch for split decode attention (largest visible span x heads).
        var scores: ?metal.Buffer = null;
        var scores_stride: u32 = 0;
        if (knobs.decode_attention != 0) {
            var words_needed: usize = 0;
            for (program.ops) |op| switch (op) {
                .attention => |at| {
                    const cap = program.states[at.keys].capacity(batch, context);
                    const span = if (at.window != 0) @min(at.window, cap) else cap;
                    scores_stride = @max(scores_stride, span);
                    words_needed = @max(words_needed, @as(usize, at.heads));
                },
                else => {},
            };
            if (scores_stride != 0) scores = try ctx.alloc(words_needed * scores_stride);
        }
        errdefer if (scores) |b| b.deinit();
        const partials: ?metal.Buffer = if (knobs.decode_argmax != 0) try ctx.alloc(@as(usize, batch) * argmax_groups * 3) else null;
        errdefer if (partials) |b| b.deinit();
        const splitk: ?metal.Buffer = if (knobs.mlx_linear) try ctx.alloc(mlx_splitk_words) else null;
        errdefer if (splitk) |b| b.deinit();
        // decode climb 2 (knob d2_gate): the packed token word needs vocab <= d2_max_vocab.
        const d2_gate_buf: ?metal.Buffer = if (knobs.d2_gate and program.meta.vocab <= inkling_plan.d2_max_vocab and d2GateSafe(program, batch, context)) try ctx.alloc(16) else null;
        errdefer if (d2_gate_buf) |b| b.deinit();
        if (d2_gate_buf) |b| @memset(b.structs(u32), 0);

        // climb 2, exact prefill: room for the widest q4 linear input.
        var x2prep: ?metal.Buffer = null;
        if (knobs.x2_linear) {
            var max_n: u32 = 16;
            for (program.ops) |op| switch (op) {
                .linear => |l| if (q4Fast(bound[l.w], l.x.width)) {
                    max_n = @max(max_n, l.x.width);
                },
                else => {},
            };
            x2prep = try ctx.alloc(exact2.prepLayout(batch, max_n).words);
        }
        errdefer if (x2prep) |b| b.deinit();
        var x2att: ?[3]metal.Buffer = null;
        var x2att_stride: u32 = 0;
        var x2att_rows: u32 = 0;
        if (knobs.x2_attention) {
            var heads: u32 = 0;
            for (program.ops) |op| switch (op) {
                .attention => |at| {
                    const cap = program.states[at.keys].capacity(batch, context);
                    x2att_stride = @max(x2att_stride, if (at.window != 0) @min(at.window, cap) else cap);
                    heads = @max(heads, at.heads);
                },
                else => {},
            };
            if (x2att_stride != 0) {
                // At most 32 M words per score array; rows beyond that run in chunks.
                x2att_rows = @intCast(@max(1, @min(@as(usize, batch), (32 << 20) / (@as(usize, heads) * x2att_stride))));
                const nwords = @as(usize, x2att_rows) * heads * x2att_stride;
                const s0 = try ctx.alloc(nwords);
                errdefer s0.deinit();
                const s1 = try ctx.alloc(nwords);
                errdefer s1.deinit();
                x2att = .{ s0, s1, try ctx.alloc(@as(usize, x2att_rows) * heads * 16) };
            }
        }
        errdefer if (x2att) |bs| for (bs) |b| b.deinit();
        var expert_tiles: ?[2]metal.Buffer = null;
        if (knobs.expert_prefill and batch >= 8) {
            const tile_rows = @min(batch, expertTileRows(knobs));
            var lists: usize = 0;
            var values: usize = 0;
            for (program.ops) |op| switch (op) {
                // After the lists: 32 segment counts an expert (knobs.expert_lists_split, MK_LIST_SEGS).
                .expert_gate_up => |e| lists = @max(lists, @as(usize, e.experts) * (tile_rows * e.top_k + 1 + list_segments)),
                .expert_down => |e| {
                    lists = @max(lists, @as(usize, e.experts) * (tile_rows * e.top_k + 1 + list_segments));
                    values = @max(values, @as(usize, e.top_k) * e.out.width * tile_rows);
                },
                else => {},
            };
            if (lists > 0) {
                const index = try ctx.alloc(lists);
                errdefer index.deinit();
                expert_tiles = .{ index, try ctx.alloc(@max(values, 1)) };
            }
        }
        errdefer if (expert_tiles) |bs| for (bs) |b| b.deinit();
        var relative_parts: ?metal.Buffer = null;
        if (knobs.relative_split and context >= min_relative_split) {
            var part_words: usize = 0;
            for (program.ops) |op| switch (op) {
                .relative_attention => |a| {
                    const nrows = if (knobs.relative_split_prefill) @min(batch, relative_row_tile) else 1;
                    part_words = @max(part_words, @as(usize, a.heads) * (a.dim + 2) * max_relative_parts * nrows);
                },
                else => {},
            };
            if (part_words != 0) relative_parts = try ctx.alloc(part_words);
        }
        var relative_half: ?[]?metal.Buffer = null;
        var stripe_parts: ?metal.Buffer = null;
        var prefill_parts: ?metal.Buffer = null;
        errdefer {
            if (prefill_parts) |b| b.deinit();
            if (relative_half) |hs| {
                for (hs) |h| if (h) |b| b.deinit();
                gpa.free(hs);
            }
            if (stripe_parts) |b| b.deinit();
        }
        if (knobs.relative_stripes) {
            const halves = try gpa.alloc(?metal.Buffer, n);
            @memset(halves, null);
            relative_half = halves;
            var part_words: usize = 0;
            var striped_heads: usize = 0;
            for (program.ops) |op| switch (op) {
                .relative_attention => |a| if (a.dim == 128 and a.heads == 3 * a.kv_heads) {
                    striped_heads = @max(striped_heads, a.heads);
                    for ([_]ir.StateId{ a.keys, a.values }) |id| if (halves[id] == null) {
                        const s = program.states[id];
                        const buf = try ctx.alloc((@as(usize, s.capacity(batch, context) + mirror_pad) * s.width + 1) / 2);
                        @memset(buf.floats(), 0);
                        halves[id] = buf;
                    };
                    const visible = if (a.window == 0) context else @min(context, a.window);
                    part_words = @max(part_words, @as(usize, a.heads) * 130 * @min(max_stripe_parts, maxStripeParts(visible, knobs.relative_long_fast)));
                },
                else => {},
            };
            if (part_words != 0) {
                stripe_parts = try ctx.alloc(part_words);
                if (knobs.relative_tile_prefill and batch >= 8) prefill_parts = try ctx.alloc(@as(usize, @min(batch, min_tile_rows)) * striped_heads * max_prefill_stripes * 130);
            }
        }
        // knobs.relative_bias16: a bias state qualifies when exactly two ops touch it,
        // the projection that writes it and an attention the staged tiles can run.
        var bias16_ok: ?[]bool = null;
        errdefer if (bias16_ok) |v| gpa.free(v);
        if (knobs.relative_bias16 and knobs.relative_tile_prefill) if (relative_half) |hs| {
            const ok = try gpa.alloc(bool, n);
            @memset(ok, false);
            bias16_ok = ok;
            const users = try gpa.alloc(u32, n);
            defer gpa.free(users);
            @memset(users, 0);
            for (program.ops) |op| {
                var reads: [3]?ir.View = undefined;
                var write: ?ir.View = null;
                plan.access(op, &reads, &write);
                for ([_]?ir.View{ reads[0], reads[1], reads[2], write }) |view| if (view) |v| {
                    users[v.state] += 1;
                };
            }
            for (program.traces) |t| users[t.view.state] += 1;
            for (program.ops) |op| switch (op) {
                .relative_attention => |a| if (a.dim == 128 and hs[a.keys] != null and hs[a.values] != null and users[a.bias.state] == 2) {
                    ok[a.bias.state] = true;
                },
                else => {},
            };
        };
        return .{
            .gpa = gpa,
            .program = program,
            .ctx = ctx,
            .store = store,
            .knobs = knobs,
            .batch = batch,
            .context = context,
            .arena = arena,
            .offsets = offsets,
            .caches = caches,
            .tables = tables,
            .table_offsets = table_offsets,
            .bound = bound,
            .steps = steps,
            .independent_launches = independent_launches,
            .head_start = head_start,
            .row_needs = row_needs,
            .bias16_ok = bias16_ok,
            .prep = prep,
            .prep_xsum = prep_xsum,
            .scores = scores,
            .scores_stride = scores_stride,
            .partials = partials,
            .splitk = splitk,
            .d2_gate_buf = d2_gate_buf,
            .x2prep = x2prep,
            .x2att = x2att,
            .x2att_stride = x2att_stride,
            .x2att_rows = x2att_rows,
            .expert_tiles = expert_tiles,
            .relative_parts = relative_parts,
            .relative_half = relative_half,
            .stripe_parts = stripe_parts,
            .prefill_parts = prefill_parts,
        };
    }

    /// Stripes of the single-row striped attention over `visible` keys: 128 keys
    /// a stripe up to 2,048 keys, 256 up to 16,384, then 512 (`nano attnbench`).
    fn stripeParts(visible: u32, fast: bool) u32 {
        const stripe: u32 = if (fast and visible <= 2048) 64 else if (visible <= 2048) 128 else if (visible <= 16384) 256 else 512;
        return (visible + stripe - 1) / stripe;
    }
    /// The most stripes any key count up to `visible` takes. The count drops where
    /// the stripe widens, so the last count is not the largest: 16,384 keys take
    /// 64 stripes and 16,385 take 33.
    fn maxStripeParts(visible: u32, fast: bool) u32 {
        return @max(stripeParts(visible, fast), @max(stripeParts(@min(visible, 2048), fast), stripeParts(@min(visible, 16384), fast)));
    }

    pub fn deinit(self: *Engine) void {
        self.d2Settle() catch {}; // decode climb 2: no committed work may outlive the buffers
        if (self.d2_redos != 0) std.debug.print("nano: d2_gate recomputed {d} token(s) whose gate gave up first\n", .{self.d2_redos});
        if (self.d2_gate_buf) |b| b.deinit();
        self.arena.deinit();
        self.tables.deinit();
        for (self.caches) |c| if (c) |b| b.deinit();
        self.gpa.free(self.caches);
        self.gpa.free(self.offsets);
        self.gpa.free(self.table_offsets);
        self.gpa.free(self.bound);
        self.gpa.free(self.steps);
        if (self.independent_launches) |v| self.gpa.free(v);
        if (self.row_needs) |v| self.gpa.free(v);
        if (self.bias16_ok) |v| self.gpa.free(v);
        if (self.prep) |b| b.deinit();
        if (self.scores) |b| b.deinit();
        if (self.partials) |b| b.deinit();
        if (self.splitk) |b| b.deinit();
        if (self.x2prep) |b| b.deinit();
        if (self.x2att) |bs| for (bs) |b| b.deinit();
        if (self.expert_tiles) |bs| for (bs) |b| b.deinit();
        if (self.relative_parts) |b| b.deinit();
        if (self.relative_half) |hs| {
            for (hs) |h| if (h) |b| b.deinit();
            self.gpa.free(hs);
        }
        if (self.stripe_parts) |b| b.deinit();
        if (self.prefill_parts) |b| b.deinit();
    }

    /// Logical reset: positions >= 0 are rewritten before any read, and conv
    /// reads of negative positions are zeros, so stale rows are never observed.
    pub fn reset(self: *Engine) void {
        self.d2Settle() catch {
            self.failed = true;
        };
        self.d2_last = null;
        self.position = 0;
        self.failed = false;
        self.out_first = 0;
        self.out_end = 0;
    }

    /// decode climb 2 (knob d2_gate): open a gate still waiting (its forward runs on
    /// the last token and writes only state at a position not yet committed, which
    /// the next forward there rewrites before reading), then wait for every
    /// committed buffer.
    pub fn d2Settle(self: *Engine) !void {
        if (self.d2_gate_buf) |g| if (self.d2_ahead != null) {
            const gw = g.structs(u32);
            @atomicStore(u32, &gw[0], self.d2_ahead_gen << @intCast(inkling_plan.d2_gen_shift) | (gw[0] & inkling_plan.d2_token_mask), .release);
            self.d2_ahead = null;
        };
        if (gpu.pending(self.ctx) != 0) try gpu.drain(self.ctx, 0);
    }

    fn words(self: *const Engine, v: ir.View) struct { off: u32, stride: u32 } {
        const s = self.program.states[v.state];
        return .{ .off = self.offsets[v.state] + v.col, .stride = s.width };
    }

    /// Bytes of persistent state: allocated, and holding information now.
    pub fn stateBytes(self: *const Engine, position: u32) struct { allocated: u64, valid: u64 } {
        var allocated: u64 = 0;
        var valid: u64 = 0;
        for (self.program.states) |s| if (s.persistent()) {
            allocated += @as(u64, s.capacity(self.batch, self.context)) * s.width * 4;
            valid += @as(u64, s.validRows(self.batch, self.context, position)) * s.width * 4;
        };
        return .{ .allocated = allocated, .valid = valid };
    }

    /// Run tokens.len (<= batch) consecutive positions. Head ops (final norm,
    /// vocabulary projection, argmax) run only for rows [out_first, tokens.len).
    pub fn forward(self: *Engine, tokens: []const u32, out_first: u32) !void {
        return self.forwardTraced(tokens, out_first, null);
    }

    pub fn forwardTraced(self: *Engine, tokens: []const u32, out_first: u32, trace: ?Trace) !void {
        const T: u32 = @intCast(tokens.len);
        if (self.failed) return error.EngineNeedsReset;
        if (T == 0 or T > self.batch) return error.InvalidBatchLength;
        if (out_first > T) return error.InvalidOutputRows;
        if (@as(u64, self.position) + T > self.context) return error.ContextExhausted;
        for (tokens) |t| if (t >= self.program.meta.vocab) return error.InvalidToken;
        if (trace) |t| if (t.data.len < self.program.traces.len * @as(usize, T) * t.width) return error.TraceTooSmall;
        errdefer self.failed = true;
        // decode climb 2 (knob d2_gate): single-row forwards with logits go gated;
        // anything else first settles a forward waiting at the gate.
        if (self.d2_gate_buf != null) {
            if (T == 1 and out_first == 0 and trace == null and self.tap == null) return d2GatedForward(self, tokens[0]);
            try self.d2Settle();
            self.d2_last = null;
        }
        const tok = self.words(self.program.tokens);
        const words_ = self.arena.structs(u32);
        for (tokens, 0..) |t, i| words_[tok.off + i * tok.stride] = t;
        const before = self.ctx.stats();
        const t_begin = io.now() catch 0; // decode climb 2
        var next_trace: usize = 0;
        self.prep_key = null;
        self.x2_key = null; // climb 2, exact prefill
        self.expert_routes = null;
        self.bias16 = null;
        var skipped_barrier = false;
        for (self.steps, 0..) |step, step_i| {
            const first = step.first();
            const head = first >= (if (trace != null) self.program.head_start else self.head_start);
            if (head and out_first == T) break;
            var row0: u32 = if (head) out_first else 0;
            // knobs.tail_trim: a step runs on the rows something reads (traced forwards on all).
            if (!head and trace == null and self.tap == null) if (self.row_needs) |needs| {
                row0 = needs[step_i].firstRow(T, out_first, self.knobs.tail_trim);
                if (row0 == T) {
                    // A skipped step that would have ended a run of overlapping launches
                    // still has to end it: the next launch was only checked against it.
                    if (self.independent_launches) |v| skipped_barrier = skipped_barrier or !v[step_i];
                    continue;
                }
            };
            // prefill objective (prefill search): only consecutive linears keep the prepared input.
            switch (step) {
                .op => |i| if (self.program.ops[i] != .linear) {
                    self.prep_key = null;
                },
                .projections, .glu => {},
                else => self.prep_key = null,
            }
            // climb 2, exact prefill: the same rule for the matrix-unit prep.
            switch (step) {
                .op => |i| if (self.program.ops[i] != .linear) {
                    self.x2_key = null;
                },
                .projections, .glu => {},
                else => self.x2_key = null,
            }
            {
                const allow_concurrent = T == 1 or switch (step) {
                    .op, .bf16_cast, .add_norm, .depth_norm, .controller_depth_norms => true,
                    else => false,
                };
                if (trace == null and self.tap == null) if (self.independent_launches) |v| {
                    if ((step_i == 0 or (allow_concurrent and v[step_i])) and !skipped_barrier) self.ctx.nano_dispatch_flags = gpu.d2_concurrent;
                };
                skipped_barrier = false;
                defer self.ctx.nano_dispatch_flags = null;
                switch (step) {
                    .op => |i| try self.encode(self.program.ops[i], row0, T - row0),
                    else => {
                        self.expert_routes = null;
                        try encodeFused(self, step, row0, T - row0);
                    },
                }
            }
            const end = first + step.len();
            if (self.tap) |tap| try self.tapStep(tap, step, T);
            if (trace) |t| while (next_trace < self.program.traces.len and self.program.traces[next_trace].after_op < end) : (next_trace += 1) {
                try self.ctx.finish();
                const v = self.program.traces[next_trace].view;
                const w = self.words(v);
                const floats = self.arena.floats();
                for (0..T) |r| {
                    const dst = t.data[(next_trace * T + r) * t.width ..][0..t.width];
                    @memcpy(dst[0..v.width], floats[w.off + r * w.stride ..][0..v.width]);
                }
            };
        }
        const t_encoded = io.now() catch 0; // decode climb 2
        try self.ctx.finish();
        const t_done = io.now() catch 0; // decode climb 2
        const after = self.ctx.stats();
        self.last = .{ .dispatches = after.dispatches - before.dispatches, .submissions = after.submissions - before.submissions, .gpu_seconds = after.gpu_seconds - before.gpu_seconds, .encode_seconds = t_encoded - t_begin, .wait_seconds = t_done - t_encoded };
        self.position += T;
        if (self.tap) |tap| tap.forwards += 1;
        self.out_first = out_first;
        self.out_end = T;
    }

    /// Copy what a tapped step produced (see `Tap`). Steps other than an attention or
    /// a router are not looked at.
    fn tapStep(self: *Engine, tap: *Tap, step: plan.Step, T: u32) !void {
        const i = switch (step) {
            .op, .relative_gate => |i| i,
            else => return,
        };
        switch (self.program.ops[i]) {
            .relative_attention => |a| {
                try self.ctx.finish();
                const o = self.words(a.out);
                const floats = self.arena.floats();
                if (tap.forwards == 0) try tap.windows.append(tap.gpa, a.window);
                tap.heads = a.heads;
                for (0..T) |r| for (0..a.heads) |h| {
                    var sum: f64 = 0;
                    for (floats[o.off + r * o.stride + h * a.dim ..][0..a.dim]) |x| sum += @as(f64, x) * x;
                    try tap.norms.append(tap.gpa, @floatCast(@sqrt(sum)));
                };
                if (tap.attention) try self.tapAttention(tap, a, T);
            },
            .route_topk => |r| {
                try self.ctx.finish();
                const o = self.words(r.out);
                const floats = self.arena.floats();
                tap.top_k = r.top_k;
                for (0..T) |row| try tap.routes.appendSlice(tap.gpa, floats[o.off + row * o.stride ..][0 .. 2 * r.top_k]);
            },
            else => {},
        }
    }

    /// Tap.attention: what mk_relative_attention computes for each row and head, again on
    /// the host in f64: the softmax over the visible keys and the sink, its weighted sum of
    /// values (compared with the engine's output), and the weights averaged over the heads.
    fn tapAttention(self: *Engine, tap: *Tap, a: anytype, T: u32) !void {
        const arena = self.arena.floats();
        const q = self.words(a.q);
        const bias = self.words(a.bias);
        const sink = self.words(a.sink);
        const o = self.words(a.out);
        const capacity = self.program.states[a.keys].capacity(self.batch, self.context);
        const keys = self.caches[a.keys].?.floats();
        const values = self.caches[a.values].?.floats();
        const width = a.kv_heads * a.dim;
        const group = a.heads / a.kv_heads;
        const first_weight = tap.weights.items.len;
        const first_sink = tap.sinks.items.len;
        try tap.weights.appendNTimes(tap.gpa, 0, @as(usize, T) * tap.context);
        try tap.sinks.appendNTimes(tap.gpa, 0, T);
        const per_head = 1.0 / @as(f64, @floatFromInt(a.heads));
        for (0..T) |r| {
            const position: u32 = self.position + @as(u32, @intCast(r));
            if (position >= tap.context) return error.TapContextTooSmall;
            const first: u32 = if (a.window != 0 and position + 1 > a.window) position + 1 - a.window else 0;
            const count = position + 1 - first;
            try tap.scratch.resize(tap.gpa, count);
            const scores = tap.scratch.items;
            const row_weights = tap.weights.items[first_weight + r * tap.context ..][0..tap.context];
            for (0..a.heads) |h| {
                const kvh = h / group;
                const qp = arena[q.off + r * q.stride + h * a.dim ..][0..a.dim];
                const sink_logit: f64 = arena[sink.off + r * sink.stride + h];
                var maximum = sink_logit;
                for (0..count) |j| {
                    const slot = (first + j) % capacity;
                    const kp = keys[slot * width + kvh * a.dim ..][0..a.dim];
                    var dot: f64 = 0;
                    for (qp, kp) |x, y| dot += @as(f64, x) * y;
                    const distance = position - (first + @as(u32, @intCast(j)));
                    const relative: f64 = if (distance < a.relative_extent) arena[bias.off + r * bias.stride + h * a.relative_extent + distance] else 0;
                    scores[j] = dot * a.scale + relative;
                    maximum = @max(maximum, scores[j]);
                }
                var denominator = @exp(sink_logit - maximum);
                for (scores) |*s| {
                    s.* = @exp(s.* - maximum);
                    denominator += s.*;
                }
                tap.sinks.items[first_sink + r] += @floatCast(@exp(sink_logit - maximum) / denominator * per_head);
                for (0..count) |j| row_weights[first + j] += @floatCast(scores[j] / denominator * per_head);
                const out = arena[o.off + r * o.stride + h * a.dim ..][0..a.dim];
                for (0..a.dim) |d| {
                    var acc: f64 = 0;
                    for (0..count) |j| acc += scores[j] / denominator * values[((first + j) % capacity) * width + kvh * a.dim + d];
                    tap.worst = @max(tap.worst, @as(f32, @floatCast(@abs(acc - out[d]))));
                }
            }
        }
    }

    /// All-vocabulary logits of batch row `row` of the last forward.
    pub fn logits(self: *const Engine, row: u32) ![]const f32 {
        if (row < self.out_first or row >= self.out_end) return error.NoLogitsForRow;
        // decode climb 2 (d2_gate): the last forward's buffer must have completed
        // (all but the one waiting at the gate).
        if (gpu.pending(self.ctx) > @intFromBool(self.d2_ahead != null)) try gpu.drain(self.ctx, @intFromBool(self.d2_ahead != null));
        const w = self.words(self.program.logits);
        return self.arena.floats()[w.off + row * w.stride ..][0..self.program.logits.width];
    }
    /// GPU argmax of batch row `row`; non-finite logits are an error.
    pub fn next(self: *const Engine, row: u32) !u32 {
        if (row < self.out_first or row >= self.out_end) return error.NoLogitsForRow;
        const w = self.words(self.program.next);
        const id = self.d2_last orelse self.arena.structs(u32)[w.off + row * w.stride]; // decode climb 2 (d2_gate)
        if (id >= self.program.meta.vocab) return error.NonFiniteLogits;
        return id;
    }

    fn quantParams(p: *gpu.Params, w: weights.Weight) void {
        p.w = w.tensor.offset;
        p.dtype = @backingInt(w.tensor.dtype);
        if (w.quant) |q| {
            p.s = q.scale_offset;
            p.b = q.bias_offset;
            p.sdtype = q.scale_dtype;
            p.bdtype = q.bias_dtype;
            p.bits = q.bits;
            p.group = q.group_size;
        }
    }
    fn quantBuffers(w: weights.Weight) [3]metal.Buffer {
        return if (w.quant) |q| .{ w.buffer, q.scales, q.biases } else .{ w.buffer, w.buffer, w.buffer };
    }

    // ---- prefill objective (prefill search) ----
    /// Prepare rows [row0, row0 + rows) of view `x` for the matrix-matrix
    /// kernels (once while they stay unchanged); returns the xsum word offset.
    fn prepare(self: *Engine, x: ir.View, row0: u32, rows: u32) !u32 {
        const key: PrepKey = .{ .state = x.state, .col = x.col, .width = x.width, .row0 = row0, .rows = rows };
        if (self.prep_key) |k| if (std.meta.eql(k, key)) return self.prep_xsum;
        const v = self.words(x);
        var p: gpu.Params = .{ .rows = rows, .row0 = row0, .pos = self.position, .x = v.off, .xs = v.stride, .n = x.width, .z = self.prep_xsum };
        try gpu.dispatch(self.ctx, "mk_q4_prep", &.{ self.arena, self.prep.? }, &p, @as(usize, rows) * (x.width / 16), 256);
        self.prep_key = key;
        return self.prep_xsum;
    }
    /// A dispatch wrote `out`: the prepared rows are stale if they overlap it.
    fn prepWritten(self: *Engine, out: ir.View) void {
        if (self.prep_key) |k| if (k.state == out.state and k.col < out.col + out.width and out.col < k.col + k.width) {
            self.prep_key = null;
        };
    }

    // ---- climb 2, exact prefill ----
    /// Prepare rows [row0, row0 + rows) of `x` (unless `x2prep` holds them) and run
    /// the matrix-unit linear with operands `p`.
    fn x2Linear(self: *Engine, x: ir.View, qb: [3]metal.Buffer, p: *gpu.Params, row0: u32, rows: u32) !void {
        const key: PrepKey = .{ .state = x.state, .col = x.col, .width = x.width, .row0 = row0, .rows = rows };
        const fresh = if (self.x2_key) |k| !std.meta.eql(k, key) else true;
        try exact2.encodeLinear(self.ctx, qb, self.arena, self.x2prep.?, p, fresh, self.knobs.x2_half);
        self.x2_key = key;
    }
    /// A dispatch wrote `out`: the matrix-unit prep is stale if it overlaps it.
    fn x2Written(self: *Engine, out: ir.View) void {
        if (self.x2_key) |k| if (k.state == out.state and k.col < out.col + out.width and out.col < k.col + k.width) {
            self.x2_key = null;
        };
    }

    fn expertTileRows(knobs: knobs_mod.Knobs) u32 {
        if (knobs.expert_tile_rows != 0) return knobs.expert_tile_rows;
        return if (knobs.expert_mlx_scale) 512 else 128;
    }

    fn expertProjection(self: *Engine, kernel: [:0]const u8, bank: ir.PackedProjection, x: ir.View, routes: ir.View, out: ir.View, p: *gpu.ExpertParams, count: usize, input_bf16: bool) !void {
        // More rows than the route lists and partial sums hold: a tile of rows at a time.
        const tile_rows = expertTileRows(self.knobs);
        if (self.expert_tiles != null and p.rows > tile_rows) {
            const row0 = p.row0;
            const rows = p.rows;
            var off: u32 = 0;
            while (off < rows) : (off += tile_rows) {
                p.row0 = row0 + off;
                p.rows = @min(tile_rows, rows - off);
                try self.expertProjection(kernel, bank, x, routes, out, p, count, input_bf16);
            }
            p.row0 = row0;
            p.rows = rows;
            self.expert_routes = null; // the lists now hold the last tile's routes only
            return;
        }
        const prev_routes = self.expert_routes;
        self.expert_routes = null;
        const v = self.words(x);
        const r = self.words(routes);
        const o = self.words(out);
        const w = self.bound[bank.payload];
        const scales = self.bound[bank.scales];
        p.x = v.off;
        p.xs = v.stride;
        p.routes = r.off;
        p.rs = r.stride;
        p.y = o.off;
        p.ys = o.stride;
        p.w = w.tensor.offset;
        p.s = scales.tensor.offset;
        var amax = w.buffer;
        var denominator = w.buffer;
        if (bank.amax) |id| {
            const a = self.bound[id];
            const d = self.bound[bank.denominator.?]; // validated by IR
            p.nvfp4 = 1;
            p.amax = a.tensor.offset;
            p.denominator = d.tensor.offset;
            amax = a.buffer;
            denominator = d.buffer;
        }
        const buffers = &.{ self.arena, w.buffer, scales.buffer, amax, denominator };
        const down = std.mem.eql(u8, kernel, "mk_expert_down");
        const projection_rows = p.n * @as(u32, if (down) 1 else 2);
        if (self.expert_tiles != null and p.rows >= 8 and p.rows <= expertTileRows(self.knobs) and p.k % 32 == 0 and projection_rows % 32 == 0) {
            const tile = self.expert_tiles.?;
            const block = self.knobs.expert_mlx_scale and input_bf16 and p.k % 64 == 0 and p.x % 4 == 0 and p.xs % 4 == 0 and (p.nvfp4 == 0 or p.w % 4 == 0);
            const key: @TypeOf(self.expert_routes.?) = .{ .off = r.off, .stride = r.stride, .row0 = p.row0, .rows = p.rows, .experts = p.experts, .top_k = p.top_k };
            const reusable = if (prev_routes) |prev| std.meta.eql(prev, key) else false;
            if (!reusable) {
                if (block and self.knobs.expert_lists_split and p.rows >= 256) {
                    try gpu.dispatch(self.ctx, "mk_expert_row_counts", &.{ self.arena, tile[0] }, p, @as(usize, p.experts) * list_segments * 32, 128);
                    try gpu.dispatch(self.ctx, "mk_expert_row_fill", &.{ self.arena, tile[0] }, p, @as(usize, p.experts) * list_segments * 32, 128);
                } else if (block)
                    try gpu.dispatch(self.ctx, "mk_expert_row_lists_simd", &.{ self.arena, tile[0] }, p, @as(usize, p.experts) * 32, 128)
                else
                    try gpu.dispatch(self.ctx, "mk_expert_row_lists", &.{ self.arena, tile[0] }, p, p.experts, 32);
                self.ctx.nano_dispatch_flags = null;
            }
            // Both list builders produce the same row-major route order. IR
            // validation forbids an expert operation's output aliasing routes.
            self.expert_routes = key;
            const block_m64n64 = block and self.knobs.expert_block_m64n64 and projection_rows % 64 == 0 and (p.nvfp4 != 0 or p.w % 16 == 0) and (down or (@as(usize, p.rows) * p.k + 1) / 2 <= tile[1].floats().len);
            const wide = self.knobs.expert_wide_k and p.k % 128 == 0;
            const bf16_storage = self.knobs.expert_bf16_tiles and input_bf16 and p.nvfp4 != 0 and p.k % 128 == 0;
            const name: [:0]const u8 = if (block_m64n64)
                (if (p.nvfp4 != 0)
                    (if (down) "mk_expert_block_down_nv64_m64" else "mk_expert_block_up_nv64_m64")
                else
                    (if (down) "mk_expert_block_down_mx64_m64" else "mk_expert_block_up_mx64_m64"))
            else if (block)
                (if (p.nvfp4 != 0)
                    (if (down) "mk_expert_block_down_nv" else "mk_expert_block_up_nv")
                else
                    (if (down) "mk_expert_block_down_mx" else "mk_expert_block_up_mx"))
            else if (bf16_storage)
                (if (down) "mk_expert_down_bf16" else "mk_expert_gate_up_bf16")
            else if (wide)
                (if (down) "mk_expert_down_wide" else "mk_expert_gate_up_wide")
            else
                (if (down) "mk_expert_down_mm" else "mk_expert_gate_up_mm");
            if (block_m64n64 and !down) {
                try gpu.dispatch(self.ctx, "mk_pack_rows_bf16", &.{ self.arena, tile[1] }, p, @as(usize, p.rows) * (p.k / 4), 256);
                self.ctx.nano_dispatch_flags = null;
            }
            const threads: usize = 128;
            const parts: usize = if (block_m64n64)
                (if (p.rows >= 2048) 8 else if (p.rows >= 256) 4 else if (p.rows >= 64) 2 else 1)
            else if (block and p.rows >= 128)
                4
            else if (block and p.rows >= 64)
                2
            else
                1;
            const col_tiles: usize = if (block_m64n64) projection_rows / 64 else projection_rows / 32;
            try gpu.dispatch(self.ctx, name, &.{ self.arena, w.buffer, scales.buffer, amax, denominator, tile[0], tile[1] }, p, @as(usize, p.experts) * col_tiles * threads * parts, threads);
            if (down) try gpu.dispatch(self.ctx, "mk_expert_combine_mm", &.{ self.arena, tile[1] }, p, @as(usize, p.rows) * p.n, 256);
        } else if (self.knobs.expert_nvfp4_fast and p.rows == 1 and p.n % 4 == 0 and p.x % 4 == 0 and p.xs % 4 == 0 and
            !down and p.top_k <= 8 and p.k % 256 == 0 and p.w % 8 == 0)
        {
            if (p.nvfp4 != 0) {
                const simds = @as(usize, p.top_k) * (p.n / 4);
                try gpu.dispatch(self.ctx, "mk_expert_gate_up_nv16", buffers, p, simds * 32, 128);
            } else {
                const simds = @as(usize, p.top_k) * (p.n / 2);
                try gpu.dispatch(self.ctx, "mk_expert_gate_up_mx_vec", buffers, p, simds * 32, 256);
            }
        } else if (self.knobs.expert_nvfp4_fast and p.rows == 1 and p.n % 4 == 0 and p.x % 4 == 0 and p.xs % 4 == 0 and
            down and p.k % 128 == 0 and (if (p.nvfp4 != 0) p.top_k == 8 and p.w % 8 == 0 else p.top_k <= 8 and p.w % 4 == 0))
        {
            if (p.nvfp4 != 0) {
                try gpu.dispatch(self.ctx, "mk_expert_down_nv8", buffers, p, @as(usize, p.n / 4) * 64, 64);
            } else {
                try gpu.dispatch(self.ctx, "mk_expert_down_mx_vec", buffers, p, @as(usize, p.n / 4) * 256, 256);
            }
        } else if (self.knobs.expert_simd and self.knobs.expert_wide_dot and p.top_k <= 8 and projection_rows % 4 == 0 and
            (if (p.nvfp4 != 0) p.k % 16 == 0 and p.w % 4 == 0 else p.k % 32 == 0))
        {
            const name: [:0]const u8 = if (down) "mk_expert_down_dot8" else "mk_expert_gate_up_dot8";
            const groups = @as(usize, p.rows) * (if (down) (count / 4) * 8 else count / 2);
            try gpu.dispatch(self.ctx, name, buffers, p, groups * 32, 256);
        } else if (self.knobs.expert_simd and p.top_k <= 8) {
            const name: [:0]const u8 = if (down) "mk_expert_down_simd" else if (self.knobs.expert_gate_pair and p.nvfp4 != 0) "mk_expert_gate_up_pair" else "mk_expert_gate_up_simd";
            try gpu.dispatch(self.ctx, name, buffers, p, @as(usize, p.rows) * count * (if (down) @as(usize, 256) else 32), 256);
        } else try gpu.dispatch(self.ctx, kernel, buffers, p, @as(usize, p.rows) * count, 256);
    }

    /// The reference schedule: exactly one dispatch per op.
    fn encode(self: *Engine, op: ir.Op, row0: u32, rows: u32) !void {
        return self.encodeRound(op, row0, rows, false);
    }

    fn encodeRound(self: *Engine, op: ir.Op, row0: u32, rows: u32, round_out: bool) !void {
        if (op != .expert_gate_up and op != .expert_down) self.expert_routes = null;
        var p: gpu.Params = .{ .rows = rows, .row0 = row0, .pos = self.position, .flags = if (round_out) gpu.flag_round_out else 0 };
        switch (op) {
            .embed => |e| {
                const w = self.bound[e.table];
                quantParams(&p, w);
                const t = self.words(e.tokens);
                const o = self.words(e.out);
                p.x = t.off;
                p.xs = t.stride;
                p.y = o.off;
                p.ys = o.stride;
                p.n = e.out.width;
                if (e.round == .bf16) p.flags |= gpu.flag_round_bf16;
                const qb = quantBuffers(w);
                try gpu.dispatch(self.ctx, "mk_embed", &.{ qb[0], qb[1], qb[2], self.arena }, &p, @as(usize, rows) * p.n, 256);
            },
            .rmsnorm => |n| {
                const x = self.words(n.x);
                const o = self.words(n.out);
                p.x = x.off;
                p.xs = x.stride;
                p.y = o.off;
                p.ys = o.stride;
                p.n = n.group;
                p.heads = n.x.width / n.group;
                p.eps = n.eps;
                if (n.round_normalized == .bf16) p.flags |= gpu.flag_round_bf16;
                var wbuf = self.arena;
                if (n.weight) |id| {
                    const w = self.bound[id];
                    p.w = w.tensor.offset;
                    p.dtype = @backingInt(w.tensor.dtype);
                    p.flags |= gpu.flag_has_weight;
                    wbuf = w.buffer;
                }
                const tg = gpu.normThreads(n.group);
                const fast = self.knobs.rmsnorm_fast and p.n % 4 == 0 and p.n <= 4 * tg and x.off % 4 == 0 and x.stride % 4 == 0 and o.off % 4 == 0 and o.stride % 4 == 0 and
                    if (n.weight) |id| ((self.bound[id].tensor.dtype == .bf16 and p.w % 8 == 0) or (self.bound[id].tensor.dtype == .f32 and p.w % 16 == 0)) else true;
                try gpu.dispatch(self.ctx, if (fast) "mk_rmsnorm_vec" else "mk_rmsnorm", &.{ wbuf, self.arena }, &p, @as(usize, rows) * p.heads * tg, tg);
            },
            .linear => |l| {
                const w = self.bound[l.w];
                quantParams(&p, w);
                const x = self.words(l.x);
                const o = self.words(l.out);
                p.x = x.off;
                p.xs = x.stride;
                p.y = o.off;
                p.ys = o.stride;
                p.n = l.x.width;
                p.m = l.out.width;
                const qb = quantBuffers(w);
                // c2-prefill-mlx (knob mlx_qmm2): the products stock MLX runs unsplit, restructured.
                if (rows > 1 and self.knobs.mlx_qmm2 and self.splitk != null and mlxApplies(w, p.n, p.m, rows) and mlx2Applies(&p, 64)) {
                    try mlx2Linear(self.ctx, qb, self.arena, &p);
                    self.prepWritten(l.out);
                } else if (rows > 1 and self.splitk != null and mlxApplies(w, p.n, p.m, rows)) {
                    // prefill-mlx (mlx-arith tier): MLX's arithmetic, not the reference's.
                    _ = try mlxLinear(self.ctx, qb, self.arena, self.splitk.?, &p);
                    self.prepWritten(l.out);
                } else if (rows > 1 and self.x2prep != null and q4Fast(w, p.n) and exact2.applies(p.sdtype, p.bdtype)) {
                    // climb 2, exact prefill
                    try self.x2Linear(l.x, qb, &p, row0, rows);
                    self.x2Written(l.out);
                    self.prepWritten(l.out);
                } else if (rows > 1 and self.prep != null and q4Fast(w, p.n)) {
                    // prefill objective (prefill search)
                    p.z = try self.prepare(l.x, row0, rows);
                    const simds = ((@as(usize, p.m) + self.knobs.mm_rows - 1) / self.knobs.mm_rows) * ((@as(usize, rows) + self.knobs.mm_tokens - 1) / self.knobs.mm_tokens);
                    try gpu.dispatch(self.ctx, mmKernel("mk_mm_q4", self.knobs), &.{ qb[0], qb[1], qb[2], self.arena, self.prep.? }, &p, simds * 32, self.knobs.mm_group);
                    self.prepWritten(l.out);
                } else if (q4Fast(w, p.n)) {
                    const R = if (rows == 1) self.knobs.decode_rows else self.knobs.prefill_rows;
                    const TT = if (rows == 1) self.knobs.decode_tokens else self.knobs.prefill_tokens;
                    const simds = ((@as(usize, p.m) + R - 1) / R) * ((@as(usize, rows) + TT - 1) / TT);
                    const kernel = if (self.knobs.decode_q4_static and rows == 1 and R == 2 and TT == 1 and p.group == 64 and p.sdtype == 2 and p.bdtype == 2)
                        switch (p.n) {
                            2048 => "mk_linear_q4_static_2048",
                            7680 => "mk_linear_q4_static_7680",
                            8192 => "mk_linear_q4_static_8192",
                            else => tileKernel("mk_linear_q4", R, TT),
                        }
                    else
                        tileKernel("mk_linear_q4", R, TT);
                    try gpu.dispatch(self.ctx, kernel, &.{ qb[0], qb[1], qb[2], self.arena }, &p, simds * 32, self.knobs.linear_group);
                } else if (self.knobs.bf16_prefill_matrix and rows >= 16 and l.input_bf16 and
                    w.tensor.dtype == .bf16 and w.quant == null and p.m >= @as(u32, if (self.knobs.bf16_matrix_small) 64 else 512) and p.m % 64 == 0 and
                    p.n >= @as(u32, if (self.knobs.bf16_matrix_small) 32 else 512) and p.n % 32 == 0 and
                    w.tensor.offset % 8 == 0 and p.x % 4 == 0 and p.xs % 4 == 0)
                {
                    const tile: u32 = if (rows >= 128) 32 else 16;
                    const groups = @as(usize, p.m / 64) * ((rows + tile - 1) / tile);
                    const name: [:0]const u8 = if (tile == 32) "mk_bf16_prefill_matrix32" else "mk_bf16_prefill_matrix16";
                    try gpu.dispatch(self.ctx, name, &.{ w.buffer, self.arena }, &p, groups * 128, 128);
                } else if (self.knobs.bf16_linear_tile and w.tensor.dtype == .bf16 and w.quant == null and
                    self.knobs.bf16_decode_splitk and rows == 1 and l.input_bf16 and
                    p.m <= 128 and p.n % 128 == 0 and w.tensor.offset % 8 == 0 and p.x % 4 == 0 and p.xs % 4 == 0)
                {
                    try gpu.dispatch(self.ctx, "mk_bf16_linear_splitk", &.{ w.buffer, self.arena }, &p, @as(usize, p.m) * inkling_plan.splitk_threads, inkling_plan.splitk_threads);
                } else if (self.knobs.bf16_linear_tile and w.tensor.dtype == .bf16 and w.quant == null and p.m % 4 == 0 and p.n % 32 == 0) {
                    const tile = self.knobs.bf16_prefill_tile;
                    const full_tile = tile != 0 and rows % tile == 0;
                    const tiles = if (full_tile) rows / tile else if (rows == 1) rows else (rows + 3) / 4;
                    const vector = p.n % 128 == 0 and p.x % 4 == 0 and p.xs % 4 == 0 and w.tensor.offset % 8 == 0;
                    const wide8 = self.knobs.bf16_decode_wide and vector and p.n % 256 == 0 and w.tensor.offset % 16 == 0;
                    const k: [:0]const u8 = if (rows == 1)
                        (if (wide8) "mk_bf16_linear_4x1_vec8" else if (self.knobs.bf16_decode_vec and vector) "mk_bf16_linear_4x1_vec" else "mk_bf16_linear_4x1")
                    else if (full_tile) switch (tile) {
                        8 => if (self.knobs.bf16_prefill_matrix and vector and rows >= 128) "mk_bf16_linear_4x8_full_vec" else "mk_bf16_linear_4x8_full",
                        16 => "mk_bf16_linear_4x16_full",
                        32 => "mk_bf16_linear_4x32_full",
                        else => unreachable,
                    } else "mk_bf16_linear_4x4";
                    try gpu.dispatch(self.ctx, k, &.{ w.buffer, self.arena }, &p, @as(usize, p.m / 4) * tiles * 32, 128);
                } else {
                    try gpu.dispatch(self.ctx, "mk_linear_generic", &.{ qb[0], qb[1], qb[2], self.arena }, &p, @as(usize, p.m) * rows * 32, 128);
                }
            },
            .binary => |b| {
                const a = self.words(b.a);
                const bb = self.words(b.b);
                const o = self.words(b.out);
                p.x = a.off;
                p.xs = a.stride;
                p.y = bb.off;
                p.ys = bb.stride;
                p.z = o.off;
                p.zs = o.stride;
                p.n = b.a.width;
                p.every = b.b_every;
                p.kind = @backingInt(b.kind);
                p.act_a = @backingInt(b.a_act);
                p.act_b = @backingInt(b.b_act);
                try gpu.dispatch(self.ctx, "mk_binary", &.{self.arena}, &p, @as(usize, rows) * p.n, 256);
            },
            .row_transform => |t| {
                const x = self.words(t.x);
                const y = self.words(t.out);
                p.x = x.off;
                p.xs = x.stride;
                p.y = y.off;
                p.ys = y.stride;
                p.n = t.out.width;
                p.flags = @backingInt(t.round);
                switch (t.transform) {
                    .copy => {},
                    .softcap => |cap| {
                        p.kind = 1;
                        p.scale = cap;
                    },
                    .log_position_scale => |s| {
                        p.kind = 2;
                        p.m = s.floor;
                        p.scale = s.alpha;
                    },
                }
                try gpu.dispatch(self.ctx, "mk_row_transform", &.{self.arena}, &p, @as(usize, rows) * p.n, 256);
            },
            .grouped_project => |g| {
                const x = self.words(g.x);
                const y = self.words(g.out);
                const w = self.bound[g.weight];
                if (w.quant != null or !w.tensor.dtype.isFloat()) return error.InvalidGroupedProjection;
                p.w = w.tensor.offset;
                p.dtype = @backingInt(w.tensor.dtype);
                p.x = x.off;
                p.xs = x.stride;
                p.y = y.off;
                p.ys = y.stride;
                p.n = g.out.width;
                p.m = g.out.width / g.groups;
                p.dim = g.x.width / g.groups;
                p.flags = @backingInt(g.round);
                // Only distances at or below the last query position can be read.
                p.taps = if (g.causal_prefix) @min(p.m, self.position + row0 + rows) else p.m;
                if (self.knobs.bf16_prefill_tile != 0 and rows % 4 == 0 and p.dim == 16 and w.tensor.dtype == .bf16 and p.w % 2 == 0) {
                    // knobs.relative_bias16: a chunk's bias for the staged tiles, as 16-bit words.
                    const pack16 = if (self.bias16_ok) |ok| ok[g.out.state] and g.causal_prefix and g.round == .bf16 and rows >= min_tile_rows else false;
                    if (pack16) self.bias16 = .{ .state = g.out.state, .row0 = row0, .rows = rows };
                    if (self.knobs.relative_bias_d8 and p.m % 8 == 0) {
                        try gpu.dispatch(self.ctx, if (pack16) "mk_grouped_project_bf16d8_pack" else "mk_grouped_project_bf16d8", &.{ self.arena, w.buffer }, &p, @as(usize, rows / 2) * g.groups * ((p.taps + 7) / 8), 256);
                    } else try gpu.dispatch(self.ctx, if (pack16) "mk_grouped_project_bf16x4_pack" else "mk_grouped_project_bf16x4", &.{ self.arena, w.buffer }, &p, @as(usize, rows / 4) * g.groups * p.taps, 256);
                } else if (self.knobs.relative_bias_decode and rows == 1 and p.dim == 16 and w.tensor.dtype == .bf16 and p.m % 8 == 0 and p.w % 2 == 0) {
                    const total_threads = @as(usize, g.groups) * ((p.taps + 7) / 8);
                    try gpu.dispatch(self.ctx, "mk_grouped_project_bf16d8_row1", &.{ self.arena, w.buffer }, &p, total_threads, @min(total_threads, 128));
                } else
                    try gpu.dispatch(self.ctx, "mk_grouped_project", &.{ self.arena, w.buffer }, &p, @as(usize, rows) * g.groups * p.taps, 256);
            },
            .concat => |c| {
                const a = self.words(c.a);
                const b = self.words(c.b);
                const o = self.words(c.out);
                p.x = a.off;
                p.xs = a.stride;
                p.y = b.off;
                p.ys = b.stride;
                p.z = o.off;
                p.zs = o.stride;
                p.n = c.a.width;
                p.m = c.b.width;
                try gpu.dispatch(self.ctx, "mk_concat", &.{self.arena}, &p, @as(usize, rows) * (p.n + p.m), 256);
            },
            .activate => |a| {
                const x = self.words(a.x);
                const o = self.words(a.out);
                p.x = x.off;
                p.xs = x.stride;
                p.z = o.off;
                p.zs = o.stride;
                p.n = a.x.width;
                p.act_a = @backingInt(a.kind);
                try gpu.dispatch(self.ctx, "mk_activate", &.{self.arena}, &p, @as(usize, rows) * p.n, 256);
            },
            .softcapped_swiglu => |s| {
                const gate = self.words(s.gate);
                const up = self.words(s.up);
                const o = self.words(s.out);
                p.x = gate.off;
                p.xs = gate.stride;
                p.y = up.off;
                p.ys = up.stride;
                p.z = o.off;
                p.zs = o.stride;
                p.n = s.gate.width;
                p.eps = s.gate_cap;
                p.scale = s.up_cap;
                p.flags = if (s.round == .bf16) gpu.flag_round_bf16 else 0;
                try gpu.dispatch(self.ctx, "mk_inkling_softcapped_swiglu", &.{self.arena}, &p, @as(usize, rows) * p.n, 256);
            },
            .depth_attention => |d| {
                const bank = self.words(d.bank);
                const partial = self.words(d.partial);
                const o = self.words(d.out);
                const query = self.bound[d.query];
                if (query.quant != null or !query.tensor.dtype.isFloat()) return error.InvalidDepthAttentionQuery;
                p.x = bank.off;
                p.xs = bank.stride;
                p.y = partial.off;
                p.ys = partial.stride;
                p.z = o.off;
                p.zs = o.stride;
                p.n = d.partial.width;
                p.m = d.bank.width / d.partial.width;
                p.heads = p.m + 1;
                p.w = query.tensor.offset + @as(usize, d.query_index) * d.partial.width * query.tensor.dtype.byteSize();
                p.dtype = @backingInt(query.tensor.dtype);
                p.scale = d.scale;
                if (self.knobs.depth_attention_fast and p.heads <= 8 and query.tensor.dtype == .f32 and p.w % 16 == 0 and
                    bank.off % 4 == 0 and bank.stride % 4 == 0 and partial.off % 4 == 0 and partial.stride % 4 == 0 and o.off % 4 == 0 and o.stride % 4 == 0 and
                    (p.n == inkling_plan.hidden or (p.n % 4 == 0 and p.n <= 1024)))
                {
                    const tg: usize = if (p.n == inkling_plan.hidden) inkling_plan.depth_reg_threads else inkling_plan.depth_vec_threads;
                    const kname: [:0]const u8 = if (p.n == inkling_plan.hidden) "mk_depth_attention_reg" else "mk_depth_attention_vec";
                    try gpu.dispatch(self.ctx, kname, &.{ self.arena, query.buffer }, &p, @as(usize, rows) * tg, tg);
                } else {
                    try gpu.dispatch(self.ctx, "mk_depth_attention", &.{ self.arena, query.buffer }, &p, @as(usize, rows) * 256, 256);
                }
            },
            .store => |s| {
                const x = self.words(s.src);
                const d = self.program.states[s.dst];
                p.x = x.off;
                p.xs = x.stride;
                p.n = s.src.width;
                p.capacity = d.capacity(self.batch, self.context);
                if (self.relative_half) |hs| if (hs[s.dst]) |half| {
                    if (self.knobs.store_pack_bf16) {
                        try gpu.dispatch(self.ctx, "mk_store_pack_bf16", &.{ self.arena, self.caches[s.dst].?, half }, &p, @as(usize, rows) * p.n, 256);
                        return;
                    }
                    try gpu.dispatch(self.ctx, "mk_store", &.{ self.arena, self.caches[s.dst].? }, &p, @as(usize, rows) * p.n, 256);
                    try gpu.dispatch(self.ctx, "mk_pack_bf16", &.{ self.arena, half }, &p, @as(usize, rows) * p.n, 256);
                    return;
                };
                try gpu.dispatch(self.ctx, "mk_store", &.{ self.arena, self.caches[s.dst].? }, &p, @as(usize, rows) * p.n, 256);
            },
            .conv => |c| {
                const w = self.bound[c.weight];
                const o = self.words(c.out);
                const src = self.program.states[c.src];
                p.w = w.tensor.offset;
                p.dtype = @backingInt(w.tensor.dtype);
                p.y = o.off;
                p.ys = o.stride;
                p.n = c.out.width;
                p.taps = c.taps;
                p.capacity = src.capacity(self.batch, self.context);
                try gpu.dispatch(self.ctx, "mk_conv", &.{ self.caches[c.src].?, w.buffer, self.arena }, &p, @as(usize, rows) * p.n, 256);
            },
            .inkling_short_conv => |c| {
                const w = self.bound[c.weight];
                const residual = self.words(c.residual);
                const o = self.words(c.out);
                const src = self.program.states[c.src];
                p.w = w.tensor.offset;
                p.dtype = @backingInt(w.tensor.dtype);
                p.x = residual.off;
                p.xs = residual.stride;
                p.z = o.off;
                p.zs = o.stride;
                p.n = c.out.width;
                p.taps = c.taps;
                p.capacity = src.capacity(self.batch, self.context);
                p.scale = 1.0 / @sqrt(@as(f32, @floatFromInt(c.taps + 1)));
                p.flags = if (c.round == .bf16) 1 else 0;
                const kernel: [:0]const u8 = if (self.knobs.conv_ring and p.capacity >= p.taps) "mk_inkling_short_conv_ring" else "mk_inkling_short_conv";
                try gpu.dispatch(self.ctx, kernel, &.{ self.caches[c.src].?, w.buffer, self.arena }, &p, @as(usize, rows) * p.n, 256);
            },
            .inkling_dynamic_conv => |c| {
                const w = self.bound[c.static_weight];
                const dynamic = self.words(c.dynamic);
                const residual = self.words(c.residual);
                const o = self.words(c.out);
                const src = self.program.states[c.src];
                p.w = w.tensor.offset;
                p.dtype = @backingInt(w.tensor.dtype);
                p.x = dynamic.off;
                p.xs = dynamic.stride;
                p.y = residual.off;
                p.ys = residual.stride;
                p.z = o.off;
                p.zs = o.stride;
                p.n = c.out.width;
                p.heads = c.heads;
                p.taps = c.taps;
                p.capacity = src.capacity(self.batch, self.context);
                p.scale = 1.0 / @sqrt(@as(f32, @floatFromInt(2 * c.taps + 1)));
                p.flags = if (c.round == .bf16) 1 else 0;
                const kernel: [:0]const u8 = if (self.knobs.conv_ring and p.capacity >= p.taps) "mk_inkling_dynamic_conv_ring" else "mk_inkling_dynamic_conv";
                try gpu.dispatch(self.ctx, kernel, &.{ self.caches[c.src].?, w.buffer, self.arena }, &p, @as(usize, rows) * p.n, 256);
            },
            .inkling_message_merge => |m| {
                const hidden = self.words(m.hidden);
                const message = self.words(m.message);
                const gates = self.words(m.gates);
                const o = self.words(m.out);
                p.x = hidden.off;
                p.xs = hidden.stride;
                p.y = message.off;
                p.ys = message.stride;
                p.z = gates.off;
                p.zs = gates.stride;
                p.w = o.off;
                p.every = o.stride;
                p.n = m.out.width;
                p.flags = if (m.round == .bf16) 1 else 0;
                try gpu.dispatch(self.ctx, "mk_inkling_message_merge", &.{self.arena}, &p, @as(usize, rows) * p.n, 256);
            },
            .rope => |r| {
                const x = self.words(r.x);
                p.x = x.off;
                p.xs = x.stride;
                p.z = self.table_offsets[r.freqs];
                p.heads = r.heads;
                p.dim = r.dim;
                p.rot = r.rot;
                if (r.fast_trig) p.flags |= gpu.flag_fast_trig;
                try gpu.dispatch(self.ctx, "mk_rope", &.{ self.arena, self.tables }, &p, @as(usize, rows) * r.heads * (r.rot / 2), 64);
            },
            .attention => |a| {
                if (try encodeDecodeAttention(self, a, null, row0, rows)) return;
                if (try encodeMlxAttention(self, a, null, row0, rows)) return;
                if (try encodeX2Attention(self, a, null, row0, rows)) return; // climb 2, exact prefill
                const q = self.words(a.q);
                const o = self.words(a.out);
                const k = self.program.states[a.keys];
                p.x = q.off;
                p.xs = q.stride;
                p.y = o.off;
                p.ys = o.stride;
                p.heads = a.heads;
                p.kv_heads = a.kv_heads;
                p.dim = a.dim;
                p.window = a.window;
                p.scale = a.scale;
                p.capacity = k.capacity(self.batch, self.context);
                try gpu.dispatch(self.ctx, "mk_attention", &.{ self.arena, self.caches[a.keys].?, self.caches[a.values].? }, &p, @as(usize, rows) * a.heads * 256, 256);
            },
            .relative_attention => |a| {
                const q = self.words(a.q);
                const bias = self.words(a.bias);
                const sink = self.words(a.sink);
                const o = self.words(a.out);
                const k = self.program.states[a.keys];
                p.x = q.off;
                p.xs = q.stride;
                p.y = o.off;
                p.ys = o.stride;
                p.z = bias.off;
                p.zs = bias.stride;
                p.s = sink.off;
                p.every = sink.stride;
                p.heads = a.heads;
                p.kv_heads = a.kv_heads;
                p.dim = a.dim;
                p.m = a.relative_extent;
                p.window = a.window;
                p.scale = a.scale;
                p.capacity = k.capacity(self.batch, self.context);
                const end = self.position + row0 + rows;
                const visible = if (p.window == 0) end else @min(end, p.window);
                // knobs.relative_bias16: a bias written 16-bit is read by the staged tiles
                // over the rows it was written for, or the forward fails.
                if (self.bias16) |b| if (b.state == a.bias.state) {
                    self.bias16 = null;
                    const tiles = self.knobs.relative_tile_prefill and rows >= min_tile_rows and a.dim == 128 and
                        (if (self.relative_half) |hs| hs[a.keys] != null else false);
                    if (!tiles or b.row0 != row0 or b.rows != rows) return error.RelativeBiasLayout;
                    p.flags |= gpu.flag_bias16;
                };
                if (self.knobs.relative_tile_prefill and rows >= 8 and a.dim == 128) {
                    // Chunks that occupy the GPU: the reference's staged tiles, off the mirrors.
                    if (rows >= min_tile_rows) if (self.relative_half) |hs| if (hs[a.keys]) |half_keys| {
                        const kname: [:0]const u8 = if (self.knobs.relative_prefill_fast) "mk_relative_prefill_fast" else "mk_relative_prefill_tiles";
                        try gpu.dispatch(self.ctx, kname, &.{ self.arena, half_keys, hs[a.values].? }, &p, @as(usize, (rows + 31) / 32) * a.heads * 128, 128);
                        return;
                    };
                    if (self.prefill_parts) |records| if (self.relative_half) |hs| if (hs[a.keys]) |half_keys| if (p.capacity >= mirror_pad) {
                        // Two stripes occupy the GPU for a sliding window; an unwindowed walk takes four (`nano attnbench`).
                        p.taps = if (a.window != 0) 2 else max_prefill_stripes;
                        try gpu.dispatch(self.ctx, "mk_relative_prefill_stripes", &.{ self.arena, half_keys, hs[a.values].?, records }, &p, @as(usize, (rows + 31) / 32) * a.heads * p.taps * 128, 128);
                        try gpu.dispatch(self.ctx, "mk_relative_prefill_merge", &.{ self.arena, records }, &p, @as(usize, rows) * a.heads * 128, 128);
                        return;
                    };
                    try gpu.dispatch(self.ctx, "mk_relative_attention_prefill", &.{ self.arena, self.caches[a.keys].?, self.caches[a.values].? }, &p, @as(usize, (rows + 31) / 32) * a.heads * 128, 128);
                    return;
                }
                if (rows == 1 and try encodeRelativeStriped(self, a, a.out, null, &p, visible)) return;
                if (self.relative_parts) |parts| {
                    const split_decode = rows == 1 and visible >= min_relative_split;
                    const split_prefill = rows > 1 and self.knobs.relative_split_prefill and visible >= 256;
                    if (split_decode or split_prefill) {
                        p.taps = @min(max_relative_parts, @max(4, (visible + 31) / 32));
                        var off: u32 = 0;
                        while (off < rows) : (off += p.rows) {
                            p.row0 = row0 + off;
                            p.rows = @min(relative_row_tile, rows - off);
                            const heads = @as(usize, a.heads) * p.rows;
                            const wide = self.knobs.relative_split_4 and split_decode and a.dim == 128 and visible <= p.taps * 32;
                            const group: usize = if (wide) 128 else 32;
                            try gpu.dispatch(self.ctx, if (wide) "mk_relative_split_keys4" else "mk_relative_split_keys", &.{ self.arena, self.caches[a.keys].?, self.caches[a.values].?, parts }, &p, heads * p.taps * group, group);
                            try gpu.dispatch(self.ctx, "mk_relative_split_merge", &.{ self.arena, parts }, &p, heads * 32, 32);
                        }
                        return;
                    }
                }
                try gpu.dispatch(self.ctx, "mk_relative_attention", &.{ self.arena, self.caches[a.keys].?, self.caches[a.values].? }, &p, @as(usize, rows) * a.heads * 32, 32);
            },
            .l2_clip => |c| {
                const x = self.words(c.x);
                const o = self.words(c.out);
                p.x = x.off;
                p.xs = x.stride;
                p.z = o.off;
                p.zs = o.stride;
                p.n = c.group;
                p.heads = c.x.width / c.group;
                p.eps = c.threshold;
                var scale_buf = self.arena;
                if (c.scale) |id| {
                    const w = self.bound[id];
                    if (w.quant != null or !w.tensor.dtype.isFloat()) return error.InvalidL2ClipScale;
                    p.w = w.tensor.offset;
                    p.dtype = @backingInt(w.tensor.dtype);
                    p.flags |= gpu.flag_has_weight;
                    scale_buf = w.buffer;
                }
                try gpu.dispatch(self.ctx, "mk_l2_clip", &.{ self.arena, scale_buf }, &p, @as(usize, rows) * p.heads * 256, 256);
            },
            .route_topk => |r| {
                const logit_view = self.words(r.logits);
                const o = self.words(r.out);
                const correction = self.bound[r.correction];
                if (correction.quant != null or correction.tensor.dtype != .f32) return error.InvalidRouterCorrection;
                p.x = logit_view.off;
                p.xs = logit_view.stride;
                p.y = o.off;
                p.ys = o.stride;
                p.n = r.experts;
                p.m = r.top_k;
                p.s = correction.tensor.offset;
                p.eps = r.eps;
                p.scale = r.scale;
                try gpu.dispatch(self.ctx, "mk_route_topk", &.{ self.arena, correction.buffer }, &p, @as(usize, rows) * 32, 256);
            },
            .expert_gate_up => |e| {
                var ep: gpu.ExpertParams = .{ .rows = rows, .row0 = row0, .k = e.x.width, .n = e.out.width / e.top_k, .experts = e.experts, .top_k = e.top_k, .gate_cap = e.gate_cap, .up_cap = e.up_cap, .round_bf16 = @intFromBool(e.round == .bf16) };
                try self.expertProjection("mk_expert_gate_up", e.bank, e.x, e.routes, e.out, &ep, e.out.width, e.input_bf16);
            },
            .expert_down => |e| {
                var ep: gpu.ExpertParams = .{ .rows = rows, .row0 = row0, .k = e.x.width / e.top_k, .n = e.out.width, .experts = e.experts, .top_k = e.top_k, .round_bf16 = @intFromBool(e.round == .bf16) };
                try self.expertProjection("mk_expert_down", e.bank, e.x, e.routes, e.out, &ep, e.out.width, e.input_bf16);
            },
            .argmax => |m| {
                const x = self.words(m.x);
                const o = self.words(m.out);
                p.x = x.off;
                p.xs = x.stride;
                p.y = o.off;
                p.ys = o.stride;
                p.n = m.x.width;
                if (self.partials) |part| {
                    // decode search: two-stage argmax (decode_argmax = 1).
                    p.every = argmax_groups;
                    p.m = inkling_plan.argmaxSpan(p.n);
                    try gpu.dispatch(self.ctx, "mk_dec_argmax_part", &.{ self.arena, part }, &p, @as(usize, rows) * argmax_groups * 256, 256);
                    try gpu.dispatch(self.ctx, "mk_dec_argmax_final", &.{ self.arena, part }, &p, @as(usize, rows) * 64, 64);
                    return;
                }
                try gpu.dispatch(self.ctx, "mk_argmax", &.{self.arena}, &p, @as(usize, rows) * 1024, 1024);
            },
        }
    }
};

fn mat(self: *const Engine, w: weights.Weight, out: ir.View) gpu.Mat {
    const o = self.words(out);
    const q = w.quant.?;
    return .{ .w = w.tensor.offset, .s = q.scale_offset, .b = q.bias_offset, .m = out.width, .y = o.off, .ys = o.stride, .sdtype = q.scale_dtype, .bdtype = q.bias_dtype };
}

fn staticFusedQ4(self: *const Engine, x: *const gpu.ParamsX) bool {
    if (!self.knobs.decode_q4_static or self.knobs.decode_rows != 2 or self.knobs.decode_tokens != 1 or
        x.p.rows != 1 or x.p.n != 2048 or x.p.group != 64) return false;
    for (x.mat[0..x.count]) |m| if (m.sdtype != 2 or m.bdtype != 2) return false;
    return true;
}

fn linearTile(self: *const Engine, rows: u32) struct { r: u32, t: u32 } {
    return if (rows == 1) .{ .r = self.knobs.decode_rows, .t = self.knobs.decode_tokens } else .{ .r = self.knobs.prefill_rows, .t = self.knobs.prefill_tokens };
}

/// Fused steps (plan.zig); each kernel reproduces the reference arithmetic.
fn encodeFused(self: *Engine, step: plan.Step, row0: u32, rows: u32) !void {
    const ops = self.program.ops;
    var p: gpu.Params = .{ .rows = rows, .row0 = row0, .pos = self.position };
    switch (step) {
        .op => unreachable,
        .bf16_cast => |i| try self.encodeRound(ops[i], row0, rows, true),
        .projections => |m| if (rows > 1 and (self.prep != null or self.splitk != null or self.x2prep != null)) {
            // prefill-mlx (knob mlx_multi): the unsplit ones in one launch, the rest alone.
            var done: [4]bool = @splat(false);
            if (self.knobs.mlx_multi and self.splitk != null) try encodeMlxMulti(self, ops[m.first..][0..m.count], &done, row0, rows);
            // prefill objective (prefill search): one prep, then each matrix-matrix linear;
            // prefill-mlx: each linear on its own, as MLX runs them.
            for (0..m.count) |k| if (!done[k]) try self.encode(ops[m.first + k], row0, rows);
        } else {
            var x: gpu.ParamsX = .{ .p = p, .count = m.count };
            const l0 = ops[m.first].linear;
            const xin = self.words(l0.x);
            x.p.x = xin.off;
            x.p.xs = xin.stride;
            x.p.n = l0.x.width;
            x.p.group = self.bound[l0.w].quant.?.group_size;
            const tile = linearTile(self, rows);
            const tiles_t = (@as(usize, rows) + tile.t - 1) / tile.t;
            var simds: usize = 0;
            for (0..m.count) |k| {
                const l = ops[m.first + k].linear;
                x.mat[k] = mat(self, self.bound[l.w], l.out);
                simds += ((@as(usize, l.out.width) + tile.r - 1) / tile.r) * tiles_t;
            }
            const kernel = if (staticFusedQ4(self, &x)) "mk_multi_q4_static_2048" else tileKernel("mk_multi_q4", tile.r, tile.t);
            try gpu.dispatch(self.ctx, kernel, &.{ self.bound[l0.w].buffer, self.arena }, &x, simds * 32, self.knobs.linear_group);
        },
        .glu => |i| {
            const l0 = ops[i].linear;
            const l1 = ops[i + 1].linear;
            const b = ops[i + 2].binary;
            if (rows > 1 and self.x2prep != null) {
                // climb 2, exact prefill: gate, up and the product as the reference's three operations.
                for (0..3) |k| try self.encode(ops[i + k], row0, rows);
                self.x2Written(b.out);
                self.prepWritten(b.out);
                return;
            }
            if (rows > 1 and self.splitk != null and mlxApplies(self.bound[l0.w], l0.x.width, l0.out.width, rows) and
                mlxApplies(self.bound[l1.w], l1.x.width, l1.out.width, rows))
            {
                // prefill-mlx: gate, up and the silu product as MLX runs them, three operations
                // (with mlx_multi gate and up in one launch when neither is split).
                // c2-prefill-mlx (knob mlx_glu2): gate, up and the product in one launch.
                if (self.knobs.mlx_glu2 and try encodeMlx2Glu(self, l0, l1, b, row0, rows)) {
                    self.prepWritten(b.out);
                    return;
                }
                var done: [4]bool = @splat(false);
                if (self.knobs.mlx_multi) try encodeMlxMulti(self, ops[i..][0..2], &done, row0, rows);
                for (0..3) |k| if (k == 2 or !done[k]) try self.encode(ops[i + k], row0, rows);
                self.prepWritten(b.out);
                return;
            }
            var x: gpu.ParamsX = .{ .p = p, .count = 2, .act_a = @backingInt(b.a_act), .act_b = @backingInt(b.b_act), .kind = @backingInt(b.kind) };
            const xin = self.words(l0.x);
            const o = self.words(b.out);
            x.p.x = xin.off;
            x.p.xs = xin.stride;
            x.p.y = o.off;
            x.p.ys = o.stride;
            x.p.n = l0.x.width;
            x.p.group = self.bound[l0.w].quant.?.group_size;
            x.mat[0] = mat(self, self.bound[l0.w], l0.out);
            x.mat[1] = mat(self, self.bound[l1.w], l1.out);
            if (rows > 1 and self.prep != null) {
                // prefill objective (prefill search)
                x.p.z = try self.prepare(l0.x, row0, rows);
                const k = self.knobs;
                const wbuf = self.bound[l0.w].buffer;
                const simds = ((@as(usize, l0.out.width) + k.mm_rows - 1) / k.mm_rows) * ((@as(usize, rows) + k.mm_tokens - 1) / k.mm_tokens);
                try gpu.dispatch(self.ctx, mmKernel("mk_mm_glu_q4", k), &.{ wbuf, self.arena, self.prep.? }, &x, simds * 32, k.mm_group);
                self.prepWritten(b.out);
                return;
            }
            const tile = linearTile(self, rows);
            const simds = ((@as(usize, l0.out.width) + tile.r - 1) / tile.r) * ((@as(usize, rows) + tile.t - 1) / tile.t);
            const kernel = if (staticFusedQ4(self, &x)) "mk_glu_q4_static_2048" else tileKernel("mk_glu_q4", tile.r, tile.t);
            try gpu.dispatch(self.ctx, kernel, &.{ self.bound[l0.w].buffer, self.arena }, &x, simds * 32, self.knobs.linear_group);
        },
        .norms => |i| {
            const n1 = ops[i].rmsnorm;
            const add = ops[i + 1].binary;
            const n2 = ops[i + 2].rmsnorm;
            const w1 = self.bound[n1.weight.?];
            const w2 = self.bound[n2.weight.?];
            const y = self.words(n1.x);
            const r = self.words(add.out);
            const o = self.words(n2.out);
            p.x = y.off;
            p.xs = y.stride;
            p.y = r.off;
            p.ys = r.stride;
            p.z = o.off;
            p.zs = o.stride;
            p.n = n1.group;
            p.eps = n1.eps;
            p.w = w1.tensor.offset;
            p.dtype = @backingInt(w1.tensor.dtype);
            p.s = w2.tensor.offset;
            p.sdtype = @backingInt(w2.tensor.dtype);
            // The add's operand order: r + t (kind 0) or t + r (kind 1).
            p.kind = if (add.a.state == n1.out.state and add.a.col == n1.out.col) 1 else 0;
            const tg = gpu.normThreads(n1.group);
            // decode climb 2 (knob d2_norm_regs): operands in registers, single-row forwards.
            const regs = self.knobs.d2_norm_regs and rows == 1 and n1.group <= 4 * tg;
            try gpu.dispatch(self.ctx, if (regs) "mk_d2_norm_add_norm" else "mk_norm_add_norm", &.{ w1.buffer, self.arena }, &p, @as(usize, rows) * tg, tg);
        },
        .conv => |i| {
            const m0 = ops[i].binary;
            const cv = ops[i + 2].conv;
            const m1 = ops[i + 3].binary;
            const w = self.bound[cv.weight];
            const conv_left = m1.a.state == cv.out.state and m1.a.col == cv.out.col;
            const other = if (conv_left) m1.b else m1.a;
            const a = self.words(m0.a);
            const b = self.words(m0.b);
            const c = self.words(other);
            const o = self.words(m1.out);
            p.x = a.off;
            p.xs = a.stride;
            p.y = b.off;
            p.ys = b.stride;
            p.z = c.off;
            p.zs = c.stride;
            p.m = o.off;
            p.every = o.stride;
            p.n = cv.out.width;
            p.taps = cv.taps;
            p.w = w.tensor.offset;
            p.dtype = @backingInt(w.tensor.dtype);
            p.capacity = self.program.states[cv.src].capacity(self.batch, self.context);
            p.act_a = @as(u32, @backingInt(m0.a_act)) | @as(u32, @backingInt(m0.b_act)) << 2;
            p.act_b = @as(u32, @backingInt(m1.a_act)) | @as(u32, @backingInt(m1.b_act)) << 2;
            p.kind = @intFromBool(conv_left);
            if (rows > 1 and self.knobs.conv_rows) {
                // prefill-mlx (exact): store every row's product, then convolve all rows in parallel.
                try gpu.dispatch(self.ctx, "mk_gated_conv_store", &.{ self.caches[cv.src].?, self.arena }, &p, @as(usize, rows) * p.n, 256);
                // c2-prefill-mlx (knob conv_rows2): the same arithmetic without a modulo per tap.
                const rows_kernel: [:0]const u8 = if (self.knobs.conv_rows2) "mk_gated_conv_rows2" else "mk_gated_conv_rows";
                try gpu.dispatch(self.ctx, rows_kernel, &.{ self.caches[cv.src].?, w.buffer, self.arena }, &p, @as(usize, rows) * p.n, 256);
                return;
            }
            try gpu.dispatch(self.ctx, "mk_gated_conv", &.{ self.caches[cv.src].?, w.buffer, self.arena }, &p, p.n, 64);
        },
        .gate => |i| {
            const a = ops[i].attention;
            const b = ops[i + 1].binary;
            if (try encodeDecodeAttention(self, a, b, row0, rows)) return;
            if (try encodeMlxAttention(self, a, b, row0, rows)) return;
            if (try encodeX2Attention(self, a, b, row0, rows)) return; // climb 2, exact prefill
            const q = self.words(a.q);
            const o = self.words(b.out);
            const g = self.words(b.b);
            p.x = q.off;
            p.xs = q.stride;
            p.y = o.off;
            p.ys = o.stride;
            p.z = g.off;
            p.zs = g.stride;
            p.act_b = @backingInt(b.b_act);
            p.heads = a.heads;
            p.kv_heads = a.kv_heads;
            p.dim = a.dim;
            p.window = a.window;
            p.scale = a.scale;
            p.capacity = self.program.states[a.keys].capacity(self.batch, self.context);
            try gpu.dispatch(self.ctx, "mk_attention_gated", &.{ self.arena, self.caches[a.keys].?, self.caches[a.values].? }, &p, @as(usize, rows) * a.heads * 256, 256);
        },
        .attn_prep => |i| try encodeAttnPrep(self, i, row0, rows),
        .add_norm => |m| {
            const b = ops[m.first].binary;
            const norm_idx = m.first + 1 + @as(u32, @intFromBool(m.round_add));
            const n = ops[norm_idx].rmsnorm;
            const a = self.words(b.a);
            const other = self.words(b.b);
            const res = self.words(b.out);
            const out = self.words(n.out);
            if (rows == 1 and n.group == 2048 and !m.round_add and !m.round_out and n.round_normalized == .none and n.weight != null) {
                const w = self.bound[n.weight.?];
                p.x = a.off + row0 * a.stride;
                p.y = other.off + row0 * other.stride;
                p.z = res.off + row0 * res.stride;
                p.every = out.off + row0 * out.stride;
                p.w = w.tensor.offset;
                p.dtype = @backingInt(w.tensor.dtype);
                p.eps = n.eps;
                try gpu.dispatch(self.ctx, "mk_add_norm_2048", &.{ w.buffer, self.arena }, &p, 512, 512);
                return;
            }
            const tg = gpu.normThreads(n.group);
            const aligned = n.group % 4 == 0 and n.group <= 4 * tg and a.off % 4 == 0 and a.stride % 4 == 0 and
                other.off % 4 == 0 and other.stride % 4 == 0 and res.off % 4 == 0 and res.stride % 4 == 0 and
                out.off % 4 == 0 and out.stride % 4 == 0 and
                (if (n.weight) |id| ((self.bound[id].tensor.dtype == .bf16 and self.bound[id].tensor.offset % 8 == 0) or
                    (self.bound[id].tensor.dtype == .f32 and self.bound[id].tensor.offset % 16 == 0)) else true);
            if (aligned) {
                var wbuf = self.arena;
                if (n.weight) |id| {
                    const w = self.bound[id];
                    p.w = w.tensor.offset;
                    p.dtype = @backingInt(w.tensor.dtype);
                    p.flags |= gpu.flag_has_weight;
                    wbuf = w.buffer;
                }
                if (m.round_add) p.flags |= 64;
                if (n.round_normalized == .bf16) p.flags |= gpu.flag_round_bf16;
                if (m.round_out) p.flags |= gpu.flag_round_out;
                p.x = a.off;
                p.xs = a.stride;
                p.y = other.off;
                p.ys = other.stride;
                p.z = res.off;
                p.zs = res.stride;
                p.every = out.off;
                p.pad_k = out.stride;
                p.n = n.group;
                p.eps = n.eps;
                try gpu.dispatch(self.ctx, "mk_add_norm_vec", &.{ wbuf, self.arena }, &p, @as(usize, rows) * tg, tg);
            } else {
                self.ctx.nano_dispatch_flags = null;
                for (0..m.ops_len) |k| try self.encode(ops[m.first + k], row0, rows);
            }
        },
        .controller_merge => |i| {
            const l0 = ops[i].linear;
            const l1 = ops[i + 1].linear;
            const m = ops[i + 2].inkling_message_merge;
            const w0 = self.bound[l0.w];
            const w1 = self.bound[l1.w];
            if (rows == 1 and w0.tensor.dtype == .bf16 and w0.quant == null and
                w1.tensor.dtype == .bf16 and w1.quant == null and
                m.out.width % 4 == 0 and w0.tensor.offset % 2 == 0 and w1.tensor.offset % 2 == 0)
            {
                const xin = self.words(l0.x);
                const hid = self.words(m.hidden);
                const msg = self.words(m.message);
                const out = self.words(m.out);
                p.w = w0.tensor.offset;
                p.s = w1.tensor.offset;
                p.x = xin.off;
                p.xs = xin.stride;
                p.y = hid.off;
                p.ys = hid.stride;
                p.z = msg.off;
                p.zs = msg.stride;
                p.every = out.off;
                p.pad_k = out.stride;
                p.n = l0.x.width;
                p.m = m.out.width;
                if (m.round == .bf16) p.flags |= gpu.flag_round_bf16;
                try gpu.dispatch(self.ctx, "mk_bf16_controller_merge", &.{ w0.buffer, w1.buffer, self.arena }, &p, ((@as(usize, p.m) + 3) / 4) * 32, self.knobs.linear_group);
            } else {
                self.ctx.nano_dispatch_flags = null;
                for (0..3) |k| try self.encode(ops[i + k], row0, rows);
            }
        },
        .bf16_projections => |m| {
            var cursor = m.first;
            var aligned = rows == 1 and self.knobs.bf16_linear_tile;
            var simds: usize = 0;
            var x: gpu.ParamsX = .{ .p = p, .count = m.count, .kind = m.round_mask };
            const l0 = ops[m.first].linear;
            const w0 = self.bound[l0.w];
            const xin = self.words(l0.x);
            x.p.x = xin.off;
            x.p.xs = xin.stride;
            x.p.n = l0.x.width;
            if (xin.off % 4 != 0 or xin.stride % 4 != 0 or x.p.n % 128 != 0) aligned = false;
            var wide8 = aligned and self.knobs.bf16_decode_wide and x.p.n % 256 == 0;
            for (0..m.count) |k| {
                const l = ops[cursor].linear;
                const w = self.bound[l.w];
                const out = self.words(l.out);
                if (w.tensor.dtype != .bf16 or w.quant != null or w.tensor.offset % 8 != 0 or l.out.width % 4 != 0) aligned = false;
                if (w.tensor.offset % 16 != 0) wide8 = false;
                x.mat[k] = .{ .w = w.tensor.offset, .m = l.out.width, .y = out.off, .ys = out.stride };
                simds += l.out.width / 4;
                const rounded = ((m.round_mask >> @intCast(k)) & 1) != 0;
                cursor += if (rounded) @as(u32, 2) else 1;
            }
            if (aligned and (self.knobs.bf16_decode_vec or wide8)) {
                const kname: [:0]const u8 = if (wide8) "mk_multi_bf16_4x1_vec8" else "mk_multi_bf16_4x1_vec";
                try gpu.dispatch(self.ctx, kname, &.{ w0.buffer, self.arena }, &x, simds * 32, self.knobs.linear_group);
            } else {
                self.ctx.nano_dispatch_flags = null;
                cursor = m.first;
                for (0..m.count) |k| {
                    const rounded = ((m.round_mask >> @intCast(k)) & 1) != 0;
                    try self.encodeRound(ops[cursor], row0, rows, rounded);
                    cursor += if (rounded) @as(u32, 2) else 1;
                }
            }
        },
        .head_softcap => |i| {
            const l = ops[i].linear;
            const cap = ops[i + 2].row_transform;
            const w = self.bound[l.w];
            const xin = self.words(l.x);
            const out = self.words(cap.out);
            const vector = l.x.width % 128 == 0 and xin.off % 4 == 0 and xin.stride % 4 == 0 and w.tensor.offset % 8 == 0;
            if (rows == 1 and self.knobs.bf16_linear_tile and w.tensor.dtype == .bf16 and w.quant == null and vector and (self.knobs.bf16_decode_vec or self.knobs.bf16_decode_wide)) {
                p.w = w.tensor.offset;
                p.x = xin.off;
                p.xs = xin.stride;
                p.y = out.off;
                p.ys = out.stride;
                p.n = l.x.width;
                p.m = cap.out.width;
                p.scale = cap.transform.softcap;
                const wide8 = self.knobs.bf16_decode_wide and p.n % 256 == 0 and w.tensor.offset % 16 == 0;
                const kname: [:0]const u8 = if (wide8) "mk_bf16_linear_4x1_softcap_vec8" else "mk_bf16_linear_4x1_softcap";
                try gpu.dispatch(self.ctx, kname, &.{ w.buffer, self.arena }, &p, ((@as(usize, p.m) + 3) / 4) * 32, self.knobs.linear_group);
            } else {
                self.ctx.nano_dispatch_flags = null;
                for (0..3) |k| try self.encode(ops[i + k], row0, rows);
            }
        },
        .controller_depth_norms => |i| {
            const d = ops[i].depth_attention;
            const n1 = ops[i + 1].rmsnorm;
            const bank = self.words(d.bank);
            const partial = self.words(d.partial);
            const msg = self.words(d.out);
            const inp = self.words(n1.x);
            const cat0 = self.words(n1.out);
            const query = self.bound[d.query];
            const width = d.partial.width;
            const heads = d.bank.width / width + 1;
            const q_off = query.tensor.offset + @as(usize, d.query_index) * width * query.tensor.dtype.byteSize();
            if (heads <= 8 and query.tensor.dtype == .f32 and q_off % 16 == 0 and
                bank.off % 4 == 0 and bank.stride % 4 == 0 and partial.off % 4 == 0 and partial.stride % 4 == 0 and
                msg.off % 4 == 0 and msg.stride % 4 == 0 and inp.off % 4 == 0 and inp.stride % 4 == 0 and
                cat0.off % 4 == 0 and cat0.stride % 4 == 0 and width % 4 == 0 and
                (width == inkling_plan.hidden or width <= 1024))
            {
                p.w = q_off;
                p.dtype = @backingInt(query.tensor.dtype);
                p.x = bank.off;
                p.xs = bank.stride;
                p.y = partial.off;
                p.ys = partial.stride;
                p.z = msg.off;
                p.zs = msg.stride;
                p.every = inp.off;
                p.pad_k = inp.stride;
                p.rot = cat0.off;
                p.kind = cat0.stride;
                p.n = width;
                p.m = heads - 1;
                p.heads = heads;
                p.eps = n1.eps;
                p.scale = d.scale;
                p.flags = gpu.flag_round_out;
                const tg: usize = if (width == inkling_plan.hidden) inkling_plan.depth_reg_threads else inkling_plan.depth_vec_threads;
                const kname: [:0]const u8 = if (width == inkling_plan.hidden) "mk_controller_depth_norms_2304" else "mk_controller_depth_norms_vec";
                try gpu.dispatch(self.ctx, kname, &.{ self.arena, query.buffer }, &p, @as(usize, rows) * tg, tg);
            } else {
                self.ctx.nano_dispatch_flags = null;
                for (0..5) |k| try self.encode(ops[i + k], row0, rows);
            }
        },
        .depth_norm => |m| {
            const d = ops[m.first].depth_attention;
            const n = ops[m.first + 1].rmsnorm;
            const bank = self.words(d.bank);
            const partial = self.words(d.partial);
            const out = self.words(n.out);
            const query = self.bound[d.query];
            const width = d.partial.width;
            const heads = d.bank.width / width + 1;
            const q_off = query.tensor.offset + @as(usize, d.query_index) * width * query.tensor.dtype.byteSize();
            const w_ok = if (n.weight) |id| ((self.bound[id].tensor.dtype == .bf16 and self.bound[id].tensor.offset % 8 == 0) or
                (self.bound[id].tensor.dtype == .f32 and self.bound[id].tensor.offset % 16 == 0)) else true;
            if (heads <= 8 and query.tensor.dtype == .f32 and q_off % 16 == 0 and w_ok and
                bank.off % 4 == 0 and bank.stride % 4 == 0 and partial.off % 4 == 0 and partial.stride % 4 == 0 and
                out.off % 4 == 0 and out.stride % 4 == 0 and width % 4 == 0 and
                (width == inkling_plan.hidden or width <= 1024))
            {
                var wbuf = self.arena;
                if (n.weight) |id| {
                    const w = self.bound[id];
                    p.s = w.tensor.offset;
                    p.sdtype = @backingInt(w.tensor.dtype);
                    p.flags |= gpu.flag_has_weight;
                    wbuf = w.buffer;
                }
                if (n.round_normalized == .bf16) p.flags |= gpu.flag_round_bf16;
                if (m.round_out) p.flags |= gpu.flag_round_out;
                p.w = q_off;
                p.dtype = @backingInt(query.tensor.dtype);
                p.x = bank.off;
                p.xs = bank.stride;
                p.y = partial.off;
                p.ys = partial.stride;
                p.z = out.off;
                p.zs = out.stride;
                p.n = width;
                p.m = heads - 1;
                p.heads = heads;
                p.eps = n.eps;
                p.scale = d.scale;
                const tg: usize = if (width == inkling_plan.hidden) inkling_plan.depth_reg_threads else inkling_plan.depth_vec_threads;
                const kname: [:0]const u8 = if (width == inkling_plan.hidden) "mk_depth_norm_2304" else "mk_depth_norm_vec";
                try gpu.dispatch(self.ctx, kname, &.{ self.arena, query.buffer, wbuf }, &p, @as(usize, rows) * tg, tg);
            } else {
                self.ctx.nano_dispatch_flags = null;
                for (0..m.ops_len) |k| try self.encode(ops[m.first + k], row0, rows);
            }
        },
        .inkling_conv => |i| {
            const c = ops[i + 1].inkling_short_conv;
            const src_state = self.program.states[c.src];
            const capacity = src_state.capacity(self.batch, self.context);
            if (rows == 1 and self.knobs.conv_ring and capacity >= c.taps) {
                const w = self.bound[c.weight];
                const residual = self.words(c.residual);
                const o = self.words(c.out);
                p.w = w.tensor.offset;
                p.dtype = @backingInt(w.tensor.dtype);
                p.x = residual.off;
                p.xs = residual.stride;
                p.z = o.off;
                p.zs = o.stride;
                p.n = c.out.width;
                p.taps = c.taps;
                p.capacity = capacity;
                p.scale = 1.0 / @sqrt(@as(f32, @floatFromInt(c.taps + 1)));
                p.flags = if (c.round == .bf16) gpu.flag_round_bf16 else 0;
                try gpu.dispatch(self.ctx, "mk_store_inkling_short_conv_ring", &.{ self.caches[c.src].?, w.buffer, self.arena }, &p, p.n, 256);
            } else {
                self.ctx.nano_dispatch_flags = null;
                try self.encode(ops[i], row0, rows);
                try self.encode(ops[i + 1], row0, rows);
            }
        },
        .relative_gate => |i| {
            const a = ops[i].relative_attention;
            const b = ops[i + 2].binary;
            const visible = if (a.window == 0) self.position + row0 + rows else @min(self.position + row0 + rows, a.window);
            if (self.tap == null and rows == 1 and self.knobs.relative_long_fast and self.bias16 == null) {
                const q = self.words(a.q);
                const bias = self.words(a.bias);
                const sink = self.words(a.sink);
                p.x = q.off;
                p.xs = q.stride;
                p.z = bias.off;
                p.zs = bias.stride;
                p.s = sink.off;
                p.every = sink.stride;
                p.heads = a.heads;
                p.kv_heads = a.kv_heads;
                p.dim = a.dim;
                p.m = a.relative_extent;
                p.window = a.window;
                p.scale = a.scale;
                p.capacity = self.program.states[a.keys].capacity(self.batch, self.context);
                if (try encodeRelativeStriped(self, a, b.out, b.b, &p, visible)) return;
            }
            self.ctx.nano_dispatch_flags = null;
            try self.encode(ops[i], row0, rows);
            try self.encode(ops[i + 1], row0, rows);
            if (self.knobs.fuse_bf16_cast) {
                try self.encodeRound(ops[i + 2], row0, rows, true);
            } else {
                try self.encode(ops[i + 2], row0, rows);
                try self.encode(ops[i + 3], row0, rows);
            }
        },
    }
}

fn encodeRelativeStriped(self: *Engine, a: @FieldType(ir.Op, "relative_attention"), out_view: ir.View, gate_view: ?ir.View, p: *gpu.Params, visible: u32) !bool {
    const hs = self.relative_half orelse return false;
    const half_keys = hs[a.keys] orelse return false;
    if (a.dim != 128 or a.heads != a.kv_heads * 3) return false;
    const fast = self.knobs.relative_long_fast;
    p.taps = Engine.stripeParts(visible, fast);
    if (p.taps > Engine.max_stripe_parts) return false;
    const parts = self.stripe_parts.?;
    if (gate_view != null and !fast) return false;
    const keys_kernel: [:0]const u8 = if (fast) "mk_relative_long_keys_fast" else "mk_relative_long_keys";
    const threads: usize = if (fast) 256 else Engine.stripe_threads;
    try gpu.dispatch(self.ctx, keys_kernel, &.{ self.arena, half_keys, hs[a.values].?, parts }, p, @as(usize, a.kv_heads) * p.taps * threads, threads);
    const o = self.words(out_view);
    p.y = o.off;
    p.ys = o.stride;
    if (gate_view) |gv| {
        const g = self.words(gv);
        p.z = g.off;
        p.zs = g.stride;
        p.flags |= gpu.flag_round_bf16 | gpu.flag_round_out;
        try gpu.dispatch(self.ctx, "mk_relative_long_merge_gated", &.{ self.arena, parts }, p, @as(usize, a.heads) * 128, 128);
    } else {
        const merge_kernel: [:0]const u8 = if (fast) "mk_relative_long_merge_fast" else "mk_relative_long_merge";
        try gpu.dispatch(self.ctx, merge_kernel, &.{ self.arena, parts }, p, @as(usize, a.heads) * 128, 128);
    }
    return true;
}

// ---- decode search ----

/// Split decode attention (knob decode_attention = 2 or 3, decode.metal): one row,
/// head dim up to 256 with ceil(dim / 32) in {1, 2, 4, 8}, up to 8 query heads per KV head.
/// Returns false (caller encodes the reference kernel) when it does not apply.
fn encodeDecodeAttention(self: *Engine, a: @FieldType(ir.Op, "attention"), gate: ?@FieldType(ir.Op, "binary"), row0: u32, rows: u32) !bool {
    if (self.knobs.decode_attention == 0 or rows != 1 or self.scores == null) return false;
    if (a.dim == 0 or a.dim > 256 or a.heads % a.kv_heads != 0) return false;
    const E = (a.dim + 31) / 32;
    const G = a.heads / a.kv_heads;
    if (!pow2upto8(E) or !pow2upto8(G)) return false;
    const k = self.program.states[a.keys];
    const capacity = k.capacity(self.batch, self.context);
    const position = self.position + row0;
    const start: u32 = if (a.window != 0 and position + 1 > a.window) position + 1 - a.window else 0;
    const span = position - start + 1;
    if (span > self.scores_stride or @as(usize, a.heads) * self.scores_stride > self.scores.?.floats().len) return false;
    const q = self.words(a.q);
    var p: gpu.Params = .{ .rows = 1, .row0 = row0, .pos = self.position };
    p.x = q.off;
    p.xs = q.stride;
    p.heads = a.heads;
    p.kv_heads = a.kv_heads;
    p.dim = a.dim;
    p.window = a.window;
    p.scale = a.scale;
    p.capacity = capacity;
    p.every = self.scores_stride;
    const chunks = (span + dec_chunk - 1) / dec_chunk;
    p.m = chunks;
    const names = comptime blk: {
        var n: [4][4][:0]const u8 = undefined;
        for (.{ 1, 2, 4, 8 }, 0..) |e, i| for (.{ 1, 2, 4, 8 }, 0..) |g, j| {
            n[i][j] = std.fmt.comptimePrint("mk_dec_scores_e{d}_g{d}", .{ e, g });
        };
        break :blk n;
    };
    const scores = self.scores.?;
    const linear = self.knobs.decode_attention == 3 and a.window == 0 and position < capacity and
        k.kind == .sequence and self.program.states[a.values].kind == .sequence and G == 4 and (a.dim == 64 or a.dim == 128);
    const scores_name = if (linear)
        (if (a.dim == 64) "mk_dec_scores_linear_e2_g4" else "mk_dec_scores_linear_e4_g4")
    else
        names[std.math.log2_int(u32, E)][std.math.log2_int(u32, G)];
    try gpu.dispatch(self.ctx, scores_name, &.{ self.arena, self.caches[a.keys].?, scores }, &p, @as(usize, a.kv_heads) * chunks * 32, 256);
    const o = if (gate) |b| self.words(b.out) else self.words(a.out);
    p.y = o.off;
    p.ys = o.stride;
    if (gate) |b| {
        const g = self.words(b.b);
        p.z = g.off;
        p.zs = g.stride;
        p.act_b = @backingInt(b.b_act);
        p.flags |= gpu.flag_dec_gated;
    }
    try gpu.dispatch(self.ctx, if (linear) "mk_dec_chain_linear" else "mk_dec_chain_block", &.{ self.arena, self.caches[a.values].?, scores }, &p, @as(usize, a.heads) * E * 256, 256);
    return true;
}
// ---- climb 2, exact prefill ----

/// Batched exact attention (knob x2_attention, exact2.metal): scores, softmax
/// scalars, value chains and merge for rows > 1, head dim 97..128 and 4 query heads
/// per KV head (the compiled variants); otherwise false (the reference runs).
fn encodeX2Attention(self: *Engine, a: @FieldType(ir.Op, "attention"), gate: ?@FieldType(ir.Op, "binary"), row0: u32, rows: u32) !bool {
    const bufs = self.x2att orelse return false;
    if (rows < 2 or !exact2.attentionApplies(a.heads, a.kv_heads, a.dim)) return false;
    const capacity = self.program.states[a.keys].capacity(self.batch, self.context);
    const span = if (a.window != 0) @min(a.window, capacity) else capacity;
    if (span > self.x2att_stride or a.heads * self.x2att_stride * @as(usize, self.x2att_rows) > bufs[0].floats().len) return false;
    const q = self.words(a.q);
    const o = if (gate) |b| self.words(b.out) else self.words(a.out);
    var p: gpu.Params = .{ .rows = rows, .row0 = row0, .pos = self.position, .x = q.off, .xs = q.stride, .y = o.off, .ys = o.stride, .heads = a.heads, .kv_heads = a.kv_heads, .dim = a.dim, .window = a.window, .capacity = capacity, .scale = a.scale, .every = self.x2att_stride };
    if (gate) |b| {
        const g = self.words(b.b);
        p.z = g.off;
        p.zs = g.stride;
        p.act_b = @backingInt(b.b_act);
        p.flags |= exact2.att_gated;
    }
    try exact2.encodeAttention(self.ctx, self.arena, self.caches[a.keys].?, self.caches[a.values].?, bufs, &p, self.x2att_rows);
    return true;
}

/// Fused attention prep (knob fuse_attn_prep, plan.matchAttnPrep): mk_dec_attn_prep.
fn encodeAttnPrep(self: *Engine, i: u32, row0: u32, rows: u32) !void {
    const ops = self.program.ops;
    const n1 = ops[i].rmsnorm;
    const n2 = ops[i + 1].rmsnorm;
    const r1 = ops[i + 2].rope;
    const r2 = ops[i + 3].rope;
    const s1 = ops[i + 4].store;
    const s2 = ops[i + 5].store;
    const wq = self.bound[n1.weight.?];
    const wk = self.bound[n2.weight.?];
    const q = self.words(n1.x);
    const k = self.words(n2.x);
    const v = self.words(s2.src);
    var p: gpu.Params = .{ .rows = rows, .row0 = row0, .pos = self.position };
    p.x = q.off;
    p.xs = q.stride;
    p.y = k.off;
    p.ys = k.stride;
    p.z = v.off;
    p.zs = v.stride;
    p.heads = r1.heads;
    p.kv_heads = r2.heads;
    p.dim = r1.dim;
    p.rot = r1.rot;
    p.n = self.table_offsets[r1.freqs];
    p.eps = n1.eps;
    p.w = wq.tensor.offset;
    p.dtype = @backingInt(wq.tensor.dtype);
    p.s = wk.tensor.offset;
    p.sdtype = @backingInt(wk.tensor.dtype);
    p.capacity = self.program.states[s1.dst].capacity(self.batch, self.context);
    if (r1.fast_trig) p.flags |= gpu.flag_fast_trig;
    const tg = gpu.normThreads(r1.dim);
    const units = @as(usize, r1.heads) + 2 * @as(usize, r2.heads);
    try gpu.dispatch(self.ctx, "mk_dec_attn_prep", &.{ wq.buffer, wk.buffer, self.arena, self.tables, self.caches[s1.dst].?, self.caches[s2.dst].? }, &p, @as(usize, rows) * units * tg, tg);
}

// ---- decode climb 2 ----

/// Knob d2_gate. Forward one token at self.position: if its forward is already
/// committed and waiting at the gate, write the token with its generation and
/// open the gate; otherwise encode and commit it now. Then encode and commit the
/// next position's forward behind a new gate, spin until this token's argmax
/// word is written (the argmax is the forward's last launch, so every launch has
/// run), and return. Buffer failures surface at the next reap or settle. A
/// non-finite result whose embedding saw a stale generation (the gate gave up
/// before the host opened it) is recomputed synchronously.
fn d2GatedForward(self: *Engine, token: u32) !void {
    const gate = self.d2_gate_buf.?;
    const gw = gate.structs(u32);
    const nx = self.words(self.program.next);
    const watch: *volatile u32 = @ptrCast(&self.arena.structs(u32)[nx.off]);
    const sentinel: u32 = 0xfffffffe;
    const pos = self.position;
    const before = self.ctx.stats();
    const t_begin = io.now() catch 0;
    // Launches of the forward that computes this token (encoded now or by the previous call).
    var own_dispatches: u64 = self.d2_ahead_dispatches;
    if (self.d2_ahead != null and self.d2_ahead.? != pos) try self.d2Settle();
    watch.* = sentinel;
    if (self.d2_ahead) |_| {
        @atomicStore(u32, &gw[0], self.d2_ahead_gen << @intCast(inkling_plan.d2_gen_shift) | token, .release);
        self.d2_ahead = null;
    } else {
        const tok = self.words(self.program.tokens);
        self.arena.structs(u32)[tok.off] = token;
        const s0 = self.ctx.stats();
        try d2Encode(self, pos, null);
        try gpu.commitAsync(self.ctx);
        own_dispatches = self.ctx.stats().dispatches - s0.dispatches;
    }
    const t_encoded = io.now() catch 0;
    // The next position's forward, behind a gate.
    const ahead = pos + 1 < self.context;
    if (ahead) {
        const s0 = self.ctx.stats();
        self.d2_gen = (self.d2_gen + 1) & 0xfff;
        try d2Encode(self, pos + 1, self.d2_gen);
        try gpu.commitAsync(self.ctx);
        self.d2_ahead = pos + 1;
        self.d2_ahead_gen = self.d2_gen;
        self.d2_ahead_dispatches = self.ctx.stats().dispatches - s0.dispatches;
    }
    // This token's buffer is the newest one not waiting at the gate.
    const n = gpu.pending(self.ctx);
    try gpu.spin(self.ctx, n - 1 - @intFromBool(ahead), watch, sentinel);
    const t_done = io.now() catch 0;
    try gpu.reap(self.ctx);
    var result = watch.*;
    if (result >= sentinel) {
        // Non-finite logits, or no argmax written after the sentinel (the forward
        // ran before the host opened its gate): wait for the buffer, then decide.
        try gpu.drain(self.ctx, @intFromBool(ahead));
        if (@atomicLoad(u32, &gw[1], .acquire) != 0) {
            @atomicStore(u32, &gw[1], 0, .release);
            try self.d2Settle();
            self.d2_redos += 1;
            watch.* = sentinel;
            const tok = self.words(self.program.tokens);
            self.arena.structs(u32)[tok.off] = token;
            try d2Encode(self, pos, null);
            try self.ctx.finish();
        }
        result = watch.*;
        if (result == sentinel) return error.GateProtocol;
    }
    self.d2_last = result;
    const after = self.ctx.stats();
    self.last = .{ .dispatches = own_dispatches, .submissions = 1, .gpu_seconds = after.gpu_seconds - before.gpu_seconds, .encode_seconds = t_encoded - t_begin, .wait_seconds = t_done - t_encoded };
    self.position = pos + 1;
    self.out_first = 0;
    self.out_end = 1;
}

/// Knob d2_gate is used only when a forward run past the last read token (the one
/// waiting at the gate, run on a stale or repeated token) cannot clobber a ring
/// row that a recomputation of that token still reads: every ring that wraps
/// keeps at least one row beyond its horizon plus the current row (capacity >=
/// horizon + 2, i.e. batch >= 2). Otherwise the engine decodes ungated.
fn d2GateSafe(p: *const ir.Program, batch: u32, context: u32) bool {
    for (p.states) |st| if (st.kind == .ring) {
        const cap = st.capacity(batch, context);
        if (cap < context and cap < st.horizon + 2) return false;
    };
    return true;
}

/// Encode the whole program for one row at position `pos`; `gen` set: a gate
/// first, and the embedding reads the gate word (mk_d2_embed_gated).
fn d2Encode(self: *Engine, pos: u32, gen: ?u32) !void {
    const saved = self.position;
    self.position = pos;
    defer self.position = saved;
    self.prep_key = null;
    self.x2_key = null; // exact prefill's prepared inputs: single-row forwards never use them
    self.expert_routes = null;
    self.bias16 = null;
    if (gen) |g| {
        if (self.independent_launches != null) self.ctx.nano_dispatch_flags = gpu.d2_concurrent;
        defer self.ctx.nano_dispatch_flags = null;
        var p: gpu.Params = .{ .every = g, .n = self.knobs.d2_gate_spins };
        try gpu.dispatch(self.ctx, "mk_d2_gate_packed", &.{self.d2_gate_buf.?}, &p, 32, 32);
    }
    for (self.steps, 0..) |step, step_i| {
        if (self.independent_launches) |v| {
            // When gated, step 0 needs a barrier after mk_d2_gate_packed before running.
            if ((gen == null and step_i == 0) or (step_i > 0 and v[step_i])) self.ctx.nano_dispatch_flags = gpu.d2_concurrent;
        }
        defer self.ctx.nano_dispatch_flags = null;
        switch (step) {
            .op => |i| {
                const op = self.program.ops[i];
                if (gen != null and op == .embed) try d2EncodeEmbed(self, op.embed, gen.?) else try self.encode(op, 0, 1);
            },
            else => {
                self.expert_routes = null;
                try encodeFused(self, step, 0, 1);
            },
        }
    }
}

fn d2EncodeEmbed(self: *Engine, e: @FieldType(ir.Op, "embed"), gen: u32) !void {
    var p: gpu.Params = .{ .rows = 1, .row0 = 0, .pos = self.position };
    const w = self.bound[e.table];
    Engine.quantParams(&p, w);
    const o = self.words(e.out);
    p.y = o.off;
    p.ys = o.stride;
    p.n = e.out.width;
    p.every = gen;
    if (e.round == .bf16) p.flags |= gpu.flag_round_bf16;
    const qb = Engine.quantBuffers(w);
    try gpu.dispatch(self.ctx, "mk_d2_embed_gated", &.{ qb[0], qb[1], qb[2], self.arena, self.d2_gate_buf.? }, &p, p.n, 256);
}

// ---- prefill-mlx (exact) ----

/// knobs.scratch_reuse: arena offsets (words) for the tokens and scratch states,
/// sharing words between states that are never live together; returns the arena
/// length. A state is live from the first op that touches it to the last, plus
/// `reuse_margin` ops: with knobs.concurrent_scratch up to 16 single-launch steps
/// (at most two ops each) run without a barrier between them, so a state's words
/// are handed on only once every launch that could still be using them has been
/// ordered before the next writer. The token ids, the logits and the argmax are
/// read by the host after the forward and keep their own words. Offsets stay
/// multiples of 64 words, as the unpacked layout's are.
pub fn packScratch(gpa: std.mem.Allocator, program: *const ir.Program, batch: u32, context: u32, offsets: []u32) !usize {
    const reuse_margin: u32 = 40;
    const never: u32 = std.math.maxInt(u32);
    const n = program.states.len;
    const first = try gpa.alloc(u32, n);
    defer gpa.free(first);
    const last = try gpa.alloc(u32, n);
    defer gpa.free(last);
    @memset(first, never);
    @memset(last, 0);
    for (program.ops, 0..) |op, i| {
        var reads: [3]?ir.View = undefined;
        var write: ?ir.View = null;
        plan.access(op, &reads, &write);
        for ([_]?ir.View{ reads[0], reads[1], reads[2], write }) |view| if (view) |v| {
            first[v.state] = @min(first[v.state], @as(u32, @intCast(i)));
            last[v.state] = @intCast(i);
        };
    }
    for ([_]ir.StateId{ program.tokens.state, program.logits.state, program.next.state }) |id| {
        first[id] = 0;
        last[id] = never;
    }
    for (program.traces) |t| last[t.view.state] = never; // read by the host during traced forwards
    const order = try gpa.alloc(u32, n);
    defer gpa.free(order);
    var count: usize = 0;
    for (program.states, 0..) |s, i| {
        offsets[i] = 0;
        if (s.kind == .tokens or s.kind == .scratch) {
            if (first[i] == never) { // touched by no op: its own words from op 0
                first[i] = 0;
                last[i] = never;
            }
            order[count] = @intCast(i);
            count += 1;
        }
    }
    const Ctx = struct {
        first: []const u32,
        fn lessThan(c: @This(), a: u32, b: u32) bool {
            return if (c.first[a] != c.first[b]) c.first[a] < c.first[b] else a < b;
        }
    };
    std.mem.sort(u32, order[0..count], Ctx{ .first = first }, Ctx.lessThan);
    const Block = struct { off: usize, len: usize, free_from: u32 };
    var live: std.ArrayList(Block) = .empty; // free_from: the first op that may take the words
    defer live.deinit(gpa);
    var free: std.ArrayList(Block) = .empty;
    defer free.deinit(gpa);
    var total: usize = 0;
    for (order[0..count]) |id| {
        const s = program.states[id];
        const need = (@as(usize, s.capacity(batch, context)) * s.width + 63) / 64 * 64;
        var k: usize = 0;
        while (k < live.items.len) {
            if (live.items[k].free_from <= first[id]) {
                try free.append(gpa, live.swapRemove(k));
            } else k += 1;
        }
        var best: ?usize = null;
        for (free.items, 0..) |b, j| {
            if (b.len >= need and (best == null or b.len < free.items[best.?].len)) best = j;
        }
        var off: usize = undefined;
        if (best) |j| {
            off = free.items[j].off;
            if (free.items[j].len == need) {
                _ = free.swapRemove(j);
            } else {
                free.items[j].off += need;
                free.items[j].len -= need;
            }
        } else {
            off = total;
            total += need;
        }
        offsets[id] = std.math.cast(u32, off) orelse return error.ArenaTooLarge;
        if (last[id] != never) try live.append(gpa, .{ .off = off, .len = need, .free_from = last[id] +| (reuse_margin + 1) });
    }
    if (total > std.math.maxInt(u32)) return error.ArenaTooLarge;
    return total;
}

test "packed scratch: states share words only when their lifetimes are a margin apart" {
    const inkling = @import("inkling2_loop.zig");
    var p = try inkling.build(std.testing.allocator, inkling.test_config);
    defer p.deinit();
    const gpa = std.testing.allocator;
    const offsets = try gpa.alloc(u32, p.states.len);
    defer gpa.free(offsets);
    const batch = 8;
    const total = try packScratch(gpa, &p, batch, 64, offsets);
    var plain: usize = 0;
    for (p.states) |s| if (s.kind == .tokens or s.kind == .scratch) {
        plain += (@as(usize, s.capacity(batch, 64)) * s.width + 63) / 64 * 64;
    };
    try std.testing.expect(total * 4 < plain);
    // Recompute every state's op span and check each pair that shares words.
    const first = try gpa.alloc(u32, p.states.len);
    defer gpa.free(first);
    const last = try gpa.alloc(u32, p.states.len);
    defer gpa.free(last);
    @memset(first, std.math.maxInt(u32));
    @memset(last, 0);
    for (p.ops, 0..) |op, i| {
        var reads: [3]?ir.View = undefined;
        var write: ?ir.View = null;
        plan.access(op, &reads, &write);
        for ([_]?ir.View{ reads[0], reads[1], reads[2], write }) |view| if (view) |v| {
            first[v.state] = @min(first[v.state], @as(u32, @intCast(i)));
            last[v.state] = @intCast(i);
        };
    }
    // Sweep by offset: an interval tree is not needed, the live set at any word is small.
    const Span = struct { off: usize, end: usize, id: u32 };
    var spans: std.ArrayList(Span) = .empty;
    defer spans.deinit(gpa);
    for (p.states, 0..) |s, i| if (s.kind == .tokens or s.kind == .scratch) {
        const len = @as(usize, s.capacity(batch, 64)) * s.width;
        try std.testing.expect(offsets[i] % 64 == 0 and offsets[i] + len <= total);
        try spans.append(gpa, .{ .off = offsets[i], .end = offsets[i] + len, .id = @intCast(i) });
    };
    const byOff = struct {
        fn lessThan(_: void, a: Span, b: Span) bool {
            return a.off < b.off;
        }
    }.lessThan;
    std.mem.sort(Span, spans.items, {}, byOff);
    var shared: usize = 0;
    for (spans.items, 0..) |a, i| {
        for (spans.items[i + 1 ..]) |b| {
            if (b.off >= a.end) break;
            shared += 1;
            for ([_]ir.StateId{ p.tokens.state, p.logits.state, p.next.state }) |keep| {
                try std.testing.expect(a.id != keep and b.id != keep);
            }
            const gap = if (first[a.id] > last[b.id]) first[a.id] - last[b.id] else if (first[b.id] > last[a.id]) first[b.id] - last[a.id] else 0;
            try std.testing.expect(gap > 40);
        }
    }
    try std.testing.expect(shared > 0);
}

/// knobs.tail_trim: the rows of a forward of T rows that an op has to compute: the
/// last `tail`, and, with `logits`, the rows with logits and `before` rows ahead of
/// the first of them.
pub const RowNeed = struct {
    tail: u32 = 0,
    logits: bool = false,
    before: u32 = 0,

    pub const all: RowNeed = .{ .tail = std.math.maxInt(u32) };

    pub fn merge(a: RowNeed, b: RowNeed) RowNeed {
        return .{ .tail = @max(a.tail, b.tail), .logits = a.logits or b.logits, .before = @max(if (a.logits) a.before else 0, if (b.logits) b.before else 0) };
    }
    /// The need of an input an op reads up to `rows` rows behind each row it computes.
    pub fn back(a: RowNeed, rows: u32) RowNeed {
        return .{ .tail = if (a.tail == 0) 0 else a.tail +| rows, .logits = a.logits, .before = if (a.logits) a.before +| rows else 0 };
    }
    /// The first row to compute (T: none) when rows [out_first, T) have logits and a
    /// trimmed op keeps at least `floor` of the last rows.
    pub fn firstRow(a: RowNeed, T: u32, out_first: u32, floor: u32) u32 {
        const from_tail = if (a.tail == 0) T else T -| @max(a.tail, floor);
        return @min(from_tail, if (a.logits) out_first -| a.before else T);
    }
};

/// knobs.tail_trim: per op, the rows something still reads. Walking the program
/// backwards, a state's need is the largest any later reader has of it; an op needs
/// what its output's state does. A sequence cache needs every row. A ring cache
/// needs its horizon's worth of the last rows for the forwards that follow, and what
/// this forward's own readers reach back to: a convolution reads `taps - 1` rows
/// behind each row it computes, windowed attention `window - 1`. Head ops run on the
/// rows with logits. Any other op computes each row from the same row of its inputs;
/// an op kind not listed here keeps every row, and so do its inputs.
pub fn rowNeeds(gpa: std.mem.Allocator, p: *const ir.Program) ![]RowNeed {
    const needs = try gpa.alloc(RowNeed, p.ops.len);
    errdefer gpa.free(needs);
    const demand = try gpa.alloc(RowNeed, p.states.len);
    defer gpa.free(demand);
    for (p.states, demand) |s, *d| d.* = switch (s.kind) {
        .sequence => RowNeed.all,
        .ring => .{ .tail = s.horizon },
        else => .{},
    };
    for ([_]ir.View{ p.logits, p.next }) |v| demand[v.state] = demand[v.state].merge(.{ .logits = true });
    var i = p.ops.len;
    while (i > 0) {
        i -= 1;
        const op = p.ops[i];
        var reads: [3]?ir.View = undefined;
        var write: ?ir.View = null;
        plan.access(op, &reads, &write);
        const row_wise = switch (op) {
            .embed, .rmsnorm, .linear, .binary, .concat, .activate, .row_transform, .grouped_project, .softcapped_swiglu, .l2_clip, .depth_attention, .inkling_message_merge, .rope, .route_topk, .expert_gate_up, .expert_down, .argmax => true,
            else => false,
        };
        var need: RowNeed = if (write) |w| demand[w.state] else .{};
        if (i >= p.head_start) need = if (row_wise) .{ .logits = true } else RowNeed.all;
        switch (op) {
            .store => |s| need = demand[s.dst],
            .inkling_short_conv => |c| demand[c.src] = demand[c.src].merge(need.back(c.taps - 1)),
            .inkling_dynamic_conv => |c| demand[c.src] = demand[c.src].merge(need.back(c.taps - 1)),
            .relative_attention => |a| for ([_]ir.StateId{ a.keys, a.values }) |id| {
                demand[id] = demand[id].merge(if (a.window == 0) RowNeed.all else need.back(a.window - 1));
            },
            .conv => |c| {
                need = RowNeed.all;
                demand[c.src] = RowNeed.all;
            },
            .attention => |a| {
                need = RowNeed.all;
                demand[a.keys] = RowNeed.all;
                demand[a.values] = RowNeed.all;
            },
            else => if (!row_wise) {
                need = RowNeed.all;
            },
        }
        needs[i] = need;
        for (reads) |r| if (r) |v| {
            demand[v.state] = demand[v.state].merge(need);
        };
    }
    return needs;
}

test "tail_trim: only the last visit's attention output and feed-forward are trimmed" {
    const inkling = @import("inkling2_loop.zig");
    var p = try inkling.build(std.testing.allocator, inkling.test_config);
    defer p.deinit();
    const needs = try rowNeeds(std.testing.allocator, &p);
    defer std.testing.allocator.free(needs);
    const all = std.math.maxInt(u32);
    var attentions: usize = 0;
    var feed_forwards: usize = 0;
    var first_trimmed: ?usize = null;
    var last_cache_store: usize = 0;
    for (p.ops, needs, 0..) |op, need, i| {
        if (i < p.head_start and need.tail != all and first_trimmed == null) first_trimmed = i;
        switch (op) {
            .store => |s| if (p.states[s.dst].kind == .sequence) {
                try std.testing.expectEqual(@as(u32, all), need.tail);
                last_cache_store = i;
            },
            .relative_attention => {
                attentions += 1;
                // 48 visits: the last one's output feeds a 4-tap convolution whose
                // output feeds the feed-forward's, so six rows back from each end.
                if (attentions == 48) try std.testing.expectEqual(RowNeed{ .tail = 6, .logits = true, .before = 6 }, need) else try std.testing.expectEqual(@as(u32, all), need.tail);
            },
            .expert_down => {
                feed_forwards += 1;
                if (feed_forwards == 48) try std.testing.expectEqual(RowNeed{ .tail = 3, .logits = true, .before = 3 }, need) else try std.testing.expectEqual(@as(u32, all), need.tail);
            },
            else => {},
        }
    }
    try std.testing.expectEqual(@as(usize, 48), attentions);
    // Nothing is trimmed before the last visit: its first trimmed op comes after
    // the 47th visit's feed-forward, and its K and V are stored for every row.
    var seen: usize = 0;
    for (p.ops[0..first_trimmed.?]) |op| seen += @intFromBool(op == .expert_down);
    try std.testing.expectEqual(@as(usize, 47), seen);
    try std.testing.expect(last_cache_store > first_trimmed.?);
    // The first row an op computes: a chunk without logits, the last chunk of a
    // prompt, a single decode row, and a floor that keeps full-chunk kernels.
    const need: RowNeed = .{ .tail = 6, .logits = true, .before = 6 };
    try std.testing.expectEqual(@as(u32, 4090), need.firstRow(4096, 4096, 1));
    try std.testing.expectEqual(@as(u32, 4089), need.firstRow(4096, 4095, 1));
    try std.testing.expectEqual(@as(u32, 0), need.firstRow(1, 0, 1));
    try std.testing.expectEqual(@as(u32, 3584), need.firstRow(4096, 4095, 512));
    try std.testing.expectEqual(@as(u32, 0), need.firstRow(4096, 0, 512));
    try std.testing.expectEqual(@as(u32, 0), RowNeed.all.firstRow(4096, 4096, 1));
    try std.testing.expectEqual(@as(u32, 4096), (RowNeed{}).firstRow(4096, 4096, 512));
    try std.testing.expectEqual(@as(u32, 7), (RowNeed{ .logits = true }).firstRow(8, 7, 512));
}

/// knobs.head_trim: one past the last op that writes a persistent state (a store):
/// what follows feeds only the logits, so rows without logits can skip it (stock
/// MLX's chunked prefill evaluates only the caches, so its lazy graph skips the same
/// ops). Every row it still runs keeps its own expression tree.
pub fn trimmedHead(p: *const ir.Program) u32 {
    var last: ?usize = null;
    for (p.ops, 0..) |op, i| if (op == .store) {
        last = i;
    };
    const t: u32 = if (last) |i| @intCast(i + 1) else 0;
    return @min(t, p.head_start);
}

/// Replace any fused step that straddles `boundary` by its ops, one step each
/// (the reference schedule for those ops); frees `steps`.
fn splitAcross(gpa: std.mem.Allocator, steps: []plan.Step, boundary: u32) ![]plan.Step {
    defer gpa.free(steps);
    var out: std.ArrayList(plan.Step) = .empty;
    errdefer out.deinit(gpa);
    for (steps) |st| {
        const f = st.first();
        const n = st.len();
        if (std.meta.activeTag(st) != .op and f < boundary and f + n > boundary) {
            for (0..n) |k| try out.append(gpa, .{ .op = f + @as(u32, @intCast(k)) });
        } else try out.append(gpa, st);
    }
    return out.toOwnedSlice(gpa);
}

test "stripe records are sized for every key count up to the context" {
    for ([_]bool{ false, true }) |fast| {
        for ([_]u32{ 1, 127, 2048, 2049, 4120, 16384, 16385, 20000, 32792, 70000 }) |context| {
            var most: u32 = 0;
            var visible: u32 = 1;
            while (visible <= context) : (visible += 1) most = @max(most, Engine.stripeParts(visible, fast));
            try std.testing.expectEqual(most, Engine.maxStripeParts(context, fast));
        }
    }
}

test "head_trim: boundary after the last store; split steps cover every op once and straddle neither boundary" {
    const lfm3 = @import("lfm3.zig");
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const config =
        \\{"model_type":"lfm3","hidden_size":64,"num_hidden_layers":3,"layer_types":["conv","full_attention","conv"],
        \\"num_attention_heads":4,"num_key_value_heads":2,"head_dim":32,"intermediate_size":96,"conv_L_cache":3,"norm_eps":1e-5,
        \\"vocab_size":40,"eos_token_id":[7],"sliding_window":4,"max_position_embeddings":128,
        \\"rope_parameters":{"full_attention":{"partial_rotary_factor":0.5,"rope_theta":1000000.0},"sliding_attention":{"rope_theta":10000}}}
    ;
    var p = try lfm3.build(std.testing.allocator, config, .{ .context = undefined, .has = struct {
        fn f(_: *const anyopaque, _: []const u8) bool {
            return true;
        }
    }.f }, .{});
    defer p.deinit();
    const t = trimmedHead(&p);
    try std.testing.expect(t < p.head_start);
    try std.testing.expect(p.ops[t - 1] == .store);
    for (p.ops[t..]) |op| try std.testing.expect(op != .store);
    var planned = p;
    planned.head_start = t;
    const info = try a.alloc(plan.WeightInfo, p.weights.len);
    @memset(info, .{ .q4 = true, .buffer = 1 });
    const steps = try splitAcross(a, try plan.build(a, &planned, .{ .projections = true, .glu = true, .norms = true, .conv = true, .gate = true }, info), p.head_start);
    var next: u32 = 0;
    for (steps) |st| {
        try std.testing.expectEqual(next, st.first());
        const end = st.first() + st.len();
        for ([_]u32{ t, p.head_start }) |b| try std.testing.expect(!(st.first() < b and end > b));
        next = end;
    }
    try std.testing.expectEqual(@as(u32, @intCast(p.ops.len)), next);
}

// ---- prefill-mlx (mlx-arith tier) ----

/// Words of the split-K partials: split * M * N <= 512 threadgroups x 32 x 32 (mlxSplitK).
pub const mlx_splitk_words: usize = 512 * 32 * 32;

/// MLX runs its steel attention (ported: mk_mlx_attention*) for float32 queries
/// longer than 8 rows; head dim 128 is the instantiation ported here.
pub fn mlxAttentionApplies(rows: u32, heads: u32, kv_heads: u32, dim: u32) bool {
    return rows > 8 and dim == 128 and kv_heads != 0 and heads % kv_heads == 0;
}

/// Encode MLX's attention for p.rows query rows (p: rows, row0, pos, heads,
/// kv_heads, dim, window, scale, capacity, x/xs, y/ys; gated: z/zs, act_b).
pub fn mlxAttention(ctx: metal.Context, arena: metal.Buffer, keys: metal.Buffer, values: metal.Buffer, p: *const gpu.Params, gated: bool) !void {
    const tgs = @as(usize, (p.rows + 31) / 32) * p.heads;
    try gpu.dispatch(ctx, if (gated) "mk_mlx_attention_gated" else "mk_mlx_attention", &.{ arena, keys, values }, p, tgs * 128, 128);
}

/// knobs.mlx_attention: MLX's attention for a multi-row forward, with the
/// sigmoid-gate epilogue when `gate` is given; false when it does not apply.
fn encodeMlxAttention(self: *Engine, a: @FieldType(ir.Op, "attention"), gate: ?@FieldType(ir.Op, "binary"), row0: u32, rows: u32) !bool {
    if (!self.knobs.mlx_attention or !mlxAttentionApplies(rows, a.heads, a.kv_heads, a.dim)) return false;
    const q = self.words(a.q);
    var p: gpu.Params = .{ .rows = rows, .row0 = row0, .pos = self.position };
    p.x = q.off;
    p.xs = q.stride;
    const o = if (gate) |b| self.words(b.out) else self.words(a.out);
    p.y = o.off;
    p.ys = o.stride;
    if (gate) |b| {
        const g = self.words(b.b);
        p.z = g.off;
        p.zs = g.stride;
        p.act_b = @backingInt(b.b_act);
    }
    p.heads = a.heads;
    p.kv_heads = a.kv_heads;
    p.dim = a.dim;
    p.window = a.window;
    p.scale = a.scale;
    p.capacity = self.program.states[a.keys].capacity(self.batch, self.context);
    // c2-prefill-mlx (knob mlx_attention2): the same arithmetic, rescheduled.
    if (self.knobs.mlx_attention2 and mlx2AttentionApplies(&p)) {
        try mlx2Attention(self.ctx, self.arena, self.caches[a.keys].?, self.caches[a.values].?, &p, gate != null);
        return true;
    }
    try mlxAttention(self.ctx, self.arena, self.caches[a.keys].?, self.caches[a.values].?, &p, gate != null);
    return true;
}

/// knob mlx_multi: the linears among `lin` (all reading one view) that MLX would run
/// unsplit, in one mk_mlx_qmm_multi launch when there are at least two of them on one
/// weight buffer with one group and scale dtype; marks them in `done`.
fn encodeMlxMulti(self: *Engine, lin: []const ir.Op, done: *[4]bool, row0: u32, rows: u32) !void {
    var x: gpu.ParamsX = .{ .p = .{ .rows = rows, .row0 = row0, .pos = self.position } };
    var pick: [4]usize = undefined;
    var buf: ?metal.Buffer = null;
    var tgs: usize = 0;
    for (lin, 0..) |op, k| {
        const l = op.linear;
        const w = self.bound[l.w];
        if (!mlxApplies(w, l.x.width, l.out.width, rows) or mlxSplitK(rows, l.out.width, l.x.width, w.quant.?.group_size) != 1) continue;
        // One buffer binding: codes, scales and biases must all live in the weight's buffer.
        if (@intFromPtr(w.quant.?.scales.handle) != @intFromPtr(w.buffer.handle) or @intFromPtr(w.quant.?.biases.handle) != @intFromPtr(w.buffer.handle)) continue;
        const mt = mat(self, w, l.out);
        if (x.count > 0) {
            const l0 = lin[pick[0]].linear;
            if (@intFromPtr(w.buffer.handle) != @intFromPtr(buf.?.handle) or w.quant.?.group_size != x.p.group or
                mt.sdtype != x.mat[0].sdtype or l.x.state != l0.x.state or l.x.col != l0.x.col or l.x.width != l0.x.width) continue;
        } else {
            const xin = self.words(l.x);
            x.p.x = xin.off;
            x.p.xs = xin.stride;
            x.p.n = l.x.width;
            x.p.group = w.quant.?.group_size;
            buf = w.buffer;
        }
        x.mat[x.count] = mt;
        pick[x.count] = k;
        x.count += 1;
        tgs += @as(usize, (l.out.width + 31) / 32) * ((rows + 31) / 32);
    }
    if (x.count < 2) return;
    // c2-prefill-mlx (knob mlx_qmm2): the same products (each unsplit, checked above) through mk_mlx2_multi.
    if (self.knobs.mlx_qmm2 and x.p.n % 64 == 0 and x.p.x % 2 == 0 and x.p.xs % 2 == 0) {
        var tgs2: usize = 0;
        for (x.mat[0..x.count]) |mt| tgs2 += @as(usize, (mt.m + mlx2_bn - 1) / mlx2_bn) * ((rows + mlx2_bm - 1) / mlx2_bm);
        try gpu.dispatch(self.ctx, mlx2Name("multi", x.mat[0].sdtype, x.p.group), &.{ buf.?, self.arena }, &x, tgs2 * 64, 64);
        for (pick[0..x.count]) |k| done[k] = true;
        return;
    }
    const names = [3][3][:0]const u8{
        .{ "mk_mlx_qmm_multi_f32_g32", "mk_mlx_qmm_multi_f32_g64", "mk_mlx_qmm_multi_f32_g128" },
        .{ "mk_mlx_qmm_multi_f16_g32", "mk_mlx_qmm_multi_f16_g64", "mk_mlx_qmm_multi_f16_g128" },
        .{ "mk_mlx_qmm_multi_bf16_g32", "mk_mlx_qmm_multi_bf16_g64", "mk_mlx_qmm_multi_bf16_g128" },
    };
    const g: usize = switch (x.p.group) {
        32 => 0,
        64 => 1,
        else => 2,
    };
    try gpu.dispatch(self.ctx, names[x.mat[0].sdtype][g], &.{ buf.?, self.arena }, &x, tgs * 128, 128);
    for (pick[0..x.count]) |k| done[k] = true;
}

/// MLX's get_qmv_batch_limit (quantized.cpp) for Ultra GPUs (architecture size
/// 'd', any generation; this repository's machine is an M3 Ultra, applegpu_g15d).
/// Other sizes have other rows in MLX's table and are not encoded here.
/// Below it stock MLX runs a matrix-vector kernel (qmv_wide), which is not
/// ported, so those batches keep the exact kernels.
pub fn mlxVectorLimit(k: u32, n: u32) u32 {
    if (k <= 2048 and n <= 2048) return 32;
    if (k <= 4096 and n <= 4096) return 18;
    return 12;
}

/// MLX qmm_splitk's number of K partitions for an [M, K] x [N, K]^T product
/// (1: it falls back to plain qmm_t).
pub fn mlxSplitK(m: u32, n: u32, k: u32, group: u32) u32 {
    const tgs = ((n + 31) / 32) * ((m + 31) / 32);
    var split: u32 = @max(1, 512 / tgs);
    const k_align: u32 = if (group > 32) group else 32;
    split = @min(split, k / k_align);
    while (split > 1 and k % (split * k_align) != 0) split -= 1;
    return @max(split, 1);
}

/// The MLX-arithmetic linear applies: 4-bit, group 32/64/128, K a multiple of
/// 32, even offsets (as the fast reference), scales and biases of one dtype
/// (float32, float16 or bfloat16: one kernel each), and at least MLX's vector limit of rows.
pub fn mlxApplies(w: weights.Weight, k: u32, n: u32, rows: u32) bool {
    const q = w.quant orelse return false;
    return q4Fast(w, k) and (q.group_size == 32 or q.group_size == 64 or q.group_size == 128) and k % 32 == 0 and
        q.scale_dtype == q.bias_dtype and q.scale_dtype <= 2 and rows >= mlxVectorLimit(k, n);
}

/// Encode stock MLX's quantized_matmul(transpose) for p.rows tokens: p.n = K,
/// p.m = N, x/xs, y/ys, and the weight's w/s/b/sdtype/bdtype/group. Same grid,
/// tiles and split as MLX; returns the split. `splitk` holds mlx_splitk_words.
pub fn mlxLinear(ctx: metal.Context, qb: [3]metal.Buffer, arena: metal.Buffer, splitk: metal.Buffer, p: *gpu.Params) !u32 {
    const split = mlxSplitK(p.rows, p.m, p.n, p.group);
    const tgs = @as(usize, (p.m + 31) / 32) * ((p.rows + 31) / 32);
    p.kind = split;
    p.dim = p.n / split;
    const g: usize = switch (p.group) {
        32 => 0,
        64 => 1,
        128 => 2,
        else => return error.NotPorted,
    };
    if (p.sdtype != p.bdtype or p.sdtype > 2) return error.NotPorted;
    const dt: usize = p.sdtype;
    if (split > 1) {
        if (@as(usize, split) * p.rows * p.m > mlx_splitk_words) return error.SplitKOverflow;
        const names = [3][3][:0]const u8{
            .{ "mk_mlx_qmm_t_splitk_f32_g32", "mk_mlx_qmm_t_splitk_f32_g64", "mk_mlx_qmm_t_splitk_f32_g128" },
            .{ "mk_mlx_qmm_t_splitk_f16_g32", "mk_mlx_qmm_t_splitk_f16_g64", "mk_mlx_qmm_t_splitk_f16_g128" },
            .{ "mk_mlx_qmm_t_splitk_bf16_g32", "mk_mlx_qmm_t_splitk_bf16_g64", "mk_mlx_qmm_t_splitk_bf16_g128" },
        };
        try gpu.dispatch(ctx, names[dt][g], &.{ qb[0], qb[1], qb[2], arena, splitk }, p, tgs * split * 128, 128);
        try gpu.dispatch(ctx, "mk_mlx_splitk_accum", &.{ splitk, arena }, p, @as(usize, p.rows) * p.m, 256);
    } else {
        const names = [3][3][:0]const u8{
            .{ "mk_mlx_qmm_t_f32_g32", "mk_mlx_qmm_t_f32_g64", "mk_mlx_qmm_t_f32_g128" },
            .{ "mk_mlx_qmm_t_f16_g32", "mk_mlx_qmm_t_f16_g64", "mk_mlx_qmm_t_f16_g128" },
            .{ "mk_mlx_qmm_t_bf16_g32", "mk_mlx_qmm_t_bf16_g64", "mk_mlx_qmm_t_bf16_g128" },
        };
        try gpu.dispatch(ctx, names[dt][g], &.{ qb[0], qb[1], qb[2], arena, splitk }, p, tgs * 128, 128);
    }
    return split;
}

// ---- c2-prefill-mlx (prefill, mlx-arith tier) ----

/// mk_mlx2_* tile (mlx2.metal): 64 tokens x 16 outputs per matrix, 64 threads.
pub const mlx2_bm: u32 = 64;
pub const mlx2_bn: u32 = 16;

/// The restructured kernels reproduce stock's unsplit affine_qmm_t only: MLX's split
/// for this product must be 1 (partitions are part of its bits). They also need K a
/// multiple of their K block and x fragments 8-byte aligned (even word offset and
/// row stride). p: rows, n = K, m = N, group, x/xs.
pub fn mlx2Applies(p: *const gpu.Params, kblock: u32) bool {
    return mlxSplitK(p.rows, p.m, p.n, p.group) == 1 and p.n % kblock == 0 and p.x % 2 == 0 and p.xs % 2 == 0 and
        (p.group == 32 or p.group == 64 or p.group == 128) and p.sdtype <= 2;
}

fn mlx2Name(comptime kind: []const u8, sdtype: u32, group: u32) [:0]const u8 {
    const names = comptime blk: {
        var n: [3][3][:0]const u8 = undefined;
        for (.{ "f32", "f16", "bf16" }, 0..) |dt, i| for (.{ 32, 64, 128 }, 0..) |g, j| {
            n[i][j] = std.fmt.comptimePrint("mk_mlx2_{s}_{s}_g{d}", .{ kind, dt, g });
        };
        break :blk n;
    };
    const g: usize = switch (group) {
        32 => 0,
        64 => 1,
        else => 2,
    };
    return names[@min(sdtype, 2)][g];
}

/// knob mlx_qmm2: one product stock MLX runs unsplit (mlxApplies and mlx2Applies(p, 64)),
/// through mk_mlx2_qmm; p as for mlxLinear.
pub fn mlx2Linear(ctx: metal.Context, qb: [3]metal.Buffer, arena: metal.Buffer, p: *const gpu.Params) !void {
    if (p.sdtype != p.bdtype or !mlx2Applies(p, 64)) return error.NotPorted;
    const tgs = @as(usize, (p.m + mlx2_bn - 1) / mlx2_bn) * ((p.rows + mlx2_bm - 1) / mlx2_bm);
    try gpu.dispatch(ctx, mlx2Name("qmm", p.sdtype, p.group), &.{ qb[0], qb[1], qb[2], arena }, p, tgs * 64, 64);
}

/// knob mlx_glu2: gate l0, up l1 and their product b (a GLU step of plan.zig) in one
/// mk_mlx2_glu launch, when both products are ones stock MLX runs unsplit with the
/// same weight buffer (codes, scales and biases), group and scale dtype. False when
/// it does not apply (the caller then runs them as before).
pub fn mlx2Glu(ctx: metal.Context, wbuf: metal.Buffer, arena: metal.Buffer, x: *const gpu.ParamsX) !void {
    if (x.count != 2 or x.mat[0].m != x.mat[1].m or x.mat[0].sdtype != x.mat[1].sdtype or x.mat[0].sdtype != x.mat[0].bdtype or
        x.mat[1].sdtype != x.mat[1].bdtype) return error.NotPorted;
    var q = x.p;
    q.m = x.mat[0].m;
    q.sdtype = x.mat[0].sdtype;
    if (!mlx2Applies(&q, 32)) return error.NotPorted;
    const tgs = @as(usize, (q.m + mlx2_bn - 1) / mlx2_bn) * ((q.rows + mlx2_bm - 1) / mlx2_bm);
    try gpu.dispatch(ctx, mlx2Name("glu", q.sdtype, q.group), &.{ wbuf, arena }, x, tgs * 64, 64);
}

fn encodeMlx2Glu(self: *Engine, l0: @FieldType(ir.Op, "linear"), l1: @FieldType(ir.Op, "linear"), b: @FieldType(ir.Op, "binary"), row0: u32, rows: u32) !bool {
    const w0 = self.bound[l0.w];
    const w1 = self.bound[l1.w];
    for ([_]weights.Weight{ w0, w1 }) |w| {
        if (@intFromPtr(w.buffer.handle) != @intFromPtr(w0.buffer.handle) or
            @intFromPtr(w.quant.?.scales.handle) != @intFromPtr(w0.buffer.handle) or
            @intFromPtr(w.quant.?.biases.handle) != @intFromPtr(w0.buffer.handle)) return false;
    }
    if (w0.quant.?.group_size != w1.quant.?.group_size) return false;
    var x: gpu.ParamsX = .{ .p = .{ .rows = rows, .row0 = row0, .pos = self.position }, .count = 2, .act_a = @backingInt(b.a_act), .act_b = @backingInt(b.b_act), .kind = @backingInt(b.kind) };
    const xin = self.words(l0.x);
    const o = self.words(b.out);
    x.p.x = xin.off;
    x.p.xs = xin.stride;
    x.p.y = o.off;
    x.p.ys = o.stride;
    x.p.n = l0.x.width;
    x.p.group = w0.quant.?.group_size;
    x.mat[0] = mat(self, w0, l0.out);
    x.mat[1] = mat(self, w1, l1.out);
    var q = x.p;
    q.m = x.mat[0].m;
    q.sdtype = x.mat[0].sdtype;
    if (x.mat[0].m != x.mat[1].m or x.mat[0].sdtype != x.mat[1].sdtype or x.mat[0].sdtype != x.mat[0].bdtype or !mlx2Applies(&q, 32)) return false;
    try mlx2Glu(self.ctx, w0.buffer, self.arena, &x);
    return true;
}

/// knob mlx_attention2: the query fragments are read as float2 (even offset and row stride).
pub fn mlx2AttentionApplies(p: *const gpu.Params) bool {
    return p.x % 2 == 0 and p.xs % 2 == 0;
}

/// mk_mlx2_attention[_gated]: mlxAttention's arithmetic and parameters, rescheduled.
pub fn mlx2Attention(ctx: metal.Context, arena: metal.Buffer, keys: metal.Buffer, values: metal.Buffer, p: *const gpu.Params, gated: bool) !void {
    if (!mlx2AttentionApplies(p)) return error.NotPorted;
    const tgs = @as(usize, (p.rows + 31) / 32) * p.heads;
    try gpu.dispatch(ctx, if (gated) "mk_mlx2_attention_gated" else "mk_mlx2_attention", &.{ arena, keys, values }, p, tgs * 128, 128);
}

/// Slices per row of the two-stage argmax.
const argmax_groups: u32 = inkling_plan.argmax_groups;

/// Positions per SIMD group in mk_dec_scores (MK_DEC_CHUNK in decode.metal).
const dec_chunk: u32 = 8;
fn pow2upto8(v: u32) bool {
    return v == 1 or v == 2 or v == 4 or v == 8;
}

fn tileKernel(comptime base: []const u8, rows: u32, tokens: u32) [:0]const u8 {
    const r: usize = switch (rows) {
        1 => 0,
        2 => 1,
        4 => 2,
        else => 3,
    };
    const t: usize = switch (tokens) {
        1 => 0,
        2 => 1,
        else => 2,
    };
    const names = comptime blk: {
        var n: [4][3][:0]const u8 = undefined;
        for (.{ 1, 2, 4, 8 }, 0..) |rr, i| for (.{ 1, 2, 4 }, 0..) |tt, j| {
            n[i][j] = std.fmt.comptimePrint("{s}_r{d}_t{d}", .{ base, rr, tt });
        };
        break :blk n;
    };
    return names[r][t];
}

/// prefill objective (prefill search): the compiled matrix-matrix variant for the knobs.
fn mmKernel(comptime base: []const u8, k: Knobs) [:0]const u8 {
    const names = comptime blk: {
        var n: [Knobs.mm_tiles.len][:0]const u8 = undefined;
        for (Knobs.mm_tiles, 0..) |t, i| n[i] = std.fmt.comptimePrint("{s}_r{d}_t{d}_{s}", .{ base, t[0], t[1], if (t[2]) "h" else "f" });
        break :blk n;
    };
    for (Knobs.mm_tiles, 0..) |t, i| if (t[0] == k.mm_rows and t[1] == k.mm_tokens and t[2] == k.mm_half) return names[i];
    unreachable; // Knobs.validate admits only compiled tiles
}

pub fn q4Fast(w: weights.Weight, cols: u32) bool {
    const q = w.quant orelse return false;
    // Even offsets: mk_word reads a 2-aligned code word as two halves; float32
    // scales and biases (dtype 0) still need 4-byte alignment.
    const sa: u64 = if (q.scale_dtype == 0) 4 else 2;
    const ba: u64 = if (q.bias_dtype == 0) 4 else 2;
    return q.bits == 4 and q.group_size % 16 == 0 and cols % 16 == 0 and w.tensor.offset % 2 == 0 and q.scale_offset % sa == 0 and q.bias_offset % ba == 0;
}

/// Group size from config.json's MLX "quantization" object (0 = unquantized).
pub fn quantGroup(a: std.mem.Allocator, config_bytes: []const u8) !u32 {
    const parsed = try std.json.parseFromSliceLeaky(std.json.Value, a, config_bytes, .{});
    if (parsed != .object) return error.InvalidConfig;
    const q = parsed.object.get("quantization") orelse return 0;
    if (q != .object) return error.InvalidConfig;
    const g = q.object.get("group_size") orelse return error.InvalidConfig;
    return switch (g) {
        .integer => |n| std.math.cast(u32, n) orelse error.InvalidConfig,
        else => error.InvalidConfig,
    };
}

pub fn modelType(a: std.mem.Allocator, config_bytes: []const u8) ![]const u8 {
    const parsed = try std.json.parseFromSliceLeaky(std.json.Value, a, config_bytes, .{});
    if (parsed != .object) return error.InvalidConfig;
    const t = parsed.object.get("model_type") orelse return error.MissingModelType;
    if (t != .string) return error.InvalidConfig;
    return t.string;
}
