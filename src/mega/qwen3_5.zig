//! Qwen3.5 / Qwen3.8 (`qwen3_5` / `qwen3_5_text`) hybrid linear-attention + GQA
//! transformer front end for the megakernel IR.
const std = @import("std");
const ir = @import("ir.zig");
const liquid = @import("lfm3.zig");

pub const Checkpoint = liquid.Checkpoint;

pub const LayerType = enum {
    linear_attention,
    full_attention,
};

pub const Config = struct {
    model_type: []const u8,
    hidden: u32,
    intermediate: u32,
    layers: []const LayerType,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    attn_output_gate: bool,
    k_heads: u32,
    v_heads: u32,
    k_head_dim: u32,
    v_head_dim: u32,
    conv_taps: u32,
    eps: f32,
    rot: u32,
    rope_theta: f64,
    tie_word_embeddings: bool,
    vocab: u32,
    eos: []const u32,
    max_context: u32,
};

const RopeParametersRaw = struct {
    rope_theta: ?f64 = null,
    partial_rotary_factor: ?f64 = null,
    rope_type: ?[]const u8 = null,
};

const TextRaw = struct {
    model_type: []const u8 = "qwen3_5_text",
    hidden_size: u32,
    intermediate_size: u32,
    num_hidden_layers: u32,
    layer_types: ?[]const LayerType = null,
    full_attention_interval: u32 = 4,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    head_dim: ?u32 = null,
    attn_output_gate: bool = true,
    attention_bias: bool = false,
    linear_conv_kernel_dim: u32 = 4,
    linear_key_head_dim: u32 = 128,
    linear_num_key_heads: u32 = 16,
    linear_num_value_heads: u32 = 48,
    linear_value_head_dim: u32 = 128,
    rms_norm_eps: f32 = 1e-6,
    rope_theta: ?f64 = null,
    partial_rotary_factor: ?f64 = null,
    rope_parameters: ?RopeParametersRaw = null,
    tie_word_embeddings: ?bool = null,
    vocab_size: u32,
    max_position_embeddings: u32,
    eos_token_id: ?std.json.Value = null,
};

fn positive(x: f64) bool {
    return std.math.isFinite(x) and x > 0;
}

fn tokenId(value: std.json.Value, vocab: u32) !u32 {
    if (value != .integer or value.integer < 0 or value.integer >= vocab) return error.InvalidEos;
    return @intCast(value.integer);
}

fn appendEos(a: std.mem.Allocator, out: *std.ArrayList(u32), value: std.json.Value, vocab: u32) !void {
    switch (value) {
        .integer => {
            const id = try tokenId(value, vocab);
            if (std.mem.indexOfScalar(u32, out.items, id) == null) try out.append(a, id);
        },
        .array => |arr| for (arr.items) |elem| {
            const id = try tokenId(elem, vocab);
            if (std.mem.indexOfScalar(u32, out.items, id) == null) try out.append(a, id);
        },
        else => return error.InvalidEos,
    }
}

pub fn parse(a: std.mem.Allocator, bytes: []const u8) !Config {
    const root = try std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{});
    if (root != .object) return error.InvalidConfig;
    const obj = root.object;
    const outer_mt = switch (obj.get("model_type") orelse return error.MissingConfigField) {
        .string => |s| s,
        else => return error.InvalidConfig,
    };

    var outer_eos: ?std.json.Value = null;
    var outer_tie: ?bool = null;
    const text_val: std.json.Value = if (std.mem.eql(u8, outer_mt, "qwen3_5")) blk: {
        outer_eos = obj.get("eos_token_id");
        if (outer_eos == null) if (obj.get("generation_config")) |gc| {
            if (gc == .object) outer_eos = gc.object.get("eos_token_id");
        };
        if (obj.get("tie_word_embeddings")) |tv| if (tv == .bool) {
            outer_tie = tv.bool;
        };
        break :blk obj.get("text_config") orelse return error.MissingTextConfig;
    } else if (std.mem.eql(u8, outer_mt, "qwen3_5_text"))
        root
    else
        return error.NotQwen35;

    const r = try std.json.parseFromValueLeaky(TextRaw, a, text_val, .{ .ignore_unknown_fields = true });
    if (!std.mem.eql(u8, r.model_type, "qwen3_5_text") and !std.mem.eql(u8, r.model_type, "qwen3_5"))
        return error.NotQwen35;
    if (r.attention_bias) return error.UnsupportedAttentionBias;

    const h = r.hidden_size;
    const heads = r.num_attention_heads;
    const kv = r.num_key_value_heads;
    if (h == 0 or heads == 0 or kv == 0 or heads % kv != 0 or r.intermediate_size == 0 or
        r.num_hidden_layers == 0 or r.vocab_size == 0 or r.max_position_embeddings == 0 or
        !positive(r.rms_norm_eps)) return error.InvalidConfig;

    const head_dim = r.head_dim orelse if (h % heads == 0) h / heads else return error.InvalidConfig;
    if (head_dim == 0 or head_dim > 256 or head_dim % 2 != 0) return error.InvalidConfig;

    const kh = r.linear_num_key_heads;
    const vh = r.linear_num_value_heads;
    const kd = r.linear_key_head_dim;
    const vd = r.linear_value_head_dim;
    if (kh == 0 or vh == 0 or vh % kh != 0 or kd == 0 or kd > 256 or vd == 0 or r.linear_conv_kernel_dim < 2)
        return error.InvalidConfig;

    const theta = r.rope_theta orelse if (r.rope_parameters) |rp| rp.rope_theta orelse 1e6 else 1e6;
    const partial = r.partial_rotary_factor orelse if (r.rope_parameters) |rp| rp.partial_rotary_factor orelse 1.0 else 1.0;
    if (!positive(theta) or !positive(partial) or partial > 1.0) return error.InvalidConfig;
    if (r.rope_parameters) |rp| if (rp.rope_type) |rt| {
        if (!std.mem.eql(u8, rt, "default")) return error.UnsupportedRopeType;
    };
    const rot_f = @round(@as(f64, @floatFromInt(head_dim)) * partial);
    if (!positive(rot_f) or rot_f > @as(f64, @floatFromInt(head_dim))) return error.InvalidConfig;
    const rot: u32 = @intFromFloat(rot_f);
    if (rot == 0 or rot > head_dim or rot % 2 != 0) return error.InvalidConfig;

    const layers: []const LayerType = if (r.layer_types) |lt| lt else blk: {
        if (r.full_attention_interval == 0) return error.InvalidConfig;
        const arr = try a.alloc(LayerType, r.num_hidden_layers);
        for (arr, 0..) |*slot, idx| {
            slot.* = if ((idx + 1) % r.full_attention_interval == 0) .full_attention else .linear_attention;
        }
        break :blk arr;
    };
    if (layers.len != r.num_hidden_layers) return error.InvalidConfig;

    var eos: std.ArrayList(u32) = .empty;
    if (outer_eos) |e| {
        try appendEos(a, &eos, e, r.vocab_size);
    } else if (r.eos_token_id) |e| {
        try appendEos(a, &eos, e, r.vocab_size);
    }

    return .{
        .model_type = outer_mt,
        .hidden = h,
        .intermediate = r.intermediate_size,
        .layers = layers,
        .heads = heads,
        .kv_heads = kv,
        .head_dim = head_dim,
        .attn_output_gate = r.attn_output_gate,
        .k_heads = kh,
        .v_heads = vh,
        .k_head_dim = kd,
        .v_head_dim = vd,
        .conv_taps = r.linear_conv_kernel_dim,
        .eps = r.rms_norm_eps,
        .rot = rot,
        .rope_theta = theta,
        .tie_word_embeddings = r.tie_word_embeddings orelse outer_tie orelse false,
        .vocab = r.vocab_size,
        .eos = eos.items,
        .max_context = r.max_position_embeddings,
    };
}

/// Partial inverse frequencies `theta ^ (-2*j / rot)` matching `mlx.nn.RoPE`.
pub fn ropeFrequencies(a: std.mem.Allocator, rot: u32, theta: f64) ![]const f32 {
    const half = rot / 2;
    const out = try a.alloc(f32, half);
    for (0..half) |j| {
        const exp = @as(f64, @floatFromInt(2 * j)) / @as(f64, @floatFromInt(rot));
        out[j] = @floatCast(1.0 / std.math.pow(f64, theta, exp));
    }
    return out;
}

fn hasWeight(checkpoint: Checkpoint, name: []const u8) bool {
    return checkpoint.has(checkpoint.context, name);
}

fn weightPrefix(a: std.mem.Allocator, checkpoint: Checkpoint) ![]const u8 {
    for ([_][]const u8{ "language_model.model.", "model.language_model.", "model.", "" }) |prefix| {
        const probe = try std.fmt.allocPrint(a, "{s}embed_tokens.weight", .{prefix});
        if (hasWeight(checkpoint, probe)) return prefix;
    }
    return "model.";
}

fn lmHeadWeightName(checkpoint: Checkpoint, prefix: []const u8) []const u8 {
    if (std.mem.eql(u8, prefix, "language_model.model.")) {
        if (hasWeight(checkpoint, "language_model.lm_head.weight") or !hasWeight(checkpoint, "lm_head.weight"))
            return "language_model.lm_head.weight";
        return "lm_head.weight";
    }
    if (std.mem.eql(u8, prefix, "model.language_model.")) {
        if (hasWeight(checkpoint, "lm_head.weight")) return "lm_head.weight";
        if (hasWeight(checkpoint, "model.lm_head.weight")) return "model.lm_head.weight";
        if (hasWeight(checkpoint, "language_model.lm_head.weight")) return "language_model.lm_head.weight";
        return "lm_head.weight";
    }
    return "lm_head.weight";
}

pub fn build(gpa: std.mem.Allocator, config_bytes: []const u8, checkpoint: Checkpoint) !ir.Program {
    var b = ir.Builder.init(gpa);
    errdefer b.deinit();
    const a = b.allocator();
    const c = try parse(a, config_bytes);
    const prefix = try weightPrefix(a, checkpoint);

    const h = c.hidden;
    const inter = c.intermediate;
    const q_dim = c.heads * c.head_dim;
    const kv_dim = c.kv_heads * c.head_dim;
    const key_dim = c.k_heads * c.k_head_dim;
    const val_dim = c.v_heads * c.v_head_dim;
    const conv_dim = 2 * key_dim + val_dim;

    var has_linear = false;
    var has_full = false;
    for (c.layers) |lt| switch (lt) {
        .linear_attention => has_linear = true,
        .full_attention => has_full = true,
    };

    const rope_table = if (has_full) try b.table(try ropeFrequencies(a, c.rot, c.rope_theta)) else 0;

    const tokens = try b.state(.{ .name = "tokens", .kind = .tokens, .width = 1 });
    _ = try b.state(.{ .name = "position", .kind = .position, .width = 1 });
    const x = try b.state(.{ .name = "hidden", .kind = .scratch, .width = h });
    const x_norm = try b.state(.{ .name = "norm", .kind = .scratch, .width = h });
    const branch_out = try b.state(.{ .name = "branch_out", .kind = .scratch, .width = h });

    const ffn_gate = try b.state(.{ .name = "ffn_gate", .kind = .scratch, .width = inter });
    const ffn_up = try b.state(.{ .name = "ffn_up", .kind = .scratch, .width = inter });
    const ffn_act = try b.state(.{ .name = "ffn_act", .kind = .scratch, .width = inter });

    var gdn_qkv: ir.StateId = undefined;
    var gdn_conv: ir.StateId = undefined;
    var gdn_act: ir.StateId = undefined;
    var gdn_z: ir.StateId = undefined;
    var gdn_b: ir.StateId = undefined;
    var gdn_a: ir.StateId = undefined;
    var gdn_out: ir.StateId = undefined;
    var gdn_norm: ir.StateId = undefined;
    var gdn_gated: ir.StateId = undefined;
    if (has_linear) {
        gdn_qkv = try b.state(.{ .name = "gdn_qkv", .kind = .scratch, .width = conv_dim });
        gdn_conv = try b.state(.{ .name = "gdn_conv", .kind = .scratch, .width = conv_dim });
        gdn_act = try b.state(.{ .name = "gdn_act", .kind = .scratch, .width = conv_dim });
        gdn_z = try b.state(.{ .name = "gdn_z", .kind = .scratch, .width = val_dim });
        gdn_b = try b.state(.{ .name = "gdn_b", .kind = .scratch, .width = c.v_heads });
        gdn_a = try b.state(.{ .name = "gdn_a", .kind = .scratch, .width = c.v_heads });
        gdn_out = try b.state(.{ .name = "gdn_out", .kind = .scratch, .width = val_dim });
        gdn_norm = try b.state(.{ .name = "gdn_norm", .kind = .scratch, .width = val_dim });
        gdn_gated = try b.state(.{ .name = "gdn_gated", .kind = .scratch, .width = val_dim });
    }

    var q_gate_raw: ir.StateId = undefined;
    var q_gate_split: ir.StateId = undefined;
    var q_normed: ir.StateId = undefined;
    var k_buf: ir.StateId = undefined;
    var k_normed: ir.StateId = undefined;
    var v_buf: ir.StateId = undefined;
    var att_out: ir.StateId = undefined;
    var att_gated: ir.StateId = undefined;
    if (has_full) {
        q_gate_raw = try b.state(.{ .name = "q_proj", .kind = .scratch, .width = if (c.attn_output_gate) 2 * q_dim else q_dim });
        if (c.attn_output_gate) q_gate_split = try b.state(.{ .name = "q_split", .kind = .scratch, .width = 2 * q_dim });
        q_normed = try b.state(.{ .name = "q", .kind = .scratch, .width = q_dim });
        k_buf = try b.state(.{ .name = "k", .kind = .scratch, .width = kv_dim });
        k_normed = try b.state(.{ .name = "k_normed", .kind = .scratch, .width = kv_dim });
        v_buf = try b.state(.{ .name = "v", .kind = .scratch, .width = kv_dim });
        att_out = try b.state(.{ .name = "att", .kind = .scratch, .width = q_dim });
        if (c.attn_output_gate) att_gated = try b.state(.{ .name = "att_gated", .kind = .scratch, .width = q_dim });
    }

    const logits = try b.state(.{ .name = "logits", .kind = .scratch, .width = c.vocab });
    const next = try b.state(.{ .name = "next", .kind = .scratch, .width = 1 });

    const embed_w = try b.weight(try b.print("{s}embed_tokens.weight", .{prefix}), &.{ c.vocab, h });
    try b.op(.{ .embed = .{ .table = embed_w, .tokens = b.view(tokens), .out = b.view(x) } });
    try b.trace(b.view(x));

    const attn_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(c.head_dim)));
    const rec_words: u32 = c.v_heads * c.v_head_dim * c.k_head_dim;

    for (c.layers, 0..) |layer_type, idx| {
        const layer: u32 = @intCast(idx);
        const in_norm_w = try b.weight(try b.print("{s}layers.{d}.input_layernorm.weight", .{ prefix, layer }), &.{h});
        try b.op(.{ .rmsnorm = .{ .x = b.view(x), .weight = in_norm_w, .out = b.view(x_norm), .group = h, .eps = c.eps } });

        switch (layer_type) {
            .linear_attention => {
                const w_qkv = try b.weight(try b.print("{s}layers.{d}.linear_attn.in_proj_qkv.weight", .{ prefix, layer }), &.{ conv_dim, h });
                const w_z = try b.weight(try b.print("{s}layers.{d}.linear_attn.in_proj_z.weight", .{ prefix, layer }), &.{ val_dim, h });
                const w_b = try b.weight(try b.print("{s}layers.{d}.linear_attn.in_proj_b.weight", .{ prefix, layer }), &.{ c.v_heads, h });
                const w_a = try b.weight(try b.print("{s}layers.{d}.linear_attn.in_proj_a.weight", .{ prefix, layer }), &.{ c.v_heads, h });
                try b.op(.{ .linear = .{ .w = w_qkv, .x = b.view(x_norm), .out = b.view(gdn_qkv) } });
                try b.op(.{ .linear = .{ .w = w_z, .x = b.view(x_norm), .out = b.view(gdn_z) } });
                try b.op(.{ .linear = .{ .w = w_b, .x = b.view(x_norm), .out = b.view(gdn_b) } });
                try b.op(.{ .linear = .{ .w = w_a, .x = b.view(x_norm), .out = b.view(gdn_a) } });

                const conv_ring = try b.state(.{ .name = try b.print("conv{d}", .{layer}), .kind = .ring, .width = conv_dim, .horizon = c.conv_taps - 1 });
                const conv_w = try b.weight(try b.print("{s}layers.{d}.linear_attn.conv1d.weight", .{ prefix, layer }), &.{ conv_dim, c.conv_taps, 1 });
                try b.op(.{ .store = .{ .src = b.view(gdn_qkv), .dst = conv_ring } });
                try b.op(.{ .conv = .{ .src = conv_ring, .weight = conv_w, .out = b.view(gdn_conv), .taps = c.conv_taps } });
                try b.op(.{ .activate = .{ .x = b.view(gdn_conv), .out = b.view(gdn_act), .kind = .silu } });

                const rec_state = try b.state(.{ .name = try b.print("rec{d}", .{layer}), .kind = .recurrent, .width = rec_words });
                const w_alog = try b.weight(try b.print("{s}layers.{d}.linear_attn.A_log", .{ prefix, layer }), &.{c.v_heads});
                const w_dt = try b.weight(try b.print("{s}layers.{d}.linear_attn.dt_bias", .{ prefix, layer }), &.{c.v_heads});
                try b.op(.{ .gated_delta_net = .{
                    .qkv = b.view(gdn_act),
                    .a = b.view(gdn_a),
                    .b = b.view(gdn_b),
                    .a_log = w_alog,
                    .dt_bias = w_dt,
                    .state = rec_state,
                    .out = b.view(gdn_out),
                    .k_heads = c.k_heads,
                    .v_heads = c.v_heads,
                    .k_head_dim = c.k_head_dim,
                    .v_head_dim = c.v_head_dim,
                    .eps = 1e-6,
                } });

                const w_gdn_norm = try b.weight(try b.print("{s}layers.{d}.linear_attn.norm.weight", .{ prefix, layer }), &.{c.v_head_dim});
                try b.op(.{ .rmsnorm = .{ .x = b.view(gdn_out), .weight = w_gdn_norm, .out = b.view(gdn_norm), .group = c.v_head_dim, .eps = c.eps } });
                try b.op(.{ .binary = .{ .kind = .mul, .a = b.view(gdn_norm), .b = b.view(gdn_z), .b_act = .silu, .out = b.view(gdn_gated) } });

                const w_out = try b.weight(try b.print("{s}layers.{d}.linear_attn.out_proj.weight", .{ prefix, layer }), &.{ h, val_dim });
                try b.op(.{ .linear = .{ .w = w_out, .x = b.view(gdn_gated), .out = b.view(branch_out) } });
            },
            .full_attention => {
                const w_q = try b.weight(try b.print("{s}layers.{d}.self_attn.q_proj.weight", .{ prefix, layer }), &.{ if (c.attn_output_gate) 2 * q_dim else q_dim, h });
                const w_k = try b.weight(try b.print("{s}layers.{d}.self_attn.k_proj.weight", .{ prefix, layer }), &.{ kv_dim, h });
                const w_v = try b.weight(try b.print("{s}layers.{d}.self_attn.v_proj.weight", .{ prefix, layer }), &.{ kv_dim, h });
                try b.op(.{ .linear = .{ .w = w_q, .x = b.view(x_norm), .out = b.view(q_gate_raw) } });
                try b.op(.{ .linear = .{ .w = w_k, .x = b.view(x_norm), .out = b.view(k_buf) } });
                try b.op(.{ .linear = .{ .w = w_v, .x = b.view(x_norm), .out = b.view(v_buf) } });

                const q_norm_w = try b.weight(try b.print("{s}layers.{d}.self_attn.q_norm.weight", .{ prefix, layer }), &.{c.head_dim});
                const k_norm_w = try b.weight(try b.print("{s}layers.{d}.self_attn.k_norm.weight", .{ prefix, layer }), &.{c.head_dim});

                var q_gate_view: ?ir.View = null;
                if (c.attn_output_gate) {
                    try b.op(.{ .select_halves = .{ .x = b.view(q_gate_raw), .out = b.view(q_gate_split), .groups = c.heads } });
                    const q_raw_view: ir.View = .{ .state = q_gate_split, .col = 0, .width = q_dim };
                    q_gate_view = .{ .state = q_gate_split, .col = q_dim, .width = q_dim };
                    try b.op(.{ .rmsnorm = .{ .x = q_raw_view, .weight = q_norm_w, .out = b.view(q_normed), .group = c.head_dim, .eps = c.eps } });
                } else {
                    try b.op(.{ .rmsnorm = .{ .x = b.view(q_gate_raw), .weight = q_norm_w, .out = b.view(q_normed), .group = c.head_dim, .eps = c.eps } });
                }
                try b.op(.{ .rmsnorm = .{ .x = b.view(k_buf), .weight = k_norm_w, .out = b.view(k_normed), .group = c.head_dim, .eps = c.eps } });
                try b.op(.{ .rope = .{ .x = b.view(q_normed), .heads = c.heads, .dim = c.head_dim, .rot = c.rot, .freqs = rope_table } });
                try b.op(.{ .rope = .{ .x = b.view(k_normed), .heads = c.kv_heads, .dim = c.head_dim, .rot = c.rot, .freqs = rope_table } });

                const k_cache = try b.state(.{ .name = try b.print("k{d}", .{layer}), .kind = .sequence, .width = kv_dim });
                const v_cache = try b.state(.{ .name = try b.print("v{d}", .{layer}), .kind = .sequence, .width = kv_dim });
                try b.op(.{ .store = .{ .src = b.view(k_normed), .dst = k_cache } });
                try b.op(.{ .store = .{ .src = b.view(v_buf), .dst = v_cache } });
                try b.op(.{ .attention = .{ .q = b.view(q_normed), .keys = k_cache, .values = v_cache, .out = b.view(att_out), .heads = c.heads, .kv_heads = c.kv_heads, .dim = c.head_dim, .window = 0, .scale = attn_scale } });

                const w_o = try b.weight(try b.print("{s}layers.{d}.self_attn.o_proj.weight", .{ prefix, layer }), &.{ h, q_dim });
                if (q_gate_view) |gview| {
                    try b.op(.{ .binary = .{ .kind = .mul, .a = b.view(att_out), .b = gview, .b_act = .sigmoid, .out = b.view(att_gated) } });
                    try b.op(.{ .linear = .{ .w = w_o, .x = b.view(att_gated), .out = b.view(branch_out) } });
                } else {
                    try b.op(.{ .linear = .{ .w = w_o, .x = b.view(att_out), .out = b.view(branch_out) } });
                }
            },
        }

        try b.op(.{ .binary = .{ .kind = .add, .a = b.view(x), .b = b.view(branch_out), .out = b.view(x) } });
        const post_norm_w = try b.weight(try b.print("{s}layers.{d}.post_attention_layernorm.weight", .{ prefix, layer }), &.{h});
        try b.op(.{ .rmsnorm = .{ .x = b.view(x), .weight = post_norm_w, .out = b.view(x_norm), .group = h, .eps = c.eps } });

        const w_gate = try b.weight(try b.print("{s}layers.{d}.mlp.gate_proj.weight", .{ prefix, layer }), &.{ inter, h });
        const w_up = try b.weight(try b.print("{s}layers.{d}.mlp.up_proj.weight", .{ prefix, layer }), &.{ inter, h });
        const w_down = try b.weight(try b.print("{s}layers.{d}.mlp.down_proj.weight", .{ prefix, layer }), &.{ h, inter });
        try b.op(.{ .linear = .{ .w = w_gate, .x = b.view(x_norm), .out = b.view(ffn_gate) } });
        try b.op(.{ .linear = .{ .w = w_up, .x = b.view(x_norm), .out = b.view(ffn_up) } });
        try b.op(.{ .binary = .{ .kind = .mul, .a = b.view(ffn_gate), .b = b.view(ffn_up), .a_act = .silu, .out = b.view(ffn_act) } });
        try b.op(.{ .linear = .{ .w = w_down, .x = b.view(ffn_act), .out = b.view(branch_out) } });
        try b.op(.{ .binary = .{ .kind = .add, .a = b.view(x), .b = b.view(branch_out), .out = b.view(x) } });
        try b.trace(b.view(x));
    }

    const head = b.mark();
    const final_norm_w = try b.weight(try b.print("{s}norm.weight", .{prefix}), &.{h});
    try b.op(.{ .rmsnorm = .{ .x = b.view(x), .weight = final_norm_w, .out = b.view(x_norm), .group = h, .eps = c.eps } });
    const lm_head_w = if (c.tie_word_embeddings)
        embed_w
    else
        try b.weight(lmHeadWeightName(checkpoint, prefix), &.{ c.vocab, h });
    try b.op(.{ .linear = .{ .w = lm_head_w, .x = b.view(x_norm), .out = b.view(logits) } });
    try b.op(.{ .argmax = .{ .x = b.view(logits), .out = b.view(next) } });

    return b.finish(.{
        .model_type = c.model_type,
        .vocab = c.vocab,
        .eos = c.eos,
        .max_context = c.max_context,
    }, head, b.view(tokens), b.view(logits), b.view(next));
}

fn hasLmModelPrefix(_: *const anyopaque, name: []const u8) bool {
    return std.mem.eql(u8, name, "language_model.model.embed_tokens.weight") or
        std.mem.eql(u8, name, "language_model.lm_head.weight");
}

fn hasModelPrefix(_: *const anyopaque, name: []const u8) bool {
    return std.mem.eql(u8, name, "model.embed_tokens.weight") or
        std.mem.eql(u8, name, "lm_head.weight");
}

test "Qwen3.5 parses text_config wrapper, detects prefixes, and lowers GDN + GQA layers" {
    const config =
        \\{
        \\  "model_type": "qwen3_5",
        \\  "eos_token_id": [29, 28],
        \\  "tie_word_embeddings": false,
        \\  "text_config": {
        \\    "model_type": "qwen3_5_text",
        \\    "hidden_size": 32,
        \\    "intermediate_size": 64,
        \\    "num_hidden_layers": 2,
        \\    "layer_types": ["linear_attention", "full_attention"],
        \\    "num_attention_heads": 6,
        \\    "num_key_value_heads": 1,
        \\    "head_dim": 16,
        \\    "attn_output_gate": true,
        \\    "linear_conv_kernel_dim": 4,
        \\    "linear_key_head_dim": 8,
        \\    "linear_num_key_heads": 2,
        \\    "linear_num_value_heads": 4,
        \\    "linear_value_head_dim": 8,
        \\    "rms_norm_eps": 1e-6,
        \\    "rope_parameters": { "partial_rotary_factor": 0.25, "rope_theta": 10000000, "rope_type": "default" },
        \\    "tie_word_embeddings": false,
        \\    "vocab_size": 40,
        \\    "max_position_embeddings": 128
        \\  }
        \\}
    ;
    var p = try build(std.testing.allocator, config, .{ .context = undefined, .has = hasLmModelPrefix });
    defer p.deinit();
    try std.testing.expectEqualStrings("qwen3_5", p.meta.model_type);
    try std.testing.expectEqualSlices(u32, &.{ 29, 28 }, p.meta.eos);
    try std.testing.expectEqual(@as(usize, 3), p.traces.len);
    // Partial RoPE rot = 16 * 0.25 = 4 -> table len = 2.
    try std.testing.expectEqual(@as(usize, 1), p.tables.len);
    try std.testing.expectEqual(@as(usize, 2), p.tables[0].len);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), p.tables[0][0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, @floatCast(1.0 / @sqrt(10000000.0))), p.tables[0][1], 1e-7);

    var gdn_ops: u32 = 0;
    var split_ops: u32 = 0;
    var recs: u32 = 0;
    var seqs: u32 = 0;
    var rings: u32 = 0;
    for (p.ops) |op| switch (op) {
        .gated_delta_net => gdn_ops += 1,
        .select_halves => split_ops += 1,
        else => {},
    };
    for (p.states) |s| switch (s.kind) {
        .recurrent => recs += 1,
        .sequence => seqs += 1,
        .ring => rings += 1,
        else => {},
    };
    try std.testing.expectEqual(@as(u32, 1), gdn_ops);
    try std.testing.expectEqual(@as(u32, 1), split_ops);
    try std.testing.expectEqual(@as(u32, 1), recs);
    try std.testing.expectEqual(@as(u32, 2), seqs);
    try std.testing.expectEqual(@as(u32, 1), rings);

    var has_alog = false;
    var has_head = false;
    for (p.weights) |w| {
        if (std.mem.eql(u8, w.name, "language_model.model.layers.0.linear_attn.A_log")) has_alog = true;
        if (std.mem.eql(u8, w.name, "language_model.lm_head.weight")) has_head = true;
    }
    try std.testing.expect(has_alog);
    try std.testing.expect(has_head);
}

test "Qwen3.5 parses qwen3_5_text directly with tied embeddings and full_attention_interval" {
    const config =
        \\{
        \\  "model_type": "qwen3_5_text",
        \\  "hidden_size": 32,
        \\  "intermediate_size": 64,
        \\  "num_hidden_layers": 2,
        \\  "full_attention_interval": 2,
        \\  "num_attention_heads": 4,
        \\  "num_key_value_heads": 2,
        \\  "head_dim": 8,
        \\  "attn_output_gate": false,
        \\  "linear_conv_kernel_dim": 4,
        \\  "linear_key_head_dim": 8,
        \\  "linear_num_key_heads": 2,
        \\  "linear_num_value_heads": 4,
        \\  "linear_value_head_dim": 8,
        \\  "rope_theta": 1000000,
        \\  "tie_word_embeddings": true,
        \\  "vocab_size": 32,
        \\  "max_position_embeddings": 64,
        \\  "eos_token_id": 5
        \\}
    ;
    var p = try build(std.testing.allocator, config, .{ .context = undefined, .has = hasModelPrefix });
    defer p.deinit();
    try std.testing.expectEqualStrings("qwen3_5_text", p.meta.model_type);
    try std.testing.expectEqualSlices(u32, &.{5}, p.meta.eos);
    for (p.ops) |op| try std.testing.expect(op != .select_halves);
    for (p.weights) |w| try std.testing.expect(std.mem.indexOf(u8, w.name, "lm_head") == null);
}
