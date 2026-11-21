//! LFM2.5 dense text configuration (LFM2 model_type). Uses the shared Liquid
//! builder for conv/GQA layers, without LFM3's attention gates or post norms.
const std = @import("std");
const ir = @import("ir.zig");
const liquid = @import("lfm3.zig");

const Raw = struct {
    model_type: []const u8,
    hidden_size: u32,
    num_hidden_layers: u32,
    layer_types: []liquid.LayerType,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    block_dim: u32,
    block_ff_dim: ?u32 = null,
    intermediate_size: ?u32 = null,
    block_auto_adjust_ff_dim: bool,
    block_ffn_dim_multiplier: ?f64 = null,
    block_multiple_of: u32,
    block_use_swiglu: bool = true,
    norm_eps: f32,
    conv_L_cache: u32,
    conv_dim: ?u32 = null,
    conv_bias: bool = false,
    use_pos_enc: bool = true,
    use_sandwich_norm: bool = false,
    tie_embedding: bool = true,
    tie_word_embeddings: bool = true,
    rope_theta: f64 = 1e6,
    rope_parameters: ?std.json.Value = null,
    rope_scaling: ?std.json.Value = null,
    full_attn_idxs: ?[]u32 = null,
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

pub fn parse(a: std.mem.Allocator, bytes: []const u8) !liquid.Config {
    const r = try std.json.parseFromSliceLeaky(Raw, a, bytes, .{ .ignore_unknown_fields = true });
    if (!std.mem.eql(u8, r.model_type, "lfm2")) return error.NotLfm2;
    if (r.conv_bias) return error.UnsupportedConvBias;
    if (!r.tie_embedding or !r.tie_word_embeddings) return error.UnsupportedUntiedHead;
    if (r.use_sandwich_norm or !r.block_use_swiglu or !r.use_pos_enc or r.rope_parameters != null or r.rope_scaling != null or r.full_attn_idxs != null)
        return error.UnsupportedLfm2Config;
    const h = r.hidden_size;
    const heads = r.num_attention_heads;
    const kv = r.num_key_value_heads;
    if (h == 0 or heads == 0 or kv == 0 or h % heads != 0 or heads % kv != 0 or h > std.math.maxInt(u32) / 3 or
        r.block_dim != h or (r.conv_dim orelse h) != h or r.conv_L_cache < 2 or
        !positive(r.norm_eps) or !positive(r.rope_theta) or r.max_position_embeddings == 0 or r.vocab_size == 0)
        return error.InvalidConfig;
    const dim = h / heads;
    if (dim % 2 != 0 or r.layer_types.len != r.num_hidden_layers or r.layer_types.len == 0) return error.InvalidConfig;
    for (r.layer_types) |layer| if (layer == .sliding_attention) return error.UnsupportedLayerType;

    var ff: u64 = r.block_ff_dim orelse r.intermediate_size orelse return error.MissingConfigField;
    if (r.block_auto_adjust_ff_dim) {
        if (r.block_multiple_of == 0) return error.InvalidConfig;
        ff = (2 * ff) / 3;
        if (r.block_ffn_dim_multiplier) |m| {
            const scaled = m * @as(f64, @floatFromInt(ff));
            if (!positive(scaled) or scaled > std.math.maxInt(u32)) return error.InvalidConfig;
            ff = @intFromFloat(scaled);
        }
        const multiple: u64 = r.block_multiple_of;
        ff = ((ff + multiple - 1) / multiple) * multiple;
    }
    if (ff == 0 or ff > std.math.maxInt(u32)) return error.InvalidConfig;

    var eos: std.ArrayList(u32) = .empty;
    if (r.eos_token_id) |e| switch (e) {
        .integer => try eos.append(a, try tokenId(e, r.vocab_size)),
        .array => |arr| for (arr.items) |t| try eos.append(a, try tokenId(t, r.vocab_size)),
        else => return error.InvalidEos,
    };
    return .{ .arch = .lfm2, .hidden = h, .layers = r.layer_types, .heads = heads, .kv_heads = kv, .head_dim = dim, .intermediate = @intCast(ff), .conv_taps = r.conv_L_cache, .eps = r.norm_eps, .vocab = r.vocab_size, .eos = eos.items, .window = 0, .max_context = r.max_position_embeddings, .full_theta = r.rope_theta, .full_partial = 1, .sliding_theta = r.rope_theta, .sliding_partial = 1 };
}

pub fn build(gpa: std.mem.Allocator, config: []const u8, checkpoint: liquid.Checkpoint, options: liquid.Options) !ir.Program {
    return liquid.buildParsed(gpa, config, checkpoint, options, parse);
}

fn modelPrefix(_: *const anyopaque, name: []const u8) bool {
    return std.mem.eql(u8, name, "model.embed_tokens.weight");
}

test "LFM2.5 builds ungated full GQA and prenorm layers with adjusted FFN width" {
    const config =
        \\{"model_type":"lfm2","hidden_size":64,"num_hidden_layers":2,"layer_types":["conv","full_attention"],
        \\"num_attention_heads":4,"num_key_value_heads":2,"block_dim":64,"block_ff_dim":144,
        \\"block_auto_adjust_ff_dim":true,"block_ffn_dim_multiplier":1.0,"block_multiple_of":64,
        \\"conv_L_cache":3,"norm_eps":1e-5,"eos_token_id":7,"vocab_size":40,"max_position_embeddings":128}
    ;
    var p = try build(std.testing.allocator, config, .{ .context = undefined, .has = modelPrefix }, .{});
    defer p.deinit();
    try std.testing.expectEqualStrings("lfm2", p.meta.model_type);
    try std.testing.expectEqualSlices(u32, &.{7}, p.meta.eos);
    try std.testing.expectEqual(@as(usize, 1 + (1 + 6 + 7) + (1 + 11 + 7) + 3), p.ops.len);
    var found = false;
    for (p.weights) |w| {
        try std.testing.expect(std.mem.indexOf(u8, w.name, "gate_proj") == null);
        try std.testing.expect(std.mem.indexOf(u8, w.name, "post_") == null);
        if (std.mem.eql(u8, w.name, "model.layers.1.self_attn.q_layernorm.weight")) found = true;
    }
    try std.testing.expect(found);
    for (p.states) |s| if (std.mem.eql(u8, s.name, "w1")) try std.testing.expectEqual(@as(u32, 128), s.width);
}
