//! Native Inkling 2 Vision encoder (`inkling2_vision`):
//! ImageIO RGB decode -> FP32 Lanczos3 resize & right/bottom 40px padding ->
//! 20x20 patch embedding + CLS + bilinear 2D pos emb + ln_pre ->
//! 23-layer ViT with interleaved 2D RoPE, bidirectional MHA, LayerScale & GELU ->
//! 2x2 spatial merge -> Linear(4096->4096) + GELU + Linear(4096->2304).
const std = @import("std");
const gpu = @import("gpu.zig");
const weights = @import("weights");
const metal = weights.metal;
const Allocator = std.mem.Allocator;

extern fn fm_load_image_rgb(path: [*:0]const u8, out_rgb: *?[*]u8, out_w: *u32, out_h: *u32) c_int;
extern fn fm_load_image_rgb_bytes(data: [*]const u8, len: usize, out_rgb: *?[*]u8, out_w: *u32, out_h: *u32) c_int;
extern fn fm_load_image_rgb_url(url: [*:0]const u8, out_rgb: *?[*]u8, out_w: *u32, out_h: *u32) c_int;
extern fn fm_free_image(rgb: [*]u8) void;

pub const ImageInput = union(enum) {
    source: []const u8,
    bytes: []const u8,
};

pub const image_token_id: u32 = 200054;
pub const default_max_image_tokens: u32 = 1024;
pub const default_max_image_dim: u32 = 4096;

pub const PatchLayout = enum { hwc, chw };

pub const Config = struct {
    text_hidden_size: u32,
    native_patch_size: u32,
    adapter_merge_size: u32,
    projection_hidden_dim: u32,
    projector_bias: bool,
    hidden_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    mlp_ratio: f32,
    posemb_grid_size: u32,
    layer_norm_eps: f32,
    rope_theta: f32 = 10000.0,
    patch_layout: PatchLayout = .hwc,

    pub fn headDim(self: Config) u32 {
        return self.hidden_size / self.num_attention_heads;
    }
    pub fn mlpDim(self: Config) u32 {
        return @intFromFloat(@round(@as(f32, @floatFromInt(self.hidden_size)) * self.mlp_ratio));
    }
    pub fn patchPixels(self: Config) u32 {
        return self.native_patch_size * self.native_patch_size * 3;
    }
    pub fn tilePixels(self: Config) u32 {
        return self.native_patch_size * self.adapter_merge_size;
    }
};

fn jsonObj(v: ?std.json.Value) !std.json.ObjectMap {
    const val = v orelse return error.MissingVisionConfig;
    return switch (val) {
        .object => |o| o,
        else => error.InvalidVisionConfig,
    };
}
fn jsonU32(v: ?std.json.Value, default: ?u32) !u32 {
    const val = v orelse return default orelse error.MissingVisionConfigField;
    return switch (val) {
        .integer => |n| std.math.cast(u32, n) orelse error.InvalidVisionConfig,
        else => error.InvalidVisionConfig,
    };
}
fn jsonF32(v: ?std.json.Value, default: ?f32) !f32 {
    const val = v orelse return default orelse error.MissingVisionConfigField;
    return switch (val) {
        .integer => |n| @floatFromInt(n),
        .float => |n| @floatCast(n),
        else => error.InvalidVisionConfig,
    };
}
fn jsonBool(v: ?std.json.Value, default: bool) !bool {
    const val = v orelse return default;
    return switch (val) {
        .bool => |b| b,
        else => error.InvalidVisionConfig,
    };
}

pub fn hasVisionConfig(a: Allocator, config_bytes: []const u8) bool {
    const parsed = std.json.parseFromSliceLeaky(std.json.Value, a, config_bytes, .{}) catch return false;
    if (parsed != .object) return false;
    const vc = parsed.object.get("vision_config") orelse return false;
    return vc == .object;
}

pub fn parseConfig(a: Allocator, config_bytes: []const u8) !Config {
    const parsed = try std.json.parseFromSliceLeaky(std.json.Value, a, config_bytes, .{});
    const root = try jsonObj(parsed);
    const vc = try jsonObj(root.get("vision_config"));
    if (vc.get("model_type")) |mt| {
        if (mt != .string or !std.mem.eql(u8, mt.string, "inkling2_vision")) return error.UnsupportedVisionModelType;
    } else return error.MissingVisionConfigField;

    const cfg: Config = .{
        .text_hidden_size = try jsonU32(vc.get("text_hidden_size"), null),
        .native_patch_size = try jsonU32(vc.get("native_patch_size"), 20),
        .adapter_merge_size = try jsonU32(vc.get("adapter_merge_size"), 2),
        .projection_hidden_dim = try jsonU32(vc.get("projection_hidden_dim"), 4096),
        .projector_bias = try jsonBool(vc.get("projector_bias"), false),
        .hidden_size = try jsonU32(vc.get("hidden_size"), 1024),
        .num_hidden_layers = try jsonU32(vc.get("num_hidden_layers"), 23),
        .num_attention_heads = try jsonU32(vc.get("num_attention_heads"), 16),
        .mlp_ratio = try jsonF32(vc.get("mlp_ratio"), 4.0),
        .posemb_grid_size = try jsonU32(vc.get("posemb_grid_size"), 32),
        .layer_norm_eps = try jsonF32(vc.get("layer_norm_eps"), 1e-5),
    };
    if (cfg.hidden_size == 0 or cfg.num_hidden_layers == 0 or cfg.num_attention_heads == 0 or
        cfg.hidden_size % cfg.num_attention_heads != 0 or cfg.headDim() % 4 != 0 or cfg.headDim() > 128 or
        cfg.native_patch_size == 0 or cfg.adapter_merge_size == 0 or cfg.posemb_grid_size == 0 or
        cfg.text_hidden_size == 0 or cfg.projection_hidden_dim == 0 or cfg.projector_bias)
    {
        return error.UnsupportedVisionConfig;
    }
    return cfg;
}

// ---- Native image decode + FP32 Lanczos3 resize + 40px padding --------------

pub const RgbImage = struct {
    pixels: [*]u8,
    width: u32,
    height: u32,
    owned_by_c: bool = true,

    pub fn slice(self: RgbImage) []const u8 {
        return self.pixels[0 .. @as(usize, self.width) * self.height * 3];
    }

    pub fn deinit(self: RgbImage) void {
        if (self.owned_by_c) fm_free_image(self.pixels);
    }
};

pub fn decodeImageBytes(bytes: []const u8) !RgbImage {
    if (bytes.len == 0) return error.ImageDecodeFailed;
    var rgb_ptr: ?[*]u8 = null;
    var w: u32 = 0;
    var h: u32 = 0;
    if (fm_load_image_rgb_bytes(bytes.ptr, bytes.len, &rgb_ptr, &w, &h) == 0 or rgb_ptr == null or w == 0 or h == 0) {
        return error.ImageDecodeFailed;
    }
    return .{ .pixels = rgb_ptr.?, .width = w, .height = h, .owned_by_c = true };
}

fn hexNibble(ch: u8) ?u8 {
    return switch (ch) {
        '0'...'9' => ch - '0',
        'a'...'f' => ch - 'a' + 10,
        'A'...'F' => ch - 'A' + 10,
        else => null,
    };
}

pub fn decodeFileUri(a: Allocator, uri: []const u8) ![]u8 {
    if (!std.mem.startsWith(u8, uri, "file://")) return error.InvalidImageUri;
    var rest = uri["file://".len..];
    if (std.mem.startsWith(u8, rest, "localhost/")) {
        rest = rest["localhost".len..];
    } else if (rest.len == 0 or rest[0] != '/') {
        return error.InvalidImageUri;
    }
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    try out.ensureTotalCapacity(a, rest.len);
    var i: usize = 0;
    while (i < rest.len) {
        if (rest[i] == '%' and i + 2 < rest.len) {
            const hi = hexNibble(rest[i + 1]) orelse return error.InvalidImageUri;
            const lo = hexNibble(rest[i + 2]) orelse return error.InvalidImageUri;
            try out.append(a, (hi << 4) | lo);
            i += 3;
        } else {
            try out.append(a, rest[i]);
            i += 1;
        }
    }
    return out.toOwnedSlice(a);
}

pub fn decodeDataUriBytes(a: Allocator, uri: []const u8) ![]u8 {
    if (!std.mem.startsWith(u8, uri, "data:")) return error.InvalidImageUri;
    const comma = std.mem.indexOfScalar(u8, uri, ',') orelse return error.InvalidImageUri;
    const header = uri["data:".len..comma];
    var is_base64 = false;
    var parts = std.mem.splitScalar(u8, header, ';');
    while (parts.next()) |part| {
        const trimmed = std.mem.trim(u8, part, " \t");
        if (std.ascii.eqlIgnoreCase(trimmed, "base64")) {
            is_base64 = true;
            break;
        }
    }
    if (!is_base64) return error.InvalidImageUri;

    const raw_payload = uri[comma + 1 ..];
    var clean: std.ArrayList(u8) = .empty;
    defer clean.deinit(a);
    try clean.ensureTotalCapacity(a, raw_payload.len);
    for (raw_payload) |ch| {
        if (ch == ' ' or ch == '\t' or ch == '\r' or ch == '\n') continue;
        try clean.append(a, ch);
    }
    const decoded_len = std.base64.standard.Decoder.calcSizeForSlice(clean.items) catch return error.InvalidImageUri;
    const out = try a.alloc(u8, decoded_len);
    errdefer a.free(out);
    std.base64.standard.Decoder.decode(out, clean.items) catch return error.InvalidImageUri;
    return out;
}

/// Decode an image from a local path, `file://` URI, `data:...;base64,...` URI, or `http(s)://` URL.
pub fn decodeImageFile(a: Allocator, source: []const u8) !RgbImage {
    const trimmed = std.mem.trim(u8, source, " \t\r\n");
    if (trimmed.len == 0) return error.ImageDecodeFailed;
    if (std.mem.startsWith(u8, trimmed, "data:")) {
        const bytes = try decodeDataUriBytes(a, trimmed);
        defer a.free(bytes);
        return decodeImageBytes(bytes);
    }
    if (std.mem.startsWith(u8, trimmed, "http://") or std.mem.startsWith(u8, trimmed, "https://")) {
        const zurl = try a.dupeSentinel(u8, trimmed, 0);
        defer a.free(zurl);
        var rgb_ptr: ?[*]u8 = null;
        var w: u32 = 0;
        var h: u32 = 0;
        if (fm_load_image_rgb_url(zurl, &rgb_ptr, &w, &h) == 0 or rgb_ptr == null or w == 0 or h == 0) {
            return error.ImageDecodeFailed;
        }
        return .{ .pixels = rgb_ptr.?, .width = w, .height = h, .owned_by_c = true };
    }
    var owned_path: ?[]u8 = null;
    defer if (owned_path) |p| a.free(p);
    const fs_path: []const u8 = if (std.mem.startsWith(u8, trimmed, "file://")) blk: {
        owned_path = try decodeFileUri(a, trimmed);
        break :blk owned_path.?;
    } else trimmed;

    const zpath = try a.dupeSentinel(u8, fs_path, 0);
    defer a.free(zpath);
    var rgb_ptr: ?[*]u8 = null;
    var w: u32 = 0;
    var h: u32 = 0;
    if (fm_load_image_rgb(zpath, &rgb_ptr, &w, &h) == 0 or rgb_ptr == null or w == 0 or h == 0) {
        return error.ImageDecodeFailed;
    }
    return .{ .pixels = rgb_ptr.?, .width = w, .height = h, .owned_by_c = true };
}

pub const TargetDims = struct {
    resized_w: u32,
    resized_h: u32,
    padded_w: u32,
    padded_h: u32,
    grid_h: u32,
    grid_w: u32,
    visual_tokens: u32,
};

pub fn computeTargetDims(w: u32, h: u32, cfg: Config, max_image_tokens: u32, max_dim: u32) TargetDims {
    const tile = cfg.tilePixels(); // 40
    var rw: u32 = @max(@as(u32, 1), w);
    var rh: u32 = @max(@as(u32, 1), h);

    // Enforce max_dim while preserving aspect ratio.
    if (rw > max_dim or rh > max_dim) {
        const s = @as(f64, @floatFromInt(max_dim)) / @as(f64, @floatFromInt(@max(rw, rh)));
        rw = @max(@as(u32, 1), @as(u32, @intFromFloat(@round(@as(f64, @floatFromInt(rw)) * s))));
        rh = @max(@as(u32, 1), @as(u32, @intFromFloat(@round(@as(f64, @floatFromInt(rh)) * s))));
    }

    // Enforce max_image_tokens budget (each visual token is tile x tile pixels).
    var tw = ((rw + tile - 1) / tile);
    var th = ((rh + tile - 1) / tile);
    if (@as(u64, tw) * th > max_image_tokens) {
        const scale = @sqrt(@as(f64, @floatFromInt(max_image_tokens)) / @as(f64, @floatFromInt(@as(u64, tw) * th)));
        rw = @max(@as(u32, 1), @as(u32, @intFromFloat(@floor(@as(f64, @floatFromInt(rw)) * scale))));
        rh = @max(@as(u32, 1), @as(u32, @intFromFloat(@floor(@as(f64, @floatFromInt(rh)) * scale))));
        tw = (rw + tile - 1) / tile;
        th = (rh + tile - 1) / tile;
        while (@as(u64, tw) * th > max_image_tokens and (rw > tile or rh > tile)) {
            if (rw >= rh and rw > tile) {
                rw = (tw - 1) * tile;
            } else if (rh > tile) {
                rh = (th - 1) * tile;
            } else break;
            tw = (rw + tile - 1) / tile;
            th = (rh + tile - 1) / tile;
        }
    }

    const padded_w = tw * tile;
    const padded_h = th * tile;
    return .{
        .resized_w = rw,
        .resized_h = rh,
        .padded_w = padded_w,
        .padded_h = padded_h,
        .grid_h = padded_h / cfg.native_patch_size,
        .grid_w = padded_w / cfg.native_patch_size,
        .visual_tokens = tw * th,
    };
}

fn lanczos3Weight(x_in: f32) f32 {
    const x = @abs(x_in);
    if (x < 1e-6) return 1.0;
    if (x >= 3.0) return 0.0;
    const pi_x = std.math.pi * x;
    const pi_x_3 = pi_x / 3.0;
    return (@sin(pi_x) / pi_x) * (@sin(pi_x_3) / pi_x_3);
}

/// Separable FP32 Lanczos3 resize matching Pillow/PyTorch coordinate centers.
pub fn resizeLanczos3Rgb(a: Allocator, src: []const u8, sw: u32, sh: u32, dw: u32, dh: u32) ![]f32 {
    const tmp = try a.alloc(f32, @as(usize, sh) * dw * 3);
    defer a.free(tmp);
    const out = try a.alloc(f32, @as(usize, dh) * dw * 3);

    // Horizontal pass: [sh, sw, 3] -> [sh, dw, 3]
    const sx_scale = @as(f32, @floatFromInt(sw)) / @as(f32, @floatFromInt(dw));
    const h_filter_scale: f32 = @max(1.0, sx_scale);
    const h_support: f32 = 3.0 * h_filter_scale;
    for (0..dw) |dx| {
        const center = (@as(f32, @floatFromInt(dx)) + 0.5) * sx_scale - 0.5;
        const left: i32 = @max(0, @as(i32, @intFromFloat(@floor(center - h_support))));
        const right: i32 = @min(@as(i32, @intCast(sw)) - 1, @as(i32, @intFromFloat(@ceil(center + h_support))));
        var w_sum: f32 = 0;
        var sx_i = left;
        while (sx_i <= right) : (sx_i += 1) {
            w_sum += lanczos3Weight((@as(f32, @floatFromInt(sx_i)) - center) / h_filter_scale);
        }
        const inv_w: f32 = if (w_sum > 0) 1.0 / w_sum else 1.0;
        for (0..sh) |y| {
            var acc = [3]f32{ 0, 0, 0 };
            sx_i = left;
            while (sx_i <= right) : (sx_i += 1) {
                const wt = lanczos3Weight((@as(f32, @floatFromInt(sx_i)) - center) / h_filter_scale) * inv_w;
                const off = (y * sw + @as(usize, @intCast(sx_i))) * 3;
                acc[0] += @as(f32, @floatFromInt(src[off + 0])) * wt;
                acc[1] += @as(f32, @floatFromInt(src[off + 1])) * wt;
                acc[2] += @as(f32, @floatFromInt(src[off + 2])) * wt;
            }
            const dst_off = (y * dw + dx) * 3;
            tmp[dst_off + 0] = std.math.clamp(acc[0], 0.0, 255.0);
            tmp[dst_off + 1] = std.math.clamp(acc[1], 0.0, 255.0);
            tmp[dst_off + 2] = std.math.clamp(acc[2], 0.0, 255.0);
        }
    }

    // Vertical pass: [sh, dw, 3] -> [dh, dw, 3]
    const sy_scale = @as(f32, @floatFromInt(sh)) / @as(f32, @floatFromInt(dh));
    const v_filter_scale: f32 = @max(1.0, sy_scale);
    const v_support: f32 = 3.0 * v_filter_scale;
    for (0..dh) |dy| {
        const center = (@as(f32, @floatFromInt(dy)) + 0.5) * sy_scale - 0.5;
        const top: i32 = @max(0, @as(i32, @intFromFloat(@floor(center - v_support))));
        const bottom: i32 = @min(@as(i32, @intCast(sh)) - 1, @as(i32, @intFromFloat(@ceil(center + v_support))));
        var w_sum: f32 = 0;
        var sy_i = top;
        while (sy_i <= bottom) : (sy_i += 1) {
            w_sum += lanczos3Weight((@as(f32, @floatFromInt(sy_i)) - center) / v_filter_scale);
        }
        const inv_w: f32 = if (w_sum > 0) 1.0 / w_sum else 1.0;
        for (0..dw) |x| {
            var acc = [3]f32{ 0, 0, 0 };
            sy_i = top;
            while (sy_i <= bottom) : (sy_i += 1) {
                const wt = lanczos3Weight((@as(f32, @floatFromInt(sy_i)) - center) / v_filter_scale) * inv_w;
                const off = (@as(usize, @intCast(sy_i)) * dw + x) * 3;
                acc[0] += tmp[off + 0] * wt;
                acc[1] += tmp[off + 1] * wt;
                acc[2] += tmp[off + 2] * wt;
            }
            const dst_off = (dy * dw + x) * 3;
            out[dst_off + 0] = std.math.clamp(acc[0], 0.0, 255.0);
            out[dst_off + 1] = std.math.clamp(acc[1], 0.0, 255.0);
            out[dst_off + 2] = std.math.clamp(acc[2], 0.0, 255.0);
        }
    }
    return out;
}

pub const PreprocessedImage = struct {
    dims: TargetDims,
    /// Flattened patches [grid_h * grid_w, native_patch_size * native_patch_size * 3].
    patches: []f32,
};

/// Right/bottom padding pixel: -1 on the 0..255 scale, normalized with mean=0.5, std=0.5.
pub const pad_pixel: f32 = (-1.0 * (1.0 / 255.0) - 0.5) / 0.5;

pub fn preprocessRgb(a: Allocator, rgb: []const u8, w: u32, h: u32, cfg: Config, max_image_tokens: u32) !PreprocessedImage {
    if (rgb.len != @as(usize, w) * h * 3) return error.InvalidImageSize;
    const dims = computeTargetDims(w, h, cfg, max_image_tokens, default_max_image_dim);

    var resized_opt: ?[]f32 = null;
    defer if (resized_opt) |r| a.free(r);
    if (dims.resized_w != w or dims.resized_h != h) {
        resized_opt = try resizeLanczos3Rgb(a, rgb, w, h, dims.resized_w, dims.resized_h);
    }

    const p = cfg.native_patch_size;
    const patch_dim = cfg.patchPixels();
    const n_patches = @as(usize, dims.grid_h) * dims.grid_w;
    const patches = try a.alloc(f32, n_patches * patch_dim);
    @memset(patches, pad_pixel);

    const scale: f32 = 2.0 / 255.0;
    for (0..dims.grid_h) |gy| {
        for (0..dims.grid_w) |gx| {
            const patch_base = (gy * dims.grid_w + gx) * patch_dim;
            for (0..p) |dy| {
                const py = gy * p + dy;
                if (py >= dims.resized_h) continue;
                for (0..p) |dx| {
                    const px = gx * p + dx;
                    if (px >= dims.resized_w) continue;
                    const src_idx = (py * dims.resized_w + px) * 3;
                    var rgb_norm: [3]f32 = undefined;
                    if (resized_opt) |rf| {
                        rgb_norm[0] = rf[src_idx + 0] * scale - 1.0;
                        rgb_norm[1] = rf[src_idx + 1] * scale - 1.0;
                        rgb_norm[2] = rf[src_idx + 2] * scale - 1.0;
                    } else {
                        rgb_norm[0] = @as(f32, @floatFromInt(rgb[src_idx + 0])) * scale - 1.0;
                        rgb_norm[1] = @as(f32, @floatFromInt(rgb[src_idx + 1])) * scale - 1.0;
                        rgb_norm[2] = @as(f32, @floatFromInt(rgb[src_idx + 2])) * scale - 1.0;
                    }
                    switch (cfg.patch_layout) {
                        .hwc => {
                            const off = patch_base + (dy * p + dx) * 3;
                            patches[off + 0] = rgb_norm[0];
                            patches[off + 1] = rgb_norm[1];
                            patches[off + 2] = rgb_norm[2];
                        },
                        .chw => {
                            const plane = p * p;
                            const spatial = dy * p + dx;
                            patches[patch_base + 0 * plane + spatial] = rgb_norm[0];
                            patches[patch_base + 1 * plane + spatial] = rgb_norm[1];
                            patches[patch_base + 2 * plane + spatial] = rgb_norm[2];
                        },
                    }
                }
            }
        }
    }

    return .{ .dims = dims, .patches = patches };
}

// ---- GPU Vision Encoder -----------------------------------------------------

const LayerWeights = struct {
    ln_1_w: weights.Weight,
    ln_1_b: weights.Weight,
    ln_2_w: weights.Weight,
    ln_2_b: weights.Weight,
    ls_1: weights.Weight,
    ls_2: weights.Weight,
    q_w: weights.Weight,
    q_b: weights.Weight,
    k_w: weights.Weight,
    k_b: weights.Weight,
    v_w: weights.Weight,
    v_b: weights.Weight,
    out_w: weights.Weight,
    out_b: weights.Weight,
    fc_w: weights.Weight,
    fc_b: weights.Weight,
    proj_w: weights.Weight,
    proj_b: weights.Weight,
};

pub const EncodedImage = struct {
    /// GPU buffer containing [tokens, text_hidden_size] float32 (BF16-rounded) visual embeddings.
    buffer: metal.Buffer,
    tokens: u32,
    grid_h: u32,
    grid_w: u32,
    width: u32,

    pub fn deinit(self: EncodedImage) void {
        self.buffer.deinit();
    }
};

pub const EncodedBatch = struct {
    /// Single contiguous GPU buffer holding all images' visual embeddings [total_tokens, width].
    buffer: metal.Buffer,
    /// Visual token count per image, in order.
    counts: []u32,
    total_tokens: u32,
    width: u32,

    pub fn deinit(self: *EncodedBatch, gpa: Allocator) void {
        self.buffer.deinit();
        gpa.free(self.counts);
    }
};

pub const VisionEncoder = struct {
    gpa: Allocator,
    ctx: metal.Context,
    cfg: Config,

    patch_emb_w: weights.Weight,
    class_emb: weights.Weight,
    cls_pos_emb: weights.Weight,
    pos_emb: weights.Weight,
    ln_pre_w: weights.Weight,
    ln_pre_b: weights.Weight,
    layers: []LayerWeights,
    proj_in_w: weights.Weight,
    proj_out_w: weights.Weight,

    pub fn init(gpa: Allocator, ctx: metal.Context, store: *const weights.Store, config_bytes: []const u8) !VisionEncoder {
        var scratch = std.heap.ArenaAllocator.init(gpa);
        defer scratch.deinit();
        const sa = scratch.allocator();

        var cfg = try parseConfig(sa, config_bytes);
        const D = cfg.hidden_size;
        const M = cfg.mlpDim();
        const P = cfg.patchPixels();
        const G = cfg.posemb_grid_size;
        const merge_dim = cfg.adapter_merge_size * cfg.adapter_merge_size * D;

        const patch_emb_w = try store.get("model.visual.backbone.embeddings.patch_embedding.weight", &.{ D, P });

        const class_emb = try store.get("model.visual.backbone.embeddings.class_embedding", &.{D});
        const cls_pos_emb = try store.get("model.visual.backbone.embeddings.cls_position_embedding", &.{D});
        const pos_emb = try store.get("model.visual.backbone.embeddings.position_embedding", &.{ G * G, D });
        const ln_pre_w = try store.get("model.visual.backbone.ln_pre.weight", &.{D});
        const ln_pre_b = try store.get("model.visual.backbone.ln_pre.bias", &.{D});
        const proj_in_w = try store.get("model.visual.proj_in.weight", &.{ cfg.projection_hidden_dim, merge_dim });
        const proj_out_w = try store.get("model.visual.proj_out.weight", &.{ cfg.text_hidden_size, cfg.projection_hidden_dim });

        const layers = try gpa.alloc(LayerWeights, cfg.num_hidden_layers);
        errdefer gpa.free(layers);
        for (layers, 0..) |*lw, l| {
            const pfx = try std.fmt.allocPrint(sa, "model.visual.backbone.layers.{d}", .{l});
            lw.* = .{
                .ln_1_w = try store.get(try std.fmt.allocPrint(sa, "{s}.ln_1.weight", .{pfx}), &.{D}),
                .ln_1_b = try store.get(try std.fmt.allocPrint(sa, "{s}.ln_1.bias", .{pfx}), &.{D}),
                .ln_2_w = try store.get(try std.fmt.allocPrint(sa, "{s}.ln_2.weight", .{pfx}), &.{D}),
                .ln_2_b = try store.get(try std.fmt.allocPrint(sa, "{s}.ln_2.bias", .{pfx}), &.{D}),
                .ls_1 = try store.get(try std.fmt.allocPrint(sa, "{s}.ls_1.gamma", .{pfx}), &.{D}),
                .ls_2 = try store.get(try std.fmt.allocPrint(sa, "{s}.ls_2.gamma", .{pfx}), &.{D}),
                .q_w = try store.get(try std.fmt.allocPrint(sa, "{s}.self_attn.q_proj.weight", .{pfx}), &.{ D, D }),
                .q_b = try store.get(try std.fmt.allocPrint(sa, "{s}.self_attn.q_proj.bias", .{pfx}), &.{D}),
                .k_w = try store.get(try std.fmt.allocPrint(sa, "{s}.self_attn.k_proj.weight", .{pfx}), &.{ D, D }),
                .k_b = try store.get(try std.fmt.allocPrint(sa, "{s}.self_attn.k_proj.bias", .{pfx}), &.{D}),
                .v_w = try store.get(try std.fmt.allocPrint(sa, "{s}.self_attn.v_proj.weight", .{pfx}), &.{ D, D }),
                .v_b = try store.get(try std.fmt.allocPrint(sa, "{s}.self_attn.v_proj.bias", .{pfx}), &.{D}),
                .out_w = try store.get(try std.fmt.allocPrint(sa, "{s}.self_attn.out_proj.weight", .{pfx}), &.{ D, D }),
                .out_b = try store.get(try std.fmt.allocPrint(sa, "{s}.self_attn.out_proj.bias", .{pfx}), &.{D}),
                .fc_w = try store.get(try std.fmt.allocPrint(sa, "{s}.mlp.c_fc.weight", .{pfx}), &.{ M, D }),
                .fc_b = try store.get(try std.fmt.allocPrint(sa, "{s}.mlp.c_fc.bias", .{pfx}), &.{M}),
                .proj_w = try store.get(try std.fmt.allocPrint(sa, "{s}.mlp.c_proj.weight", .{pfx}), &.{ D, M }),
                .proj_b = try store.get(try std.fmt.allocPrint(sa, "{s}.mlp.c_proj.bias", .{pfx}), &.{D}),
            };
        }

        return .{
            .gpa = gpa,
            .ctx = ctx,
            .cfg = cfg,
            .patch_emb_w = patch_emb_w,
            .class_emb = class_emb,
            .cls_pos_emb = cls_pos_emb,
            .pos_emb = pos_emb,
            .ln_pre_w = ln_pre_w,
            .ln_pre_b = ln_pre_b,
            .layers = layers,
            .proj_in_w = proj_in_w,
            .proj_out_w = proj_out_w,
        };
    }

    pub fn deinit(self: *VisionEncoder) void {
        self.gpa.free(self.layers);
    }

    fn dispatchLinear(
        self: *const VisionEncoder,
        x_buf: metal.Buffer,
        w: weights.Weight,
        bias_opt: ?weights.Weight,
        y_buf: metal.Buffer,
        rows: u32,
        m: u32,
        n: u32,
        apply_gelu: bool,
        round_out: bool,
    ) !void {
        const b_w = bias_opt orelse w;
        var p: gpu.Params = .{
            .w = w.tensor.offset,
            .b = if (bias_opt) |bw| bw.tensor.offset else 0,
            .rows = rows,
            .xs = n,
            .ys = m,
            .n = n,
            .m = m,
            .dtype = @backingInt(w.tensor.dtype),
            .bdtype = if (bias_opt) |bw| @backingInt(bw.tensor.dtype) else 0,
            .flags = (if (bias_opt != null) gpu.flag_has_weight else 0) |
                (if (round_out) gpu.flag_round_out else 0),
            .act_a = if (apply_gelu) 4 else 0,
        };
        const tgs: usize = ((@as(usize, m) + 31) / 32) * ((@as(usize, rows) + 31) / 32);
        try gpu.dispatch(self.ctx, "mk_vis_linear_mma", &.{ x_buf, w.buffer, b_w.buffer, y_buf }, &p, tgs * 128, 128);
    }

    fn dispatchLayerNorm(
        self: *const VisionEncoder,
        x_buf: metal.Buffer,
        w: weights.Weight,
        b: weights.Weight,
        y_buf: metal.Buffer,
        rows: u32,
        dim: u32,
    ) !void {
        var p: gpu.Params = .{
            .w = w.tensor.offset,
            .b = b.tensor.offset,
            .rows = rows,
            .xs = dim,
            .ys = dim,
            .n = dim,
            .dtype = @backingInt(w.tensor.dtype),
            .bdtype = @backingInt(b.tensor.dtype),
            .eps = self.cfg.layer_norm_eps,
        };
        try gpu.dispatch(self.ctx, "mk_vis_layernorm", &.{ x_buf, w.buffer, b.buffer, y_buf }, &p, @as(usize, rows) * 256, 256);
    }

    pub fn encodePreprocessed(self: *const VisionEncoder, prep: PreprocessedImage) !EncodedImage {
        const c = self.cfg;
        const D = c.hidden_size;
        const M = c.mlpDim();
        const P = c.patchPixels();
        const gh = prep.dims.grid_h;
        const gw = prep.dims.grid_w;
        const n_patches: u32 = gh * gw;
        const seq_len: u32 = 1 + n_patches;
        const n_vis: u32 = prep.dims.visual_tokens;
        const merge_dim: u32 = c.adapter_merge_size * c.adapter_merge_size * D;

        // Allocate temporary Metal buffers for this image's vision forward pass.
        const patches_buf = try self.ctx.alloc(@as(usize, n_patches) * P);
        defer patches_buf.deinit();
        @memcpy(patches_buf.floats(), prep.patches);

        const patch_proj_buf = try self.ctx.alloc(@as(usize, n_patches) * D);
        defer patch_proj_buf.deinit();
        const h_buf = try self.ctx.alloc(@as(usize, seq_len) * D);
        defer h_buf.deinit();
        const norm_buf = try self.ctx.alloc(@as(usize, seq_len) * D);
        defer norm_buf.deinit();
        const q_buf = try self.ctx.alloc(@as(usize, seq_len) * D);
        defer q_buf.deinit();
        const k_buf = try self.ctx.alloc(@as(usize, seq_len) * D);
        defer k_buf.deinit();
        const v_buf = try self.ctx.alloc(@as(usize, seq_len) * D);
        defer v_buf.deinit();
        const att_buf = try self.ctx.alloc(@as(usize, seq_len) * D);
        defer att_buf.deinit();
        const mlp_buf = try self.ctx.alloc(@as(usize, seq_len) * M);
        defer mlp_buf.deinit();
        const merged_buf = try self.ctx.alloc(@as(usize, n_vis) * merge_dim);
        defer merged_buf.deinit();
        const proj_mid_buf = try self.ctx.alloc(@as(usize, n_vis) * c.projection_hidden_dim);
        defer proj_mid_buf.deinit();
        const out_buf = try self.ctx.alloc(@as(usize, n_vis) * c.text_hidden_size);
        errdefer out_buf.deinit();

        // 1. Patch projection: [n_patches, P] @ W^T -> [n_patches, D]
        try self.dispatchLinear(patches_buf, self.patch_emb_w, null, patch_proj_buf, n_patches, D, P, false, false);

        // 2. Prepend CLS + add bilinearly interpolated 2D position embeddings -> [seq_len, D]
        {
            var p: gpu.Params = .{
                .w = self.pos_emb.tensor.offset,
                .s = self.class_emb.tensor.offset,
                .b = self.cls_pos_emb.tensor.offset,
                .xs = D,
                .ys = D,
                .n = D,
                .heads = gh,
                .kv_heads = gw,
                .dim = c.posemb_grid_size,
                .dtype = @backingInt(self.pos_emb.tensor.dtype),
            };
            try gpu.dispatch(
                self.ctx,
                "mk_vis_embed_prep",
                &.{ patch_proj_buf, self.pos_emb.buffer, self.class_emb.buffer, self.cls_pos_emb.buffer, norm_buf },
                &p,
                @as(usize, seq_len) * D,
                256,
            );
        }

        // 3. ln_pre: norm_buf -> h_buf
        try self.dispatchLayerNorm(norm_buf, self.ln_pre_w, self.ln_pre_b, h_buf, seq_len, D);

        // 4. 23 ViT transformer layers
        const hd = c.headDim();
        const att_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
        for (self.layers) |lw| {
            // Pre-attention LayerNorm
            try self.dispatchLayerNorm(h_buf, lw.ln_1_w, lw.ln_1_b, norm_buf, seq_len, D);

            // Q, K, V projections
            try self.dispatchLinear(norm_buf, lw.q_w, lw.q_b, q_buf, seq_len, D, D, false, false);
            try self.dispatchLinear(norm_buf, lw.k_w, lw.k_b, k_buf, seq_len, D, D, false, false);
            try self.dispatchLinear(norm_buf, lw.v_w, lw.v_b, v_buf, seq_len, D, D, false, false);

            // Interleaved 2D RoPE on Q and K (row 0 = CLS unrotated)
            {
                var rp: gpu.Params = .{
                    .rows = seq_len,
                    .xs = D,
                    .heads = c.num_attention_heads,
                    .dim = hd,
                    .window = gh,
                    .capacity = gw,
                    .scale = c.rope_theta,
                };
                const rope_threads = @as(usize, seq_len) * c.num_attention_heads * (hd / 2);
                try gpu.dispatch(self.ctx, "mk_vis_rope2d", &.{q_buf}, &rp, rope_threads, 256);
                try gpu.dispatch(self.ctx, "mk_vis_rope2d", &.{k_buf}, &rp, rope_threads, 256);
            }

            // Bidirectional MHA
            {
                var ap: gpu.Params = .{
                    .rows = seq_len,
                    .xs = D,
                    .ys = D,
                    .zs = D,
                    .window = D,
                    .heads = c.num_attention_heads,
                    .dim = hd,
                    .scale = att_scale,
                };
                const att_threads = @as(usize, seq_len) * c.num_attention_heads * 32;
                try gpu.dispatch(self.ctx, "mk_vis_attention", &.{ q_buf, k_buf, v_buf, att_buf }, &ap, att_threads, 128);
            }

            // Output projection -> norm_buf (reused scratch)
            try self.dispatchLinear(att_buf, lw.out_w, lw.out_b, norm_buf, seq_len, D, D, false, false);

            // LayerScale 1 + residual: h_buf += ls_1 * norm_buf
            {
                var lsp: gpu.Params = .{
                    .w = lw.ls_1.tensor.offset,
                    .rows = seq_len,
                    .xs = D,
                    .ys = D,
                    .n = D,
                    .dtype = @backingInt(lw.ls_1.tensor.dtype),
                };
                try gpu.dispatch(self.ctx, "mk_vis_layerscale_add", &.{ h_buf, norm_buf, lw.ls_1.buffer }, &lsp, @as(usize, seq_len) * D, 256);
            }

            // Pre-MLP LayerNorm
            try self.dispatchLayerNorm(h_buf, lw.ln_2_w, lw.ln_2_b, norm_buf, seq_len, D);

            // MLP: c_fc + GELU -> mlp_buf, c_proj -> att_buf
            try self.dispatchLinear(norm_buf, lw.fc_w, lw.fc_b, mlp_buf, seq_len, M, D, true, false);
            try self.dispatchLinear(mlp_buf, lw.proj_w, lw.proj_b, att_buf, seq_len, D, M, false, false);

            // LayerScale 2 + residual: h_buf += ls_2 * att_buf
            {
                var lsp: gpu.Params = .{
                    .w = lw.ls_2.tensor.offset,
                    .rows = seq_len,
                    .xs = D,
                    .ys = D,
                    .n = D,
                    .dtype = @backingInt(lw.ls_2.tensor.dtype),
                };
                try gpu.dispatch(self.ctx, "mk_vis_layerscale_add", &.{ h_buf, att_buf, lw.ls_2.buffer }, &lsp, @as(usize, seq_len) * D, 256);
            }
        }

        // 5. Drop CLS + 2x2 spatial merge -> [n_vis, merge_dim]
        {
            var mp: gpu.Params = .{
                .xs = D,
                .ys = merge_dim,
                .n = D,
                .heads = gh,
                .kv_heads = gw,
                .dim = c.adapter_merge_size,
            };
            try gpu.dispatch(self.ctx, "mk_vis_spatial_merge", &.{ h_buf, merged_buf }, &mp, @as(usize, n_vis) * merge_dim, 256);
        }

        // 6. Multimodal projector: proj_in + GELU -> proj_out (rounded to BF16 for LLM residual stream)
        try self.dispatchLinear(merged_buf, self.proj_in_w, null, proj_mid_buf, n_vis, c.projection_hidden_dim, merge_dim, true, false);
        try self.dispatchLinear(proj_mid_buf, self.proj_out_w, null, out_buf, n_vis, c.text_hidden_size, c.projection_hidden_dim, false, true);
        try self.ctx.finish();

        return .{
            .buffer = out_buf,
            .tokens = n_vis,
            .grid_h = gh / c.adapter_merge_size,
            .grid_w = gw / c.adapter_merge_size,
            .width = c.text_hidden_size,
        };
    }

    pub fn encodeRgb(self: *const VisionEncoder, a: Allocator, rgb: []const u8, w: u32, h: u32, max_image_tokens: u32) !EncodedImage {
        const prep = try preprocessRgb(a, rgb, w, h, self.cfg, max_image_tokens);
        defer a.free(prep.patches);
        return self.encodePreprocessed(prep);
    }

    pub fn encodeBytes(self: *const VisionEncoder, a: Allocator, bytes: []const u8, max_image_tokens: u32) !EncodedImage {
        const img = try decodeImageBytes(bytes);
        defer img.deinit();
        return self.encodeRgb(a, img.slice(), img.width, img.height, max_image_tokens);
    }

    pub fn encodeFile(self: *const VisionEncoder, a: Allocator, path: []const u8, max_image_tokens: u32) !EncodedImage {
        const img = try decodeImageFile(a, path);
        defer img.deinit();
        return self.encodeRgb(a, img.slice(), img.width, img.height, max_image_tokens);
    }

    pub fn encodeInput(self: *const VisionEncoder, a: Allocator, input: ImageInput, max_image_tokens: u32) !EncodedImage {
        return switch (input) {
            .source => |s| self.encodeFile(a, s, max_image_tokens),
            .bytes => |b| self.encodeBytes(a, b, max_image_tokens),
        };
    }

    /// Encode one or more image inputs (paths/URLs/data URIs or raw encoded bytes) into one contiguous GPU buffer.
    pub fn encodeInputs(self: *const VisionEncoder, a: Allocator, inputs: []const ImageInput, max_image_tokens: u32) !EncodedBatch {
        if (inputs.len == 0) return error.NoImages;
        const encoded = try a.alloc(EncodedImage, inputs.len);
        var built: usize = 0;
        errdefer for (encoded[0..built]) |e| e.deinit();
        defer for (encoded[0..built]) |e| e.deinit();

        const counts = try self.gpa.alloc(u32, inputs.len);
        errdefer self.gpa.free(counts);

        var total: u32 = 0;
        for (inputs, 0..) |inp, i| {
            encoded[i] = try self.encodeInput(a, inp, max_image_tokens);
            built += 1;
            counts[i] = encoded[i].tokens;
            total += encoded[i].tokens;
        }
        const W = self.cfg.text_hidden_size;
        const buf = try self.ctx.alloc(@as(usize, total) * W);
        errdefer buf.deinit();
        const dst = buf.floats();
        var row: usize = 0;
        for (encoded) |e| {
            const words = @as(usize, e.tokens) * W;
            @memcpy(dst[row * W ..][0..words], e.buffer.floats()[0..words]);
            row += e.tokens;
        }
        return .{ .buffer = buf, .counts = counts, .total_tokens = total, .width = W };
    }

    /// Encode one or more image files/URIs and pack their visual embeddings into one contiguous GPU buffer.
    pub fn encodeFiles(self: *const VisionEncoder, a: Allocator, paths: []const []const u8, max_image_tokens: u32) !EncodedBatch {
        const inputs = try a.alloc(ImageInput, paths.len);
        defer a.free(inputs);
        for (paths, 0..) |p, i| inputs[i] = .{ .source = p };
        return self.encodeInputs(a, inputs, max_image_tokens);
    }
};

// ---- Unit tests -------------------------------------------------------------

fn cpuErf(x_in: f64) f64 {
    const sign: f64 = if (x_in < 0) -1.0 else 1.0;
    const x = @abs(x_in);
    const t = 1.0 / (1.0 + 0.3275911 * x);
    const poly = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
    return sign * (1.0 - poly * @exp(-x * x));
}

fn cpuGelu(x: f64) f64 {
    return 0.5 * x * (1.0 + cpuErf(x * std.math.sqrt1_2));
}

test "parseConfig accepts Inkling 2 Mini vision_config and computes dimensions" {
    const cfg_json =
        \\{"model_type":"inkling2_loop","vision_config":{
        \\"model_type":"inkling2_vision","text_hidden_size":2304,"temporal_patch_size":2,
        \\"native_patch_size":20,"adapter_merge_size":2,"projection_hidden_dim":4096,
        \\"projector_bias":false,"hidden_size":1024,"num_hidden_layers":23,
        \\"num_attention_heads":16,"mlp_ratio":4.0,"posemb_grid_size":32,"layer_norm_eps":1e-5}}
    ;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const cfg = try parseConfig(arena.allocator(), cfg_json);
    try std.testing.expectEqual(@as(u32, 2304), cfg.text_hidden_size);
    try std.testing.expectEqual(@as(u32, 64), cfg.headDim());
    try std.testing.expectEqual(@as(u32, 4096), cfg.mlpDim());
    try std.testing.expectEqual(@as(u32, 1200), cfg.patchPixels());
    try std.testing.expectEqual(@as(u32, 40), cfg.tilePixels());

    // Exact multiple of 40 -> no resize
    const d1 = computeTargetDims(80, 120, cfg, 1024, 4096);
    try std.testing.expectEqual(@as(u32, 80), d1.padded_w);
    try std.testing.expectEqual(@as(u32, 120), d1.padded_h);
    try std.testing.expectEqual(@as(u32, 6), d1.visual_tokens);
    try std.testing.expectEqual(@as(u32, 6), d1.grid_h);
    try std.testing.expectEqual(@as(u32, 4), d1.grid_w);

    // Non-multiple of 40 -> padded right/bottom
    const d2 = computeTargetDims(50, 30, cfg, 1024, 4096);
    try std.testing.expectEqual(@as(u32, 50), d2.resized_w);
    try std.testing.expectEqual(@as(u32, 30), d2.resized_h);
    try std.testing.expectEqual(@as(u32, 80), d2.padded_w);
    try std.testing.expectEqual(@as(u32, 40), d2.padded_h);
    try std.testing.expectEqual(@as(u32, 2), d2.visual_tokens);

    // Large image respects max_image_tokens budget
    const d3 = computeTargetDims(3840, 2160, cfg, 256, 4096);
    try std.testing.expect(d3.visual_tokens <= 256);
    try std.testing.expect(d3.padded_w % 40 == 0 and d3.padded_h % 40 == 0);
}

test "vision Metal kernels match CPU float64 reference" {
    const gpa = std.testing.allocator;
    const ctx = try gpu.open();
    defer ctx.deinit();

    // 1. mk_vis_linear_mma (with bias + GELU) on non-multiple-of-32 dims (rows=5, N=12, M=19)
    {
        const R: u32 = 5;
        const N: u32 = 12;
        const M: u32 = 19;
        const x = try ctx.alloc(R * N);
        defer x.deinit();
        const w = try ctx.alloc(M * N);
        defer w.deinit();
        const b = try ctx.alloc(M);
        defer b.deinit();
        const y = try ctx.alloc(R * M);
        defer y.deinit();
        for (x.floats(), 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 13)) - 6)) * 0.25;
        for (w.floats(), 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 7) % 11)) - 5)) * 0.15;
        for (b.floats(), 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 5)) - 2)) * 0.1;

        var p: gpu.Params = .{ .rows = R, .xs = N, .ys = M, .n = N, .m = M, .dtype = 0, .bdtype = 0, .flags = gpu.flag_has_weight, .act_a = 4 };
        try gpu.dispatch(ctx, "mk_vis_linear_mma", &.{ x, w, b, y }, &p, 128, 128);
        try ctx.finish();

        for (0..R) |r| {
            for (0..M) |m_i| {
                var acc: f64 = b.floats()[m_i];
                for (0..N) |k| acc += @as(f64, x.floats()[r * N + k]) * w.floats()[m_i * N + k];
                const expected: f32 = @floatCast(cpuGelu(acc));
                try std.testing.expectApproxEqAbs(expected, y.floats()[r * M + m_i], 1e-4);
            }
        }
    }

    // 2. mk_vis_layernorm
    {
        const R: u32 = 3;
        const D: u32 = 32;
        const x = try ctx.alloc(R * D);
        defer x.deinit();
        const g = try ctx.alloc(D);
        defer g.deinit();
        const b = try ctx.alloc(D);
        defer b.deinit();
        const y = try ctx.alloc(R * D);
        defer y.deinit();
        for (x.floats(), 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 17)) - 8)) * 0.3;
        for (g.floats(), 0..) |*v, i| v.* = 1.0 + @as(f32, @floatFromInt(i % 5)) * 0.1;
        for (b.floats(), 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 7)) - 3)) * 0.05;

        var p: gpu.Params = .{ .rows = R, .xs = D, .ys = D, .n = D, .dtype = 0, .bdtype = 0, .eps = 1e-5 };
        try gpu.dispatch(ctx, "mk_vis_layernorm", &.{ x, g, b, y }, &p, R * 256, 256);
        try ctx.finish();

        for (0..R) |r| {
            var mean: f64 = 0;
            for (0..D) |c| mean += x.floats()[r * D + c];
            mean /= D;
            var vr: f64 = 0;
            for (0..D) |c| {
                const d = x.floats()[r * D + c] - mean;
                vr += d * d;
            }
            const inv = 1.0 / @sqrt(vr / D + 1e-5);
            for (0..D) |c| {
                const exp: f32 = @floatCast((x.floats()[r * D + c] - mean) * inv * g.floats()[c] + b.floats()[c]);
                try std.testing.expectApproxEqAbs(exp, y.floats()[r * D + c], 1e-5);
            }
        }
    }

    // 3. mk_vis_rope2d + mk_vis_attention + mk_vis_spatial_merge + mk_scatter_embeds
    {
        const gh: u32 = 2;
        const gw: u32 = 2;
        const S: u32 = 1 + gh * gw;
        const heads: u32 = 2;
        const hd: u32 = 8;
        const D: u32 = heads * hd;

        const q = try ctx.alloc(S * D);
        defer q.deinit();
        const k = try ctx.alloc(S * D);
        defer k.deinit();
        const v = try ctx.alloc(S * D);
        defer v.deinit();
        const out = try ctx.alloc(S * D);
        defer out.deinit();
        for (q.floats(), 0..) |*val, i| val.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 9)) - 4)) * 0.2;
        for (k.floats(), 0..) |*val, i| val.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 3) % 9)) - 4)) * 0.2;
        for (v.floats(), 0..) |*val, i| val.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 5) % 11)) - 5)) * 0.1;

        const q_cls_before = try gpa.dupe(f32, q.floats()[0..D]);
        defer gpa.free(q_cls_before);
        const q_patch1_before = try gpa.dupe(f32, q.floats()[2 * D .. 3 * D]); // patch idx 1 => (py=0, px=1) => (row=1, col=2)
        defer gpa.free(q_patch1_before);

        var rp: gpu.Params = .{ .rows = S, .xs = D, .heads = heads, .dim = hd, .window = gh, .capacity = gw, .scale = 10000.0 };
        try gpu.dispatch(ctx, "mk_vis_rope2d", &.{q}, &rp, S * heads * (hd / 2), 64);
        try gpu.dispatch(ctx, "mk_vis_rope2d", &.{k}, &rp, S * heads * (hd / 2), 64);
        var ap: gpu.Params = .{ .rows = S, .xs = D, .ys = D, .zs = D, .window = D, .heads = heads, .dim = hd, .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd))) };
        try gpu.dispatch(ctx, "mk_vis_attention", &.{ q, k, v, out }, &ap, S * heads * 32, 128);
        try ctx.finish();

        // CLS row (row 0) must be unrotated by 2D RoPE
        try std.testing.expectEqualSlices(f32, q_cls_before, q.floats()[0..D]);

        // Patch 1 (py=0, px=1) rotates first half (pair 0) by col+1=2.0, second half (pair 2) by row+1=1.0
        {
            const a0 = q_patch1_before[0];
            const b0 = q_patch1_before[1];
            try std.testing.expectApproxEqAbs(a0 * @cos(@as(f32, 2.0)) - b0 * @sin(@as(f32, 2.0)), q.floats()[2 * D + 0], 1e-5);
            try std.testing.expectApproxEqAbs(a0 * @sin(@as(f32, 2.0)) + b0 * @cos(@as(f32, 2.0)), q.floats()[2 * D + 1], 1e-5);
            const a2 = q_patch1_before[4];
            const b2 = q_patch1_before[5];
            try std.testing.expectApproxEqAbs(a2 * @cos(@as(f32, 1.0)) - b2 * @sin(@as(f32, 1.0)), q.floats()[2 * D + 4], 1e-5);
            try std.testing.expectApproxEqAbs(a2 * @sin(@as(f32, 1.0)) + b2 * @cos(@as(f32, 1.0)), q.floats()[2 * D + 5], 1e-5);
        }

        // Verify bidirectional attention vs CPU reference
        for (0..S) |r| {
            for (0..heads) |h_i| {
                var scores: [5]f64 = undefined;
                var m: f64 = -std.math.inf(f64);
                for (0..S) |j| {
                    var dot: f64 = 0;
                    for (0..hd) |d_i| dot += @as(f64, q.floats()[r * D + h_i * hd + d_i]) * k.floats()[j * D + h_i * hd + d_i];
                    scores[j] = dot / @sqrt(@as(f64, @floatFromInt(hd)));
                    m = @max(m, scores[j]);
                }
                var denom: f64 = 0;
                for (0..S) |j| {
                    scores[j] = @exp(scores[j] - m);
                    denom += scores[j];
                }
                for (0..hd) |d_i| {
                    var acc: f64 = 0;
                    for (0..S) |j| acc += (scores[j] / denom) * v.floats()[j * D + h_i * hd + d_i];
                    try std.testing.expectApproxEqAbs(@as(f32, @floatCast(acc)), out.floats()[r * D + h_i * hd + d_i], 1e-5);
                }
            }
        }

        // Spatial merge: [1 + 2*2, D] -> [1, 4*D], dropping CLS at row 0
        const merged = try ctx.alloc(4 * D);
        defer merged.deinit();
        var mp: gpu.Params = .{ .xs = D, .ys = 4 * D, .n = D, .heads = gh, .kv_heads = gw, .dim = 2 };
        try gpu.dispatch(ctx, "mk_vis_spatial_merge", &.{ out, merged }, &mp, 4 * D, 64);
        try ctx.finish();
        for (0..4) |sub| {
            try std.testing.expectEqualSlices(f32, out.floats()[(1 + sub) * D ..][0..D], merged.floats()[sub * D ..][0..D]);
        }

        // Scatter embeds into arena
        const arena_buf = try ctx.alloc(4 * D);
        defer arena_buf.deinit();
        @memset(arena_buf.floats(), 99.0);
        const vis_buf = try ctx.alloc(2 * D);
        defer vis_buf.deinit();
        for (vis_buf.floats(), 0..) |*val, i| val.* = @as(f32, @floatFromInt(i + 1));
        const row_map = try ctx.alloc(4);
        defer row_map.deinit();
        @memcpy(row_map.structs(i32), &[_]i32{ -1, 0, 1, -1 });
        var sp: gpu.Params = .{ .rows = 4, .n = D, .xs = D, .ys = D };
        try gpu.dispatch(ctx, "mk_scatter_embeds", &.{ arena_buf, vis_buf, row_map }, &sp, 4 * D, 64);
        try ctx.finish();
        try std.testing.expectEqual(@as(f32, 99.0), arena_buf.floats()[0]);
        try std.testing.expectEqualSlices(f32, vis_buf.floats()[0..D], arena_buf.floats()[D .. 2 * D]);
        try std.testing.expectEqualSlices(f32, vis_buf.floats()[D .. 2 * D], arena_buf.floats()[2 * D .. 3 * D]);
        try std.testing.expectEqual(@as(f32, 99.0), arena_buf.floats()[3 * D]);
    }
}

test "inkling_vision decodes file:// URIs, data: base64 URIs, and raw PNG bytes" {
    const a = std.testing.allocator;
    const p1 = try decodeFileUri(a, "file:///tmp/hello%20world/img%231.png");
    defer a.free(p1);
    try std.testing.expectEqualStrings("/tmp/hello world/img#1.png", p1);

    const p2 = try decodeFileUri(a, "file://localhost/var/img.jpg");
    defer a.free(p2);
    try std.testing.expectEqualStrings("/var/img.jpg", p2);

    try std.testing.expectError(error.InvalidImageUri, decodeFileUri(a, "http://example.com/a.png"));
    try std.testing.expectError(error.InvalidImageUri, decodeDataUriBytes(a, "data:image/png,notbase64"));

    // 1x1 opaque red PNG (#FF0000)
    const red_png_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC";
    const img = try decodeImageFile(a, red_png_uri);
    defer img.deinit();
    try std.testing.expectEqual(@as(u32, 1), img.width);
    try std.testing.expectEqual(@as(u32, 1), img.height);
    try std.testing.expectEqualSlices(u8, &.{ 255, 0, 0 }, img.slice());

    const raw_bytes = try decodeDataUriBytes(a, red_png_uri);
    defer a.free(raw_bytes);
    const img2 = try decodeImageBytes(raw_bytes);
    defer img2.deinit();
    try std.testing.expectEqual(@as(u32, 1), img2.width);
    try std.testing.expectEqual(@as(u32, 1), img2.height);
    try std.testing.expectEqualSlices(u8, &.{ 255, 0, 0 }, img2.slice());
}
