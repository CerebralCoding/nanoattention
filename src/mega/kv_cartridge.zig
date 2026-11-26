//! Binary persistent KV/ring cartridge export & import (`NANOKV01`).
//! Stores persistent state rows in chronological logical order (`first_pos .. first_pos + rows - 1`)
//! so cartridges are independent of engine batch size and rebuild derived caches on load.
const std = @import("std");
const exec = @import("exec.zig");
const io = @import("io.zig");
const ir = @import("ir.zig");
const weights = @import("weights");
const metal = weights.metal;

extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
extern "c" fn close(c_int) c_int;
extern "c" fn rename([*:0]const u8, [*:0]const u8) c_int;
extern "c" fn unlink([*:0]const u8) c_int;

pub const magic = "NANOKV01";
pub const format_version: u32 = 1;
const header_bytes: usize = 64;
const dir_fixed_bytes: usize = 40;
const zero_pad: [8]u8 = @splat(0);

pub const DType = enum(u32) {
    f32 = 0,
    bf16 = 1,

    pub fn elemBytes(self: DType) u64 {
        return switch (self) {
            .f32 => 4,
            .bf16 => 2,
        };
    }

    pub fn name(self: DType) []const u8 {
        return switch (self) {
            .f32 => "f32",
            .bf16 => "bf16",
        };
    }
};

pub fn parseDType(s: []const u8) ?DType {
    if (std.mem.eql(u8, s, "f32")) return .f32;
    if (std.mem.eql(u8, s, "bf16")) return .bf16;
    return null;
}

pub const SaveSummary = struct {
    position: u32,
    states: u32,
    bytes: u64,
    dtype: DType,
};

pub const LoadSummary = struct {
    position: u32,
    states: u32,
    bytes: u64,
    dtype: DType,
};

fn align8(v: u64) u64 {
    return (v + 7) & ~@as(u64, 7);
}

fn padBytes(len: u64) []const u8 {
    const p: usize = @intCast(align8(len) - len);
    return zero_pad[0..p];
}

const SaveEntry = struct {
    state_id: usize,
    s: ir.State,
    first_pos: u32,
    rows: u32,
    data_offset: u64,
    data_bytes: u64,
};

/// Atomically save all persistent `.sequence` and `.ring` states from `engine` to `path`.
pub fn save(engine: *exec.Engine, a: std.mem.Allocator, path: []const u8, dtype: DType, meta_json: []const u8) !SaveSummary {
    try engine.d2Settle();
    const pos = engine.position;
    const model_type = engine.program.meta.model_type;
    const elem_bytes = dtype.elemBytes();

    var entries: std.ArrayList(SaveEntry) = .empty;
    defer entries.deinit(a);

    var header_cursor: u64 = header_bytes;
    header_cursor += align8(model_type.len);
    header_cursor += align8(meta_json.len);

    for (engine.program.states, 0..) |s, id| {
        if (!s.persistent()) continue;
        const rows: u32 = switch (s.kind) {
            .sequence => pos,
            .ring => @min(pos, s.horizon),
            else => unreachable,
        };
        const first_pos: u32 = pos - rows;
        const data_bytes: u64 = @as(u64, rows) * s.width * elem_bytes;
        try entries.append(a, .{
            .state_id = id,
            .s = s,
            .first_pos = first_pos,
            .rows = rows,
            .data_offset = 0,
            .data_bytes = data_bytes,
        });
        header_cursor += dir_fixed_bytes + align8(s.name.len);
    }

    var data_cursor: u64 = align8(header_cursor);
    for (entries.items) |*e| {
        e.data_offset = data_cursor;
        data_cursor += align8(e.data_bytes);
    }

    const tmp = try std.fmt.allocPrintSentinel(a, "{s}.tmp", .{path}, 0);
    const final = try a.dupeSentinel(u8, path, 0);
    // Darwin: O_WRONLY | O_CREAT | O_TRUNC.
    const fd = open(tmp, 0x1 | 0x200 | 0x400, @as(c_uint, 0o644));
    if (fd < 0) return error.CannotCreateFile;
    errdefer {
        _ = close(fd);
        _ = unlink(tmp);
    }

    var hdr: [header_bytes]u8 = @splat(0);
    @memcpy(hdr[0..8], magic);
    std.mem.writeInt(u32, hdr[8..12], format_version, .little);
    std.mem.writeInt(u32, hdr[12..16], @intFromEnum(dtype), .little);
    std.mem.writeInt(u32, hdr[16..20], pos, .little);
    std.mem.writeInt(u32, hdr[20..24], @intCast(entries.items.len), .little);
    std.mem.writeInt(u32, hdr[24..28], @intCast(model_type.len), .little);
    std.mem.writeInt(u32, hdr[28..32], @intCast(meta_json.len), .little);
    std.mem.writeInt(u32, hdr[32..36], 0, .little);
    try io.writeAll(fd, &hdr);
    try io.writeAll(fd, model_type);
    try io.writeAll(fd, padBytes(model_type.len));
    try io.writeAll(fd, meta_json);
    try io.writeAll(fd, padBytes(meta_json.len));

    for (entries.items) |e| {
        var rec: [dir_fixed_bytes]u8 = @splat(0);
        std.mem.writeInt(u32, rec[0..4], @intCast(e.s.name.len), .little);
        std.mem.writeInt(u32, rec[4..8], @backingInt(e.s.kind), .little);
        std.mem.writeInt(u32, rec[8..12], e.s.width, .little);
        std.mem.writeInt(u32, rec[12..16], e.s.horizon, .little);
        std.mem.writeInt(u32, rec[16..20], e.first_pos, .little);
        std.mem.writeInt(u32, rec[20..24], e.rows, .little);
        std.mem.writeInt(u64, rec[24..32], e.data_offset, .little);
        std.mem.writeInt(u64, rec[32..40], e.data_bytes, .little);
        try io.writeAll(fd, &rec);
        try io.writeAll(fd, e.s.name);
        try io.writeAll(fd, padBytes(e.s.name.len));
    }

    var stage_u16: std.ArrayList(u16) = .empty;
    defer stage_u16.deinit(a);
    const max_stage_elems: usize = 32768;

    for (entries.items) |e| {
        if (e.rows > 0) {
            const buf = engine.caches[e.state_id] orelse return error.MissingBackingCache;
            const floats = buf.floats();
            const cap = e.s.capacity(engine.batch, engine.context);
            const width: usize = e.s.width;
            if (e.rows > cap) return error.CartridgeCapacityExceeded;
            if (dtype == .f32) {
                if (e.s.kind == .sequence and e.first_pos == 0) {
                    const total_floats = @as(usize, e.rows) * width;
                    try io.writeAll(fd, std.mem.sliceAsBytes(floats[0..total_floats]));
                } else {
                    for (0..e.rows) |r| {
                        const slot: usize = (e.first_pos + @as(u32, @intCast(r))) % cap;
                        try io.writeAll(fd, std.mem.sliceAsBytes(floats[slot * width ..][0..width]));
                    }
                }
            } else {
                const chunk_rows: u32 = @max(@as(u32, 1), @as(u32, @intCast(max_stage_elems / @max(width, 1))));
                try stage_u16.resize(a, @as(usize, @min(e.rows, chunk_rows)) * width);
                var r: u32 = 0;
                while (r < e.rows) {
                    const take = @min(chunk_rows, e.rows - r);
                    for (0..take) |dr| {
                        const slot: usize = (e.first_pos + r + @as(u32, @intCast(dr))) % cap;
                        const src_row = floats[slot * width ..][0..width];
                        const dst_row = stage_u16.items[dr * width ..][0..width];
                        for (src_row, dst_row) |v, *h| h.* = @truncate(@as(u32, @bitCast(v)) >> 16);
                    }
                    try io.writeAll(fd, std.mem.sliceAsBytes(stage_u16.items[0 .. @as(usize, take) * width]));
                    r += take;
                }
            }
            try io.writeAll(fd, padBytes(e.data_bytes));
        }
    }

    if (close(fd) != 0) return error.OutputFailed;
    if (rename(tmp, final) != 0) return error.CannotRenameFile;

    return .{
        .position = pos,
        .states = @intCast(entries.items.len),
        .bytes = data_cursor,
        .dtype = dtype,
    };
}

const LoadEntry = struct {
    state_id: usize,
    s: ir.State,
    first_pos: u32,
    rows: u32,
    data_offset: usize,
    data_bytes: usize,
};

/// Memory-map `path`, validate against `engine.program`, load state rows into physical
/// cache slots, and rebuild derived BF16 mirrors / LSA Q4 pooled blocks.
pub fn load(engine: *exec.Engine, a: std.mem.Allocator, path: []const u8) !LoadSummary {
    try engine.d2Settle();
    const map = try metal.Mapping.open(a, path);
    defer map.deinit();
    const raw = map.bytes;
    if (raw.len < header_bytes) return error.TruncatedCartridge;
    if (!std.mem.eql(u8, raw[0..8], magic)) return error.InvalidCartridgeMagic;
    if (std.mem.readInt(u32, raw[8..12], .little) != format_version) return error.UnsupportedCartridgeVersion;
    const dtype_raw = std.mem.readInt(u32, raw[12..16], .little);
    const dtype: DType = switch (dtype_raw) {
        0 => .f32,
        1 => .bf16,
        else => return error.InvalidCartridgeDType,
    };
    const pos = std.mem.readInt(u32, raw[16..20], .little);
    const state_count = std.mem.readInt(u32, raw[20..24], .little);
    const model_type_len = std.mem.readInt(u32, raw[24..28], .little);
    const meta_json_len = std.mem.readInt(u32, raw[28..32], .little);
    if (pos > engine.context) return error.ContextExhausted;

    var cursor: usize = header_bytes;
    const model_padded: usize = @intCast(align8(model_type_len));
    if (cursor + model_padded > raw.len) return error.TruncatedCartridge;
    const model_type = raw[cursor..][0..model_type_len];
    if (!std.mem.eql(u8, model_type, engine.program.meta.model_type)) return error.CartridgeModelMismatch;
    cursor += model_padded;

    const meta_padded: usize = @intCast(align8(meta_json_len));
    if (cursor + meta_padded > raw.len) return error.TruncatedCartridge;
    cursor += meta_padded;

    const states = engine.program.states;
    const seen = try a.alloc(bool, states.len);
    defer a.free(seen);
    @memset(seen, false);

    const entries = try a.alloc(LoadEntry, state_count);
    defer a.free(entries);

    const elem_bytes = dtype.elemBytes();
    for (0..state_count) |i| {
        if (cursor + dir_fixed_bytes > raw.len) return error.TruncatedCartridge;
        const rec = raw[cursor..][0..dir_fixed_bytes];
        const name_len = std.mem.readInt(u32, rec[0..4], .little);
        const kind_raw = std.mem.readInt(u32, rec[4..8], .little);
        const width = std.mem.readInt(u32, rec[8..12], .little);
        const horizon = std.mem.readInt(u32, rec[12..16], .little);
        const first_pos = std.mem.readInt(u32, rec[16..20], .little);
        const rows = std.mem.readInt(u32, rec[20..24], .little);
        const data_offset = std.mem.readInt(u64, rec[24..32], .little);
        const data_bytes = std.mem.readInt(u64, rec[32..40], .little);

        const name_padded: usize = @intCast(align8(name_len));
        if (cursor + dir_fixed_bytes + name_padded > raw.len) return error.TruncatedCartridge;
        const name = raw[cursor + dir_fixed_bytes ..][0..name_len];
        cursor += dir_fixed_bytes + name_padded;

        var matched_id: ?usize = null;
        for (states, 0..) |s, sid| {
            if (std.mem.eql(u8, s.name, name)) {
                matched_id = sid;
                break;
            }
        }
        const sid = matched_id orelse return error.UnknownCartridgeState;
        const s = states[sid];
        if (!s.persistent()) return error.UnknownCartridgeState;
        if (seen[sid]) return error.DuplicateCartridgeState;
        seen[sid] = true;

        if (kind_raw != @backingInt(s.kind) or width != s.width or horizon != s.horizon) return error.CartridgeStateMismatch;
        if (first_pos > pos or @as(u64, first_pos) + rows > pos) return error.InvalidCartridgeRange;
        const cap = s.capacity(engine.batch, engine.context);
        if (rows > cap) return error.CartridgeCapacityExceeded;
        const expected_bytes: u64 = @as(u64, rows) * width * elem_bytes;
        if (data_bytes != expected_bytes) return error.CartridgeDataSizeMismatch;
        if (data_offset > raw.len or data_bytes > raw.len - data_offset) return error.TruncatedCartridge;

        entries[i] = .{
            .state_id = sid,
            .s = s,
            .first_pos = first_pos,
            .rows = rows,
            .data_offset = @intCast(data_offset),
            .data_bytes = @intCast(data_bytes),
        };
    }

    for (states, seen) |s, was_seen| {
        if (s.persistent() and !was_seen) return error.MissingCartridgeState;
    }

    engine.reset();
    for (engine.caches) |c| if (c) |buf| @memset(buf.floats(), 0);

    for (entries) |e| {
        if (e.rows == 0) continue;
        const buf = engine.caches[e.state_id] orelse return error.MissingBackingCache;
        const dst = buf.floats();
        const cap = e.s.capacity(engine.batch, engine.context);
        const width: usize = e.s.width;
        const src_bytes = raw[e.data_offset..][0..e.data_bytes];
        if (dtype == .f32) {
            if (e.s.kind == .sequence and e.first_pos == 0) {
                @memcpy(std.mem.sliceAsBytes(dst[0 .. @as(usize, e.rows) * width]), src_bytes);
            } else {
                const row_bytes = width * 4;
                for (0..e.rows) |r| {
                    const slot: usize = (e.first_pos + @as(u32, @intCast(r))) % cap;
                    @memcpy(std.mem.sliceAsBytes(dst[slot * width ..][0..width]), src_bytes[r * row_bytes ..][0..row_bytes]);
                }
            }
        } else {
            const row_bytes = width * 2;
            for (0..e.rows) |r| {
                const slot: usize = (e.first_pos + @as(u32, @intCast(r))) % cap;
                const d_row = dst[slot * width ..][0..width];
                const s_row = src_bytes[r * row_bytes ..][0..row_bytes];
                for (0..width) |c| {
                    const h = std.mem.readInt(u16, s_row[c * 2 ..][0..2], .little);
                    d_row[c] = @bitCast(@as(u32, h) << 16);
                }
            }
        }
    }

    try engine.syncDerivedCaches(pos);

    return .{
        .position = pos,
        .states = state_count,
        .bytes = raw.len,
        .dtype = dtype,
    };
}

extern "c" fn mkdtemp([*:0]u8) ?[*:0]u8;
extern "c" fn rmdir([*:0]const u8) c_int;

test "nkv cartridge round-trips cross-batch f32, bf16+stripes+LSA, and error cases" {
    const inkling = @import("inkling2_loop.zig");
    const gpu = @import("gpu.zig");
    const gpa = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    var c = try inkling.parse(a, inkling.test_config);
    c.hidden = 128;
    c.intermediate = 128;
    c.heads = 3;
    c.kv_heads = 1;
    c.head_dim = 128;
    c.rel_dim = 4;
    c.rel_extent = 4;
    c.window = 4;
    c.log_scaling_floor = 3;
    c.vocab = 16;
    c.unpadded_vocab = 14;
    c.eos = try a.dupe(u32, &.{13});
    c.gate_rank = 4;
    c.max_context = 48;

    var p = try inkling.lower(gpa, c);
    defer p.deinit();

    var tmp = "/tmp/nano-nkv-XXXXXX".*;
    const dir = std.mem.span(mkdtemp(&tmp) orelse return error.TempDirectoryFailed);
    defer _ = rmdir(dir);
    const weights_path = try std.fmt.allocPrintSentinel(a, "{s}/model.safetensors", .{dir}, 0);
    defer _ = unlink(weights_path);
    const kv_f32_path = try std.fmt.allocPrintSentinel(a, "{s}/cart_f32.nkv", .{dir}, 0);
    defer _ = unlink(kv_f32_path);
    const kv_bf16_path = try std.fmt.allocPrintSentinel(a, "{s}/cart_bf16.nkv", .{dir}, 0);
    defer _ = unlink(kv_bf16_path);
    try inkling.writeFixture(a, &p, weights_path);

    const ctx = try gpu.open();
    defer ctx.deinit();
    var store = try weights.Store.init(a, ctx, dir, 0);
    defer store.deinit();

    const tokens = [_]u32{ 1, 4, 11, 2, 3, 6, 9, 1, 5, 7, 10, 2, 8, 3, 12, 0, 4, 6, 2, 9, 1, 7, 11, 3, 5, 8, 10, 4 };
    const Engine = exec.Engine;

    var eng_a = try Engine.init(gpa, &p, ctx, &store, .{ .batch = 16 }, 48);
    defer eng_a.deinit();
    try eng_a.forward(tokens[0..16], 16);
    try eng_a.forward(tokens[16..20], 4);
    const saved_f32 = try save(&eng_a, a, kv_f32_path, .f32, "{\"doc\":\"test\"}");
    try std.testing.expectEqual(@as(u32, 20), saved_f32.position);
    const saved_bf16 = try save(&eng_a, a, kv_bf16_path, .bf16, "{}");
    try std.testing.expectEqual(@as(u32, 20), saved_bf16.position);
    try std.testing.expect(saved_bf16.bytes < saved_f32.bytes);

    var eng_ref = try Engine.init(gpa, &p, ctx, &store, .{ .batch = 8 }, 48);
    defer eng_ref.deinit();
    try eng_ref.forward(tokens[0..8], 8);
    try eng_ref.forward(tokens[8..16], 8);
    try eng_ref.forward(tokens[16..24], 8);
    try eng_ref.forward(tokens[24..28], 3);

    var eng_b = try Engine.init(gpa, &p, ctx, &store, .{ .batch = 4 }, 48);
    defer eng_b.deinit();
    const loaded_f32 = try load(&eng_b, a, kv_f32_path);
    try std.testing.expectEqual(@as(u32, 20), loaded_f32.position);
    try std.testing.expectEqual(@as(u32, 20), eng_b.position);
    try eng_b.forward(tokens[20..24], 4);
    try eng_b.forward(tokens[24..28], 3);
    try std.testing.expectEqualSlices(f32, try eng_ref.logits(3), try eng_b.logits(3));
    try std.testing.expectEqual(try eng_ref.next(3), try eng_b.next(3));

    for (0..4) |_| {
        const tok_ref = try eng_ref.next(if (eng_ref.out_end == 1) 0 else 3);
        const tok_b = try eng_b.next(if (eng_b.out_end == 1) 0 else 3);
        try std.testing.expectEqual(tok_ref, tok_b);
        try eng_ref.forward(&.{tok_ref}, 0);
        try eng_b.forward(&.{tok_b}, 0);
        try std.testing.expectEqualSlices(f32, try eng_ref.logits(0), try eng_b.logits(0));
    }

    // BF16 cartridge load into smaller context? Check ContextExhausted error path.
    var eng_small = try Engine.init(gpa, &p, ctx, &store, .{ .batch = 4 }, 16);
    defer eng_small.deinit();
    try std.testing.expectError(error.ContextExhausted, load(&eng_small, a, kv_f32_path));

    // BF16 load with relative_stripes + store_pack_bf16 + LSA enabled exercises derived cache rebuild.
    var eng_lsa = try Engine.init(gpa, &p, ctx, &store, .{
        .batch = 8,
        .relative_stripes = true,
        .store_pack_bf16 = true,
        .lsa = true,
        .lsa_budget = 8,
        .lsa_window = 8,
        .lsa_coarse_threshold = 0,
    }, 48);
    defer eng_lsa.deinit();
    try std.testing.expect(eng_lsa.relative_half != null);
    try std.testing.expect(eng_lsa.lsa_layers != null);
    const loaded_bf16 = try load(&eng_lsa, a, kv_bf16_path);
    try std.testing.expectEqual(@as(u32, 20), loaded_bf16.position);
    try eng_lsa.forward(tokens[20..28], 7);
    for (try eng_lsa.logits(7)) |v| try std.testing.expect(std.math.isFinite(v));

    // Also test direct 1-token decode right after cartridge load (uses striped/LSA decode kernels).
    try eng_lsa.forward(&.{1}, 0);
    for (try eng_lsa.logits(0)) |v| try std.testing.expect(std.math.isFinite(v));

    // Corrupt magic fails cleanly.
    const bad_path = try std.fmt.allocPrintSentinel(a, "{s}/bad.nkv", .{dir}, 0);
    defer _ = unlink(bad_path);
    var bad_bytes: [72]u8 = @splat(0);
    @memcpy(bad_bytes[0..8], "NOTAKV01");
    try io.writeFile(a, bad_path, &bad_bytes);
    try std.testing.expectError(error.InvalidCartridgeMagic, load(&eng_b, a, bad_path));
}
