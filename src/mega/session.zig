//! Persistent JSONL session over the megakernel, the `flint session` protocol:
//! request {id, tokens, max_tokens, reset, return_logits}; tokens are only the
//! new suffix; every generated token (EOS included) is committed to state.
const std = @import("std");
const exec = @import("exec.zig");
const dflash2 = @import("dflash2.zig");
const io = @import("io.zig");
const kv_cartridge = @import("kv_cartridge.zig");
const Allocator = std.mem.Allocator;

extern "c" fn read(c_int, [*]u8, usize) isize;

pub const max_line_bytes = 1024 * 1024;

pub const Request = struct {
    id: u32 = 0,
    tokens: []const u32,
    max_tokens: u32 = 128,
    reset: bool = false,
    return_logits: bool = false,
    kv_in: ?[]const u8 = null,
    kv_out: ?[]const u8 = null,
    kv_dtype: ?[]const u8 = null,
};

pub const Response = struct {
    id: u32,
    tokens: []const u32,
    eos: bool,
    cache_before: u32,
    cache_after: u32,
    prefill_tokens: usize,
    decode_forward_tokens: usize,
    reused_prefix_tokens: u32,
    prefill_seconds: f64,
    decode_seconds: f64,
    kv_allocated_bytes: u64,
    kv_valid_bytes: u64,
    kv_storage: []const u8 = "fp32",
    speculative_batches: usize = 0,
    speculative_proposed: usize = 0,
    speculative_accepted: usize = 0,
    prefill_forwards: usize,
    dispatches: u64,
    prefill_logits: ?[]const f32,
    final_logits: ?[]const f32,
};

/// Bounded framing; same rules as src/session.zig's LineBuffer.
pub const LineBuffer = struct {
    storage: []u8,
    len: usize = 0,

    pub fn push(self: *LineBuffer, byte: u8) !?[]const u8 {
        if (byte == '\n') {
            const line = self.storage[0..self.len];
            self.len = 0;
            return line;
        }
        if (self.len >= @min(self.storage.len, max_line_bytes)) return error.LineTooLong;
        self.storage[self.len] = byte;
        self.len += 1;
        return null;
    }
    pub fn finish(self: *LineBuffer) ?[]const u8 {
        if (self.len == 0) return null;
        const line = self.storage[0..self.len];
        self.len = 0;
        return line;
    }
};

fn jsonU32(value: std.json.Value) !u32 {
    return switch (value) {
        .integer => |n| std.math.cast(u32, n) orelse error.InvalidRequestInteger,
        else => error.InvalidRequestInteger,
    };
}
fn jsonBool(value: std.json.Value) !bool {
    return switch (value) {
        .bool => |b| b,
        else => error.InvalidRequestBoolean,
    };
}
fn jsonString(value: std.json.Value) ![]const u8 {
    return switch (value) {
        .string => |s| s,
        else => error.InvalidRequestString,
    };
}

/// Rejects coercions, duplicate and unknown fields, and missing tokens.
pub fn parse(a: Allocator, line: []const u8) !Request {
    if (line.len > max_line_bytes) return error.LineTooLong;
    const value = try std.json.parseFromSliceLeaky(std.json.Value, a, line, .{
        .allocate = .alloc_always,
        .duplicate_field_behavior = .@"error",
        .max_value_len = max_line_bytes,
    });
    const object = switch (value) {
        .object => |object| object,
        else => return error.InvalidRequestObject,
    };
    var fields = object.iterator();
    while (fields.next()) |field| {
        const name = field.key_ptr.*;
        inline for (.{ "id", "tokens", "max_tokens", "reset", "return_logits", "kv_in", "kv_out", "kv_dtype" }) |known| {
            if (std.mem.eql(u8, name, known)) break;
        } else return error.UnknownRequestField;
    }
    const token_value = object.get("tokens") orelse return error.MissingTokens;
    const items = switch (token_value) {
        .array => |array| array.items,
        else => return error.InvalidTokens,
    };
    if (items.len == 0) return error.EmptyPrompt;
    const tokens = try a.alloc(u32, items.len);
    for (items, tokens) |item, *token| token.* = try jsonU32(item);
    return .{
        .id = if (object.get("id")) |v| try jsonU32(v) else 0,
        .tokens = tokens,
        .max_tokens = if (object.get("max_tokens")) |v| try jsonU32(v) else 128,
        .reset = if (object.get("reset")) |v| try jsonBool(v) else false,
        .return_logits = if (object.get("return_logits")) |v| try jsonBool(v) else false,
        .kv_in = if (object.get("kv_in")) |v| try jsonString(v) else null,
        .kv_out = if (object.get("kv_out")) |v| try jsonString(v) else null,
        .kv_dtype = if (object.get("kv_dtype")) |v| try jsonString(v) else null,
    };
}

pub const Preflight = struct { cache_before: u32, required: u64 };

/// Reserve a forward for every generated token, including EOS and the last.
pub fn validate(request: Request, position: u32, context: u32, vocab: u32) !Preflight {
    if (request.tokens.len == 0) return error.EmptyPrompt;
    for (request.tokens) |token| if (token >= vocab) return error.InvalidToken;
    const before = if (request.reset) @as(u32, 0) else position;
    const required = @as(u64, before) + request.tokens.len + request.max_tokens;
    if (required > context) return error.ContextExhausted;
    return .{ .cache_before = before, .required = required };
}

fn finite(logits: []const f32) !void {
    for (logits) |v| if (!std.math.isFinite(v)) return error.NonFiniteLogits;
}

fn handle(engine: *exec.Engine, gpa: Allocator, line: []const u8, draft: ?*dflash2.Draft) !void {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const request = try parse(a, line);
    if (draft != null and request.kv_in != null) return error.DraftWithKvCartridgeUnsupported;
    if (request.kv_in) |kv_path| {
        _ = try kv_cartridge.load(engine, a, kv_path);
    }
    var check_req = request;
    if (request.kv_in != null) check_req.reset = false;
    const preflight = try validate(check_req, engine.position, engine.context, engine.program.meta.vocab);
    if (engine.failed and !check_req.reset and request.kv_in == null) return error.ModelNeedsReset;
    const vocab = engine.program.meta.vocab;
    const generated = try a.alloc(u32, request.max_tokens);
    const prefill_copy: ?[]f32 = if (request.return_logits) try a.alloc(f32, vocab) else null;
    const final_copy: ?[]f32 = if (request.return_logits) try a.alloc(f32, vocab) else null;
    if (request.kv_in == null and request.reset) {
        engine.reset();
        if (draft) |d| d.reset();
    }

    var dispatches: u64 = 0;
    var forwards: usize = 0;
    const prefill_start = try io.now();
    var row: u32 = 0;
    if (draft) |d| {
        const pf = try d.prefill(engine, request.tokens);
        dispatches += pf.dispatches;
        forwards += pf.forwards;
        row = pf.last_row;
    } else {
        var i: usize = 0;
        while (i < request.tokens.len) {
            const n = @min(engine.batch, request.tokens.len - i);
            const last = i + n == request.tokens.len;
            try engine.forward(request.tokens[i..][0..n], if (last) @intCast(n - 1) else @intCast(n));
            dispatches += engine.last.dispatches;
            forwards += 1;
            row = @intCast(n - 1);
            i += n;
        }
    }
    try finite(try engine.logits(row));
    const prefill_seconds = try io.now() - prefill_start;
    if (prefill_copy) |c| @memcpy(c, try engine.logits(row));

    const decode_start = try io.now();
    var count: usize = 0;
    var eos = false;
    var spec_batches: usize = 0;
    var spec_proposed: usize = 0;
    var spec_accepted: usize = 0;
    if (draft) |d| {
        if (request.max_tokens > 0) {
            const first_tok = try engine.next(row);
            const res = try d.decodeGreedy(engine, first_tok, generated, .{ .stop_on_eos = true, .commit_all = true });
            count = res.count;
            eos = res.eos;
            row = res.last_row;
            spec_batches = res.batches;
            spec_proposed = res.proposed;
            spec_accepted = res.accepted;
            dispatches += res.dispatches;
        }
    } else {
        while (count < request.max_tokens) {
            const token = try engine.next(row);
            try engine.forward(&.{token}, 0);
            dispatches += engine.last.dispatches;
            row = 0;
            generated[count] = token;
            count += 1;
            if (std.mem.indexOfScalar(u32, engine.program.meta.eos, token) != null) {
                eos = true;
                break;
            }
        }
    }
    try finite(try engine.logits(row));
    const decode_seconds = try io.now() - decode_start;
    if (final_copy) |c| @memcpy(c, try engine.logits(row));
    if (engine.position != preflight.cache_before + request.tokens.len + count) return error.CachePositionMismatch;
    if (request.kv_out) |out_path| {
        const dtype = if (request.kv_dtype) |ds| kv_cartridge.parseDType(ds) orelse return error.InvalidKvDType else .f32;
        _ = try kv_cartridge.save(engine, a, out_path, dtype, "{}");
    }
    const bytes = engine.stateBytes(engine.position);
    const response: Response = .{
        .id = request.id,
        .tokens = generated[0..count],
        .eos = eos,
        .cache_before = preflight.cache_before,
        .cache_after = engine.position,
        .prefill_tokens = request.tokens.len,
        .decode_forward_tokens = count,
        .reused_prefix_tokens = preflight.cache_before,
        .prefill_seconds = prefill_seconds,
        .decode_seconds = decode_seconds,
        .kv_allocated_bytes = bytes.allocated,
        .kv_valid_bytes = bytes.valid,
        .speculative_batches = spec_batches,
        .speculative_proposed = spec_proposed,
        .speculative_accepted = spec_accepted,
        .prefill_forwards = forwards,
        .dispatches = dispatches,
        .prefill_logits = prefill_copy,
        .final_logits = final_copy,
    };
    const json = try std.json.Stringify.valueAlloc(a, response, .{});
    try io.out(json);
    try io.out("\n");
}

pub fn run(engine: *exec.Engine, gpa: Allocator) !void {
    return runWithDraft(engine, gpa, null);
}

pub fn runWithDraft(engine: *exec.Engine, gpa: Allocator, draft: ?*dflash2.Draft) !void {
    return runImpl(engine, gpa, draft) catch |err| {
        std.debug.print("Session failed: {s}\n", .{@errorName(err)});
        return err;
    };
}

fn runImpl(engine: *exec.Engine, gpa: Allocator, draft: ?*dflash2.Draft) !void {
    const storage = try gpa.alloc(u8, max_line_bytes);
    defer gpa.free(storage);
    var lines: LineBuffer = .{ .storage = storage };
    var chunk: [8192]u8 = undefined;
    while (true) {
        const n = read(0, &chunk, chunk.len);
        if (n < 0) return error.InputFailed;
        if (n == 0) {
            if (lines.finish()) |line| try handle(engine, gpa, line, draft);
            return;
        }
        for (chunk[0..@intCast(n)]) |byte| {
            if (try lines.push(byte)) |line| try handle(engine, gpa, line, draft);
        }
    }
}

test "session request parsing matches the Flint protocol" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const r = try parse(a, "{\"tokens\":[2,3]}");
    try std.testing.expectEqual(@as(u32, 128), r.max_tokens);
    try std.testing.expect(!r.reset and !r.return_logits);
    try std.testing.expectError(error.UnknownRequestField, parse(a, "{\"tokens\":[1],\"typo\":0}"));
    try std.testing.expectError(error.InvalidRequestInteger, parse(a, "{\"tokens\":[1.0]}"));
    try std.testing.expectError(error.EmptyPrompt, parse(a, "{\"tokens\":[]}"));
    try std.testing.expectError(error.DuplicateField, parse(a, "{\"tokens\":[1],\"tokens\":[2]}"));
}

test "session preflight reserves every generated token and checks tokens first" {
    const req: Request = .{ .tokens = &.{1}, .max_tokens = 2 };
    try std.testing.expectEqual(@as(u64, 10), (try validate(req, 7, 10, 4)).required);
    try std.testing.expectError(error.ContextExhausted, validate(req, 8, 10, 4));
    try std.testing.expectError(error.InvalidToken, validate(.{ .tokens = &.{ 1, 4 }, .reset = true }, 9, 100, 4));
    try std.testing.expectEqual(@as(u32, 0), (try validate(.{ .tokens = &.{1}, .max_tokens = 0, .reset = true }, 10, 10, 4)).cache_before);
}

test "session framing keeps a final unterminated line" {
    var storage: [16]u8 = undefined;
    var lines: LineBuffer = .{ .storage = &storage };
    for ("one\ntwo") |byte| if (try lines.push(byte)) |line| try std.testing.expectEqualStrings("one", line);
    try std.testing.expectEqualStrings("two", lines.finish().?);
    try std.testing.expect(lines.finish() == null);
}
