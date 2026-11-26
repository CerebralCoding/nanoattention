//! C ABI for embedding `nano` (LFM / Inkling / megakernel) in iOS and macOS apps.
const std = @import("std");
const builtin = @import("builtin");
const exec = @import("exec.zig");
const loader = @import("loader.zig");
const frontend = @import("frontend.zig");
const chat = @import("chat.zig");
const tokenizer = @import("tokenizer.zig");
const io = @import("io.zig");
const Knobs = @import("knobs.zig").Knobs;
const metal = @import("weights").metal;

pub const std_options_debug_threaded_io: ?*std.Io.Threaded = if (builtin.os.tag == .ios) null else std.Io.Threaded.global_single_threaded;
pub const std_options_debug_io: std.Io = if (builtin.os.tag == .ios) io.ios_debug_io else std.Io.Threaded.global_single_threaded.io();

const version_str: [*:0]const u8 = "0.2.0";
const lfm_ios_preset = @embedFile("presets/lfm-ios.json");
const exact_preset = @embedFile("presets/exact.json");
const mlx_preset = @embedFile("presets/mlx-prefill.json");

threadlocal var last_error_buf: [2048]u8 = @splat(0);

fn clearError() void {
    last_error_buf[0] = 0;
}

fn setError(msg: []const u8) void {
    const n = @min(msg.len, last_error_buf.len - 1);
    @memcpy(last_error_buf[0..n], msg[0..n]);
    last_error_buf[n] = 0;
}

fn setErrorFmt(comptime fmt: []const u8, args: anytype) void {
    const s = std.fmt.bufPrint(last_error_buf[0 .. last_error_buf.len - 1], fmt, args) catch last_error_buf[0 .. last_error_buf.len - 1];
    last_error_buf[s.len] = 0;
}

pub const NanoConfig = extern struct {
    context: u32 = 0,
    batch: u32 = 0,
    preset: ?[*:0]const u8 = null,
    use_tokenizer: i32 = 1,
};

pub const NanoStats = extern struct {
    prompt_tokens: u32 = 0,
    generated_tokens: u32 = 0,
    position: u32 = 0,
    load_ms: f64 = 0,
    prefill_ms: f64 = 0,
    decode_ms: f64 = 0,
    dispatches: u64 = 0,
};

pub const NanoTokenCallback = ?*const fn (u32, ?[*]const u8, usize, ?*anyopaque) callconv(.c) c_int;

pub const NanoSession = struct {
    gpa: std.mem.Allocator,
    load_arena: *std.heap.ArenaAllocator,
    loaded: *loader.Loaded,
    tok: ?tokenizer.Tokenizer,
    template: ?chat.Template,
    model_type: [:0]u8,
    device_name: [:0]u8,
    load_ms: f64,
};

pub export fn nano_version() callconv(.c) [*:0]const u8 {
    return version_str;
}

pub export fn nano_last_error() callconv(.c) [*:0]const u8 {
    return @ptrCast(&last_error_buf);
}

fn selectKnobs(a: std.mem.Allocator, config: ?*const NanoConfig) !Knobs {
    const default_preset = if (builtin.os.tag == .ios) "lfm-ios" else "exact";
    const preset_name = if (config) |c| (if (c.preset) |p| std.mem.span(p) else default_preset) else default_preset;
    const json_bytes: []const u8 = if (std.mem.eql(u8, preset_name, "lfm-ios"))
        lfm_ios_preset
    else if (std.mem.eql(u8, preset_name, "exact"))
        exact_preset
    else if (std.mem.eql(u8, preset_name, "mlx-prefill"))
        mlx_preset
    else if (preset_name.len > 0 and preset_name[0] == '{')
        preset_name
    else {
        setErrorFmt("unknown preset \"{s}\"", .{preset_name});
        return error.InvalidPreset;
    };
    var knobs = try Knobs.parseJson(a, json_bytes);
    if (config) |c| {
        if (c.batch != 0) knobs.batch = c.batch;
    }
    try knobs.validate();
    return knobs;
}

pub export fn nano_open(checkpoint_dir: ?[*:0]const u8, config: ?*const NanoConfig) callconv(.c) ?*NanoSession {
    clearError();
    const dir_ptr = checkpoint_dir orelse {
        setError("checkpoint_dir is null");
        return null;
    };
    const dir = std.mem.span(dir_ptr);
    if (dir.len == 0) {
        setError("checkpoint_dir is empty");
        return null;
    }

    const gpa = std.heap.c_allocator;
    const load_arena = gpa.create(std.heap.ArenaAllocator) catch {
        setError("out of memory allocating load_arena");
        return null;
    };
    load_arena.* = std.heap.ArenaAllocator.init(gpa);
    errdefer {
        load_arena.deinit();
        gpa.destroy(load_arena);
    }
    const a = load_arena.allocator();

    const default_ctx: u32 = if (builtin.os.tag == .ios) 2048 else 4096;
    const context: u32 = if (config) |c| (if (c.context != 0) c.context else default_ctx) else default_ctx;
    const use_tokenizer: bool = if (config) |c| c.use_tokenizer != 0 else true;

    const knobs = selectKnobs(a, config) catch |err| {
        if (last_error_buf[0] == 0) setErrorFmt("knobs error: {s}", .{@errorName(err)});
        return null;
    };

    const t0 = io.now() catch 0;
    const loaded = loader.load(gpa, a, dir, knobs, context, .{}) catch |err| {
        if (err == error.MetalFailure) {
            setErrorFmt("cannot load checkpoint \"{s}\": MetalFailure ({s})", .{ dir, metal.lastError() });
        } else {
            setErrorFmt("cannot load checkpoint \"{s}\": {s}", .{ dir, @errorName(err) });
        }
        return null;
    };
    errdefer {
        loaded.deinit();
        gpa.destroy(loaded);
    }
    const t1 = io.now() catch t0;

    var tok: ?tokenizer.Tokenizer = null;
    errdefer if (tok) |*t| t.deinit();
    if (use_tokenizer) {
        const tok_path = std.fs.path.join(a, &.{ dir, "tokenizer.json" }) catch {
            setError("out of memory joining tokenizer path");
            return null;
        };
        const map = metal.Mapping.open(a, tok_path) catch |err| {
            setErrorFmt("cannot read tokenizer \"{s}\": {s}", .{ tok_path, @errorName(err) });
            return null;
        };
        defer map.deinit();
        var diag: tokenizer.Diagnostic = .{};
        tok = tokenizer.Tokenizer.init(gpa, map.bytes, &diag) catch |err| {
            if (err == error.UnsupportedTokenizer) {
                setError(diag.message());
            } else {
                setErrorFmt("cannot initialize tokenizer: {s}", .{@errorName(err)});
            }
            return null;
        };
    }

    const mt = gpa.dupeSentinel(u8, loaded.program.meta.model_type, 0) catch {
        setError("out of memory duplicating model_type");
        return null;
    };
    errdefer gpa.free(mt);

    const dev = gpa.dupeSentinel(u8, loaded.ctx.deviceName(), 0) catch {
        setError("out of memory duplicating device_name");
        return null;
    };
    errdefer gpa.free(dev);

    const session = gpa.create(NanoSession) catch {
        setError("out of memory creating NanoSession");
        return null;
    };
    session.* = .{
        .gpa = gpa,
        .load_arena = load_arena,
        .loaded = loaded,
        .tok = tok,
        .template = frontend.chatTemplate(loaded.program.meta.model_type),
        .model_type = mt,
        .device_name = dev,
        .load_ms = (t1 - t0) * 1000.0,
    };
    return session;
}

pub export fn nano_close(session: ?*NanoSession) callconv(.c) void {
    const s = session orelse return;
    const gpa = s.gpa;
    const load_arena = s.load_arena;
    if (s.tok) |*t| t.deinit();
    gpa.free(s.model_type);
    gpa.free(s.device_name);
    s.loaded.deinit();
    gpa.destroy(s.loaded);
    load_arena.deinit();
    gpa.destroy(load_arena);
    gpa.destroy(s);
}

pub export fn nano_device_name(session: ?*const NanoSession) callconv(.c) [*:0]const u8 {
    const s = session orelse return "none";
    return s.device_name.ptr;
}

pub export fn nano_model_type(session: ?*const NanoSession) callconv(.c) [*:0]const u8 {
    const s = session orelse return "none";
    return s.model_type.ptr;
}

pub export fn nano_reset(session: ?*NanoSession) callconv(.c) c_int {
    clearError();
    const s = session orelse {
        setError("session is null");
        return -1;
    };
    s.loaded.engine.reset();
    return 0;
}

fn isEos(s: *const NanoSession, id: u32) bool {
    return std.mem.indexOfScalar(u32, s.loaded.program.meta.eos, id) != null;
}

pub export fn nano_generate_tokens(
    session: ?*NanoSession,
    prompt_tokens: ?[*]const u32,
    prompt_len: usize,
    max_tokens: u32,
    cb: NanoTokenCallback,
    user_data: ?*anyopaque,
    out_stats: ?*NanoStats,
) callconv(.c) c_int {
    clearError();
    if (out_stats) |st| st.* = .{};
    const s = session orelse {
        setError("session is null");
        return -1;
    };
    const tok_ptr = prompt_tokens orelse {
        setError("prompt_tokens is null");
        return -1;
    };
    if (prompt_len == 0) {
        setError("prompt_len must be > 0");
        return -1;
    }

    const prompt = tok_ptr[0..prompt_len];
    const engine = &s.loaded.engine;
    const disp_before = s.loaded.ctx.stats().dispatches;
    const t_start = io.now() catch 0;

    var i: usize = 0;
    var row: u32 = 0;
    while (i < prompt.len) {
        const n = @min(engine.batch, prompt.len - i);
        const last = i + n == prompt.len;
        const out_first: u32 = if (last and max_tokens > 0) @intCast(n - 1) else @intCast(n);
        engine.forward(prompt[i..][0..n], out_first) catch |err| {
            setErrorFmt("prefill failed: {s}", .{@errorName(err)});
            return -1;
        };
        row = @intCast(n - 1);
        i += n;
    }
    const t_prefill = io.now() catch t_start;

    var stream: tokenizer.Stream = .{};
    defer stream.deinit(s.gpa);
    var piece: std.ArrayList(u8) = .empty;
    defer piece.deinit(s.gpa);

    var generated: u32 = 0;
    var t_first = t_prefill;
    var t_last = t_prefill;

    while (generated < max_tokens) {
        const id = engine.next(row) catch |err| {
            setErrorFmt("decode next failed: {s}", .{@errorName(err)});
            return -1;
        };
        const t_now = io.now() catch t_last;
        if (generated == 0) t_first = t_now;
        t_last = t_now;
        generated += 1;

        const stop = isEos(s, id);
        var user_stop = false;
        piece.clearRetainingCapacity();
        if (!stop) {
            if (s.tok) |*tok| {
                stream.push(s.gpa, tok, id, true, &piece) catch |err| {
                    setErrorFmt("tokenizer stream failed: {s}", .{@errorName(err)});
                    return -1;
                };
            }
        } else {
            stream.finish(s.gpa, &piece) catch |err| {
                setErrorFmt("tokenizer flush failed: {s}", .{@errorName(err)});
                return -1;
            };
        }

        if (cb) |callback| {
            const p_ptr: [*]const u8 = if (piece.items.len > 0) piece.items.ptr else "".ptr;
            if (callback(id, p_ptr, piece.items.len, user_data) != 0) user_stop = true;
        }

        const done = stop or user_stop or generated == max_tokens;
        if (engine.position >= engine.context) break;
        engine.forward(&.{id}, 0) catch |err| {
            setErrorFmt("decode forward failed: {s}", .{@errorName(err)});
            return -1;
        };
        row = 0;
        if (done) break;
    }

    if (generated == max_tokens) {
        piece.clearRetainingCapacity();
        stream.finish(s.gpa, &piece) catch {};
        if (piece.items.len > 0) {
            if (cb) |callback| {
                _ = callback(0, piece.items.ptr, piece.items.len, user_data);
            }
        }
    }

    if (out_stats) |st| {
        const disp_after = s.loaded.ctx.stats().dispatches;
        st.* = .{
            .prompt_tokens = @intCast(prompt_len),
            .generated_tokens = generated,
            .position = engine.position,
            .load_ms = s.load_ms,
            .prefill_ms = (t_prefill - t_start) * 1000.0,
            .decode_ms = if (generated > 1) (t_last - t_first) * 1000.0 else 0.0,
            .dispatches = disp_after - disp_before,
        };
    }
    return 0;
}

pub export fn nano_generate_text(
    session: ?*NanoSession,
    system_prompt: ?[*:0]const u8,
    user_prompt: ?[*:0]const u8,
    raw_prompt: i32,
    max_tokens: u32,
    cb: NanoTokenCallback,
    user_data: ?*anyopaque,
    out_stats: ?*NanoStats,
) callconv(.c) c_int {
    clearError();
    if (out_stats) |st| st.* = .{};
    const s = session orelse {
        setError("session is null");
        return -1;
    };
    const tok = if (s.tok) |*t| t else {
        setError("session initialized with use_tokenizer=0");
        return -1;
    };
    const user_ptr = user_prompt orelse {
        setError("user_prompt is null");
        return -1;
    };
    const user_slice = std.mem.span(user_ptr);

    var arena = std.heap.ArenaAllocator.init(s.gpa);
    defer arena.deinit();
    const a = arena.allocator();

    var prompt_text: []const u8 = user_slice;
    if (raw_prompt == 0) {
        const tmpl = s.template orelse {
            setErrorFmt("no native chat template for model_type \"{s}\"; pass raw_prompt=1", .{s.model_type});
            return -1;
        };
        var msgs: std.ArrayList(chat.Message) = .empty;
        if (system_prompt) |sp| {
            const sys_slice = std.mem.span(sp);
            if (sys_slice.len > 0) {
                var clean_sys: std.ArrayList(u8) = .empty;
                _ = tokenizer.appendLossy(a, &clean_sys, sys_slice, true) catch {
                    setError("out of memory sanitizing system_prompt");
                    return -1;
                };
                msgs.append(a, .{ .role = .system, .content = clean_sys.items }) catch {
                    setError("out of memory building messages");
                    return -1;
                };
            }
        }
        var clean_user: std.ArrayList(u8) = .empty;
        _ = tokenizer.appendLossy(a, &clean_user, user_slice, true) catch {
            setError("out of memory sanitizing user_prompt");
            return -1;
        };
        msgs.append(a, .{ .role = .user, .content = clean_user.items }) catch {
            setError("out of memory building messages");
            return -1;
        };
        prompt_text = tmpl.render(a, msgs.items, .{ .thinking = false, .generate = true }) catch |err| {
            setErrorFmt("chat template render failed: {s}", .{@errorName(err)});
            return -1;
        };
    }

    var ids: std.ArrayList(u32) = .empty;
    tok.encode(a, prompt_text, &ids) catch |err| {
        setErrorFmt("tokenize failed: {s}", .{@errorName(err)});
        return -1;
    };
    if (ids.items.len == 0) {
        setError("tokenized prompt is empty");
        return -1;
    }
    return nano_generate_tokens(s, ids.items.ptr, ids.items.len, max_tokens, cb, user_data, out_stats);
}
