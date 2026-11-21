//! Decode attention: `nano attnbench [repeats] [keys...] [rows=N]`. GPU time of the
//! single-row relative attention of one Inkling layer (24 heads, 8 KV heads,
//! head size 128, relative extent 1024) over synthetic bfloat16-valued K and V:
//! a global layer at each key count, and the 1,024-key sliding window in a
//! wrapped ring when the count is 1024. Variants: the reference kernel, the
//! split the tuned knobs use today and the stripes at several geometries; then
//! a prefill chunk of N rows (128 unless rows=N) ending at the last key, the tiled kernel against its
//! tiles in stripes of keys loaded straight from the mirrors and against the
//! staged tiles off the mirrors (inkling_tiles.metal). Each variant's output is compared with the
//! reference kernel's. Timing guidance only; run it under the inference lock
//! and confirm any gain with `nano bench`.
const std = @import("std");
const gpu = @import("gpu.zig");
const io = @import("io.zig");

const heads = 24;
const kv_heads = 8;
const dim = 128;
const extent = 1024;
const width = kv_heads * dim;

fn bf16(v: f32) f32 {
    const bits: u32 = @bitCast(v);
    return @bitCast((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000);
}

pub fn run(a: std.mem.Allocator, args: []const []const u8) !void {
    const repeats: u32 = if (args.len > 0) try std.fmt.parseInt(u32, args[0], 10) else 20;
    var lengths: std.ArrayList(u32) = .empty;
    var batch: u32 = 128; // rows of the prefill chunk; decode uses row 0
    for (args[@min(args.len, 1)..]) |v| {
        if (std.mem.startsWith(u8, v, "rows=")) {
            batch = try std.fmt.parseInt(u32, v[5..], 10);
            if (batch < 64 or batch % 32 != 0 or batch > 4096) return error.InvalidRows;
        } else try lengths.append(a, try std.fmt.parseInt(u32, v, 10));
    }
    if (lengths.items.len == 0) try lengths.appendSlice(a, &.{ 4096, 8192, 16384, 32768 });
    const ctx = try gpu.open();
    defer ctx.deinit();
    var prng = std.Random.DefaultPrng.init(7);
    const r = prng.random();

    const q_off: u32 = 0;
    const bias_off: u32 = q_off + batch * heads * dim;
    const sink_off: u32 = bias_off + batch * heads * extent;
    const out_off: u32 = sink_off + batch * heads;
    const arena = try ctx.alloc(out_off + 2 * batch * heads * dim);
    defer arena.deinit();
    const f = arena.floats();
    for (f[q_off..][0 .. batch * heads * dim]) |*v| v.* = bf16(r.floatNorm(f32));
    for (f[bias_off..][0 .. batch * heads * extent]) |*v| v.* = bf16(0.5 * r.floatNorm(f32));
    for (f[sink_off..][0 .. batch * heads]) |*v| v.* = bf16(r.floatNorm(f32));

    for (lengths.items) |keys| {
        const window: u32 = if (keys == 1024) 1024 else 0;
        const capacity: u32 = if (window != 0) window - 1 + batch else keys;
        const k32 = try ctx.alloc(@as(usize, capacity) * width);
        defer k32.deinit();
        const v32 = try ctx.alloc(@as(usize, capacity) * width);
        defer v32.deinit();
        // The mirrors repeat their first 16 rows after their last (mk_pack_bf16).
        const k16 = try ctx.alloc(@as(usize, capacity + 16) * width / 2);
        defer k16.deinit();
        const v16 = try ctx.alloc(@as(usize, capacity + 16) * width / 2);
        defer v16.deinit();
        for ([_][2]@TypeOf(k32){ .{ k32, k16 }, .{ v32, v16 } }) |pair| {
            const half = std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(pair[1].floats()));
            for (pair[0].floats(), half[0 .. @as(usize, capacity) * width]) |*v, *h| {
                v.* = bf16(r.floatNorm(f32));
                h.* = @intCast(@as(u32, @bitCast(v.*)) >> 16);
            }
            @memcpy(half[@as(usize, capacity) * width ..][0 .. 16 * width], half[0 .. 16 * width]);
        }
        var p: gpu.Params = .{ .rows = 1, .pos = if (window != 0) 4 * batch + 904 else keys - 1, .x = q_off, .xs = heads * dim, .y = out_off, .ys = heads * dim, .z = bias_off, .zs = heads * extent, .s = sink_off, .every = heads, .heads = heads, .kv_heads = kv_heads, .dim = dim, .m = extent, .window = window, .capacity = capacity, .scale = 1 / @sqrt(@as(f32, dim)) };
        try gpu.dispatch(ctx, "mk_relative_attention", &.{ arena, k32, v32 }, &p, heads * 32, 32);
        try ctx.finish();
        const reference = try a.dupe(f32, f[out_off..][0 .. heads * dim]);
        p.y = out_off + batch * heads * dim;
        const out = f[p.y..][0 .. heads * dim];
        const parts = try ctx.alloc(@as(usize, heads) * 1024 * 130);
        defer parts.deinit();
        try io.print(a, "{d} keys, window {d} ({d:.0} MB of float K and V)\n", .{ keys, window, @as(f64, @floatFromInt(keys)) * width * 8 / 1e6 });

        const Timer = struct {
            fn report(al: std.mem.Allocator, name: []const u8, us: f64, expected: []const f32, actual: []const f32) !void {
                var worst: f32 = 0;
                for (expected, actual) |e, x| worst = @max(worst, @abs(e - x));
                try io.print(al, "  {s:<34} {d:>9.1} us   max |diff| {e:.2}\n", .{ name, us, worst });
            }
        };
        {
            p.taps = @min(64, @max(4, (keys + 31) / 32));
            const wide = keys <= p.taps * 32;
            const group: usize = if (wide) 128 else 32;
            @memset(out, 0);
            const before = ctx.stats();
            for (0..repeats) |_| {
                try gpu.dispatch(ctx, if (wide) "mk_relative_split_keys4" else "mk_relative_split_keys", &.{ arena, k32, v32, parts }, &p, heads * p.taps * group, group);
                try gpu.dispatch(ctx, "mk_relative_split_merge", &.{ arena, parts }, &p, heads * 32, 32);
            }
            try ctx.finish();
            const us = (ctx.stats().gpu_seconds - before.gpu_seconds) / @as(f64, @floatFromInt(repeats)) * 1e6;
            try Timer.report(a, if (wide) "split, 4 dots a part (current)" else "split 64 parts (current)", us, reference, out);
        }
        for ([_]u32{ 32, 64, 128, 256, 512 }) |stripe| for ([_]u32{ 4, 8, 16 }) |simds| {
            p.taps = (keys + stripe - 1) / stripe;
            if (p.taps > 1024 or p.taps < 2) continue;
            @memset(out, 0);
            var us: [2]f64 = undefined;
            for (0..2) |with_merge| {
                const before = ctx.stats();
                for (0..repeats) |_| {
                    try gpu.dispatch(ctx, "mk_relative_long_keys", &.{ arena, k16, v16, parts }, &p, @as(usize, kv_heads) * p.taps * 32 * simds, 32 * simds);
                    if (with_merge == 1) try gpu.dispatch(ctx, "mk_relative_long_merge", &.{ arena, parts }, &p, heads * 128, 128);
                }
                try ctx.finish();
                us[with_merge] = (ctx.stats().gpu_seconds - before.gpu_seconds) / @as(f64, @floatFromInt(repeats)) * 1e6;
            }
            const name = try std.fmt.allocPrint(a, "stripes of {d}, {d} simd (keys {d:.0})", .{ stripe, simds, us[0] });
            try Timer.report(a, name, us[1], reference, out);
        };
        // A prefill chunk: the last `batch` positions.
        if (p.pos + 1 >= batch) {
            p.rows = batch;
            p.pos -= batch - 1;
            const chunk = batch * heads * dim;
            p.y = out_off;
            var tiled: [3]f64 = undefined;
            for ([_]u32{ 32, 64, batch }, &tiled) |rows, *us| {
                p.rows = rows;
                const before = ctx.stats();
                for (0..repeats) |_| try gpu.dispatch(ctx, "mk_relative_attention_prefill", &.{ arena, k32, v32 }, &p, @as(usize, rows / 32) * heads * 128, 128);
                try ctx.finish();
                us.* = (ctx.stats().gpu_seconds - before.gpu_seconds) / @as(f64, @floatFromInt(repeats)) * 1e6;
            }
            try io.print(a, "  prefill tiled (current): 32 rows {d:.1} us, 64 rows {d:.1} us, {d} rows {d:.1} us\n", .{ tiled[0], tiled[1], batch, tiled[2] });
            const want = f[out_off..][0..chunk];
            p.y = out_off + chunk;
            const got = f[p.y..][0..chunk];
            for ([_]u32{ 1, 2, 4, 8 }) |taps| {
                p.taps = taps;
                const records = try ctx.alloc(@as(usize, batch) * heads * taps * 130);
                defer records.deinit();
                @memset(got, 0);
                const before = ctx.stats();
                for (0..repeats) |_| {
                    try gpu.dispatch(ctx, "mk_relative_prefill_stripes", &.{ arena, k16, v16, records }, &p, @as(usize, batch / 32) * heads * taps * 128, 128);
                    try gpu.dispatch(ctx, "mk_relative_prefill_merge", &.{ arena, records }, &p, chunk, 128);
                }
                try ctx.finish();
                const us = (ctx.stats().gpu_seconds - before.gpu_seconds) / @as(f64, @floatFromInt(repeats)) * 1e6;
                const name = try std.fmt.allocPrint(a, "prefill, {d} stripes", .{taps});
                try Timer.report(a, name, us, want, got);
            }
            {
                // The reference's staged 16-key tiles off the mirrors (inkling_tiles.metal).
                @memset(got, 0);
                const before = ctx.stats();
                for (0..repeats) |_| try gpu.dispatch(ctx, "mk_relative_prefill_tiles", &.{ arena, k16, v16 }, &p, @as(usize, batch / 32) * heads * 128, 128);
                try ctx.finish();
                const us = (ctx.stats().gpu_seconds - before.gpu_seconds) / @as(f64, @floatFromInt(repeats)) * 1e6;
                try Timer.report(a, "prefill, staged tiles off mirrors", us, want, got);
            }
        }
    }
}
