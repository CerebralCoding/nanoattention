//! Minimal libc I/O: stdout, whole-file writes, a monotonic clock.
const std = @import("std");

extern "c" fn write(c_int, [*]const u8, usize) isize;
extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
extern "c" fn close(c_int) c_int;
extern "c" fn rename([*:0]const u8, [*:0]const u8) c_int;
const Timespec = extern struct { sec: i64, nsec: i64 };
extern "c" fn clock_gettime(c_int, *Timespec) c_int;

pub fn writeAll(fd: c_int, bytes: []const u8) !void {
    var offset: usize = 0;
    while (offset < bytes.len) {
        // Darwin write(2) rejects single transfers >= 2 GiB with EINVAL.
        const chunk = @min(bytes.len - offset, 128 * 1024 * 1024);
        const n = write(fd, bytes[offset..].ptr, chunk);
        if (n <= 0) return error.OutputFailed;
        offset += @intCast(n);
    }
}
pub fn out(bytes: []const u8) !void {
    return writeAll(1, bytes);
}
pub fn print(a: std.mem.Allocator, comptime fmt: []const u8, args: anytype) !void {
    return out(try std.fmt.allocPrint(a, fmt, args));
}

/// Write via a temporary file and rename, so readers never see a partial file.
pub fn writeFile(a: std.mem.Allocator, path: []const u8, bytes: []const u8) !void {
    const tmp = try std.fmt.allocPrintSentinel(a, "{s}.tmp", .{path}, 0);
    const final = try a.dupeSentinel(u8, path, 0);
    // Darwin: O_WRONLY | O_CREAT | O_TRUNC.
    const fd = open(tmp, 0x1 | 0x200 | 0x400, @as(c_uint, 0o644));
    if (fd < 0) return error.CannotCreateFile;
    writeAll(fd, bytes) catch |err| {
        _ = close(fd);
        return err;
    };
    if (close(fd) != 0) return error.OutputFailed;
    if (rename(tmp, final) != 0) return error.CannotRenameFile;
}

/// Darwin CLOCK_MONOTONIC seconds, as the Flint session and benchmarks use.
pub fn now() !f64 {
    var ts: Timespec = undefined;
    if (clock_gettime(6, &ts) != 0) return error.ClockFailed;
    return @as(f64, @floatFromInt(ts.sec)) + @as(f64, @floatFromInt(ts.nsec)) * 1e-9;
}

// Zig 0.17 std.Io.Threaded fails to compile spawnDarwin on iOS (NullFile lacks .fd).
// Provide a minimal std.Io for std.debug on iOS that writes stderr directly via fd 2.
var ios_stderr_lock: std.atomic.Value(bool) = .init(false);
var ios_stderr_writer: std.Io.File.Writer = .{
    .io = std.Io.failing,
    .file = .{ .handle = 2, .flags = .{ .nonblocking = false } },
    .interface = .{
        .vtable = &.{ .drain = iosStderrDrain },
        .buffer = &.{},
    },
};

fn iosStderrDrain(w: *std.Io.Writer, data: []const []const u8, splat: usize) std.Io.Writer.Error!usize {
    if (w.end > 0) {
        writeAll(2, w.buffer[0..w.end]) catch {};
        w.end = 0;
    }
    var consumed: usize = 0;
    for (data, 0..) |slice, idx| {
        const reps: usize = if (idx + 1 == data.len) splat else 1;
        for (0..reps) |_| {
            if (slice.len > 0) writeAll(2, slice) catch {};
            consumed += slice.len;
        }
    }
    return consumed;
}

fn iosSwapCancelProtection(_: ?*anyopaque, _: std.Io.CancelProtection) std.Io.CancelProtection {
    return .blocked;
}

fn iosLockStderr(_: ?*anyopaque, _: ?std.Io.Terminal.Mode) std.Io.Cancelable!std.Io.LockedStderr {
    while (ios_stderr_lock.cmpxchgWeak(false, true, .acquire, .monotonic) != null) {}
    ios_stderr_writer.interface.buffer = &.{};
    ios_stderr_writer.interface.end = 0;
    return .{ .file_writer = &ios_stderr_writer, .terminal_mode = .no_color };
}

fn iosTryLockStderr(_: ?*anyopaque, _: ?std.Io.Terminal.Mode) std.Io.Cancelable!?std.Io.LockedStderr {
    if (ios_stderr_lock.cmpxchgWeak(false, true, .acquire, .monotonic) != null) return null;
    ios_stderr_writer.interface.buffer = &.{};
    ios_stderr_writer.interface.end = 0;
    return .{ .file_writer = &ios_stderr_writer, .terminal_mode = .no_color };
}

fn iosUnlockStderr(_: ?*anyopaque) void {
    ios_stderr_lock.store(false, .release);
}

pub const ios_debug_io: std.Io = blk: {
    var vt = std.Io.failing.vtable.*;
    vt.swapCancelProtection = iosSwapCancelProtection;
    vt.lockStderr = iosLockStderr;
    vt.tryLockStderr = iosTryLockStderr;
    vt.unlockStderr = iosUnlockStderr;
    const frozen = vt;
    break :blk .{ .userdata = null, .vtable = &frozen };
};
