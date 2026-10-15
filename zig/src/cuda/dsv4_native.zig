//! Source-built reference kernels for original FP4/E8M0 and FP8/E8M0 storage.
const std = @import("std");
const module = @import("module.zig");
const kernels = @import("kernels.zig");
const launch = @import("launch.zig");
const Driver = @import("driver.zig").Driver;
const Stream = @import("stream.zig").Stream;

pub const Ops = struct {
    mod: module.Module,
    quant: module.Function,
    matmul: module.Function,
    dequant: module.Function,

    pub fn load(d: *const Driver) !Ops {
        if (!kernels.available) return error.BuiltWithoutKernels;
        return loadImage(d, kernels.dsv4_native);
    }

    pub fn loadImage(d: *const Driver, image: []const u8) !Ops {
        var mod = try module.Module.load(d, image);
        errdefer mod.unload();
        return .{ .mod = mod, .quant = try mod.function("tf_dsv4_native_quant"), .matmul = try mod.function("tf_dsv4_native_matmul"), .dequant = try mod.function("tf_dsv4_native_dequant") };
    }

    pub fn unload(o: *Ops) void {
        o.mod.unload();
    }

    pub fn scratchBytes(rows: u32, k: u32) !struct { quant: usize, scales: usize } {
        if (k == 0 or k % 128 != 0) return error.Invalid;
        const count = try std.math.mul(usize, rows, k);
        return .{ .quant = count, .scales = count / 128 };
    }

    pub fn run(o: Ops, stream: Stream, x: u64, w: u64, scales: u64, bias: u64, y: u64, q: u64, qs: u64, rows: u32, k: u32, n: u32, fp4: bool, input_f32: bool, output_f32: bool) !void {
        _ = try scratchBytes(rows, k);
        if (n == 0) return error.Invalid;
        if (rows == 0) return;
        var qa: launch.Args = .{};
        inline for (.{ x, q, qs, k, @as(u32, @intFromBool(input_f32)) }) |v| qa.add(v);
        try launch.launch(o.quant, .{ .grid = .{ .x = k / 128, .y = rows }, .block = .{ .x = 128 } }, stream, &qa);
        var ma: launch.Args = .{};
        inline for (.{ q, qs, w, scales, bias, y, k, n, @as(u32, @intFromBool(fp4)), @as(u32, @intFromBool(output_f32)) }) |v| ma.add(v);
        try launch.launch(o.matmul, .{ .grid = .{ .x = (n + 127) / 128, .y = rows }, .block = .{ .x = 128 } }, stream, &ma);
    }

    pub fn toBf16(o: Ops, s: Stream, w: u64, scales: u64, y: u64, k: u32, n: u32) !void {
        _ = try scratchBytes(1, k);
        if (n == 0) return error.Invalid;
        const count = try std.math.mul(u32, k, n);
        var args: launch.Args = .{};
        inline for (.{ w, scales, y, k, n }) |v| args.add(v);
        try launch.launch(o.dequant, .{ .grid = .{ .x = (count + 255) / 256 }, .block = .{ .x = 256 } }, s, &args);
    }
};

test "native scratch retains FP8 activations and one byte per scale" {
    const size = try Ops.scratchBytes(4, 4096);
    try std.testing.expectEqual(@as(usize, 16384), size.quant);
    try std.testing.expectEqual(@as(usize, 128), size.scales);
    try std.testing.expectError(error.Invalid, Ops.scratchBytes(1, 65));
}
