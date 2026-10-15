//! Source-built EXL3 CUDA projections, without a captured Triton or PyTorch kernel kit.
const std = @import("std");
const module = @import("module.zig");
const kernels = @import("kernels.zig");
const launch = @import("launch.zig");
const Stream = @import("stream.zig").Stream;
const Driver = @import("driver.zig").Driver;

pub const Codebook = enum(u32) { inst3, mcg, mul1 };
pub const Params = extern struct {
    k: u32,
    n: u32,
    rows: u32,
    halves: u32,
    codebook: Codebook,
    multiplier: u32,
    su_signs: u32 = 0,
    sv_signs: u32 = 0,
    input_f32: u32 = 0,
    output_f32: u32 = 0,
    tile_k_stride: u64,
    tile_n_stride: u64,

    pub fn validate(p: Params) !void {
        if (p.k == 0 or p.n == 0 or p.k % 128 != 0 or p.n % 128 != 0) return error.UnsupportedExl3Shape;
        if (p.halves < 2 or p.halves > 16 or
            (p.halves % 2 == 1 and (p.codebook != .mul1 or p.halves > 7))) return error.UnsupportedExl3Width;
        if (p.tile_n_stride < p.halves * 16 or p.tile_n_stride % 2 != 0 or
            p.tile_k_stride < p.tile_n_stride * (p.n / 16) or p.tile_k_stride % 2 != 0)
            return error.InvalidExl3Stride;
        if (p.su_signs > 1 or p.sv_signs > 1 or p.input_f32 > 1 or p.output_f32 > 1) return error.Invalid;
        if (p.multiplier == 0) return error.InvalidCodebookMultiplier;
    }

    pub fn scratchBytes(p: Params) !struct { rotated: usize, sums: usize } {
        try p.validate();
        return .{
            .rotated = try std.math.mul(usize, try std.math.mul(usize, p.rows, p.k), 2),
            .sums = try std.math.mul(usize, try std.math.mul(usize, p.rows, p.n), 4),
        };
    }
};

pub const Projection = struct {
    mod: module.Module,
    input: module.Function,
    product: module.Function,
    output: module.Function,

    pub fn load(d: *const Driver) !Projection {
        if (!kernels.available) return error.BuiltWithoutKernels;
        return loadImage(d, kernels.exl3);
    }

    pub fn loadImage(d: *const Driver, image: []const u8) !Projection {
        var mod = try module.Module.load(d, image);
        errdefer mod.unload();
        return .{ .mod = mod, .input = try mod.function("tf_exl3_rotate_in"), .product = try mod.function("tf_exl3_matmul"), .output = try mod.function("tf_exl3_rotate_out") };
    }

    pub fn unload(p: *Projection) void {
        p.mod.unload();
    }

    /// Caller owns disjoint input, output and scratch spans until this stream finishes.
    pub fn run(p: Projection, stream: Stream, args: Params, x: u64, trellis: u64, su: u64, sv: u64, bias: u64, y: u64, rotated: u64, sums: u64) !void {
        try args.validate();
        if (args.rows == 0) return;
        for ([_]u64{ x, trellis, su, sv, y, rotated, sums }) |ptr| if (ptr == 0) return error.Invalid;
        var a: launch.Args = .{};
        a.add(x);
        a.add(su);
        a.add(rotated);
        a.add(args);
        try launch.launch(p.input, .{ .grid = .{ .x = args.k / 128, .y = args.rows }, .block = .{ .x = 128 } }, stream, &a);
        a = .{};
        a.add(rotated);
        a.add(trellis);
        a.add(sums);
        a.add(args);
        try launch.launch(p.product, .{ .grid = .{ .x = (args.n + 127) / 128, .y = args.rows }, .block = .{ .x = 128 } }, stream, &a);
        a = .{};
        a.add(sums);
        a.add(sv);
        a.add(bias);
        a.add(y);
        a.add(args);
        try launch.launch(p.output, .{ .grid = .{ .x = args.n / 128, .y = args.rows }, .block = .{ .x = 128 } }, stream, &a);
    }
};

test "the CUDA EXL3 ABI preserves strides and per-tensor codebooks" {
    try std.testing.expectEqual(@as(usize, 56), @sizeOf(Params));
    try std.testing.expectEqual(@as(usize, 40), @offsetOf(Params, "tile_k_stride"));
    var p: Params = .{ .k = 4096, .n = 2048, .rows = 4, .halves = 8, .codebook = .mcg, .multiplier = 0xCBAC1FED, .tile_k_stride = 128 * 128, .tile_n_stride = 128 };
    try p.validate();
    try std.testing.expectEqual(@as(usize, 32768), (try p.scratchBytes()).rotated);
    p.halves = 7;
    try std.testing.expectError(error.UnsupportedExl3Width, p.validate());
    p.codebook = .mul1;
    try p.validate();
}
