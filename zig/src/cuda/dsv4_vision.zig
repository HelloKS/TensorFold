//! Explicit launches for native ViT attention, normalization and fused-projection unpacking.
const std = @import("std");
const module = @import("module.zig");
const kernels = @import("kernels.zig");
const launch = @import("launch.zig");
const Driver = @import("driver.zig").Driver;
const Stream = @import("stream.zig").Stream;
const Kind = @import("dsv4_tensor.zig").Kind;

pub const Op = enum { norm, split, attention, swiglu, span };
pub const Ops = struct {
    mod: module.Module,
    functions: [5]module.Function,

    pub fn load(d: *const Driver) !Ops {
        if (!kernels.available) return error.BuiltWithoutKernels;
        return loadImage(d, kernels.dsv4_vision);
    }

    pub fn loadImage(d: *const Driver, image: []const u8) !Ops {
        var mod = try module.Module.load(d, image);
        errdefer mod.unload();
        var funcs: [5]module.Function = undefined;
        inline for (@typeInfo(Op).@"enum".field_names, 0..) |name, i|
            funcs[i] = try mod.function("tf_dsv4_vision_" ++ name);
        return .{ .mod = mod, .functions = funcs };
    }

    pub fn unload(o: *Ops) void {
        o.mod.unload();
    }

    pub fn call(o: Ops, comptime op: Op, s: Stream, grid: launch.Dim3, threads: u32, shared: u32, values: anytype) !void {
        var args: launch.Args = .{};
        inline for (values) |v| args.add(v);
        try launch.launch(o.functions[@backingInt(op)], .{ .grid = grid, .block = .{ .x = threads }, .shared = shared }, s, &args);
    }

    pub fn norm(o: Ops, s: Stream, x: u64, w: u64, y: u64, rows: u32, dim: u32, kind: Kind) !void {
        if (dim == 0) return error.Invalid;
        if (rows != 0) try o.call(.norm, s, .{ .x = rows }, 256, 0, .{ x, w, y, dim, @as(u32, @backingInt(kind)) });
    }

    pub fn split(o: Ops, s: Stream, fused: u64, a: u64, b: u64, c: u64, rows: u32, dim: u32, parts: u32) !void {
        if (dim == 0 or (parts != 2 and parts != 3) or (parts == 3 and c == 0)) return error.Invalid;
        if (rows != 0) try o.call(.split, s, .{ .x = (dim + 127) / 128, .y = rows }, 128, 0, .{ fused, a, b, c, dim, parts });
    }

    pub fn attention(o: Ops, s: Stream, q: u64, k: u64, v: u64, y: u64, rows: u32, heads: u32, dim: u32) !void {
        if (rows == 0 or rows > 4096 or heads == 0 or dim == 0 or dim > 128) return error.Invalid;
        try o.call(.attention, s, .{ .x = rows, .y = heads }, 128, rows * 4, .{ q, k, v, y, rows, heads, dim });
    }

    pub fn swiglu(o: Ops, s: Stream, gate: u64, up: u64, out: u64, rows: u32, dim: u32) !void {
        if (dim == 0) return error.Invalid;
        if (rows != 0) try o.call(.swiglu, s, .{ .x = (dim + 127) / 128, .y = rows }, 128, 0, .{ gate, up, out, dim });
    }

    pub fn span(o: Ops, s: Stream, raster: u64, special: u64, map: u64, out: u64, tokens: u32, dim: u32) !void {
        if (tokens == 0 or tokens > 384 or dim == 0) return error.Invalid;
        try o.call(.span, s, .{ .x = (dim + 127) / 128, .y = tokens }, 128, 0, .{ raster, special, map, out, dim });
    }
};
