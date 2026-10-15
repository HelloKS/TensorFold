//! Typed launch contracts for the source-built V4 model operators.
const std = @import("std");
const module = @import("module.zig");
const kernels = @import("kernels.zig");
const launch = @import("launch.zig");
const Driver = @import("driver.zig").Driver;
const Stream = @import("stream.zig").Stream;

pub const Op = enum { route, sinkhorn, hc_pre, hc_post, qat, rope, pool, sparse_attn, swiglu, unfold };

pub const Sparse = struct {
    rows: u32,
    heads: u32,
    selected: u32,
    dim: u32,

    pub fn sharedBytes(s: Sparse) !u32 {
        if (s.heads == 0 or s.heads > 64 or s.dim == 0 or s.dim > 512 or s.dim % 32 != 0 or
            s.selected > 4096) return error.UnsupportedSparseShape;
        return (s.selected + 256) * 4;
    }
};

pub const Ops = struct {
    mod: module.Module,
    functions: [10]module.Function,

    pub fn load(d: *const Driver) !Ops {
        if (!kernels.available) return error.BuiltWithoutKernels;
        return loadImage(d, kernels.dsv4_ops);
    }

    pub fn loadImage(d: *const Driver, image: []const u8) !Ops {
        var mod = try module.Module.load(d, image);
        errdefer mod.unload();
        var functions: [10]module.Function = undefined;
        inline for (@typeInfo(Op).@"enum".field_names, 0..) |name, i|
            functions[i] = try mod.function("tf_dsv4_" ++ name);
        return .{ .mod = mod, .functions = functions };
    }

    pub fn unload(o: *Ops) void {
        o.mod.unload();
    }

    fn call(o: Ops, comptime op: Op, stream: Stream, grid: launch.Dim3, threads: u32, shared: u32, values: anytype) !void {
        var args: launch.Args = .{};
        inline for (values) |v| args.add(v);
        try launch.launch(o.functions[@backingInt(op)], .{ .grid = grid, .block = .{ .x = threads }, .shared = shared }, stream, &args);
    }

    pub fn route(o: Ops, s: Stream, logits: u64, ids: u64, table: u64, bias: u64, bias_vl: u64, picks: u64, weights: u64, rows: u32, vocab: u32, hash: bool, scale: f32) !void {
        if (rows == 0) return;
        if (vocab == 0 or (hash and table == 0) or !std.math.isFinite(scale) or scale <= 0) return error.Invalid;
        try o.call(.route, s, .{ .x = rows }, 256, 0, .{ logits, ids, table, bias, bias_vl, picks, weights, vocab, @as(u32, @intFromBool(hash)), scale });
    }

    pub fn sinkhorn(o: Ops, s: Stream, mixes: u64, scale: u64, base: u64, pre: u64, post: u64, comb: u64, rows: u32, iterations: u32, eps: f32) !void {
        if (rows == 0) return;
        if (iterations == 0 or iterations > 100 or eps <= 0 or !std.math.isFinite(eps)) return error.Invalid;
        try o.call(.sinkhorn, s, .{ .x = rows }, 32, 0, .{ mixes, scale, base, pre, post, comb, iterations, eps });
    }

    pub fn hcPre(o: Ops, s: Stream, residual: u64, pre: u64, out: u64, rows: u32, width: u32) !void {
        if (rows == 0) return;
        if (width == 0) return error.Invalid;
        try o.call(.hc_pre, s, .{ .x = (width + 127) / 128, .y = rows }, 128, 0, .{ residual, pre, out, width });
    }

    pub fn hcPost(o: Ops, s: Stream, x: u64, residual: u64, post: u64, comb: u64, out: u64, rows: u32, width: u32) !void {
        if (rows == 0) return;
        if (width == 0) return error.Invalid;
        try o.call(.hc_post, s, .{ .x = (width + 127) / 128, .y = rows }, 128, 0, .{ x, residual, post, comb, out, width });
    }

    pub fn qat(o: Ops, s: Stream, x: u64, rows: u32, width: u32, group: u32, fp4: bool, stride: u32) !void {
        if (rows == 0) return;
        if ((group != 32 and group != 64 and group != 128) or width == 0 or width % group != 0 or stride < width)
            return error.Invalid;
        try o.call(.qat, s, .{ .x = width / group, .y = rows }, 128, 0, .{ x, width, group, @as(u32, @intFromBool(fp4)), stride });
    }

    pub fn rope(o: Ops, s: Stream, x: u64, cos: u64, sin: u64, positions: u64, rows: u32, heads: u32, dim: u32, rope_dim: u32, inverse: bool) !void {
        if (rows == 0) return;
        if (heads == 0 or rope_dim == 0 or rope_dim % 2 != 0 or rope_dim > dim or rope_dim > 512)
            return error.Invalid;
        try o.call(.rope, s, .{ .x = rows, .y = heads }, 256, 0, .{ x, cos, sin, positions, heads, dim, rope_dim, @as(u32, @intFromBool(inverse)) });
    }

    pub fn pool(o: Ops, s: Stream, kv: u64, scores: u64, out: u64, rows: u32, slots: u32, dim: u32) !void {
        if (rows == 0) return;
        if (slots == 0 or slots > 128 or dim == 0 or dim > 512) return error.Invalid;
        try o.call(.pool, s, .{ .x = (dim + 127) / 128, .y = rows }, 128, 0, .{ kv, scores, out, slots, dim });
    }

    pub fn sparse(o: Ops, s: Stream, q: u64, kv: u64, indices: u64, sink: u64, out: u64, shape: Sparse) !void {
        const bytes = try shape.sharedBytes();
        if (shape.rows == 0) return;
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(shape.dim)));
        try o.call(.sparse_attn, s, .{ .x = shape.rows, .y = shape.heads }, 256, bytes, .{ q, kv, indices, sink, out, shape.heads, shape.selected, shape.dim, scale });
    }

    pub fn swiglu(o: Ops, s: Stream, gate: u64, up: u64, weights: u64, out: u64, rows: u32, width: u32, limit: f32) !void {
        if (rows == 0) return;
        if (width == 0 or limit <= 0 or !std.math.isFinite(limit)) return error.Invalid;
        try o.call(.swiglu, s, .{ .x = (width + 127) / 128, .y = rows }, 128, 0, .{ gate, up, weights, out, width, limit });
    }

    pub fn unfold(o: Ops, s: Stream, x: u64, out: u64, h: u32, w: u32, channels: u32) !void {
        if (h == 0 or w == 0 or channels == 0 or h > 1024 or w > 1024 or channels > 1024) return error.Invalid;
        const rows = ((h + 2) / 3) * ((w + 2) / 3);
        try o.call(.unfold, s, .{ .x = (channels * 9 + 127) / 128, .y = rows }, 128, 0, .{ x, out, h, w, channels });
    }
};

test "sparse attention shared storage covers the 256K ratio-128 context" {
    const shape: Sparse = .{ .rows = 4, .heads = 32, .selected = 262144 / 128 + 128, .dim = 512 };
    try std.testing.expectEqual(@as(u32, 9728), try shape.sharedBytes());
    var bad = shape;
    bad.selected = 4097;
    try std.testing.expectError(error.UnsupportedSparseShape, bad.sharedBytes());
}
