//! Lossless checkpoint-to-device uploads, for either source tensors or a validated TP2 rank cache.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const Catalog = @import("catalog.zig").Catalog;
const weights = @import("weights.zig");
const fmt = core.exl3_format;
const Kind = cuda.dsv4_tensor.Kind;
const Device = cuda.DeviceBuffer;

pub fn kind(dtype: core.safetensors.DType) !Kind {
    return switch (dtype) {
        .f32 => .f32,
        .f16 => .f16,
        .bf16 => .bf16,
        else => error.UnsupportedDType,
    };
}

fn upload(c: *const Catalog, a: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, key: []const u8) !Device {
    const tensor = try c.read(a, io, key);
    defer a.free(tensor.bytes);
    return Device.fromHost(d, tensor.bytes);
}

pub const Raw = struct {
    data: Device,
    dtype: core.safetensors.DType,

    pub fn load(c: *const Catalog, a: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, key: []const u8) !Raw {
        const header = c.headers.get(key) orelse return error.MissingTensor;
        return .{ .data = try upload(c, a, io, d, key), .dtype = header.dtype };
    }

    pub fn free(r: *Raw) void {
        r.data.free();
    }
};

pub const Scratch = struct {
    rotated: ?Device = null,
    sums: ?Device = null,

    pub fn free(s: *Scratch) void {
        if (s.rotated) |*b| b.free();
        if (s.sums) |*b| b.free();
        s.* = .{};
    }

    fn ensure(s: *Scratch, d: *const cuda.Driver, stream: cuda.Stream, rotated: usize, sums: usize) !void {
        if (s.rotated == null or s.rotated.?.len < rotated) {
            try stream.synchronize();
            const new = try Device.alloc(d, rotated);
            if (s.rotated) |*old| old.free();
            s.rotated = new;
        }
        if (s.sums == null or s.sums.?.len < sums) {
            try stream.synchronize();
            const new = try Device.alloc(d, sums);
            if (s.sums) |*old| old.free();
            s.sums = new;
        }
    }
};

pub const Matrix = struct {
    data: Device,
    su: ?Device = null,
    sv: ?Device = null,
    bias: ?Device = null,
    trellis_params: ?cuda.dsv4_exl3.Params = null,
    dtype: Kind = .f32,
    k: u32,
    n: u32,

    pub fn load(c: *const Catalog, gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, key: []const u8, k: usize, n: usize, codebook: fmt.Codebook) !Matrix {
        var arena: std.heap.ArenaAllocator = .init(gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const tname = try std.fmt.allocPrint(a, "{s}.trellis", .{key});
        var m: Matrix = .{ .data = undefined, .k = @intCast(k), .n = @intCast(n) };
        if (c.headers.get(tname)) |t| {
            const suname = try std.fmt.allocPrint(a, "{s}.suh", .{key});
            const svname = try std.fmt.allocPrint(a, "{s}.svh", .{key});
            const su = c.headers.get(suname) orelse return error.MissingTensor;
            const sv = c.headers.get(svname) orelse return error.MissingTensor;
            const shape = try weights.Matrix.parse(t, su, sv, k, n, codebook);
            var multiplier: u32 = switch (codebook) {
                .inst3 => fmt.Inst3Mul,
                .mcg => fmt.McgMul,
                .mul1 => fmt.Mul1Mul,
            };
            if (codebook == .mul1) {
                const marker = try c.read(gpa, io, try std.fmt.allocPrint(a, "{s}.mul1", .{key}));
                defer gpa.free(marker.bytes);
                if (marker.dtype != .i32 or marker.rank != 0 or marker.bytes.len != 4) return error.BadCodebook;
                multiplier = std.mem.readInt(u32, marker.bytes[0..4], .little);
            }
            m.trellis_params = .{ .k = m.k, .n = m.n, .rows = 0, .halves = shape.width.halves, .codebook = @fromBackingInt(@as(u32, @intCast(@backingInt(codebook)))), .multiplier = multiplier, .su_signs = @intFromBool(su.dtype == .i16), .sv_signs = @intFromBool(sv.dtype == .i16), .tile_n_stride = shape.width.tileWords() * 2, .tile_k_stride = n / 16 * shape.width.tileWords() * 2 };
            try m.trellis_params.?.validate();
            m.data = try upload(c, gpa, io, d, tname);
            errdefer m.data.free();
            m.su = try upload(c, gpa, io, d, suname);
            errdefer m.su.?.free();
            m.sv = try upload(c, gpa, io, d, svname);
        } else {
            const name = try std.fmt.allocPrint(a, "{s}.weight", .{key});
            const header = c.headers.get(name) orelse return error.MissingTensor;
            if (header.rank != 2 or header.shape[0] != n or header.shape[1] != k) return error.UnexpectedTensor;
            m.dtype = try kind(header.dtype);
            m.data = try upload(c, gpa, io, d, name);
        }
        errdefer m.free();
        const bname = try std.fmt.allocPrint(a, "{s}.bias", .{key});
        if (c.headers.get(bname)) |b| {
            if (b.rank != 1 or b.shape[0] != n) return error.UnexpectedTensor;
            const tensor = try c.read(gpa, io, bname);
            defer gpa.free(tensor.bytes);
            if (m.trellis_params != null) {
                if (b.dtype != .f16) return error.UnsupportedBiasType;
                m.bias = try Device.fromHost(d, tensor.bytes);
            } else {
                const floats = try gpa.alloc(f32, n);
                defer gpa.free(floats);
                try toFloat(tensor, floats);
                m.bias = try Device.fromHost(d, std.mem.sliceAsBytes(floats));
            }
        }
        return m;
    }

    pub fn free(m: *Matrix) void {
        m.data.free();
        if (m.su) |*b| b.free();
        if (m.sv) |*b| b.free();
        if (m.bias) |*b| b.free();
        m.* = undefined;
    }

    pub fn run(m: Matrix, exl3: cuda.dsv4_exl3.Projection, tensor: cuda.dsv4_tensor.Ops, stream: cuda.Stream, scratch: *Scratch, x: u64, y: u64, rows: u32, input: Kind, output: Kind) !void {
        if (rows == 0) return;
        if (input == .f16 or output == .f16) return error.UnsupportedActivationType;
        if (m.trellis_params) |p| {
            var params = p;
            params.rows = rows;
            params.input_f32 = @intFromBool(input == .f32);
            params.output_f32 = @intFromBool(output == .f32);
            const needed = try params.scratchBytes();
            try scratch.ensure(m.data.d, stream, needed.rotated, needed.sums);
            try exl3.run(stream, params, x, m.data.ptr, m.su.?.ptr, m.sv.?.ptr, if (m.bias) |b| b.ptr else 0, y, scratch.rotated.?.ptr, scratch.sums.?.ptr);
        } else try tensor.dense(stream, x, m.data.ptr, if (m.bias) |b| b.ptr else 0, y, rows, m.k, m.n, input, m.dtype, output, false, 0);
    }
};

pub fn toFloat(t: core.safetensors.Tensor, out: []f32) !void {
    if (out.len != t.numel()) return error.UnexpectedTensor;
    for (out, 0..) |*v, i| v.* = switch (t.dtype) {
        .f32 => @bitCast(std.mem.readInt(u32, t.bytes[i * 4 ..][0..4], .little)),
        .f16 => @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, t.bytes[i * 2 ..][0..2], .little)))),
        .bf16 => @bitCast(@as(u32, std.mem.readInt(u16, t.bytes[i * 2 ..][0..2], .little)) << 16),
        else => return error.UnsupportedDType,
    };
}

test "raw BF16 weights widen to FP32 without loss" {
    const bytes = [_]u8{ 0x80, 0x3f, 0x80, 0xbf };
    const tensor: core.safetensors.Tensor = .{ .bytes = &bytes, .dtype = .bf16, .rank = 1, .shape = .{ 2, 1, 1, 1, 1 } };
    var out: [2]f32 = undefined;
    try toFloat(tensor, &out);
    try std.testing.expectEqualSlices(f32, &.{ 1, -1 }, &out);
}
