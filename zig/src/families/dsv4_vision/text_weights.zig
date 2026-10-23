//! Target text weights from an already validated, losslessly partitioned rank cache.
const std = @import("std");
const cuda = @import("cuda");
const gpu = @import("gpu_weights.zig");
const Catalog = @import("catalog.zig").Catalog;
const Config = @import("config.zig").Config;
pub const Weights = struct {
    a: std.mem.Allocator,
    matrices: std.StringHashMapUnmanaged(gpu.Matrix) = .empty,
    raw: std.StringHashMapUnmanaged(gpu.Raw) = .empty,
    pub fn load(a: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, s: cuda.Stream, text: cuda.dsv4_text.Ops, cat: *const Catalog, c: Config, rank: u32) !Weights {
        var w: Weights = .{ .a = a };
        errdefer w.free();
        var arena: std.heap.ArenaAllocator = .init(a);
        defer arena.deinit();
        for (try @import("weights.zig").requirements(arena.allocator(), c)) |spec| {
            if (std.mem.startsWith(u8, spec.key, "vision.") or std.mem.startsWith(u8, spec.key, "aligner.") or std.mem.startsWith(u8, spec.key, "image_")) continue;
            const part = try @import("parallel.zig").partition(c, spec, rank);
            if (!part.keep) continue;
            const key = try a.dupe(u8, spec.key);
            errdefer a.free(key);
            if (spec.kind == .matrix) {
                var shape = spec.shape;
                if (part.axis) |axis| shape[axis] = part.count;
                var value = try gpu.Matrix.load(cat, a, io, d, key, shape[0], shape[1], c.storage);
                errdefer value.free();
                try w.matrices.put(a, key, value);
            } else {
                var value = try gpu.Raw.load(cat, a, io, d, key);
                errdefer value.free();
                if (spec.kind == .float and !std.mem.eql(u8, key, "embed.weight") and value.dtype != .f32) {
                    const kind = try gpu.kind(value.dtype);
                    const header = cat.headers.get(key).?;
                    const count = (header.end - header.begin) / header.dtype.size();
                    var converted = try cuda.DeviceBuffer.alloc(d, count * 4);
                    errdefer converted.free();
                    try text.call("cast", s, @intCast((count + 127) / 128), 128, .{ value.data.ptr, converted.ptr, @as(u32, @intCast(count)), @as(u32, @backingInt(kind)) });
                    try s.synchronize();
                    value.free();
                    value = .{ .data = converted, .dtype = .f32 };
                }
                if (spec.kind == .ids and value.dtype == .i32) {
                    const count = value.data.len / 4;
                    var converted = try cuda.DeviceBuffer.alloc(d, count * 8);
                    errdefer converted.free();
                    try text.call("ids", s, @intCast((count + 127) / 128), 128, .{ value.data.ptr, converted.ptr, @as(u32, @intCast(count)) });
                    try s.synchronize();
                    value.free();
                    value = .{ .data = converted, .dtype = .i64 };
                }
                try w.raw.put(a, key, value);
            }
        }
        return w;
    }
    pub fn matrix(w: *Weights, key: []const u8) !*gpu.Matrix {
        return w.matrices.getPtr(key) orelse error.MissingTensor;
    }
    pub fn tensor(w: *Weights, key: []const u8) !gpu.Raw {
        return w.raw.get(key) orelse error.MissingTensor;
    }
    pub fn ptr(w: *Weights, key: []const u8) !u64 {
        return (try w.tensor(key)).data.ptr;
    }
    pub fn free(w: *Weights) void {
        var it = w.matrices.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.free();
            w.a.free(entry.key_ptr.*);
        }
        w.matrices.deinit(w.a);
        var raw = w.raw.iterator();
        while (raw.next()) |entry| {
            entry.value_ptr.free();
            w.a.free(entry.key_ptr.*);
        }
        w.raw.deinit(w.a);
    }
};
