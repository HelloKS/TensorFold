//! Physical byte slices derived from logical TP2 matrices, with quantization-scale boundaries preserved.
const std = @import("std");
const st = @import("core").safetensors;
const Config = @import("config.zig").Config;
const weights = @import("weights.zig");
const parallel = @import("parallel.zig");

pub const Payload = struct {
    name: []const u8,
    dtype: st.DType,
    shape: []const usize,
    ranks: [2]parallel.Slice,
    bf16_conversion: bool = false,
};

const Builder = struct {
    a: std.mem.Allocator,
    headers: st.Header,
    items: std.ArrayList(Payload) = .empty,

    fn add(b: *Builder, key: []const u8, suffix: []const u8, parts: [2]parallel.Slice, conversion: bool) !void {
        const name = try std.fmt.allocPrint(b.a, "{s}{s}", .{ key, suffix });
        const t = b.headers.get(name) orelse return error.MissingTensor;
        try b.items.append(b.a, .{ .name = name, .dtype = t.dtype, .shape = try b.a.dupe(usize, t.shape[0..t.rank]), .ranks = parts, .bf16_conversion = conversion });
    }
};

fn physical(parts: [2]parallel.Slice, reverse: bool, affected: ?u8, factors: [2]usize) ![2]parallel.Slice {
    var out = parts;
    for (&out) |*p| {
        if (p.axis) |axis| {
            if (affected != null and axis != affected.?) {
                p.axis = null;
                p.first = 0;
                p.count = 0;
            } else {
                if (p.first % factors[axis] != 0 or p.count % factors[axis] != 0) return error.UnalignedScaleSlice;
                p.first /= factors[axis];
                p.count /= factors[axis];
                if (affected != null) p.axis = 0 else if (reverse) p.axis = 1 - axis;
            }
        }
    }
    return out;
}

pub fn select(a: std.mem.Allocator, c: Config, headers: st.Header) ![]const Payload {
    var b: Builder = .{ .a = a, .headers = headers };
    for (try weights.requirements(a, c)) |spec| {
        const parts = [2]parallel.Slice{ try parallel.partition(c, spec, 0), try parallel.partition(c, spec, 1) };
        if (spec.kind != .matrix) {
            try b.add(spec.key, "", parts, false);
            continue;
        }
        const tn = try std.fmt.allocPrint(a, "{s}.trellis", .{spec.key});
        if (c.storage == .exl3 and headers.contains(tn)) {
            try b.add(spec.key, ".trellis", try physical(parts, false, null, .{ 16, 16 }), false);
            inline for (.{ "suh", "svh" }, 0..) |suffix, axis| {
                const name = try std.fmt.allocPrint(a, "{s}.{s}", .{ spec.key, suffix });
                const factor: usize = if (headers.get(name).?.dtype == .i16) 16 else 1;
                try b.add(spec.key, "." ++ suffix, try physical(parts, false, axis, .{ factor, factor }), false);
            }
            const marker = try std.fmt.allocPrint(a, ".{t}", .{c.storage.exl3});
            if (headers.contains(try std.fmt.allocPrint(a, "{s}{s}", .{ spec.key, marker })))
                try b.add(spec.key, marker, .{ .{ .keep = parts[0].keep }, .{ .keep = parts[1].keep } }, false);
        } else {
            const t = headers.get(try std.fmt.allocPrint(a, "{s}.weight", .{spec.key})).?;
            const fp4 = t.dtype == .i8;
            const conversion = c.storage == .native_fp4_fp8 and std.mem.endsWith(u8, spec.key, ".attn.wo_a");
            try b.add(spec.key, ".weight", try physical(parts, true, null, .{ if (fp4) 2 else 1, 1 }), conversion);
            if (t.dtype == .i8 or t.dtype == .f8_e4m3) {
                try b.add(spec.key, ".scale", try physical(parts, true, null, .{ if (fp4) 32 else 128, if (fp4) 1 else 128 }), false);
            }
        }
    }
    return b.items.toOwnedSlice(a);
}

pub const DtypeBytes = struct { dtype: st.DType, bytes: u64 };
pub const RankBytes = struct {
    payload_bytes: u64 = 0,
    by_dtype: []const DtypeBytes,
    converted_bf16_bytes: u64 = 0,
    replaced_fp8_bytes: u64 = 0,
    resident_weight_bytes: u64 = 0,
    released_scale_bytes: u64 = 0,
    conversion_old_weight_peak: u64 = 0,
    host_staging_bytes: u64 = 8 << 20,
};

pub fn memory(a: std.mem.Allocator, items: []const Payload, rank: u32) !RankBytes {
    if (rank > 1) return error.BadRank;
    var sums: [@typeInfo(st.DType).@"enum".field_names.len]u64 = @splat(0);
    var out: RankBytes = .{ .by_dtype = &.{} };
    for (items) |item| {
        const p = item.ranks[rank];
        if (!p.keep) continue;
        var elements: u64 = 1;
        for (item.shape, 0..) |size, axis| elements = try std.math.mul(u64, elements, if (p.axis != null and p.axis.? == axis) p.count else size);
        const bytes = try std.math.mul(u64, elements, item.dtype.size());
        sums[@backingInt(item.dtype)] += bytes;
        out.payload_bytes += bytes;
        if (item.bf16_conversion) {
            out.converted_bf16_bytes += elements * 2;
            out.replaced_fp8_bytes += bytes;
            out.conversion_old_weight_peak = @max(out.conversion_old_weight_peak, bytes);
        }
        if (std.mem.endsWith(u8, item.name, ".attn.wo_a.scale")) out.released_scale_bytes += bytes;
    }
    var list: std.ArrayList(DtypeBytes) = .empty;
    for (sums, 0..) |bytes, i| if (bytes != 0) try list.append(a, .{ .dtype = @fromBackingInt(@intCast(i)), .bytes = bytes });
    out.by_dtype = try list.toOwnedSlice(a);
    out.resident_weight_bytes = out.payload_bytes - out.replaced_fp8_bytes - out.released_scale_bytes + out.converted_bf16_bytes;
    return out;
}

test "FP4 K slices divide packed bytes by two and scales by 32" {
    const parts = [2]parallel.Slice{ .{ .axis = 0, .first = 0, .count = 1024 }, .{ .axis = 0, .first = 1024, .count = 1024 } };
    const w = try physical(parts, true, null, .{ 2, 1 });
    const s = try physical(parts, true, null, .{ 32, 1 });
    try std.testing.expectEqual(@as(?u8, 1), w[1].axis);
    try std.testing.expectEqual(@as(usize, 512), w[1].first);
    try std.testing.expectEqual(@as(usize, 32), s[1].first);
    try std.testing.expectEqual(@as(usize, 32), s[1].count);
    var bad = parts;
    bad[0].count = 16;
    try std.testing.expectError(error.UnalignedScaleSlice, physical(bad, true, null, .{ 32, 1 }));
}

test "EXL3 output vector uses vector axis zero rather than matrix axis one" {
    const parts = [2]parallel.Slice{ .{ .axis = 1, .first = 0, .count = 128 }, .{ .axis = 1, .first = 128, .count = 128 } };
    const sv = try physical(parts, false, 1, .{ 1, 1 });
    const su = try physical(parts, false, 0, .{ 1, 1 });
    try std.testing.expectEqual(@as(?u8, 0), sv[1].axis);
    try std.testing.expectEqual(@as(usize, 128), sv[1].first);
    try std.testing.expectEqual(@as(?u8, null), su[1].axis);
}
