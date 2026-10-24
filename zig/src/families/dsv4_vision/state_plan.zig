//! Byte ranges needed by a prefix snapshot or by undoing one appended target row.
const std = @import("std");
const Config = @import("config.zig").Config;
pub const Region = struct { part: usize, offset: usize, bytes: usize };

/// One cache row of `dim` channels: the bf16 halves, or --kv-cache fp8's e4m3 bytes and their fp32 scale.
pub fn rowBytes(dim: usize, fp8: bool) usize {
    return if (fp8) dim + 4 else dim * 2;
}

const Builder = struct {
    a: std.mem.Allocator,
    regions: std.ArrayList(Region) = .empty,
    part: usize = 0,
    fn add(b: *Builder, offset: usize, size: usize) !void {
        if (size > 0) try b.regions.append(b.a, .{ .part = b.part, .offset = offset, .bytes = size });
    }
    fn recurrent(b: *Builder, at: usize, ratio: usize, dim: usize, undo: bool) !void {
        const overlap: usize = if (ratio == 4) 2 else 1;
        const row = dim * overlap * 4;
        for (0..2) |_| {
            try b.add(if (undo) at % (ratio * overlap) * row else 0, if (undo) row else ratio * overlap * row);
            b.part += 1;
        }
    }
};
pub fn regions(a: std.mem.Allocator, c: Config, at: usize, undo: bool, fp8: bool) ![]Region {
    if (at > @import("config.zig").default_context) return error.ContextExceeded;
    var b: Builder = .{ .a = a };
    errdefer b.regions.deinit(a);
    const row = rowBytes(c.head_dim, fp8);
    for (c.compress_ratios[0..c.num_hidden_layers]) |ratio| {
        try b.add(if (undo) at % c.sliding_window * row else 0, if (undo) row else @min(at, c.sliding_window) * row);
        if (!undo and ratio > 0) try b.add((c.sliding_window + c.vision_max_n_token) * row, at / ratio * row);
        b.part += 1;
        if (ratio > 0) try b.recurrent(at, ratio, c.head_dim, undo);
        if (ratio == 4) {
            try b.recurrent(at, ratio, c.index_head_dim, undo);
            if (!undo) try b.add(0, at / ratio * rowBytes(c.index_head_dim, fp8));
            b.part += 1;
        }
    }
    for (0..c.num_nextn_predict_layers) |_| {
        try b.add(if (undo) at % c.sliding_window * row else 0, if (undo) row else @min(at, c.sliding_window) * row);
        b.part += 1;
    }
    try b.add(0, c.hidden_size * c.dspark_target_layer_ids.len * 2);
    b.part += 1;
    try b.add(0, c.hidden_size * 2);
    return b.regions.toOwnedSlice(a);
}
pub fn bytes(c: Config, at: usize, undo: bool, fp8: bool) !usize {
    const a = std.heap.page_allocator;
    const list = try regions(a, c, at, undo, fp8);
    defer a.free(list);
    var total: usize = 0;
    for (list) |r| total = try std.math.add(usize, total, r.bytes);
    return total;
}

test "snapshots include compressed prefixes, undo holds only overwritten rows" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try Config.parse(a, @import("config.zig").native_fixture);
    try std.testing.expect(try bytes(c, 262144, false, false) > try bytes(c, 128, false, false));
    try std.testing.expectEqual(try bytes(c, 128, true, false), try bytes(c, 262144, true, false));
    const before = try regions(a, c, 127, true, false);
    const after = try regions(a, c, 128, true, false);
    try std.testing.expectEqual(@as(usize, 127 * c.head_dim * 2), before[0].offset);
    try std.testing.expectEqual(@as(usize, 0), after[0].offset);
    try std.testing.expect(try bytes(c, 262144, true, false) < 2 << 20);
}

test "an fp8 cache plans e4m3 rows with a scale each, and leaves the recurrent rings alone" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try Config.parse(a, @import("config.zig").native_fixture);
    try std.testing.expectEqual(@as(usize, 516), rowBytes(c.head_dim, true));
    try std.testing.expectEqual(@as(usize, 1024), rowBytes(c.head_dim, false));
    try std.testing.expect(try bytes(c, 262144, false, true) < try bytes(c, 262144, false, false));
    const wide = try regions(a, c, 4096, false, true);
    const plain = try regions(a, c, 4096, false, false);
    try std.testing.expectEqual(wide.len, plain.len); // the same parts in the same order: only the rows' widths change
    const prefix = 4096 / 4 * rowBytes(c.head_dim, true);
    var found = false;
    for (wide) |w| {
        if (w.bytes == prefix) found = true;
    }
    try std.testing.expect(found); // one e4m3 row and its scale for every four tokens of a compressed prefix
    var same: usize = 0;
    for (wide, plain) |w, p| {
        if (w.bytes == p.bytes) same += 1;
    }
    try std.testing.expect(same > 100); // the compressor rings and the taps are FP32 whatever the cache's rows are
}
