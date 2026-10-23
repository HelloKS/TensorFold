//! Rank-zero aligner outputs share the prompt-cache budget and are keyed before position-specific span packing.
const std = @import("std");
const cuda = @import("cuda");
pub const Key = [32]u8;
const Entry = struct { key: Key, data: cuda.DeviceBuffer, stamp: u64 };
pub const Cache = struct {
    entries: std.ArrayList(Entry) = .empty,
    limit: u64 = 0,
    used: u64 = 0,
    clock: u64 = 0,
    hits: u64 = 0,
    misses: u64 = 0,
    pub fn get(c: *Cache, digest: Key) ?cuda.DeviceBuffer {
        c.clock += 1;
        for (c.entries.items) |*entry| if (std.mem.eql(u8, &entry.key, &digest)) {
            entry.stamp = c.clock;
            c.hits += 1;
            var view = entry.data;
            view.borrowed = true;
            return view;
        };
        c.misses += 1;
        return null;
    }
    pub fn put(c: *Cache, a: std.mem.Allocator, digest: Key, raster: cuda.DeviceBuffer, stream: cuda.Stream) !void {
        if (c.limit == 0 or raster.len > c.limit) return;
        while (c.used + raster.len > c.limit) {
            var oldest: usize = 0;
            for (c.entries.items, 0..) |entry, i| if (entry.stamp < c.entries.items[oldest].stamp) { oldest = i; };
            var dropped = c.entries.orderedRemove(oldest);
            c.used -= dropped.data.len;
            dropped.data.free();
        }
        var copy = try cuda.DeviceBuffer.alloc(raster.d, raster.len);
        errdefer { stream.synchronize() catch {}; copy.free(); }
        try copy.copyFrom(0, raster.ptr, raster.len, stream.handle);
        try stream.synchronize();
        c.clock += 1;
        try c.entries.append(a, .{ .key = digest, .data = copy, .stamp = c.clock });
        c.used += copy.len;
    }
    pub fn free(c: *Cache, a: std.mem.Allocator) void {
        for (c.entries.items) |*entry| entry.data.free();
        c.entries.deinit(a);
    }
};
pub fn key(identity: [32]u8, preprocessing: [32]u8, pixels: []const f32, gh: u32, gw: u32) Key {
    var sha = std.crypto.hash.sha2.Sha256.init(.{});
    sha.update("dsv4/aligner-cache/v1\x00");
    sha.update(&identity);
    sha.update(&preprocessing);
    var grid: [8]u8 = undefined;
    std.mem.writeInt(u32, grid[0..4], gh, .little);
    std.mem.writeInt(u32, grid[4..8], gw, .little);
    sha.update(&grid);
    sha.update(std.mem.sliceAsBytes(pixels));
    return sha.finalResult();
}
test "aligner keys include actual pixels, grid, preprocessing and execution identity" {
    const original = key(@splat(1), @splat(2), &.{0, 1}, 2, 3);
    for ([_]Key{ key(@splat(3), @splat(2), &.{0, 1}, 2, 3), key(@splat(1), @splat(4), &.{0, 1}, 2, 3),
        key(@splat(1), @splat(2), &.{1, 0}, 2, 3), key(@splat(1), @splat(2), &.{0, 1}, 3, 2) }) |other|
        try std.testing.expect(!std.mem.eql(u8, &original, &other));
}
