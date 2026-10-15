//! Encoded PNG/JPEG to a positioned N-layout embedding span, ready for the text prefill graph.
const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const Catalog = @import("catalog.zig").Catalog;
const gpu = @import("gpu_weights.zig");
const vision = @import("vision_graph.zig");
const picture = @import("picture.zig");
const layout = @import("image_layout.zig");
const cache = @import("cache_key.zig");
const Sha = std.crypto.hash.sha2.Sha256;

pub const Image = struct {
    ids: []u32,
    embeddings: cuda.DeviceBuffer,
    key: cache.Image,

    pub fn free(image: *Image, a: std.mem.Allocator) void {
        a.free(image.ids);
        image.embeddings.free();
    }
};

pub const Encoder = struct {
    tower: vision.Tower,
    special: cuda.DeviceBuffer,
    preprocessing: [32]u8,

    pub fn load(ck: *const Catalog, a: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, c: Config) !Encoder {
        var tower = try vision.Tower.load(ck, a, io, d, c);
        errdefer tower.free(a);
        const values = try a.alloc(f32, c.hidden_size * 4);
        defer a.free(values);
        inline for (.{ "image_start", "image_pad", "image_newline", "image_end" }, 0..) |name, i| {
            const t = try ck.read(a, io, name);
            defer a.free(t.bytes);
            if (t.rank != 1 or t.shape[0] != c.hidden_size) return error.UnexpectedTensor;
            try gpu.toFloat(t, values[i * c.hidden_size ..][0..c.hidden_size]);
        }
        const special = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(values));
        return .{ .tower = tower, .special = special, .preprocessing = preprocessingKey(c) };
    }

    pub fn free(e: *Encoder, a: std.mem.Allocator) void {
        e.tower.free(a);
        e.special.free();
    }

    pub fn encode(e: *Encoder, a: std.mem.Allocator, runtime: vision.Runtime, s: cuda.Stream, bytes: []const u8, jpeg: ?*const picture.jpeg.Decoder, position: u32, context: u32) !Image {
        if (position >= context or context > @import("config.zig").default_context) return error.ContextExceeded;
        const c = e.tower.c;
        const rgb = try picture.decode(a, bytes, c, jpeg);
        defer a.free(rgb.patches);
        const block = try layout.block(a, rgb.grid, position);
        defer block.deinit(a);
        if (position > context or block.types.len > context - position) return error.ContextExceeded;
        const map = try a.alloc(i32, block.types.len);
        defer a.free(map);
        try mapRows(block, map);
        const ids = try block.tokens(a, @intCast(c.vocab_size));
        errdefer a.free(ids);
        var patches = try cuda.DeviceBuffer.fromHost(e.special.d, std.mem.sliceAsBytes(rgb.patches));
        defer patches.free();
        var raster = try e.tower.run(runtime, s, patches.ptr, @intCast(rgb.grid.height / c.vision_patch_size), @intCast(rgb.grid.width / c.vision_patch_size));
        defer raster.free();
        var mapping = try cuda.DeviceBuffer.fromHost(e.special.d, std.mem.sliceAsBytes(map));
        defer mapping.free();
        var embeddings = try cuda.DeviceBuffer.alloc(e.special.d, ids.len * c.hidden_size * 2);
        errdefer embeddings.free();
        defer s.synchronize() catch {};
        try runtime.vision.span(s, raster.ptr, e.special.ptr, mapping.ptr, embeddings.ptr, @intCast(ids.len), @intCast(c.hidden_size));
        try s.synchronize();
        var content: [32]u8 = undefined;
        Sha.hash(bytes, &content, .{});
        const leading = 3 - position % 4;
        return .{ .ids = ids, .embeddings = embeddings, .key = .{
            .span = .{ .start = position + leading, .end = position + @as(u32, @intCast(ids.len)) },
            .content = content,
            .preprocessing = e.preprocessing,
        } };
    }
};

pub fn mapRows(block: layout.Block, out: []i32) !void {
    if (out.len != block.types.len) return error.InvalidImageMap;
    var image: usize = 0;
    for (block.types, out) |kind, *index| index.* = switch (kind) {
        .start => -1,
        .pad => -2,
        .newline => -3,
        .end => -4,
        .image => blk: {
            if (image >= block.permutation.len) return error.InvalidImageMap;
            const row = block.permutation[image];
            if (row >= block.permutation.len) return error.InvalidImageMap;
            image += 1;
            break :blk @intCast(row);
        },
    };
    if (image != block.permutation.len) return error.InvalidImageMap;
}

pub fn preprocessingKey(c: Config) [32]u8 {
    var hash: Sha = .init(.{});
    hash.update("TensorFold/dsv4/rgb-bicubic-bf16-nlayout/v1\x00");
    for ([_]usize{ c.vision_patch_size, c.vision_downsample_ratio, c.vision_max_n_token, c.vision_min_pixels, c.vision_max_wh_ratio }) |value| {
        var word: [8]u8 = undefined;
        std.mem.writeInt(u64, &word, value, .little);
        hash.update(&word);
    }
    return hash.finalResult();
}

test "positioned image mapping preserves N-layout and explicit special embeddings" {
    const a = std.testing.allocator;
    const block = try layout.block(a, layout.grid(126, 126, 14, 3), 2);
    defer block.deinit(a);
    const map = try a.alloc(i32, block.types.len);
    defer a.free(map);
    try mapRows(block, map);
    try std.testing.expectEqual(@as(i32, -2), map[0]);
    try std.testing.expectEqual(@as(i32, -1), map[1]);
    try std.testing.expectEqual(@as(i32, 0), map[2]);
    try std.testing.expectEqual(@as(i32, 3), map[3]);
    try std.testing.expectEqual(@as(i32, -4), map[map.len - 1]);
}
