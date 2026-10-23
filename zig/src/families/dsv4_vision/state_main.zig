//! Model-free device regression for rollback and moving a complete prefix between request slots.
const std = @import("std");
const cuda = @import("cuda");
const state = @import("text_state.zig");
const plan = @import("state_plan.zig");
fn stamp(s: *state.Sequence, stream: cuda.Stream, value: u8) !void {
    const rs = try plan.regions(s.a, s.c, s.len, true);
    defer s.a.free(rs);
    for (rs) |r| {
        const view: cuda.DeviceBuffer = .{ .d = s.taps.d, .ptr = s.part(r.part).ptr + r.offset, .len = r.bytes, .borrowed = true };
        try view.fill8(value, stream.handle);
    }
    for (s.layers, 0..) |layer, i| {
        const ratio = s.c.compress_ratios[i];
        if (ratio == 0 or (s.len + 1) % ratio != 0) continue;
        const offset = @as(usize, s.len / ratio) * s.c.head_dim * 2;
        const view: cuda.DeviceBuffer = .{ .d = s.taps.d, .ptr = layer.compressor.?.cache.ptr + offset, .len = s.c.head_dim * 2, .borrowed = true };
        try view.fill8(value, stream.handle);
        if (layer.indexer) |index| {
            const indexed: cuda.DeviceBuffer = .{ .d = s.taps.d, .ptr = index.cache.ptr + @as(usize, s.len / ratio) * s.c.index_head_dim * 2, .len = s.c.index_head_dim * 2, .borrowed = true };
            try indexed.fill8(value, stream.handle);
        }
    }
    try stream.synchronize();
    s.len += 1;
}
fn equal(a: std.mem.Allocator, expected: state.Snapshot, s: *state.Sequence, stream: cuda.Stream) !void {
    var actual = try state.Snapshot.capture(s, false, stream);
    defer actual.free();
    if (actual.at != expected.at or actual.data.len != expected.data.len) return error.StateShape;
    const lhs = try a.alloc(u8, actual.data.len);
    defer a.free(lhs);
    const rhs = try a.alloc(u8, lhs.len);
    defer a.free(rhs);
    try actual.data.download(0, lhs);
    try expected.data.download(0, rhs);
    if (!std.mem.eql(u8, lhs, rhs)) return error.StateMismatch;
}
pub fn main(init: std.process.Init) !void {
    const a = init.gpa;
    const c = try @import("config.zig").Config.parse(init.arena.allocator(), @import("config.zig").native_fixture);
    var d = try cuda.Driver.open();
    defer d.close();
    var ctx = try cuda.Context.init(&d, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&d, true);
    defer stream.deinit();
    var seq = try state.Sequence.init(a, &d, c, 512);
    defer seq.free();
    var other = try state.Sequence.init(a, &d, c, 512);
    defer other.free();
    try seq.reset(stream);
    try other.reset(stream);
    for (0..128) |i| try stamp(&seq, stream, @intCast(i));
    var saved = try state.Snapshot.capture(&seq, false, stream);
    defer saved.free();
    for (0..6) |i| { try seq.before(stream); try stamp(&seq, stream, @intCast(128 + i)); }
    try seq.keep(128, stream);
    try equal(a, saved, &seq, stream);
    try saved.restore(&other, stream);
    try equal(a, saved, &other, stream);
    // A partially accepted verify restores from its own post-accepted compressor/indexer state.
    for (0..2) |i| { try seq.before(stream); try stamp(&seq, stream, @intCast(200 + i)); }
    var partial = try state.Snapshot.capture(&seq, false, stream);
    defer partial.free();
    for (0..4) |i| { try seq.before(stream); try stamp(&seq, stream, @intCast(220 + i)); }
    try seq.keep(130, stream);
    try equal(a, partial, &seq, stream);
    var raster = try cuda.DeviceBuffer.alloc(&d, 512);
    defer raster.free();
    var images: @import("image_cache.zig").Cache = .{ .limit = 1024 };
    defer images.free(a);
    try raster.fill8(31, stream.handle);
    try images.put(a, @splat(1), raster, stream);
    try raster.fill8(47, stream.handle);
    try images.put(a, @splat(2), raster, stream);
    _ = images.get(@splat(1)) orelse return error.ImageCacheMiss;
    try images.put(a, @splat(3), raster, stream);
    if (images.get(@splat(2)) != null or images.used != 1024) return error.ImageCacheBudget;
    const reused = images.get(@splat(1)) orelse return error.ImageCacheMiss;
    var pixels: [512]u8 = undefined;
    try reused.download(0, &pixels);
    for (pixels) |byte| if (byte != 31) return error.ImageCacheCorrupt;
    try other.reset(stream);
    if (other.len != 0) return error.ResetPosition;
    std.debug.print("{{\"rollback\":true,\"partial_accept\":true,\"prefix_transfer\":true,\"slot_reset\":true,\"image_cache\":true,\"snapshot_bytes\":{d},\"undo_bytes\":{d}}}\n", .{saved.data.len, try plan.bytes(c, 128, true)});
}
