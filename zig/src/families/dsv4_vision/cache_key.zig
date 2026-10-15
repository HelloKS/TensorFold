//! Media-aware cache keys have prompt-length parity but never enter the model as token IDs.
const std = @import("std");
const Span = @import("image_layout.zig").Span;
const Sha = std.crypto.hash.sha2.Sha256;

pub fn namespace(storage: @import("storage.zig").Storage, checkpoint: [32]u8, tokenizer: [32]u8) [32]u8 {
    var hash: Sha = .init(.{});
    hash.update("TensorFold/dsv4/inference-state/v2\x00");
    hash.update(storage.name());
    if (storage == .exl3) hash.update(@tagName(storage.exl3));
    hash.update(&checkpoint);
    hash.update(&tokenizer);
    return hash.finalResult();
}

pub const Image = struct {
    span: Span,
    content: [32]u8,
    preprocessing: [32]u8,
};

/// The namespace includes the immutable checkpoint and arithmetic ABI; returned IDs are for cache lookup only.
pub fn tokens(a: std.mem.Allocator, prompt: []const u32, images: []const Image, model_namespace: [32]u8) ![]u32 {
    const out = try a.dupe(u32, prompt);
    errdefer a.free(out);
    var previous: u32 = 0;
    for (images) |image| {
        const span = image.span;
        if (span.start < previous or span.end > prompt.len or span.end <= span.start or span.end - span.start < 8)
            return error.InvalidImageSpans;
        var hash: Sha = .init(.{});
        hash.update("TensorFold/dsv4/image-prefix/v1\x00");
        hash.update(&model_namespace);
        hash.update(&image.content);
        hash.update(&image.preprocessing);
        var position: [8]u8 = undefined;
        std.mem.writeInt(u32, position[0..4], span.start, .little);
        std.mem.writeInt(u32, position[4..8], span.end, .little);
        hash.update(&position);
        const digest = hash.finalResult();
        for (0..8) |i| out[span.start + i] = std.mem.readInt(u32, digest[i * 4 ..][0..4], .little);
        previous = span.end;
    }
    return out;
}

pub fn snapshotAllowed(position: u32, images: []const Image) bool {
    for (images) |image| if (image.span.cuts(position)) return false;
    return true;
}

test "the same placeholder cannot reuse another image's prefix state" {
    const a = std.testing.allocator;
    const prompt: [32]u32 = @splat(129282);
    const ns: [32]u8 = @splat(4);
    var first = [_]Image{.{ .span = .{ .start = 8, .end = 24 }, .content = @splat(0), .preprocessing = @splat(1) }};
    const key_a = try tokens(a, &prompt, &first, ns);
    defer a.free(key_a);
    first[0].content[0] = 2;
    const key_b = try tokens(a, &prompt, &first, ns);
    defer a.free(key_b);
    try std.testing.expectEqualSlices(u32, prompt[0..8], key_a[0..8]);
    try std.testing.expect(!std.mem.eql(u32, key_a, key_b));
    try std.testing.expect(!snapshotAllowed(9, &first));
    try std.testing.expect(snapshotAllowed(8, &first) and snapshotAllowed(24, &first));
}

test "preprocessing and model identity participate in media cache keys" {
    const a = std.testing.allocator;
    const prompt: [16]u32 = @splat(1);
    const images = [_]Image{.{ .span = .{ .start = 0, .end = 16 }, .content = @splat(1), .preprocessing = @splat(2) }};
    const first = try tokens(a, &prompt, &images, @splat(0));
    defer a.free(first);
    const second = try tokens(a, &prompt, &images, @splat(1));
    defer a.free(second);
    try std.testing.expect(!std.mem.eql(u32, first, second));
}

test "native and EXL3 cannot share a state namespace at the same model revision" {
    const native = namespace(.native_fp4_fp8, @splat(1), @splat(2));
    const exl3 = namespace(.{ .exl3 = .mul1 }, @splat(1), @splat(2));
    try std.testing.expect(!std.mem.eql(u8, &native, &exl3));
    const changed = namespace(.native_fp4_fp8, @splat(3), @splat(2));
    try std.testing.expect(!std.mem.eql(u8, &native, &changed));
}
