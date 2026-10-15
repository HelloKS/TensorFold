//! V4 image geometry feeds the shared native RGB resampler; V4.1's grid and token layout are not used.
const std = @import("std");
const Config = @import("config.zig").Config;
const layout = @import("image_layout.zig");
pub const rgb = @import("rgb.zig");
pub const png = @import("png.zig");
pub const jpeg = @import("jpeg.zig");

pub const Picture = struct {
    grid: layout.Geometry,
    patches: []u16,

    pub fn deinit(p: Picture, a: std.mem.Allocator) void {
        a.free(p.patches);
    }
};

pub fn fromRgb(a: std.mem.Allocator, image: rgb.Rgb, c: Config) !Picture {
    if (image.w == 0 or image.h == 0 or image.px.len != image.w * image.h * 3) return error.BadRgb;
    const g = try layout.geometry(c, image.h, image.w);
    var padded = if (image.w >= c.vision_max_wh_ratio * image.h)
        try rgb.resize(a, image, g.width, g.height)
    else
        try rgb.pad(a, image, g.width, g.height);
    defer padded.deinit(a);
    return .{ .grid = g, .patches = try rgb.patchify(a, padded, c.vision_patch_size) };
}

pub fn fromPng(a: std.mem.Allocator, bytes: []const u8, c: Config) !Picture {
    if (bytes.len > 32 << 20) return error.ImageTooLarge;
    var image = try png.decodePng(a, bytes);
    defer image.deinit(a);
    return fromRgb(a, image, c);
}

pub fn decode(a: std.mem.Allocator, bytes: []const u8, c: Config, jpg: ?*const jpeg.Decoder) !Picture {
    if (std.mem.startsWith(u8, bytes, "\x89PNG")) return fromPng(a, bytes, c);
    const decoder = jpg orelse return error.JpegLibraryUnavailable;
    var image = try decoder.decode(a, bytes);
    defer image.deinit(a);
    return fromRgb(a, image, c);
}

test "native normalization preserves channel-major patch order and BF16 ties" {
    const a = std.testing.allocator;
    var pixels: [12]u8 = .{ 0, 127, 255, 255, 127, 0, 0, 127, 255, 255, 127, 0 };
    const patches = try rgb.patchify(a, .{ .w = 2, .h = 2, .px = &pixels }, 2);
    defer a.free(patches);
    try std.testing.expectEqual(@as(u16, 0xbf80), patches[0]);
    try std.testing.expectEqual(@as(u16, 0x3f80), patches[1]);
    try std.testing.expectEqual(@as(u16, 0x3f80), patches[8]);
    try std.testing.expectEqual(@as(u16, 0xbf80), patches[9]);
    try std.testing.expectEqual(@as(u16, 0x3f80), rgb.bf16(@bitCast(@as(u32, 0x3f808000))));
}

test "PNG CRC failures are refused without a Python decoder" {
    const bad = "\x89PNG\r\n\x1a\n\x00\x00\x00\x00IEND\x00\x00\x00\x00";
    try std.testing.expectError(error.BadPng, png.decodePng(std.testing.allocator, bad));
}
