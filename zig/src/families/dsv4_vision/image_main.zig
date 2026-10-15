//! Native image-preprocessing oracle runner, without a model load or GPU.
const std = @import("std");
const config = @import("config.zig");
const picture = @import("picture.zig");
const layout = @import("image_layout.zig");

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 4 or args.len > 6) {
        std.debug.print("usage: tf-dsv4-image CONFIG IMAGE PATCHES_OUT [POSITION] [LIBTURBOJPEG]\n", .{});
        return 2;
    }
    const text = try std.Io.Dir.cwd().readFileAlloc(init.io, args[1], a, .limited(1 << 20));
    const c = try config.Config.parse(a, text);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, args[2], a, .limited(32 << 20));
    var jpeg: ?picture.jpeg.Decoder = if (std.mem.startsWith(u8, bytes, "\xff\xd8\xff"))
        try picture.jpeg.Decoder.openPath(if (args.len == 6) args[5] else if (@import("builtin").os.tag == .windows) "turbojpeg.dll" else "libturbojpeg.so.0")
    else
        null;
    defer if (jpeg) |*d| d.close();
    const image = try picture.decode(a, bytes, c, if (jpeg) |*d| d else null);
    const position = if (args.len >= 5) try std.fmt.parseInt(usize, args[4], 10) else 0;
    const block = try layout.block(a, image.grid, position);
    const kinds = try a.alloc(u8, block.types.len);
    for (kinds, block.types) |*k, t| k.* = @backingInt(t);
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[3], .data = std.mem.sliceAsBytes(image.patches) });
    const json = try std.json.Stringify.valueAlloc(a, .{ .geometry = image.grid, .types = kinds, .permutation = block.permutation }, .{ .emit_strings_as_arrays = true });
    try std.Io.File.stdout().writeStreamingAll(init.io, json);
    return 0;
}
