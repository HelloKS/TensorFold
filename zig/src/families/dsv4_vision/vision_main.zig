//! Execute the complete native ViT and aligner for independent checkpoint qualification.
const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const Catalog = @import("catalog.zig").Catalog;
const vision = @import("vision_graph.zig");
const media = @import("media.zig");
const picture = @import("picture.zig");

const Trace = struct {
    a: std.mem.Allocator,
    io: std.Io,
    dir: []const u8,

    fn write(ptr: *anyopaque, index: u32, state: cuda.DeviceBuffer) !void {
        const t: *Trace = @ptrCast(@alignCast(ptr));
        const name = try std.fmt.allocPrint(t.a, "layer{d}.bf16", .{index});
        defer t.a.free(name);
        const path = try std.fs.path.join(t.a, &.{ t.dir, name });
        defer t.a.free(path);
        const bytes = try t.a.alloc(u8, state.len);
        defer t.a.free(bytes);
        try state.download(0, bytes);
        try std.Io.Dir.cwd().writeFile(t.io, .{ .sub_path = path, .data = bytes });
    }
};

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len >= 2 and std.mem.eql(u8, args[1], "--image")) return imageMain(init, args);
    if (args.len != 7 and args.len != 8) {
        std.debug.print("usage: tf-dsv4-vision MODEL PATCHES_BF16 GRID_H GRID_W OUTPUT_BF16 KERNEL_DIR [TRACE_DIR]\n", .{});
        return 2;
    }
    const height = try std.fmt.parseInt(u32, args[3], 10);
    const width = try std.fmt.parseInt(u32, args[4], 10);
    if (height == 0 or width == 0 or height > 4096 or width > 4096 or height * width > 4096) return error.InvalidImageShape;
    const config_path = try std.fs.path.join(a, &.{ args[1], "config.json" });
    const c = try Config.parse(a, try std.Io.Dir.cwd().readFileAlloc(init.io, config_path, a, .limited(1 << 20)));
    const ck = try Catalog.open(a, init.io, args[1]);
    var d = try cuda.Driver.open();
    defer d.close();
    var context = try cuda.Context.init(&d, 0);
    defer context.deinit();
    var s = try cuda.Stream.init(&d, true);
    defer s.deinit();
    var runtime = try vision.Runtime.open(init.gpa, init.io, &d, c, args[6]);
    defer runtime.close();
    var tower = try vision.Tower.load(&ck, init.gpa, init.io, &d, c);
    defer tower.free(init.gpa);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, args[2], a, .limited(16 << 20));
    if (bytes.len != @as(usize, height) * width * 3 * c.vision_patch_size * c.vision_patch_size * 2) return error.BadInputSize;
    var x = try cuda.DeviceBuffer.fromHost(&d, bytes);
    defer x.free();
    var trace: Trace = .{ .a = init.gpa, .io = init.io, .dir = if (args.len == 8) args[7] else "" };
    const observer: ?vision.Observer = if (args.len == 8) .{ .ptr = &trace, .after = Trace.write } else null;
    var out = try tower.runObserved(runtime, s, x.ptr, height, width, observer);
    defer out.free();
    const output = try a.alloc(u8, out.len);
    try out.download(0, output);
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[5], .data = output });
    return 0;
}

fn imageMain(init: std.process.Init, args: []const []const u8) !u8 {
    if (args.len != 7 and args.len != 8) {
        std.debug.print("usage: tf-dsv4-vision --image MODEL IMAGE OUTPUT_BF16 KERNEL_DIR POSITION [LIBTURBOJPEG]\n", .{});
        return 2;
    }
    const a = init.arena.allocator();
    const config_path = try std.fs.path.join(a, &.{ args[2], "config.json" });
    const c = try Config.parse(a, try std.Io.Dir.cwd().readFileAlloc(init.io, config_path, a, .limited(1 << 20)));
    const ck = try Catalog.open(a, init.io, args[2]);
    var d = try cuda.Driver.open();
    defer d.close();
    var context = try cuda.Context.init(&d, 0);
    defer context.deinit();
    var s = try cuda.Stream.init(&d, true);
    defer s.deinit();
    var runtime = try vision.Runtime.open(init.gpa, init.io, &d, c, args[5]);
    defer runtime.close();
    var encoder = try media.Encoder.load(&ck, init.gpa, init.io, &d, c);
    defer encoder.free(init.gpa);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, args[3], a, .limited(32 << 20));
    var jpeg: ?picture.jpeg.Decoder = if (std.mem.startsWith(u8, bytes, "\xff\xd8\xff"))
        if (args.len == 8) try picture.jpeg.Decoder.openPath(args[7]) else try picture.jpeg.Decoder.open()
    else
        null;
    defer if (jpeg) |*decoder| decoder.close();
    const position = try std.fmt.parseInt(u32, args[6], 10);
    var image = try encoder.encode(init.gpa, runtime, s, bytes, if (jpeg) |*decoder| decoder else null, position, @import("config.zig").default_context);
    defer image.free(init.gpa);
    const output = try a.alloc(u8, image.embeddings.len);
    try image.embeddings.download(0, output);
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[4], .data = output });
    const json = try std.json.Stringify.valueAlloc(a, .{ .storage = c.storage.name(), .ids = image.ids, .span = image.key.span, .content_sha256 = std.fmt.bytesToHex(image.key.content, .lower), .preprocessing_sha256 = std.fmt.bytesToHex(image.key.preprocessing, .lower) }, .{});
    try std.Io.File.stdout().writeStreamingAll(init.io, json);
    return 0;
}
