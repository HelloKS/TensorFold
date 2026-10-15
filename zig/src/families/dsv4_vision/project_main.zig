//! Run one real checkpoint projection natively on CUDA, for independent per-layer qualification.
const std = @import("std");
const cuda = @import("cuda");
const config = @import("config.zig");
const Catalog = @import("catalog.zig").Catalog;
const gpu = @import("gpu_weights.zig");

fn image(a: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8) ![]align(16) u8 {
    const path = try std.fs.path.join(a, &.{ dir, name });
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(64 << 20));
    const out = try a.alignedAlloc(u8, .@"16", bytes.len);
    @memcpy(out, bytes);
    return out;
}

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 9) {
        std.debug.print("usage: tf-dsv4-project MODEL TENSOR K N INPUT_BF16 OUTPUT_F32 ROWS KERNEL_DIR\n", .{});
        return 2;
    }
    const k = try std.fmt.parseInt(usize, args[3], 10);
    const n = try std.fmt.parseInt(usize, args[4], 10);
    const rows = try std.fmt.parseInt(u32, args[7], 10);
    if (k == 0 or n == 0 or rows == 0 or rows > 2048) return error.BadShape;
    const cfg = try std.Io.Dir.cwd().readFileAlloc(init.io, try std.fs.path.join(a, &.{ args[1], "config.json" }), a, .limited(1 << 20));
    const c = try config.Config.parse(a, cfg);
    const ck = try Catalog.open(a, init.io, args[1]);
    var driver = try cuda.Driver.open();
    defer driver.close();
    var context = try cuda.Context.init(&driver, 0);
    defer context.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    var exl3: ?cuda.dsv4_exl3.Projection = if (c.storage == .exl3)
        try cuda.dsv4_exl3.Projection.loadImage(&driver, try image(a, init.io, args[8], "exl3.fatbin"))
    else
        null;
    defer if (exl3) |*ops| ops.unload();
    var native: ?cuda.dsv4_native.Ops = if (c.storage == .native_fp4_fp8)
        try cuda.dsv4_native.Ops.loadImage(&driver, try image(a, init.io, args[8], "dsv4_native.fatbin"))
    else
        null;
    defer if (native) |*ops| ops.unload();
    var tensor = try cuda.dsv4_tensor.Ops.loadImage(&driver, try image(a, init.io, args[8], "dsv4_tensor.fatbin"));
    defer tensor.unload();
    var matrix = try gpu.Matrix.load(&ck, init.gpa, init.io, &driver, args[2], k, n, c.storage);
    defer matrix.free();
    const bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, args[5], a, .limited(128 << 20));
    if (bytes.len != @as(usize, rows) * k * 2) return error.BadInputSize;
    var x = try cuda.DeviceBuffer.fromHost(&driver, bytes);
    defer x.free();
    var y = try cuda.DeviceBuffer.alloc(&driver, @as(usize, rows) * n * 4);
    defer y.free();
    var scratch: gpu.Scratch = .{};
    defer scratch.free();
    try matrix.run(exl3, native, tensor, stream, &scratch, x.ptr, y.ptr, rows, .bf16, .f32);
    try stream.synchronize();
    const output = try a.alloc(u8, y.len);
    try y.download(0, output);
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[6], .data = output });
    return 0;
}
