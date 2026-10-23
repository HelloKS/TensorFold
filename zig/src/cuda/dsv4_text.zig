//! Source-built text graph glue; each operation has explicit BF16/FP32 boundaries.
const std = @import("std");
const module = @import("module.zig");
const launch = @import("launch.zig");
const cuda = @import("root.zig");
pub const Ops = struct {
    mod: module.Module,
    pub fn load(d: *const cuda.Driver) !Ops {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        return loadImage(d, cuda.kernels.dsv4_text);
    }
    pub fn loadImage(d: *const cuda.Driver, image: []const u8) !Ops {
        return .{ .mod = try module.Module.load(d, image) };
    }
    pub fn unload(o: *Ops) void {
        o.mod.unload();
    }
    pub fn call(o: Ops, comptime name: []const u8, s: cuda.Stream, blocks: u32, threads: u32, values: anytype) !void {
        var args: launch.Args = .{};
        inline for (values) |v| args.add(v);
        try launch.launch(try o.mod.function("tf_dsv4_text_" ++ name), .{ .grid = .{ .x = blocks }, .block = .{ .x = threads } }, s, &args);
    }
};
