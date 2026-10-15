//! Development probes and the model share the same source-built CUDA module loader.
const std = @import("std");
const cuda = @import("cuda");

pub fn load(comptime T: type, a: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: ?[]const u8, name: []const u8) !T {
    const directory = dir orelse return T.load(d);
    const path = try std.fs.path.join(a, &.{ directory, name });
    defer a.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(64 << 20));
    defer a.free(bytes);
    const aligned = try a.alignedAlloc(u8, .@"16", bytes.len);
    defer a.free(aligned);
    @memcpy(aligned, bytes);
    return T.loadImage(d, aligned);
}
