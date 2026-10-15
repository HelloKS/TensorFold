//! CUDA's shared-library loader, including Windows for development-time device probes.
const std = @import("std");
pub const Library = if (@import("builtin").os.tag == .windows) Windows else std.DynLib;

const Windows = struct {
    handle: *anyopaque,
    extern "kernel32" fn LoadLibraryW([*:0]const u16) callconv(.winapi) ?*anyopaque;
    extern "kernel32" fn FreeLibrary(*anyopaque) callconv(.winapi) c_int;
    extern "kernel32" fn GetProcAddress(*anyopaque, [*:0]const u8) callconv(.winapi) ?*anyopaque;

    pub fn open(path: []const u8) !Windows {
        var buffer: [32768]u16 = undefined;
        const n = try std.unicode.utf8ToUtf16Le(buffer[0 .. buffer.len - 1], path);
        buffer[n] = 0;
        return .{ .handle = LoadLibraryW(buffer[0..n :0]) orelse return error.DriverUnavailable };
    }

    pub fn lookup(lib: *Windows, comptime T: type, name: [:0]const u8) ?T {
        return @ptrCast(GetProcAddress(lib.handle, name) orelse return null);
    }

    pub fn close(lib: *Windows) void {
        _ = FreeLibrary(lib.handle);
    }
};
