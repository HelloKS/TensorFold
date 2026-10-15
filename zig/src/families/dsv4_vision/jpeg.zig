//! TurboJPEG's stable C API is loaded directly; image serving needs no Python process.
const std = @import("std");
const Rgb = @import("rgb.zig").Rgb;
const builtin = @import("builtin");
const Library = if (builtin.os.tag == .windows) WindowsLibrary else std.DynLib;

const WindowsLibrary = struct {
    handle: *anyopaque,
    extern "kernel32" fn LoadLibraryW([*:0]const u16) callconv(.winapi) ?*anyopaque;
    extern "kernel32" fn FreeLibrary(*anyopaque) callconv(.winapi) c_int;
    extern "kernel32" fn GetProcAddress(*anyopaque, [*:0]const u8) callconv(.winapi) ?*anyopaque;

    fn open(path: []const u8) !WindowsLibrary {
        var buffer: [32768]u16 = undefined;
        const n = try std.unicode.utf8ToUtf16Le(buffer[0 .. buffer.len - 1], path);
        buffer[n] = 0;
        return .{ .handle = LoadLibraryW(buffer[0..n :0]) orelse return error.JpegLibraryUnavailable };
    }

    fn lookup(lib: *WindowsLibrary, comptime T: type, name: [:0]const u8) ?T {
        return @ptrCast(GetProcAddress(lib.handle, name) orelse return null);
    }

    fn close(lib: *WindowsLibrary) void {
        _ = FreeLibrary(lib.handle);
    }
};

const Handle = ?*anyopaque;
const Api = struct {
    tjInitDecompress: *const fn () callconv(.c) Handle,
    tjDestroy: *const fn (Handle) callconv(.c) c_int,
    tjDecompressHeader3: *const fn (Handle, [*]const u8, c_ulong, *c_int, *c_int, *c_int, *c_int) callconv(.c) c_int,
    tjDecompress2: *const fn (Handle, [*]const u8, c_ulong, [*]u8, c_int, c_int, c_int, c_int, c_int) callconv(.c) c_int,
};

pub const Decoder = struct {
    library: Library,
    api: Api,

    pub fn open() !Decoder {
        return openPath(if (builtin.os.tag == .windows) "turbojpeg.dll" else "libturbojpeg.so.0");
    }

    pub fn openPath(path: []const u8) !Decoder {
        var lib = try Library.open(path);
        errdefer lib.close();
        var api: Api = undefined;
        const fields = @typeInfo(Api).@"struct";
        inline for (fields.field_names, fields.field_types) |name, T|
            @field(api, name) = lib.lookup(T, name) orelse return error.MissingJpegSymbol;
        return .{ .library = lib, .api = api };
    }

    pub fn close(d: *Decoder) void {
        d.library.close();
    }

    pub fn decode(d: *const Decoder, a: std.mem.Allocator, bytes: []const u8) !Rgb {
        if (bytes.len < 3 or !std.mem.eql(u8, bytes[0..3], "\xff\xd8\xff")) return error.NotJpeg;
        if (bytes.len > 32 << 20) return error.ImageTooLarge;
        const handle = d.api.tjInitDecompress() orelse return error.JpegInitFailed;
        defer _ = d.api.tjDestroy(handle);
        var width: c_int = 0;
        var height: c_int = 0;
        var subsamp: c_int = 0;
        var colorspace: c_int = 0;
        if (d.api.tjDecompressHeader3(handle, bytes.ptr, @intCast(bytes.len), &width, &height, &subsamp, &colorspace) != 0) return error.BadJpeg;
        if (width <= 0 or height <= 0 or width > 100000 or height > 100000) return error.BadImageDimensions;
        const w: usize = @intCast(width);
        const h: usize = @intCast(height);
        if (w * h > 32 * 1024 * 1024) return error.ImageTooLarge;
        const cmyk = colorspace == 3 or colorspace == 4;
        const pixels = try a.alloc(u8, w * h * (if (cmyk) @as(usize, 4) else 3));
        var owns_pixels = true;
        defer if (owns_pixels) a.free(pixels);
        const flags = 4096 + 8192;
        if (d.api.tjDecompress2(handle, bytes.ptr, @intCast(bytes.len), pixels.ptr, width, 0, height, if (cmyk) 11 else 0, flags) != 0) return error.BadJpeg;
        if (!cmyk) {
            owns_pixels = false;
            return .{ .w = w, .h = h, .px = pixels };
        }
        const out = try a.alloc(u8, w * h * 3);
        for (0..w * h) |i| for (0..3) |channel| {
            const product = @as(u32, pixels[i * 4 + channel]) * pixels[i * 4 + 3] + 128;
            out[i * 3 + channel] = @intCast((product + (product >> 8)) >> 8);
        };
        return .{ .w = w, .h = h, .px = out };
    }
};
