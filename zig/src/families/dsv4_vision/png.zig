// Copyright 2026 BertholomusAI (Albert Lee). Apache-2.0.

// Adapted from deepseek-v41-zig dfc8fe50; see THIRD_PARTY_NOTICES.md.

//! Native PNG decoder; pixel layout matches Pillow convert("RGB").
const std = @import("std");
const Allocator = std.mem.Allocator;
const Rgb = @import("rgb.zig").Rgb;
pub const Error = error{ NotPng, BadPng, UnsupportedPng, TooLarge, OutOfMemory };

const signature = "\x89PNG\r\n\x1a\n";

fn be32(b: []const u8) u32 {
    return std.mem.readInt(u32, b[0..4], .big);
}

pub const max_pixels: usize = 32 * 1024 * 1024;

pub fn pixels(data: []const u8) ?usize {
    if (data.len < 24 or !std.mem.eql(u8, data[0..8], "\x89PNG\r\n\x1a\n")) return null;
    return @as(usize, std.mem.readInt(u32, data[16..20], .big)) * std.mem.readInt(u32, data[20..24], .big);
}

pub fn decodePng(a: Allocator, data: []const u8) Error!Rgb {
    if (data.len < 8 or !std.mem.eql(u8, data[0..8], signature)) return error.NotPng;
    if (data.len > 32 << 20) return error.TooLarge;
    var at: usize = 8;
    var w: usize = 0;
    var h: usize = 0;
    var depth: u8 = 0;
    var ctype: u8 = 0;
    var interlace: u8 = 0;
    var palette: [256][3]u8 = @splat(.{ 0, 0, 0 });
    var have_ihdr = false;
    var idat: std.ArrayList(u8) = .empty;
    defer idat.deinit(a);
    while (true) {
        if (at + 12 > data.len) return error.BadPng;
        const len = be32(data[at..]);
        const kind = data[at + 4 .. at + 8];
        if (len > data.len - at - 12) return error.BadPng;
        const body = data[at + 8 .. at + 8 + len];
        const crc = be32(data[at + 8 + len ..]);
        var hc = std.hash.Crc32.init();
        hc.update(kind);
        hc.update(body);
        if (hc.final() != crc) return error.BadPng;
        at += 12 + len;
        if (std.mem.eql(u8, kind, "IHDR")) {
            if (len != 13 or have_ihdr or at != 33) return error.BadPng;
            w = be32(body[0..]);
            h = be32(body[4..]);
            if (w == 0 or h == 0 or w > 100000 or h > 100000 or w * h > max_pixels) return error.TooLarge;
            depth = body[8];
            ctype = body[9];
            if (body[10] != 0 or body[11] != 0 or body[12] > 1) return error.UnsupportedPng;
            interlace = body[12];
            have_ihdr = true;
        } else if (std.mem.eql(u8, kind, "PLTE")) {
            if (len % 3 != 0 or len > 768) return error.BadPng;
            for (0..len / 3) |i| palette[i] = .{ body[3 * i], body[3 * i + 1], body[3 * i + 2] };
        } else if (std.mem.eql(u8, kind, "IDAT")) {
            try idat.appendSlice(a, body);
        } else if (std.mem.eql(u8, kind, "IEND")) {
            break;
        }
    }
    if (!have_ihdr or w == 0 or h == 0) return error.BadPng;
    if (w * h > max_pixels) return error.TooLarge;
    const channels: usize = switch (ctype) {
        0 => 1,
        2 => 3,
        3 => 1,
        4 => 2,
        6 => 4,
        else => return error.UnsupportedPng,
    };
    const ok_depth = switch (ctype) {
        0 => depth == 1 or depth == 2 or depth == 4 or depth == 8 or depth == 16,
        3 => depth == 1 or depth == 2 or depth == 4 or depth == 8,
        else => depth == 8 or depth == 16,
    };
    if (!ok_depth) return error.UnsupportedPng;
    const bits = channels * depth;
    const bpp = @max(1, bits / 8);

    const raw_buffer = try a.alloc(u8, h * ((w * bits + 7) / 8 + 16) + 64);
    defer a.free(raw_buffer);
    var raw: std.Io.Writer = .fixed(raw_buffer);
    {
        var in: std.Io.Reader = .fixed(idat.items);
        var z: std.compress.flate.Decompress = .init(&in, .zlib, &.{});
        _ = z.reader.streamRemaining(&raw) catch return error.BadPng;
    }
    const zdata = raw.buffered();

    const samples = try a.alloc(u16, w * h * channels);
    defer a.free(samples);
    const passes: [7][4]usize = .{ .{ 0, 0, 8, 8 }, .{ 4, 0, 8, 8 }, .{ 0, 4, 4, 8 }, .{ 2, 0, 4, 4 }, .{ 0, 2, 2, 4 }, .{ 1, 0, 2, 2 }, .{ 0, 1, 1, 2 } };
    const one: [1][4]usize = .{.{ 0, 0, 1, 1 }};
    const list: []const [4]usize = if (interlace == 1) &passes else &one;
    var off: usize = 0;
    var prev: std.ArrayList(u8) = .empty;
    defer prev.deinit(a);
    var cur: std.ArrayList(u8) = .empty;
    defer cur.deinit(a);
    for (list) |ps| {
        const x0 = ps[0];
        const y0 = ps[1];
        const dx = ps[2];
        const dy = ps[3];
        if (x0 >= w or y0 >= h) continue;
        const pw = (w - x0 + dx - 1) / dx;
        const ph = (h - y0 + dy - 1) / dy;
        const stride = (pw * bits + 7) / 8;
        try prev.resize(a, stride);
        @memset(prev.items, 0);
        try cur.resize(a, stride);
        for (0..ph) |row| {
            if (off + 1 + stride > zdata.len) return error.BadPng;
            const filter = zdata[off];
            const line = zdata[off + 1 .. off + 1 + stride];
            off += 1 + stride;
            const c = cur.items;
            const p = prev.items;
            for (0..stride) |i| {
                const left: u8 = if (i >= bpp) c[i - bpp] else 0;
                const up = p[i];
                const ul: u8 = if (i >= bpp) p[i - bpp] else 0;
                c[i] = switch (filter) {
                    0 => line[i],
                    1 => line[i] +% left,
                    2 => line[i] +% up,
                    3 => line[i] +% @as(u8, @intCast((@as(u16, left) + up) / 2)),
                    4 => line[i] +% paeth(left, up, ul),
                    else => return error.BadPng,
                };
            }
            const y = y0 + row * dy;
            for (0..pw) |col| {
                const x = x0 + col * dx;
                for (0..channels) |ch| {
                    const k = col * channels + ch;
                    samples[(y * w + x) * channels + ch] = sampleAt(c, k, depth);
                }
            }
            std.mem.swap(std.ArrayList(u8), &prev, &cur);
        }
    }

    if (off != zdata.len) return error.BadPng;
    const px = try a.alloc(u8, w * h * 3);
    errdefer a.free(px);
    for (0..w * h) |i| {
        const s = samples[i * channels ..][0..channels];
        const rgb: [3]u8 = switch (ctype) {
            0 => blk: {
                const v = grey(s[0], depth);
                break :blk .{ v, v, v };
            },
            2, 6 => .{ hi(s[0], depth), hi(s[1], depth), hi(s[2], depth) },
            4 => blk: {
                const v = hi(s[0], depth);
                break :blk .{ v, v, v };
            },
            3 => palette[@min(s[0], 255)],
            else => unreachable,
        };
        px[i * 3 ..][0..3].* = rgb;
    }
    return .{ .w = w, .h = h, .px = px };
}

fn paeth(a: u8, b: u8, c: u8) u8 {
    const p: i16 = @as(i16, a) + b - c;
    const pa = @abs(p - a);
    const pb = @abs(p - b);
    const pc = @abs(p - c);
    if (pa <= pb and pa <= pc) return a;
    if (pb <= pc) return b;
    return c;
}

fn sampleAt(row: []const u8, k: usize, depth: u8) u16 {
    return switch (depth) {
        16 => (@as(u16, row[2 * k]) << 8) | row[2 * k + 1],
        8 => row[k],
        else => blk: {
            const per = 8 / depth;
            const byte = row[k / per];
            const shift: u3 = @intCast(8 - depth * (k % per + 1));
            const mask: u8 = @intCast((@as(u16, 1) << @intCast(depth)) - 1);
            break :blk (byte >> shift) & mask;
        },
    };
}

fn hi(v: u16, depth: u8) u8 {
    return if (depth == 16) @intCast(v >> 8) else @intCast(v);
}

fn grey(v: u16, depth: u8) u8 {
    return switch (depth) {
        1 => if (v != 0) 255 else 0,
        2 => @intCast(v * 85),
        4 => @intCast(v * 17),
        8 => @intCast(v),
        else => @intCast(@min(v, 255)),
    };
}
