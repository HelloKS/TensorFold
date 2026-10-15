//! Image resize geometry and N-layout, reimplemented from DeepSeek's MIT image_processor.py.
const std = @import("std");
const Config = @import("config.zig").Config;

pub const Kind = enum(u8) { start, pad, image, newline, end };
pub const Geometry = struct { height: usize, width: usize, rows: usize, cols: usize, tokens: usize };

fn ceil(n: usize, d: usize) usize {
    return n / d + @intFromBool(n % d != 0);
}

pub fn grid(height: usize, width: usize, patch: usize, down: usize) Geometry {
    const h = ceil(height / patch, down);
    const w = ceil(width / patch, down);
    const even_h = h + h % 2;
    return .{
        .height = height,
        .width = width,
        .rows = h,
        .cols = w,
        .tokens = even_h * (w + 1) + 2 + (even_h / 2 * (w + 1)) % 2 * 2,
    };
}

pub fn geometry(c: Config, height: usize, width: usize) !Geometry {
    if (height == 0 or width == 0 or height > 100000 or width > 100000) return error.BadImageDimensions;
    var h = height;
    var w = @min(width, height * c.vision_max_wh_ratio);
    if (h * w < c.vision_min_pixels) {
        const scale = @sqrt(@as(f64, @floatFromInt(c.vision_min_pixels)) / @as(f64, @floatFromInt(h * w)));
        w = @intFromFloat(@as(f64, @floatFromInt(w)) * scale);
        h = @intFromFloat(@as(f64, @floatFromInt(h)) * scale);
    }
    const p = c.vision_patch_size;
    const d = c.vision_downsample_ratio;
    var g = grid(ceil(h, p) * p, ceil(w, p) * p, p, d);
    const limit = c.vision_max_n_token - 3;
    var budget = limit;
    while (g.tokens > limit) : (budget -= 1) {
        if (budget < 10) return error.ImageBudgetTooSmall;
        const ratio = @as(f64, @floatFromInt(h)) / @as(f64, @floatFromInt(w));
        const max_wf = @sqrt(@as(f64, @floatFromInt(budget - 2)) / ratio + 0.25) - 0.5;
        const max_hf = max_wf * ratio;
        var best_w: usize = undefined;
        var best_h: usize = undefined;
        if (max_wf < 1) {
            const mh = (budget - 2) / 2;
            best_w = p * d;
            best_h = (mh - mh % 2) * p * d;
        } else if (max_hf < 2) {
            best_h = 2 * p * d;
            best_w = ((budget - 2) / 2 - 1) * p * d;
        } else {
            const mw: usize = @intFromFloat(@floor(max_wf));
            var mh: usize = @intFromFloat(@floor(max_hf));
            mh -= mh % 2;
            const beta = @min(@as(f64, @floatFromInt(mw * p * d)) / @as(f64, @floatFromInt(w)), @as(f64, @floatFromInt(mh * p * d)) / @as(f64, @floatFromInt(h)));
            best_w = @as(usize, @intFromFloat(@floor(@as(f64, @floatFromInt(w)) * beta /
                @as(f64, @floatFromInt(p))))) * p;
            best_h = @as(usize, @intFromFloat(@floor(@as(f64, @floatFromInt(h)) * beta /
                @as(f64, @floatFromInt(p))))) * p;
        }
        g = grid(best_h, best_w, p, d);
    }
    return g;
}

pub const Block = struct {
    types: []Kind,
    permutation: []u32,

    pub fn deinit(b: Block, a: std.mem.Allocator) void {
        a.free(b.types);
        a.free(b.permutation);
    }

    pub fn tokens(b: Block, a: std.mem.Allocator, vocab: u32) ![]u32 {
        const out = try a.alloc(u32, b.types.len);
        for (out, b.types) |*id, kind| id.* = vocab + @as(u32, @backingInt(kind));
        return out;
    }
};

pub fn block(a: std.mem.Allocator, g: Geometry, position: usize) !Block {
    if (g.rows == 0 or g.cols == 0 or g.tokens > 381) return error.BadImageGrid;
    const leading = 3 - position % 4;
    const types = try a.alloc(Kind, leading + g.tokens);
    errdefer a.free(types);
    const perm = try a.alloc(u32, g.rows * g.cols);
    errdefer a.free(perm);
    @memset(types, .pad);
    types[leading] = .start;
    var at = leading + 1;
    var pi: usize = 0;
    const even_h = g.rows + g.rows % 2;
    for (0..even_h / 2) |pair| {
        for (0..g.cols + 1) |col| {
            for (0..2) |offset| {
                const row = pair * 2 + offset;
                types[at] = if (row >= g.rows) .pad else if (col == g.cols) .newline else .image;
                if (types[at] == .image) {
                    perm[pi] = @intCast(row * g.cols + col);
                    pi += 1;
                }
                at += 1;
            }
        }
    }
    types[types.len - 1] = .end;
    return .{ .types = types, .permutation = perm };
}

pub const Span = struct {
    start: u32,
    end: u32,

    pub fn cuts(s: Span, position: u32) bool {
        return position > s.start and position < s.end;
    }
};

pub fn chunkEnd(start: u32, proposed: u32, prompt_len: u32, spans: []const Span) !u32 {
    if (proposed <= start or proposed > prompt_len) return error.BadChunk;
    var end = proposed;
    var previous: u32 = 0;
    for (spans) |s| {
        if (s.start < previous or s.end <= s.start or s.end > prompt_len or s.cuts(start)) return error.BadImageSpans;
        if (s.cuts(end)) end = if (s.start > start) s.start else s.end;
        previous = s.end;
    }
    return end;
}

pub fn window(position: u32, prompt_len: u32, image: ?Span, width: u32) !Span {
    if (position >= prompt_len or width == 0) return error.BadPosition;
    var left: u32 = 0;
    var right: u32 = 0;
    if (image) |s| {
        if (s.start > position or s.end <= position or s.end > prompt_len) return error.BadImageSpans;
        left = @min(position - s.start, 383);
        right = @min(s.end - 1 - position, 384);
    }
    const extra = left -| (width - 1);
    return .{ .start = position -| (width - 1 + extra), .end = position + right + 1 };
}

test "odd-height N-layout has the official ordering and compression alignment" {
    const g = grid(126, 126, 14, 3);
    try std.testing.expectEqual(@as(usize, 3), g.rows);
    const b = try block(std.testing.allocator, g, 0);
    defer b.deinit(std.testing.allocator);
    try std.testing.expectEqualSlices(u32, &.{ 0, 3, 1, 4, 2, 5, 6, 7, 8 }, b.permutation);
    try std.testing.expectEqual(Kind.start, b.types[3]);
    try std.testing.expectEqual(Kind.end, b.types[b.types.len - 1]);
    try std.testing.expectEqual(@as(usize, 1), b.types.len % 4);
}

test "resize stays within the image budget at extreme aspect ratios" {
    const c = try Config.parse(std.testing.allocator, @import("config.zig").fixture);
    for ([_][2]usize{ .{ 1, 1 }, .{ 10000, 1 }, .{ 1, 10000 }, .{ 1080, 1920 }, .{ 100000, 100000 } }) |shape| {
        const g = try geometry(c, shape[0], shape[1]);
        try std.testing.expect(g.height % 14 == 0 and g.width % 14 == 0);
        for (0..4) |offset| {
            const b = try block(std.testing.allocator, g, offset);
            defer b.deinit(std.testing.allocator);
            try std.testing.expect(b.types.len <= 384);
            try std.testing.expectEqual(@as(usize, 1), (offset + b.types.len) % 4);
        }
    }
    try std.testing.expectError(error.BadImageDimensions, geometry(c, 0, 2));
}

test "chunk and attention boundaries preserve image spans" {
    const spans = [_]Span{.{ .start = 120, .end = 400 }};
    try std.testing.expectEqual(@as(u32, 120), try chunkEnd(0, 256, 1000, &spans));
    try std.testing.expectEqual(@as(u32, 400), try chunkEnd(120, 256, 1000, &spans));
    try std.testing.expectError(error.BadImageSpans, chunkEnd(121, 256, 1000, &spans));
    const seen = try window(140, 1000, spans[0], 128);
    try std.testing.expectEqual(@as(u32, 13), seen.start);
    try std.testing.expectEqual(@as(u32, 400), seen.end);
    const late = try window(300, 1000, spans[0], 128);
    try std.testing.expectEqual(@as(u32, 120), late.start);
}
