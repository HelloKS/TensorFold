//! CPU image preparation for HTTP and rank-zero tower execution on the engine thread.
const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const layout = @import("image_layout.zig");
const picture = @import("picture.zig");
const media = @import("media.zig");
const wire = @import("control.zig");
pub const Limits = struct { image_tokens: u32 = 1536, max_images: u32 = 4 };
pub const Prepared = struct { pixels: []f32, gh: u32, gw: u32, tokens: u32, hash: u64 };
pub const Span = struct {
    at: u32,
    start: u32,
    end: u32,
    embeddings: cuda.DeviceBuffer,
    pub fn free(s: *Span) void { s.embeddings.free(); }
};
pub const Vision = struct {
    c: Config,
    encoder: media.Encoder,
    marks: struct { image: u32, open: u32 = 0, close: u32 = 0 },
    mark_ids: [1]u32,
    limits: Limits,
    cache: @import("image_cache.zig").Cache = .{},
    pub const template_wraps = true;
    pub fn open(a: std.mem.Allocator, io: std.Io, m: anytype, dir: []const u8, cache: []const u8, limits: Limits) !Vision {
        if (limits.image_tokens == 0 or limits.image_tokens > 1536 or limits.max_images == 0 or limits.max_images > 4) return error.UnsupportedVisionLimits;
        var arena: std.heap.ArenaAllocator = .init(a);
        defer arena.deinit();
        const temp = arena.allocator();
        const cfg = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(temp, &.{dir, "config.json"}), temp, .limited(1 << 20));
        var problem: []const u8 = "";
        const local = try @import("rank_cache.zig").Cache.open(temp, io, cache, 0, cfg, &problem);
        var encoder = try media.Encoder.load(&local.tensors, a, io, m.d, m.c);
        errdefer encoder.free(a);
        var tok = try @import("core").tokenizer.loadTokenizer(io, a, try std.fs.path.join(temp, &.{dir, "tokenizer.json"}));
        defer tok.deinit();
        const id = tok.vocab.get("<｜deepseek_image｜>") orelse return error.MissingImageToken;
        return .{ .c = m.c, .encoder = encoder, .marks = .{ .image = id }, .mark_ids = .{id}, .limits = limits };
    }
    pub fn free(v: *Vision, a: std.mem.Allocator) void { v.cache.free(a); v.encoder.free(a); }
    pub fn prepare(v: *Vision, a: std.mem.Allocator, bytes: []const u8, max_tokens: u32) !Prepared {
        if (!std.mem.startsWith(u8, bytes, "\x89PNG") and !std.mem.startsWith(u8, bytes, "\xff\xd8")) return error.UnsupportedImageFormat;
        var c = v.c;
        c.vision_max_n_token = @min(c.vision_max_n_token, max_tokens);
        if (c.vision_max_n_token < 16) return error.ImageBudgetTooSmall;
        var jpg = if (std.mem.startsWith(u8, bytes, "\xff\xd8")) try picture.jpeg.Decoder.open() else null;
        defer if (jpg) |*decoder| decoder.close();
        const rgb = try picture.decode(a, bytes, c, if (jpg) |*decoder| decoder else null);
        defer rgb.deinit(a);
        const pixels = try a.alloc(f32, rgb.patches.len);
        errdefer a.free(pixels);
        for (pixels, rgb.patches) |*value, bits| value.* = @bitCast(@as(u32, bits) << 16);
        var hash = std.crypto.hash.sha2.Sha256.init(.{});
        hash.update(bytes);
        hash.update(&media.preprocessingKey(c));
        const digest = hash.finalResult();
        return .{ .pixels = pixels, .gh = @intCast(rgb.grid.height / c.vision_patch_size), .gw = @intCast(rgb.grid.width / c.vision_patch_size),
            .tokens = @intCast(rgb.grid.tokens + 3), .hash = std.mem.readInt(u64, digest[0..8], .little) };
    }
    pub fn position(v: *Vision, a: std.mem.Allocator, gh: u32, gw: u32, at: u32) ![]u32 {
        const g = layout.grid(@as(usize, gh) * v.c.vision_patch_size, @as(usize, gw) * v.c.vision_patch_size, v.c.vision_patch_size, v.c.vision_downsample_ratio);
        const block = try layout.block(a, g, at);
        defer block.deinit(a);
        return block.tokens(a, @intCast(v.c.vocab_size));
    }
    pub fn explain(err: anyerror) ?[]const u8 {
        return switch (err) { error.UnsupportedImageFormat => "DeepSeek image input supports PNG and JPEG", error.JpegLibraryUnavailable, error.FileNotFound => "native JPEG input requires libturbojpeg", else => null };
    }
    pub fn install(v: *Vision, m: anytype, leader: anytype, slot: u32, img: anytype) !void {
        const g = layout.grid(@as(usize, img.gh) * v.c.vision_patch_size, @as(usize, img.gw) * v.c.vision_patch_size, v.c.vision_patch_size, v.c.vision_downsample_ratio);
        const block = try layout.block(m.a, g, img.at);
        defer block.deinit(m.a);
        if (block.types.len != img.tokens or img.at + img.tokens > m.context or img.pixels.len != @as(usize, img.gh) * img.gw * 3 * v.c.vision_patch_size * v.c.vision_patch_size) return error.InvalidImageSpan;
        const bf16 = try m.a.alloc(u16, img.pixels.len);
        defer m.a.free(bf16);
        for (bf16, img.pixels) |*bits, value| { if (!std.math.isFinite(value)) return error.InvalidImagePixels; bits.* = picture.rgb.bf16(value); }
        var patches = try cuda.DeviceBuffer.fromHost(m.d, std.mem.sliceAsBytes(bf16));
        defer patches.free();
        const raster_key = @import("image_cache.zig").key(m.identity, v.encoder.preprocessing, img.pixels, img.gh, img.gw);
        const cached = v.cache.get(raster_key);
        var raster = cached orelse try v.encoder.tower.run(m.r.operators, m.stream, patches.ptr, img.gh, img.gw);
        defer raster.free();
        if (cached == null) try v.cache.put(m.a, raster_key, raster, m.stream);
        const map = try m.a.alloc(i32, img.tokens);
        defer m.a.free(map);
        try media.mapRows(block, map);
        var mapping = try cuda.DeviceBuffer.fromHost(m.d, std.mem.sliceAsBytes(map));
        defer mapping.free();
        var rows = try cuda.DeviceBuffer.alloc(m.d, @as(usize, img.tokens) * v.c.hidden_size * 2);
        defer rows.free();
        defer m.sync() catch {};
        try m.r.operators.vision.span(m.stream, raster.ptr, v.encoder.special.ptr, mapping.ptr, rows.ptr, img.tokens, @intCast(v.c.hidden_size));
        try m.sync();
        const host = try m.a.alloc(u8, rows.len);
        defer m.a.free(host);
        try rows.download(0, host);
        const command = try wire.encode(m.a, .{ .op = .media, .slot = slot, .id = img.tokens, .at = img.at, .data = host });
        defer m.a.free(command);
        try leader.command(command);
    }
};
pub fn receive(m: anytype, c: wire.Command) !void {
    if (c.id == 0 or c.id > m.c.vision_max_n_token or c.at + c.id > m.context or c.data.len != c.id * m.c.hidden_size * 2) return error.InvalidImageSpan;
    const got = try m.media.getOrPut(m.a, c.slot);
    if (!got.found_existing) got.value_ptr.* = .empty;
    if (got.value_ptr.items.len >= 4) return error.TooManyImages;
    if (got.value_ptr.items.len > 0 and got.value_ptr.items[got.value_ptr.items.len - 1].end > c.at) return error.InvalidImageSpan;
    var rows = try cuda.DeviceBuffer.fromHost(m.d, c.data);
    errdefer rows.free();
    try got.value_ptr.append(m.a, .{ .at = c.at, .start = c.at + 3 - c.at % 4, .end = @intCast(c.at + c.id), .embeddings = rows });
}
