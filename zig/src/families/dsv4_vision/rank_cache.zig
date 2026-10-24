//! A rank cache is accepted only with its exact config, schema and payload digest; no missing weights are fetched.
const std = @import("std");
const catalog = @import("catalog.zig");
const config = @import("config.zig");
const weights = @import("weights.zig");
const parallel = @import("parallel.zig");
const Sha = std.crypto.hash.sha2.Sha256;

pub const Cache = struct {
    tensors: catalog.Catalog,
    identity: [32]u8,
    rank: u32,

    pub fn open(a: std.mem.Allocator, io: std.Io, dir: []const u8, rank: u32, config_text: []const u8, problem: *[]const u8) !Cache {
        if (rank > 1) return error.BadRank;
        const c = try config.Config.parse(a, config_text);
        const path = try std.fs.path.join(a, &.{ dir, "manifest.json" });
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 20));
        const doc = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
        const identity_text = try string(try field(doc, "identity"));
        var identity_sha: [32]u8 = undefined;
        Sha.hash(identity_text, &identity_sha, .{});
        if (!std.mem.eql(u8, &identity_sha, &try digest(try field(doc, "identity_sha256"))))
            return error.CacheIdentityMismatch;
        const identity = try std.json.parseFromSliceLeaky(std.json.Value, a, identity_text, .{});
        if (try integer(try field(identity, "format_version")) != 2 or
            try integer(try field(identity, "tp")) != 2 or
            !std.mem.eql(u8, try string(try field(identity, "model_revision")), config.model_revision))
            return error.IncompatibleRankCache;
        if (!std.mem.eql(u8, try string(try field(identity, "storage")), c.storage.name()) or
            try integer(try field(identity, "partition_version")) != 2) return error.IncompatibleRankCache;
        if (!std.mem.eql(u8, &@import("cache_abi.zig").digest(), &try digest(try field(identity, "cache_code_sha256"))))
            return error.CacheCodeChanged;
        var config_sha: [32]u8 = undefined;
        Sha.hash(config_text, &config_sha, .{});
        const sources = try field(identity, "sources");
        if (!std.mem.eql(u8, &config_sha, &try digest(try field(sources, "config.json"))))
            return error.CacheConfigChanged;
        const ranks = try field(doc, "ranks");
        if (ranks != .array or ranks.array.items.len != 2) return error.BadCacheManifest;
        const entry = ranks.array.items[rank];
        const name = try string(try field(entry, "file"));
        if (!catalog.safeName(name)) return error.BadShardName;
        const rank_path = try std.fs.path.join(a, &.{ dir, name });
        const expected = try digest(try field(entry, "sha256"));
        const expected_bytes = try integer(try field(entry, "bytes"));
        const skip = skipPayloadHash(if (std.c.getenv("TF_DSV4_SKIP_PAYLOAD_HASH")) |v| std.mem.span(v) else null);
        if (skip) std.log.warn("rank {d}: payload SHA256 skipped; cache corruption is unchecked", .{rank});
        try verifyPayload(io, rank_path, expected_bytes, expected, skip);
        var tensors: catalog.Catalog = .{};
        try tensors.addFile(a, io, rank_path);
        var local: std.ArrayList(weights.Spec) = .empty;
        for (try weights.requirements(a, c)) |spec| {
            const part = try parallel.partition(c, spec, rank);
            if (!part.keep) continue;
            var selected = spec;
            if (part.axis) |axis| selected.shape[axis] = part.count;
            try local.append(a, selected);
        }
        _ = try weights.validateSpecs(a, c, local.items, tensors.headers, problem);
        return .{ .tensors = tensors, .identity = try digest(try field(doc, "identity_sha256")), .rank = rank };
    }
};

fn field(value: std.json.Value, name: []const u8) !std.json.Value {
    if (value != .object) return error.BadCacheManifest;
    return value.object.get(name) orelse return error.BadCacheManifest;
}

fn string(value: std.json.Value) ![]const u8 {
    if (value != .string) return error.BadCacheManifest;
    return value.string;
}

fn integer(value: std.json.Value) !u64 {
    if (value != .integer or value.integer < 0) return error.BadCacheManifest;
    return @intCast(value.integer);
}

fn digest(value: std.json.Value) ![32]u8 {
    const hex = try string(value);
    if (hex.len != 64) return error.BadCacheManifest;
    var out: [32]u8 = undefined;
    _ = std.fmt.hexToBytes(&out, hex) catch return error.BadCacheManifest;
    return out;
}

fn skipPayloadHash(value: ?[]const u8) bool {
    return std.mem.eql(u8, value orelse "0", "1");
}

fn verifyPayload(io: std.Io, path: []const u8, expected_bytes: u64, expected: [32]u8, skip: bool) !void {
    var file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const len = try file.length(io);
    if (len != expected_bytes) return error.CacheSizeMismatch;
    if (skip) return;
    var hash: Sha = .init(.{});
    var buffer: [64 * 1024]u8 = undefined;
    var at: u64 = 0;
    while (at < len) {
        const n: usize = @intCast(@min(buffer.len, len - at));
        if (try file.readPositionalAll(io, buffer[0..n], at) != n) return error.TruncatedTensor;
        hash.update(buffer[0..n]);
        at += n;
    }
    if (!std.mem.eql(u8, &hash.finalResult(), &expected)) return error.CacheDigestMismatch;
}

test "payload hashing defaults on and skipping still rejects a truncated cache" {
    const t = std.testing;
    for ([_]?[]const u8{ null, "", "0", "true", "01" }) |value| try t.expect(!skipPayloadHash(value));
    try t.expect(skipPayloadHash("1"));
    var temp = t.tmpDir(.{});
    defer temp.cleanup();
    try temp.dir.writeFile(t.io, .{ .sub_path = "rank.safetensors", .data = "original" });
    var buf: [256]u8 = undefined;
    const path = try std.fmt.bufPrint(&buf, ".zig-cache/tmp/{s}/rank.safetensors", .{temp.sub_path});
    var expected: [32]u8 = undefined;
    Sha.hash("original", &expected, .{});
    try verifyPayload(t.io, path, 8, expected, false);
    try temp.dir.writeFile(t.io, .{ .sub_path = "rank.safetensors", .data = "modified" });
    try t.expectError(error.CacheDigestMismatch, verifyPayload(t.io, path, 8, expected, false));
    try verifyPayload(t.io, path, 8, expected, true);
    try t.expectError(error.CacheSizeMismatch, verifyPayload(t.io, path, 9, expected, true));
    try t.expectError(error.CacheSizeMismatch, verifyPayload(t.io, path, 7, expected, false));
}

test "cache digests and manifests reject malformed identities" {
    try std.testing.expectError(error.BadCacheManifest, digest(.{ .string = "abc" }));
    try std.testing.expectError(error.BadCacheManifest, integer(.{ .integer = -1 }));
    try std.testing.expectError(error.BadCacheManifest, field(.null, "ranks"));
    var problem: []const u8 = "";
    try std.testing.expectError(error.BadRank, Cache.open(std.testing.allocator, std.testing.io, ".", 2, config.fixture, &problem));
}

test "cache identity tampering and a different storage format are refused before GPU loading" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var temp = std.testing.tmpDir(.{});
    defer temp.cleanup();
    const path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{temp.sub_path});
    const identity = try std.json.Stringify.valueAlloc(a, .{ .format_version = 2, .tp = 2, .model_revision = config.model_revision, .storage = "exl3", .partition_version = 2 }, .{});
    var sha: [32]u8 = undefined;
    Sha.hash(identity, &sha, .{});
    const doc = try std.json.Stringify.valueAlloc(a, .{ .identity = identity, .identity_sha256 = std.fmt.bytesToHex(sha, .lower) }, .{});
    try temp.dir.writeFile(io, .{ .sub_path = "manifest.json", .data = doc });
    var problem: []const u8 = "";
    try std.testing.expectError(error.IncompatibleRankCache, Cache.open(a, io, path, 0, config.native_fixture, &problem));
    const bad = try std.mem.replaceOwned(u8, a, doc, "exl3", "fp16");
    try temp.dir.writeFile(io, .{ .sub_path = "manifest.json", .data = bad });
    try std.testing.expectError(error.CacheIdentityMismatch, Cache.open(a, io, path, 0, config.native_fixture, &problem));
}
