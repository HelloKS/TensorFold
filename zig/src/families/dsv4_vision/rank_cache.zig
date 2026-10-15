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
        const identity = try field(doc, "identity");
        if (try integer(try field(identity, "format_version")) != 1 or
            try integer(try field(identity, "tp")) != 2 or
            !std.mem.eql(u8, try string(try field(identity, "model_revision")), config.model_revision))
            return error.IncompatibleRankCache;
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
        if (!std.mem.eql(u8, &try fileDigest(io, rank_path, expected_bytes), &expected)) return error.CacheDigestMismatch;
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

fn fileDigest(io: std.Io, path: []const u8, expected_bytes: u64) ![32]u8 {
    var file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const len = try file.length(io);
    if (len != expected_bytes) return error.CacheSizeMismatch;
    var hash: Sha = .init(.{});
    var buffer: [64 * 1024]u8 = undefined;
    var at: u64 = 0;
    while (at < len) {
        const n: usize = @intCast(@min(buffer.len, len - at));
        if (try file.readPositionalAll(io, buffer[0..n], at) != n) return error.TruncatedTensor;
        hash.update(buffer[0..n]);
        at += n;
    }
    return hash.finalResult();
}

test "cache digests and manifests reject malformed identities" {
    try std.testing.expectError(error.BadCacheManifest, digest(.{ .string = "abc" }));
    try std.testing.expectError(error.BadCacheManifest, integer(.{ .integer = -1 }));
    try std.testing.expectError(error.BadCacheManifest, field(.null, "ranks"));
    var problem: []const u8 = "";
    try std.testing.expectError(error.BadRank, Cache.open(std.testing.allocator, std.testing.io, ".", 2, config.fixture, &problem));
}
