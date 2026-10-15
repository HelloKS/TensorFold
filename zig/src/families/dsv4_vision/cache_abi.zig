//! Weight caches are invalidated when the storage, splitting or native loading source changes.
const std = @import("std");
const Sha = std.crypto.hash.sha2.Sha256;

fn update(hash: *Sha, source: []const u8) void {
    for (source, 0..) |byte, i| {
        if (byte == '\r' and i + 1 < source.len and source[i + 1] == '\n') continue;
        hash.update(&.{byte});
    }
}

pub fn digest() [32]u8 {
    var hash: Sha = .init(.{});
    hash.update("TensorFold/dsv4/cache-code/v2\x00");
    inline for (.{ "config.zig", "storage.zig", "weights.zig", "parallel.zig", "payloads.zig", "gpu_weights.zig", "rank_cache.zig", "cache_abi.zig" }) |file| {
        hash.update(file ++ "\x00");
        update(&hash, @embedFile(file));
        hash.update("\x00");
    }
    return hash.finalResult();
}

test "cache source fingerprints normalize Git line endings but preserve code changes" {
    var a: Sha = .init(.{});
    var b: Sha = .init(.{});
    update(&a, "first\r\nsecond\r\n");
    update(&b, "first\nsecond\n");
    const ad = a.finalResult();
    const bd = b.finalResult();
    try std.testing.expectEqualSlices(u8, &ad, &bd);
    var c: Sha = .init(.{});
    update(&c, "first\nchanged\n");
    try std.testing.expect(!std.mem.eql(u8, &ad, &c.finalResult()));
}
