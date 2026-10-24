//! TP2 partition rules: expert intermediates and attention groups are split at complete Hadamard blocks.
const std = @import("std");
const Spec = @import("weights.zig").Spec;
const Config = @import("config.zig").Config;

pub const Slice = struct {
    keep: bool = true,
    axis: ?u8 = null,
    first: usize = 0,
    count: usize = 0,
};

fn ends(key: []const u8, suffix: []const u8) bool {
    return std.mem.endsWith(u8, key, suffix);
}

pub fn partition(c: Config, spec: Spec, rank: u32) !Slice {
    if (rank >= 2) return error.BadRank;
    const key = spec.key;
    if (std.mem.startsWith(u8, key, "vision.") or std.mem.startsWith(u8, key, "aligner.") or
        std.mem.startsWith(u8, key, "image_")) return .{ .keep = rank == 0 };
    if (std.mem.indexOf(u8, key, ".attn.wo_a.slice.")) |at| {
        const group = try std.fmt.parseInt(usize, key[at + ".attn.wo_a.slice.".len ..], 10);
        if (group >= c.o_groups) return error.BadGroup;
        return .{ .keep = group / (c.o_groups / 2) == rank };
    }
    const expert = std.mem.indexOf(u8, key, ".ffn.experts.") != null or
        std.mem.indexOf(u8, key, ".ffn.shared_experts.") != null;
    const output = spec.kind == .matrix and (std.mem.eql(u8, key, "head") or
        ends(key, ".attn.wo_a") or ends(key, ".attn.wq_b") or ends(key, ".attn.indexer.wq_b") or
        ends(key, ".markov_head.markov_w2") or (expert and (ends(key, ".w1") or ends(key, ".w3"))));
    const input = spec.kind == .matrix and (ends(key, ".attn.wo_b") or (expert and ends(key, ".w2")));
    const rows = spec.kind != .matrix and (ends(key, ".attn.attn_sink") or
        ends(key, ".attn.indexer.weights_proj.weight"));
    if (!output and !input and !rows) return .{};
    const axis: u8 = if (output) 1 else 0;
    const dimension = spec.shape[axis];
    if (dimension % 2 != 0 or (spec.kind == .matrix and dimension % 256 != 0)) return error.BadParallelShape;
    return .{ .axis = axis, .first = rank * (dimension / 2), .count = dimension / 2 };
}

pub const Memory = struct {
    context: usize,
    streams: usize,
    cache_bytes: u64,
    compressor_bytes: u64,
    draft_bytes: u64,

    pub fn calculate(c: Config, context: usize, streams: usize, fp8: bool) !Memory {
        if (context == 0 or context > @import("config.zig").default_context or streams == 0 or streams > 4)
            return error.UnsupportedProfile;
        const rows = @import("state_plan.zig").rowBytes;
        const row = rows(c.head_dim, fp8); // an e4m3 row and its FP32 scale, or the bf16 halves
        const indexed = rows(c.index_head_dim, fp8);
        var cache: u64 = 0;
        var compressor: u64 = 0;
        for (c.compress_ratios[0..c.num_hidden_layers]) |ratio| {
            cache += c.sliding_window * row;
            if (ratio == 0) continue;
            cache += (context / ratio) * row;
            const overlap: u64 = if (ratio == 4) 2 else 1;
            compressor += overlap * ratio * overlap * c.head_dim * 4 * 2;
            if (ratio == 4) {
                cache += (context / ratio) * indexed;
                compressor += 2 * ratio * 2 * c.index_head_dim * 4 * 2;
            }
        }
        const draft = c.num_nextn_predict_layers * c.sliding_window * row;
        return .{ .context = context, .streams = streams, .cache_bytes = cache * streams, .compressor_bytes = compressor * streams, .draft_bytes = draft * streams };
    }

    pub fn total(m: Memory) u64 {
        return m.cache_bytes + m.compressor_bytes + m.draft_bytes;
    }
};

test "expert and attention TP2 slices do not mix Hadamard blocks" {
    const c = try Config.parse(std.testing.allocator, @import("config.zig").fixture);
    const w: Spec = .{ .key = "layers.0.ffn.experts.0.w2", .shape = .{ 2048, 4096, 1 }, .rank = 2, .kind = .matrix };
    const a = try partition(c, w, 0);
    const b = try partition(c, w, 1);
    try std.testing.expectEqual(@as(?u8, 0), a.axis);
    try std.testing.expectEqual(@as(usize, 1024), a.count);
    try std.testing.expectEqual(a.count, b.first);
    const group: Spec = .{ .key = "layers.0.attn.wo_a.slice.4", .shape = .{ 4096, 1024, 1 }, .rank = 2, .kind = .matrix };
    try std.testing.expect(!(try partition(c, group, 0)).keep);
    try std.testing.expect((try partition(c, group, 1)).keep);
}

test "the KV estimate scales with streams and rejects a silent profile downgrade" {
    const c = try Config.parse(std.testing.allocator, @import("config.zig").fixture);
    const one = try Memory.calculate(c, 262144, 1, false);
    const four = try Memory.calculate(c, 262144, 4, false);
    try std.testing.expectEqual(one.total() * 4, four.total());
    try std.testing.expectError(error.UnsupportedProfile, Memory.calculate(c, 1048576, 4, false));
    const fp8_rows = try Memory.calculate(c, 262144, 1, true); // --kv-cache fp8 buys back room, never spends it
    try std.testing.expect(fp8_rows.cache_bytes < one.cache_bytes);
    try std.testing.expectEqual(one.compressor_bytes, fp8_rows.compressor_bytes); // the rings stay FP32
    try std.testing.expect(fp8_rows.draft_bytes < one.draft_bytes);
}
