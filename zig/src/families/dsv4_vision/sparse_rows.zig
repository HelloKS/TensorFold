//! Sparse cache addresses become dense row numbers after FP8 gather; both target and draft share this launch.
const std = @import("std");
const Config = @import("config.zig").Config;

pub const max_selected = 4096;

pub fn capacity(c: Config, context: usize) usize {
    var compressed: usize = 0;
    for (c.compress_ratios[0..c.num_hidden_layers]) |ratio| {
        if (ratio == 0) continue;
        compressed = @max(compressed, if (ratio == 4) @min(context / ratio, c.index_topk) else context / ratio);
    }
    return c.sliding_window + @max(c.vision_max_n_token + compressed, c.dspark_block_size);
}

pub fn run(m: anytype, staging: anytype, cache: u64, sink: u64, count: usize) !void {
    if (count > max_selected) return error.UnsupportedSparseShape;
    var rows = cache;
    var indices = m.r.p(.selected);
    if (m.kv8) {
        const dense = staging orelse return error.MissingStaging;
        if (count * m.c.head_dim * 2 > dense.len) return error.SparseStagingTooSmall;
        try m.r.text.call("gather8", m.stream, @intCast(count), 256,
            .{ cache, indices, dense.ptr, @as(u32, @intCast(count)), @as(u32, @intCast(m.c.head_dim)),
                @as(u32, @intCast(@import("state_plan.zig").rowBytes(m.c.head_dim, true))) });
        rows = dense.ptr;
        indices = m.r.p(.linear);
    }
    try m.r.operators.model.sparse(m.stream, m.r.p(.q), rows, indices, sink, m.r.p(.attn),
        .{ .rows = 1, .heads = @intCast(m.c.num_attention_heads / 2), .selected = @intCast(count),
            .dim = @intCast(m.c.head_dim) });
}

test "FP8 staging covers ratio-128 contexts as well as top-k and image rows" {
    const c = try Config.parse(std.testing.allocator, @import("config.zig").native_fixture);
    const n = capacity(c, 262144);
    try std.testing.expect(n >= c.sliding_window + 262144 / 128);
    try std.testing.expect(n >= c.sliding_window + c.vision_max_n_token + c.index_topk);
    try std.testing.expect(n <= max_selected);
}

test "FP8 attention uses compact row indices after gathering noncontiguous cache rows" {
    const Probe = struct {
        const Self = @This();
        const B = enum { selected, linear, q, attn };
        const Buffer = struct { ptr: u64, len: usize };
        const Text = struct {
            pub fn call(_: @This(), comptime name: []const u8, _: void, blocks: u32, _: u32, args: anytype) !void {
                try std.testing.expectEqualStrings("gather8", name);
                try std.testing.expectEqual(@as(u32, 3), blocks);
                try std.testing.expectEqual(@as(u64, 200), args[0]);
                try std.testing.expectEqual(@as(u64, 10), args[1]);
                try std.testing.expectEqual(@as(u64, 300), args[2]);
            }
        };
        const Operator = struct {
            pub fn sparse(_: @This(), _: void, _: u64, rows: u64, indices: u64, _: u64, _: u64, shape: @import("cuda").dsv4_ops.Sparse) !void {
                try std.testing.expectEqual(@as(u64, 300), rows);
                try std.testing.expectEqual(@as(u64, 20), indices);
                try std.testing.expectEqual(@as(u32, 3), shape.selected);
            }
        };
        const Runtime = struct {
            text: Text = .{},
            operators: struct { model: Operator = .{} } = .{},
            pub fn p(_: @This(), b: B) u64 { return switch (b) { .selected => 10, .linear => 20, .q => 30, .attn => 40 }; }
        };
        kv8: bool = true,
        c: Config,
        r: Runtime = .{},
        stream: void = {},
    };
    var m: Probe = .{ .c = try Config.parse(std.testing.allocator, @import("config.zig").native_fixture) };
    const staging: ?Probe.Buffer = .{ .ptr = 300, .len = 3 * m.c.head_dim * 2 };
    try run(&m, staging, 200, 400, 3);
    try std.testing.expectError(error.SparseStagingTooSmall, run(&m, staging, 200, 400, 4));
}
