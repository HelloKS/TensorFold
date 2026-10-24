//! Shared precision and launch helpers for a single TP rank's reference text graph.
const std = @import("std");
const cuda = @import("cuda");
const gpu = @import("gpu_weights.zig");
pub const B = enum { h, residual, x, norm, qr, q, kv, attn, grouped, out, gate, up, act, expert, accum, mixes, pre, post, comb, logits, ixq, ixw, score, selected, linear, picks, routeweights, ids, gather, tmp };
pub const Runtime = struct {
    operators: @import("vision_graph.zig").Runtime,
    text: cuda.dsv4_text.Ops,
    scratch: gpu.Scratch = .{},
    buffers: [@typeInfo(B).@"enum".field_names.len]cuda.DeviceBuffer,
    pub fn init(a: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, c: @import("config.zig").Config, dir: ?[]const u8) !Runtime {
        var r: Runtime = undefined;
        r.operators = try @import("vision_graph.zig").Runtime.open(a, io, d, c, dir);
        errdefer r.operators.close();
        r.text = try @import("gpu_modules.zig").load(cuda.dsv4_text.Ops, a, io, d, dir, "dsv4_text.fatbin");
        errdefer r.text.unload();
        r.scratch = .{};
        var n: usize = 0;
        errdefer for (r.buffers[0..n]) |*v| v.free();
        for (&r.buffers, 0..) |*v, i| {
            const bytes: usize = if (i == @intFromEnum(B.linear)) @import("sparse_rows.zig").max_selected * 4 else 1 << 20;
            v.* = try cuda.DeviceBuffer.alloc(d, bytes);
            n += 1;
        }
        var linear: [@import("sparse_rows.zig").max_selected]i32 = undefined;
        for (&linear, 0..) |*index, i| index.* = @intCast(i);
        try r.buf(.linear).upload(0, std.mem.asBytes(&linear));
        return r;
    }
    pub fn free(r: *Runtime) void {
        for (&r.buffers) |*v| v.free();
        r.scratch.free();
        r.text.unload();
        r.operators.close();
    }
    pub fn buf(r: *Runtime, b: B) cuda.DeviceBuffer {
        return r.buffers[@backingInt(b)];
    }
    pub fn p(r: *Runtime, b: B) u64 {
        return r.buf(b).ptr;
    }
};
pub fn frequencies(c: @import("config.zig").Config, compressed: bool) [32]f32 {
    var out: [32]f32 = undefined;
    const base: f32 = @floatCast(if (compressed) c.compress_rope_theta else c.rope_theta);
    const dim: f32 = @floatFromInt(c.qk_rope_head_dim);
    const original: f32 = @floatFromInt(c.original_max_position_embeddings);
    const lo = @max(0, @floor(dim * @log(original / (@as(f32, @floatCast(c.beta_fast)) * 2 * std.math.pi)) / (2 * @log(base))));
    const hi = @min(dim - 1, @ceil(dim * @log(original / (@as(f32, @floatCast(c.beta_slow)) * 2 * std.math.pi)) / (2 * @log(base))));
    for (&out, 0..) |*v, i| {
        const index: f32 = @floatFromInt(i);
        const plain = 1 / std.math.pow(f32, base, 2 * index / dim);
        const ramp = @max(0, @min(1, (index - lo) / (if (lo == hi) hi + 0.001 - lo else hi - lo)));
        v.* = if (compressed) plain / @as(f32, @floatCast(c.rope_factor)) * ramp + plain * (1 - ramp) else plain;
    }
    return out;
}
pub const Modes = struct {
    dspark: bool = false,
    graph: @import("graphs.zig").Settings = .{},
    p2p: bool = false,
    roce: bool = false,
};
pub fn fingerprint(a: std.mem.Allocator, io: std.Io, dir: ?[]const u8, storage: @import("storage.zig").Storage, batched: bool, rows_cap: u32, kv8: bool, wc: bool, modes: Modes) ![32]u8 {
    var sha = std.crypto.hash.sha2.Sha256.init(.{});
    sha.update("dsv4-multimodal-dspark-tp2/v3");
    inline for (.{ "text_forward.zig", "text_runtime.zig", "text_weights.zig", "text_state.zig", "text_model.zig", "text_identity.zig", "comm.zig", "link.zig", "state_plan.zig", "control.zig", "distributed_state.zig", "dspark.zig", "vision_forward.zig", "vision_serving.zig", "serving.zig", "native.zig", "text_admission.zig", "vision_graph.zig", "media.zig", "image_layout.zig", "picture.zig", "rgb.zig", "png.zig", "jpeg.zig", "weights.zig", "parallel.zig", "config.zig", "storage.zig", "image_cache.zig", "rowtab.zig", "rounds.zig", "graphs.zig", "round_graph.zig", "kv_pages.zig", "window_cost.zig", "sparse_rows.zig" }) |file| {
        const source = @embedFile(file);
        for (source, 0..) |byte, i| {
            if (byte == '\r' and i + 1 < source.len and source[i + 1] == '\n') continue;
            sha.update(&.{byte});
        }
    }
    inline for (.{ "exl3", "dsv4_native", "dsv4_ops", "dsv4_tensor", "dsv4_vision", "dsv4_text" }) |name| {
        const needed = if (comptime std.mem.eql(u8, name, "exl3")) storage == .exl3 else if (comptime std.mem.eql(u8, name, "dsv4_native")) storage != .exl3 else true;
        if (needed) {
            if (dir) |path| {
                const file = try std.fs.path.join(a, &.{ path, name ++ ".fatbin" });
                defer a.free(file);
                const image = try std.Io.Dir.cwd().readFileAlloc(io, file, a, .limited(64 << 20));
                defer a.free(image);
                sha.update(image);
            } else sha.update(@field(cuda.kernels, name));
        }
    }
    var gate: [16]u8 = undefined; // the round's own gates: batched or serial, its row cap, the cache's row width, and the open-time cost measurement
    std.mem.writeInt(u32, gate[0..4], @as(u32, @intFromBool(batched)), .little);
    std.mem.writeInt(u32, gate[4..8], rows_cap, .little);
    std.mem.writeInt(u32, gate[8..12], @as(u32, @intFromBool(kv8)), .little); // an fp8 rank and a bf16 rank must never form a pair
    std.mem.writeInt(u32, gate[12..16], @as(u32, @intFromBool(wc)), .little); // measuring ranks mirror passes; a measuring and a silent rank would desync
    sha.update(&gate);
    sha.update(&.{ @intFromBool(modes.dspark), @intFromBool(modes.graph.on), @intFromBool(modes.p2p), @intFromBool(modes.roce) });
    if (modes.graph.on) {
        sha.update(&.{@intFromBool(modes.graph.floor_hold)});
        var shape: [20]u8 = undefined;
        std.mem.writeInt(u32, shape[0..4], modes.graph.max, .little);
        std.mem.writeInt(u32, shape[4..8], modes.graph.bucket, .little);
        std.mem.writeInt(u32, shape[8..12], modes.graph.rows_max, .little);
        std.mem.writeInt(u64, shape[12..20], @bitCast(modes.graph.grow), .little);
        sha.update(&shape);
    }
    if (batched) {
        if (dir) |path| {
            const file = try std.fs.path.join(a, &.{ path, "dsv4_batch.fatbin" });
            defer a.free(file);
            const image = try std.Io.Dir.cwd().readFileAlloc(io, file, a, .limited(64 << 20));
            defer a.free(image);
            sha.update(image);
        } else if (@hasDecl(cuda.kernels, "dsv4_batch")) sha.update(cuda.kernels.dsv4_batch);
    }
    return sha.finalResult();
}

test "the execution identity covers the cache's row width, so an fp8 rank and a bf16 rank cannot form a pair" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const storage: @import("storage.zig").Storage = .native_fp4_fp8;
    const wide = try fingerprint(a, std.testing.io, null, storage, false, 0, true, false, .{});
    const plain = try fingerprint(a, std.testing.io, null, storage, false, 0, false, false, .{});
    try std.testing.expect(!std.mem.eql(u8, &wide, &plain)); // identity.validate answers RankIdentityMismatch for a pair that differs only here
    const again = try fingerprint(a, std.testing.io, null, storage, false, 0, true, false, .{});
    try std.testing.expectEqualSlices(u8, &wide, &again); // the same gates hash the same: the refusal is the flag, never the digest drifting
    const measured = try fingerprint(a, std.testing.io, null, storage, false, 0, true, true, .{});
    try std.testing.expect(!std.mem.eql(u8, &wide, &measured)); // a measuring rank and a silent rank must never form a pair either
}

test "ranks refuse different startup passes, graph collectives and transport negotiation modes" {
    const a = std.testing.allocator;
    const plain = try fingerprint(a, std.testing.io, null, .native_fp4_fp8, true, 24, false, true, .{});
    for ([_]Modes{ .{ .dspark = true }, .{ .graph = .{ .on = true } }, .{ .p2p = true }, .{ .roce = true } }) |mode| {
        const other = try fingerprint(a, std.testing.io, null, .native_fp4_fp8, true, 24, false, true, mode);
        try std.testing.expect(!std.mem.eql(u8, &plain, &other));
    }
    const graph = try fingerprint(a, std.testing.io, null, .native_fp4_fp8, true, 24, false, true, .{ .graph = .{ .on = true } });
    const capped = try fingerprint(a, std.testing.io, null, .native_fp4_fp8, true, 24, false, true, .{ .graph = .{ .on = true, .rows_max = 2 } });
    try std.testing.expect(!std.mem.eql(u8, &graph, &capped));
    const held = try fingerprint(a, std.testing.io, null, .native_fp4_fp8, true, 24, false, true, .{ .graph = .{ .on = true, .floor_hold = true } });
    try std.testing.expect(!std.mem.eql(u8, &graph, &held));
}

test "plain and compressed YaRN frequencies match the pinned official float32 reference" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try @import("config.zig").Config.parse(a, @import("config.zig").native_fixture);
    const doc = try std.json.parseFromSliceLeaky(std.json.Value, a, @embedFile("fixtures/text_frequencies.json"), .{});
    inline for (.{ "plain", "compressed" }, 0..) |name, index| {
        const expected = doc.object.get(name).?.array.items;
        const actual = frequencies(c, index == 1);
        for (actual, expected) |value, reference| try std.testing.expectApproxEqRel(@as(f32, @floatCast(reference.float)), value, 1e-5);
    }
}
