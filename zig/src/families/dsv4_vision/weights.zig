//! Required tensor graph and EXL3 storage validation, including split vision projections and DSpark heads.
const std = @import("std");
const core = @import("core");
const Config = @import("config.zig").Config;
const st = core.safetensors;
const fmt = core.exl3_format;
const storage = @import("storage.zig");

pub const Kind = enum { float, ids, matrix };
pub const Spec = struct { key: []const u8, shape: [3]usize = @splat(1), rank: usize, kind: Kind };

const Builder = struct {
    a: std.mem.Allocator,
    items: std.ArrayList(Spec) = .empty,

    fn tensor(b: *Builder, prefix: []const u8, suffix: []const u8, shape: []const usize, kind: Kind) !void {
        var spec: Spec = .{ .key = try std.fmt.allocPrint(b.a, "{s}{s}", .{ prefix, suffix }), .rank = shape.len, .kind = kind };
        @memcpy(spec.shape[0..shape.len], shape);
        try b.items.append(b.a, spec);
    }

    fn linear(b: *Builder, prefix: []const u8, suffix: []const u8, k: usize, n: usize) !void {
        try b.tensor(prefix, suffix, &.{ k, n }, .matrix);
    }

    fn compressor(b: *Builder, p: []const u8, dim: usize, ratio: usize, head: usize) !void {
        const channels = head * (1 + @as(usize, @intFromBool(ratio == 4)));
        try b.linear(p, ".wkv", dim, channels);
        try b.linear(p, ".wgate", dim, channels);
        try b.tensor(p, ".ape", &.{ ratio, channels }, .float);
        try b.tensor(p, ".norm.weight", &.{head}, .float);
    }

    fn layer(b: *Builder, c: Config, p: []const u8, ratio: usize, hash: bool, vision: bool) !void {
        const d = c.hidden_size;
        const mix = (2 + c.hc_mult) * c.hc_mult;
        inline for (.{ "attn", "ffn" }) |part| {
            try b.tensor(p, ".hc_" ++ part ++ "_fn", &.{ mix, d * c.hc_mult }, .float);
            try b.tensor(p, ".hc_" ++ part ++ "_base", &.{mix}, .float);
            try b.tensor(p, ".hc_" ++ part ++ "_scale", &.{3}, .float);
            try b.tensor(p, "." ++ part ++ "_norm.weight", &.{d}, .float);
        }
        try b.linear(p, ".attn.wq_a", d, c.q_lora_rank);
        try b.linear(p, ".attn.wq_b", c.q_lora_rank, c.num_attention_heads * c.head_dim);
        try b.linear(p, ".attn.wkv", d, c.head_dim);
        try b.linear(p, ".attn.wo_b", c.o_groups * c.o_lora_rank, d);
        if (c.storage == .native_fp4_fp8) {
            try b.linear(p, ".attn.wo_a", c.num_attention_heads / c.o_groups * c.head_dim, c.o_groups * c.o_lora_rank);
        } else for (0..c.o_groups) |group| {
            const key = try std.fmt.allocPrint(b.a, "{s}.attn.wo_a.slice.{d}", .{ p, group });
            try b.linear(key, "", c.num_attention_heads / c.o_groups * c.head_dim, c.o_lora_rank);
        }
        try b.tensor(p, ".attn.q_norm.weight", &.{c.q_lora_rank}, .float);
        try b.tensor(p, ".attn.kv_norm.weight", &.{c.head_dim}, .float);
        try b.tensor(p, ".attn.attn_sink", &.{c.num_attention_heads}, .float);
        if (ratio > 0) {
            const cp = try std.fmt.allocPrint(b.a, "{s}.attn.compressor", .{p});
            try b.compressor(cp, d, ratio, c.head_dim);
            if (ratio == 4) {
                try b.linear(p, ".attn.indexer.wq_b", c.q_lora_rank, c.index_n_heads * c.index_head_dim);
                try b.tensor(p, ".attn.indexer.weights_proj.weight", &.{ c.index_n_heads, d }, .float);
                const ip = try std.fmt.allocPrint(b.a, "{s}.attn.indexer.compressor", .{p});
                try b.compressor(ip, d, ratio, c.index_head_dim);
            }
        }
        try b.tensor(p, ".ffn.gate.weight", &.{ c.n_routed_experts, d }, .float);
        try b.tensor(p, ".ffn.gate.bias", &.{c.n_routed_experts}, .float);
        if (vision) try b.tensor(p, ".ffn.gate.bias_vl", &.{c.n_routed_experts}, .float);
        if (hash) try b.tensor(p, ".ffn.gate.tid2eid", &.{ c.vocab_size, c.num_experts_per_tok }, .ids);
        for (0..c.n_routed_experts + 1) |e| {
            const ep = if (e < c.n_routed_experts)
                try std.fmt.allocPrint(b.a, "{s}.ffn.experts.{d}", .{ p, e })
            else
                try std.fmt.allocPrint(b.a, "{s}.ffn.shared_experts", .{p});
            try b.linear(ep, ".w1", d, c.moe_intermediate_size);
            try b.linear(ep, ".w3", d, c.moe_intermediate_size);
            try b.linear(ep, ".w2", c.moe_intermediate_size, d);
        }
    }

    fn hcHead(b: *Builder, p: []const u8, c: Config) !void {
        try b.tensor(p, "hc_head_fn", &.{ c.hc_mult, c.hc_mult * c.hidden_size }, .float);
        try b.tensor(p, "hc_head_base", &.{c.hc_mult}, .float);
        try b.tensor(p, "hc_head_scale", &.{1}, .float);
    }
};

/// Names and their slices are arena-owned; linear shapes are [K,N], dense file shapes are [N,K].
pub fn requirements(a: std.mem.Allocator, c: Config) ![]const Spec {
    var b: Builder = .{ .a = a };
    try b.tensor("embed", ".weight", &.{ c.vocab_size, c.hidden_size }, .float);
    try b.tensor("norm", ".weight", &.{c.hidden_size}, .float);
    try b.linear("head", "", c.hidden_size, c.vocab_size);
    try b.hcHead("", c);
    for (0..c.num_hidden_layers) |i| {
        const p = try std.fmt.allocPrint(a, "layers.{d}", .{i});
        try b.layer(c, p, c.compress_ratios[i], i < c.num_hash_layers, true);
    }
    for (0..c.num_nextn_predict_layers) |i| {
        const p = try std.fmt.allocPrint(a, "mtp.{d}", .{i});
        try b.layer(c, p, 0, false, false);
    }
    try b.linear("mtp.0.main_proj", "", c.hidden_size * c.dspark_target_layer_ids.len, c.hidden_size);
    try b.tensor("mtp.0.main_norm", ".weight", &.{c.hidden_size}, .float);
    const last = try std.fmt.allocPrint(a, "mtp.{d}.", .{c.num_nextn_predict_layers - 1});
    try b.hcHead(last, c);
    try b.tensor(last, "norm.weight", &.{c.hidden_size}, .float);
    try b.linear(last, "markov_head.markov_w1", c.dspark_markov_rank, c.vocab_size);
    try b.linear(last, "markov_head.markov_w2", c.dspark_markov_rank, c.vocab_size);
    try b.linear(last, "confidence_head.proj", c.hidden_size + c.dspark_markov_rank, 1);
    try b.tensor("vision.patch_embed.proj", ".weight", &.{ c.vision_dim, 3 * c.vision_patch_size * c.vision_patch_size }, .float);
    try b.tensor("vision.patch_embed.proj", ".bias", &.{c.vision_dim}, .float);
    for (0..c.vision_n_layers) |i| {
        const p = try std.fmt.allocPrint(a, "vision.blocks.{d}", .{i});
        inline for (.{ "norm1", "norm2" }) |n| try b.tensor(p, "." ++ n ++ ".weight", &.{c.vision_dim}, .float);
        if (c.storage == .native_fp4_fp8) {
            try b.linear(p, ".attn.wqkv", c.vision_dim, 3 * c.vision_dim);
            try b.tensor(p, ".attn.wqkv.bias", &.{3 * c.vision_dim}, .float);
            try b.linear(p, ".attn.wo", c.vision_dim, c.vision_dim);
            try b.tensor(p, ".attn.wo.bias", &.{c.vision_dim}, .float);
            try b.linear(p, ".mlp.w1", c.vision_dim, 2 * c.vision_inter_dim);
        } else {
            inline for (.{ "q_proj", "k_proj", "v_proj", "wo" }) |n| {
                try b.linear(p, ".attn." ++ n, c.vision_dim, c.vision_dim);
                try b.tensor(p, ".attn." ++ n ++ ".bias", &.{c.vision_dim}, .float);
            }
            inline for (.{ "gate_proj", "up_proj" }) |n|
                try b.linear(p, ".mlp." ++ n, c.vision_dim, c.vision_inter_dim);
        }
        try b.linear(p, ".mlp.w2", c.vision_inter_dim, c.vision_dim);
    }
    try b.tensor("vision.norm", ".weight", &.{c.vision_dim}, .float);
    try b.linear("aligner.w1", "", c.vision_dim * c.vision_downsample_ratio * c.vision_downsample_ratio, c.hidden_size);
    try b.linear("aligner.w2", "", c.hidden_size, c.hidden_size);
    try b.tensor("aligner.w1", ".bias", &.{c.hidden_size}, .float);
    try b.tensor("aligner.w2", ".bias", &.{c.hidden_size}, .float);
    inline for (.{ "image_start", "image_pad", "image_newline", "image_end" }) |name|
        try b.tensor(name, "", &.{c.hidden_size}, .float);
    return b.items.toOwnedSlice(a);
}

pub fn floatType(t: st.DType) bool {
    return t == .f16 or t == .bf16 or t == .f32;
}

pub fn matches(t: st.Entry, spec: Spec) bool {
    const dtype = if (spec.kind == .ids) t.dtype == .i32 or t.dtype == .i64 else floatType(t.dtype);
    return dtype and t.rank == spec.rank and std.mem.eql(usize, t.shape[0..t.rank], spec.shape[0..spec.rank]);
}

pub const Matrix = struct {
    k: usize,
    n: usize,
    width: fmt.Width,
    codebook: fmt.Codebook,

    pub fn parse(t: st.Entry, suh: st.Entry, svh: st.Entry, k: usize, n: usize, cb: fmt.Codebook) !Matrix {
        if (k == 0 or n == 0 or k % 128 != 0 or n % 128 != 0 or t.dtype != .i16 or t.rank != 3 or
            t.shape[0] != k / 16 or t.shape[1] != n / 16 or t.shape[2] % 8 != 0)
            return error.BadExl3Shape;
        const halves = t.shape[2] / 8;
        if (halves < 2 or halves > 16 or (halves % 2 == 1 and (cb != .mul1 or halves > 7)))
            return error.UnsupportedExl3Width;
        if (!scale(suh, k) or !scale(svh, n)) return error.BadExl3Scale;
        return .{ .k = k, .n = n, .width = .{ .halves = @intCast(halves) }, .codebook = cb };
    }

    fn scale(t: st.Entry, n: usize) bool {
        return t.rank == 1 and ((t.dtype == .f16 and t.shape[0] == n) or
            (t.dtype == .i16 and t.shape[0] == n / 16));
    }
};

pub const Summary = struct { matrices: usize = 0, dense: usize = 0, raw: usize = 0, fp4: usize = 0, fp8: usize = 0 };

/// The headers already passed safetensors bounds validation; errors name the missing or incompatible tensor.
pub fn validate(a: std.mem.Allocator, c: Config, headers: st.Header, problem: *[]const u8) !Summary {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    return validateSpecs(a, c, try requirements(arena.allocator(), c), headers, problem);
}

pub fn validateSpecs(a: std.mem.Allocator, c: Config, specs: []const Spec, headers: st.Header, problem: *[]const u8) !Summary {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const aa = arena.allocator();
    var summary: Summary = .{};
    for (specs) |spec| {
        if (spec.kind != .matrix) {
            const t = headers.get(spec.key) orelse {
                problem.* = try a.dupe(u8, spec.key);
                return error.MissingTensor;
            };
            if (!matches(t, spec)) {
                problem.* = try a.dupe(u8, spec.key);
                return error.UnexpectedTensor;
            }
            summary.raw += 1;
            continue;
        }
        const trellis_name = try std.fmt.allocPrint(aa, "{s}.trellis", .{spec.key});
        if (c.storage == .native_fp4_fp8) {
            if (headers.contains(trellis_name)) {
                problem.* = try a.dupe(u8, trellis_name);
                return error.MixedStorage;
            }
            try validateNative(a, aa, spec, headers, problem, &summary);
        } else if (headers.get(trellis_name)) |t| {
            const su = try std.fmt.allocPrint(aa, "{s}.suh", .{spec.key});
            const sv = try std.fmt.allocPrint(aa, "{s}.svh", .{spec.key});
            const marker = try std.fmt.allocPrint(aa, "{s}.{t}", .{ spec.key, c.storage.exl3 });
            const suh = headers.get(su) orelse {
                problem.* = try a.dupe(u8, su);
                return error.MissingTensor;
            };
            const svh = headers.get(sv) orelse {
                problem.* = try a.dupe(u8, sv);
                return error.MissingTensor;
            };
            if (c.storage.exl3 == .mul1) {
                const m = headers.get(marker) orelse {
                    problem.* = try a.dupe(u8, marker);
                    return error.MissingTensor;
                };
                if (m.dtype != .i32 or m.rank != 0) {
                    problem.* = try a.dupe(u8, marker);
                    return error.UnexpectedTensor;
                }
            }
            _ = Matrix.parse(t, suh, svh, spec.shape[0], spec.shape[1], c.storage.exl3) catch |e| {
                problem.* = try a.dupe(u8, trellis_name);
                return e;
            };
            summary.matrices += 1;
        } else {
            const name = try std.fmt.allocPrint(aa, "{s}.weight", .{spec.key});
            const t = headers.get(name) orelse {
                problem.* = try a.dupe(u8, name);
                return error.MissingTensor;
            };
            var dense_spec = spec;
            dense_spec.shape[0] = spec.shape[1];
            dense_spec.shape[1] = spec.shape[0];
            if (!matches(t, dense_spec)) {
                problem.* = try a.dupe(u8, name);
                return error.UnexpectedTensor;
            }
            summary.dense += 1;
        }
    }
    return summary;
}

fn validateNative(a: std.mem.Allocator, aa: std.mem.Allocator, spec: Spec, headers: st.Header, problem: *[]const u8, summary: *Summary) !void {
    const name = try std.fmt.allocPrint(aa, "{s}.weight", .{spec.key});
    const t = headers.get(name) orelse {
        problem.* = try a.dupe(u8, name);
        return error.MissingTensor;
    };
    const sn = try std.fmt.allocPrint(aa, "{s}.scale", .{spec.key});
    if (storage.dense(spec.key)) {
        var dense = spec;
        dense.shape[0] = spec.shape[1];
        dense.shape[1] = spec.shape[0];
        if (t.dtype != .bf16 or !matches(t, dense) or headers.contains(sn)) {
            problem.* = try a.dupe(u8, name);
            return error.UnexpectedTensor;
        }
        summary.dense += 1;
        return;
    }
    const scale = headers.get(sn) orelse {
        problem.* = try a.dupe(u8, sn);
        return error.MissingTensor;
    };
    const q = storage.matrix(t, scale, spec.key, spec.shape[0], spec.shape[1]) catch |e| {
        problem.* = try a.dupe(u8, if (e == error.BadNativeScale) sn else name);
        return e;
    };
    if (q == .fp4) summary.fp4 += 1 else summary.fp8 += 1;
}

test "missing image bias is rejected before any CUDA work" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try Config.parse(a, @import("config.zig").fixture);
    var header: st.Header = .empty;
    for (try requirements(a, c)) |spec| {
        if (std.mem.eql(u8, spec.key, "layers.0.ffn.gate.bias_vl")) continue;
        var e: st.Entry = .{ .dtype = if (spec.kind == .ids) .i64 else .bf16, .rank = @intCast(spec.rank), .shape = @splat(1), .begin = 0, .end = 0 };
        @memcpy(e.shape[0..spec.rank], spec.shape[0..spec.rank]);
        var key = spec.key;
        if (spec.kind == .matrix) {
            e.shape[0] = spec.shape[1];
            e.shape[1] = spec.shape[0];
            key = try std.fmt.allocPrint(a, "{s}.weight", .{key});
        }
        try header.put(a, key, e);
    }
    var problem: []const u8 = "";
    try std.testing.expectError(error.MissingTensor, validate(a, c, header, &problem));
    try std.testing.expectEqualStrings("layers.0.ffn.gate.bias_vl", problem);
}

test "EXL3 width comes from the tensor rather than the checkpoint's average bpw" {
    const su: st.Entry = .{ .dtype = .f16, .rank = 1, .shape = .{ 4096, 1, 1, 1, 1 }, .begin = 0, .end = 8192 };
    const sv: st.Entry = .{ .dtype = .f16, .rank = 1, .shape = .{ 2048, 1, 1, 1, 1 }, .begin = 0, .end = 4096 };
    var t: st.Entry = .{ .dtype = .i16, .rank = 3, .shape = .{ 256, 128, 48, 1, 1 }, .begin = 0, .end = 0 };
    try std.testing.expectEqual(@as(u16, 6), (try Matrix.parse(t, su, sv, 4096, 2048, .mul1)).width.halves);
    t.shape[2] = 64;
    try std.testing.expectEqual(@as(u16, 8), (try Matrix.parse(t, su, sv, 4096, 2048, .mcg)).width.halves);
    t.shape[2] = 56;
    try std.testing.expectError(error.UnsupportedExl3Width, Matrix.parse(t, su, sv, 4096, 2048, .mcg));
}

test "native missing scales and mixed EXL3 storage report the exact tensor" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try Config.parse(a, @import("config.zig").native_fixture);
    var h: st.Header = .empty;
    const spec: Spec = .{ .key = "layers.0.attn.wq_a", .rank = 2, .shape = .{ 4096, 1024, 1 }, .kind = .matrix };
    const w: st.Entry = .{ .dtype = .f8_e4m3, .rank = 2, .shape = .{ 1024, 4096, 1, 1, 1 }, .begin = 0, .end = 0 };
    try h.put(a, "layers.0.attn.wq_a.weight", w);
    var problem: []const u8 = "";
    try std.testing.expectError(error.MissingTensor, validateSpecs(a, c, &.{spec}, h, &problem));
    try std.testing.expectEqualStrings("layers.0.attn.wq_a.scale", problem);
    try h.put(a, "layers.0.attn.wq_a.trellis", w);
    try std.testing.expectError(error.MixedStorage, validateSpecs(a, c, &.{spec}, h, &problem));
    try std.testing.expectEqualStrings("layers.0.attn.wq_a.trellis", problem);
}

test "native vision requires fused projections while EXL3 requires the selected split projections" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try Config.parse(a, @import("config.zig").native_fixture);
    var fused = false;
    for (try requirements(a, c)) |spec| {
        if (std.mem.eql(u8, spec.key, "vision.blocks.0.attn.wqkv")) {
            try std.testing.expectEqual(@as(usize, 3072), spec.shape[1]);
            fused = true;
        }
        try std.testing.expect(!std.mem.eql(u8, spec.key, "vision.blocks.0.attn.q_proj"));
    }
    try std.testing.expect(fused);
}
