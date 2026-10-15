//! DeepSeek-V4 Flash Vision's checkpoint contract; quantization widths belong to tensors, not the model average.
const std = @import("std");
const fmt = @import("core").exl3_format;

pub const model_revision = "6821d6ad3681a4b137b066b76094fa82ebd0a380";
pub const default_context = 262144;
pub const world_size = 2;
pub const max_streams = 4;

pub const Config = struct {
    num_hidden_layers: usize,
    num_nextn_predict_layers: usize,
    hidden_size: usize,
    vocab_size: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    qk_rope_head_dim: usize,
    q_lora_rank: usize,
    o_lora_rank: usize,
    o_groups: usize,
    moe_intermediate_size: usize,
    n_routed_experts: usize,
    n_shared_experts: usize,
    num_experts_per_tok: usize,
    num_hash_layers: usize,
    index_n_heads: usize,
    index_head_dim: usize,
    index_topk: usize,
    sliding_window: usize,
    hc_mult: usize,
    hc_sinkhorn_iters: usize,
    max_position_embeddings: usize,
    dspark_block_size: usize,
    dspark_noise_token_id: usize,
    dspark_markov_rank: usize,
    vision_n_layers: usize,
    vision_dim: usize,
    vision_n_heads: usize,
    vision_inter_dim: usize,
    vision_patch_size: usize,
    vision_downsample_ratio: usize,
    vision_max_n_token: usize,
    vision_min_pixels: usize,
    vision_max_wh_ratio: usize,
    bos_token_id: u32,
    eos_token_id: u32,
    rms_norm_eps: f64,
    hc_eps: f64,
    routed_scaling_factor: f64,
    swiglu_limit: f64,
    rope_theta: f64,
    compress_rope_theta: f64,
    vision_rope_theta: f64,
    compress_ratios: [46]u8,
    dspark_target_layer_ids: [3]u8,
    storage: @import("storage.zig").Storage,
    rope_factor: f64,
    beta_fast: f64,
    beta_slow: f64,
    original_max_position_embeddings: usize,

    pub fn parse(a: std.mem.Allocator, text: []const u8) !Config {
        const parsed = try std.json.parseFromSlice(std.json.Value, a, text, .{});
        defer parsed.deinit();
        if (parsed.value != .object) return error.BadConfig;
        const o = parsed.value.object;
        if (!stringIs(o, "model_type", "deepseek_v4")) return error.WrongModel;
        if (!stringIs(o, "scoring_func", "sqrtsoftplus")) return error.UnsupportedRouting;
        var c: Config = undefined;
        c.storage = try @import("storage.zig").Storage.parse(o);
        inline for (@typeInfo(Config).@"struct".field_names) |field| {
            const T = @FieldType(Config, field);
            if (comptime (T == usize or T == u32) and !std.mem.eql(u8, field, "original_max_position_embeddings"))
                @field(c, field) = std.math.cast(T, try integer(o, field)) orelse
                    return error.BadConfig;
            if (comptime T == f64 and !isRopeField(field)) @field(c, field) = try number(o, field);
        }
        const r = try object(o, "rope_scaling");
        c.rope_factor = try number(r, "factor");
        c.beta_fast = try number(r, "beta_fast");
        c.beta_slow = try number(r, "beta_slow");
        c.original_max_position_embeddings = std.math.cast(usize, try integer(r, "original_max_position_embeddings")) orelse return error.BadConfig;
        try readIds(o, "compress_ratios", &c.compress_ratios);
        try readIds(o, "dspark_target_layer_ids", &c.dspark_target_layer_ids);
        try c.validate();
        return c;
    }

    pub fn validate(c: Config) !void {
        if (c.num_hidden_layers != 43 or c.num_nextn_predict_layers != 3 or c.hidden_size != 4096 or
            c.vocab_size != 129280 or c.num_attention_heads != 64 or c.num_key_value_heads != 1 or
            c.head_dim != 512 or c.qk_rope_head_dim != 64 or c.q_lora_rank != 1024 or
            c.o_lora_rank != 1024 or c.o_groups != 8 or c.moe_intermediate_size != 2048 or
            c.n_routed_experts != 256 or c.n_shared_experts != 1 or c.num_experts_per_tok != 6 or
            c.num_hash_layers != 3 or c.index_n_heads != 64 or c.index_head_dim != 128 or
            c.index_topk != 512 or c.sliding_window != 128 or c.hc_mult != 4 or
            c.hc_sinkhorn_iters != 20 or c.dspark_block_size != 5 or c.dspark_markov_rank != 256)
            return error.UnsupportedTextShape;
        if (c.vision_n_layers != 32 or c.vision_dim != 1024 or c.vision_n_heads != 16 or
            c.vision_inter_dim != 2816 or c.vision_patch_size != 14 or c.vision_downsample_ratio != 3 or
            c.vision_max_n_token != 384 or c.vision_min_pixels != 147456 or c.vision_max_wh_ratio != 8)
            return error.UnsupportedVisionShape;
        if (c.max_position_embeddings < default_context or c.dspark_noise_token_id >= c.vocab_size or
            c.bos_token_id >= c.vocab_size or c.eos_token_id >= c.vocab_size) return error.BadConfig;
        for (c.compress_ratios, 0..) |ratio, i| {
            const expected: u8 = if (i < 2 or i >= 43) 0 else if (i % 2 == 0) 4 else 128;
            if (ratio != expected) return error.UnsupportedAttentionSchedule;
        }
        if (!std.mem.eql(u8, &c.dspark_target_layer_ids, &.{ 40, 41, 42 })) return error.UnsupportedDraftTaps;
        if (c.rms_norm_eps <= 0 or c.hc_eps <= 0 or c.rope_theta <= 0 or c.compress_rope_theta <= 0 or
            c.vision_rope_theta <= 0 or c.rope_factor <= 0 or c.beta_slow <= 0 or c.beta_fast <= c.beta_slow or
            c.original_max_position_embeddings == 0 or c.routed_scaling_factor <= 0 or c.swiglu_limit <= 0)
            return error.BadConfig;
    }
};

fn isRopeField(comptime name: []const u8) bool {
    inline for (.{ "rope_factor", "beta_fast", "beta_slow" }) |field| {
        if (std.mem.eql(u8, name, field)) return true;
    }
    return false;
}

fn object(o: std.json.ObjectMap, key: []const u8) !std.json.ObjectMap {
    const v = o.get(key) orelse return error.MissingConfigField;
    if (v != .object) return error.BadConfig;
    return v.object;
}

fn stringIs(o: std.json.ObjectMap, key: []const u8, expected: []const u8) bool {
    const v = o.get(key) orelse return false;
    return v == .string and std.mem.eql(u8, v.string, expected);
}

fn integer(o: std.json.ObjectMap, key: []const u8) !i64 {
    const v = o.get(key) orelse return error.MissingConfigField;
    if (v != .integer or v.integer < 0) return error.BadConfig;
    return v.integer;
}

fn number(o: std.json.ObjectMap, key: []const u8) !f64 {
    const v = o.get(key) orelse return error.MissingConfigField;
    const n: f64 = switch (v) {
        .integer => @floatFromInt(v.integer),
        .float => v.float,
        else => return error.BadConfig,
    };
    if (!std.math.isFinite(n)) return error.BadConfig;
    return n;
}

fn readIds(o: std.json.ObjectMap, key: []const u8, out: []u8) !void {
    const v = o.get(key) orelse return error.MissingConfigField;
    if (v != .array or v.array.items.len != out.len) return error.BadConfig;
    for (out, v.array.items) |*dest, item| {
        if (item != .integer) return error.BadConfig;
        dest.* = std.math.cast(u8, item.integer) orelse return error.BadConfig;
    }
}

pub const fixture = @embedFile("fixtures/config.json");

test "the real EXL3 config accepts arbitrary model average bits" {
    const c = try Config.parse(std.testing.allocator, fixture);
    try std.testing.expectEqual(fmt.Codebook.mul1, c.storage.exl3);
    try std.testing.expectEqual(@as(u8, 128), c.compress_ratios[3]);
    const four = try std.mem.replaceOwned(u8, std.testing.allocator, fixture, "3.04", "4.0");
    defer std.testing.allocator.free(four);
    const other = try Config.parse(std.testing.allocator, four);
    try std.testing.expectEqual(c.hidden_size, other.hidden_size);
}

test "wrong models, missing vision and mismatched compression cannot claim this family" {
    const a = std.testing.allocator;
    for ([_]struct { []const u8, []const u8, anyerror }{
        .{ "deepseek_v4", "deepseek_v41", error.WrongModel },
        .{ "\"vision_n_layers\": 32", "\"vision_n_layers\": 0", error.UnsupportedVisionShape },
        .{ "\"codebook\": \"mul1\"", "\"codebook\": \"other\"", error.UnsupportedCodebook },
        .{ "\"hidden_size\": 4096", "\"hidden_size\": 5120", error.UnsupportedTextShape },
    }) |case| {
        const s = try std.mem.replaceOwned(u8, a, fixture, case[0], case[1]);
        defer a.free(s);
        try std.testing.expectError(case[2], Config.parse(a, s));
    }
    var c = try Config.parse(a, fixture);
    c.compress_ratios[3] = 2;
    try std.testing.expectError(error.UnsupportedAttentionSchedule, c.validate());
}

pub const native_fixture = @embedFile("fixtures/native_config.json");

test "native storage requires the original FP4 FP8 quantization contract" {
    const a = std.testing.allocator;
    const c = try Config.parse(a, native_fixture);
    try std.testing.expect(c.storage == .native_fp4_fp8);
    inline for (.{ .{ "ue8m0", "float32" }, .{ "e4m3", "e5m2" }, .{ "dynamic", "static" } }) |pair| {
        const bad = try std.mem.replaceOwned(u8, a, native_fixture, pair[0], pair[1]);
        defer a.free(bad);
        try std.testing.expectError(error.UnsupportedQuantization, Config.parse(a, bad));
    }
}
