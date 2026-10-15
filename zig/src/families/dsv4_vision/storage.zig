//! Checkpoint storage is independent of model architecture; native FP4 is packed E2M1, not NVFP4.
const std = @import("std");
const core = @import("core");
const st = core.safetensors;

pub const Storage = union(enum) {
    exl3: core.exl3_format.Codebook,
    native_fp4_fp8,

    pub fn name(s: Storage) []const u8 {
        return switch (s) {
            .exl3 => "exl3",
            .native_fp4_fp8 => "native-fp4-fp8",
        };
    }

    pub fn parse(o: std.json.ObjectMap) !Storage {
        const qv = o.get("quantization_config") orelse return error.MissingConfigField;
        if (qv != .object) return error.BadConfig;
        const q = qv.object;
        if (is(q, "quant_method", "exl3")) {
            const cb = q.get("codebook") orelse return error.MissingCodebook;
            if (cb != .string) return error.MissingCodebook;
            return .{ .exl3 = std.meta.stringToEnum(core.exl3_format.Codebook, cb.string) orelse
                return error.UnsupportedCodebook };
        }
        if (!is(q, "quant_method", "fp8") or !is(q, "fmt", "e4m3") or
            !is(q, "scale_fmt", "ue8m0") or !is(q, "activation_scheme", "dynamic") or
            !is(o, "expert_dtype", "fp4")) return error.UnsupportedQuantization;
        const bs = q.get("weight_block_size") orelse return error.MissingConfigField;
        if (bs != .array or bs.array.items.len != 2) return error.UnsupportedQuantization;
        for (bs.array.items) |v| if (v != .integer or v.integer != 128) return error.UnsupportedQuantization;
        return .native_fp4_fp8;
    }
};

fn is(o: std.json.ObjectMap, key: []const u8, value: []const u8) bool {
    const v = o.get(key) orelse return false;
    return v == .string and std.mem.eql(u8, v.string, value);
}

pub const Quant = enum(u32) { fp8, fp4 };

pub fn expert(key: []const u8) bool {
    return std.mem.indexOf(u8, key, ".ffn.experts.") != null;
}

pub fn dense(key: []const u8) bool {
    return std.mem.startsWith(u8, key, "vision.") or std.mem.startsWith(u8, key, "aligner.") or
        std.mem.eql(u8, key, "head") or std.mem.indexOf(u8, key, ".compressor.") != null or
        std.mem.indexOf(u8, key, ".markov_head.") != null or std.mem.endsWith(u8, key, ".confidence_head.proj");
}

pub fn matrix(t: st.Entry, scale: st.Entry, key: []const u8, k: usize, n: usize) !Quant {
    const fp4 = expert(key);
    if (k == 0 or n == 0 or k % 128 != 0 or t.rank != 2 or
        t.dtype != (if (fp4) st.DType.i8 else st.DType.f8_e4m3) or
        t.shape[0] != n or t.shape[1] != k / (if (fp4) @as(usize, 2) else 1))
        return error.BadNativeWeight;
    if (scale.dtype != .f8_e8m0 or scale.rank != 2 or
        scale.shape[0] != (if (fp4) n else (n + 127) / 128) or
        scale.shape[1] != k / (if (fp4) @as(usize, 32) else 128)) return error.BadNativeScale;
    return if (fp4) .fp4 else .fp8;
}

pub fn validateBytes(dtype: st.DType, bytes: []const u8) !void {
    if (dtype == .f8_e8m0) {
        if (std.mem.indexOfScalar(u8, bytes, 255) != null) return error.NonfiniteScale;
    } else if (dtype == .f8_e4m3) {
        for (bytes) |v| if ((v & 127) == 127) return error.NonfiniteWeight;
    }
}

test "packed FP4 accepts every byte but native FP8 and E8M0 reject NaN codes" {
    try validateBytes(.i8, &.{ 127, 255 });
    try validateBytes(.f8_e4m3, &.{ 0, 126, 128, 254 });
    try validateBytes(.f8_e8m0, &.{ 0, 127, 254 });
    try std.testing.expectError(error.NonfiniteWeight, validateBytes(.f8_e4m3, &.{255}));
    try std.testing.expectError(error.NonfiniteScale, validateBytes(.f8_e8m0, &.{255}));
}

test "I8 requires an expert name and its exact E8M0 scale geometry" {
    var w: st.Entry = .{ .dtype = .i8, .rank = 2, .shape = .{ 2048, 2048, 1, 1, 1 }, .begin = 0, .end = 0 };
    var s: st.Entry = .{ .dtype = .f8_e8m0, .rank = 2, .shape = .{ 2048, 128, 1, 1, 1 }, .begin = 0, .end = 0 };
    try std.testing.expectEqual(Quant.fp4, try matrix(w, s, "layers.0.ffn.experts.0.w1", 4096, 2048));
    try std.testing.expectError(error.BadNativeWeight, matrix(w, s, "layers.0.attn.wq_a", 4096, 2048));
    s.shape[1] = 32;
    try std.testing.expectError(error.BadNativeScale, matrix(w, s, "layers.0.ffn.experts.0.w1", 4096, 2048));
    w.dtype = .f8_e4m3;
    w.shape[1] = 4096;
    s.shape[0] = 16;
    try std.testing.expectEqual(Quant.fp8, try matrix(w, s, "layers.0.attn.wq_a", 4096, 2048));
    s.dtype = .f32;
    try std.testing.expectError(error.BadNativeScale, matrix(w, s, "layers.0.attn.wq_a", 4096, 2048));
}
