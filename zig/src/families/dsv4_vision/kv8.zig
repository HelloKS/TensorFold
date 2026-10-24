//! CPU reference for the e4m3 KV rows dsv4_text.cu stores under --kv-cache fp8; never an inference path.
const std = @import("std");
const fp8 = @import("core").quant.fp8;

pub fn bf(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

pub fn bfBits(x: f32) u16 {
    const b: u32 = @bitCast(x);
    return @truncate((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}

/// One value's e4m3 code, the sign kept apart the way __nv_cvt_float_to_fp8 does it.
pub fn code(x: f32) u8 {
    const sign: u8 = @intCast((@as(u32, @bitCast(x)) >> 24) & 0x80);
    const v: f32 = @floatCast(@abs(fp8.toE4m3(x)));
    const b: u32 = @bitCast(v);
    return sign | if (v < 0x1p-6) @as(u8, @intFromFloat(v * 512)) else @as(u8, @intCast((((b >> 23) - 120) << 3) | ((b >> 20) & 7)));
}

/// A code's BF16 value once its row's scale is folded back in, as tf_dsv4_text_gather8 writes it.
pub fn decoded(bits: u8, scale: f32) u16 {
    const value: f32 = @floatCast(fp8.e4m3(bits));
    return bfBits(value * scale);
}

/// One cache row: BF16 values in, e4m3 codes out, and the FP32 scale quant_row<WIDTH> records beside them.
pub fn pack(row: []const u16, out: []u8) !f32 {
    if (row.len == 0 or row.len != out.len) return error.BadRow;
    var amax: f32 = 0;
    for (row) |x| {
        const v = bf(x);
        if (!std.math.isFinite(v)) return error.NonFiniteRow;
        amax = @max(amax, @abs(v));
    }
    const inv: f32 = if (amax > 0) 448 / amax else 1;
    const scale: f32 = if (amax > 0) amax / 448 else 1;
    for (row, out) |x, *slot| slot.* = code(bf(x) * inv);
    return scale;
}

/// A row's bytes as the cache holds them: the codes, then the scale that brings them back to BF16.
pub fn rowBytes(dim: usize) usize {
    return dim + 4;
}

test "the reference quantizes the way kv_fp8.cuh does, ties even and a scale a row" {
    try std.testing.expectEqual(@as(u8, 0x38), code(1.0625)); // 17/16 is a tie between two e4m3 codes: rounds to even
    try std.testing.expectEqual(@as(u8, 0x3a), code(1.1875));
    try std.testing.expectEqual(@as(u8, 0x80), code(-0.0));
    var codes: [4]u8 = undefined;
    try std.testing.expectError(error.BadRow, pack(&.{}, &codes));
    try std.testing.expectError(error.NonFiniteRow, pack(&.{0x7f80, 0, 0, 0}, &codes)); // +inf would poison the row's scale
    var scale = try pack(&.{ 0, 0, 0, 0 }, &codes);
    try std.testing.expectEqual(@as(f32, 1), scale); // an empty row keeps the neutral scale its kernel writes
    try std.testing.expectEqual(@as(u8, 0), codes[0]);
    const values = [_]u16{ bfBits(3), bfBits(-1.5), bfBits(0.75), bfBits(0) };
    scale = try pack(&values, &codes);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0 / 448.0), scale, 1e-9);
    for (codes, values) |c, x| {
        const back = bf(decoded(c, scale));
        try std.testing.expectApproxEqAbs(@abs(bf(x)), @abs(back), 0.02); // a code times its row's scale lands near the value
    }
    var wide: [512]u16 = undefined;
    for (&wide, 0..) |*x, i| x.* = bfBits(@as(f32, @floatFromInt(i)) - 256);
    var many: [512]u8 = undefined;
    const big = try pack(&wide, &many);
    try std.testing.expectApproxEqAbs(@as(f32, 256.0 / 448.0), big, 1e-5);
    try std.testing.expectEqual(@as(usize, 516), rowBytes(512)); // 512 codes and one FP32 scale
}

test "every finite BF16 magnitude survives a row's round trip" {
    var codes: [1]u8 = undefined;
    for (0..65536) |raw| {
        const bits: u16 = @intCast(raw);
        if (!std.math.isFinite(bf(bits))) continue;
        const scale = try pack(&.{bits}, &codes);
        try std.testing.expect(std.math.isFinite(scale) and scale > 0);
        try std.testing.expect(std.math.isFinite(bf(decoded(codes[0], scale))));
    }
}
