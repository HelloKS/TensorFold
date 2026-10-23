//! Fixed-width TP2 handshake refuses mismatched execution identities before any collective.
const std = @import("std");
pub const bytes = 72;
pub fn encode(checkpoint: [32]u8, execution: [32]u8, context: u32, rank: u32) ![bytes]u8 {
    if (rank > 1 or context == 0 or context > 262144) return error.UnsupportedProfile;
    var out: [bytes]u8 = undefined;
    @memcpy(out[0..32], &checkpoint);
    @memcpy(out[32..64], &execution);
    std.mem.writeInt(u32, out[64..68], context, .little);
    std.mem.writeInt(u32, out[68..72], rank, .little);
    return out;
}
pub fn validate(local: [bytes]u8, peer: []const u8) !void {
    if (peer.len != bytes) return error.BadRankHandshake;
    if (!std.mem.eql(u8, local[0..68], peer[0..68]) or std.mem.readInt(u32, peer[68..72], .little) != 1 - std.mem.readInt(u32, local[68..72], .little)) return error.RankIdentityMismatch;
}
test "TP2 refuses truncated frames, a different checkpoint, execution, context or rank" {
    const t = std.testing;
    const a = try encode(@splat(1), @splat(2), 262144, 0);
    const b = try encode(@splat(1), @splat(2), 262144, 1);
    try validate(a, &b);
    try t.expectError(error.BadRankHandshake, validate(a, b[0..71]));
    try t.expectError(error.RankIdentityMismatch, validate(a, &try encode(@splat(3), @splat(2), 262144, 1)));
    try t.expectError(error.RankIdentityMismatch, validate(a, &try encode(@splat(1), @splat(3), 262144, 1)));
    try t.expectError(error.RankIdentityMismatch, validate(a, &try encode(@splat(1), @splat(2), 32768, 1)));
    try t.expectError(error.RankIdentityMismatch, validate(a, &a));
}
