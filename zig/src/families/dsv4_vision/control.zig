//! Versioned, bounded family commands carried by the shared cluster command envelope.
const std = @import("std");
pub const limit = 16 << 20;
pub const Op = enum(u8) { reset = 1, save, restore, drop, draft, media, room };
pub const Command = struct { op: Op, slot: u32 = 0, id: u64 = 0, at: u32 = 0, data: []const u8 = &.{} };
pub fn encode(a: std.mem.Allocator, c: Command) ![]u8 {
    if (c.data.len > limit - 18) return error.FrameTooLong;
    const out = try a.alloc(u8, 18 + c.data.len);
    out[0] = 1;
    out[1] = @backingInt(c.op);
    std.mem.writeInt(u32, out[2..6], c.slot, .little);
    std.mem.writeInt(u64, out[6..14], c.id, .little);
    std.mem.writeInt(u32, out[14..18], c.at, .little);
    @memcpy(out[18..], c.data);
    return out;
}
pub fn decode(bytes: []const u8) !Command {
    if (bytes.len < 18 or bytes.len > limit or bytes[0] != 1) return error.BadFamilyCommand;
    return .{ .op = std.enums.fromInt(Op, bytes[1]) orelse return error.BadFamilyCommand,
        .slot = std.mem.readInt(u32, bytes[2..6], .little), .id = std.mem.readInt(u64, bytes[6..14], .little),
        .at = std.mem.readInt(u32, bytes[14..18], .little), .data = bytes[18..] };
}
test "family commands retain identifiers and reject unknown versions and operations" {
    const a = std.testing.allocator;
    const bytes = try encode(a, .{ .op = .media, .slot = 3, .id = 77, .at = 123, .data = "payload" });
    defer a.free(bytes);
    const c = try decode(bytes);
    try std.testing.expectEqual(@as(u64, 77), c.id);
    try std.testing.expectEqual(@as(u32, 123), c.at);
    try std.testing.expectEqualStrings("payload", c.data);
    bytes[0] = 2;
    try std.testing.expectError(error.BadFamilyCommand, decode(bytes));
    bytes[0] = 1; bytes[1] = 255;
    try std.testing.expectError(error.BadFamilyCommand, decode(bytes));
}
