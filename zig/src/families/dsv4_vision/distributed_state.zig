//! Snapshot operations finish on both ranks before a saved prefix is published to the cache.
const std = @import("std");
const wire = @import("control.zig");
const Snapshot = @import("text_state.zig").Snapshot;

pub fn apply(m: anytype, c: wire.Command) !void {
    if (m.failed and c.op != .drop) return error.RankStateInvalid;
    switch (c.op) {
        .reset => try (try m.sequence(c.slot)).reset(m.stream),
        .save => {
            const seq = try m.sequence(c.slot);
            if (seq.len != c.at or m.snapshots.contains(c.id)) return error.SnapshotPosition;
            var image = try Snapshot.capture(seq, false, m.stream);
            errdefer image.free();
            try m.snapshots.put(m.a, c.id, image);
        },
        .restore => {
            const image = m.snapshots.get(c.id) orelse return error.SnapshotStale;
            if (image.at != c.at) return error.SnapshotPosition;
            const seq = try m.sequence(c.slot);
            try image.restore(seq, m.stream);
        },
        .drop => drop(m, c.id),
        .draft => {
            if (c.data.len != 4 or c.at > m.c.dspark_block_size) return error.InvalidDraft;
            const token = std.mem.readInt(u32, c.data[0..4], .little);
            try @import("dspark.zig").propose(m, try m.sequence(c.slot), token, m.proposed[0..c.at]);
        },
        .media => try @import("vision_serving.zig").receive(m, c),
        .room => {
            if (c.data.len != 8) return error.BadFamilyCommand;
            m.room_result = if (m.rank == 0) std.mem.readInt(u64, c.data[0..8], .little)
                else try @import("text_admission.zig").room(m.a, m.io, m.ctx);
        },
    }
}
pub fn drop(m: anytype, id: u64) void {
    if (m.snapshots.fetchRemove(id)) |entry| { var image = entry.value; image.free(); }
}
pub fn command(m: anytype, bytes: []const u8) !void {
    const Actions = struct {
        pub fn run(rank: anytype, c: wire.Command) !void {
            apply(rank, c) catch |err| {
                std.log.warn("rank {d} {s} failed: {t}", .{rank.rank, @tagName(c.op), err});
                return err;
            };
        }
        pub const discard = drop;
        pub fn fail(rank: anytype) void { rank.poison(); }
    };
    return commandWith(m, bytes, Actions) catch |err| {
        switch (err) {
            error.PeerDown, error.PeerTimeout, error.BadRankAck, error.BadFamilyCommand, error.FrameTooLong => {
                m.poison();
            },
            else => {},
        }
        return err;
    };
}
pub fn commandWith(m: anytype, bytes: []const u8, comptime Actions: type) !void {
    const c = try wire.decode(bytes);
    const result = Actions.run(m, c);
    var local: [9]u8 = @splat(0);
    local[0] = if (result) |_| 0 else |_| 1;
    std.mem.writeInt(u64, local[1..9], m.room_result, .little);
    var peer: [9]u8 = undefined;
    if (m.rank == 0) {
        try m.link.send(&local);
        if ((try m.link.recv(&peer)).len != 9) return error.BadRankAck;
    } else {
        if ((try m.link.recv(&peer)).len != 9) return error.BadRankAck;
        try m.link.send(&local);
    }
    if (c.op == .room) m.room_result = @min(m.room_result, std.mem.readInt(u64, peer[1..9], .little));
    if (local[0] != 0 or peer[0] != 0) {
        if (c.op == .save) Actions.discard(m, c.id);
        if (c.op == .restore or c.op == .draft) Actions.fail(m);
        if (m.rank == 0) {
            try result;
            return error.PeerOperationFailed;
        }
    }
}
