//! The lane adapter owns two-rank prompt snapshots, image preparation and DSpark proposal requests.
const std = @import("std");
const lanes = @import("lanes");
const cluster = @import("cluster_backend");
const model = @import("text_model.zig");
const wire = @import("control.zig");
pub const Vision = @import("vision_serving.zig").Vision;
pub const State = struct {
    a: std.mem.Allocator,
    m: *model.Model,
    control: model.Control,
    leader: cluster.Leader,
    vision: ?Vision = null,
    next_snapshot: u64 = 1,
    pub fn send(s: *State, c: wire.Command) !void {
        if (s.m.failed) {
            if (c.op == .drop) { @import("distributed_state.zig").drop(s.m, c.id); return; }
            return error.RankStateInvalid;
        }
        const payload = try wire.encode(s.a, c);
        defer s.a.free(payload);
        s.leader.command(payload) catch |err| {
            switch (err) {
                error.PeerDown, error.PeerTimeout, error.BadRankAck, error.Corrupt, error.FrameTooLong => s.m.poison(),
                else => {},
            }
            return err;
        };
    }
};
const Saved = struct { id: u64, at: u32, identity: [32]u8 };
fn self(p: *anyopaque) *State { return @ptrCast(@alignCast(p)); }
fn owner(p: ?*anyopaque) !*lanes.Stream { return @ptrCast(@alignCast(p orelse return error.SnapshotNeedsStream)); }
pub fn bytes(p: *anyopaque, at: u32) u64 {
    const size = @import("state_plan.zig").bytes(self(p).m.c, at, false) catch return std.math.maxInt(u64);
    return 2 * size + @sizeOf(Saved);
}
pub fn ready(p: *anyopaque, budget: u64) !u64 {
    const s = self(p);
    if (s.vision) |*v| {
        const reserved = @min(budget / 8, 64 << 20);
        v.cache.limit = reserved;
        return reserved;
    }
    return 0;
}
pub fn save(p: *anyopaque, stream: ?*anyopaque, at: u32) anyerror!*anyopaque {
    const s = self(p);
    const current = try owner(stream);
    for (current.images) |img| if (at > img.at and at < img.at + img.tokens) return error.SnapshotCutsImage;
    const slot = s.leader.slot(current) orelse return error.UnknownStream;
    const saved = try s.a.create(Saved);
    errdefer s.a.destroy(saved);
    saved.* = .{ .id = s.next_snapshot, .at = at, .identity = s.m.identity };
    s.next_snapshot = try std.math.add(u64, s.next_snapshot, 1);
    try s.send(.{ .op = .save, .slot = slot, .id = saved.id, .at = at });
    return saved;
}
pub fn restore(p: *anyopaque, stream: ?*anyopaque, image: *anyopaque) anyerror!void {
    const s = self(p);
    const saved: *Saved = @ptrCast(@alignCast(image));
    if (!std.mem.eql(u8, &saved.identity, &s.m.identity)) return error.SnapshotIdentity;
    const slot = s.leader.slot(try owner(stream)) orelse return error.RestoredInPromptPass;
    try s.send(.{ .op = .restore, .slot = slot, .id = saved.id, .at = saved.at });
}
pub fn drop(p: *anyopaque, image: *anyopaque) void {
    const s = self(p);
    const saved: *Saved = @ptrCast(@alignCast(image));
    s.send(.{ .op = .drop, .id = saved.id }) catch { s.m.poison(); };
    s.a.destroy(saved);
}
pub fn begin(p: *anyopaque, current: *lanes.Stream, slot: u32) anyerror!u32 {
    const s = self(p);
    try s.send(.{ .op = .reset, .slot = slot });
    var at: u32 = 0;
    if (current.reuse.saved) |image| {
        const saved: *Saved = @ptrCast(@alignCast(image));
        if (saved.at != current.reuse.at or saved.at >= current.prompt_len or !std.mem.eql(u8, &saved.identity, &s.m.identity)) return error.SnapshotIdentity;
        try s.send(.{ .op = .restore, .slot = slot, .id = saved.id, .at = saved.at });
        at = saved.at;
    }
    if (current.images.len > 0) {
        const v = if (s.vision) |*vision| vision else return error.VisionDisabled;
        for (current.images) |img| {
            if (img.at + img.tokens <= at) continue;
            if (img.at < at) return error.SnapshotCutsImage;
            try v.install(s.m, &s.leader, slot, img);
        }
    }
    return at;
}
pub fn end(_: *anyopaque, current: *lanes.Stream, start: u32, proposed: u32) anyerror!u32 {
    var finish = proposed;
    for (current.images) |img| {
        const visible = img.at + 3 - img.at % 4;
        const after = img.at + img.tokens;
        if (start > visible and start < after) return error.ChunkCutsImage;
        if (finish > visible and finish < after) finish = if (visible > start) visible else after;
    }
    return finish;
}
pub fn unused(_: *anyopaque, _: []const u32, _: ?lanes.Sampling, _: u64, _: []u32) anyerror!void { return error.DraftNeedsSlot; }
pub fn guess(p: *anyopaque, slot: u32, history: []const u32, _: ?lanes.Sampling, position: u64, out: []u32) anyerror!void {
    const s = self(p);
    if (history.len == 0 or history.len != position or out.len > s.m.c.dspark_block_size) return error.InvalidDraft;
    var token: [4]u8 = undefined;
    std.mem.writeInt(u32, &token, history[history.len - 1], .little);
    try s.send(.{ .op = .draft, .slot = slot, .at = @intCast(out.len), .data = &token });
    @memcpy(out, s.m.proposed[0..out.len]);
}

pub fn room(p: *anyopaque, local: u64) !u64 {
    const s = self(p);
    var raw: [8]u8 = undefined;
    std.mem.writeInt(u64, &raw, local, .little);
    try s.send(.{ .op = .room, .data = &raw });
    return s.m.room_result;
}
