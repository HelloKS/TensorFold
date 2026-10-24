//! The lane adapter owns two-rank prompt snapshots, image preparation and DSpark proposal requests.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const cluster = @import("cluster_backend");
const kv = @import("core").kv_pool;
const model = @import("text_model.zig");
const wire = @import("control.zig");
const kv_pages = @import("kv_pages.zig");
const states = @import("text_state.zig");
pub const Vision = @import("vision_serving.zig").Vision;
pub const State = struct {
    a: std.mem.Allocator,
    m: *model.Model,
    control: model.Control,
    leader: cluster.Leader,
    vision: ?Vision = null,
    next_snapshot: u64 = 1,
    layout: ?kv.Layout = null, // the layout the host planned this open's streams from (null: each keeps its own window)
    pages: ?kv_pages.Pages = null, // the plan's page ids and one lease per stream, made by attach
    kv8: bool = false, // --kv-cache fp8: e4m3 rows, so a page and a kept state's slot are smaller
    peer: ?kv_pages.Peer = null, // rank 1's room when it is the smaller: the host admits by the smallest
    snap_host: bool = false, // the cache's kept states in the host's pinned memory, as the native host placed them
    snap_rows: ?cuda.DeviceBuffer = null, // their slots on the card, when the cache fit there
    snap_pin: ?cuda.HostBuffer = null, // ... or in pinned host memory, when it did not
    snap_ring: ?states.Ring = null, // the host's budget, handed out one kept state at a time
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
    const size = @import("state_plan.zig").bytes(self(p).m.c, at, false, self(p).kv8) catch return std.math.maxInt(u64);
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

/// The host's page plan for this open: the layout its streams come from, after the room both ranks agree exactly once.
pub fn plan(s: *State, o: anytype) !?kv.Layout {
    s.kv8 = if (@hasField(@TypeOf(o), "kv8")) o.kv8 else false;
    try agree(s, try @import("text_admission.zig").room(s.a, s.m.io, s.m.ctx)); // once per open, as the host's rankRoom did
    s.layout = kv_pages.layoutOf(s.m.c, o);
    return s.layout;
}

/// The room the ranks hold, agreed once over the control link: rank 1's when it is the smaller, so both admit by it.
fn agree(s: *State, local: u64) !void {
    const held = try room(s, local); // the ranks' own command leaves both holding the smaller figure
    s.peer = kv_pages.smaller(local, held, 1); // the host names the smaller rank in its own refusal to admit
}

/// The plan's streams and pages, one lease each, from the layout the host just planned the streams from.
pub fn attach(p: *anyopaque, streams: u32, pages: usize) !void {
    const s = self(p);
    const l = s.layout orelse return error.NoPages;
    if (s.pages) |*held| held.deinit(); // a reopen's leases go back before the new plan's are made
    s.pages = null; // never leave a freed plan behind if the new one cannot be made
    s.pages = try kv_pages.Pages.init(s.a, l, streams, pages);
}

/// Where the kept states go, as the native host found the cache fits: the host's pinned memory, or the card.
pub fn place(p: *anyopaque, host_mode: bool) void {
    self(p).snap_host = kv_pages.placed(kv_pages.enabled(), host_mode); // placement learns to choose only on the paged path
}

/// The cache's `n` kept-state slots, made where `place` put them; 0: the old per-save device buffer still stands.
pub fn slots(p: *anyopaque, n: usize) !u64 {
    const s = self(p);
    const l = s.layout orelse return 0; // the pool unnamed: nothing to make, and the host's own accounting stands
    if (n == 0) return 0;
    dropSlots(s); // a reopen's ring is gone before the new one is charged for
    const want = n * l.snap_bytes; // exactly what the host budgeted a kept state at, so a slot cannot come up short
    const taken = try s.a.alloc(bool, n);
    errdefer s.a.free(taken);
    @memset(taken, false);
    if (s.snap_host) {
        s.snap_pin = try cuda.HostBuffer.allocMapped(s.m.d, want); // the engine reaches its own states through the device map
        errdefer if (s.snap_pin) |*pin| pin.free();
        s.snap_ring = .{ .rows = .{ .d = s.m.d, .ptr = try s.snap_pin.?.device(), .len = want, .borrowed = true }, .taken = taken, .bytes = l.snap_bytes };
    } else {
        s.snap_rows = try cuda.DeviceBuffer.alloc(s.m.d, want);
        errdefer if (s.snap_rows) |*rows| rows.free();
        s.snap_ring = .{ .rows = s.snap_rows.?, .taken = taken, .bytes = l.snap_bytes };
    }
    states.ring = &s.snap_ring.?; // from here a kept state is written into the host's slot, not a buffer of our own
    return want;
}

/// The pages and the cache's slots go back before the ranks close; nothing to give when the pool was never named.
pub fn free(s: *State) void {
    if (s.pages) |*held| held.deinit();
    s.pages = null; // a second close, or a close after a failed reopen, must not hand the same pages back twice
    dropSlots(s);
}

/// One ring's worth of memory and bookkeeping, back the way it came: captures fall back to their own buffers after it.
fn dropSlots(s: *State) void {
    if (s.snap_ring) |*r| {
        if (states.ring) |live| {
            if (live == r) states.ring = null;
        }
        s.a.free(r.taken);
    }
    s.snap_ring = null;
    if (s.snap_rows) |*rows| rows.free();
    s.snap_rows = null;
    if (s.snap_pin) |*pin| pin.free();
    s.snap_pin = null;
}
