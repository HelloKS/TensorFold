//! The actual two-rank publication barrier under asymmetric allocation and restore failures.
const std = @import("std");
const state = @import("distributed_state.zig");
const wire = @import("control.zig");
const Bus = struct { data: [2][9]u8 = undefined, ready: [2]std.atomic.Value(bool) = .{ .init(false), .init(false) } };
const Link = struct {
    bus: *Bus,
    rank: usize,
    pub fn send(l: Link, bytes: []const u8) !void {
        if (bytes.len != 9) return error.BadAck;
        @memcpy(&l.bus.data[l.rank], bytes);
        l.bus.ready[l.rank].store(true, .release);
    }
    pub fn recv(l: Link, bytes: []u8) ![]u8 {
        const started = std.Io.Clock.awake.now(std.Options.debug_io).nanoseconds;
        while (!l.bus.ready[1 - l.rank].load(.acquire)) {
            if (std.Io.Clock.awake.now(std.Options.debug_io).nanoseconds - started > 2 * std.time.ns_per_s) return error.PeerTimeout;
            std.atomic.spinLoopHint();
        }
        @memcpy(bytes, &l.bus.data[1 - l.rank]);
        return bytes;
    }
};
const Comm = struct { aborted: bool = false, pub fn abort(c: *Comm) void { c.aborted = true; } };
const Rank = struct { rank: u32, link: Link, failed: bool = false, saved: bool = false, refuse: bool, room_result: u64, comm: Comm = .{} };
const Actions = struct {
    pub fn run(r: *Rank, c: wire.Command) !void {
        if (r.refuse) return error.OutOfMemory;
        if (c.op == .save) r.saved = true;
    }
    pub fn discard(r: *Rank, _: u64) void { r.saved = false; }
    pub fn fail(r: *Rank) void { r.failed = true; r.comm.abort(); }
};
const Worker = struct {
    rank: *Rank,
    command: []const u8,
    err: ?anyerror = null,
    fn run(w: *Worker) void { state.commandWith(w.rank, w.command, Actions) catch |err| { w.err = err; }; }
};
fn check(op: wire.Op, refuse: ?u32) !void {
    const a = std.testing.allocator;
    const command = try wire.encode(a, .{ .op = op, .slot = 3, .id = 11, .at = 128 });
    defer a.free(command);
    var bus: Bus = .{};
    var head: Rank = .{ .rank = 0, .link = .{ .bus = &bus, .rank = 0 }, .refuse = refuse == 0, .room_result = 100 };
    var peer: Rank = .{ .rank = 1, .link = .{ .bus = &bus, .rank = 1 }, .refuse = refuse == 1, .room_result = 40 };
    var worker: Worker = .{ .rank = &peer, .command = command };
    const thread = try std.Thread.spawn(.{}, Worker.run, .{&worker});
    const result = state.commandWith(&head, command, Actions);
    thread.join();
    try std.testing.expect(worker.err == null);
    if (refuse) |rank| {
        if (rank == 0) try std.testing.expectError(error.OutOfMemory, result)
        else try std.testing.expectError(error.PeerOperationFailed, result);
        if (op == .save) try std.testing.expect(!head.saved and !peer.saved);
        if (op == .restore) try std.testing.expect(head.failed and peer.failed and head.comm.aborted and peer.comm.aborted);
    } else {
        try result;
        if (op == .save) try std.testing.expect(head.saved and peer.saved);
        if (op == .room) try std.testing.expectEqual(@as(u64, 40), head.room_result);
    }
}
test "distributed snapshots publish only after both ranks succeed" {
    try check(.save, null);
    try check(.save, 0);
    try check(.save, 1);
}
test "restore failures poison both ranks and memory admission uses the smaller pool" {
    try check(.restore, 0);
    try check(.restore, 1);
    try check(.room, null);
}
