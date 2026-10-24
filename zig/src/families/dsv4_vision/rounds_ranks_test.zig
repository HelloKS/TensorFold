//! Fake ranks on the cluster plane over the batched round: identical plans, one exchange per round, joint failure.
const std = @import("std");
const lanes = @import("lanes");
const cluster = @import("cluster_backend");
const rounds = @import("rounds.zig");
const rowtab = @import("rowtab.zig");

const gpa = std.testing.allocator;
const testing = std.testing;
const exchange = cluster.exchange;
const round = cluster.round;

const vocab = 64;
const vh = vocab / 2;

const RSpan = struct { at: u32, start: u32, end: u32 };
const RMedia = struct { items: []const RSpan };
const RSeq = struct { slot: u32 = 0, len: u32 = 0, befores: u32 = 0 };

const Peer = struct {
    seen: [2]std.atomic.Value(u64),
    counts: [2]std.atomic.Value(u64),

    fn init() Peer {
        return .{ .seen = .{ .init(0), .init(0) }, .counts = .{ .init(0), .init(0) } };
    }

    /// A collective: publish my Nth count, wait for the peer's Nth, fail on a mismatch or a missing peer.
    fn arrive(p: *Peer, me: u32, n: u64, count: u64) !void {
        p.counts[me].store(count, .release);
        p.seen[me].store(n, .release);
        const other: usize = if (me == 0) 1 else 0;
        var spins: u32 = 0;
        while (p.seen[other].load(.acquire) < n) : (spins += 1) {
            if (spins > 5_000_000) return error.NcclMissing;
            std.Thread.yield() catch {};
        }
        if (p.counts[other].load(.acquire) != count) return error.NcclShapeMismatch;
    }
};

const Rank = struct {
    id: u32,
    peer: *Peer,
    batched: bool,
    dspark: bool = false,
    media: std.AutoHashMapUnmanaged(u32, RMedia) = .empty,
    seqs: std.AutoHashMapUnmanaged(u32, RSeq) = .empty,
    failed: bool = false,
    gathers: u64 = 0,
    expected: u64 = 0,
    media_rounds: u64 = 0,
    rows_run: u64 = 0,
    digest: u64 = 0,
    logits_slot: [vh]f32 = undefined,
    exch: [rowtab.cap_max * vh]f32 = undefined,

    pub fn sequence(r: *Rank, slot: u32) !*RSeq {
        const got = try r.seqs.getOrPut(gpa, slot);
        if (!got.found_existing) got.value_ptr.* = .{ .slot = slot };
        return got.value_ptr;
    }

    fn gather(r: *Rank, count: u64) !void {
        r.gathers += 1;
        try r.peer.arrive(r.id, r.gathers, count);
    }

    fn mix(r: *Rank, p: *const rounds.Plan) void {
        r.digest = fold(r.digest, @as(u64, @intFromEnum(p.fallback)));
        r.digest = fold(r.digest, p.n);
        r.digest = fold(r.digest, p.R);
        r.digest = fold(r.digest, p.n_sampled);
        r.digest = fold(r.digest, p.n_segs);
        for (p.rows[0..p.n]) |row| {
            r.digest = fold(r.digest, row.slot);
            r.digest = fold(r.digest, row.token);
            r.digest = fold(r.digest, row.index);
            r.digest = fold(r.digest, @intFromBool(row.sample));
        }
    }

    fn mixRows(r: *Rank, rows: []const round.Row) void {
        r.digest = fold(r.digest, 0xff);
        r.digest = fold(r.digest, @as(u64, rows.len));
        for (rows) |row| {
            r.digest = fold(r.digest, row.slot);
            r.digest = fold(r.digest, row.token);
            r.digest = fold(r.digest, row.index);
            r.digest = fold(r.digest, @intFromBool(row.sample));
        }
    }
};

fn fold(hash: u64, value: u64) u64 {
    var h = (hash ^ value) *% 0x100000001b3;
    h ^= h >> 29;
    return h;
}

fn logitsFor(token: u32, j: u32) f32 {
    return @floatFromInt(token *% 7 +% j);
}

fn rowFn(r: *Rank, seq: *RSeq, token: u32, sample: bool) !void {
    if (sample) {
        for (0..vh) |j| r.logits_slot[j] = logitsFor(token, @intCast(j));
    }
    seq.len += 1;
    r.rows_run += 1;
}

const RankSink = struct {
    pub fn stage(r: *Rank, p: *const rounds.Plan) !void {
        _ = r;
        _ = p;
    }
    pub fn snapshot(_: *Rank, seq: *RSeq) !void {
        seq.befores += 1;
    }
    pub fn headRow(r: *Rank, p: *const rounds.Plan, i: usize) !void {
        _ = p;
        @memcpy(r.exch[i * vh ..][0..vh], r.logits_slot[0..vh]);
    }
    pub fn round(r: *Rank, p: *const rounds.Plan, logits: ?[]f32) !void {
        try r.gather(@as(u64, p.R) * vh);
        const out = logits orelse return;
        for (0..p.n_sampled) |k| {
            const row = p.sampled[k];
            for (0..vh) |j| {
                out[k * vocab + j] = r.exch[row * vh + j];
                out[k * vocab + vh + j] = r.exch[row * vh + j];
            }
        }
    }
};

fn forwardFn(p: *anyopaque, a: std.mem.Allocator, x: exchange.Exchange, rows: []const round.Row, logits: ?[]f32) anyerror!void {
    _ = a;
    _ = x;
    const r: *Rank = @ptrCast(@alignCast(p));
    if (r.failed) return error.RankStateInvalid;
    errdefer r.failed = true;
    if (r.batched) {
        var pl = rounds.plan(r, rows, rowtab.cap_max);
        r.mix(&pl);
        if (pl.fallback == .media) r.media_rounds += 1;
        if (pl.fallback == .none) {
            r.expected += 1;
            return rounds.verify(r, RankSink, rowFn, &pl, logits);
        }
    }
    r.mixRows(rows);
    var drawn: usize = 0;
    for (rows) |row| {
        const seq = try r.sequence(row.slot);
        if (seq.len != row.index) return error.PositionMismatch;
        try rowFn(r, seq, row.token, row.sample);
        if (row.sample) {
            r.expected += 1;
            try r.gather(vh);
            if (logits) |out| {
                for (0..vh) |j| {
                    out[drawn * vocab + j] = r.logits_slot[j];
                    out[drawn * vocab + vh + j] = r.logits_slot[j];
                }
            }
            drawn += 1;
        }
    }
}

fn keepFn(p: *anyopaque, slot: u32, len: u32) void {
    const r: *Rank = @ptrCast(@alignCast(p));
    if (r.seqs.getPtr(slot)) |s| {
        if (len <= s.len) s.len = len;
    }
}

fn releaseFn(p: *anyopaque, slot: u32) void {
    const r: *Rank = @ptrCast(@alignCast(p));
    _ = r.seqs.remove(slot);
}

fn vocabFn(_: *anyopaque) u32 {
    return vocab;
}

fn shardOf(r: *Rank) cluster.Shard {
    return .{ .ptr = r, .vtable = &.{ .forward = forwardFn, .keep = keepFn, .release = releaseFn, .vocab = vocabFn } };
}

fn guessFn(_: *anyopaque, history: []const u32, _: ?lanes.Sampling, position: u64, out: []u32) anyerror!void {
    const last: u32 = if (history.len == 0) 1 else history[history.len - 1];
    for (out, 0..) |*o, j| o.* = @mod(last +% @as(u32, @intCast(position + j)), vocab - 1) + 1;
}

fn followRank(x: exchange.Exchange, shard: cluster.Shard, err: *?anyerror) void {
    cluster.follow(gpa, x, shard) catch |e| {
        err.* = e;
    };
}

const Run = struct {
    emitted: [][]u32,
    leader_rounds: u64,
    leader_gathers: u64,
    leader_expected: u64,
    follower_gathers: u64,
    leader_rows: u64,
    follower_rows: u64,
    leader_digest: u64,
    follower_digest: u64,
    leader_failed: bool,
    follower_failed: bool,
    follower_media: u64,
    follower_err: ?anyerror,
    step_err: ?anyerror,
};

fn config() !lanes.Config {
    var costs: [16]lanes.config.Cost = undefined;
    for (&costs, 1..) |*c, width| c.* = .{ .width = @intCast(width), .ms = 5.0 + 0.8 * @as(f64, @floatFromInt(width)) };
    return lanes.Config.init(gpa, .{ .exact_width = 16, .gpu_tokens = true, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 4, .window_costs = &costs, .mtp_step_ms = 0.5, .hidden_rows = true, .batch_rows = 24, .max_streams = 4, .draft_streams = true }, 16, 15);
}

/// One serve on two fake ranks: the leader thread runs the engine, the other thread runs cluster.follow too.
fn serve(batched: bool, prompts: []const []const u32, inject: struct { media_slot: ?u32 = null, seq_shift_slot: ?u32 = null }) !Run {
    var cfg = try config();
    defer cfg.deinit(gpa);
    const mem = try exchange.Mem.init(gpa, 2);
    defer mem.deinit();
    mem.timeout_ns = 2 * std.time.ns_per_s;
    var peer: Peer = .init();
    const lead = try gpa.create(Rank);
    lead.* = .{ .id = 0, .peer = &peer, .batched = batched };
    const peer_rank = try gpa.create(Rank);
    peer_rank.* = .{ .id = 1, .peer = &peer, .batched = batched };
    defer {
        lead.seqs.deinit(gpa);
        lead.media.deinit(gpa);
        peer_rank.seqs.deinit(gpa);
        peer_rank.media.deinit(gpa);
        gpa.destroy(lead);
        gpa.destroy(peer_rank);
    }
    var inject_spans: ?[]RSpan = null;
    defer if (inject_spans) |sp| gpa.free(sp);
    if (inject.media_slot) |slot| {
        const spans = try gpa.alloc(RSpan, 1);
        spans[0] = .{ .at = 0, .start = 0, .end = 1 << 20 };
        try peer_rank.media.put(gpa, slot, .{ .items = spans });
        inject_spans = spans;
    }
    if (inject.seq_shift_slot) |slot| {
        (try peer_rank.sequence(slot)).len = 1;
    }
    var b: cluster.Leader = .{ .gpa = gpa, .x = mem.endpoint(0), .shard = shardOf(lead), .prefill_rows = 8 };
    defer b.deinit();
    b.drafter = .{ .ptr = undefined, .guess = guessFn };
    var clock: lanes.fake.FixedClock = .{};
    var engine = lanes.Engine.init(gpa, &cfg, b.backend(), clock.clock());
    defer engine.deinit();
    const streams = try gpa.alloc(lanes.Stream, prompts.len);
    defer gpa.free(streams);
    for (prompts, streams) |prompt, *s| s.* = try lanes.Stream.init(gpa, .{ .id = "s", .prompt = prompt, .max_new = 8, .sampling = null, .drafts = true });
    defer for (streams) |*s| s.deinit(gpa);
    var follower_err: ?anyerror = null;
    const follower = try std.Thread.spawn(.{}, followRank, .{ mem.endpoint(1), shardOf(peer_rank), &follower_err });
    var joined = false;
    defer if (!joined) follower.join();
    var step_err: ?anyerror = null;
    for (streams) |*s| {
        engine.addStream(s) catch |e| {
            step_err = e;
            break;
        };
    }
    while (step_err == null and engine.activeCount() > 0) {
        engine.step() catch |e| {
            step_err = e;
            break;
        };
    }
    b.stop() catch {};
    follower.join();
    joined = true;
    const emitted = try gpa.alloc([]u32, prompts.len);
    for (emitted, streams) |*o, *s| o.* = try gpa.dupe(u32, s.emitted());
    return .{
        .emitted = emitted,
        .leader_rounds = b.rounds,
        .leader_gathers = lead.gathers,
        .leader_expected = lead.expected,
        .follower_gathers = peer_rank.gathers,
        .leader_rows = lead.rows_run,
        .follower_rows = peer_rank.rows_run,
        .leader_digest = lead.digest,
        .follower_digest = peer_rank.digest,
        .leader_failed = lead.failed,
        .follower_failed = peer_rank.failed,
        .follower_media = peer_rank.media_rounds,
        .follower_err = follower_err,
        .step_err = step_err,
    };
}

fn freeRun(run: *Run) void {
    for (run.emitted) |e| gpa.free(e);
    gpa.free(run.emitted);
}

const p1 = [_]u32{ 3, 14, 15, 92, 65, 35, 89, 79 };
const p2 = [_]u32{ 27, 18, 28, 18, 28, 45, 90, 45, 23, 53 };
const p3 = [_]u32{ 7, 7, 7, 12, 34, 19 };

test "batched ranks: one verify and one exchange per round, plans identical, tokens identical to serial" {
    const prompts = [_][]const u32{ &p1, &p2, &p3 };
    var batched_run = try serve(true, &prompts, .{});
    defer freeRun(&batched_run);
    try testing.expect(batched_run.step_err == null);
    try testing.expect(!batched_run.leader_failed and !batched_run.follower_failed);
    try testing.expectEqual(batched_run.leader_digest, batched_run.follower_digest);
    try testing.expect(batched_run.leader_rounds > 4);
    try testing.expectEqual(batched_run.leader_expected, batched_run.leader_gathers);
    try testing.expectEqual(batched_run.leader_gathers, batched_run.follower_gathers);
    try testing.expect(batched_run.leader_gathers <= batched_run.leader_rounds);
    try testing.expect(batched_run.leader_rows > batched_run.leader_rounds);
    try testing.expectEqual(batched_run.leader_rows, batched_run.follower_rows);
    var serial_run = try serve(false, &prompts, .{});
    defer freeRun(&serial_run);
    try testing.expect(serial_run.step_err == null);
    try testing.expect(serial_run.leader_gathers > serial_run.leader_rounds);
    for (batched_run.emitted, serial_run.emitted) |b, s| try testing.expectEqualSlices(u32, s, b);
}

test "divergence: a follower-only image span fails the round on both ranks together" {
    const prompts = [_][]const u32{ &p1, &p2, &p3 };
    var run = try serve(true, &prompts, .{ .media_slot = 0 });
    defer freeRun(&run);
    try testing.expect(run.follower_media > 0);
    try testing.expect(run.step_err != null);
    try testing.expect(run.leader_failed);
    try testing.expect(run.follower_failed);
    try testing.expect(run.follower_err != null);
}

test "divergence: a follower sequence off the round's index fails both ranks together" {
    const prompts = [_][]const u32{ &p1, &p2, &p3 };
    var run = try serve(true, &prompts, .{ .seq_shift_slot = 0 });
    defer freeRun(&run);
    try testing.expect(run.step_err != null);
    try testing.expectEqual(error.PositionMismatch, run.follower_err.?);
    try testing.expect(run.leader_failed);
    try testing.expect(run.follower_failed);
}
