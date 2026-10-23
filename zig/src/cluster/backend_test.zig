//! Exactness across the cluster on the toy model: drafted == plain, shared == solo, and the same bits at 1, 2 and 4 nodes.
const std = @import("std");
const lanes = @import("lanes");
const exchange = @import("exchange.zig");
const round = @import("round.zig");
const toy = @import("toy.zig");
const backend = @import("backend.zig");
const toy_cluster = @import("toy_cluster.zig");

const Cluster = toy_cluster.Cluster;
const Variant = toy_cluster.Variant;

const gpa = std.testing.allocator;
const Sampling = lanes.Sampling;

/// A draft head that knows the model: one node's own decode, wrong every fifth position so rollbacks happen.
const Oracle = struct {
    w: *const toy.Weights,

    fn drafter(o: *Oracle) backend.Drafter {
        return .{ .ptr = o, .guess = guess };
    }

    fn guess(ptr: *anyopaque, history: []const u32, s: ?Sampling, position: u64, out: []u32) anyerror!void {
        const o: *Oracle = @ptrCast(@alignCast(ptr));
        const toks = try decode(o.w, history, s, out.len);
        defer gpa.free(toks);
        for (out, toks, 0..) |*d, t, j| d.* = if ((position + j) % 5 == 0) (t + 1) % o.w.dims.vocab else t;
    }
};

/// One node feeding `history` and drawing `n` tokens one at a time: the model's own decode.
pub fn decode(w: *const toy.Weights, history: []const u32, s: ?Sampling, n: usize) ![]u32 {
    const c = try Cluster.init(gpa, w, 1, .{});
    defer c.deinit();
    const l = &c.leader;
    const rows = try gpa.alloc(round.Row, history.len);
    defer gpa.free(rows);
    for (rows, history, 0..) |*r, t, i| r.* = .{ .slot = 0, .token = t, .index = @intCast(i), .sample = i + 1 == history.len };
    var logits = try l.run(rows);
    const out = try gpa.alloc(u32, n);
    for (out, 0..) |*t, j| {
        t.* = try l.draw(logits, s, history.len + j);
        gpa.free(logits);
        logits = try l.run(&.{.{ .slot = 0, .token = t.*, .index = @intCast(history.len + j), .sample = true }});
    }
    gpa.free(logits);
    return out;
}

fn config() !lanes.Config {
    var costs: [16]lanes.config.Cost = undefined;
    for (&costs, 1..) |*c, width| c.* = .{ .width = @intCast(width), .ms = 5.0 + 0.8 * @as(f64, @floatFromInt(width)) };
    return lanes.Config.init(gpa, .{ .exact_width = 16, .gpu_tokens = true, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 4, .window_costs = &costs, .mtp_step_ms = 0.5, .hidden_rows = true, .batch_rows = 32, .max_streams = 8, .draft_streams = true }, 16, 15);
}

const Case = struct { prompt: []const u32, max_new: u32 = 24, sampling: ?Sampling = null, drafts: bool = true };

/// The last serve's leader counters: rounds, and rows those rounds carried.
var last: struct { rounds: u64 = 0, rows: u64 = 0 } = .{};

/// Every case's emitted tokens, admitted together into one engine over an n-node cluster.
fn serve(w: *const toy.Weights, n: u32, cases: []const Case, v: Variant) ![][]u32 {
    var cfg = try config();
    defer cfg.deinit(gpa);
    const c = try Cluster.init(gpa, w, n, v);
    defer c.deinit();
    var oracle: Oracle = .{ .w = w };
    c.leader.drafter = oracle.drafter();
    var clock: lanes.fake.FixedClock = .{};
    var engine = lanes.Engine.init(gpa, &cfg, c.leader.backend(), clock.clock());
    defer engine.deinit();
    const streams = try gpa.alloc(lanes.Stream, cases.len);
    defer gpa.free(streams);
    for (cases, streams) |k, *s| s.* = try lanes.Stream.init(gpa, .{ .id = "s", .prompt = k.prompt, .max_new = k.max_new, .sampling = k.sampling, .drafts = k.drafts });
    defer for (streams) |*s| s.deinit(gpa);
    for (streams) |*s| try engine.addStream(s);
    while (engine.activeCount() > 0) try engine.step();
    last = .{ .rounds = c.leader.rounds, .rows = c.leader.rows_run };
    const out = try gpa.alloc([]u32, cases.len);
    for (out, streams) |*o, *s| o.* = try gpa.dupe(u32, s.emitted());
    return out;
}

fn free(runs: [][]u32) void {
    for (runs) |r| gpa.free(r);
    gpa.free(runs);
}

const p1 = [_]u32{ 3, 14, 15, 92, 65, 35, 89, 79 };
const p2 = [_]u32{ 27, 18, 28, 18, 28, 45, 90, 45, 23, 53 };
const hot: Sampling = .{ .seed = 77, .temperature = 0.8, .top_k = 0, .top_p = 0.95 };

test "four nodes: drafted rounds commit the model's own decode, greedy and sampled" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const w = try toy.Weights.init(arena.allocator(), .{}, 1);
    for ([_]?Sampling{ null, hot }) |s| {
        const drafted = try serve(&w, 4, &.{.{ .prompt = &p1, .sampling = s }}, .{});
        defer free(drafted);
        try std.testing.expect(last.rounds < drafted[0].len and last.rows > last.rounds + drafted[0].len);
        const plain = try serve(&w, 4, &.{.{ .prompt = &p1, .sampling = s, .drafts = false }}, .{});
        defer free(plain);
        const own = try decode(&w, &p1, s, drafted[0].len);
        defer gpa.free(own);
        try std.testing.expectEqualSlices(u32, plain[0], drafted[0]);
        try std.testing.expectEqualSlices(u32, own, drafted[0]);
    }
}

test "one, two and four nodes emit the same tokens; two streams sharing rounds commit what each commits alone" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const w = try toy.Weights.init(arena.allocator(), .{}, 2);
    const cases = [_]Case{ .{ .prompt = &p1 }, .{ .prompt = &p2, .sampling = hot, .max_new = 20 } };
    const four = try serve(&w, 4, &cases, .{});
    defer free(four);
    for ([_]u32{ 1, 2 }) |n| {
        const fewer = try serve(&w, n, &cases, .{});
        defer free(fewer);
        for (four, fewer) |x, y| try std.testing.expectEqualSlices(u32, x, y);
    }
    const one = try serve(&w, 4, cases[0..1], .{});
    defer free(one);
    const two = try serve(&w, 4, cases[1..2], .{});
    defer free(two);
    try std.testing.expectEqualSlices(u32, one[0], four[0]);
    try std.testing.expectEqualSlices(u32, two[0], four[1]);
}

/// Slot 0's next row's logits: alone, or in a round with slot 1's window and slot 2's prompt chunk; at `n` nodes.
fn rowBits(w: *const toy.Weights, n: u32, mixed: bool, v: Variant) ![]u32 {
    const c = try Cluster.init(gpa, w, n, v);
    defer c.deinit();
    const l = &c.leader;
    for ([_][]const u32{ &p1, &p2 }, 0..) |p, slot| {
        const rows = try gpa.alloc(round.Row, p.len);
        defer gpa.free(rows);
        for (rows, p, 0..) |*r, t, i| r.* = .{ .slot = @intCast(slot), .token = t, .index = @intCast(i), .sample = false };
        gpa.free(try l.run(rows));
    }
    const target: round.Row = .{ .slot = 0, .token = 41, .index = p1.len, .sample = true };
    const rows: []const round.Row = if (!mixed) &.{target} else &.{
        .{ .slot = 1, .token = 7, .index = p2.len, .sample = true },
        .{ .slot = 1, .token = 8, .index = p2.len + 1, .sample = true },
        target,
        .{ .slot = 2, .token = 5, .index = 0, .sample = false },
        .{ .slot = 2, .token = 6, .index = 1, .sample = false },
        .{ .slot = 2, .token = 9, .index = 2, .sample = true },
    };
    const logits = try l.run(rows);
    defer gpa.free(logits);
    const at: usize = if (mixed) 2 else 0;
    const vocab = w.dims.vocab;
    const out = try gpa.alloc(u32, vocab);
    for (out, logits[at * vocab ..][0..vocab]) |*o, x| o.* = @bitCast(x);
    return out;
}

test "a row's logits are the same bits alone, beside other streams' rows and a prompt chunk, and at 1 or 4 nodes" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const w = try toy.Weights.init(arena.allocator(), .{}, 3);
    const solo = try rowBits(&w, 4, false, .{});
    defer gpa.free(solo);
    for ([_]struct { u32, bool }{ .{ 4, true }, .{ 1, false }, .{ 1, true }, .{ 2, true } }) |case| {
        const got = try rowBits(&w, case[0], case[1], .{});
        defer gpa.free(got);
        try std.testing.expectEqualSlices(u32, solo, got);
    }
    const naive_one = try rowBits(&w, 1, false, .{ .naive = true });
    defer gpa.free(naive_one);
    const naive_four = try rowBits(&w, 4, false, .{ .naive = true });
    defer gpa.free(naive_four);
    try std.testing.expect(!std.mem.eql(u32, naive_one, naive_four));
}

test "the same rounds over the fabric's one-sided channel: four ranks emit what the memory exchange's four emit" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const w = try toy.Weights.init(arena.allocator(), .{}, 4);
    const cases = [_]Case{ .{ .prompt = &p1 }, .{ .prompt = &p2, .sampling = hot, .max_new = 16 } };
    const mem = try serve(&w, 4, &cases, .{});
    defer free(mem);
    const fab = try serve(&w, 4, &cases, .{ .fabric = true });
    defer free(fab);
    for (mem, fab) |x, y| try std.testing.expectEqualSlices(u32, x, y);
    const bits = try rowBits(&w, 4, true, .{ .fabric = true });
    defer gpa.free(bits);
    const solo = try rowBits(&w, 1, false, .{});
    defer gpa.free(solo);
    try std.testing.expectEqualSlices(u32, solo, bits);
}

test "by streams, split sums, Thunderbolt's two-sided links: the same bits at 1-4 nodes, the same tokens at 4" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const w = try toy.Weights.init(arena.allocator(), .{}, 5);
    const solo = try rowBits(&w, 1, false, .{});
    defer gpa.free(solo);
    const variants = [_]Variant{ .{ .by_streams = true }, .{ .split_sums = true }, .{ .by_streams = true, .split_sums = true }, .{ .by_streams = true, .fabric = true }, .{ .two_sided = true }, .{ .two_sided = true, .by_streams = true, .split_sums = true } };
    for (variants) |v| for ([_]u32{ 1, 2, 3, 4 }) |n| {
        const got = try rowBits(&w, n, true, v);
        defer gpa.free(got);
        try std.testing.expectEqualSlices(u32, solo, got);
    };
    const cases = [_]Case{ .{ .prompt = &p1 }, .{ .prompt = &p2, .sampling = hot, .max_new = 16 } };
    const plain = try serve(&w, 4, &cases, .{});
    defer free(plain);
    const dp = try serve(&w, 4, &cases, .{ .by_streams = true, .split_sums = true });
    defer free(dp);
    for (plain, dp) |x, y| try std.testing.expectEqualSlices(u32, x, y);
    const tb = try serve(&w, 4, &cases, .{ .two_sided = true, .by_streams = true, .split_sums = true });
    defer free(tb);
    for (plain, tb) |x, y| try std.testing.expectEqualSlices(u32, x, y);
}

test "every expert split by its intermediate columns: the same bits at 1, 2, 3 and 4 nodes, and exact rounds" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const w = try toy.Weights.init(arena.allocator(), .{}, 8);
    const one = try rowBits(&w, 1, false, .{ .split_experts = true });
    defer gpa.free(one);
    for ([_]Variant{ .{ .split_experts = true }, .{ .split_experts = true, .split_sums = true, .two_sided = true } }) |v| for ([_]u32{ 2, 3, 4 }) |n| {
        const got = try rowBits(&w, n, true, v);
        defer gpa.free(got);
        try std.testing.expectEqualSlices(u32, one, got);
    };
    const cases = [_]Case{ .{ .prompt = &p1 }, .{ .prompt = &p2, .sampling = hot, .max_new = 16 } };
    const four = try serve(&w, 4, &cases, .{ .split_experts = true, .two_sided = true });
    defer free(four);
    const solo = try serve(&w, 4, cases[0..1], .{ .split_experts = true });
    defer free(solo);
    const plain = try serve(&w, 2, &.{.{ .prompt = &p1, .drafts = false }}, .{ .split_experts = true });
    defer free(plain);
    try std.testing.expectEqualSlices(u32, solo[0], four[0]);
    try std.testing.expectEqualSlices(u32, plain[0], four[0]);
}

test "a draw refuses a row holding a NaN or an infinity, greedy or sampled; finite extremes draw as before" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const w = try toy.Weights.init(arena.allocator(), .{}, 1);
    const c = try Cluster.init(gpa, &w, 1, .{});
    defer c.deinit();
    const l = &c.leader;
    const row = try gpa.alloc(f32, 2048);
    defer gpa.free(row);
    for ([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }) |bad| for ([_]?Sampling{ hot, null }) |s| {
        for (row, 0..) |*x, i| x.* = if (i % 2 == 1) bad else @floatFromInt(i % 7);
        try std.testing.expectError(error.NonfiniteLogits, l.draw(row, s, 0));
        for ([_]usize{ 0, row.len - 1 }) |at| {
            @memset(row, 1);
            row[at] = bad;
            try std.testing.expectError(error.NonfiniteLogits, l.draw(row, s, 0));
        }
    };
    const max = std.math.floatMax(f32);
    const kept = [_]f32{ -max, std.math.floatTrueMin(f32), max, -0.0, max };
    var values: [kept.len]f64 = undefined;
    for (&values, kept) |*v, x| v.* = x;
    const ids = [_]u64{ 0, 1, 2, 3, 4 };
    for (0..16) |position| {
        try std.testing.expectEqual(2, try l.draw(&kept, null, position));
        try std.testing.expectEqual(try lanes.sampling.choose(gpa, &values, &ids, position, hot), try l.draw(&kept, hot, position));
    }
}

test "chunked cluster prefill matches whole prefill and cancelled chunks release both ranks" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const w = try toy.Weights.init(arena.allocator(), .{}, 29);
    const c = try Cluster.init(gpa, &w, 2, .{});
    defer c.deinit();
    c.leader.prefill_rows = 2;
    const b = c.leader.backend();
    var first = try lanes.Stream.init(gpa, .{ .id = "chunked", .prompt = &p1, .max_new = 4, .drafts = false });
    defer first.deinit(gpa);
    var steps: usize = 1;
    while (!try b.vtable.prefill_step.?(b.ptr, &first)) steps += 1;
    try std.testing.expectEqual((p1.len + 1) / 2, steps);
    const expected = try decode(&w, &p1, null, 1);
    defer gpa.free(expected);
    try std.testing.expectEqual(expected[0], try b.read(try b.first(&first, p1.len)));
    b.release(&first);
    var cut = try lanes.Stream.init(gpa, .{ .id = "cancelled", .prompt = &p1, .max_new = 4, .drafts = false });
    defer cut.deinit(gpa);
    try std.testing.expect(!try b.vtable.prefill_step.?(b.ptr, &cut));
    b.release(&cut);
    var next = try lanes.Stream.init(gpa, .{ .id = "next", .prompt = &p2, .max_new = 4, .drafts = false });
    defer next.deinit(gpa);
    try b.prefill(&next);
    const fresh = try decode(&w, &p2, null, 1);
    defer gpa.free(fresh);
    try std.testing.expectEqual(fresh[0], try b.read(try b.first(&next, p2.len)));
    b.release(&next);
}

// exchange.Mem rendezvouses every step, so these tests drive their own follower thread and their own Shard vtables.

const Done = struct { ok: bool = false, err: ?anyerror = null };

fn follow_to(alloc: std.mem.Allocator, x: exchange.Exchange, shard: backend.Shard, done: *Done) void {
    backend.follow(alloc, x, shard) catch |e| {
        done.err = e;
        return;
    };
    done.ok = true;
}

/// A shard whose only working vtable is `command`: enough for tests that never send a round.
const Sink = struct {
    bytes: [32]u8 = undefined,
    len: usize = 0,
    fn command(ptr: *anyopaque, bytes: []const u8) anyerror!void {
        const s: *Sink = @ptrCast(@alignCast(ptr));
        if (bytes.len > s.bytes.len) return error.PayloadTooLong;
        @memcpy(s.bytes[0..bytes.len], bytes);
        s.len = bytes.len;
    }
};

fn unreachable_round(_: *anyopaque, _: std.mem.Allocator, _: exchange.Exchange, _: []const round.Row, _: ?[]f32) anyerror!void {
    return error.TestingRound;
}
fn nothing(_: *anyopaque, _: u32, _: u32) void {}
fn nothing_slot(_: *anyopaque, _: u32) void {}
fn one_vocab(_: *anyopaque) u32 {
    return 1;
}

fn hooked(sink: *Sink) backend.Shard {
    return .{ .ptr = sink, .vtable = &.{ .forward = unreachable_round, .keep = nothing, .release = nothing_slot, .vocab = one_vocab, .command = Sink.command } };
}

fn mute() backend.Shard {
    return .{ .ptr = undefined, .vtable = &.{ .forward = unreachable_round, .keep = nothing, .release = nothing_slot, .vocab = one_vocab } };
}

/// A real peer rank whose release calls are recorded beside the cache they free.
const Tracked = struct {
    r: *toy.Rank,
    releases: [16]u32 = undefined,
    count: std.atomic.Value(u32) = .init(0),

    fn shard(t: *Tracked) backend.Shard {
        return .{ .ptr = t, .vtable = &.{ .forward = forward, .keep = keep, .release = release, .vocab = vocab } };
    }

    fn forward(ptr: *anyopaque, a: std.mem.Allocator, x: exchange.Exchange, rows: []const round.Row, logits: ?[]f32) anyerror!void {
        const t: *Tracked = @ptrCast(@alignCast(ptr));
        return t.r.forward(a, x, rows, logits);
    }

    fn keep(ptr: *anyopaque, slot: u32, len: u32) void {
        const t: *Tracked = @ptrCast(@alignCast(ptr));
        t.r.keep(slot, len);
    }

    fn release(ptr: *anyopaque, slot: u32) void {
        const t: *Tracked = @ptrCast(@alignCast(ptr));
        const i = t.count.load(.monotonic);
        t.releases[i] = slot;
        t.count.store(i + 1, .release);
        t.r.release(slot);
    }

    fn vocab(ptr: *anyopaque) u32 {
        const t: *Tracked = @ptrCast(@alignCast(ptr));
        return t.r.w.dims.vocab;
    }
};

/// A drafter that only works through the slot-aware hook and records the slot it was given.
const SlotOracle = struct {
    oracle: Oracle,
    slots: std.atomic.Value(u32) = .init(std.math.maxInt(u32)),

    fn drafter(o: *SlotOracle) backend.Drafter {
        return .{ .ptr = o, .guess = wrongPath, .guess_slot = guess };
    }

    fn wrongPath(_: *anyopaque, _: []const u32, _: ?Sampling, _: u64, _: []u32) anyerror!void {
        return error.PlainGuessPath;
    }

    fn guess(ptr: *anyopaque, slot: u32, history: []const u32, s: ?Sampling, position: u64, out: []u32) anyerror!void {
        const o: *SlotOracle = @ptrCast(@alignCast(ptr));
        o.slots.store(slot, .release);
        return Oracle.guess(&o.oracle, history, s, position, out);
    }
};

test "a family command reaches the shard hook on every rank" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const mem = try exchange.Mem.init(a, 2);
    defer mem.deinit();
    var lead_sink: Sink = .{};
    var peer_sink: Sink = .{};
    var done: Done = .{};
    const peer = try std.Thread.spawn(.{}, follow_to, .{ gpa, mem.endpoint(1), hooked(&peer_sink), &done });
    var l: backend.Leader = .{ .gpa = a, .x = mem.endpoint(0), .shard = hooked(&lead_sink) };
    defer l.deinit();
    try l.command("snapshot");
    try l.stop();
    peer.join();
    if (done.err) |e| return e;
    try std.testing.expect(done.ok);
    try std.testing.expectEqualStrings("snapshot", peer_sink.bytes[0..peer_sink.len]);
    try std.testing.expectEqualStrings("snapshot", lead_sink.bytes[0..lead_sink.len]); // the leader runs its own half
}

test "a shard without a command hook refuses a family command on the leader and the follower" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const mem = try exchange.Mem.init(a, 2);
    defer mem.deinit();
    var done: Done = .{};
    const peer = try std.Thread.spawn(.{}, follow_to, .{ gpa, mem.endpoint(1), mute(), &done });
    var l: backend.Leader = .{ .gpa = a, .x = mem.endpoint(0), .shard = mute() };
    defer l.deinit();
    // no stop() here: the follower leaves follow() on this command, and a stop broadcast would wait on it
    try std.testing.expectError(error.UnsupportedCommand, l.command("nope"));
    peer.join();
    if (done.err) |e| {
        try std.testing.expectEqual(error.UnsupportedCommand, e);
        return;
    }
    return error.MissingUnsupportedCommand;
}

test "the slot-aware drafter hook is the one the leader calls, with the lane's slot" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const w = try toy.Weights.init(arena.allocator(), .{}, 1);
    var cfg = try config();
    defer cfg.deinit(gpa);
    const c = try Cluster.init(gpa, &w, 1, .{});
    defer c.deinit();
    var rec: SlotOracle = .{ .oracle = .{ .w = &w } };
    c.leader.drafter = rec.drafter();
    var clock: lanes.fake.FixedClock = .{};
    var engine = lanes.Engine.init(gpa, &cfg, c.leader.backend(), clock.clock());
    defer engine.deinit();
    var one = try lanes.Stream.init(gpa, .{ .id = "slot1", .prompt = &p1, .max_new = 6, .drafts = true });
    defer one.deinit(gpa);
    try engine.addStream(&one);
    while (engine.activeCount() > 0) try engine.step();
    // the plain Drafter.guess above errors, so arriving here proves the leader routed to guess_slot
    try std.testing.expectEqual(@as(u32, 0), rec.slots.load(.acquire));
    const own = try decode(&w, &p1, null, one.emitted().len);
    defer gpa.free(own);
    try std.testing.expectEqualSlices(u32, own, one.emitted()); // and the drafts still commit the model's own decode
    rec.slots.store(std.math.maxInt(u32), .monotonic);
    var two = try lanes.Stream.init(gpa, .{ .id = "slot2", .prompt = &p2, .max_new = 6, .drafts = true });
    defer two.deinit(gpa);
    try engine.addStream(&two);
    while (engine.activeCount() > 0) try engine.step();
    try std.testing.expectEqual(@as(u32, 1), rec.slots.load(.acquire)); // the second lane got the second slot
}

test "a fresh leader restarts drawn handles after the last lane releases" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const w = try toy.Weights.init(arena.allocator(), .{}, 29);
    const c = try Cluster.init(gpa, &w, 1, .{});
    defer c.deinit();
    const b = c.leader.backend();
    var one = try lanes.Stream.init(gpa, .{ .id = "first", .prompt = &p1, .max_new = 4, .drafts = false });
    defer one.deinit(gpa);
    try b.prefill(&one);
    const h1 = try b.first(&one, p1.len);
    try std.testing.expectEqual(@as(u64, 0), h1);
    _ = try b.read(h1);
    const h2 = try b.queue(&one, .{ .handle = h1 }, p1.len + 1);
    _ = try b.read(h2); // the host contract: every handle is read while its lane is still in the rounds
    b.release(&one); // the last lane: this ends the drawn generation (backend.zig's clearRetainingCapacity)
    var two = try lanes.Stream.init(gpa, .{ .id = "second", .prompt = &p2, .max_new = 4, .drafts = false });
    defer two.deinit(gpa);
    try b.prefill(&two);
    const h3 = try b.first(&two, p2.len);
    try std.testing.expectEqual(@as(u64, 0), h3); // without the clear this handle would continue at 2
    const expected = try decode(&w, &p2, null, 1);
    defer gpa.free(expected);
    try std.testing.expectEqual(expected[0], try b.read(h3));
    // reading h1/h2 here is out of contract: their generation ended with the release above
    b.release(&two);
}

test "prefilling a stream twice releases its first slot on every rank" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try toy.Weights.init(arena.allocator(), .{}, 43);
    const mem = try exchange.Mem.init(a, 2);
    defer mem.deinit();
    const lead_rank = try a.create(toy.Rank);
    lead_rank.* = try toy.Rank.init(a, &w, 0, 2);
    defer lead_rank.deinit();
    const peer_rank = try a.create(toy.Rank);
    peer_rank.* = try toy.Rank.init(a, &w, 1, 2);
    defer peer_rank.deinit();
    var tracked: Tracked = .{ .r = peer_rank };
    var done: Done = .{};
    const peer = try std.Thread.spawn(.{}, follow_to, .{ gpa, mem.endpoint(1), tracked.shard(), &done });
    var l: backend.Leader = .{ .gpa = a, .x = mem.endpoint(0), .shard = toy_cluster.shardOf(lead_rank) };
    defer l.deinit();
    const b = l.backend();
    var s = try lanes.Stream.init(gpa, .{ .id = "twice", .prompt = &p1, .max_new = 4, .drafts = false });
    defer s.deinit(gpa);
    try b.prefill(&s); // opens slot 0 on both ranks; no release yet
    try std.testing.expectEqual(@as(u32, 0), tracked.count.load(.acquire));
    try b.prefill(&s); // re-prefill: release(slot 0) broadcast first, the recreated slot's round after it
    // the peer joins a step only after applying the previous one, so the release is already recorded here
    try std.testing.expectEqual(@as(u32, 1), tracked.count.load(.acquire));
    try std.testing.expectEqual(@as(u32, 0), tracked.releases[0]);
    b.release(&s); // releases the recreated slot (slot 1)
    try l.stop();
    peer.join();
    if (done.err) |e| return e;
    try std.testing.expect(done.ok);
    try std.testing.expectEqual(@as(u32, 2), tracked.count.load(.acquire));
    try std.testing.expectEqual(@as(u32, 1), tracked.releases[1]);
    // the hazard itself: the re-prefilled slot's caches were freed on the peer (and here), not leaked
    try std.testing.expect(peer_rank.caches.get(0) == null);
    try std.testing.expect(peer_rank.caches.get(1) == null);
    try std.testing.expect(lead_rank.caches.get(0) == null);
}

test "a trailing byte corrupts an ordinary command; only extensions carry payloads" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const frames = [_]round.Command{
        .{ .kind = .round, .number = 3, .rows = &.{.{ .slot = 0, .token = 1, .index = 0, .sample = true }} },
        .{ .kind = .release, .releases = &.{4} },
        .{ .kind = .stop },
    };
    for (frames) |f| {
        const bytes = try round.encode(a, f);
        const longer = try a.alloc(u8, bytes.len + 1);
        @memcpy(longer[0..bytes.len], bytes);
        longer[longer.len - 1] = 0;
        try std.testing.expectError(error.Corrupt, round.decode(a, longer));
    }
}