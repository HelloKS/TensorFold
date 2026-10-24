//! Host tests for the graphed decode round: keys, the cut at the collectives, cache policy, ranks and eager parity.

const std = @import("std");
const cuda = @import("cuda");
const graphs = @import("graphs.zig");
const rg = @import("round_graph.zig");

const testing = std.testing;

test {
    _ = @import("round_graph_kv8_test.zig"); // the kv8 x graphs composition fixture (perf-kv's seven points), its own file
}
pub const no_stream: cuda.Stream = .{ .d = undefined, .handle = null };
pub const call: rg.Call = .{ .all_gather = .{ .send = 0x1000, .recv = 0x2000, .count = 3, .dtype = .f32 } };

/// A fake capture side: an exec is the ops its piece recorded; `log` is everything that ran, in order, both paths.
pub const Fake = struct {
    log: std.ArrayList(u8) = .empty,
    begins: usize = 0,
    ends: usize = 0,
    frees: usize = 0,
    calls: usize = 0,
    live: usize = 0,
    fail_begin: bool = false,
    fail_end: bool = false,
    open: ?*std.ArrayList(u8) = null,

    pub fn pieces(f: *Fake) rg.Pieces {
        return .{ .ctx = f, .begin = begin, .end = end, .launch = launch, .free = free, .step = step };
    }

    pub fn collective(f: *Fake) rg.Collective {
        return .{ .ctx = f, .allGather = colGather, .allReduce = colReduce };
    }

    fn say(f: *Fake, c: u8) !void {
        try f.log.append(testing.allocator, c);
    }

    fn begin(ctx: *anyopaque, _: cuda.Stream) anyerror!void {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        if (f.fail_begin) return error.CaptureFailed;
        f.begins += 1;
        const ops = try testing.allocator.create(std.ArrayList(u8));
        ops.* = .empty;
        f.open = ops;
    }

    fn end(ctx: *anyopaque, _: cuda.Stream, nodes: *usize) anyerror!*anyopaque {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        if (f.fail_end) {
            if (f.open) |ops| { // a failed end owns the half-recorded piece: free it, keep nothing
                ops.deinit(testing.allocator);
                testing.allocator.destroy(ops);
                f.open = null;
            }
            return error.CaptureFailed;
        }
        f.ends += 1;
        const ops = f.open.?;
        f.open = null;
        nodes.* = ops.items.len; // the fake's node count: what the step seam recorded, as cuGraphGetNodes would say
        f.live += 1;
        return ops;
    }

    fn step(ctx: *anyopaque, op: u8) void {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        f.open.?.append(testing.allocator, op) catch unreachable;
    }

    fn launch(ctx: *anyopaque, exec: *anyopaque, _: cuda.Stream) anyerror!void {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        const ops: *const std.ArrayList(u8) = @ptrCast(@alignCast(exec));
        for (ops.items) |op| try f.say(op);
    }

    fn free(ctx: *anyopaque, exec: *anyopaque) void {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        const ops: *std.ArrayList(u8) = @ptrCast(@alignCast(exec));
        ops.deinit(testing.allocator);
        testing.allocator.destroy(ops);
        f.frees += 1;
        f.live -= 1;
    }

    fn colGather(ctx: *anyopaque, _: u64, _: u64, _: usize, _: cuda.nccl.DataType, _: cuda.Stream) anyerror!void {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        f.calls += 1;
        try f.say('c');
    }

    fn colReduce(ctx: *anyopaque, _: u64, _: u64, _: usize, _: cuda.nccl.DataType, _: cuda.nccl.RedOp, _: cuda.Stream) anyerror!void {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        f.calls += 1;
        try f.say('c');
    }
};

/// One round's program (the batched verify's order): rows a b, the rank exchange, rows d e, on either path.
fn program(ctx: *anyopaque, rec: ?*rg.Recorder, collective: ?rg.Collective, stream: cuda.Stream) anyerror!void {
    const f: *Fake = @ptrCast(@alignCast(ctx));
    if (rec) |r| {
        try r.step('a');
        try r.step('b');
        try r.cut(call);
        try r.step('d');
        try r.step('e');
        return;
    }
    const col = collective.?;
    try f.say('a');
    try f.say('b');
    try col.allGather(col.ctx, call.all_gather.send, call.all_gather.recv, call.all_gather.count, call.all_gather.dtype, stream);
    try f.say('d');
    try f.say('e');
}

pub fn body(f: *Fake) rg.Body {
    return .{ .ctx = f, .run = program };
}

/// The shared cache's eager entry, mirroring Graphs.runBody: the same body, no Recorder, issuing its own collectives.
pub fn runBody(ctx: *anyopaque, stream: cuda.Stream) anyerror!void {
    const r: *const rg.Run = @ptrCast(@alignCast(ctx));
    try r.body.run(r.body.ctx, null, r.collective, stream);
}

fn makeEngine(f: *Fake, e: *rg.RoundEngine, s: rg.gc.Settings) graphs.Cache {
    e.* = .{ .gpa = testing.allocator, .pieces_side = f.pieces(), .collective = f.collective() };
    return graphs.Cache.init(testing.allocator, e.engine(), s);
}

test "cut at the collective: recorded with nothing run, replayed as the graphs and the call in record order" {
    var f: Fake = .{};
    defer f.log.deinit(testing.allocator);
    var e: rg.RoundEngine = undefined;
    var c = makeEngine(&f, &e, .{});
    defer c.deinit();
    var run = rg.Run{ .body = body(&f), .collective = f.collective() };
    const key: graphs.Key = .{ .rows = 5, .ctx = 2 };
    try testing.expectEqual(rg.gc.Outcome.captured, try c.run(key, no_stream, 7, .{ .ctx = &run, .run = runBody }));
    try testing.expectEqualStrings("abcde", f.log.items); // the capture's own replay ran the round once
    try testing.expectEqual(@as(usize, 1), f.calls);
    f.log.clearRetainingCapacity();
    try testing.expectEqual(rg.gc.Outcome.replayed, try c.run(key, no_stream, 7, .{ .ctx = &run, .run = runBody }));
    try testing.expectEqualStrings("abcde", f.log.items); // graph, collective, graph - no host idle gap inside a piece
    try testing.expectEqual(@as(usize, 2), f.calls);
    try testing.expectEqual(@as(usize, 2), f.begins); // two stretches of launches, one graph each
}

test "two collectives with no launches between them keep no empty graph" {
    var f: Fake = .{};
    defer f.log.deinit(testing.allocator);
    var rec: rg.Recorder = .{ .gpa = testing.allocator, .stream = no_stream, .capture = f.pieces(), .inner = f.collective() };
    try rec.begin();
    try rec.step('a');
    try rec.cut(call);
    try rec.cut(call); // nothing launched between the two calls
    try rec.step('b');
    var split = (try rec.finish()) orelse return error.NotRecorded;
    try testing.expectEqual(@as(usize, 4), split.pieces.len); // graph, call, call, graph - no empty piece
    try testing.expect(split.pieces[0] == .graph and split.pieces[1] == .call and split.pieces[2] == .call and split.pieces[3] == .graph);
    try testing.expectEqual(@as(usize, 3), f.ends); // the empty stretch still had to end its capture
    try split.launchOn(no_stream);
    try testing.expectEqualStrings("accb", f.log.items); // both collectives ran, in order, between the graphs
    split.deinit();
}

test "the key is (rows, context bucket): a different bucket or row count never shares a graph" {
    var f: Fake = .{};
    defer f.log.deinit(testing.allocator);
    var e: rg.RoundEngine = undefined;
    var c = makeEngine(&f, &e, .{});
    defer c.deinit();
    var run = rg.Run{ .body = body(&f), .collective = f.collective() };
    _ = try c.run(.{ .rows = 4, .ctx = 1 }, no_stream, 9, .{ .ctx = &run, .run = runBody });
    _ = try c.run(.{ .rows = 4, .ctx = 2 }, no_stream, 9, .{ .ctx = &run, .run = runBody });
    _ = try c.run(.{ .rows = 5, .ctx = 1 }, no_stream, 9, .{ .ctx = &run, .run = runBody });
    try testing.expectEqual(@as(usize, 3), c.count());
    try testing.expect(c.has(.{ .rows = 4, .ctx = 1 }));
    try testing.expect(!c.has(.{ .rows = 4, .ctx = 3 }));
    try testing.expectEqual(@as(usize, 6), f.live); // each key owns its two pieces; none was shared or freed early
}

test "the cap evicts the least recently used key, and a touched key is the last to go" {
    var f: Fake = .{};
    defer f.log.deinit(testing.allocator);
    var e: rg.RoundEngine = undefined;
    var c = makeEngine(&f, &e, .{ .max = 2 });
    defer c.deinit();
    var run = rg.Run{ .body = body(&f), .collective = f.collective() };
    _ = try c.run(.{ .rows = 1, .ctx = 0 }, no_stream, 1, .{ .ctx = &run, .run = runBody });
    _ = try c.run(.{ .rows = 2, .ctx = 0 }, no_stream, 1, .{ .ctx = &run, .run = runBody });
    try testing.expectEqual(@as(u64, 0), c.stats.evicted);
    _ = try c.run(.{ .rows = 3, .ctx = 0 }, no_stream, 1, .{ .ctx = &run, .run = runBody });
    try testing.expectEqual(@as(u64, 1), c.stats.evicted); // the rows=1 key (untouched longest) went first
    try testing.expect(!c.has(.{ .rows = 1, .ctx = 0 }));
    try testing.expectEqual(@as(usize, 2), f.frees); // its two pieces
    _ = try c.run(.{ .rows = 2, .ctx = 0 }, no_stream, 1, .{ .ctx = &run, .run = runBody }); // touch the rows=2 key
    _ = try c.run(.{ .rows = 4, .ctx = 0 }, no_stream, 1, .{ .ctx = &run, .run = runBody }); // so rows=3 is oldest now
    try testing.expect(c.has(.{ .rows = 2, .ctx = 0 }));
    try testing.expect(!c.has(.{ .rows = 3, .ctx = 0 }));
}

test "one new fingerprint (every rank's same step) drops every graph and recaptures" {
    var f: Fake = .{};
    defer f.log.deinit(testing.allocator);
    var e: rg.RoundEngine = undefined;
    var c = makeEngine(&f, &e, .{});
    defer c.deinit();
    var run = rg.Run{ .body = body(&f), .collective = f.collective() };
    _ = try c.run(.{ .rows = 1, .ctx = 0 }, no_stream, 11, .{ .ctx = &run, .run = runBody });
    _ = try c.run(.{ .rows = 2, .ctx = 0 }, no_stream, 11, .{ .ctx = &run, .run = runBody });
    try testing.expectEqual(@as(usize, 2), c.count());
    _ = try c.run(.{ .rows = 1, .ctx = 0 }, no_stream, 12, .{ .ctx = &run, .run = runBody }); // a reallocated table
    try testing.expectEqual(@as(u64, 2), c.stats.dropped);
    try testing.expectEqual(@as(usize, 1), c.count()); // the rows=1 key recaptured; rows=2 is a miss again
    try testing.expectEqual(@as(usize, 4), f.frees); // the two dropped keys' pieces; the recapture made two more
    try testing.expectEqual(@as(usize, 2), f.live);
}

test "a body that reads the device on the host abandons the recording: nothing runs, nothing is kept" {
    var f: Fake = .{};
    defer f.log.deinit(testing.allocator);
    const host = struct {
        var runs: usize = 0;

        fn run(ctx: *anyopaque, rec: ?*rg.Recorder, collective: ?rg.Collective, stream: cuda.Stream) anyerror!void {
            _ = ctx;
            if (rec) |r| {
                try r.step('a');
                try r.hostRead(); // the moe route's sync + download: an audit miss, so the recording is dropped
                return;
            }
            runs += 1;
            const col = collective.?;
            try col.allGather(col.ctx, 0, 0, 1, .f32, stream);
        }
    };
    var e: rg.RoundEngine = .{ .gpa = testing.allocator, .pieces_side = f.pieces(), .collective = f.collective() };
    var c = graphs.Cache.init(testing.allocator, e.engine(), .{});
    defer c.deinit();
    var run = rg.Run{ .body = .{ .ctx = undefined, .run = host.run }, .collective = f.collective() };
    try testing.expectEqual(rg.gc.Outcome.eager, try c.run(.{ .rows = 1, .ctx = 0 }, no_stream, 1, .{ .ctx = &run, .run = runBody }));
    try testing.expectEqual(@as(usize, 0), c.count()); // nothing kept
    try testing.expectEqual(@as(usize, 0), f.live); // every exec the abandoned capture made was freed
    try testing.expect(c.failed); // off after the failed capture: the plan's audit promised no host reads
    try testing.expectEqual(@as(usize, 1), host.runs); // the round ran exactly once, eagerly
    try testing.expectEqual(@as(usize, 1), f.calls); // and its collective ran exactly once
}

test "a failed capture keeps no piece and the cache runs the round eagerly instead" {
    var f: Fake = .{ .fail_end = true };
    defer f.log.deinit(testing.allocator);
    var e: rg.RoundEngine = .{ .gpa = testing.allocator, .pieces_side = f.pieces(), .collective = f.collective() };
    var c = graphs.Cache.init(testing.allocator, e.engine(), .{});
    defer c.deinit();
    var run = rg.Run{ .body = body(&f), .collective = f.collective() };
    try testing.expectEqual(rg.gc.Outcome.eager, try c.run(.{ .rows = 1, .ctx = 0 }, no_stream, 1, .{ .ctx = &run, .run = runBody }));
    try testing.expectEqual(@as(usize, 0), c.count());
    try testing.expectEqual(@as(usize, 0), f.live);
    try testing.expectEqualStrings("abcde", f.log.items); // the eager path ran the same program, same order
}

test "a collective issued raw while capturing is refused, never baked, and the round runs eager exactly once" {
    var f: Fake = .{};
    defer f.log.deinit(testing.allocator);
    const baked = struct {
        var runs: usize = 0;

        fn run(ctx: *anyopaque, rec: ?*rg.Recorder, collective: ?rg.Collective, stream: cuda.Stream) anyerror!void {
            if (rec) |r| {
                try r.step('a');
                const col = collective.?;
                try col.allGather(col.ctx, 0, 0, 1, .f32, stream); // the bug this guard exists for: raw NCCL mid-capture
                return;
            }
            runs += 1;
            const fl: *Fake = @ptrCast(@alignCast(ctx));
            try fl.say('a');
            const col = collective.?;
            try col.allGather(col.ctx, 0, 0, 1, .f32, stream);
        }
    };
    var e: rg.RoundEngine = .{ .gpa = testing.allocator, .pieces_side = f.pieces(), .collective = f.collective() };
    var c = graphs.Cache.init(testing.allocator, e.engine(), .{});
    defer c.deinit();
    var run = rg.Run{ .body = .{ .ctx = &f, .run = baked.run }, .collective = f.collective() };
    try testing.expectEqual(rg.gc.Outcome.eager, try c.run(.{ .rows = 1, .ctx = 0 }, no_stream, 1, .{ .ctx = &run, .run = runBody }));
    try testing.expectEqual(@as(usize, 1), baked.runs); // once, eagerly - the refused capture executed nothing
    try testing.expectEqual(@as(usize, 1), f.calls); // the eager one; the refused raw call never reached NCCL
    try testing.expectEqual(@as(usize, 0), c.count());
    try testing.expectEqual(@as(usize, 0), f.live);
}

test "the audit's word is assertable: a graphable body that host-reads runs eager once and matches the eager path" {
    var f: Fake = .{};
    defer f.log.deinit(testing.allocator);
    const hosty = struct {
        var runs: usize = 0;

        fn run(ctx: *anyopaque, rec: ?*rg.Recorder, collective: ?rg.Collective, stream: cuda.Stream) anyerror!void {
            if (rec) |r| {
                try r.step('a');
                try r.hostRead(); // the plan said graphable; the body read the device anyway: the audit lied
                return;
            }
            runs += 1;
            const fl: *Fake = @ptrCast(@alignCast(ctx));
            try fl.say('a');
            const col = collective.?;
            try col.allGather(col.ctx, 0, 0, 1, .f32, stream);
        }
    };
    var e: rg.RoundEngine = .{ .gpa = testing.allocator, .pieces_side = f.pieces(), .collective = f.collective() };
    var c = graphs.Cache.init(testing.allocator, e.engine(), .{});
    defer c.deinit();
    var run = rg.Run{ .body = .{ .ctx = &f, .run = hosty.run }, .collective = f.collective() };
    try testing.expectEqual(rg.gc.Outcome.eager, try c.run(.{ .rows = 1, .ctx = 0 }, no_stream, 1, .{ .ctx = &run, .run = runBody }));
    try testing.expectEqual(@as(usize, 1), hosty.runs); // once, eagerly - never twice
    const after_capture = try f.log.toOwnedSlice(testing.allocator);
    defer testing.allocator.free(after_capture);
    try testing.expectEqualStrings("ac", after_capture);
    try testing.expectEqual(@as(usize, 0), c.count()); // nothing kept from the abandoned recording
    try testing.expectEqual(@as(usize, 0), f.live);
    try testing.expect(c.failed);
    try run.body.run(&f, null, f.collective(), no_stream); // a later plain eager round: the same program, same order
    try testing.expectEqualSlices(u8, after_capture, f.log.items);
}

test "graphed and eager run the same program: one order of launches and collectives on both paths" {
    var f: Fake = .{};
    defer f.log.deinit(testing.allocator);
    var e: rg.RoundEngine = undefined;
    var c = makeEngine(&f, &e, .{});
    defer c.deinit();
    var run = rg.Run{ .body = body(&f), .collective = f.collective() };
    _ = try c.run(.{ .rows = 5, .ctx = 0 }, no_stream, 3, .{ .ctx = &run, .run = runBody });
    const graphed = try f.log.toOwnedSlice(testing.allocator);
    defer testing.allocator.free(graphed);
    try body(&f).run(&f, null, f.collective(), no_stream); // the same body with no cache at all
    try testing.expectEqualSlices(u8, "abcde", graphed); // piece, collective, piece
    try testing.expectEqualSlices(u8, graphed, f.log.items); // bits equal: one order on both paths
}

// -- the ranks: votes on misses only, never on a hit (cuda/graph_cache_ranks_test.zig's shape, the family's key) ------

const Board = struct {
    votes: [2][16]bool = undefined,
    cast: [2]std.atomic.Value(u32) = undefined,

    fn zero(b: *Board) void {
        b.cast[0].store(0, .monotonic);
        b.cast[1].store(0, .monotonic);
    }

    fn vote(b: *Board, rank: u1, ok: bool) !bool {
        const r = b.cast[rank].load(.monotonic);
        if (r == b.votes[rank].len) return error.TooManyVotes;
        b.votes[rank][r] = ok;
        b.cast[rank].store(r + 1, .release);
        const other = 1 - rank;
        var spins: usize = 0;
        while (b.cast[other].load(.acquire) <= r) {
            spins += 1;
            if (spins > 200_000) return error.PeerNeverVoted; // the off rank never votes (the digest test below)
            std.Thread.yield() catch {};
        }
        return ok and b.votes[other][r];
    }
};

var mixed_prog_runs: usize = 0;

fn liveRange(ctx: *anyopaque, from: usize, to: usize, _: cuda.Stream) anyerror!void {
    const f: *Fake = @ptrCast(@alignCast(ctx));
    mixed_prog_runs += 1;
    for (from..to) |i| try f.say(@intCast('0' + i));
}

const mixed = struct {
    fn run(ctx: *anyopaque, rec: ?*rg.Recorder, collective: ?rg.Collective, stream: cuda.Stream) anyerror!void {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        if (rec) |r| {
            try r.step('a');
            try r.step('b');
            try r.deferSteps(2, null); // step 2 reads the device on the host, step 3 launches with its value
            try r.cut(call);
            try r.step('d');
            try r.step('e');
            return;
        }
        try f.say('a');
        try f.say('b');
        try f.say('2');
        try f.say('3');
        const col = collective.?;
        try col.allGather(col.ctx, 0x1000, 0x2000, 3, .f32, stream);
        try f.say('d');
        try f.say('e');
    }

    fn short(ctx: *anyopaque, rec: ?*rg.Recorder, collective: ?rg.Collective, stream: cuda.Stream) anyerror!void {
        if (rec) |r| {
            try r.step('a');
            try r.step('b');
            try r.deferSteps(2, null);
            try r.cut(call);
            try r.step('d'); // the seventh step never comes: the program was mis-counted
            return;
        }
        return mixed.run(ctx, null, collective, stream);
    }
};

test "mixed round: device stretches replay as graphs and the deferred range re-runs live exactly once per round" {
    var f: Fake = .{};
    defer f.log.deinit(testing.allocator);
    mixed_prog_runs = 0;
    var e: rg.RoundEngine = .{ .gpa = testing.allocator, .pieces_side = f.pieces(), .collective = f.collective() };
    var c = graphs.Cache.init(testing.allocator, e.engine(), .{});
    defer c.deinit();
    var run = rg.Run{ .body = .{ .ctx = &f, .run = mixed.run, .program = .{ .ctx = &f, .n = 7, .run = liveRange } }, .collective = f.collective() };
    try testing.expectEqual(rg.gc.Outcome.captured, try c.run(.{ .rows = 2, .ctx = 0 }, no_stream, 1, .{ .ctx = &run, .run = runBody }));
    try testing.expectEqualStrings("ab23cde", f.log.items); // the round ran once, in program order, after the capture
    try testing.expectEqual(@as(usize, 1), mixed_prog_runs); // the host read ran live at launch, never during recording
    const first = try f.log.toOwnedSlice(testing.allocator);
    defer testing.allocator.free(first);
    try testing.expectEqual(rg.gc.Outcome.replayed, try c.run(.{ .rows = 2, .ctx = 0 }, no_stream, 1, .{ .ctx = &run, .run = runBody }));
    try testing.expectEqualSlices(u8, first, f.log.items); // the replay is the same program, in the same order
    try testing.expectEqual(@as(usize, 2), mixed_prog_runs); // the live stretch ran once per round, both rounds
    try testing.expectEqual(@as(usize, 1), c.count());
    try testing.expectEqual(@as(usize, 2), f.live); // two graph pieces live: (a,b) and (d,e)
}

test "a body that consumes the wrong number of program steps records nothing and runs eager" {
    var f: Fake = .{};
    defer f.log.deinit(testing.allocator);
    var e: rg.RoundEngine = .{ .gpa = testing.allocator, .pieces_side = f.pieces(), .collective = f.collective() };
    var c = graphs.Cache.init(testing.allocator, e.engine(), .{});
    defer c.deinit();
    var run = rg.Run{ .body = .{ .ctx = &f, .run = mixed.short, .program = .{ .ctx = &f, .n = 7, .run = liveRange } }, .collective = f.collective() };
    try testing.expectEqual(rg.gc.Outcome.eager, try c.run(.{ .rows = 2, .ctx = 0 }, no_stream, 1, .{ .ctx = &run, .run = runBody }));
    try testing.expectEqual(@as(usize, 0), c.count()); // a mis-counted program is refused, not half-recorded
    try testing.expectEqual(@as(usize, 0), f.live);
    try testing.expectEqualStrings("ab23cde", f.log.items); // eager once, the same order
}

test "deferring without a program is refused: a host-free round cannot defer steps" {
    var f: Fake = .{};
    defer f.log.deinit(testing.allocator);
    const nod = struct {
        fn run(ctx: *anyopaque, rec: ?*rg.Recorder, collective: ?rg.Collective, stream: cuda.Stream) anyerror!void {
            const ff: *Fake = @ptrCast(@alignCast(ctx));
            if (rec) |r| {
                try r.step('a');
                try r.deferSteps(1, null); // no program on this body: refused
                return;
            }
            try ff.say('a');
            const col = collective.?;
            try col.allGather(col.ctx, 0, 0, 1, .f32, stream);
        }
    };
    var e: rg.RoundEngine = .{ .gpa = testing.allocator, .pieces_side = f.pieces(), .collective = f.collective() };
    var c = graphs.Cache.init(testing.allocator, e.engine(), .{});
    defer c.deinit();
    var run = rg.Run{ .body = .{ .ctx = &f, .run = nod.run }, .collective = f.collective() };
    try testing.expectEqual(rg.gc.Outcome.eager, try c.run(.{ .rows = 1, .ctx = 0 }, no_stream, 1, .{ .ctx = &run, .run = runBody }));
    try testing.expectEqual(@as(usize, 0), c.count());
    try testing.expectEqual(@as(usize, 1), f.calls); // the eager collective ran once; the recording executed nothing
}

pub fn alwaysRoom() bool {
    return true;
}

const Rank = struct {
    board: *Board,
    id: u1,
    fake: Fake = .{},
    outcomes: [5]rg.gc.Outcome = undefined,
    err: ?anyerror = null,

    fn all(ctx: *anyopaque, ok: bool) anyerror!bool {
        const r: *Rank = @ptrCast(@alignCast(ctx));
        return r.board.vote(r.id, ok);
    }

    fn steps(r: *Rank, ks: []const graphs.Key) void {
        defer r.fake.log.deinit(testing.allocator);
        var e: rg.RoundEngine = .{ .gpa = testing.allocator, .pieces_side = r.fake.pieces(), .collective = r.fake.collective() };
        var c = graphs.Cache.init(testing.allocator, e.engine(), .{ .on = true, .max = 4 });
        defer c.deinit();
        c.agree = .{ .ctx = r, .all = all };
        c.room = alwaysRoom;
        var run = rg.Run{ .body = body(&r.fake), .collective = r.fake.collective() };
        for (ks, 0..) |k, i| {
            r.outcomes[i] = c.run(k, no_stream, 7, .{ .ctx = &run, .run = runBody }) catch |err| {
                r.err = err;
                return;
            };
        }
    }
};

const rank_keys = [_]graphs.Key{
    .{ .rows = 1, .ctx = 0 }, .{ .rows = 2, .ctx = 0 }, .{ .rows = 1, .ctx = 0 }, .{ .rows = 2, .ctx = 0 }, .{ .rows = 1, .ctx = 0 },
};

test "two ranks capture together and vote on each miss, never on a hit" {
    var board: Board = .{};
    board.zero();
    var ranks = [2]Rank{ .{ .board = &board, .id = 0 }, .{ .board = &board, .id = 1 } };
    const t = try std.Thread.spawn(.{}, Rank.steps, .{ &ranks[1], &rank_keys });
    ranks[0].steps(&rank_keys);
    t.join();
    const want = [_]rg.gc.Outcome{ .captured, .captured, .replayed, .replayed, .replayed };
    for (&ranks) |*r| {
        try testing.expectEqual(@as(?anyerror, null), r.err);
        try testing.expectEqualSlices(rg.gc.Outcome, &want, r.outcomes[0..rank_keys.len]);
        try testing.expectEqual(@as(u32, 4), board.cast[r.id].load(.acquire)); // room and capture per miss; hits none
    }
}

test "a rank whose cache is off never votes, so an on rank waits at its first vote and the digests tell it first" {
    const d: rg.gc.Settings = .{};
    try testing.expect(!std.mem.eql(u8, &d.digest(), &d.effective("off").digest())); // start-up agreement would differ
    var board: Board = .{};
    board.zero();
    var on: Rank = .{ .board = &board, .id = 0 };
    on.steps(&.{.{ .rows = 1, .ctx = 0 }}); // the off peer (settings.on=false) returns eager and casts no vote
    try testing.expectEqual(@as(?anyerror, error.PeerNeverVoted), on.err);
    try testing.expectEqual(@as(u32, 1), board.cast[0].load(.acquire));
    try testing.expectEqual(@as(u32, 0), board.cast[1].load(.acquire));
}

// -- the reset-failure fixture (flashnext/hip/state_test.zig's discipline, at the graph boundary) --------------------

/// A pooled slot whose reset failed keeps its buffers: the graphs that name those addresses never see them freed.
const Slot = struct {
    buffers: [2]u64 = .{ 0xA000, 0xB000 },
    freed: usize = 0,
    fail_reset: bool = false,

    fn reset(s: *Slot) !void {
        if (s.fail_reset) return error.ResetFailed;
    }

    fn freeBuffers(s: *Slot) void {
        s.freed += 2;
    }
};

test "a slot whose reset fails keeps its buffers: the graphs that name them never see them freed" {
    var f: Fake = .{};
    defer f.log.deinit(testing.allocator);
    var e: rg.RoundEngine = undefined;
    var c = makeEngine(&f, &e, .{});
    defer c.deinit();
    var slot: Slot = .{};
    var run = rg.Run{ .body = body(&f), .collective = f.collective() };
    const fp = rg.fingerprint(&.{ slot.buffers[0], slot.buffers[1] });
    _ = try c.run(.{ .rows = 1, .ctx = 0 }, no_stream, fp, .{ .ctx = &run, .run = runBody });
    try testing.expectEqual(@as(usize, 1), c.count()); // the graph names both buffers
    slot.fail_reset = true;
    try testing.expectError(error.ResetFailed, slot.reset()); // the pooled slot's reset failed: keep it as it stands
    _ = try c.run(.{ .rows = 1, .ctx = 0 }, no_stream, fp, .{ .ctx = &run, .run = runBody }); // a replay keeps it live
    try testing.expectEqual(@as(usize, 1), c.count()); // still held: its graph still names those addresses
    try testing.expectEqual(@as(usize, 0), f.frees); // nothing freed while a graph names the slot's buffers
    try testing.expectEqual(@as(usize, 0), slot.freed); // and the buffers themselves stay owned by the kept slot
    slot.fail_reset = false;
    try slot.reset(); // only a clean reset may return it to the pool
    c.dropAll(); // the graph that named them goes first; the buffers outlive every graph naming them
    try testing.expectEqual(@as(usize, 2), f.frees); // its two pieces
    slot.freeBuffers();
    try testing.expectEqual(@as(usize, 2), slot.freed);
}
