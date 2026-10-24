//! kv8 x graphs composition: production RoundAddrs names every baked address; the fixture constrains the real thing.

const std = @import("std");
const graphs = @import("graphs.zig");
const rg = @import("round_graph.zig");
const rgt = @import("round_graph_test.zig");
const st = @import("text_state.zig");
const runtime = @import("text_runtime.zig");

const testing = std.testing;

const ring_bytes: usize = 4096;

/// The kv8 fingerprint folded the production way: the real text_runtime.fingerprint through the shared digestWord.
fn folded(kv8: bool) ![32]u8 {
    return runtime.fingerprint(testing.allocator, std.testing.io, null, .native_fp4_fp8, false, 0, kv8, false);
}

/// A fabricated device buffer: the graphWords walk only ever reads ptr, so a real struct needs no driver.
fn fake(ptr: u64) @FieldType(st.Sequence, "taps") {
    return .{ .d = undefined, .ptr = ptr, .len = 0x1000, .borrowed = true };
}

/// A ratio-4 layer: window rows plus a compressor and an indexer, exactly the shapes part() walks.
fn richLayer(base: u64) st.Layer {
    return .{
        .kv = fake(base),
        .compressor = .{ .kv = fake(base + 0x100), .scores = fake(base + 0x200), .cache = fake(base + 0x300) },
        .indexer = .{ .kv = fake(base + 0x400), .scores = fake(base + 0x500), .cache = fake(base + 0x600) },
    };
}

fn plainLayer(base: u64) st.Layer {
    return .{ .kv = fake(base), .compressor = .{ .kv = fake(base + 0x100), .scores = fake(base + 0x200), .cache = fake(base + 0x300) } };
}

const Store = struct {
    layers: [2]st.Layer,
    draft: [1]st.Layer, // dspark's draft rows are baked by an absorbing round too
    journal: [0]?st.Snapshot = .{},
};

fn makeStore(base: u64) Store {
    return .{
        .layers = .{ richLayer(base + 0x2000), plainLayer(base + 0x2800) },
        .draft = .{.{ .kv = fake(base + 0x2E00) }},
    };
}

/// One batched round over two REAL Sequences: only fields move here; the word list is perf-kv's production graphWords.
const Round = struct {
    base: rg.RoundAddrs = .{ .table = 0x1000, .ids = 0x2000, .exch = 0x3000, .recv = 0x4000, .out = 0x5000, .kv = 0x6000, .gate = 0x6100, .up = 0x6200, .selected = 0x7100, .identity = 0, .seqs = &.{} },
    cfg: @FieldType(st.Sequence, "c") = undefined,
    stores: [2]Store = undefined,
    seqs: [2]st.Sequence = undefined,
    fails: [2]bool = .{ false, false },
    freed: [2]usize = .{ 0, 0 },

    fn blank() !Round {
        return .{
            .cfg = try @import("config.zig").Config.parse(testing.allocator, @import("config.zig").fixture), // a real Config: Storage cannot be zeroed
            .stores = .{ makeStore(0x9000), makeStore(0xB000) },
        };
    }

    /// Called on a settled local: each sequence's slices then address this Round's own stores for good.
    fn wire(r: *Round) void {
        for (&r.seqs, &r.stores, 0..) |*s, *store, i| {
            const base: u64 = if (i == 0) 0x9000 else 0xB000;
            s.* = .{
                .a = testing.allocator,
                .c = r.cfg,
                .layers = &store.layers,
                .draft = &store.draft,
                .taps = fake(base + 0x2C00),
                .main = fake(base + 0x2D00),
                .journal = &store.journal,
                .kv8 = true,
                .staging = fake(base + 0x3000),
            };
        }
    }

    fn walkSeq(ctx: *anyopaque, out: []u64) anyerror!usize {
        const s: *st.Sequence = @ptrCast(@alignCast(ctx));
        return s.graphWords(out); // F1: the fixture constrains perf-kv's walk, never re-derives it
    }

    fn fingerprint(r: *const Round, kv8: bool) !u64 {
        var buf: [64]u64 = undefined;
        var set = r.base;
        set.identity = rg.digestWord(try folded(kv8));
        set.seqs = &.{
            .{ .ctx = @constCast(@ptrCast(&r.seqs[0])), .words = walkSeq },
            .{ .ctx = @constCast(@ptrCast(&r.seqs[1])), .words = walkSeq },
        };
        return set.fingerprint(&buf);
    }

    fn reset(r: *Round, s: usize) !void {
        if (r.fails[s]) return error.ResetFailed;
    }

    fn freeSeq(r: *Round, s: usize) void {
        const seq = &r.seqs[s];
        var n: usize = seq.partCount(); // the same production count graphWords walks
        if (seq.staging != null) n += 1;
        r.freed[s] += n;
    }
};

/// perf-kv's grown operand set, one mover each: every address the kv8 round bakes must be named by the production list.
const Mover = enum { staging, selected, q8_scratch, window_rows, compressor_rows, indexer_cache, draft_rows, taps, main_rows };

fn move(r: *Round, m: Mover) void {
    switch (m) {
        .staging => {
            if (r.seqs[0].staging) |*d| {
                d.ptr += ring_bytes; // gather8's write target, sparse's read
            }
        },
        .selected => r.base.selected += ring_bytes, // the doubly-read selected list register
        .q8_scratch => r.base.kv += ring_bytes, // the BF16 scratch q8 reads
        .window_rows => r.seqs[0].layers[0].kv.ptr += ring_bytes, // q8's destination rows, sparse's read
        .compressor_rows => {
            if (r.seqs[0].layers[0].compressor) |*cp| {
                cp.cache.ptr += ring_bytes; // the pool's rows are a view inside the layer's kv: a real move takes both
            }
            r.seqs[0].layers[0].kv.ptr += ring_bytes;
        },
        .indexer_cache => {
            if (r.seqs[0].layers[0].indexer) |*ix| {
                ix.cache.ptr += ring_bytes; // index8's reads
            }
        },
        .draft_rows => r.seqs[0].draft[0].kv.ptr += ring_bytes, // dspark writes the draft layer's rows
        .taps => r.seqs[0].taps.ptr += ring_bytes, // absorb writes the taps
        .main_rows => r.seqs[0].main.ptr += ring_bytes, // and the main rows the round reads
    }
}

fn cache(f: *rgt.Fake, e: *rg.RoundEngine, on: bool) graphs.Cache {
    e.* = .{ .gpa = testing.allocator, .pieces_side = f.pieces(), .collective = f.collective() };
    var c = graphs.Cache.init(testing.allocator, e.engine(), .{ .on = on });
    c.room = rgt.alwaysRoom;
    return c;
}

test "kv8 round: captured once, replayed N steps with zero dropAll, every launch naming the same addresses" {
    var f: rgt.Fake = .{};
    defer f.log.deinit(testing.allocator);
    var e: rg.RoundEngine = undefined;
    var c = cache(&f, &e, true);
    defer c.deinit();
    var round = try Round.blank();
    round.wire();
    var run = rg.Run{ .body = rgt.body(&f), .collective = f.collective() };
    const fp = try round.fingerprint(true);
    var i: usize = 0;
    while (i < 4) : (i += 1) {
        const want: rg.gc.Outcome = if (i == 0) .captured else .replayed;
        try testing.expectEqual(want, try c.run(.{ .rows = 2, .ctx = 0 }, rgt.no_stream, fp, .{ .ctx = &run, .run = rgt.runBody }));
    }
    try testing.expectEqual(@as(usize, 0), f.frees); // zero dropAll: part buffers, staging and taps never moved
    try testing.expectEqual(@as(usize, 1), c.count());
    try testing.expectEqual(@as(usize, 2), f.live); // the round's two graph pieces, captured once
}

test "realloc of one baked address - operand or cache row - drops every graph and recaptures" {
    const movers = [_]Mover{ .staging, .selected, .q8_scratch, .window_rows, .compressor_rows, .indexer_cache, .draft_rows, .taps, .main_rows };
    for (movers) |m| {
        var f: rgt.Fake = .{};
        defer f.log.deinit(testing.allocator);
        var e: rg.RoundEngine = undefined;
        var c = cache(&f, &e, true);
        defer c.deinit();
        var round = try Round.blank();
    round.wire();
        var run = rg.Run{ .body = rgt.body(&f), .collective = f.collective() };
        const before = try round.fingerprint(true);
        _ = try c.run(.{ .rows = 2, .ctx = 0 }, rgt.no_stream, before, .{ .ctx = &run, .run = rgt.runBody });
        move(&round, m); // the owning field moves; the production list decides whether anything notices
        const after = try round.fingerprint(true);
        try testing.expect(before != after); // named: a move of this address cannot hide from the fingerprint
        _ = try c.run(.{ .rows = 2, .ctx = 0 }, rgt.no_stream, after, .{ .ctx = &run, .run = rgt.runBody });
        try testing.expectEqual(@as(usize, 2), f.frees); // the stale graph is gone before anything replays it
        try testing.expectEqual(@as(usize, 1), c.count());
        try testing.expectEqual(@as(usize, 2), f.live); // recaptured on the new addresses
    }
}

test "kv8 on and off never share a graph at one (rows, bucket): the REAL digest words separate them" {
    var f: rgt.Fake = .{};
    defer f.log.deinit(testing.allocator);
    var e: rg.RoundEngine = undefined;
    var c = cache(&f, &e, true);
    defer c.deinit();
    var round = try Round.blank();
    round.wire();
    var run = rg.Run{ .body = rgt.body(&f), .collective = f.collective() };
    const on = try round.fingerprint(true);
    const off = try round.fingerprint(false);
    try testing.expect(on != off); // the composition fixture closes its own loop: the kv8 bit reaches the word through text_runtime
    try testing.expectEqual(rg.gc.Outcome.captured, try c.run(.{ .rows = 2, .ctx = 0 }, rgt.no_stream, on, .{ .ctx = &run, .run = rgt.runBody }));
    try testing.expectEqual(rg.gc.Outcome.captured, try c.run(.{ .rows = 2, .ctx = 0 }, rgt.no_stream, off, .{ .ctx = &run, .run = rgt.runBody }));
    try testing.expectEqual(@as(usize, 2), f.frees); // the kv8-off graph is dropped, never replayed by the on program
    try testing.expectEqual(@as(usize, 1), c.count()); // one fingerprint epoch: the changed digest dropped it and recaptured
    try testing.expectEqual(@as(usize, 4), f.begins); // two captures of two stretches each: no cross-program replay ever
}

test "a ring slot given and re-taken names the identical address through the real Ring.slot" {
    var f: rgt.Fake = .{};
    defer f.log.deinit(testing.allocator);
    var e: rg.RoundEngine = undefined;
    var c = cache(&f, &e, true);
    defer c.deinit();
    var taken: [4]bool = .{ false, false, false, false };
    var ring: st.Ring = .{ .rows = .{ .d = undefined, .ptr = 0x8000, .len = 4 * ring_bytes, .borrowed = true }, .taken = &taken, .bytes = ring_bytes };
    _ = ring.take(); // three kept states ahead of ours
    _ = ring.take();
    _ = ring.take();
    const at = ring.take() orelse return error.RingFull; // slot 3
    var round = try Round.blank();
    round.wire();
    round.seqs[0].staging = ring.slot(at); // State's rows.ptr + at * ring.bytes, from the real accessor
    var run = rg.Run{ .body = rgt.body(&f), .collective = f.collective() };
    const fp = try round.fingerprint(true);
    _ = try c.run(.{ .rows = 2, .ctx = 0 }, rgt.no_stream, fp, .{ .ctx = &run, .run = rgt.runBody });
    ring.give(at); // Snapshot.free on a ring-backed state: the slot returns, the bytes never move
    const again = ring.take() orelse return error.RingFull;
    try testing.expectEqual(ring.slot(again).ptr, (round.seqs[0].staging orelse return error.RingFull).ptr);
    _ = try c.run(.{ .rows = 2, .ctx = 0 }, rgt.no_stream, fp, .{ .ctx = &run, .run = rgt.runBody });
    try testing.expectEqual(@as(usize, 2), f.begins); // one capture of two stretches: the ring's churn moved no address
    try testing.expectEqual(@as(usize, 0), f.frees);
}

test "a failed reset retires a kv8 sequence keeping every baked address until dropAll" {
    var f: rgt.Fake = .{};
    defer f.log.deinit(testing.allocator);
    var e: rg.RoundEngine = undefined;
    var c = cache(&f, &e, true);
    defer c.deinit();
    var round = try Round.blank();
    round.wire();
    var run = rg.Run{ .body = rgt.body(&f), .collective = f.collective() };
    const fp = try round.fingerprint(true);
    _ = try c.run(.{ .rows = 2, .ctx = 0 }, rgt.no_stream, fp, .{ .ctx = &run, .run = rgt.runBody });
    round.fails[0] = true;
    try testing.expectError(error.ResetFailed, round.reset(0)); // the pooled sequence's reset failed: keep every address
    _ = try c.run(.{ .rows = 2, .ctx = 0 }, rgt.no_stream, fp, .{ .ctx = &run, .run = rgt.runBody }); // the replay names them all
    try testing.expectEqual(@as(usize, 0), f.frees); // no free on the retire path: the graph still names them
    try testing.expectEqual(@as(usize, 0), round.freed[0]);
    round.fails[0] = false;
    try round.reset(0);
    c.dropAll(); // graphs go first, buffers only after every graph naming them is gone
    try testing.expectEqual(@as(usize, 2), f.frees);
    round.freeSeq(0);
    try testing.expectEqual(@as(usize, 13), round.freed[0]); // staging + the walk's 11 parts: layer rows, compressor kv/scores, the indexer's three, draft, taps, main
}

test "two sequences name their own staging once each, and a move on either drops the round whole" {
    var f: rgt.Fake = .{};
    defer f.log.deinit(testing.allocator);
    var e: rg.RoundEngine = undefined;
    var c = cache(&f, &e, true);
    defer c.deinit();
    var round = try Round.blank();
    round.wire();
    const s0 = [_]u64{ (round.seqs[0].staging orelse return error.RingFull).ptr, round.seqs[0].layers[0].kv.ptr, round.seqs[0].draft[0].kv.ptr, round.seqs[0].taps.ptr, round.seqs[0].main.ptr };
    const s1 = [_]u64{ (round.seqs[1].staging orelse return error.RingFull).ptr, round.seqs[1].layers[0].kv.ptr, round.seqs[1].draft[0].kv.ptr, round.seqs[1].taps.ptr, round.seqs[1].main.ptr };
    try testing.expect(s0[0] != s1[0]); // seq0.staging and seq1.staging: distinct, one entry per sequence, never shared
    for (s0) |a| for (s1) |b| try testing.expect(a != b); // and the two sequences' sets never overlap
    var run = rg.Run{ .body = rgt.body(&f), .collective = f.collective() };
    _ = try c.run(.{ .rows = 2, .ctx = 0 }, rgt.no_stream, try round.fingerprint(true), .{ .ctx = &run, .run = rgt.runBody });
    move(&round, .staging); // seq0 reallocs its one staging: the round's whole graph set drops, seq1 named intact
    _ = try c.run(.{ .rows = 2, .ctx = 0 }, rgt.no_stream, try round.fingerprint(true), .{ .ctx = &run, .run = rgt.runBody });
    try testing.expectEqual(@as(usize, 2), f.frees);
    if (round.seqs[1].layers[1].compressor) |*cp| cp.scores.ptr += ring_bytes; // and the same on a compressor part of the other sequence
    _ = try c.run(.{ .rows = 2, .ctx = 0 }, rgt.no_stream, try round.fingerprint(true), .{ .ctx = &run, .run = rgt.runBody });
    try testing.expectEqual(@as(usize, 4), f.frees);
    try testing.expectEqual(@as(usize, 2), f.live); // recaptured on the new set
}

test "graphs off captures nothing: the kv8 program runs eager, exactly the T2 program, at no fingerprint cost" {
    var f: rgt.Fake = .{};
    defer f.log.deinit(testing.allocator);
    var e: rg.RoundEngine = undefined;
    var c = cache(&f, &e, false); // TF_DSV4_GRAPHS unset
    defer c.deinit();
    var round = try Round.blank();
    round.wire();
    var run = rg.Run{ .body = rgt.body(&f), .collective = f.collective() };
    try testing.expectEqual(rg.gc.Outcome.eager, try c.run(.{ .rows = 2, .ctx = 0 }, rgt.no_stream, try round.fingerprint(true), .{ .ctx = &run, .run = rgt.runBody }));
    try testing.expectEqual(@as(usize, 0), f.begins); // nothing captured: kv8 with no graphs is exactly perf-kv's T2 program
    try testing.expectEqual(@as(usize, 0), c.count());
    try testing.expectEqualStrings("abcde", f.log.items); // the round ran once, eagerly
}
