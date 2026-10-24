//! Fake-rank round shapes: plan determinism, batched==serial parity, one exchange per round, failure, fallbacks.
const std = @import("std");
const rounds = @import("rounds.zig");
const rowtab = @import("rowtab.zig");

const testing = std.testing;
const Row = rounds.Row;
const vh = 4;
const vocab = vh * 2;

const FakeSpan = struct { at: u32, start: u32, end: u32 };
const MediaList = struct { items: []const FakeSpan };
const Exec = struct { slot: u32, token: u32, at: u32, sample: bool };

const FakeSeq = struct {
    slot: u32 = 0,
    len: u32 = 0,
    befores: u32 = 0,
};

const Fake = struct {
    a: std.mem.Allocator,
    dspark: bool = false,
    media: std.AutoHashMapUnmanaged(u32, MediaList) = .empty,
    seqs: std.AutoHashMapUnmanaged(u32, FakeSeq) = .empty,
    rows_seen: std.ArrayList(Exec) = .empty,
    logits_slot: [vh]f32 = undefined,
    exch: [rowtab.cap_max * vh]f32 = undefined,
    executed: usize = 0,
    fail_at: ?usize = null,
    staged_when_idle: bool = false,
    stages: usize = 0,
    snapshots: usize = 0,
    all_gathers: usize = 0,

    pub fn sequence(f: *Fake, slot: u32) !*FakeSeq {
        const got = try f.seqs.getOrPut(f.a, slot);
        if (!got.found_existing) got.value_ptr.* = .{ .slot = slot };
        return got.value_ptr;
    }
};

fn halfOf(token: u32, j: u32) f32 {
    return @floatFromInt(token *% 7 +% j);
}

fn rowBody(m: *Fake, seq: *FakeSeq, token: u32, sample: bool) !void {
    if (m.fail_at != null and m.executed == m.fail_at.?) return error.RowFailed;
    try m.rows_seen.append(m.a, .{ .slot = seq.slot, .token = token, .at = seq.len, .sample = sample });
    if (sample) {
        for (0..vh) |j| m.logits_slot[j] = halfOf(token, @intCast(j));
    }
    m.executed += 1;
    seq.len += 1;
}

const Sink = struct {
    pub fn stage(m: *Fake, p: *const rounds.Plan) !void {
        _ = p;
        m.stages += 1;
        m.staged_when_idle = m.executed == 0;
    }
    pub fn snapshot(m: *Fake, seq: *FakeSeq) !void {
        m.snapshots += 1;
        seq.befores += 1;
    }
    pub fn headRow(m: *Fake, p: *const rounds.Plan, i: usize) !void {
        _ = p;
        @memcpy(m.exch[i * vh ..][0..vh], m.logits_slot[0..vh]);
    }
    pub fn round(m: *Fake, p: *const rounds.Plan, logits: ?[]f32) !void {
        m.all_gathers += 1;
        const out = logits orelse return;
        for (0..p.n_sampled) |k| {
            const row = p.sampled[k];
            for (0..vh) |j| out[k * vocab + j] = m.exch[row * vh + j];
            for (0..vh) |j| out[k * vocab + vh + j] = m.exch[row * vh + j];
        }
    }
};

fn verifyRows() [6]Row {
    return .{
        .{ .slot = 2, .token = 101, .index = 10, .sample = true },
        .{ .slot = 2, .token = 102, .index = 11, .sample = true },
        .{ .slot = 0, .token = 103, .index = 40, .sample = true },
        .{ .slot = 1, .token = 104, .index = 7, .sample = true },
        .{ .slot = 1, .token = 105, .index = 8, .sample = true },
        .{ .slot = 3, .token = 106, .index = 0, .sample = true },
    };
}

fn expectPlanShape(p: *const rounds.Plan, n: u32, R: u32, ns: u32, fallback: rounds.Fallback) !void {
    try testing.expectEqual(fallback, p.fallback);
    try testing.expectEqual(n, p.n);
    try testing.expectEqual(R, p.R);
    try testing.expectEqual(ns, p.n_sampled);
}

/// The sequence lengths the four verify streams must already hold for the command's rows to sit at their indices.
fn windUp(f: *Fake) !void {
    for (verifyRows()) |r| {
        if (!f.seqs.contains(r.slot)) {
            (try f.sequence(r.slot)).len = r.index;
        }
    }
}

fn expectPlansAgree(x: *const rounds.Plan, y: *const rounds.Plan) !void {
    try testing.expectEqual(x.fallback, y.fallback);
    try testing.expectEqual(x.n, y.n);
    try testing.expectEqual(x.R, y.R);
    try testing.expectEqual(x.n_sampled, y.n_sampled);
    try testing.expectEqual(x.n_segs, y.n_segs);
    try testing.expectEqual(x.last_pos, y.last_pos);
    try testing.expectEqualSlices(Row, x.rows[0..x.n], y.rows[0..y.n]);
    try testing.expectEqualSlices(u32, x.sampled[0..x.n_sampled], y.sampled[0..y.n_sampled]);
    try testing.expectEqualSlices(rowtab.Seg, x.segs[0..x.n_segs], y.segs[0..x.n_segs]);
}

fn argmax(x: []const f32, base: usize) usize {
    var b: usize = 0;
    for (1..vocab) |j| {
        if (x[base + j] > x[base + b]) b = j;
    }
    return b;
}

test "plan: one bucket for the round, windows in command order, sampled list in table order" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    var m: Fake = .{ .a = arena.allocator() };
    const rows = verifyRows();
    const p = rounds.plan(&m, &rows, rowtab.cap_max);
    try expectPlanShape(&p, 6, 6, 6, .none);
    try testing.expectEqual(@as(u32, 4), p.n_segs);
    try testing.expectEqual(rowtab.Seg{ .slot = 2, .start = 10, .rows = 2 }, p.segs[0]);
    try testing.expectEqual(rowtab.Seg{ .slot = 3, .start = 0, .rows = 1 }, p.segs[3]);
    try testing.expectEqual(@as(u32, 40), p.last_pos);
}

test "plan: ordinary decode rounds are graphable, dspark rounds and serial fallbacks are not" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    var plain: Fake = .{ .a = arena.allocator() };
    const rows = [_]Row{
        .{ .slot = 0, .token = 1, .index = 10, .sample = true },
        .{ .slot = 1, .token = 2, .index = 20, .sample = true },
    };
    try testing.expect(rounds.plan(&plain, &rows, rowtab.cap_max).graphable);
    var spec: Fake = .{ .a = arena.allocator(), .dspark = true };
    try testing.expect(!rounds.plan(&spec, &rows, rowtab.cap_max).graphable);
    try testing.expect(!rounds.plan(&plain, &.{.{ .slot = 0, .token = 1, .index = 0, .sample = true }}, 24).graphable);
}

test "multirow prefill stays eager because graph replay advances each sequence only after the round" {
    var m: Fake = .{ .a = testing.allocator };
    const rows = [_]Row{
        .{ .slot = 0, .token = 1, .index = 10, .sample = false },
        .{ .slot = 0, .token = 2, .index = 11, .sample = false },
    };
    const p = rounds.plan(&m, &rows, rowtab.cap_max);
    try testing.expectEqual(rounds.Fallback.none, p.fallback);
    try testing.expect(!p.graphable);
}

test "rowProgram: steps replay rows [from, to) through the rowBody, clamped to the plan" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    var m: Fake = .{ .a = arena.allocator() };
    try windUp(&m);
    const rows = verifyRows();
    const p = rounds.plan(&m, &rows, rowtab.cap_max);
    var box: rounds.StepBox = undefined;
    const prog = rounds.rowProgram(&box, Fake, rowBody, &m, &p);
    try testing.expectEqual(@as(u32, 6), prog.n);
    try prog.run(prog.ctx, 0, 2, undefined);
    try testing.expectEqual(@as(usize, 2), m.executed);
    try prog.run(prog.ctx, 2, 4, undefined);
    try prog.run(prog.ctx, 4, 99, undefined);
    try testing.expectEqual(@as(usize, 6), m.executed);
    try testing.expectEqual(@as(u32, 3), m.rows_seen.items[5].slot);
}

test "plan: the shapes that stay on the serial path" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var m: Fake = .{ .a = a };
    try expectPlanShape(&rounds.plan(&m, &.{.{ .slot = 0, .token = 1, .index = 0, .sample = true }}, 24), 0, 0, 0, .single);
    try expectPlanShape(&rounds.plan(&m, &.{}, 24), 0, 0, 0, .single);
    var many: [25]Row = undefined;
    for (&many, 0..) |*r, i| r.* = .{ .slot = 0, .token = 1, .index = @intCast(i), .sample = true };
    try expectPlanShape(&rounds.plan(&m, &many, 24), 0, 0, 0, .cap);
    const split = [_]Row{
        .{ .slot = 1, .token = 1, .index = 4, .sample = true },
        .{ .slot = 2, .token = 2, .index = 0, .sample = true },
        .{ .slot = 1, .token = 3, .index = 5, .sample = true },
    };
    try expectPlanShape(&rounds.plan(&m, &split, 24), 0, 0, 0, .order);
    const gap = [_]Row{
        .{ .slot = 0, .token = 1, .index = 3, .sample = true },
        .{ .slot = 0, .token = 2, .index = 5, .sample = true },
    };
    try expectPlanShape(&rounds.plan(&m, &gap, 24), 0, 0, 0, .order);
    const spans = [_]FakeSpan{.{ .at = 8, .start = 8, .end = 12 }};
    try m.media.put(a, 1, .{ .items = &spans });
    const rows = verifyRows();
    try expectPlanShape(&rounds.plan(&m, &rows, 24), 0, 0, 0, .media);
    m.media.clearRetainingCapacity();
    try expectPlanShape(&rounds.plan(&m, &rows, 24), 6, 6, 6, .none);
}

test "plan: ranks derive identical plans from identical decoded rows regardless of local sequence state" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const rows = verifyRows();
    var one: Fake = .{ .a = a };
    var two: Fake = .{ .a = a };
    (try two.sequence(2)).len = 999;
    (try two.sequence(0)).len = 3;
    (try two.sequence(9)).len = 17;
    const p1 = rounds.plan(&one, &rows, rowtab.cap_max);
    const p2 = rounds.plan(&two, &rows, rowtab.cap_max);
    try expectPlansAgree(&p1, &p2);
}

test "verify: batched rows equal the serial path's tokens, draws and one exchange" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const rows = verifyRows();
    var batched: Fake = .{ .a = a };
    try windUp(&batched);
    var p = rounds.plan(&batched, &rows, rowtab.cap_max);
    var got: [6 * vocab]f32 = undefined;
    try rounds.verify(&batched, Sink, rowBody, &p, &got);
    try testing.expectEqual(@as(usize, 1), batched.stages);
    try testing.expectEqual(true, batched.staged_when_idle);
    try testing.expectEqual(@as(usize, 6), batched.executed);
    try testing.expectEqual(@as(usize, 6), batched.snapshots);
    try testing.expectEqual(@as(usize, 1), batched.all_gathers);
    var serial: Fake = .{ .a = a };
    try windUp(&serial);
    var want: [6 * vocab]f32 = undefined;
    var drawn: usize = 0;
    for (rows) |r| {
        const seq = try serial.sequence(r.slot);
        try testing.expectEqual(r.index, seq.len);
        if (r.sample) seq.befores += 1;
        try rowBody(&serial, seq, r.token, r.sample);
        if (r.sample) {
            serial.all_gathers += 1;
            for (0..vh) |j| want[drawn * vocab + j] = serial.logits_slot[j];
            for (0..vh) |j| want[drawn * vocab + vh + j] = serial.logits_slot[j];
            drawn += 1;
        }
    }
    try testing.expectEqual(@as(usize, 6), serial.all_gathers);
    try testing.expectEqualSlices(f32, &want, &got);
    for (0..6) |k| try testing.expectEqual(argmax(&want, k * vocab), argmax(&got, k * vocab));
    for (batched.rows_seen.items, rows) |seen, r| {
        try testing.expectEqual(r.slot, seen.slot);
        try testing.expectEqual(r.token, seen.token);
        try testing.expectEqual(r.index, seen.at);
        try testing.expectEqual(r.sample, seen.sample);
    }
}

test "verify: padded bucket rows execute nothing and every real row keeps its own head" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var rows: [9]Row = undefined;
    const bases = [_]usize{ 0, 4, 6, 8 };
    for (&rows, 0..) |*r, i| {
        const seg: usize = if (i < 4) 0 else if (i < 6) 1 else if (i < 8) 2 else 3;
        r.* = .{ .slot = @intCast(seg), .token = 200 + @as(u32, @intCast(i)), .index = @intCast(i - bases[seg]), .sample = true };
    }
    var m: Fake = .{ .a = a };
    const p = rounds.plan(&m, &rows, rowtab.cap_max);
    try expectPlanShape(&p, 9, 12, 9, .none);
    var got: [9 * vocab]f32 = undefined;
    try rounds.verify(&m, Sink, rowBody, &p, &got);
    try testing.expectEqual(@as(usize, 9), m.executed);
    try testing.expectEqual(@as(usize, 1), m.all_gathers);
    for (0..9) |k| try testing.expectEqual(halfOf(200 + @as(u32, @intCast(k)), 0), got[k * vocab]);
}

test "verify: a failing row fails the whole round before any exchange" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const rows = verifyRows();
    var m: Fake = .{ .a = a, .fail_at = 3 };
    try windUp(&m);
    var p = rounds.plan(&m, &rows, rowtab.cap_max);
    var got: [6 * vocab]f32 = undefined;
    try testing.expectError(error.RowFailed, rounds.verify(&m, Sink, rowBody, &p, &got));
    try testing.expectEqual(@as(usize, 0), m.all_gathers);
    try testing.expectEqual(@as(usize, 3), m.executed);
}

test "verify: a rank whose sequence moved off the round's index fails on that row" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const rows = verifyRows();
    var m: Fake = .{ .a = a };
    try windUp(&m);
    (try m.sequence(0)).len = 33;
    var p = rounds.plan(&m, &rows, rowtab.cap_max);
    var got: [6 * vocab]f32 = undefined;
    try testing.expectError(error.PositionMismatch, rounds.verify(&m, Sink, rowBody, &p, &got));
    try testing.expectEqual(@as(usize, 0), m.all_gathers);
}

test "verify: follower rounds carry no logits out yet still run the round exchange" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const rows = verifyRows();
    var m: Fake = .{ .a = a };
    try windUp(&m);
    var p = rounds.plan(&m, &rows, rowtab.cap_max);
    try rounds.verify(&m, Sink, rowBody, &p, null);
    try testing.expectEqual(@as(usize, 1), m.all_gathers);
    try testing.expectEqual(@as(usize, 6), m.executed);
}

test "gates: TF_DSV4_SERIAL wins ties, the batched path needs its flag, caps must be buckets" {
    try testing.expectEqual(rounds.Mode.serial, rounds.modeFrom(null, null));
    try testing.expectEqual(rounds.Mode.batched, rounds.modeFrom("1", null));
    try testing.expectEqual(rounds.Mode.serial, rounds.modeFrom("1", "1"));
    try testing.expectEqual(rounds.Mode.serial, rounds.modeFrom(null, "1"));
    try testing.expectEqual(rounds.Mode.serial, rounds.modeFrom("0", null));
    try testing.expectEqual(@as(u32, 24), try rounds.capFromEnv(null));
    try testing.expectEqual(@as(u32, 16), try rounds.capFromEnv("16"));
    try testing.expectError(error.RowsCap, rounds.capFromEnv("17"));
    try testing.expectError(error.RowsCap, rounds.capFromEnv("0"));
    try testing.expectError(error.RowsCap, rounds.capFromEnv("two"));
}

test "batch image symbols all carry the tf_dsv4_batch_ prefix" {
    for (rounds.symbols) |name| try testing.expect(std.mem.startsWith(u8, name, "tf_dsv4_batch_"));
    try testing.expectEqual(@as(usize, 3), rounds.symbols.len);
}

test "Rounds.bytes budgets the table, ids, exchange, receive and output lines" {
    const cap: u32 = 8;
    const words: usize = 4 * 4 * cap + 4 * cap + 4 * cap + 4 * cap * vh + 8 * cap * vh + 4 * cap * vocab;
    try testing.expectEqual(words, rounds.Rounds.bytes(cap, vocab));
}
