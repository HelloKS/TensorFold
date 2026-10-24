//! The batched lane round's row table: rows in command order, padded to a bucket as int32 device columns.
const std = @import("std");

pub const buckets = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24 };
pub const cap_max = buckets[buckets.len - 1];
/// dspark_block_size is fixed 5, so one stream's verify window is at most pending + 5 drafts.
pub const max_seg_rows = 6;
pub const max_streams = 4;

pub const Col = enum(u32) { ids, pos, slot, sample };
pub const ncol = 4;

comptime {
    std.debug.assert(@intFromEnum(Col.sample) + 1 == ncol);
}

/// The smallest bucket of `cap` or fewer rows holding `n` (null: more than the largest bucket the cap allows).
pub fn bucketRows(n: u32, cap: u32) ?u32 {
    for (buckets) |b| if (b >= n and b <= cap) return b;
    return null;
}

/// One stream's window inside a round: its slot, first position and rows, in command order.
pub const Seg = struct { slot: u32, start: u32, rows: u32 };

pub const Error = error{ NoRows, SlotTwice, SegmentRows, TooManyRows, PastLimit, SlotRange, RunOrder, StreamLimit };

pub fn totalRows(segs: []const Seg) u32 {
    var n: u32 = 0;
    for (segs) |s| n += s.rows;
    return n;
}

/// The last position any row of the round holds (the context bucket's input).
pub fn lastPos(segs: []const Seg) u32 {
    var last: u32 = 0;
    for (segs) |s| last = @max(last, s.start + s.rows - 1);
    return last;
}

/// Checks a round's windows: at most `max_streams` of 1..max_seg_rows rows, no slot twice, within `limit`.
pub fn check(segs: []const Seg, limit: u32) Error!void {
    if (segs.len == 0) return error.NoRows;
    if (segs.len > max_streams) return error.StreamLimit;
    var seen: u64 = 0;
    for (segs) |s| {
        if (s.slot >= 64) return error.SlotRange;
        const bit = @as(u64, 1) << @intCast(s.slot);
        if (seen & bit != 0) return error.SlotTwice;
        seen |= bit;
        if (s.rows < 1 or s.rows > max_seg_rows) return error.SegmentRows;
        if (s.start + s.rows > limit) return error.PastLimit;
    }
}

/// Byte offsets of the device table: the int32 columns [ncol][rmax], then the i32 sampled-row list [rmax].
pub fn tableBytes(rmax: u32) usize {
    return 4 * ncol * @as(usize, rmax) + 4 * @as(usize, rmax);
}

pub fn colOffset(c: Col, rmax: u32) usize {
    return 4 * @as(usize, @intFromEnum(c)) * rmax;
}

pub fn sampledOffset(rmax: u32) usize {
    return 4 * ncol * @as(usize, rmax);
}

/// Columns of `rows` padded to `R` (pad mirrors the last row, slot -1); `sampled` gets logits rows; the sampled count.
pub fn plan(rows: []const Row, R: u32, rmax: u32, out: []i32, sampled: []i32) Error!u32 {
    const n: u32 = @intCast(rows.len);
    if (n == 0) return error.NoRows;
    if (n > R or R > rmax) return error.TooManyRows;
    std.debug.assert(out.len >= ncol * rmax and sampled.len >= rmax);
    const col = struct {
        fn at(o: []i32, rm: u32, c: Col) []i32 {
            return o[@intFromEnum(c) * rm ..][0..rm];
        }
    }.at;
    const ids = col(out, rmax, .ids);
    const pos = col(out, rmax, .pos);
    const slot = col(out, rmax, .slot);
    const flag = col(out, rmax, .sample);
    var ns: u32 = 0;
    for (rows, 0..) |r, i| {
        ids[i] = @bitCast(r.token);
        pos[i] = @bitCast(r.index);
        slot[i] = @bitCast(r.slot);
        flag[i] = @intFromBool(r.sample);
        if (r.sample) {
            sampled[ns] = @intCast(i);
            ns += 1;
        }
    }
    for (n..R) |i| {
        ids[i] = ids[n - 1];
        pos[i] = pos[n - 1];
        slot[i] = -1;
        flag[i] = 0;
    }
    return ns;
}

/// Groups command-order rows into per-stream windows in `out`: contiguous runs, a slot never returns, runs stay whole.
pub fn segments(rows: []const Row, out: *[max_streams]Seg) Error![]const Seg {
    if (rows.len == 0) return error.NoRows;
    var n: usize = 0;
    var at: usize = 0;
    while (at < rows.len) {
        if (n == max_streams) return error.StreamLimit;
        const slot = rows[at].slot;
        var end = at;
        while (end < rows.len and rows[end].slot == slot) : (end += 1) {
            if (end > at and rows[end].index != rows[end - 1].index + 1) return error.RunOrder;
            if (end - at >= max_seg_rows) return error.SegmentRows;
        }
        if (end == at) return error.RunOrder;
        out[n] = .{ .slot = slot, .start = rows[at].index, .rows = @intCast(end - at) };
        n += 1;
        at = end;
    }
    for (0..n) |i| for (0..n) |j| if (i != j and out[i].slot == out[j].slot) return error.SlotTwice;
    return out[0..n];
}

pub const Row = struct {
    slot: u32,
    token: u32,
    index: u32,
    sample: bool,
};

const testing = std.testing;

test "buckets: the smallest holding n, and the cap cutting the largest" {
    try testing.expectEqual(@as(?u32, 1), bucketRows(1, 24));
    try testing.expectEqual(@as(?u32, 5), bucketRows(5, 24));
    try testing.expectEqual(@as(?u32, 7), bucketRows(7, 24));
    try testing.expectEqual(@as(?u32, 8), bucketRows(8, 24));
    try testing.expectEqual(@as(?u32, 12), bucketRows(9, 24));
    try testing.expectEqual(@as(?u32, 16), bucketRows(13, 24));
    try testing.expectEqual(@as(?u32, 24), bucketRows(17, 24));
    try testing.expectEqual(@as(?u32, null), bucketRows(25, 24));
    try testing.expectEqual(@as(?u32, 6), bucketRows(6, 6));
    try testing.expectEqual(@as(?u32, null), bucketRows(7, 6));
    try testing.expectEqual(@as(?u32, null), bucketRows(9, 8));
}

test "segments: two streams' verify windows, whole and in command order" {
    const rows = [_]Row{
        .{ .slot = 2, .token = 7, .index = 10, .sample = true },
        .{ .slot = 2, .token = 8, .index = 11, .sample = true },
        .{ .slot = 0, .token = 3, .index = 40, .sample = true },
    };
    var buf: [max_streams]Seg = undefined;
    const segs = try segments(&rows, &buf);
    try testing.expectEqual(@as(usize, 2), segs.len);
    try testing.expectEqual(Seg{ .slot = 2, .start = 10, .rows = 2 }, segs[0]);
    try testing.expectEqual(Seg{ .slot = 0, .start = 40, .rows = 1 }, segs[1]);
    try testing.expectEqual(@as(u32, 40), lastPos(segs));
    try testing.expectEqual(@as(u32, 3), totalRows(segs));
    try check(segs, 262144);
}

test "segments refuse a split run, a slot twice, a window past the block and the stream cap" {
    var b: [max_streams]Seg = undefined;
    try testing.expectError(error.RunOrder, segments(&.{
        .{ .slot = 1, .token = 1, .index = 4, .sample = true },
        .{ .slot = 1, .token = 2, .index = 6, .sample = true },
    }, &b));
    try testing.expectError(error.SlotTwice, segments(&.{
        .{ .slot = 1, .token = 1, .index = 0, .sample = true },
        .{ .slot = 2, .token = 2, .index = 0, .sample = true },
        .{ .slot = 1, .token = 3, .index = 5, .sample = true },
    }, &b));
    var over: [max_seg_rows + 1]Row = undefined;
    for (&over, 0..) |*r, i| r.* = .{ .slot = 0, .token = 1, .index = @intCast(i), .sample = true };
    try testing.expectError(error.SegmentRows, segments(&over, &b));
    var many: [max_streams + 1]Row = undefined;
    for (&many, 0..) |*r, i| r.* = .{ .slot = @intCast(i), .token = 1, .index = 0, .sample = true };
    try testing.expectError(error.StreamLimit, segments(&many, &b));
    try testing.expectError(error.NoRows, segments(&.{}, &b));
}

test "check bounds the context and refuses empty rounds" {
    try check(&.{.{ .slot = 3, .start = 5, .rows = 6 }}, 100);
    try testing.expectError(error.PastLimit, check(&.{.{ .slot = 0, .start = 99, .rows = 2 }}, 100));
    try testing.expectError(error.SlotRange, check(&.{.{ .slot = 64, .start = 0, .rows = 1 }}, 100));
    try testing.expectError(error.NoRows, check(&.{}, 100));
}

test "plan pads the bucket mirroring the last real row and lists the sampled table rows" {
    const rows = [_]Row{
        .{ .slot = 1, .token = 7, .index = 10, .sample = true },
        .{ .slot = 1, .token = 8, .index = 11, .sample = false },
        .{ .slot = 0, .token = 100, .index = 40, .sample = true },
    };
    const rmax = 8;
    var t: [ncol * rmax]i32 = undefined;
    var picks: [rmax]i32 = undefined;
    const ns = try plan(&rows, 4, rmax, &t, &picks);
    const C = struct {
        fn at(x: []const i32, rm: u32, c: Col) []const i32 {
            return x[@intFromEnum(c) * rm ..][0..4];
        }
    }.at;
    try testing.expectEqual(@as(u32, 2), ns);
    try testing.expectEqualSlices(i32, &.{ 7, 8, 100, 100 }, C(&t, rmax, .ids));
    try testing.expectEqualSlices(i32, &.{ 10, 11, 40, 40 }, C(&t, rmax, .pos));
    try testing.expectEqualSlices(i32, &.{ 1, 1, 0, -1 }, C(&t, rmax, .slot));
    try testing.expectEqualSlices(i32, &.{ 1, 0, 1, 0 }, C(&t, rmax, .sample));
    try testing.expectEqualSlices(i32, &.{ 0, 2 }, picks[0..ns]);
    try testing.expectError(error.TooManyRows, plan(&rows, 2, rmax, &t, &picks));
    try testing.expectError(error.NoRows, plan(&.{}, 4, rmax, &t, &picks));
    try testing.expectEqual(@as(usize, 4 * 4 * 8 + 4 * 8), tableBytes(8));
    try testing.expectEqual(@as(usize, 32), colOffset(.pos, 8));
    try testing.expectEqual(@as(usize, 128), sampledOffset(8));
}
