//! Measured forward cost per row width at open: admission takes the timed table, static arithmetic when unmeasured.
const std = @import("std");
const cluster = @import("cluster_backend");
const lanes = @import("lanes");
pub const max_width = @import("rowtab.zig").cap_max;
pub const Error = error{ InvalidWindowMeasurement, WidthPastTable };

/// The timed widths, one Cost per row count from 1; an incomplete table never reaches admission.
pub const Table = struct {
    listed: [max_width]lanes.config.Cost = undefined,
    timed: u32 = 0,
    wanted: u32 = 0,

    /// Widths land in order; a gap or a bad number fails the whole measurement to the static path.
    pub fn set(t: *Table, width: u32, ms: f64) !void {
        if (width < 1 or width > max_width) return error.WidthPastTable;
        if (width != t.timed + 1) return error.InvalidWindowMeasurement;
        if (!(ms > 0) or !std.math.isFinite(ms)) return error.InvalidWindowMeasurement;
        t.listed[width - 1] = .{ .width = width, .ms = ms };
        t.timed = width;
    }

    /// The startup-cost contract: every width 1..wanted timed with a finite positive ms.
    pub fn complete(t: *const Table) bool {
        return t.wanted != 0 and t.timed == t.wanted;
    }

    pub fn costs(t: *const Table) []const lanes.config.Cost {
        return t.listed[0..t.timed];
    }
};

/// Two warm passes then one timed pass per width, on scratch slots the serving lanes never hold.
pub fn measure(m: anytype, t: *Table, widths: u32) !void {
    if (widths < 1 or widths > max_width) return error.WidthPastTable;
    t.wanted = widths;
    var rows: [max_width]cluster.round.Row = undefined;
    for (1..widths + 1) |w| {
        const width: u32 = @intCast(w);
        const slot: u32 = 64 + width;
        defer m.releaseRows(slot);
        for (0..w) |i| rows[i] = .{ .slot = slot, .token = 0, .index = @intCast(i), .sample = i + 1 == w };
        const span = rows[0..w];
        var pass: usize = 0;
        while (pass < 3) : (pass += 1) {
            m.releaseRows(slot);
            const started = m.now();
            try m.timeRows(span);
            const ns = m.now() -| started;
            if (pass == 2) {
                const ms = @as(f64, @floatFromInt(ns)) / 1e6;
                if (ms <= 0 or !std.math.isFinite(ms)) return error.InvalidWindowMeasurement;
                try t.set(width, ms);
            }
        }
    }
}

test "a failed timing pass returns its scratch sequence to the serving pool" {
    const Failing = struct {
        held: bool = false,
        pub fn now(_: *@This()) u64 { return 1; }
        pub fn timeRows(m: *@This(), _: []const cluster.round.Row) !void {
            m.held = true;
            return error.ForwardFailed;
        }
        pub fn releaseRows(m: *@This(), _: u32) void { m.held = false; }
    };
    var m: Failing = .{};
    var table: Table = .{};
    try std.testing.expectError(error.ForwardFailed, measure(&m, &table, 6));
    try std.testing.expect(!m.held);
}

const testing = std.testing;

const Probe = struct {
    calls: usize = 0,
    releases: usize = 0,
    rows_seen: usize = 0,
    step_ns: u64,

    fn now(p: *Probe) u64 {
        p.calls += 1;
        return p.calls * p.step_ns;
    }
    fn timeRows(p: *Probe, rows: []const cluster.round.Row) !void {
        p.rows_seen += rows.len;
    }
    fn releaseRows(p: *Probe, slot: u32) void {
        p.releases += 1;
        _ = slot;
    }
};

test "measure: three passes per width land a complete ordered table" {
    var probe: Probe = .{ .step_ns = 1_500_000 };
    var table: Table = .{};
    try measure(&probe, &table, 4);
    try testing.expect(table.complete());
    try testing.expectEqual(@as(u32, 4), table.timed);
    const got = table.costs();
    try testing.expectEqual(@as(usize, 4), got.len);
    for (got, 1..) |c, width| {
        try testing.expectEqual(@as(u32, @intCast(width)), c.width);
        try testing.expectApproxEqAbs(@as(f64, 1.5), c.ms, 0.0001);
    }
    try testing.expectEqual(@as(usize, 12), probe.calls / 2);
    try testing.expectEqual(@as(usize, 16), probe.releases);
}

test "measure: a zero-duration timed pass fails the measurement, table stays incomplete" {
    var probe: Probe = .{ .step_ns = 0 };
    var table: Table = .{};
    try testing.expectError(error.InvalidWindowMeasurement, measure(&probe, &table, 3));
    try testing.expect(!table.complete());
    try testing.expectEqual(@as(u32, 0), table.timed);
}

test "table: an empty table is the static fallback, costs are empty" {
    var table: Table = .{};
    try testing.expect(!table.complete());
    try testing.expectEqual(@as(usize, 0), table.costs().len);
}

test "table: widths must arrive in order, positive and finite, within the table" {
    var table: Table = .{};
    try testing.expectError(error.WidthPastTable, table.set(0, 1));
    try testing.expectError(error.WidthPastTable, table.set(max_width + 1, 1));
    try testing.expectError(error.InvalidWindowMeasurement, table.set(2, 1));
    try testing.expectError(error.InvalidWindowMeasurement, table.set(1, 0));
    try testing.expectError(error.InvalidWindowMeasurement, table.set(1, std.math.nan(f64)));
    try table.set(1, 2);
    try testing.expectEqual(@as(u32, 1), table.timed);
    try testing.expect(!table.complete());
}
