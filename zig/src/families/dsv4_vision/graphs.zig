//! CUDA graphs for the dsv4_vision decode round: cuda/graph_cache keyed (rows, ctx bucket); TF_DSV4_GRAPHS=1 opts in.

const std = @import("std");
const cuda = @import("cuda");
const Comm = @import("comm.zig").Comm;

pub const gc = cuda.graph_cache;

/// the graphed row counts (a window's rows are exact; this caps them and the buckets above it)
pub const row_buckets = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32, 48, 64 };

/// context buckets 0-15 are `size` tokens each; past them each ends `grow` x further
pub const linear = 16;

pub const Settings = struct {
    on: bool = false,
    max: u32 = 48,
    bucket: u32 = 2048,
    grow: f64 = 1.25,
    rows_max: u32 = 64,
    room: Room = .{},
    /// TF_DSV4_GRAPH_FLOOR=hold (default `shed`): under the floor a miss runs eagerly and held graphs stay
    floor_hold: bool = false,

    pub fn validate(s: Settings) !void {
        if (s.bucket == 0 or s.rows_max == 0 or s.rows_max > 64 or s.max > std.math.maxInt(u32) / 2) return error.BadGraphBucket;
        if (!std.math.isFinite(s.grow) or s.grow < 1 or s.grow > 16) return error.BadGraphBucket;
        if (!std.math.isFinite(s.room.cost_gib) or s.room.cost_gib < 0 or !std.math.isFinite(s.room.keep_gib) or s.room.keep_gib < 0) return error.BadGraphFloor;
        if (s.room.floor_gib) |f| if (!std.math.isFinite(f) or f < 0) return error.BadGraphFloor;
    }

    /// The knobs; graphs stay off until TF_DSV4_GRAPHS says otherwise (every rank must agree).
    pub fn fromEnv() !Settings {
        return fromEnvOr(false);
    }

    /// The knobs; graphs `on_default` only where hardware has qualified them (every rank must agree).
    pub fn fromEnvOr(on_default: bool) !Settings {
        var s: Settings = .{ .on = on_default };
        if (std.c.getenv("TF_DSV4_GRAPHS")) |v| s.on = !std.mem.eql(u8, std.mem.span(v), "0");
        if (std.c.getenv("TF_DSV4_GRAPHS_MAX")) |v| s.max = try std.fmt.parseInt(u32, std.mem.span(v), 10);
        if (std.c.getenv("TF_DSV4_GRAPH_BUCKET")) |v| s.bucket = try std.fmt.parseInt(u32, std.mem.span(v), 10);
        if (std.c.getenv("TF_DSV4_GRAPH_BUCKET_GROW")) |v| s.grow = try std.fmt.parseFloat(f64, std.mem.span(v));
        if (std.c.getenv("TF_DSV4_GRAPH_ROWS_MAX")) |v| s.rows_max = try std.fmt.parseInt(u32, std.mem.span(v), 10);
        s.room = try Room.fromEnv();
        if (std.c.getenv("TF_DSV4_GRAPH_FLOOR")) |v| {
            const m = std.mem.span(v);
            if (std.mem.eql(u8, m, "hold")) s.floor_hold = true else if (!(m.len == 0 or std.mem.eql(u8, m, "shed"))) return error.BadGraphFloor;
        }
        try s.validate();
        return s;
    }
};

/// The end (exclusive) of bucket `b`: `size` steps, then x `grow` (truncated) rounded up to `size`, at least one step.
pub fn endOf(b: u32, size: u64, grow: f64) u64 {
    var e: u64 = size * @min(b + 1, linear);
    var i: u32 = linear;
    while (i <= b) : (i += 1) {
        const nxt: u64 = if (grow > 1) blk: {
            const prod: u64 = @intFromFloat(@trunc(@as(f64, @floatFromInt(e)) * grow));
            break :blk ((prod + size - 1) / size) * size;
        } else e + size;
        e = @max(nxt, e + size);
    }
    return e;
}

/// The context bucket of a window whose last position is `last` (the first bucket ending past it).
pub fn bucketOf(last: u64, size: u64, grow: f64) u32 {
    var b: u32 = 0;
    while (endOf(b, size, grow) <= last) b += 1;
    return b;
}

/// The last position bucket `b`'s graphs score keys for.
pub fn bucketEnd(b: u32, size: u64, limit: u64, grow: f64) u64 {
    return @min(endOf(b, size, grow) - 1, limit - 1);
}

/// `part`: 0 a whole round; 1 / 2 its head / tail graph (a round split so the tail launches while the head runs)
pub const Key = struct { rows: u32, ctx: u32, part: u8 = 0 };

pub const Cache = gc.Cache(Key);

/// The key of a round of `rows` rows whose last position is `last`, at the planned rows.
pub fn keyOf(rows: u32, last: u64, s: Settings) Key {
    return .{ .rows = rows, .ctx = bucketOf(last, s.bucket, s.grow) };
}

/// The smallest bucketed row count >= `rows`, or null past the last one (the round then runs eager).
pub fn bucketRows(rows: u32) ?u32 {
    for (row_buckets) |b| if (rows <= b) return b;
    return null;
}

/// The round's start for emitting its program: its end lands at the bucket's end (the keys' shared NK).
pub fn emitStart(n: u32, b: u32, s: Settings, limit: u64) i64 {
    return @as(i64, @intCast(bucketEnd(b, s.bucket, limit, s.grow))) + 1 - n;
}

/// One more graph fits when its price leaves the hard floor (a capture's driver memory is what the room rule prices).
pub const Room = struct {
    /// TF_DSV4_GRAPH_FLOOR_GIB set: MemAvailable at least this instead
    floor_gib: ?f64 = null,
    /// TF_DSV4_GRAPH_COST_MIB: a capture's price
    cost_gib: f64 = 64.0 / 1024.0,
    /// TF_DSV4_GRAPH_KEEP_GIB: what a capture must leave free (default the 4 GiB hard floor)
    keep_gib: f64 = 4.0,

    pub fn fromEnv() !Room {
        var r: Room = .{};
        if (std.c.getenv("TF_DSV4_GRAPH_FLOOR_GIB")) |v| r.floor_gib = try std.fmt.parseFloat(f64, std.mem.span(v));
        if (std.c.getenv("TF_DSV4_GRAPH_COST_MIB")) |v| r.cost_gib = try std.fmt.parseFloat(f64, std.mem.span(v)) / 1024.0;
        r.keep_gib = gib("TF_DSV4_GRAPH_KEEP_GIB", 4.0);
        return r;
    }

    pub fn fits(r: Room, avail_gib: f64) bool {
        if (r.floor_gib) |f| return avail_gib >= f;
        return avail_gib - r.cost_gib >= r.keep_gib;
    }
};

fn gib(name: [:0]const u8, default: f64) f64 {
    const v = std.c.getenv(name) orelse return default;
    return std.fmt.parseFloat(f64, std.mem.span(v)) catch default;
}

/// The rule every graph cache asks (`room`): set once from the environment by the first cache made.
pub var rule: ?Room = null;

/// This rank's room for one more graph (true where /proc/meminfo cannot be read).
pub fn room() bool {
    const r = rule orelse blk: {
        rule = Room.fromEnv() catch Room{};
        break :blk rule.?;
    };
    return r.fits(memAvailableGiB() orelse return true);
}

/// MemAvailable now, in GiB, for a capture's measured cost (null: unknown).
pub fn meter() ?f64 {
    return memAvailableGiB();
}

/// MemAvailable in GiB (null where /proc/meminfo cannot be read, as on this Windows host).
pub fn memAvailableGiB() ?f64 {
    if (@import("builtin").os.tag != .linux) return null;
    var buf: [4096]u8 = undefined;
    const fd = std.c.open("/proc/meminfo", .{ .ACCMODE = .RDONLY });
    if (fd < 0) return null;
    defer _ = std.c.close(fd);
    const n = std.c.read(fd, &buf, buf.len);
    if (n <= 0) return null;
    return memAvailableOf(buf[0..@intCast(n)]);
}

fn memAvailableOf(text: []const u8) ?f64 {
    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |line| {
        if (!std.mem.startsWith(u8, line, "MemAvailable:")) continue;
        var f = std.mem.tokenizeScalar(u8, line["MemAvailable:".len..], ' ');
        const kb = std.fmt.parseInt(u64, f.next() orelse return null, 10) catch return null;
        return @as(f64, @floatFromInt(kb)) / (1 << 20);
    }
    return null;
}

/// Every rank's verdict over NCCL (one all-gathered i32 per rank, outside any capture): true when all agree.
pub const Agreement = struct {
    comm: *Comm,
    stream: cuda.Stream,
    dev: cuda.DeviceBuffer,
    host: cuda.HostBuffer,

    pub fn init(d: *const cuda.Driver, comm: *Comm, stream: cuda.Stream) !Agreement {
        const w: u32 = @intCast(comm.g.world);
        const dev = try cuda.DeviceBuffer.alloc(d, 4 * (w + 1));
        errdefer {
            var b = dev;
            b.free();
        }
        return .{ .comm = comm, .stream = stream, .dev = dev, .host = try cuda.HostBuffer.alloc(d, 4 * (w + 1)) };
    }

    pub fn deinit(a: *Agreement) void {
        a.dev.free();
        a.host.free();
    }

    pub fn agree(a: *Agreement) gc.Agree {
        return .{ .ctx = a, .all = all };
    }

    fn all(ctx: *anyopaque, ok: bool) anyerror!bool {
        const a: *Agreement = @ptrCast(@alignCast(ctx));
        const w: u32 = @intCast(a.comm.g.world);
        try a.stream.synchronize();
        const h = a.host.slice(i32);
        h[0] = @intFromBool(ok);
        try a.dev.uploadAsync(0, std.mem.sliceAsBytes(h[0..1]), a.stream.handle);
        try a.comm.allGather(a.dev.ptr, a.dev.ptr + 4, 1, .i32, a.stream);
        try a.dev.downloadAsync(4, std.mem.sliceAsBytes(h[1 .. 1 + w]), a.stream.handle);
        try a.stream.synchronize();
        for (h[1 .. 1 + w]) |v| if (v != 1) return false;
        return true;
    }
};

/// A graphed round's inputs (ids int64 [rows_max], start int32), copied from two alternating pinned staging halves.
pub const Statics = struct {
    pub const ids_off = 0;
    rows_max: u32,
    dev: cuda.DeviceBuffer,
    stages: [2]cuda.copy_source.Stage = @splat(.{}),
    half: u1 = 0,

    pub fn bytes(rows_max: u32) usize {
        return 8 * @as(usize, rows_max) + 256;
    }

    pub fn startOff(rows_max: u32) usize {
        return 8 * @as(usize, rows_max);
    }

    pub fn init(d: *const cuda.Driver, rows_max: u32) !Statics {
        return .{ .rows_max = rows_max, .dev = try cuda.DeviceBuffer.alloc(d, bytes(rows_max)) };
    }

    pub fn drain(s: *Statics) !void {
        for (&s.stages) |*st| try st.wait();
    }

    // Called only before first use or after drain succeeds: a pinned half a copy may still read is never freed.
    pub fn deinit(s: *Statics) void {
        for (&s.stages) |*st| st.close() catch unreachable;
        s.dev.free();
    }

    pub fn idsDev(s: *const Statics) u64 {
        return s.dev.ptr;
    }

    pub fn startDev(s: *const Statics) u64 {
        return s.dev.ptr + startOff(s.rows_max);
    }

    pub fn stage(s: *Statics, stream: cuda.Stream, ids: []const u32, start: u64) !void {
        if (ids.len > s.rows_max or start > std.math.maxInt(i32)) return error.TooManyRows;
        const st = &s.stages[s.half];
        const b = try st.take(stream.d, bytes(s.rows_max));
        @memset(b, 0);
        for (ids, 0..) |id, i| std.mem.writeInt(i64, b[8 * i ..][0..8], id, .little);
        std.mem.writeInt(i32, b[startOff(s.rows_max)..][0..4], @intCast(start), .little);
        try st.copyTo(s.dev, 0, stream);
        s.half +%= 1;
    }
};

// -- host tests ---------------------------------------------------------------------------------------------------

test "graphs: context buckets as v41's (2,048 steps to 32K, then 1.25x rounded up to 2,048)" {
    const want = [_]u64{ 2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384, 18432, 20480, 22528, 24576, 26624, 28672, 30720, 32768, 40960, 51200, 65536, 81920, 102400, 129024 };
    for (want, 0..) |e, b| try std.testing.expectEqual(e, endOf(@intCast(b), 2048, 1.25));
    try std.testing.expectEqual(@as(u32, 0), bucketOf(0, 2048, 1.25));
    try std.testing.expectEqual(@as(u32, 0), bucketOf(2047, 2048, 1.25));
    try std.testing.expectEqual(@as(u32, 1), bucketOf(2048, 2048, 1.25));
    try std.testing.expectEqual(@as(u32, 16), bucketOf(32768, 2048, 1.25));
    try std.testing.expectEqual(@as(u32, 17), bucketOf(40960, 2048, 1.25));
    try std.testing.expectEqual(@as(u64, 2047), bucketEnd(0, 2048, 1 << 20, 1.25));
    try std.testing.expectEqual(@as(u64, 3999), bucketEnd(1, 2048, 4000, 1.25));
    try std.testing.expectEqual(@as(u64, 2048 * 20), endOf(19, 2048, 1.0));
}

test "graphs: emitted start puts the round's end at the bucket's end; keys and row buckets agree" {
    const s: Settings = .{};
    try std.testing.expectEqual(@as(i64, 2043), emitStart(5, 0, s, 1 << 20));
    try std.testing.expectEqual(Key{ .rows = 5, .ctx = 0 }, keyOf(5, 2047, s));
    try std.testing.expectEqual(Key{ .rows = 5, .ctx = 1 }, keyOf(5, 2048, s));
    try std.testing.expectEqual(@as(?u32, 1), bucketRows(1));
    try std.testing.expectEqual(@as(?u32, 24), bucketRows(17));
    try std.testing.expectEqual(@as(?u32, 64), bucketRows(64));
    try std.testing.expectEqual(@as(?u32, null), bucketRows(65));
}

test "graphs: MemAvailable in GiB" {
    try std.testing.expectEqual(@as(?f64, 6.0), memAvailableOf("MemTotal: 1 kB\nMemAvailable:   6291456 kB\n"));
    try std.testing.expectEqual(@as(?f64, null), memAvailableOf("MemTotal: 1 kB\n"));
}

test "graphs: one more graph fits when its price leaves the kept floor, or above the fixed one" {
    const served: Room = .{ .keep_gib = 4.0 };
    try std.testing.expect(served.fits(4.66));
    try std.testing.expect(!served.fits(4.03));
    const high: Room = .{ .keep_gib = 4.5 };
    try std.testing.expect(high.fits(4.66) and !high.fits(4.55));
    const fixed: Room = .{ .floor_gib = 5.0 };
    try std.testing.expect(!fixed.fits(4.66) and fixed.fits(5.1));
}

test "graphs: settings default OFF and validate zero buckets, nonfinite growth and oversized row windows" {
    try std.testing.expect(!(Settings{}).on); // opt-in like v41's graphs_default=false; the env flips it on every rank
    try std.testing.expectError(error.BadGraphBucket, (Settings{ .bucket = 0 }).validate());
    try std.testing.expectError(error.BadGraphBucket, (Settings{ .grow = std.math.nan(f64) }).validate());
    try std.testing.expectError(error.BadGraphBucket, (Settings{ .grow = std.math.inf(f64) }).validate());
    try std.testing.expectError(error.BadGraphBucket, (Settings{ .rows_max = 65 }).validate());
    try std.testing.expectError(error.BadGraphFloor, (Settings{ .room = .{ .keep_gib = -1 } }).validate());
    try std.testing.expectError(error.BadGraphFloor, (Settings{ .room = .{ .floor_gib = std.math.nan(f64) } }).validate());
    try (Settings{}).validate();
}
