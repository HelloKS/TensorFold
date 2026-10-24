//! One batched lane round: several streams' rows through one verify and one rank exchange, keyed by the bucketed rows.
const std = @import("std");
const cuda = @import("cuda");
const rowtab = @import("rowtab.zig");

pub const Row = rowtab.Row;
pub const Plan = struct {
    rows: [rowtab.cap_max]Row = undefined,
    sampled: [rowtab.cap_max]u32 = undefined,
    segs: [rowtab.max_streams]rowtab.Seg = undefined,
    n: u32 = 0,
    R: u32 = 0,
    n_sampled: u32 = 0,
    n_segs: u32 = 0,
    last_pos: u32 = 0,
    fallback: Fallback = .single,
    graphable: bool = false,
};

pub const Fallback = enum { none, single, cap, media, order };

pub const Error = error{ RowsCap, BuiltWithoutBatchKernels };

pub const Mode = enum { serial, batched };

/// TF_DSV4_SERIAL=1 always wins; the batched path only answers TF_DSV4_BATCH=1; serial is the shipped default.
pub fn modeFrom(batch: ?[]const u8, serial: ?[]const u8) Mode {
    if (serial) |s| if (std.mem.eql(u8, s, "1")) return .serial;
    if (batch) |b| if (std.mem.eql(u8, b, "1")) return .batched;
    return .serial;
}

/// TF_DSV4_ROWS_CAP: the largest bucket one round takes (default 24); a non-bucket value refuses to open.
pub fn capFromEnv(value: ?[]const u8) !u32 {
    const s = value orelse return rowtab.cap_max;
    const n = std.fmt.parseInt(u32, s, 10) catch return error.RowsCap;
    if (rowtab.bucketRows(n, rowtab.cap_max) != n) return error.RowsCap;
    return n;
}

/// The round's shape from command-order rows: window grouping, the bucket and the sampled list, or why it stays serial.
pub fn plan(m: anytype, rows: anytype, cap: u32) Plan {
    const n: u32 = @intCast(rows.len);
    if (n < 2) return .{ .fallback = .single };
    if (n > rowtab.cap_max) return .{ .fallback = .cap };
    const R = rowtab.bucketRows(n, cap) orelse return .{ .fallback = .cap };
    var p: Plan = .{ .fallback = .none, .n = n, .R = R, .graphable = !m.dspark };
    for (rows, 0..) |r, i| {
        p.rows[i] = .{ .slot = r.slot, .token = r.token, .index = r.index, .sample = r.sample };
        if (mediaCovers(m, p.rows[i])) return .{ .fallback = .media };
        if (r.sample) {
            p.sampled[p.n_sampled] = @intCast(i);
            p.n_sampled += 1;
        }
    }
    var segbuf: [rowtab.max_streams]rowtab.Seg = undefined;
    const segs = rowtab.segments(p.rows[0..@min(n, rowtab.cap_max)], &segbuf) catch return .{ .fallback = .order };
    for (segs, 0..) |s, i| {
        p.segs[i] = s;
        p.n_segs += 1;
        if (s.rows != 1) p.graphable = false;
    }
    p.last_pos = rowtab.lastPos(segs);
    return p;
}

fn mediaCovers(m: anytype, row: Row) bool {
    if (m.media.get(row.slot)) |spans| for (spans.items) |span| {
        if (row.index >= span.at and row.index < span.end) return true;
    };
    return false;
}

/// The verify loop: every row through `rowBody` on its sequence, then the round's one rank exchange through `sink`.
pub fn verify(m: anytype, sink: anytype, comptime rowBody: anytype, p: *const Plan, logits: ?[]f32) !void {
    try sink.stage(m, p);
    var i: usize = 0;
    while (i < p.n) : (i += 1) {
        const row = p.rows[i];
        const seq = try m.sequence(row.slot);
        if (seq.len != row.index) return error.PositionMismatch;
        if (row.sample) try sink.snapshot(m, seq);
        try rowBody(m, seq, row.token, row.sample);
        if (row.sample) try sink.headRow(m, p, i);
    }
    try sink.round(m, p, logits);
}

/// One program step is one row's rowBody call: step i replays row i, matching the recorder's step cadence.
pub const Program = struct {
    ctx: *anyopaque,
    n: u32,
    run: *const fn (ctx: *anyopaque, from: u32, to: u32, stream: cuda.Stream) anyerror!void,
};

/// Caller-owned storage for one row program, the same discipline as rowtab.segments' out slice.
pub const StepBox = struct {
    m: *anyopaque,
    p: *const Plan,
};

/// The round's replayable row program over rows [0, p.n); the caller keeps the box alive for the run.
pub fn rowProgram(box: *StepBox, comptime M: type, comptime rowBody: anytype, m: *M, p: *const Plan) Program {
    box.* = .{ .m = m, .p = p };
    return .{ .ctx = box, .n = p.n, .run = runSteps(M, rowBody) };
}

fn runSteps(comptime M: type, comptime rowBody: anytype) *const fn (*anyopaque, u32, u32, cuda.Stream) anyerror!void {
    const Impl = struct {
        fn run(ctx: *anyopaque, from: u32, to: u32, stream: cuda.Stream) anyerror!void {
            const b: *StepBox = @ptrCast(@alignCast(ctx));
            const m: *M = @ptrCast(@alignCast(b.m));
            var i: u32 = from;
            const end = @min(to, b.p.n);
            while (i < end) : (i += 1) {
                const row = b.p.rows[i];
                const seq = try m.sequence(row.slot);
                if (seq.len != row.index) return error.PositionMismatch;
                try rowBody(m, seq, row.token, row.sample);
            }
            _ = stream;
        }
    };
    return &Impl.run;
}

/// Source-built batch glue; one fatbin image, tf_dsv4_batch_ prefixed extern "C" kernels.
pub const Ops = struct {
    mod: cuda.Module,
    pub fn load(d: *const cuda.Driver) !Ops {
        const image = if (@hasDecl(cuda.kernels, "dsv4_batch")) cuda.kernels.dsv4_batch else &.{};
        if (image.len == 0) return error.BuiltWithoutBatchKernels;
        return loadImage(d, image);
    }
    pub fn loadImage(d: *const cuda.Driver, image: []const u8) !Ops {
        return .{ .mod = try cuda.Module.load(d, image) };
    }
    pub fn unload(o: *Ops) void {
        o.mod.unload();
    }
    pub fn call(o: Ops, comptime name: []const u8, s: cuda.Stream, blocks: u32, threads: u32, values: anytype) !void {
        var args: cuda.launch.Args = .{};
        inline for (values) |v| args.add(v);
        try cuda.launch.launch(try o.mod.function("tf_dsv4_batch_" ++ name), .{ .grid = .{ .x = blocks }, .block = .{ .x = threads } }, s, &args);
    }
};

/// Every symbol the batch image must export; a wrong name cannot hide behind an unused launch path.
pub const symbols = [_][:0]const u8{ "tf_dsv4_batch_rows", "tf_dsv4_batch_ids", "tf_dsv4_batch_merge" };

/// The device state one batched round stages into: the row table, its ids column, and the round's rank-exchange line.
pub const Rounds = struct {
    a: std.mem.Allocator,
    ops: Ops,
    rows_max: u32,
    vh: u32,
    table: cuda.DeviceBuffer,
    ids: cuda.DeviceBuffer,
    exch: cuda.DeviceBuffer,
    recv: cuda.DeviceBuffer,
    out: cuda.DeviceBuffer,
    host: []i32,

    pub fn bytes(cap: u32, vocab: usize) usize {
        const vh = vocab / 2;
        return rowtab.tableBytes(cap) + 4 * cap + 4 * cap * vh + 8 * cap * vh + 4 * cap * vocab;
    }

    pub fn init(a: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: ?[]const u8, vocab: usize, cap: u32) !*Rounds {
        const r = try a.create(Rounds);
        var assembled = false;
        errdefer if (assembled) r.free() else a.destroy(r);
        const vh = vocab / 2;
        r.* = .{
            .a = a,
            .ops = try @import("gpu_modules.zig").load(Ops, a, io, d, dir, "dsv4_batch.fatbin"),
            .rows_max = cap,
            .vh = @intCast(vh),
            .table = try cuda.DeviceBuffer.alloc(d, rowtab.tableBytes(cap)),
            .ids = try cuda.DeviceBuffer.alloc(d, 4 * cap),
            .exch = try cuda.DeviceBuffer.alloc(d, 4 * cap * vh),
            .recv = try cuda.DeviceBuffer.alloc(d, 8 * cap * vh),
            .out = try cuda.DeviceBuffer.alloc(d, 4 * cap * vocab),
            .host = try a.alloc(i32, rowtab.ncol * cap + cap),
        };
        assembled = true;
        return r;
    }

    pub fn free(r: *Rounds) void {
        r.a.free(r.host);
        r.out.free();
        r.recv.free();
        r.exch.free();
        r.ids.free();
        r.table.free();
        r.ops.unload();
        r.a.destroy(r);
    }
};

/// The device sink: staging, the per-row head copy, and the round's single allGather, scatter, sync and download.
pub const Device = struct {
    pub fn stage(m: anytype, p: *const Plan) !void {
        const rb = m.rb.?;
        const cap = rb.rows_max;
        _ = try rowtab.plan(p.rows[0..p.n], p.R, cap, rb.host[0..rowtab.ncol * cap], rb.host[rowtab.ncol * cap ..]);
        try rb.table.upload(0, std.mem.sliceAsBytes(rb.host[0 .. rowtab.ncol * cap + p.n_sampled]));
        try rb.ops.call("ids", m.stream, (p.R + 127) / 128, 128, .{ rb.table.ptr, rb.ids.ptr, p.R });
    }
    pub fn snapshot(m: anytype, seq: anytype) !void {
        try seq.before(m.stream);
    }
    pub fn headRow(m: anytype, p: *const Plan, i: usize) !void {
        _ = p;
        const rb = m.rb.?;
        const vb = rb.vh * 4;
        try rb.exch.copyFrom(i * vb, m.r.p(.logits), vb, m.stream.handle);
    }
    pub fn round(m: anytype, p: *const Plan, logits: ?[]f32) !void {
        const rb = m.rb.?;
        try m.comm.allGather(rb.exch.ptr, rb.recv.ptr, @as(usize, p.R) * rb.vh, .f32, m.stream);
        if (logits) |out| {
            if (p.n_sampled > 0) {
                try rb.ops.call("rows", m.stream, p.n_sampled, 256, .{ rb.recv.ptr, rb.table.ptr + rowtab.sampledOffset(rb.rows_max), rb.out.ptr, p.R, rb.vh, rb.vh * 2 });
                try m.sync();
                try rb.out.download(0, std.mem.sliceAsBytes(out[0 .. p.n_sampled * rb.vh * 2]));
            } else try m.sync();
        } else try m.sync();
    }
};
