//! A decode round recorded as graphs cut at its NCCL collectives, which run between the replays; capture runs nothing.

const std = @import("std");
const cuda = @import("cuda");
const graphs = @import("graphs.zig");
const Comm = @import("comm.zig").Comm;

pub const gc = graphs.gc;

/// One NCCL call of a round, kept with the buffers it recorded with (a graphed round's addresses stay fixed).
pub const Call = union(enum) {
    all_gather: struct { send: u64, recv: u64, count: usize, dtype: cuda.nccl.DataType },
    all_reduce: struct { send: u64, recv: u64, count: usize, dtype: cuda.nccl.DataType, op: cuda.nccl.RedOp },

    fn issue(c: Call, col: Collective, stream: cuda.Stream) !void {
        switch (c) {
            .all_gather => |g| try col.allGather(col.ctx, g.send, g.recv, g.count, g.dtype, stream),
            .all_reduce => |r| try col.allReduce(col.ctx, r.send, r.recv, r.count, r.dtype, r.op, stream),
        }
    }
};

/// The round's rank collective: one thin vtable, so a transport swap (tp_group, roce, P2P) cannot fork the cut.
pub const Collective = struct {
    ctx: *anyopaque,
    allGather: *const fn (ctx: *anyopaque, send: u64, recv: u64, count: usize, dtype: cuda.nccl.DataType, stream: cuda.Stream) anyerror!void,
    allReduce: *const fn (ctx: *anyopaque, send: u64, recv: u64, count: usize, dtype: cuda.nccl.DataType, op: cuda.nccl.RedOp, stream: cuda.Stream) anyerror!void,
};

/// The NCCL collective over the family's comm (comm.zig read-only); the caller keeps the comm alive.
pub fn nccl(comm: *Comm) Collective {
    const C = struct {
        fn allGather(ctx: *anyopaque, send: u64, recv: u64, count: usize, dtype: cuda.nccl.DataType, stream: cuda.Stream) anyerror!void {
            const c: *Comm = @ptrCast(@alignCast(ctx));
            return c.allGather(send, recv, count, dtype, stream);
        }

        fn allReduce(ctx: *anyopaque, send: u64, recv: u64, count: usize, dtype: cuda.nccl.DataType, op: cuda.nccl.RedOp, stream: cuda.Stream) anyerror!void {
            const c: *Comm = @ptrCast(@alignCast(ctx));
            return c.g.lib.check(c.g.lib.api.ncclAllReduce(send, recv, count, dtype, op, c.g.comm, stream.handle), "ncclAllReduce");
        }
    };
    return .{ .ctx = comm, .allGather = C.allGather, .allReduce = C.allReduce };
}

pub const Range = struct { from: usize, to: usize };

/// A deferred stretch's live body: runs uncaptured each replay; the Split owning the piece destroys its ctx.
pub const Thunk = struct {
    ctx: *anyopaque,
    run: *const fn (ctx: *anyopaque, stream: cuda.Stream) anyerror!void,
    dtor: ?*const fn (ctx: *anyopaque) void = null,
};

/// One sequence's address walk, supplied by the caller: production runs Sequence.graphWords, tests a fake part walk.
pub const SeqWalk = struct {
    ctx: *anyopaque,
    words: *const fn (ctx: *anyopaque, out: []u64) anyerror!usize,
};

/// One round's baked addresses: registers once, then a walked set per row; production and tests list through this.
pub const RoundAddrs = struct {
    table: u64, // rowtab, staged before the round
    ids: u64, // the ids column the batch ids kernel fills
    exch: u64, // the rank exchange line
    recv: u64, // the peer halves' landing
    out: u64, // sampled logits staging (graphable rounds sample none, but its launches name it)
    kv: u64, // q8's BF16 scratch register
    gate: u64, // the compressor gate register q8 reads
    up: u64, // the compressor up register q8 reads
    selected: u64, // the selected list gather8 and sparse both read
    identity: u64, // text_runtime.fingerprint's digest folded by digestWord: the kv8 bit rides its gate block
    seqs: []const SeqWalk, // one walk per row's sequence, in row order

    pub fn fingerprint(r: RoundAddrs, buf: []u64) !u64 {
        const fixed = [_]u64{ r.table, r.ids, r.exch, r.recv, r.out, r.kv, r.gate, r.up, r.selected, r.identity };
        if (buf.len < fixed.len) return error.AddressSetTooSmall;
        var n: usize = fixed.len;
        @memcpy(buf[0..n], &fixed);
        for (r.seqs) |s| n += try s.words(s.ctx, buf[n..]); // each walk bounds-checks its own slice
        return gc.fingerprintOf(buf[0..n]);
    }
};

/// The shared fold from the execution fingerprint to one word: production and tests fold identically through this.
pub fn digestWord(digest: [32]u8) u64 {
    return std.mem.readInt(u64, digest[0..8], .little);
}

/// The round's step-addressable program: a deferred range re-runs launches and host reads live, not collectives.
pub const Program = struct {
    ctx: *anyopaque,
    n: u32, // the shape rounds.rowProgram exports; step counts meet it by value
    run: *const fn (ctx: *anyopaque, from: usize, to: usize, stream: cuda.Stream) anyerror!void,
};

/// One recorded piece of a round: a graph, a collective between the graphs, or a stretch re-run live at replay.
pub const Piece = union(enum) {
    graph: *anyopaque,
    call: Call,
    /// A deferred stretch: its thunk when the body supplied one, else the program answers for [from,to).
    eager: struct { range: Range, thunk: ?Thunk },
};

/// The capture side the recorder cuts with; `step` is a test seam, the driver records the stream's launches itself.
pub const Pieces = struct {
    ctx: *anyopaque,
    begin: *const fn (ctx: *anyopaque, stream: cuda.Stream) anyerror!void,
    /// end reports the captured node count (cuGraphGetNodes on the driver, ops length in tests): zero keeps no piece.
    end: *const fn (ctx: *anyopaque, stream: cuda.Stream, nodes: *usize) anyerror!*anyopaque,
    launch: *const fn (ctx: *anyopaque, exec: *anyopaque, stream: cuda.Stream) anyerror!void,
    free: *const fn (ctx: *anyopaque, exec: *anyopaque) void,
    step: ?*const fn (ctx: *anyopaque, op: u8) void = null,
};

/// The driver's pieces: thread-local stream capture, instantiate and upload once; launchOn replays in order.
pub const CudaPieces = struct {
    const Held = struct { exec: cuda.graph.Exec };

    pub fn pieces() Pieces {
        return .{ .ctx = undefined, .begin = begin, .end = end, .launch = launch, .free = free };
    }

    fn begin(_: *anyopaque, stream: cuda.Stream) anyerror!void {
        try cuda.graph.beginCapture(stream, .thread_local);
    }

    fn end(_: *anyopaque, stream: cuda.Stream, nodes: *usize) anyerror!*anyopaque {
        var g = try cuda.graph.endCapture(stream);
        defer g.deinit();
        nodes.* = try g.nodeCount();
        const h = try std.heap.c_allocator.create(Held);
        errdefer std.heap.c_allocator.destroy(h);
        h.exec = try g.instantiate();
        errdefer h.exec.deinit();
        try h.exec.upload(stream);
        return h;
    }

    fn launch(_: *anyopaque, exec: *anyopaque, stream: cuda.Stream) anyerror!void {
        const h: *Held = @ptrCast(@alignCast(exec));
        try h.exec.launchOn(stream);
    }

    fn free(_: *anyopaque, exec: *anyopaque) void {
        const h: *Held = @ptrCast(@alignCast(exec));
        h.exec.deinit();
        std.heap.c_allocator.destroy(h);
    }
};

/// A recorded round: its graphs, collectives and deferred live stretches in order, on the stream it recorded on.
pub const Split = struct {
    gpa: std.mem.Allocator,
    pieces: []Piece,
    collective: ?Collective,
    capture: Pieces,
    program: ?Program,

    pub fn deinit(s: *Split) void {
        for (s.pieces) |*p| switch (p.*) {
            .graph => |x| s.capture.free(s.capture.ctx, x),
            .eager => |e| if (e.thunk) |t| if (t.dtor) |d| d(t.ctx),
            .call => {},
        };
        s.gpa.free(s.pieces);
        s.* = undefined;
    }

    /// Graphs launch, collectives run between them, deferred stretches re-run live: once per replay, in order.
    pub fn launchOn(s: Split, stream: cuda.Stream) !void {
        for (s.pieces) |p| switch (p) {
            .graph => |x| try s.capture.launch(s.capture.ctx, x, stream),
            .call => |c| try c.issue(s.collective orelse return error.CollectiveFailed, stream),
            .eager => |e| {
                if (e.thunk) |t| try t.run(t.ctx, stream) else {
                    const p2 = s.program orelse return error.ProgramRequired;
                    try p2.run(p2.ctx, e.range.from, e.range.to, stream);
                }
            },
        };
    }
};

/// Records a round from `begin` to `finish`: launches go into graphs, cut wherever a collective goes through `cut`.
pub const Recorder = struct {
    gpa: std.mem.Allocator,
    stream: cuda.Stream,
    capture: Pieces,
    /// The round's communicator (null: one rank, no collectives).
    inner: ?Collective,
    pieces: std.ArrayList(Piece) = .empty,
    /// A cut failed or the body read the device on the host: nothing is kept and the round runs eagerly.
    failed: bool = false,
    capturing: bool = false,
    /// The program when the round defers host-dependent stretches: ranges replay live, so the recorder counts steps.
    program: ?Program = null,
    to: usize = 0,

    pub fn begin(r: *Recorder) !void {
        if (r.capturing or r.failed) return;
        r.capture.begin(r.capture.ctx, r.stream) catch {
            r.failed = true;
            return error.CaptureFailed;
        };
        r.capturing = true;
    }

    /// One launch step of the round's program: the driver records it into the open capture; `op` labels it for tests.
    pub fn step(r: *Recorder, op: u8) !void {
        if (r.failed) return error.CaptureFailed;
        if (!r.capturing) try r.begin();
        r.to += 1;
        if (r.capture.step) |s| s(r.capture.ctx, op);
    }

    /// A host-dependent stretch: recording executes nothing; the thunk (or program range) re-runs it on every replay.
    pub fn deferSteps(r: *Recorder, k: usize, thunk: ?Thunk) !void {
        if (r.failed) return error.CaptureFailed;
        if (k == 0) return;
        if (thunk == null and r.program == null) return error.ProgramRequired;
        r.closePiece() catch |e| {
            r.failed = true;
            return e;
        };
        r.pieces.append(r.gpa, .{ .eager = .{ .range = .{ .from = r.to, .to = r.to + k }, .thunk = thunk } }) catch |e| {
            r.failed = true;
            return e;
        };
        r.to += k;
    }

    /// One collective of the round: the open graph closes, the call is kept, and the next capture starts.
    pub fn cut(r: *Recorder, c: Call) !void {
        if (r.failed) return error.CollectiveFailed;
        r.closePiece() catch |e| {
            r.failed = true;
            return e;
        };
        r.pieces.append(r.gpa, .{ .call = c }) catch |e| {
            r.failed = true;
            return e;
        };
        r.to += 1;
        try r.begin();
    }

    /// A sync + download the body branches on: no graph may answer the host, so the whole recording is abandoned.
    pub fn hostRead(r: *Recorder) !void {
        return r.abandon();
    }

    /// Drops everything recorded: the captured launches never ran, so the round keeps its buffers and runs eager.
    pub fn abandon(r: *Recorder) !void {
        if (r.capturing) {
            var nodes: usize = 0;
            if (r.endCapture(&nodes)) |x| r.capture.free(r.capture.ctx, x) else |_| {}
        }
        r.dropPieces();
        r.failed = true;
    }

    fn dropPieces(r: *Recorder) void {
        for (r.pieces.items) |*p| switch (p.*) {
            .graph => |x| r.capture.free(r.capture.ctx, x),
            .eager => |e| if (e.thunk) |t| if (t.dtor) |d| d(t.ctx),
            .call => {},
        };
        r.pieces.clearRetainingCapacity();
    }

    fn closePiece(r: *Recorder) !void {
        if (!r.capturing) return;
        var nodes: usize = 0;
        const x = r.endCapture(&nodes) catch {
            r.failed = true;
            return error.CaptureFailed;
        };
        if (nodes == 0) {
            r.capture.free(r.capture.ctx, x); // the driver recorded nothing into it: no piece, no failure
            return;
        }
        r.pieces.append(r.gpa, .{ .graph = x }) catch |e| {
            r.capture.free(r.capture.ctx, x);
            r.failed = true;
            return e;
        };
    }

    fn endCapture(r: *Recorder, nodes: *usize) !*anyopaque {
        const x = try r.capture.end(r.capture.ctx, r.stream, nodes);
        r.capturing = false;
        return x;
    }

    /// The collective the body gets while capturing: a raw call is a graph-baked NCCL, so it is refused, not baked.
    pub fn guarded(r: *Recorder) Collective {
        const G = struct {
            fn gather(ctx: *anyopaque, send: u64, recv: u64, count: usize, dtype: cuda.nccl.DataType, stream: cuda.Stream) anyerror!void {
                const rec: *Recorder = @ptrCast(@alignCast(ctx));
                if (rec.capturing) return error.CollectiveBaked;
                const c = rec.inner.?;
                try c.allGather(c.ctx, send, recv, count, dtype, stream);
            }

            fn reduce(ctx: *anyopaque, send: u64, recv: u64, count: usize, dtype: cuda.nccl.DataType, op: cuda.nccl.RedOp, stream: cuda.Stream) anyerror!void {
                const rec: *Recorder = @ptrCast(@alignCast(ctx));
                if (rec.capturing) return error.CollectiveBaked;
                const c = rec.inner.?;
                try c.allReduce(c.ctx, send, recv, count, dtype, op, stream);
            }
        };
        return .{ .ctx = r, .allGather = G.gather, .allReduce = G.reduce };
    }

    /// The recorded round, or null when it was abandoned or a piece could not be kept; the recorder is empty after.
    pub fn finish(r: *Recorder) !?Split {
        if (!r.failed) r.closePiece() catch {
            r.failed = true;
        };
        if (r.failed) return r.discard();
        if (r.program) |p| if (r.to != @as(usize, p.n)) { // the body must consume exactly its program: a missing step cannot replay
            r.failed = true;
            return r.discard();
        };
        const pieces = r.pieces.toOwnedSlice(r.gpa) catch return r.discard();
        return .{ .gpa = r.gpa, .pieces = pieces, .collective = r.inner, .capture = r.capture, .program = r.program };
    }

    /// Frees what was recorded: the round runs eagerly instead.
    fn discard(r: *Recorder) ?Split {
        r.dropPieces();
        r.pieces.deinit(r.gpa);
        r.pieces = .empty;
        return null;
    }
};

/// One round through the cache: the same body runs captured (with a Recorder) or eager (without one), in one order.
pub const Body = struct {
    ctx: *anyopaque,
    run: *const fn (ctx: *anyopaque, rec: ?*Recorder, collective: ?Collective, stream: cuda.Stream) anyerror!void,
    /// The step-addressable program when the round defers host-dependent stretches; null for host-free rounds.
    program: ?Program = null,
};

/// The cache's body context: the round's program plus the collective it issues itself when not being captured.
pub const Run = struct {
    body: Body,
    collective: ?Collective,
};

/// The shared graph cache's engine over recorded rounds: capture records the Split, launch replays its pieces.
pub const RoundEngine = struct {
    gpa: std.mem.Allocator,
    pieces_side: Pieces,
    collective: ?Collective,

    pub fn engine(e: *RoundEngine) gc.Engine {
        return .{ .ctx = e, .capture = capture, .launch = launch, .free = free };
    }

    const Held = struct { split: Split };

    /// The stream is capturing while the body runs: its launches record, its collectives cut, nothing executes.
    fn capture(ctx: *anyopaque, stream: cuda.Stream, body: gc.Body) anyerror!*anyopaque {
        const e: *RoundEngine = @ptrCast(@alignCast(ctx));
        const r: *const Run = @ptrCast(@alignCast(body.ctx));
        var rec: Recorder = .{ .gpa = e.gpa, .stream = stream, .capture = e.pieces_side, .inner = e.collective, .program = r.body.program };
        try rec.begin();
        const col: ?Collective = if (e.collective != null) rec.guarded() else null;
        r.body.run(r.body.ctx, &rec, col, stream) catch {
            // nothing ran (captures execute nothing): drop what was recorded and let the cache's verdict run it eager
            rec.abandon() catch {};
            return error.CaptureFailed;
        };
        const split = (try rec.finish()) orelse return error.CaptureFailed;
        const h = try std.heap.c_allocator.create(Held);
        errdefer std.heap.c_allocator.destroy(h);
        h.split = split;
        return h;
    }

    fn launch(_: *anyopaque, exec: *anyopaque, stream: cuda.Stream) anyerror!void {
        const h: *Held = @ptrCast(@alignCast(exec));
        try h.split.launchOn(stream);
    }

    fn free(_: *anyopaque, exec: *anyopaque) void {
        const h: *Held = @ptrCast(@alignCast(exec));
        var s = h.split;
        s.deinit();
        std.heap.c_allocator.destroy(h);
    }
};

/// The graphed rounds' state: the shared cache keyed (rows, context bucket), one engine over the ranks' stream.
pub const Graphs = struct {
    gpa: std.mem.Allocator,
    settings: graphs.Settings,
    rows_max: u32,
    engine: RoundEngine,
    agreement: ?graphs.Agreement = null,
    cache: graphs.Cache,

    pub fn init(g: *Graphs, gpa: std.mem.Allocator, s: graphs.Settings, capture: Pieces, collective: ?Collective, agree: ?gc.Agree) void {
        g.* = .{
            .gpa = gpa,
            .settings = s,
            .rows_max = s.rows_max,
            .engine = .{ .gpa = gpa, .pieces_side = capture, .collective = collective },
            .cache = undefined,
        };
        g.cache = graphs.Cache.init(gpa, g.engine.engine(), .{ .on = true, .max = s.max, .hold = s.floor_hold });
        g.cache.agree = agree;
        g.cache.room = graphs.room;
        g.cache.meter = graphs.meter;
        g.cache.name = "decode rounds";
        graphs.rule = s.room;
    }

    /// null: the round stays the caller's to run; else the cache ran it, and the outcome says who advances.
    pub fn runRound(g: *Graphs, rows: u32, last: u64, stream: cuda.Stream, fp: u64, graphable: bool, body: Body) !?gc.Outcome {
        if (!graphable or !g.settings.on or rows == 0 or rows > g.rows_max or graphs.bucketRows(rows) == null) return null;
        const key = graphs.keyOf(rows, last, g.settings);
        var run = Run{ .body = body, .collective = g.engine.collective };
        return try g.cache.run(key, stream, fp, .{ .ctx = &run, .run = runBody });
    }

    /// The eager path of the shared cache: the same body, no Recorder, issuing its own collectives in the same order.
    fn runBody(ctx: *anyopaque, stream: cuda.Stream) anyerror!void {
        const r: *const Run = @ptrCast(@alignCast(ctx));
        try r.body.run(r.body.ctx, null, r.collective, stream);
    }

    /// Every graph freed (a fingerprint change, a poison): a slot's buffers outlive every graph that names them.
    pub fn dropAll(g: *Graphs) void {
        g.cache.dropAll();
    }

    pub fn deinit(g: *Graphs) void {
        const cs = g.cache.stats;
        std.log.scoped(.dsv4v).info("graphs: {d} captured, {d} replayed, {d} eager, {d} keys held, {d} evicted, {d} dropped{s}", .{ cs.captured, cs.replayed, cs.eager, g.cache.count(), cs.evicted, cs.dropped, if (g.cache.failed) " (off after a failed capture)" else "" });
        g.cache.deinit();
        if (g.agreement) |*a| a.deinit();
    }
};

/// The addresses a graphed round bakes in (staged statics, the row table, the gathered exchange): any change drops all.
pub fn fingerprint(words: []const u64) u64 {
    return gc.fingerprintOf(words);
}

test "the production graph cache retains its owning engine address" {
    var g: Graphs = undefined;
    g.init(std.testing.allocator, .{ .on = true }, undefined, null, null);
    defer g.deinit();
    try std.testing.expectEqual(@intFromPtr(&g.engine), @intFromPtr(g.cache.engine.ctx));
}
