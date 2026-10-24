//! Two-rank text model: source kernels, exact cache identity and bounded per-request state.
const std = @import("std");
const cuda = @import("cuda");
const cluster = @import("cluster_backend");
const Config = @import("config.zig").Config;
const state = @import("text_state.zig");
const runtime = @import("text_runtime.zig");
const Link = @import("link.zig").Link;
const exchange = cluster.exchange;
const rg = @import("round_graph.zig");
const graphs = @import("graphs.zig");
pub const Model = struct {
    a: std.mem.Allocator,
    io: std.Io,
    d: *const cuda.Driver,
    ctx: *const cuda.Context,
    room_result: u64 = 0,
    c: Config,
    rank: u32,
    context: usize,
    stream: cuda.Stream,
    r: runtime.Runtime,
    w: @import("text_weights.zig").Weights,
    link: Link,
    comm: @import("comm.zig").Comm,
    freq: [2]cuda.DeviceBuffer,
    sequences: std.AutoHashMapUnmanaged(u32, state.Sequence) = .empty,
    idle: [4]state.Sequence,
    idle_n: usize = 4,
    failed: bool = false,
    identity: [32]u8,
    snapshots: std.AutoHashMapUnmanaged(u64, state.Snapshot) = .empty,
    media: std.AutoHashMapUnmanaged(u32, std.ArrayList(@import("vision_serving.zig").Span)) = .empty,
    proposed: [5]u32 = @splat(0),
    confidence: [5]f32 = @splat(0),
    rb: ?*@import("rounds.zig").Rounds = null,
    kv8: bool = false, // --kv-cache fp8: e4m3 rows across this model's caches, and part of the identity its rank shares
    dspark: bool = false, // the dspark window at open: those rounds start off the graphed path
    wc: @import("window_cost.zig").Table = .{}, // the widths timed at open; an incomplete table keeps lane admission static
    gr: ?*rg.Graphs = null, // TF_DSV4_GRAPHS decode rounds: rank-local execution, never a fingerprint byte
    glive: [@import("rowtab.zig").cap_max]GLive = @splat(.{}),
    const GLive = struct { slot: u32 = 0, token: u32 = 0 }; // the per-round refresh deferred stretches read live
    pub fn open(a: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, cache: []const u8, kernels: ?[]const u8, rank: u32, ip: [4]u8, port: u16, context: usize, kv8: bool) !*Model {
        if (rank > 1 or context == 0 or context > 262144) return error.UnsupportedProfile;
        var arena: std.heap.ArenaAllocator = .init(a);
        defer arena.deinit();
        const temp = arena.allocator();
        const config_path = try std.fs.path.join(temp, &.{ dir, "config.json" });
        const config_text = try std.Io.Dir.cwd().readFileAlloc(io, config_path, temp, .limited(1 << 20));
        const c = try Config.parse(temp, config_text);
        var problem: []const u8 = "";
        const local = @import("rank_cache.zig").Cache.open(temp, io, cache, rank, config_text, &problem) catch |err| {
            std.log.err("rank {d}: cache {t}: {s}", .{ rank, err, problem });
            return err;
        };
        try @import("text_admission.zig").check(a, io, ctx, c, &local.tensors, context, rank, kv8);
        var link = if (rank == 0) try Link.listen(ip, port) else try Link.connect(io, ip, port, 120);
        errdefer link.close();
        const identity = @import("text_identity.zig");
        const native = @import("native.zig");
        const rd = @import("rounds.zig");
        const batched = rd.modeFrom(native.env("TF_DSV4_BATCH"), native.env("TF_DSV4_SERIAL")) == .batched;
        const rows_cap = try rd.capFromEnv(native.env("TF_DSV4_ROWS_CAP"));
        const dspark = std.mem.eql(u8, native.env("TF_DSV4_DSPARK") orelse "0", "1");
        const wc_on = std.mem.eql(u8, native.env("TF_DSV4_WC") orelse "1", "1");
        const gs: graphs.Settings = if (batched) try graphs.Settings.fromEnv() else .{};
        const execution = try runtime.fingerprint(a, io, kernels, c.storage, batched, rows_cap, kv8, wc_on, .{
            .dspark = dspark, .graph = gs,
            .p2p = std.mem.eql(u8, native.env("TF_DSV4_P2P") orelse "0", "1"),
            .roce = std.mem.eql(u8, native.env("TF_DSV4_ROCE") orelse "0", "1"),
        });
        const hello = try identity.encode(local.identity, execution, @intCast(context), rank);
        var peer: [72]u8 = undefined;
        if (rank == 0) {
            try link.send(&hello);
            try identity.validate(hello, try link.recv(&peer));
        } else {
            try identity.validate(hello, try link.recv(&peer));
            try link.send(&hello);
        }
        var comm = try @import("comm.zig").Comm.init(link, rank, 2);
        errdefer comm.deinit();
        var s = try cuda.Stream.init(ctx.d, true);
        errdefer s.deinit();
        var r = try runtime.Runtime.init(a, io, ctx.d, c, kernels);
        errdefer r.free();
        var w = try @import("text_weights.zig").Weights.load(a, io, ctx.d, s, r.text, &local.tensors, c, rank);
        errdefer w.free();
        var matrices = w.matrices.valueIterator();
        while (matrices.next()) |matrix| try matrix.prepareDense(r.operators.native, s);
        var idle: [4]state.Sequence = undefined;
        var ready: usize = 0;
        errdefer for (idle[0..ready]) |*seq| seq.free();
        for (&idle) |*seq| {
            seq.* = try state.Sequence.init(a, ctx.d, c, context, kv8);
            ready += 1;
        }
        var freq: [2]cuda.DeviceBuffer = undefined;
        var made: usize = 0;
        errdefer for (freq[0..made]) |*f| f.free();
        for (&freq, 0..) |*f, i| {
            const values = runtime.frequencies(c, i == 1);
            f.* = try cuda.DeviceBuffer.fromHost(ctx.d, std.mem.asBytes(&values));
            made += 1;
        }
        var rb: ?*@import("rounds.zig").Rounds = null;
        if (batched) rb = try @import("rounds.zig").Rounds.init(a, io, ctx.d, kernels, c.vocab_size, rows_cap);
        errdefer if (rb) |x| x.free();
        const m = try a.create(Model);
        errdefer a.destroy(m);
        var done: [8]u8 = undefined;
        if (rank == 0) {
            try link.send("ready-v1");
            if ((try link.recv(&done)).len != done.len) return error.RankNotReady;
        } else {
            if ((try link.recv(&done)).len != done.len) return error.RankNotReady;
            try link.send("ready-v1");
        }
        if (!std.mem.eql(u8, &done, "ready-v1")) return error.RankNotReady;
        m.* = .{ .a = a, .io = io, .d = ctx.d, .c = c, .rank = rank, .context = context, .stream = s, .r = r, .w = w, .link = link, .comm = comm, .freq = freq, .idle = idle, .identity = @import("cache_key.zig").namespace(c.storage, local.identity, execution), .rb = rb, .ctx = ctx, .kv8 = kv8, .dspark = dspark };
        if (rb != null) {
            if (gs.on) {
                m.gr = try a.create(rg.Graphs);
                errdefer {
                    m.gr.?.deinit();
                    a.destroy(m.gr.?);
                    m.gr = null;
                }
                m.gr.?.init(a, gs, rg.CudaPieces.pieces(), rg.nccl(&m.comm), null);
                m.gr.?.agreement = try graphs.Agreement.init(ctx.d, &m.comm, m.stream);
                m.gr.?.cache.agree = m.gr.?.agreement.?.agree();
            }
        }
        if (wc_on) {
            const wc = @import("window_cost.zig");
            const capped = @min(@as(u32, @intCast(c.dspark_block_size)) +| 1, wc.max_width);
            wc.measure(m, &m.wc, if (dspark) capped else 1) catch |err| std.log.warn("rank {d}: window costs stay static: {t}", .{ rank, err });
        }
        return m;
    }
    pub fn now(m: *Model) u64 {
        return @intCast(std.Io.Clock.awake.now(m.io).nanoseconds);
    }
    /// One timed pass over command-order rows: the batched verify when the plan allows it, the serial rows otherwise.
    pub fn timeRows(m: *Model, rows: []const cluster.round.Row) !void {
        if (m.rb) |rb| {
            const rd = @import("rounds.zig");
            var pl = rd.plan(m, rows, rb.rows_max);
            if (pl.fallback == .none) {
                try rd.verify(m, rd.Device, roundRow, &pl, null);
                try m.sync();
                return;
            }
        }
        try forwardRows(m, rows, null);
        try m.sync();
    }
    pub fn releaseRows(m: *Model, slot: u32) void {
        release(m, slot);
    }
    pub fn sync(m: *Model) !void {
        const start = std.Io.Clock.awake.now(m.io).nanoseconds;
        while (!try m.stream.done()) {
            m.comm.checkAsync() catch |err| {
                m.poison();
                return err;
            };
            if (std.Io.Clock.awake.now(m.io).nanoseconds - start > 120 * std.time.ns_per_s) {
                m.poison();
                return error.PeerTimeout;
            }
            try std.Io.sleep(m.io, .fromMilliseconds(1), .awake);
        }
    }
    pub fn poison(m: *Model) void {
        m.failed = true;
        if (m.gr) |g| g.dropAll(); // a poisoned rank replays nothing: every graph dies where its trust did
        m.comm.abort();
        m.link.shutdown();
    }
    pub fn close(m: *Model) void {
        m.sync() catch {};
        var it = m.sequences.valueIterator();
        while (it.next()) |s| s.free();
        m.sequences.deinit(m.a);
        var snapshots = m.snapshots.valueIterator();
        while (snapshots.next()) |image| image.free();
        m.snapshots.deinit(m.a);
        var media = m.media.valueIterator();
        while (media.next()) |list| { for (list.items) |*span| span.free(); list.deinit(m.a); }
        m.media.deinit(m.a);
        for (m.idle[0..m.idle_n]) |*s| s.free();
        for (&m.freq) |*f| f.free();
        if (m.gr) |g| {
            g.deinit(); // execs reference the stream and every staged buffer: they die first
            m.a.destroy(g);
            m.gr = null;
        }
        if (m.rb) |x| x.free();
        m.w.free();
        m.r.free();
        m.stream.deinit();
        m.comm.deinit();
        m.link.close();
        m.a.destroy(m);
    }
    pub fn shard(m: *Model) cluster.Shard {
        return .{ .ptr = m, .vtable = &.{ .forward = forward, .keep = keep, .release = release, .vocab = vocab, .command = command } };
    }
    fn command(p: *anyopaque, bytes: []const u8) anyerror!void { try @import("distributed_state.zig").command(self(p), bytes); }
    pub fn sequence(m: *Model, slot: u32) !*state.Sequence {
        if (m.sequences.getPtr(slot)) |s| return s;
        if (m.idle_n == 0) return error.TooManyActiveStreams;
        const got = try m.sequences.getOrPut(m.a, slot);
        m.idle_n -= 1;
        got.value_ptr.* = m.idle[m.idle_n];
        try got.value_ptr.reset(m.stream);
        return got.value_ptr;
    }
    fn self(p: *anyopaque) *Model {
        return @ptrCast(@alignCast(p));
    }
    fn vocab(p: *anyopaque) u32 {
        return @intCast(self(p).c.vocab_size);
    }
    fn forward(p: *anyopaque, a: std.mem.Allocator, x: exchange.Exchange, rows: []const cluster.round.Row, logits: ?[]f32) anyerror!void {
        _ = a;
        _ = x;
        const m = self(p);
        if (m.failed) return error.RankStateInvalid;
        errdefer {
            m.poison();
        }
        if (m.rb) |rb| {
            const rd = @import("rounds.zig");
            var pl = rd.plan(m, rows, rb.rows_max);
            if (pl.fallback == .none) {
                if (try graphRound(m, &pl)) return; // captured, replayed or cache-eager: the round ran
                return rd.verify(m, rd.Device, roundRow, &pl, logits);
            }
        }
        try forwardRows(m, rows, logits);
    }
    /// The graphed round: the plan's own audit, perf-kv's walk for the fingerprint, one engine call; true means it ran.
    fn graphRound(m: *Model, pl: *const @import("rounds.zig").Plan) !bool {
        const rd = @import("rounds.zig");
        const g = m.gr orelse return false;
        if (!pl.graphable or pl.n_sampled != 0) return false; // dspark and sampling rounds stay verify's
        const rb = m.rb.?;
        const rowtab = @import("rowtab.zig");
        try rd.Device.stage(m, pl); // fresh rowtab bytes every round: positions never live inside a graph
        var seqs: [rowtab.cap_max]*state.Sequence = undefined;
        var total: usize = 10; // RoundAddrs' fixed registers
        for (0..pl.n) |i| {
            const seq = try m.sequence(pl.rows[i].slot);
            if (seq.len != pl.rows[i].index) return error.PositionMismatch; // verify's audit, on the graphed path too
            seqs[i] = seq;
            m.glive[i] = .{ .slot = pl.rows[i].slot, .token = pl.rows[i].token };
            total += seq.partCount() + 1;
        }
        var walks: [rowtab.cap_max]rg.SeqWalk = undefined;
        for (0..pl.n) |i| walks[i] = .{ .ctx = seqs[i], .words = seqWords };
        const set: rg.RoundAddrs = .{
            .table = rb.table.ptr, .ids = rb.ids.ptr, .exch = rb.exch.ptr, .recv = rb.recv.ptr, .out = rb.out.ptr,
            .kv = m.r.p(.kv), .gate = m.r.p(.gate), .up = m.r.p(.up), .selected = m.r.p(.selected),
            .identity = rg.digestWord(m.identity), // the execution digest, kv8 gate included, folded to its word
            .seqs = walks[0..pl.n],
        };
        const words = try m.a.alloc(u64, total);
        defer m.a.free(words);
        const fp = try set.fingerprint(words);
        var run = GraphRun{ .m = m, .pl = pl };
        const outcome = try g.runRound(pl.n, pl.last_pos, m.stream, fp, true, .{ .ctx = &run, .run = graphBody }) orelse return false; // null: the round stays ours, and verify's path runs it
        switch (outcome) {
            .captured, .replayed => {
                for (0..pl.n) |i| seqs[i].len += 1; // the launch is the round's step: this is where rows advance
                try m.sync();
            },
            .eager => {}, // the cache's eager pass advanced and drained inside each row itself
        }
        return true;
    }
    const GraphRun = struct { m: *Model, pl: *const @import("rounds.zig").Plan };
    /// One round's program through the cache: rows in order, then the round's single allGather (cut when capturing).
    fn graphBody(ctx: *anyopaque, rec: ?*rg.Recorder, collective: ?rg.Collective, stream: cuda.Stream) anyerror!void {
        const b: *const GraphRun = @ptrCast(@alignCast(ctx));
        const m = b.m;
        const pl = b.pl;
        const tf = @import("text_forward.zig");
        var i: usize = 0;
        while (i < pl.n) : (i += 1) {
            const row = pl.rows[i];
            const seq = try m.sequence(row.slot);
            if (seq.len != row.index) return error.PositionMismatch;
            try tf.bodyRec(m, seq, row.token, 0, .none, null, @intCast(i), rec);
        }
        const rb = m.rb.?;
        const count: usize = @as(usize, pl.R) * rb.vh;
        if (rec) |r| return r.cut(.{ .all_gather = .{ .send = rb.exch.ptr, .recv = rb.recv.ptr, .count = count, .dtype = .f32 } });
        if (collective) |c| try c.allGather(c.ctx, rb.exch.ptr, rb.recv.ptr, count, .f32, stream) else try m.comm.allGather(rb.exch.ptr, rb.recv.ptr, count, .f32, m.stream);
        try m.sync(); // Device.round's drain, mirrored: the exchange completes before the caller returns
    }
    /// F1's production half: the graphed step fingerprints through perf-kv's snapshot walk, per row's sequence.
    fn seqWords(ctx: *anyopaque, out: []u64) anyerror!usize {
        const s: *state.Sequence = @ptrCast(@alignCast(ctx));
        return s.graphWords(out);
    }
    fn roundRow(m: anytype, seq: anytype, token: u32, sample: bool) anyerror!void {
        try @import("text_forward.zig").body(m, seq, token, 0, if (sample) .partial else .none, null);
    }
    fn forwardRows(m: *Model, rows: []const cluster.round.Row, logits: ?[]f32) !void {
        var n: usize = 0;
        var r: usize = 0;
        while (r < rows.len) {
            const row = rows[r];
            const seq = try m.sequence(row.slot);
            if (seq.len != row.index) return error.PositionMismatch;
            var embedded: u64 = 0;
            var image: ?@import("vision_serving.zig").Span = null;
            if (m.media.get(row.slot)) |spans| for (spans.items) |span| {
                if (row.index >= span.at and row.index < span.end) {
                    embedded = span.embeddings.ptr + @as(usize, row.index - span.at) * m.c.hidden_size * 2;
                    if (row.index == span.start) image = span;
                    break;
                }
            };
            if (image) |span| {
                const count = span.end - span.start;
                if (count > rows.len - r) return error.ChunkCutsImage;
                for (rows[r..][0..count]) |part| if (part.slot != row.slot) return error.ChunkCutsImage;
                const sample = rows[r + count - 1].sample;
                try @import("vision_forward.zig").run(m, seq, span, rows[r..][0..count], if (logits != null and sample) logits.?[n * m.c.vocab_size ..][0..m.c.vocab_size] else null);
                if (sample) n += 1;
                r += count;
                continue;
            }
            if (row.sample) try seq.before(m.stream);
            try @import("text_forward.zig").step(m, seq, row.token, row.sample, if (logits != null and row.sample) logits.?[n * m.c.vocab_size ..][0..m.c.vocab_size] else null, embedded);
            if (row.sample) n += 1;
            r += 1;
        }
    }
    fn keep(p: *anyopaque, slot: u32, len: u32) void {
        const m = self(p);
        if (m.sequences.getPtr(slot)) |s| s.keep(len, m.stream) catch {
            m.poison();
        };
    }
    fn release(p: *anyopaque, slot: u32) void {
        const m = self(p);
        if (m.media.fetchRemove(slot)) |entry| {
            var list = entry.value;
            for (list.items) |*span| span.free();
            list.deinit(m.a);
        }
        if (m.sequences.fetchRemove(slot)) |entry| {
            m.idle[m.idle_n] = entry.value;
            m.idle_n += 1;
        }
    }
};
pub const Control = struct {
    link: Link,
    rank: u32,
    buffer: []u8,
    pub fn view(c: *Control) exchange.Exchange {
        return .{ .ptr = c, .vtable = &.{ .rank = rankFn, .size = sizeFn, .exchange = step } };
    }
    fn rankFn(p: *anyopaque) u32 {
        const c: *Control = @ptrCast(@alignCast(p));
        return c.rank;
    }
    fn sizeFn(_: *anyopaque) u32 {
        return 2;
    }
    fn step(p: *anyopaque, chunks: []const []const u8, out: [][]const u8) exchange.Error!void {
        const c: *Control = @ptrCast(@alignCast(p));
        var received: []u8 = undefined;
        if (c.rank == 0) {
            c.link.send(chunks[1]) catch return error.PeerTimeout;
            received = c.link.recv(c.buffer) catch return error.PeerTimeout;
        } else {
            received = c.link.recvIdle(c.buffer) catch return error.PeerTimeout;
            c.link.send(chunks[0]) catch return error.PeerTimeout;
        }
        out[c.rank] = chunks[c.rank];
        out[1 - c.rank] = received;
    }
};
