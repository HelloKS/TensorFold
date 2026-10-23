//! Two-rank text model: source kernels, exact cache identity and bounded per-request state.
const std = @import("std");
const cuda = @import("cuda");
const cluster = @import("cluster_backend");
const Config = @import("config.zig").Config;
const state = @import("text_state.zig");
const runtime = @import("text_runtime.zig");
const Link = @import("link.zig").Link;
const exchange = cluster.exchange;
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
    pub fn open(a: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, cache: []const u8, kernels: ?[]const u8, rank: u32, ip: [4]u8, port: u16, context: usize) !*Model {
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
        try @import("text_admission.zig").check(a, io, ctx, c, &local.tensors, context, rank);
        var link = if (rank == 0) try Link.listen(ip, port) else try Link.connect(io, ip, port, 120);
        errdefer link.close();
        const identity = @import("text_identity.zig");
        const execution = try runtime.fingerprint(a, io, kernels, c.storage);
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
            seq.* = try state.Sequence.init(a, ctx.d, c, context);
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
        m.* = .{ .a = a, .io = io, .d = ctx.d, .c = c, .rank = rank, .context = context, .stream = s, .r = r, .w = w, .link = link, .comm = comm, .freq = freq, .idle = idle, .identity = @import("cache_key.zig").namespace(c.storage, local.identity, execution), .ctx = ctx };
        return m;
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
