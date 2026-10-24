//! NCCL gathers FP32 partials in rank order; with TF_DSV4_P2P=1 peer GPUs do the add directly.

// Copyright 2026 BertholomusAI (Albert Lee). Apache-2.0; adapted from dsv41/comm.zig.

const std = @import("std");
const cuda = @import("cuda");
const link_mod = @import("link.zig");
const Link = link_mod.Link;

const nccl = cuda.nccl;
const p2p = cuda.p2p;
const tp_watch = cuda.tp_watch;

pub const Comm = struct {
    g: cuda.tp_group.Group, // the shared rank plumbing: NCCL stays the default transport, byte for byte
    rank: u32, // the frozen surface: graphs.zig and text_model.zig read these directly
    world: u32,
    pa: ?p2p.Api = null, // the peer entry table, held exactly while the P2P exchange is live
    x: ?p2p.Exchange = null,
    mine: [3]u64 = .{ 0, 0, 0 }, // my two staging slots and my mailbox, while allocated
    allocs: usize = 0,
    roce_agreed: bool = false, // both ranks said yes in the P2P-v2 vote frame
    roce_tried: bool = false,
    drv: ?*cuda.Driver = null,

    /// Rank 0 makes the id and sends it; the others take it from the link. Needs the CUDA context current.
    pub fn init(link: Link, rank: u32, world: u32) !Comm {
        var lib = try nccl.Library.open();
        errdefer lib.close();
        try lib.requireSupportedVersion();
        var id: nccl.UniqueId = undefined;
        if (rank == 0) {
            try lib.check(lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId");
            try link.send(std.mem.asBytes(&id));
        } else {
            var buf: [@sizeOf(nccl.UniqueId)]u8 = undefined;
            const got = try link.recv(&buf);
            if (got.len != buf.len) return error.BadUniqueId;
            id = std.mem.bytesToValue(nccl.UniqueId, got);
        }
        var comm: nccl.Comm = null;
        if (envOn("TF_DSV4_WATCH")) startWatch(rank, world); // the ranks' lines before NCCL (as glm/nemotron): a rank lost in boot stops the other
        try lib.check(lib.api.ncclCommInitRank(&comm, @intCast(world), id, @intCast(rank)), "ncclCommInitRank");
        var c: Comm = .{ .g = .{ .lib = lib, .comm = comm, .rank = rank, .world = world }, .rank = rank, .world = world };
        if (world == 2) c.openTransports(link); // the vote rides the link; no frames at all while both envs stay off
        return c;
    }

    pub fn deinit(c: *Comm) void {
        if (envOn("TF_DSV4_WATCH") and tp_watch.open()) tp_watch.ended(); // a clean goodnight, not a deadline
        c.dropTransports();
        c.g.close();
        c.* = undefined;
    }

    pub fn abort(c: *Comm) void {
        c.g.abort();
        if (envOn("TF_DSV4_WATCH")) {
            var name: [40]u8 = undefined;
            const label = std.fmt.bufPrint(&name, "dsv4-rank-{d}: communicator abort", .{c.g.rank}) catch "dsv4-rank: abort";
            tp_watch.fail(label); // the named peer learns in about a second, not when a deadline expires
        }
    }

    pub fn checkAsync(c: *Comm) !void {
        if (c.g.aborted.load(.acquire)) return error.NcclAborted;
        var status: nccl.Result = 0;
        try c.g.lib.check(c.g.lib.api.ncclCommGetAsyncError(c.g.comm, &status), "ncclCommGetAsyncError");
        if (status != 0) {
            c.abort();
            return error.NcclFailed;
        }
        if (c.g.fast != null) {
            c.g.check() catch {
                c.abort();
                return error.NcclFailed; // a dead QP is a dead communicator: Model.sync sees exactly the two tags it sees today
            };
        }
    }

    /// `count` elements of `dtype` from every rank into `recv`, rank after rank.
    pub fn allGather(c: *Comm, send: u64, recv: u64, count: usize, dtype: nccl.DataType, stream: cuda.Stream) !void {
        if (c.roce_agreed and !c.roce_tried) c.openRoceOnce(stream);
        if (c.x) |*x| {
            if (dtype == .f32 and p2p.fits(count * 4)) { // fits() is deterministic: both ranks pick the same transport per round
                const my: u32 = @intCast(c.g.rank);
                p2p.allAdd(&c.pa.?, x, stream.handle, my, send, recv, count * 4) catch |err| {
                    c.abort(); // mid-stream P2P failure poisons: a per-round fallback would split the sequence numbers
                    return err;
                };
                return;
            }
        }
        if (c.g.fast != null and dtype == .f32) return c.g.allGather(stream, send, recv, count * 4);
        try c.g.lib.check(c.g.lib.api.ncclAllGather(send, recv, count, dtype, c.g.comm, stream.handle), "ncclAllGather");
    }

    /// The named peer lines to rank 0: NCCL_COMM_ID's host at its port plus one answers fatal() in about a second.
    fn startWatch(rank: u32, world: u32) void {
        var buf: [32]u8 = undefined;
        const ip = link_mod.bind_ip;
        const id = std.fmt.bufPrint(&buf, "{d}.{d}.{d}.{d}:{d}", .{ ip[0], ip[1], ip[2], ip[3], link_mod.bind_port }) catch return;
        buf[id.len] = 0;
        const idz: [*:0]const u8 = buf[0..id.len :0];
        if (env.setenv("NCCL_COMM_ID", idz, 1) != 0) return;
        tp_watch.start(std.Options.debug_io, rank, world) catch {}; // best effort: a missing line never delays open
    }

    // -- the vote's transports: TF_DSV4_P2P=1 and TF_DSV4_ROCE=1; off keeps today's path

    fn openTransports(c: *Comm, link: Link) void {
        const want_p2p = envOn("TF_DSV4_P2P");
        const want_roce = envOn("TF_DSV4_ROCE");
        if (!want_p2p and !want_roce) return; // zero frames, zero waits, zero allocations: the untouched path
        var api = p2p.Api.open() catch return; // the driver may simply not carry the peer symbols: NCCL, quietly
        if (!api.complete()) {
            api.close();
            return;
        }
        var offer = p2p.Probe{ .ok = false, .memops = false, .roce = want_roce, .bytes = p2p.staging_bytes_max };
        if (want_p2p) {
            if (p2p.memopsSupported(&api) catch false) { // staging exists before the vote so its handles can ride the frame
                offer.ok = c.allocMine(&api);
                offer.memops = offer.ok;
            }
        }
        var mapper: Mapper = .{ .api = &api };
        const pipe = linkPipe(link);
        var buf: [256]u8 = undefined;
        const vote: p2p.Vote = if (c.g.rank == 0)
            p2p.leaderVote(pipe, offer, &buf, Mapper.mapThree, &mapper) catch p2p.Vote{}
        else
            p2p.followerVote(pipe, offer, &buf, Mapper.mapThree, &mapper) catch p2p.Vote{};
        if (!vote.p2p) {
            for (mapper.out[0..mapper.n]) |p| p2p.unmapPeer(&api, p); // unmap exactly what got mapped before the vote went bad
            c.dropMine(&api); // nothing survives a lost vote: no allocation, no map, no dlopen handle
            api.close();
            return;
        }
        c.pa = api;
        c.x = .{
            .staging = .{ c.mine[0], c.mine[1] },
            .peer_staging = .{ mapper.out[0], mapper.out[1] },
            .mailbox = c.mine[2],
            .peer_mailbox = mapper.out[2],
            .seq = 0,
        };
        c.roce_agreed = vote.roce; // the small-gather opens on the first gather, where the hardware truth votes for real
    }

    fn allocMine(c: *Comm, api: *const p2p.Api) bool {
        while (c.allocs < 3) : (c.allocs += 1) {
            const bytes = if (c.allocs == 2) p2p.mailbox_bytes else p2p.staging_bytes_max;
            c.mine[c.allocs] = p2p.alloc(api, bytes) catch return false;
        }
        p2p.zeroD32(api, c.mine[2], p2p.mailbox_bytes / 4) catch return false; // both ranks wait for the first ring
        return true;
    }

    fn dropMine(c: *Comm, api: *const p2p.Api) void {
        var i: usize = 0;
        while (i < c.allocs) : (i += 1) p2p.free(api, c.mine[i]);
        c.allocs = 0;
    }

    fn dropTransports(c: *Comm) void {
        if (c.pa) |*api| {
            c.dropMine(api);
            if (c.x) |*x| {
                for (x.peer_staging) |p| p2p.unmapPeer(api, p); // unmap exactly what the vote mapped, in order
                p2p.unmapPeer(api, x.peer_mailbox);
            }
            api.close(); // the second dlopen closes with the exchange it carried
            c.pa = null;
        }
        c.x = null;
    }

    /// The small-gather's truth table is roce.zig's NCCL vote; reaching it together is what the link vote guaranteed.
    fn openRoceOnce(c: *Comm, stream: cuda.Stream) void {
        c.roce_tried = true;
        const d = c.driver() catch {
            c.abort(); // one rank cannot vote alone: dying loudly beats a rank split across transports
            return;
        };
        c.g.openRoce(std.heap.page_allocator, std.Options.debug_io, d, stream, p2p.staging_bytes_max) catch {
            c.abort();
            return;
        };
    }

    fn driver(c: *Comm) !*const cuda.Driver {
        if (c.drv) |d| return d;
        const box = try std.heap.page_allocator.create(cuda.Driver);
        errdefer std.heap.page_allocator.destroy(box);
        box.* = try cuda.Driver.open();
        c.drv = box; // the driver outlives the communicator, as the NCCL library does
        return box;
    }
};

const env = struct {
    extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int; // as glm/cuda_native.zig names it
};

fn envOn(name: [*:0]const u8) bool {
    const v = std.c.getenv(name) orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

/// The link's frames as the vote's transport; unread() hands a foreign frame back so ready-v1 still finds it whole.
var pipe_link: Link = undefined;

fn linkPipe(link: Link) p2p.Pipe {
    pipe_link = link;
    return .{ .ctx = &pipe_link, .send = pipeSend, .recvTimeout = pipeRecv, .unread = pipeUnread };
}

fn pipeSend(ctx: *anyopaque, bytes: []const u8) anyerror!void {
    const l: *const Link = @ptrCast(@alignCast(ctx));
    try l.send(bytes);
}

fn pipeRecv(ctx: *anyopaque, buf: []u8, ms: u32) anyerror!?[]u8 {
    const l: *const Link = @ptrCast(@alignCast(ctx));
    return l.recvTimeout(buf, ms);
}

fn pipeUnread(ctx: *anyopaque, bytes: []const u8) void {
    const l: *const Link = @ptrCast(@alignCast(ctx));
    l.unread(bytes);
}

/// The vote's mapping side: I only agree to ring buffers I actually mapped through the peer's IPC handle.
const Mapper = struct {
    api: *const p2p.Api,
    out: [3]u64 = .{ 0, 0, 0 },
    n: usize = 0,

    fn mapThree(ctx: *anyopaque, probe: p2p.Probe) bool {
        const m: *Mapper = @ptrCast(@alignCast(ctx));
        for (0..p2p.Probe.count) |i| {
            const p = p2p.mapPeer(m.api, probe.handles[i]) catch return false;
            m.out[i] = p;
            m.n = i + 1;
        }
        return true;
    }
};
