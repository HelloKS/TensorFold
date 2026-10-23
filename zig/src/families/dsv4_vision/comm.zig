//! NCCL gathers FP32 partials, then every rank adds them in rank order.

// Copyright 2026 BertholomusAI (Albert Lee). Apache-2.0; adapted from dsv41/comm.zig.

const std = @import("std");
const cuda = @import("cuda");
const Link = @import("link.zig").Link;

const nccl = cuda.nccl;

pub const Comm = struct {
    lib: nccl.Library,
    comm: nccl.Comm,
    rank: u32,
    world: u32,
    aborted: bool = false,

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
        try lib.check(lib.api.ncclCommInitRank(&comm, @intCast(world), id, @intCast(rank)), "ncclCommInitRank");
        return .{ .lib = lib, .comm = comm, .rank = rank, .world = world };
    }

    pub fn deinit(c: *Comm) void {
        if (!c.aborted) _ = c.lib.api.ncclCommDestroy(c.comm);
        c.lib.close();
        c.* = undefined;
    }

    pub fn abort(c: *Comm) void {
        if (!c.aborted) _ = c.lib.api.ncclCommAbort(c.comm);
        c.aborted = true;
    }

    pub fn checkAsync(c: *Comm) !void {
        if (c.aborted) return error.NcclAborted;
        var status: nccl.Result = 0;
        try c.lib.check(c.lib.api.ncclCommGetAsyncError(c.comm, &status), "ncclCommGetAsyncError");
        if (status != 0) {
            c.abort();
            return error.NcclFailed;
        }
    }

    /// `count` elements of `dtype` from every rank into `recv`, rank after rank.
    pub fn allGather(c: Comm, send: u64, recv: u64, count: usize, dtype: nccl.DataType, stream: cuda.Stream) !void {
        try c.lib.check(c.lib.api.ncclAllGather(send, recv, count, dtype, c.comm, stream.handle), "ncclAllGather");
    }
};
