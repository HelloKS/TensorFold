//! P2P rank plumbing: probe, IPC staging and stream memops order a two-rank FP32 all-add; missing symbols fall back.

const std = @import("std");
const platform = @import("sys");
const abi = @import("abi.zig");

const R = abi.Result;
const D = abi.DevicePtr;
const S = abi.Stream;

pub const Error = error{ P2pUnavailable, P2pFailed, DriverUnavailable, MissingSymbol, BadFrame };

pub const Handle = extern struct { bytes: [64]u8 = @splat(0) }; // CUipcMemHandle (cuda.h): one per cuMemAlloc allocation

pub const handle_bytes = @sizeOf(Handle);
pub const staging_bytes_max: usize = 0x5F_0000; // one rank's round partial: 24-row bucket × vocab/2 f32 = 6,205,440 B, aligned slack
pub const slots = 2; // seq-parity double-buffered staging, as roce.zig's doorbell slots
pub const mailbox_bytes = 64; // ready[2] then ack[2] u32 words, zeroed at open

pub const ready: u32 = 0; // word r: rank r's staging holds seq (written by rank r into the peer's mailbox)
pub const acked: u32 = 2; // word 2+r: rank r finished reading the peer's staging up to that seq

pub const peer_access_already_enabled: R = 203;
pub const peer_access_not_enabled: R = 204;
pub const lazy_enable_peer_access: c_uint = 1; // cuIpcOpenMemHandle enables the peer link itself when needed
pub const wait_value_geq: c_uint = 0; // CU_STREAM_WAIT_VALUE_GEQ: wrap-safe (signed-delta) compare
pub const attribute_can_use_stream_mem_ops_v2: c_int = 106;

pub const CanAccessPeer = *const fn (*c_int, c_int, c_int) callconv(.c) R;
pub const CtxEnablePeerAccess = *const fn (abi.Context, c_uint) callconv(.c) R;
pub const CtxDisablePeerAccess = *const fn (abi.Context) callconv(.c) R;
pub const IpcGetMemHandle = *const fn (*Handle, D) callconv(.c) R;
pub const IpcOpenMemHandle = *const fn (*D, Handle, c_uint) callconv(.c) R;
pub const IpcCloseMemHandle = *const fn (D) callconv(.c) R;
pub const StreamWriteValue32 = *const fn (S, D, u32, c_uint) callconv(.c) R;
pub const StreamWaitValue32 = *const fn (S, D, u32, c_uint) callconv(.c) R;
pub const DtoDAsync = *const fn (D, D, usize, S) callconv(.c) R;
pub const MemsetD32 = *const fn (D, c_uint, usize) callconv(.c) R;
pub const CtxGetDevice = *const fn (*abi.Device) callconv(.c) R;
pub const DeviceGetAttribute = *const fn (*c_int, c_int, abi.Device) callconv(.c) R;
pub const MemAlloc = *const fn (*D, usize) callconv(.c) R;
pub const MemFree = *const fn (D) callconv(.c) R;

/// The peer entry points the all-add needs, each optional: an absent symbol means no P2P, never a dead rank.
pub const Api = struct {
    lib: platform.DynLib = undefined,
    can_access_peer: ?CanAccessPeer = null,
    ctx_enable_peer_access: ?CtxEnablePeerAccess = null,
    ctx_disable_peer_access: ?CtxDisablePeerAccess = null,
    ipc_get_mem_handle: ?IpcGetMemHandle = null,
    ipc_open_mem_handle: ?IpcOpenMemHandle = null,
    ipc_close_mem_handle: ?IpcCloseMemHandle = null,
    stream_write_value32: ?StreamWriteValue32 = null,
    stream_wait_value32: ?StreamWaitValue32 = null,
    memcpy_dto_d_async: ?DtoDAsync = null,
    memset_d32: ?MemsetD32 = null,
    ctx_get_device: ?CtxGetDevice = null,
    device_get_attribute: ?DeviceGetAttribute = null,
    mem_alloc: ?MemAlloc = null,
    mem_free: ?MemFree = null,

    /// Opens the driver a second time for the peer symbols, hand-written: an absent symbol simply stays null.
    pub fn open() Error!Api {
        var lib = platform.DynLib.open(platform.cuda_driver) catch return error.DriverUnavailable;
        errdefer lib.close();
        return .{
            .lib = lib,
            .can_access_peer = lib.lookup(CanAccessPeer, "cuDeviceCanAccessPeer"),
            .ctx_enable_peer_access = lib.lookup(CtxEnablePeerAccess, "cuCtxEnablePeerAccess"),
            .ctx_disable_peer_access = lib.lookup(CtxDisablePeerAccess, "cuCtxDisablePeerAccess"),
            .ipc_get_mem_handle = lib.lookup(IpcGetMemHandle, "cuIpcGetMemHandle"),
            .ipc_open_mem_handle = lib.lookup(IpcOpenMemHandle, "cuIpcOpenMemHandle"),
            .ipc_close_mem_handle = lib.lookup(IpcCloseMemHandle, "cuIpcCloseMemHandle"),
            .stream_write_value32 = lib.lookup(StreamWriteValue32, "cuStreamWriteValue32_v2"),
            .stream_wait_value32 = lib.lookup(StreamWaitValue32, "cuStreamWaitValue32_v2"),
            .memcpy_dto_d_async = lib.lookup(DtoDAsync, "cuMemcpyDtoDAsync_v2"),
            .memset_d32 = lib.lookup(MemsetD32, "cuMemsetD32_v2"),
            .ctx_get_device = lib.lookup(CtxGetDevice, "cuCtxGetDevice"),
            .device_get_attribute = lib.lookup(DeviceGetAttribute, "cuDeviceGetAttribute"),
            .mem_alloc = lib.lookup(MemAlloc, "cuMemAlloc_v2"),
            .mem_free = lib.lookup(MemFree, "cuMemFree_v2"),
        };
    }

    pub fn close(a: *Api) void {
        a.lib.close();
        a.* = undefined;
    }

    /// True only when every entry the protocol calls is present; the caller then keeps NCCL.
    pub fn complete(a: *const Api) bool {
        return a.can_access_peer != null and a.ctx_enable_peer_access != null and a.ctx_disable_peer_access != null and
            a.ipc_get_mem_handle != null and a.ipc_open_mem_handle != null and a.ipc_close_mem_handle != null and
            a.stream_write_value32 != null and a.stream_wait_value32 != null and a.memcpy_dto_d_async != null and
            a.memset_d32 != null and a.ctx_get_device != null and a.device_get_attribute != null and
            a.mem_alloc != null and a.mem_free != null;
    }
};

fn need(comptime T: type, f: ?T) !T {
    return f orelse error.MissingSymbol;
}

/// Both directions of cuDeviceCanAccessPeer; false simply means the caller keeps NCCL.
pub fn probe(a: *const Api, x: c_int, y: c_int) Error!bool {
    const can = try need(CanAccessPeer, a.can_access_peer);
    var ok: c_int = 0;
    if (can(&ok, x, y) != abi.success or ok == 0) return false;
    if (can(&ok, y, x) != abi.success or ok == 0) return false;
    return true;
}

/// cuCtxEnablePeerAccess, tolerant of a link the driver (or cuIpcOpenMemHandle) already raised.
pub fn enablePeerAccess(a: *const Api, peer: abi.Context) Error!void {
    const en = try need(CtxEnablePeerAccess, a.ctx_enable_peer_access);
    switch (en(peer, 0)) {
        abi.success, peer_access_already_enabled => return,
        else => return error.P2pUnavailable,
    }
}

pub fn disablePeerAccess(a: *const Api, peer: abi.Context) void {
    const dis = a.ctx_disable_peer_access orelse return;
    _ = dis(peer); // 204 (not enabled) and teardown races are noise: the contexts are being released
}

/// Maps the peer's allocation into this context, letting the driver raise the peer link itself.
pub fn mapPeer(a: *const Api, h: Handle) Error!D {
    const open = try need(IpcOpenMemHandle, a.ipc_open_mem_handle);
    var p: D = 0;
    if (open(&p, h, lazy_enable_peer_access) != abi.success) return error.P2pUnavailable;
    return p;
}

pub fn unmapPeer(a: *const Api, p: D) void {
    if (a.ipc_close_mem_handle) |close| _ = close(p);
}

pub fn handleOf(a: *const Api, p: D) Error!Handle {
    const get = try need(IpcGetMemHandle, a.ipc_get_mem_handle);
    var h: Handle = undefined;
    if (get(&h, p) != abi.success) return error.P2pUnavailable;
    return h;
}

/// Whether this context's device accepts stream memops (the wait/write the exchange orders on).
pub fn memopsSupported(a: *const Api) Error!bool {
    const get_dev = try need(CtxGetDevice, a.ctx_get_device);
    const attr = try need(DeviceGetAttribute, a.device_get_attribute);
    var dev: abi.Device = 0;
    if (get_dev(&dev) != abi.success) return error.P2pUnavailable;
    var v: c_int = 0;
    if (attr(&v, attribute_can_use_stream_mem_ops_v2, dev) != abi.success) return false;
    return v != 0;
}

pub fn writeValue32(a: *const Api, s: S, p: D, v: u32) Error!void {
    const w = try need(StreamWriteValue32, a.stream_write_value32);
    if (w(s, p, v, 0) != abi.success) return error.P2pFailed; // write value, default membar
}

/// Stream-ordered wait: proceed when `*p` has reached `v` (wrap-safe GEQ compare).
pub fn waitValue32(a: *const Api, s: S, p: D, v: u32) Error!void {
    const w = try need(StreamWaitValue32, a.stream_wait_value32);
    if (w(s, p, v, wait_value_geq) != abi.success) return error.P2pFailed;
}

pub fn copyDtoD(a: *const Api, s: S, dst: D, src: D, bytes: usize) Error!void {
    const c = try need(DtoDAsync, a.memcpy_dto_d_async);
    if (c(dst, src, bytes, s) != abi.success) return error.P2pFailed;
}

pub fn zeroD32(a: *const Api, p: D, words: usize) Error!void {
    const m = try need(MemsetD32, a.memset_d32);
    if (m(p, 0, words) != abi.success) return error.P2pFailed;
}

/// A protocol-private allocation (raw cuMemAlloc so the IPC handle names its exact base).
pub fn alloc(a: *const Api, bytes: usize) Error!D {
    const m = try need(MemAlloc, a.mem_alloc);
    var p: D = 0;
    if (m(&p, bytes) != abi.success) return error.P2pUnavailable;
    return p;
}

pub fn free(a: *const Api, p: D) void {
    if (a.mem_free) |f| _ = f(p);
}

/// A gather this way only when its bytes fit a slot and suit the copy engine; both ranks decide alike (same size).
pub fn fits(bytes: usize) bool {
    return bytes > 0 and bytes <= staging_bytes_max and bytes % 16 == 0;
}

/// The seq-parity staging slot a sequence owns.
pub fn slotOf(seq: u32) u32 {
    return seq & (slots - 1);
}

/// Wrap-safe `now >= want` for a monotonic u32 sequence (the memop GEQ's own compare, host-side twin).
pub fn reached(now: u32, want: u32) bool {
    return @as(i32, @bitCast(now -% want)) >= 0;
}

/// One rank's side of the exchange: its staging slots, the peer's mapped ones, both mailbox views, one sequence.
pub const Exchange = struct {
    staging: [slots]D = .{ 0, 0 }, // this rank's cuMemAlloc'd slots (handles cross the link at setup)
    peer_staging: [slots]D = .{ 0, 0 }, // the peer's slots mapped into this context
    mailbox: D = 0, // this rank's flag words (the peer writes them)
    peer_mailbox: D = 0, // the peer's flag words mapped here (this rank writes them)
    seq: u32 = 0,

    pub fn word(base: D, index: u32) D {
        return base + @as(D, index) * 4;
    }
};

/// One FP32 rank partial into gather in rank order: slot copy, ring ready, the peer's half in over the link, ack.
pub fn allAdd(a: *const Api, x: *Exchange, s: S, my: u32, partial: D, gather: D, bytes: usize) Error!void {
    const seq = x.seq +% 1;
    const slot = slotOf(seq);
    const other = 1 - my;
    try waitValue32(a, s, Exchange.word(x.mailbox, acked + other), seq -% slots); // slot free: peer read up to seq-2
    try copyDtoD(a, s, x.staging[slot], partial, bytes); // my partial to my slot (the peer reads slots, never my buffer)
    try copyDtoD(a, s, gather + @as(D, my) * bytes, partial, bytes); // my own half of the gather, rank order
    try writeValue32(a, s, Exchange.word(x.peer_mailbox, ready + my), seq); // ring: my slot holds seq
    try waitValue32(a, s, Exchange.word(x.mailbox, ready + other), seq); // the peer's slot holds seq
    try copyDtoD(a, s, gather + @as(D, other) * bytes, x.peer_staging[slot], bytes); // its half in, over the link
    try writeValue32(a, s, Exchange.word(x.peer_mailbox, acked + my), seq); // I read its slot: its next slot write may pass
    x.seq = seq;
}

// -- negotiation: the P2P-v1 frame and the vote every mixed-version pair must survive --------------------------------

pub const magic = "TFP2P2"; // the frame tag; anything else (an old peer's next frame) is simply not ours

pub const Probe = struct {
    ok: bool, // this rank probed peer-capable (staging allocated, handles ready)
    memops: bool, // and its device takes stream memops
    roce: bool, // this rank wants the optional RoCE small-gather (TF_DSV4_ROCE=1)
    bytes: u32, // the staging slot size both ranks must share
    handles: [slots + 1]Handle = @splat(.{}), // staging slots, then the mailbox

    pub const header: usize = magic.len + 8; // magic, ok, memops, roce, bytes(4), handle count
    pub const count: usize = slots + 1; // two staging slots and the mailbox

    pub fn frameLen(n: usize) usize {
        return header + n * handle_bytes;
    }
};

pub fn encodeProbe(p: Probe, out: []u8) !void {
    if (out.len < Probe.frameLen(Probe.count)) return error.BadFrame;
    @memcpy(out[0..magic.len], magic);
    out[magic.len] = @intFromBool(p.ok);
    out[magic.len + 1] = @intFromBool(p.memops);
    out[magic.len + 2] = @intFromBool(p.roce);
    std.mem.writeInt(u32, out[magic.len + 3 ..][0..4], p.bytes, .little);
    out[magic.len + 7] = Probe.count;
    for (0..Probe.count) |i| @memcpy(out[Probe.header + i * handle_bytes ..][0..handle_bytes], &p.handles[i].bytes);
}

/// The P2P-v2 frame from `bytes`, or null when it is not ours (an old leader's silence, or some other frame).
pub fn decodeProbe(bytes: []const u8) ?Probe {
    if (bytes.len != Probe.frameLen(Probe.count) or !std.mem.startsWith(u8, bytes, magic)) return null;
    if (bytes[magic.len + 7] != Probe.count) return null; // a frame claiming other handles is not this protocol's
    var p: Probe = .{
        .ok = bytes[magic.len] != 0,
        .memops = bytes[magic.len + 1] != 0,
        .roce = bytes[magic.len + 2] != 0,
        .bytes = std.mem.readInt(u32, bytes[magic.len + 3 ..][0..4], .little),
    };
    for (0..Probe.count) |i| @memcpy(&p.handles[i].bytes, bytes[Probe.header + i * handle_bytes ..][0..handle_bytes]);
    return p;
}

/// The leader's commit word after it mapped the follower's buffers; only the leader ever sends it.
pub const ack_magic = "P2PA";
pub const ack_len = ack_magic.len + 2; // the p2p verdict, then the roce verdict

pub const Ack = struct { p2p: bool = false, roce: bool = false };

pub fn encodeAck(a: Ack, out: []u8) !void {
    if (out.len < ack_len) return error.BadFrame;
    @memcpy(out[0..ack_magic.len], ack_magic);
    out[ack_magic.len] = @intFromBool(a.p2p);
    out[ack_magic.len + 1] = @intFromBool(a.roce);
}

pub fn decodeAck(bytes: []const u8) ?Ack {
    if (bytes.len != ack_len or !std.mem.startsWith(u8, bytes, ack_magic)) return null;
    return .{ .p2p = bytes[ack_magic.len] != 0, .roce = bytes[ack_magic.len + 1] != 0 };
}

/// p2p needs both ranks' probes; roce needs both ranks to want it (hardware truth votes later, inside roce.zig).
pub fn p2pAgree(offer: Probe, reply: Probe) bool {
    return offer.ok and offer.memops and reply.ok and reply.memops and offer.bytes == reply.bytes;
}

pub fn roceAgree(offer: Probe, reply: Probe) bool {
    return offer.roce and reply.roce;
}

/// Transport the vote rides on (link.zig's frames in the rank's process; a scripted queue in host tests).
pub const Pipe = struct {
    ctx: *anyopaque,
    send: *const fn (ctx: *anyopaque, bytes: []const u8) anyerror!void,
    recvTimeout: *const fn (ctx: *anyopaque, buf: []u8, ms: u32) anyerror!?[]u8,
    unread: *const fn (ctx: *anyopaque, bytes: []const u8) void, // a frame that was not ours goes back: stream stays whole
};

pub const vote_ms: u32 = 2000; // only at open, beside the UniqueId frame: no rank ever spins or dies on the vote

pub const Vote = struct { p2p: bool = false, roce: bool = false };

/// Leader: offer, await the reply, map the follower's buffers, ack both verdicts. Silence or foreign frames mean NCCL.
pub fn leaderVote(pipe: Pipe, my: Probe, buf: []u8, map: *const fn (ctx: *anyopaque, p: Probe) bool, map_ctx: *anyopaque) Error!Vote {
    var out: [Probe.frameLen(Probe.count)]u8 = undefined;
    try encodeProbe(my, &out);
    pipe.send(pipe.ctx, &out) catch return error.P2pFailed;
    const got = pipe.recvTimeout(pipe.ctx, buf, vote_ms) catch return error.P2pFailed;
    const reply = decodeProbe(got orelse return .{}) orelse {
        if (got) |bytes| pipe.unread(pipe.ctx, bytes); // a foreign frame belongs to the next reader (an old peer's ready-v1)
        return .{};
    };
    const a: Ack = .{ .p2p = p2pAgree(my, reply) and map(map_ctx, reply), .roce = roceAgree(my, reply) };
    encodeAck(a, &out) catch return error.P2pFailed;
    pipe.send(pipe.ctx, out[0..ack_len]) catch return error.P2pFailed;
    return .{ .p2p = a.p2p, .roce = a.roce };
}

/// Follower: await the offer (an old leader sends nothing), map, reply, and adopt only the leader's acked word.
pub fn followerVote(pipe: Pipe, my: Probe, buf: []u8, map: *const fn (ctx: *anyopaque, p: Probe) bool, map_ctx: *anyopaque) Error!Vote {
    const got = pipe.recvTimeout(pipe.ctx, buf, vote_ms) catch return error.P2pFailed;
    const offer = decodeProbe(got orelse return .{}) orelse {
        if (got) |bytes| pipe.unread(pipe.ctx, bytes); // no P2P-v2 came (or none ours): the leader is old or off; NCCL
        return .{};
    };
    var reply = my;
    reply.ok = my.ok and p2pAgree(offer, my) and map(map_ctx, offer); // I ring only buffers I mapped: agreed, never assumed
    var out: [Probe.frameLen(Probe.count)]u8 = undefined;
    try encodeProbe(reply, &out);
    pipe.send(pipe.ctx, &out) catch return error.P2pFailed;
    const acked_bytes = pipe.recvTimeout(pipe.ctx, buf, vote_ms) catch return error.P2pFailed;
    const verdict = decodeAck(acked_bytes orelse return .{}) orelse return error.P2pFailed;
    return .{ .p2p = verdict.p2p, .roce = verdict.roce }; // the leader decided both; I adopt exactly its word
}
