//! Host fakes for p2p.zig: the vote protocol, the gather ordering and the frame codec, run only by test builds.

const std = @import("std");
const abi = @import("abi.zig");
const p2p = @import("p2p.zig");

const R = abi.Result;
const D = abi.DevicePtr;
const S = abi.Stream;


const fake = struct {
    var can = true;
    var open_fails = false;
    var writes: [64]u32 = undefined;
    var writes_n: usize = 0;
    var waits: [64]u32 = undefined;
    var waits_n: usize = 0;
    var copies: [64]u64 = undefined;
    var copies_n: usize = 0;

    fn reset() void {
        can = true;
        open_fails = false;
        writes_n = 0;
        waits_n = 0;
        copies_n = 0;
    }

    fn canAccessPeer(out: *c_int, x: c_int, _: c_int) callconv(.c) R {
        out.* = @intFromBool(can and x >= 0);
        return abi.success;
    }
    fn ipcOpen(p: *D, h: p2p.Handle, _: c_uint) callconv(.c) R {
        if (open_fails) return 1;
        p.* = @as(D, h.bytes[0]) * 4096; // deterministic stand-in addresses so tests can tell slots apart
        return abi.success;
    }
    fn write32(_: S, p: D, v: u32, _: c_uint) callconv(.c) R {
        writes[writes_n] = v *% 64 + @as(u32, @intCast((p / 4) % 8)); // (value, word) pairs for the order assertions
        writes_n += 1;
        return abi.success;
    }
    fn wait32(_: S, p: D, v: u32, _: c_uint) callconv(.c) R {
        waits[waits_n] = v *% 64 + @as(u32, @intCast((p / 4) % 8));
        waits_n += 1;
        return abi.success;
    }
    fn copy(dst: D, src: D, _: usize, _: S) callconv(.c) R {
        copies[copies_n] = dst * 16 + src / 4096;
        copies_n += 1;
        return abi.success;
    }
    fn getDevice(dev: *abi.Device) callconv(.c) R {
        dev.* = 0;
        return abi.success;
    }
    fn getAttribute(out: *c_int, attr: c_int, _: abi.Device) callconv(.c) R {
        out.* = @intFromBool(attr == p2p.attribute_can_use_stream_mem_ops_v2 and can);
        return abi.success;
    }
};

fn fakeApi() p2p.Api {
    fake.reset();
    var a: p2p.Api = .{};
    a.can_access_peer = fake.canAccessPeer;
    a.ipc_open_mem_handle = fake.ipcOpen;
    a.stream_write_value32 = fake.write32;
    a.stream_wait_value32 = fake.wait32;
    a.memcpy_dto_d_async = fake.copy;
    a.ctx_get_device = fake.getDevice;
    a.device_get_attribute = fake.getAttribute;
    return a;
}

const Script = struct {
    inbox: [4][512]u8 = undefined,
    in_len: [4]usize = .{ 0, 0, 0, 0 },
    in_n: usize = 0,
    in_at: usize = 0,
    outbox: [4][512]u8 = undefined,
    out_len: [4]usize = .{ 0, 0, 0, 0 },
    out_n: usize = 0,
    silence: bool = false, // every read times out, as if the peer never spoke
    map_ok: bool = true,
    maps: usize = 0,

    fn put(s: *Script, bytes: []const u8) void {
        @memcpy(s.inbox[s.in_n][0..bytes.len], bytes);
        s.in_len[s.in_n] = bytes.len;
        s.in_n += 1;
    }
    fn send(c: *anyopaque, bytes: []const u8) anyerror!void {
        const s: *Script = @ptrCast(@alignCast(c));
        @memcpy(s.outbox[s.out_n][0..bytes.len], bytes);
        s.out_len[s.out_n] = bytes.len;
        s.out_n += 1;
    }
    fn recvTimeout(c: *anyopaque, buf: []u8, _: u32) anyerror!?[]u8 {
        const s: *Script = @ptrCast(@alignCast(c));
        if (s.silence or s.in_at >= s.in_n) return null; // a timed-out read consumed nothing: the stream stays whole
        const f = s.inbox[s.in_at][0..s.in_len[s.in_at]];
        if (buf.len < f.len) return error.BufferTooSmall;
        @memcpy(buf[0..f.len], f);
        s.in_at += 1;
        return buf[0..f.len];
    }
    fn unread(c: *anyopaque, bytes: []const u8) void {
        const s: *Script = @ptrCast(@alignCast(c));
        _ = bytes;
        s.in_at -= 1; // the rejected frame returns to the stream: the next reader (ready-v1) sees it untouched
    }
    fn mapAll(c: *anyopaque, _: p2p.Probe) bool {
        const s: *Script = @ptrCast(@alignCast(c));
        s.maps += 1;
        return s.map_ok;
    }
    fn pipe(s: *Script) p2p.Pipe {
        return .{ .ctx = s, .send = send, .recvTimeout = recvTimeout, .unread = unread };
    }
};

test "P2P probe: both directions or nothing, and a declined map is never half-adopted" {
    var a = fakeApi();
    try std.testing.expect(try p2p.probe(&a, 0, 1));
    fake.can = false;
    try std.testing.expectEqual(false, try p2p.probe(&a, 0, 1));
    try std.testing.expectEqual(false, try p2p.memopsSupported(&a));
    fake.can = true;
    try std.testing.expect(try p2p.memopsSupported(&a));
    const h = p2p.Handle{ .bytes = @splat(7) };
    fake.open_fails = true;
    try std.testing.expectError(error.P2pUnavailable, p2p.mapPeer(&a, h)); // the caller keeps NCCL, adopting nothing
    fake.open_fails = false;
    _ = try p2p.mapPeer(&a, h);
}

test "the exchange orders slot, ring, wait, link copy and ack on every gather, and the slot guard precedes reuse" {
    var a = fakeApi();
    var x: p2p.Exchange = .{ .staging = .{ 4096, 8192 }, .peer_staging = .{ 12288, 16384 }, .mailbox = 0, .peer_mailbox = 1024 };
    for (0..4) |k| {
        const seq: u32 = @intCast(k + 1);
        const slot = p2p.slotOf(seq);
        try p2p.allAdd(&a, &x, null, 0, 20480, 32768, 4096);
        try std.testing.expectEqual(seq, x.seq);
        try std.testing.expectEqual(@as(u32, seq -% p2p.slots) *% 64 + p2p.acked + 1, fake.waits[2 * k]); // slot guard first
        try std.testing.expectEqual(seq *% 64 + p2p.ready + 1, fake.waits[2 * k + 1]); // the peer's ring
        try std.testing.expectEqual(seq *% 64 + p2p.ready, fake.writes[2 * k]); // ring after my copies
        try std.testing.expectEqual(seq *% 64 + p2p.acked, fake.writes[2 * k + 1]); // my ack last
        try std.testing.expectEqual(@as(u64, x.staging[slot]) * 16 + 20480 / 4096, fake.copies[3 * k]); // partial to slot
        try std.testing.expectEqual(32768 * 16 + 20480 / 4096, fake.copies[3 * k + 1]); // my own half of the gather
        const peer_dst: u64 = 32768 + 1 * 4096; // the peer's half of the gather, over the mapped link
        try std.testing.expectEqual(peer_dst * 16 + x.peer_staging[slot] / 4096, fake.copies[3 * k + 2]);
    }
    try std.testing.expectEqual(@as(u32, 4), x.seq);
}

test "the vote never half-adopts: silence, foreign frames and every decline keep both ranks on NCCL" {
    var buf: [256]u8 = undefined;
    const good = p2p.Probe{ .ok = true, .memops = true, .roce = false, .bytes = p2p.staging_bytes_max };
    var lead: Script = .{};
    const lonly = try p2p.leaderVote(lead.pipe(), good, &buf, Script.mapAll, &lead);
    try std.testing.expectEqual(false, lonly.p2p); // old follower: silence meant NCCL, and no ack went out
    try std.testing.expectEqual(@as(usize, 1), lead.out_n); // only the offer rode the wire
    var foll: Script = .{};
    try std.testing.expectEqual(false, (try p2p.followerVote(foll.pipe(), good, &buf, Script.mapAll, &foll)).p2p); // old leader
    try std.testing.expectEqual(@as(usize, 0), foll.out_n); // a rank with nothing to say about a silent leader sends nothing
    var foreign: [12]u8 = .{ 'r', 'e', 'a', 'd', 'y', '-', 'v', '1', '!', '!', '!', '!' };
    foll = .{};
    foll.put(&foreign);
    try std.testing.expectEqual(false, (try p2p.followerVote(foll.pipe(), good, &buf, Script.mapAll, &foll)).p2p);
    try std.testing.expectEqual(@as(usize, 0), foll.out_n); // foreign frame: NCCL, no reply, stream untouched
    try std.testing.expectEqual(@as(usize, 0), foll.in_at); // and the frame went back: ready-v1 will still find it whole
    try std.testing.expectEqual(foreign.len, foll.in_len[0]);
}

test "a failed handle-map after the vote means NCCL on both ranks, never a half-adopted P2P" {
    var lbuf: [256]u8 = undefined;
    var fbuf: [256]u8 = undefined;
    const good = p2p.Probe{ .ok = true, .memops = true, .roce = false, .bytes = p2p.staging_bytes_max };
    var lead: Script = .{};
    var foll: Script = .{};
    foll.map_ok = false; // the follower cannot map the leader's staging: it must decline in its reply
    var offer: [p2p.Probe.frameLen(p2p.Probe.count)]u8 = undefined;
    try p2p.encodeProbe(good, &offer);
    foll.put(&offer);
    var ack: [6]u8 = undefined;
    try p2p.encodeAck(.{ .p2p = false, .roce = false }, &ack);
    foll.put(&ack); // the verdict the leader will send once it reads the decline
    const fv = try p2p.followerVote(foll.pipe(), good, &fbuf, Script.mapAll, &foll);
    try std.testing.expectEqual(false, fv.p2p);
    try std.testing.expectEqual(@as(usize, 1), foll.maps); // it tried, failed, and said so
    lead.put(foll.outbox[0][0..foll.out_len[0]]); // the leader reads that very decline
    const lv = try p2p.leaderVote(lead.pipe(), good, &lbuf, Script.mapAll, &lead);
    try std.testing.expectEqual(false, lv.p2p); // both ranks chose NCCL: agreed, not assumed
    try std.testing.expectEqual(@as(usize, 0), lead.maps); // the leader never even mapped a declined offer
    lead = .{};
    lead.map_ok = false; // now the leader's side of the exchange fails after an agreeing reply
    foll = .{};
    var reply: [p2p.Probe.frameLen(p2p.Probe.count)]u8 = undefined;
    try p2p.encodeProbe(good, &reply);
    lead.put(&reply);
    const lv2 = try p2p.leaderVote(lead.pipe(), good, &lbuf, Script.mapAll, &lead);
    try std.testing.expectEqual(false, lv2.p2p and lv2.roce);
    try std.testing.expectEqual(@as(usize, 1), lead.maps); // it tried to map and told the follower no in the ack
    try std.testing.expectEqual(@as(usize, 0), lead.outbox[1][p2p.ack_magic.len]); // the ack the follower will adopt says NCCL
}

test "agreeing probes adopt p2p (and roce) identically on both ranks, and the ack word is the follower's law" {
    var lbuf: [256]u8 = undefined;
    var fbuf: [256]u8 = undefined;
    const good = p2p.Probe{ .ok = true, .memops = true, .roce = true, .bytes = p2p.staging_bytes_max };
    var lead: Script = .{};
    var foll: Script = .{};
    var offer: [p2p.Probe.frameLen(p2p.Probe.count)]u8 = undefined;
    try p2p.encodeProbe(good, &offer);
    foll.put(&offer);
    var ack: [6]u8 = undefined;
    try p2p.encodeAck(.{ .p2p = true, .roce = true }, &ack);
    foll.put(&ack);
    const fv = try p2p.followerVote(foll.pipe(), good, &fbuf, Script.mapAll, &foll);
    try std.testing.expect(fv.p2p and fv.roce); // the follower adopts exactly the leader's acked word
    var mismatch: p2p.Probe = good;
    mismatch.bytes = p2p.staging_bytes_max / 2; // one slot size for both ranks or none
    var m: Script = .{};
    m.put(&offer);
    try p2p.encodeAck(.{}, &ack);
    m.put(&ack);
    const mv = try p2p.followerVote(m.pipe(), mismatch, &fbuf, Script.mapAll, &m);
    try std.testing.expectEqual(false, mv.p2p);
    try std.testing.expectEqual(@as(usize, 0), m.maps); // a disagreed vote never maps anything
    lead.put(foll.outbox[0][0..foll.out_len[0]]); // the leader reads the follower's agreeing reply
    const lv = try p2p.leaderVote(lead.pipe(), good, &lbuf, Script.mapAll, &lead);
    try std.testing.expect(lv.p2p and lv.roce); // both ranks reached the same transport from the same two frames
}

test "seq wrap and slot parity agree with the memop's signed-delta compare" {
    try std.testing.expect(p2p.reached(4, 3) and p2p.reached(3, 3) and !p2p.reached(3, 4));
    try std.testing.expect(p2p.reached(1, 0xFFFF_FFFE)); // a step past the wrap still "reached" the far-back want
    try std.testing.expect(p2p.reached(0, 0xFFFF_FFFF) and !p2p.reached(0xFFFF_FFFF, 0));
    try std.testing.expect(p2p.reached(0xFFFF_FFFF, 0x8000_0000) and !p2p.reached(0x7FFF_FFFF, 0x8000_0000)); // half the space ahead, half behind
    try std.testing.expectEqual(@as(u32, 1), p2p.slotOf(1));
    try std.testing.expectEqual(@as(u32, 0), p2p.slotOf(2));
    try std.testing.expectEqual(p2p.slotOf(0xFFFF_FFFF), p2p.slotOf(1));
}

test "fits decides both ranks alike and the frame round-trips or refuses" {
    try std.testing.expect(p2p.fits(4096) and p2p.fits(p2p.staging_bytes_max));
    try std.testing.expect(!p2p.fits(0) and !p2p.fits(15) and !p2p.fits(p2p.staging_bytes_max + 16)); // same size -> same choice on both ranks
    var p = p2p.Probe{ .ok = true, .memops = true, .roce = false, .bytes = p2p.staging_bytes_max };
    p.handles[0].bytes[0] = 3;
    p.handles[p2p.slots].bytes[63] = 9;
    var out: [p2p.Probe.frameLen(p2p.Probe.count)]u8 = undefined;
    try p2p.encodeProbe(p, &out);
    const got = p2p.decodeProbe(&out).?;
    try std.testing.expect(got.ok and got.memops);
    try std.testing.expectEqual(p.bytes, got.bytes);
    try std.testing.expectEqualSlices(u8, &p.handles[0].bytes, &got.handles[0].bytes);
    try std.testing.expectEqualSlices(u8, &p.handles[p2p.slots].bytes, &got.handles[p2p.slots].bytes);
    try std.testing.expectEqual(@as(usize, 206), out.len);
    var tampered = out;
    tampered[0] = 'X';
    try std.testing.expectEqual(@as(?p2p.Probe, null), p2p.decodeProbe(&tampered)); // any other frame (an old peer's) is not ours
    tampered[0] = p2p.magic[0];
    tampered[p2p.magic.len + 7] = 4;
    try std.testing.expectEqual(@as(?p2p.Probe, null), p2p.decodeProbe(&tampered)); // a frame claiming other than slots+1 handles is refused
    try std.testing.expectEqual(@as(?p2p.Probe, null), p2p.decodeProbe(out[0 .. out.len - 1])); // a short frame is not ours either
    var ack: [p2p.ack_len]u8 = undefined;
    try p2p.encodeAck(.{ .p2p = true, .roce = false }, &ack);
    const got_ack = p2p.decodeAck(&ack).?;
    try std.testing.expect(got_ack.p2p and !got_ack.roce);
    try std.testing.expectEqual(@as(?p2p.Ack, null), p2p.decodeAck(ack[0 .. p2p.ack_len - 1])); // a short word is not the leader's
    ack[0] = 'X';
    try std.testing.expectEqual(@as(?p2p.Ack, null), p2p.decodeAck(&ack)); // a foreign word is not the leader's either
}