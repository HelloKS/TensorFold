//! Length-prefixed TP2 control frames over a persistent TCP connection with bounded reads.

// Copyright 2026 BertholomusAI (Albert Lee). Apache-2.0; adapted from dsv41/link.zig.

const std = @import("std");
const c = std.c;

pub var bind_ip: [4]u8 = .{ 0, 0, 0, 0 }; // the address named at listen/connect: the rank plumbing derives NCCL_COMM_ID from it
pub var bind_port: u16 = 0;

var pending: [512]u8 = undefined; // one link per rank process: a frame the vote rejected is parked here, whole
var pending_len: usize = 0;

pub const Link = struct {
    fd: c.fd_t,

    fn socket() !c.fd_t {
        const fd = c.socket(c.AF.INET, c.SOCK.STREAM, 0);
        if (fd < 0) return error.SocketFailed;
        tune(fd);
        return fd;
    }

    // Bound socket waits; control traffic stays on the model's persistent TCP connection.
    fn tune(fd: c.fd_t) void {
        const one: c_int = 1;
        const timeout: c.timeval = .{ .sec = 120, .usec = 0 };
        _ = c.setsockopt(fd, c.IPPROTO.TCP, std.posix.TCP.NODELAY, std.mem.asBytes(&one), @sizeOf(c_int));
        _ = c.setsockopt(fd, c.SOL.SOCKET, c.SO.RCVTIMEO, std.mem.asBytes(&timeout), @sizeOf(c.timeval));
        _ = c.setsockopt(fd, c.SOL.SOCKET, c.SO.SNDTIMEO, std.mem.asBytes(&timeout), @sizeOf(c.timeval));
    }

    fn addr(ip: [4]u8, port: u16) c.sockaddr.in {
        return .{ .port = std.mem.nativeToBig(u16, port), .addr = @bitCast(ip) };
    }

    /// Rank 0: listen on `ip`:`port` and take the one peer's connection.
    pub fn listen(ip: [4]u8, port: u16) !Link {
        const fd = try socket();
        errdefer _ = c.close(fd);
        const one: c_int = 1;
        _ = c.setsockopt(fd, c.SOL.SOCKET, c.SO.REUSEADDR, std.mem.asBytes(&one), @sizeOf(c_int));
        var a = addr(ip, port);
        if (c.bind(fd, @ptrCast(&a), @sizeOf(c.sockaddr.in)) != 0) return error.BindFailed;
        bind_ip = ip;
        bind_port = port;
        if (c.listen(fd, 1) != 0) return error.ListenFailed;
        const peer = c.accept(fd, null, null);
        if (peer < 0) return error.AcceptFailed;
        _ = c.close(fd);
        tune(peer);
        return .{ .fd = peer };
    }

    /// Rank 1: connect to rank 0, retrying for up to `seconds` while it starts.
    pub fn connect(io: std.Io, ip: [4]u8, port: u16, seconds: u32) !Link {
        var tries: u32 = 0;
        while (true) : (tries += 1) {
            const fd = try socket();
            var a = addr(ip, port);
            if (c.connect(fd, @ptrCast(&a), @sizeOf(c.sockaddr.in)) == 0) {
                bind_ip = ip;
                bind_port = port;
                return .{ .fd = fd };
            }
            _ = c.close(fd);
            if (tries >= seconds * 4) return error.ConnectFailed;
            std.Io.sleep(io, .fromMilliseconds(250), .awake) catch {};
        }
    }

    pub fn close(l: *Link) void {
        _ = c.close(l.fd);
        l.* = undefined;
    }
    pub fn shutdown(l: Link) void {
        _ = c.shutdown(l.fd, c.SHUT.RDWR);
    }

    fn writeAll(l: Link, bytes: []const u8) !void {
        var done: usize = 0;
        while (done < bytes.len) {
            const n = c.send(l.fd, bytes.ptr + done, bytes.len - done, c.MSG.NOSIGNAL);
            if (n <= 0) {
                if (n < 0 and c.errno(n) == .INTR) continue;
                return error.PeerDown;
            }
            done += @intCast(n);
        }
    }

    fn readAll(l: Link, out: []u8) !void {
        var done: usize = 0;
        while (done < out.len) {
            const n = c.recv(l.fd, out.ptr + done, out.len - done, 0);
            if (n <= 0) {
                if (n < 0 and c.errno(n) == .INTR) continue;
                return error.PeerDown;
            }
            done += @intCast(n);
        }
    }

    /// One frame: its length (u32, little-endian), then its bytes.
    pub fn send(l: Link, bytes: []const u8) !void {
        var len: [4]u8 = undefined;
        std.mem.writeInt(u32, &len, @intCast(bytes.len), .little);
        try l.writeAll(&len);
        try l.writeAll(bytes);
    }

    /// The next frame into `buf` (a frame longer than `buf` is refused); its bytes.
    pub fn recv(l: Link, buf: []u8) ![]u8 {
        if (pending_len > 0) return l.takePending(buf); // a frame the vote rejected belongs to this reader, first and whole
        var len: [4]u8 = undefined;
        try l.readAll(&len);
        const n = std.mem.readInt(u32, &len, .little);
        if (n > buf.len) return error.FrameTooLong;
        try l.readAll(buf[0..n]);
        return buf[0..n];
    }

    /// Hand back a frame the reader refused (a foreign vote frame): the next recv finds it again, untouched.
    pub fn unread(l: Link, bytes: []const u8) void {
        _ = l;
        if (bytes.len > pending.len) return; // a frame larger than the parking slot is a protocol bug, never parked half
        @memcpy(pending[0..bytes.len], bytes);
        pending_len = bytes.len;
    }

    fn takePending(l: Link, buf: []u8) ![]u8 {
        _ = l;
        if (buf.len < pending_len) return error.FrameTooLong;
        const len = pending_len;
        @memcpy(buf[0..len], pending[0..len]);
        pending_len = 0;
        return buf[0..len];
    }

    /// One frame, waiting at most `ms` (only the TF_DSV4_P2P vote asks): null on timeout having consumed nothing.
    pub fn recvTimeout(l: Link, buf: []u8, ms: u32) !?[]u8 {
        if (pending_len > 0) return try l.takePending(buf);
        const t: c.timeval = .{ .sec = @intCast(ms / 1000), .usec = @intCast((ms % 1000) * 1000) };
        _ = c.setsockopt(l.fd, c.SOL.SOCKET, c.SO.RCVTIMEO, std.mem.asBytes(&t), @sizeOf(c.timeval));
        defer tune(l.fd); // the socket keeps its ordinary 120 s deadline for every other reader
        var len: [4]u8 = undefined;
        var done: usize = 0;
        while (done < 4) {
            const n = c.recv(l.fd, len[done..].ptr, 4 - done, 0);
            if (n > 0) {
                done += @intCast(n);
                continue;
            }
            if (n < 0 and c.errno(n) == .INTR) continue;
            if (n < 0 and c.errno(n) == .AGAIN and done == 0) return null; // the peer never spoke: the stream stays whole
            return error.PeerDown; // a peer that died mid-frame is down, bounded by ms and loud about it
        }
        const n = std.mem.readInt(u32, &len, .little);
        if (n > buf.len) return error.FrameTooLong;
        try l.readAll(buf[0..n]); // the peer committed this frame; a stall now is a dead peer, still bounded by ms
        return buf[0..n];
    }

    // Idle workers may wait for a new request; partial command frames still have the socket deadline.
    pub fn recvIdle(l: Link, buf: []u8) ![]u8 {
        if (pending_len > 0) return l.takePending(buf); // the vote's rejected frame reaches its real reader before any command
        var len: [4]u8 = undefined;
        while (true) {
            const n = c.recv(l.fd, &len, 1, 0);
            if (n == 1) break;
            if (n == 0) return error.PeerDown;
            switch (c.errno(n)) {
                .AGAIN, .INTR => continue,
                else => return error.PeerDown,
            }
        }
        try l.readAll(len[1..]);
        const n = std.mem.readInt(u32, &len, .little);
        if (n > buf.len) return error.FrameTooLong;
        try l.readAll(buf[0..n]);
        return buf[0..n];
    }
};

/// "192.0.2.7" as four bytes.
pub fn parseIp(text: []const u8) ![4]u8 {
    var out: [4]u8 = undefined;
    var it = std.mem.splitScalar(u8, text, '.');
    for (&out) |*b| b.* = try std.fmt.parseInt(u8, it.next() orelse return error.BadAddress, 10);
    if (it.next() != null) return error.BadAddress;
    return out;
}

test "addresses parse as four bytes" {
    try std.testing.expectEqual([4]u8{ 192, 0, 2, 7 }, try parseIp("192.0.2.7"));
    try std.testing.expectError(error.BadAddress, parseIp("192.0.2"));
}

test "a parked control frame retains its length and survives a short receive buffer" {
    const link: Link = .{ .fd = undefined };
    defer pending_len = 0;
    link.unread("ready-v1");
    var short: [4]u8 = undefined;
    try std.testing.expectError(error.FrameTooLong, link.takePending(&short));
    var out: [32]u8 = undefined;
    try std.testing.expectEqualStrings("ready-v1", try link.takePending(&out));
    try std.testing.expectEqual(@as(usize, 0), pending_len);
}
