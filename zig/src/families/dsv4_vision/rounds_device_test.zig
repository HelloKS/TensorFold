//! Exercise the production round sink with host-backed device buffers and rank-distinct logits.
const std = @import("std");
const rounds = @import("rounds.zig");
const rowtab = @import("rowtab.zig");
const t = std.testing;

const Buffer = struct {
    ptr: u64,
    len: usize,
    fn of(data: []u8) Buffer { return .{ .ptr = @intFromPtr(data.ptr), .len = data.len }; }
    fn bytes(b: Buffer) []u8 { return @as([*]u8, @ptrFromInt(b.ptr))[0..b.len]; }
    pub fn upload(b: Buffer, at: usize, src: []const u8) !void {
        if (at > b.len or src.len > b.len - at) return error.BufferOverrun;
        @memcpy(b.bytes()[at..][0..src.len], src);
    }
    pub fn download(b: Buffer, at: usize, dst: []u8) !void {
        if (at > b.len or dst.len > b.len - at) return error.BufferOverrun;
        @memcpy(dst, b.bytes()[at..][0..dst.len]);
    }
    pub fn copyFrom(b: Buffer, at: usize, src: u64, n: usize, _: ?*anyopaque) !void {
        try b.upload(at, @as([*]const u8, @ptrFromInt(src))[0..n]);
    }
};
const Ops = struct {
    pub fn call(_: Ops, comptime name: []const u8, _: anytype, blocks: u32, _: u32, v: anytype) !void {
        if (comptime std.mem.eql(u8, name, "ids")) {
            const table: [*]const i32 = @ptrFromInt(v[0]);
            const ids: [*]u32 = @ptrFromInt(v[1]);
            for (0..v[2]) |i| ids[i] = @bitCast(table[i]);
        } else if (comptime std.mem.eql(u8, name, "rows")) {
            const gathered: [*]const f32 = @ptrFromInt(v[0]);
            const sampled: [*]const i32 = @ptrFromInt(v[1]);
            const out: [*]f32 = @ptrFromInt(v[2]);
            for (0..blocks) |k| {
                const at: usize = @intCast(sampled[k]);
                if (at >= v[3]) return error.BadSampledRow;
                for (0..v[5]) |j| out[k * v[5] + j] = if (j < v[4])
                    gathered[at * v[4] + j] else gathered[(v[3] + at) * v[4] + j - v[4]];
            }
        } else return error.UnexpectedKernel;
    }
};
const Comm = struct {
    pub fn allGather(_: Comm, send: u64, recv: u64, count: usize, _: anytype, _: anytype) !void {
        const src: [*]const f32 = @ptrFromInt(send);
        const dst: [*]f32 = @ptrFromInt(recv);
        for (0..count) |i| { dst[i] = src[i]; dst[count + i] = 100 + src[i]; }
    }
};
const Scratch = struct {
    logits: Buffer,
    pub fn buf(s: Scratch, _: anytype) Buffer { return s.logits; }
    pub fn p(s: Scratch, _: anytype) u64 { return s.logits.ptr; }
};
const Batch = struct {
    rows_max: u32,
    vh: u32,
    host: []i32,
    table: Buffer,
    ids: Buffer,
    exch: Buffer,
    recv: Buffer,
    out: Buffer,
    ops: Ops = .{},
};
const Model = struct {
    rb: ?*Batch,
    r: Scratch,
    stream: struct { handle: ?*anyopaque = null } = .{},
    comm: Comm = .{},
    pub fn sync(_: *Model) !void {}
};

test "production batch staging and exchange retain each sampled row in rank order" {
    const cap = 8;
    var host: [5 * cap]i32 = @splat(-1);
    var table: [5 * cap]i32 = @splat(-1);
    var ids: [cap]u32 = @splat(0);
    var exch: [cap * 2]f32 = @splat(-999);
    var recv: [cap * 4]f32 = @splat(-999);
    var out: [cap * 4]f32 = @splat(-999);
    var scratch: [2]f32 = .{ 1, 2 };
    var rb: Batch = .{ .rows_max = cap, .vh = 2, .host = &host,
        .table = Buffer.of(std.mem.asBytes(&table)), .ids = Buffer.of(std.mem.asBytes(&ids)),
        .exch = Buffer.of(std.mem.asBytes(&exch)), .recv = Buffer.of(std.mem.asBytes(&recv)),
        .out = Buffer.of(std.mem.asBytes(&out)) };
    var m: Model = .{ .rb = &rb, .r = .{ .logits = Buffer.of(std.mem.asBytes(&scratch)) } };
    var p: rounds.Plan = .{ .n = 3, .R = 4, .n_sampled = 2 };
    p.rows[0] = .{ .slot = 0, .token = 10, .index = 1, .sample = true };
    p.rows[1] = .{ .slot = 0, .token = 11, .index = 2, .sample = false };
    p.rows[2] = .{ .slot = 1, .token = 12, .index = 7, .sample = true };
    try rounds.Device.stage(&m, &p);
    try t.expectEqualSlices(i32, &.{ 0, 2 }, table[rowtab.ncol * cap ..][0..2]);
    try t.expectEqualSlices(u32, &.{ 10, 11, 12, 12 }, ids[0..4]);
    try rounds.Device.headRow(&m, &p, 0);
    scratch = .{ 3, 4 };
    try rounds.Device.headRow(&m, &p, 2);
    var actual: [8]f32 = undefined;
    try rounds.Device.round(&m, &p, &actual);
    try t.expectEqualSlices(f32, &.{ 1, 2, 101, 102, 3, 4, 103, 104 }, &actual);
    try t.expectEqualSlices(f32, &.{ 3, 4 }, &scratch);
}
