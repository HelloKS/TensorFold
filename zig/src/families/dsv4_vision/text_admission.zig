//! Conservative TP2 admission includes the rank payload, four sequences, conversion/staging and shared host memory.
const std = @import("std");
const cuda = @import("cuda");
pub fn check(a: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, c: @import("config.zig").Config, cat: *const @import("catalog.zig").Catalog, context: usize, rank: u32, kv8: bool) !void {
    var payload: u64 = 0;
    for (cat.headers.values()) |t| payload += t.end - t.begin;
    const plan = @import("state_plan.zig");
    const state = try @import("parallel.zig").Memory.calculate(c, context, 4, kv8); // e4m3 rows are narrower, and admission books what the engine actually makes
    const rollback = 4 * (c.dspark_block_size + 1) * try plan.bytes(c, 0, true, kv8);
    const image_rows = 4 * 4 * c.vision_max_n_token * c.hidden_size * 2;
    const sequences = state.total() + rollback + image_rows + (c.num_hidden_layers + c.num_nextn_predict_layers) * c.vision_max_n_token * plan.rowBytes(c.head_dim, kv8) * 4;
    const scratch_stage: u64 = 2 << 30;
    const rd = @import("rounds.zig");
    const nt = @import("native.zig");
    const round_bytes: u64 = if (rd.modeFrom(nt.env("TF_DSV4_BATCH"), nt.env("TF_DSV4_SERIAL")) == .batched)
        rd.Rounds.bytes(try rd.capFromEnv(nt.env("TF_DSV4_ROWS_CAP")), c.vocab_size)
    else 0;
    const need = payload + sequences + scratch_stage + round_bytes;
    const pool = try inspect(a, io, ctx);
    const granted = pool.room;
    const reserve = pool.reserve;
    const available = pool.available;
    std.log.info("rank {d} admission: payload {d}, 4 sequence states {d}, conversion/scratch/staging {d}, batched rounds {d}, reserve {d}, available {d} bytes", .{ rank, payload, sequences, scratch_stage, round_bytes, reserve, available });
    if (need > granted) return error.MemoryBudgetExceeded;
}
const Pool = struct { available: u64, reserve: u64, room: u64 };
const Counts = struct { total: u64, available: u64 };
pub fn room(a: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context) !u64 { return (try inspect(a, io, ctx)).room; }
fn inspect(a: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context) !Pool {
    const card = try ctx.memInfo();
    const unified = (try ctx.attribute(.integrated)) != 0;
    var host_text: ?[]u8 = null;
    defer if (host_text) |text| a.free(text);
    if (unified) {
        var file = try std.Io.Dir.cwd().openFile(io, "/proc/meminfo", .{});
        defer file.close(io);
        var bytes: [4096]u8 = undefined;
        var reader = file.readerStreaming(io, &bytes);
        host_text = try reader.interface.allocRemaining(a, .limited(1 << 20));
    }
    const counts = try memoryCounts(.{ .total = card.total, .available = card.free }, unified, host_text);
    const env = @import("native.zig").env;
    const used = cuda.usage(false);
    return budget(counts, env("TENSORFOLD_MEMORY_RESERVE_GIB"), env("TENSORFOLD_CUDA_MEMORY_LIMIT_GB"), used.device, used.host);
}

fn memoryCounts(card: Counts, unified: bool, host_text: ?[]const u8) !Counts {
    var available = card.available;
    var total = card.total;
    if (unified) {
        var lines = std.mem.tokenizeScalar(u8, host_text orelse return error.HostMemoryUnavailable, '\n');
        var host_free: ?u64 = null;
        var host_total: ?u64 = null;
        while (lines.next()) |line| {
            var words = std.mem.tokenizeAny(u8, line, ": \t");
            const name = words.next() orelse continue;
            if (!std.mem.eql(u8, name, "MemAvailable") and !std.mem.eql(u8, name, "MemTotal")) continue;
            const n = std.fmt.parseInt(u64, words.next() orelse return error.HostMemoryUnavailable, 10) catch
                return error.HostMemoryUnavailable;
            if (!std.mem.eql(u8, words.next() orelse return error.HostMemoryUnavailable, "kB") or
                words.next() != null) return error.HostMemoryUnavailable;
            const bytes = std.math.mul(u64, n, 1024) catch return error.HostMemoryUnavailable;
            if (std.mem.eql(u8, name, "MemAvailable")) {
                if (host_free != null) return error.HostMemoryUnavailable;
                host_free = bytes;
            } else {
                if (host_total != null) return error.HostMemoryUnavailable;
                host_total = bytes;
            }
        }
        // On integrated CUDA devices, MemAvailable includes reclaimable file cache that CUDA free may exclude.
        available = host_free orelse return error.HostMemoryUnavailable;
        total = host_total orelse return error.HostMemoryUnavailable;
        if (total == 0 or host_free.? > total) return error.HostMemoryUnavailable;
    }
    return .{ .total = total, .available = available };
}

fn budget(counts: Counts, reserve_text: ?[]const u8, limit_text: ?[]const u8, device: u64, host: u64) !Pool {
    const available = counts.available;
    const reserve = if (reserve_text) |s| try gib(s) else @max(@as(u64, 4 << 30), counts.total / 10);
    var granted = available -| reserve;
    if (limit_text) |s| {
        granted = @min(granted, (try gib(s)) -| device -| host);
    }
    return .{ .available = available, .reserve = reserve, .room = granted };
}

test "GB10 admission counts reclaimable file cache instead of clamping to CUDA free" {
    const card: Counts = .{ .total = 128 << 30, .available = 3371315200 };
    const text = "MemTotal: 127535248 kB\nMemFree: 3683400 kB\nMemAvailable: 109608980 kB\n";
    const counts = try memoryCounts(card, true, text);
    try std.testing.expectEqual(@as(u64, 109608980) * 1024, counts.available);
    const pool = try budget(counts, null, null, 0, 0);
    const need: u64 = 84609416440 + 3808047424 + 2147483648 + 31027776;
    try std.testing.expect(need <= pool.room);
    try std.testing.expectEqual(counts.total / 10, pool.reserve);
    try std.testing.expectEqual(card, try memoryCounts(card, false, text));
    try std.testing.expect(need > (try budget(card, null, null, 0, 0)).room);
}

test "unified admission preserves reserves and memory limits and rejects invalid host counts" {
    const g: u64 = 1 << 30;
    const counts: Counts = .{ .total = 128 * g, .available = 100 * g };
    try std.testing.expectEqual(80 * g, (try budget(counts, "20", null, 0, 0)).room);
    try std.testing.expectEqual(57 * g, (try budget(counts, null, "60", g, 2 * g)).room);
    try std.testing.expectEqual(@as(u64, 0), (try budget(counts, "120", null, 0, 0)).room);
    try std.testing.expectEqual(@as(u64, 0), (try budget(counts, null, "2", 3 * g, 0)).room);
    try std.testing.expectError(error.InvalidMemoryBudget, budget(counts, "nan", null, 0, 0));
    for ([_]?[]const u8{
        null, "", "MemTotal: 1024 kB\n", "MemTotal: bad kB\nMemAvailable: 1 kB\n",
        "MemTotal: 1024 MB\nMemAvailable: 1 kB\n", "MemTotal: 0 kB\nMemAvailable: 0 kB\n",
        "MemTotal: 1 kB\nMemAvailable: 2 kB\n", "MemTotal: 18446744073709551615 kB\nMemAvailable: 1 kB\n",
        "MemTotal: 2 kB\nMemTotal: 2 kB\nMemAvailable: 1 kB\n",
    }) |text| try std.testing.expectError(error.HostMemoryUnavailable, memoryCounts(counts, true, text));
    try std.testing.expectEqual(counts, try memoryCounts(counts, false, null));
}

fn gib(s: []const u8) !u64 {
    const v = try std.fmt.parseFloat(f64, s);
    if (!std.math.isFinite(v) or v < 2 or v >= 0x1p34) return error.InvalidMemoryBudget;
    return @intFromFloat(v * 1073741824);
}
