//! Conservative TP2 admission includes the rank payload, four sequences, conversion/staging and shared host memory.
const std = @import("std");
const cuda = @import("cuda");
pub fn check(a: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, c: @import("config.zig").Config, cat: *const @import("catalog.zig").Catalog, context: usize, rank: u32) !void {
    var payload: u64 = 0;
    for (cat.headers.values()) |t| payload += t.end - t.begin;
    const state = try @import("parallel.zig").Memory.calculate(c, context, 4);
    const rollback = 4 * (c.dspark_block_size + 1) * try @import("state_plan.zig").bytes(c, 0, true);
    const image_rows = 4 * 4 * c.vision_max_n_token * c.hidden_size * 2;
    const sequences = state.total() + rollback + image_rows + (c.num_hidden_layers + c.num_nextn_predict_layers) * c.vision_max_n_token * c.head_dim * 2 * 4;
    const scratch_stage: u64 = 2 << 30;
    const need = payload + sequences + scratch_stage;
    const pool = try inspect(a, io, ctx);
    const granted = pool.room;
    const reserve = pool.reserve;
    const available = pool.available;
    std.log.info("rank {d} admission: payload {d}, 4 sequence states {d}, conversion/scratch/staging {d}, reserve {d}, available {d} bytes", .{ rank, payload, sequences, scratch_stage, reserve, available });
    if (need > granted) return error.MemoryBudgetExceeded;
}
const Pool = struct { available: u64, reserve: u64, room: u64 };
pub fn room(a: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context) !u64 { return (try inspect(a, io, ctx)).room; }
fn inspect(a: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context) !Pool {
    const card = try ctx.memInfo();
    var available: u64 = card.free;
    var total: u64 = card.total;
    const unified = (try ctx.attribute(.integrated)) != 0;
    if (unified) {
        var file = try std.Io.Dir.cwd().openFile(io, "/proc/meminfo", .{});
        defer file.close(io);
        var bytes: [4096]u8 = undefined;
        var reader = file.readerStreaming(io, &bytes);
        const data = try reader.interface.allocRemaining(a, .limited(1 << 20));
        defer a.free(data);
        var lines = std.mem.tokenizeScalar(u8, data, '\n');
        var host_free: ?u64 = null;
        var host_total: ?u64 = null;
        while (lines.next()) |line| {
            var words = std.mem.tokenizeAny(u8, line, ": \t");
            const name = words.next() orelse continue;
            if (!std.mem.eql(u8, name, "MemAvailable") and !std.mem.eql(u8, name, "MemTotal")) continue;
            const n = try std.fmt.parseInt(u64, words.next() orelse return error.HostMemoryUnavailable, 10);
            if (std.mem.eql(u8, name, "MemAvailable")) host_free = n * 1024 else host_total = n * 1024;
        }
        available = @min(available, host_free orelse return error.HostMemoryUnavailable);
        total = host_total orelse return error.HostMemoryUnavailable;
    }
    const env = @import("native.zig").env;
    const reserve = if (env("TENSORFOLD_MEMORY_RESERVE_GIB")) |s| try gib(s) else @max(@as(u64, 4 << 30), total / 10);
    var granted = available -| reserve;
    if (env("TENSORFOLD_CUDA_MEMORY_LIMIT_GB")) |s| {
        const used = cuda.usage(false);
        granted = @min(granted, (try gib(s)) -| used.device -| used.host);
    }
    return .{ .available = available, .reserve = reserve, .room = granted };
}

fn gib(s: []const u8) !u64 {
    const v = try std.fmt.parseFloat(f64, s);
    if (!std.math.isFinite(v) or v < 2 or v >= 0x1p34) return error.InvalidMemoryBudget;
    return @intFromFloat(v * 1073741824);
}
