//! Offline checkpoint inspection, separate from serving qualification.
const std = @import("std");
const core = @import("core");
const config = @import("config.zig");
const weights = @import("weights.zig");
const catalog = @import("catalog.zig");
const payloads = @import("payloads.zig");
const parallel = @import("parallel.zig");

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 2) {
        std.debug.print("usage: tf-dsv4-inspect MODEL_DIR [--catalog HEADERS.json] [--plan]\n", .{});
        return 2;
    }
    var catalog_path: ?[]const u8 = null;
    var plan = false;
    var i: usize = 2;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--plan")) plan = true else if (std.mem.eql(u8, args[i], "--catalog")) {
            i += 1;
            if (i == args.len or catalog_path != null) return 2;
            catalog_path = args[i];
        } else return 2;
    }
    const path = try std.fs.path.join(a, &.{ args[1], "config.json" });
    const text = try std.Io.Dir.cwd().readFileAlloc(init.io, path, a, .limited(1 << 20));
    const c = config.Config.parse(a, text) catch |e| {
        std.debug.print("config.json: {t}\n", .{e});
        return 1;
    };
    var problem: []const u8 = "";
    const headers = if (catalog_path) |cp| h: {
        const json = try std.Io.Dir.cwd().readFileAlloc(init.io, cp, a, .limited(64 << 20));
        break :h try core.safetensors.parseHeader(a, json, std.math.maxInt(usize));
    } else (try catalog.Catalog.open(a, init.io, args[1])).headers;
    const summary = weights.validate(a, c, headers, &problem) catch |e| {
        std.debug.print("checkpoint: {t}: {s}\n", .{ e, problem });
        return 1;
    };
    if (plan) {
        const Item = struct { spec: weights.Spec, ranks: [2]parallel.Slice };
        const specs = try weights.requirements(a, c);
        const items = try a.alloc(Item, specs.len);
        for (items, specs) |*item, spec| item.* = .{ .spec = spec, .ranks = .{ try parallel.partition(c, spec, 0), try parallel.partition(c, spec, 1) } };
        const physical = try payloads.select(a, c, headers);
        const rank_bytes = [2]payloads.RankBytes{ try payloads.memory(a, physical, 0), try payloads.memory(a, physical, 1) };
        const json = try std.json.Stringify.valueAlloc(a, .{ .version = 2, .cache_code_sha256 = std.fmt.bytesToHex(@import("cache_abi.zig").digest(), .lower), .memory_qualified = false, .serving_backend_complete = false, .storage = c.storage.name(), .payloads = physical, .weight_memory = rank_bytes, .items = items, .kv_memory = try parallel.Memory.calculate(c, 262144, 4), .summary = summary }, .{});
        try std.Io.File.stdout().writeStreamingAll(init.io, json);
        return 0;
    }
    var out: std.Io.Writer.Allocating = .init(a);
    try out.writer.print("DeepSeek-V4-Flash-Vision-Exp {s}\n" ++
        "validated: {d} packed projections, {d} dense projections, {d} raw tensors, {d} FP4, {d} FP8\n" ++
        "profile: TP2, 262144 tokens/request, 4 streams\n" ++
        "This is a storage-contract check; it does not qualify inference or memory capacity.\n", .{ c.storage.name(), summary.matrices, summary.dense, summary.raw, summary.fp4, summary.fp8 });
    try std.Io.File.stdout().writeStreamingAll(init.io, out.written());
    return 0;
}
