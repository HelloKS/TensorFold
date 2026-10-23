//! Rank 1 follows the same lane commands and NCCL order as the HTTP process on rank 0.
const std = @import("std");
const cuda = @import("cuda");
const cluster = @import("cluster_backend");
pub fn main(init: std.process.Init) !u8 {
    const a = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 4) {
        std.debug.print("usage: tf-dsv4-worker MODEL CONTEXT KERNEL_DIR\n", .{});
        return 2;
    }
    var d = try cuda.Driver.open();
    defer d.close();
    var ctx = try cuda.Context.init(&d, 0);
    defer ctx.deinit();
    const native = @import("native.zig");
    const context = try std.fmt.parseInt(usize, args[2], 10);
    const m = try @import("text_model.zig").Model.open(a, init.io, &ctx, args[1], try native.cacheDir(), args[3], 1, try native.master(), try native.port(), context);
    var live = true;
    defer if (live) m.close();
    const bytes = try a.alloc(u8, @import("control.zig").limit + 64);
    defer a.free(bytes);
    var control: @import("text_model.zig").Control = .{ .link = m.link, .rank = 1, .buffer = bytes };
    try cluster.follow(a, control.view(), m.shard());
    const identity = m.identity;
    const used = cuda.usage(false);
    const before = try ctx.memInfo();
    m.close();
    live = false;
    if (native.env("TF_DSV4_WORKER_RECEIPT")) |path| {
        const receipt = try std.json.Stringify.valueAlloc(a, .{ .rank = 1, .state_identity = identity,
            .tracked = used, .after_cleanup = cuda.usage(false), .cuda_free_before_cleanup = before.free,
            .cuda_free_after_cleanup = (try ctx.memInfo()).free }, .{ .whitespace = .indent_2 });
        defer a.free(receipt);
        try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = path, .data = receipt });
    }
    return 0;
}
