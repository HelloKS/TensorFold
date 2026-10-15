//! V4 tensor operations keep checkpoint dtypes explicit at every projection boundary.
const module = @import("module.zig");
const kernels = @import("kernels.zig");
const launch = @import("launch.zig");
const Driver = @import("driver.zig").Driver;
const Stream = @import("stream.zig").Stream;

pub const Kind = enum(u32) { f32, f16, bf16 };
pub const Op = enum { dense, embed, expand, tap, collapse, add, gelu, vision_rope };

pub const Ops = struct {
    mod: module.Module,
    functions: [8]module.Function,

    pub fn load(d: *const Driver) !Ops {
        if (!kernels.available) return error.BuiltWithoutKernels;
        return loadImage(d, kernels.dsv4_tensor);
    }

    pub fn loadImage(d: *const Driver, image: []const u8) !Ops {
        var mod = try module.Module.load(d, image);
        errdefer mod.unload();
        var funcs: [8]module.Function = undefined;
        inline for (@typeInfo(Op).@"enum".field_names, 0..) |name, i|
            funcs[i] = try mod.function("tf_dsv4_" ++ name);
        return .{ .mod = mod, .functions = funcs };
    }

    pub fn unload(o: *Ops) void {
        o.mod.unload();
    }

    pub fn run(o: Ops, comptime op: Op, stream: Stream, grid: launch.Dim3, threads: u32, values: anytype) !void {
        var args: launch.Args = .{};
        inline for (values) |v| args.add(v);
        try launch.launch(o.functions[@backingInt(op)], .{ .grid = grid, .block = .{ .x = threads } }, stream, &args);
    }

    pub fn dense(o: Ops, s: Stream, x: u64, w: u64, bias: u64, y: u64, rows: u32, k: u32, n: u32, xkind: Kind, wkind: Kind, ykind: Kind, normalize: bool, eps: f32) !void {
        if (rows == 0) return;
        if (k == 0 or n == 0 or ykind == .f16 or eps < 0) return error.Invalid;
        try o.run(.dense, s, .{ .x = (n + 127) / 128, .y = rows }, 128, .{ x, w, bias, y, k, n, @as(u32, @backingInt(xkind)), @as(u32, @backingInt(wkind)), @as(u32, @backingInt(ykind)), @as(u32, @intFromBool(normalize)), eps });
    }
};
