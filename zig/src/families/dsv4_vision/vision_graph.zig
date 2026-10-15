//! ViT and aligner forward graph shared by fused native and split EXL3 projections.
const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const gpu = @import("gpu_weights.zig");
const Catalog = @import("catalog.zig").Catalog;
const Device = cuda.DeviceBuffer;

pub const Observer = struct {
    ptr: *anyopaque,
    after: *const fn (*anyopaque, u32, Device) anyerror!void,
};

pub const Runtime = struct {
    exl3: ?cuda.dsv4_exl3.Projection,
    native: ?cuda.dsv4_native.Ops,
    tensor: cuda.dsv4_tensor.Ops,
    model: cuda.dsv4_ops.Ops,
    vision: cuda.dsv4_vision.Ops,

    pub fn open(a: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, c: Config, dir: ?[]const u8) !Runtime {
        const load = @import("gpu_modules.zig").load;
        var exl3: ?cuda.dsv4_exl3.Projection = if (c.storage == .exl3) try load(cuda.dsv4_exl3.Projection, a, io, d, dir, "exl3.fatbin") else null;
        errdefer if (exl3) |*o| o.unload();
        var native: ?cuda.dsv4_native.Ops = if (c.storage == .native_fp4_fp8) try load(cuda.dsv4_native.Ops, a, io, d, dir, "dsv4_native.fatbin") else null;
        errdefer if (native) |*o| o.unload();
        var tensor = try load(cuda.dsv4_tensor.Ops, a, io, d, dir, "dsv4_tensor.fatbin");
        errdefer tensor.unload();
        var model = try load(cuda.dsv4_ops.Ops, a, io, d, dir, "dsv4_ops.fatbin");
        errdefer model.unload();
        const vision = try load(cuda.dsv4_vision.Ops, a, io, d, dir, "dsv4_vision.fatbin");
        return .{ .exl3 = exl3, .native = native, .tensor = tensor, .model = model, .vision = vision };
    }

    pub fn close(r: *Runtime) void {
        if (r.exl3) |*o| o.unload();
        if (r.native) |*o| o.unload();
        r.tensor.unload();
        r.model.unload();
        r.vision.unload();
    }

    fn project(r: Runtime, m: *gpu.Matrix, s: cuda.Stream, scratch: *gpu.Scratch, x: u64, y: u64, rows: u32) !void {
        try m.run(r.exl3, r.native, r.tensor, s, scratch, x, y, rows, .bf16, .bf16);
    }
};

const Split = struct {
    matrices: [3]?gpu.Matrix = @splat(null),
    parts: u32,
    width: u32,
    fused: bool,

    fn load(catalog: *const Catalog, a: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, c: Config, prefix: []const u8, fused: []const u8, split: []const []const u8, width: usize) !Split {
        var out: Split = .{ .parts = @intCast(split.len), .width = @intCast(width), .fused = c.storage == .native_fp4_fp8 };
        errdefer out.free();
        if (out.fused) {
            const key = try std.fmt.allocPrint(a, "{s}{s}", .{ prefix, fused });
            defer a.free(key);
            out.matrices[0] = try gpu.Matrix.load(catalog, a, io, d, key, c.vision_dim, split.len * width, c.storage);
        } else for (split, 0..) |suffix, i| {
            const key = try std.fmt.allocPrint(a, "{s}{s}", .{ prefix, suffix });
            defer a.free(key);
            out.matrices[i] = try gpu.Matrix.load(catalog, a, io, d, key, c.vision_dim, width, c.storage);
        }
        return out;
    }

    fn free(m: *Split) void {
        for (&m.matrices) |*slot| if (slot.*) |*matrix| matrix.free();
    }

    fn run(m: *Split, r: Runtime, s: cuda.Stream, scratch: *gpu.Scratch, x: u64, wide: u64, out: [3]u64, rows: u32) !void {
        if (m.fused) {
            try r.project(&m.matrices[0].?, s, scratch, x, wide, rows);
            try r.vision.split(s, wide, out[0], out[1], out[2], rows, m.width, m.parts);
        } else for (m.matrices[0..m.parts], 0..) |*slot, i| try r.project(&slot.*.?, s, scratch, x, out[i], rows);
    }
};

const Block = struct {
    norm1: gpu.Raw,
    norm2: gpu.Raw,
    qkv: Split,
    first: Split,
    wo: gpu.Matrix,
    w2: gpu.Matrix,

    fn load(ck: *const Catalog, a: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, c: Config, index: usize) !Block {
        const p = try std.fmt.allocPrint(a, "vision.blocks.{d}", .{index});
        defer a.free(p);
        const n1 = try std.fmt.allocPrint(a, "{s}.norm1.weight", .{p});
        defer a.free(n1);
        const n2 = try std.fmt.allocPrint(a, "{s}.norm2.weight", .{p});
        defer a.free(n2);
        var norm1 = try gpu.Raw.load(ck, a, io, d, n1);
        errdefer norm1.free();
        var norm2 = try gpu.Raw.load(ck, a, io, d, n2);
        errdefer norm2.free();
        var qkv = try Split.load(ck, a, io, d, c, p, ".attn.wqkv", &.{ ".attn.q_proj", ".attn.k_proj", ".attn.v_proj" }, c.vision_dim);
        errdefer qkv.free();
        var first = try Split.load(ck, a, io, d, c, p, ".mlp.w1", &.{ ".mlp.gate_proj", ".mlp.up_proj" }, c.vision_inter_dim);
        errdefer first.free();
        const wo_name = try std.fmt.allocPrint(a, "{s}.attn.wo", .{p});
        defer a.free(wo_name);
        const w2_name = try std.fmt.allocPrint(a, "{s}.mlp.w2", .{p});
        defer a.free(w2_name);
        var wo = try gpu.Matrix.load(ck, a, io, d, wo_name, c.vision_dim, c.vision_dim, c.storage);
        errdefer wo.free();
        const w2 = try gpu.Matrix.load(ck, a, io, d, w2_name, c.vision_inter_dim, c.vision_dim, c.storage);
        return .{ .norm1 = norm1, .norm2 = norm2, .qkv = qkv, .first = first, .wo = wo, .w2 = w2 };
    }

    fn free(b: *Block) void {
        b.norm1.free();
        b.norm2.free();
        b.qkv.free();
        b.first.free();
        b.wo.free();
        b.w2.free();
    }
};

pub const Tower = struct {
    c: Config,
    patch: gpu.Matrix,
    blocks: []Block,
    norm: gpu.Raw,
    align1: gpu.Matrix,
    align2: gpu.Matrix,

    pub fn load(ck: *const Catalog, a: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, c: Config) !Tower {
        var selected: std.ArrayList(@import("weights.zig").Spec) = .empty;
        defer selected.deinit(a);
        var arena: std.heap.ArenaAllocator = .init(a);
        defer arena.deinit();
        for (try @import("weights.zig").requirements(arena.allocator(), c)) |spec| {
            if (std.mem.startsWith(u8, spec.key, "vision.") or std.mem.startsWith(u8, spec.key, "aligner."))
                try selected.append(a, spec);
        }
        var problem: []const u8 = "";
        _ = @import("weights.zig").validateSpecs(arena.allocator(), c, selected.items, ck.headers, &problem) catch |e| {
            std.log.err("vision checkpoint: {t}: {s}", .{ e, problem });
            return e;
        };
        var patch = try gpu.Matrix.load(ck, a, io, d, "vision.patch_embed.proj", 3 * c.vision_patch_size * c.vision_patch_size, c.vision_dim, c.storage);
        errdefer patch.free();
        const blocks = try a.alloc(Block, c.vision_n_layers);
        var loaded: usize = 0;
        errdefer {
            for (blocks[0..loaded]) |*b| b.free();
            a.free(blocks);
        }
        for (blocks, 0..) |*b, index| {
            b.* = try Block.load(ck, a, io, d, c, index);
            loaded += 1;
        }
        var norm = try gpu.Raw.load(ck, a, io, d, "vision.norm.weight");
        errdefer norm.free();
        var align1 = try gpu.Matrix.load(ck, a, io, d, "aligner.w1", c.vision_dim * 9, c.hidden_size, c.storage);
        errdefer align1.free();
        const align2 = try gpu.Matrix.load(ck, a, io, d, "aligner.w2", c.hidden_size, c.hidden_size, c.storage);
        return .{ .c = c, .patch = patch, .blocks = blocks, .norm = norm, .align1 = align1, .align2 = align2 };
    }

    pub fn free(t: *Tower, a: std.mem.Allocator) void {
        t.patch.free();
        for (t.blocks) |*b| b.free();
        a.free(t.blocks);
        t.norm.free();
        t.align1.free();
        t.align2.free();
    }

    /// Returns raster-ordered aligned embeddings; N-layout and special tokens are applied at their prompt position.
    pub fn run(t: *Tower, r: Runtime, s: cuda.Stream, patches: u64, height: u32, width: u32) !Device {
        return t.runObserved(r, s, patches, height, width, null);
    }

    pub fn runObserved(t: *Tower, r: Runtime, s: cuda.Stream, patches: u64, height: u32, width: u32, observer: ?Observer) !Device {
        if (height == 0 or width == 0 or height > 4096 or width > 4096 or height * width > 4096) return error.InvalidImageShape;
        const rows = height * width;
        const cells = ((height + 2) / 3) * ((width + 2) / 3);
        if (cells > t.c.vision_max_n_token) return error.InvalidImageShape;
        const dim: u32 = @intCast(t.c.vision_dim);
        const inter: u32 = @intCast(t.c.vision_inter_dim);
        const hidden: u32 = @intCast(t.c.hidden_size);
        const heads: u32 = @intCast(t.c.vision_n_heads);
        const d = t.patch.data.d;
        const sizes = [_]usize{ rows * dim, rows * dim, rows * dim, rows * dim, rows * dim, rows * dim, rows * @max(3 * dim, 2 * inter), rows * inter, rows * inter, cells * 9 * dim, cells * hidden };
        var buffers: [sizes.len]Device = undefined;
        var allocated: usize = 0;
        defer for (buffers[0..allocated]) |*b| b.free();
        for (&buffers, sizes) |*b, size| {
            b.* = try Device.alloc(d, size * 2);
            allocated += 1;
        }
        errdefer s.synchronize() catch {};
        var scratch: gpu.Scratch = .{};
        defer scratch.free();
        defer s.synchronize() catch {};
        const x = buffers[0].ptr;
        const normalized = buffers[1].ptr;
        const update = buffers[2].ptr;
        const q = buffers[3].ptr;
        const k = buffers[4].ptr;
        const v = buffers[5].ptr;
        const wide = buffers[6].ptr;
        const gate = buffers[7].ptr;
        const up = buffers[8].ptr;
        const unfolded = buffers[9].ptr;
        const aligned = buffers[10].ptr;
        try r.project(&t.patch, s, &scratch, patches, x, rows);
        try observe(observer, s, 0, buffers[0]);
        for (t.blocks, 0..) |*b, index| {
            try r.vision.norm(s, x, b.norm1.data.ptr, normalized, rows, dim, try gpu.kind(b.norm1.dtype));
            try b.qkv.run(r, s, &scratch, normalized, wide, .{ q, k, v }, rows);
            inline for (.{ q, k }) |ptr| try r.tensor.run(.vision_rope, s, .{ .x = rows, .y = heads }, 128, .{ ptr, width, heads, dim / heads, @as(f32, @floatCast(t.c.vision_rope_theta)) });
            try r.vision.attention(s, q, k, v, normalized, rows, heads, dim / heads);
            try r.project(&b.wo, s, &scratch, normalized, update, rows);
            try r.tensor.run(.add, s, .{ .x = (rows * dim + 127) / 128 }, 128, .{ x, update, rows * dim });
            try r.vision.norm(s, x, b.norm2.data.ptr, normalized, rows, dim, try gpu.kind(b.norm2.dtype));
            try b.first.run(r, s, &scratch, normalized, wide, .{ gate, up, 0 }, rows);
            try r.vision.swiglu(s, gate, up, gate, rows, inter);
            try r.project(&b.w2, s, &scratch, gate, update, rows);
            try r.tensor.run(.add, s, .{ .x = (rows * dim + 127) / 128 }, 128, .{ x, update, rows * dim });
            try observe(observer, s, @intCast(index + 1), buffers[0]);
        }
        try r.vision.norm(s, x, t.norm.data.ptr, normalized, rows, dim, try gpu.kind(t.norm.dtype));
        try r.model.unfold(s, normalized, unfolded, height, width, dim);
        try r.project(&t.align1, s, &scratch, unfolded, aligned, cells);
        try r.tensor.run(.gelu, s, .{ .x = (cells * hidden + 127) / 128 }, 128, .{ aligned, cells * hidden });
        var out = try Device.alloc(d, @as(usize, cells) * hidden * 2);
        errdefer out.free();
        try r.project(&t.align2, s, &scratch, aligned, out.ptr, cells);
        try s.synchronize();
        return out;
    }
};

fn observe(observer: ?Observer, s: cuda.Stream, index: u32, state: Device) !void {
    const o = observer orelse return;
    try s.synchronize();
    try o.after(o.ptr, index, state);
}
