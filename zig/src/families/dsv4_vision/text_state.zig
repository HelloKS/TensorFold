//! Per-sequence compressed KV and compressor state, shared by serial prefill and decode steps.
const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const D = cuda.DeviceBuffer;
const plan = @import("state_plan.zig");
pub const Snapshot = struct {
    data: D,
    at: u32,
    undo: bool,
    pub fn capture(s: *Sequence, undo: bool, stream: cuda.Stream) !Snapshot {
        const rs = try plan.regions(s.a, s.c, s.len, undo);
        defer s.a.free(rs);
        var size: usize = 0;
        for (rs) |r| size += r.bytes;
        var data = try D.alloc(s.taps.d, size);
        errdefer { stream.synchronize() catch {}; data.free(); }
        var at: usize = 0;
        for (rs) |r| {
            try data.copyFrom(at, s.part(r.part).ptr + r.offset, r.bytes, stream.handle);
            at += r.bytes;
        }
        try stream.synchronize();
        return .{ .data = data, .at = s.len, .undo = undo };
    }
    pub fn restore(image: Snapshot, s: *Sequence, stream: cuda.Stream) !void {
        const rs = try plan.regions(s.a, s.c, image.at, image.undo);
        defer s.a.free(rs);
        var at: usize = 0;
        for (rs) |r| {
            try s.part(r.part).copyFrom(r.offset, image.data.ptr + at, r.bytes, stream.handle);
            at += r.bytes;
        }
        if (at != image.data.len) return error.SnapshotShape;
        try stream.synchronize();
        s.len = image.at;
    }
    pub fn free(image: *Snapshot) void { image.data.free(); }
};
pub const Compressor = struct {
    kv: D,
    scores: D,
    cache: D,
    pub fn init(d: *const cuda.Driver, context: usize, ratio: usize, dim: usize, external: ?D) !Compressor {
        const channels = dim * (if (ratio == 4) @as(usize, 2) else 1);
        var kv = try D.alloc(d, ratio * channels * (channels / dim) * 4);
        errdefer kv.free();
        var scores = try D.alloc(d, kv.len);
        errdefer scores.free();
        try kv.fill8(0, null);
        try scores.fill32(@bitCast(-std.math.inf(f32)), null);
        return .{ .kv = kv, .scores = scores, .cache = external orelse try D.alloc(d, (context / ratio) * dim * 2) };
    }
    pub fn free(c: *Compressor) void {
        c.kv.free();
        c.scores.free();
        c.cache.free();
    }
};
pub const Layer = struct {
    kv: D,
    compressor: ?Compressor = null,
    indexer: ?Compressor = null,
    pub fn init(d: *const cuda.Driver, c: Config, context: usize, ratio: usize) !Layer {
        const bytes = (c.sliding_window + c.vision_max_n_token) * c.head_dim * 2;
        var l: Layer = .{ .kv = try D.alloc(d, bytes + (if (ratio > 0) context / ratio * c.head_dim * 2 else 0)) };
        errdefer l.free();
        if (ratio > 0) {
            l.compressor = try Compressor.init(d, context, ratio, c.head_dim, .{ .d = d, .ptr = l.kv.ptr + bytes, .len = l.kv.len - bytes, .borrowed = true });
        }
        if (ratio == 4) l.indexer = try Compressor.init(d, context, ratio, c.index_head_dim, null);
        return l;
    }
    pub fn free(l: *Layer) void {
        l.kv.free();
        if (l.compressor) |*x| x.free();
        if (l.indexer) |*x| x.free();
    }
};
pub const Sequence = struct {
    a: std.mem.Allocator,
    c: Config,
    layers: []Layer,
    draft: []Layer,
    taps: D,
    main: D,
    journal: []?Snapshot,
    len: u32 = 0,
    pub fn part(s: *Sequence, index: usize) D {
        var i: usize = 0;
        for (s.layers) |l| {
            if (index == i) return l.kv;
            i += 1;
            if (l.compressor) |cp| {
                if (index == i) return cp.kv;
                if (index == i + 1) return cp.scores;
                i += 2;
            }
            if (l.indexer) |cp| {
                if (index == i) return cp.kv;
                if (index == i + 1) return cp.scores;
                if (index == i + 2) return cp.cache;
                i += 3;
            }
        }
        if (index < i + s.draft.len) return s.draft[index - i].kv;
        return if (index == i + s.draft.len) s.taps else s.main;
    }
    pub fn before(s: *Sequence, stream: cuda.Stream) !void {
        const i = s.len % s.journal.len;
        if (s.journal[i]) |*old| old.free();
        s.journal[i] = null;
        s.journal[i] = try Snapshot.capture(s, true, stream);
    }
    pub fn keep(s: *Sequence, len: u32, stream: cuda.Stream) !void {
        if (len > s.len or s.len - len > s.journal.len) return error.UnsupportedRollback;
        while (s.len > len) {
            const row = s.len - 1;
            const saved = s.journal[row % s.journal.len] orelse return error.MissingRollback;
            if (saved.at != row) return error.MissingRollback;
            try saved.restore(s, stream);
        }
    }
    pub fn reset(s: *Sequence, stream: cuda.Stream) !void {
        s.len = 0;
        for (s.journal) |*slot| {
            if (slot.*) |*old| old.free();
            slot.* = null;
        }
        for (s.layers) |l| {
            try l.kv.fill8(0, stream.handle);
            if (l.compressor) |cp| {
                try cp.kv.fill8(0, stream.handle);
                try cp.scores.fill32(@bitCast(-std.math.inf(f32)), stream.handle);
            }
            if (l.indexer) |cp| {
                try cp.kv.fill8(0, stream.handle);
                try cp.scores.fill32(@bitCast(-std.math.inf(f32)), stream.handle);
                try cp.cache.fill8(0, stream.handle);
            }
        }
        for (s.draft) |l| try l.kv.fill8(0, stream.handle);
        try s.taps.fill8(0, stream.handle);
        try s.main.fill8(0, stream.handle);
        try stream.synchronize();
    }
    pub fn init(a: std.mem.Allocator, d: *const cuda.Driver, c: Config, context: usize) !Sequence {
        const layers = try a.alloc(Layer, c.num_hidden_layers);
        errdefer a.free(layers);
        var n: usize = 0;
        errdefer for (layers[0..n]) |*l| l.free();
        for (layers, 0..) |*l, i| {
            l.* = try Layer.init(d, c, context, c.compress_ratios[i]);
            n += 1;
        }
        const draft = try a.alloc(Layer, c.num_nextn_predict_layers);
        errdefer a.free(draft);
        var made: usize = 0;
        errdefer for (draft[0..made]) |*l| l.free();
        for (draft) |*l| { l.* = try Layer.init(d, c, context, 0); made += 1; }
        var taps = try D.alloc(d, c.hidden_size * c.dspark_target_layer_ids.len * 2);
        errdefer taps.free();
        var main = try D.alloc(d, c.hidden_size * 2);
        errdefer main.free();
        const journal = try a.alloc(?Snapshot, c.dspark_block_size + 1);
        @memset(journal, null);
        return .{ .a = a, .c = c, .layers = layers, .draft = draft, .taps = taps, .main = main, .journal = journal };
    }
    pub fn free(s: *Sequence) void {
        for (s.layers) |*l| l.free();
        s.a.free(s.layers);
        for (s.draft) |*l| l.free();
        s.a.free(s.draft);
        for (s.journal) |*slot| if (slot.*) |*image| image.free();
        s.a.free(s.journal);
        s.taps.free();
        s.main.free();
    }
};
