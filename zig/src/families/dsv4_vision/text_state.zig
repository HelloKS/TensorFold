//! Per-sequence compressed KV and compressor state, shared by serial prefill and decode steps.
const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const D = cuda.DeviceBuffer;
const plan = @import("state_plan.zig");
/// The cache's slot ring, named once by the host's budget: kept states live there, not in buffers of their own.
pub const Ring = struct {
    rows: D, // the ring's bytes on the card, or a borrowed view of the host's pinned states
    taken: []bool, // which slots a kept state holds right now
    bytes: usize, // one slot, as the host budgeted it: the largest snapshot this open can hold
    pub fn take(r: *Ring) ?usize {
        for (r.taken, 0..) |used, i| {
            if (!used) {
                r.taken[i] = true;
                return i;
            }
        }
        return null; // full: a state falls back to a buffer of its own rather than the cache refusing the prompt
    }
    pub fn give(r: *Ring, at: usize) void {
        r.taken[at] = false;
    }
    pub fn slot(r: *Ring, at: usize) D {
        return .{ .d = r.rows.d, .ptr = r.rows.ptr + at * r.bytes, .len = r.bytes, .borrowed = true };
    }
};
pub var ring: ?*Ring = null; // one engine per process: the native host names the ring before any state is kept
pub const Snapshot = struct {
    data: D,
    at: u32,
    undo: bool,
    size: usize = 0, // the bytes a capture wrote: shorter than a ring slot, which is sized for the longest prompt
    slot: ?usize = null, // the ring's slot it lives in, when the cache paid for it
    from: ?*Ring = null,
    pub fn capture(s: *Sequence, undo: bool, stream: cuda.Stream) !Snapshot {
        const rs = try plan.regions(s.a, s.c, s.len, undo, s.kv8);
        defer s.a.free(rs);
        var size: usize = 0;
        for (rs) |r| size += r.bytes;
        var data: D = undefined;
        var owned: bool = true;
        var slot: ?usize = null;
        var from: ?*Ring = null;
        if (!undo) { // only the cache's kept states take the host's slots; a step's own undo record stays ours to free
            if (ring) |r| if (r.take()) |at| {
                slot = at;
                from = r;
                owned = false;
                data = r.slot(at);
            };
        }
        if (owned) data = try D.alloc(s.taps.d, size);
        errdefer if (owned) {
            stream.synchronize() catch {};
            data.free();
        };
        var at: usize = 0;
        for (rs) |r| {
            try data.copyFrom(at, s.part(r.part).ptr + r.offset, r.bytes, stream.handle);
            at += r.bytes;
        }
        try stream.synchronize();
        return .{ .data = data, .at = s.len, .undo = undo, .size = size, .slot = slot, .from = from };
    }
    pub fn restore(image: Snapshot, s: *Sequence, stream: cuda.Stream) !void {
        const rs = try plan.regions(s.a, s.c, image.at, image.undo, s.kv8);
        defer s.a.free(rs);
        var at: usize = 0;
        for (rs) |r| {
            try s.part(r.part).copyFrom(r.offset, image.data.ptr + at, r.bytes, stream.handle);
            at += r.bytes;
        }
        if (at != image.size) return error.SnapshotShape; // a ring slot is wider than the state it holds
        try stream.synchronize();
        s.len = image.at;
    }
    pub fn free(image: *Snapshot) void {
        if (image.slot) |at| {
            if (image.from) |r| r.give(at);
            image.slot = null; // a second free must not hand the same slot to two states
            return;
        }
        image.data.free();
    }
};
pub const Compressor = struct {
    kv: D,
    scores: D,
    cache: D,
    pub fn init(d: *const cuda.Driver, context: usize, ratio: usize, dim: usize, external: ?D, fp8: bool) !Compressor {
        const channels = dim * (if (ratio == 4) @as(usize, 2) else 1);
        var kv = try D.alloc(d, ratio * channels * (channels / dim) * 4);
        errdefer kv.free();
        var scores = try D.alloc(d, kv.len);
        errdefer scores.free();
        try kv.fill8(0, null);
        try scores.fill32(@bitCast(-std.math.inf(f32)), null);
        return .{ .kv = kv, .scores = scores, .cache = external orelse try D.alloc(d, (context / ratio) * plan.rowBytes(dim, fp8)) };
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
    pub fn init(d: *const cuda.Driver, c: Config, context: usize, ratio: usize, fp8: bool) !Layer {
        const row = plan.rowBytes(c.head_dim, fp8); // e4m3 bytes and one FP32 scale a row, or the bf16 halves
        const bytes = (c.sliding_window + c.vision_max_n_token) * row;
        var l: Layer = .{ .kv = try D.alloc(d, bytes + (if (ratio > 0) context / ratio * row else 0)) };
        errdefer l.free();
        if (ratio > 0) {
            l.compressor = try Compressor.init(d, context, ratio, c.head_dim, .{ .d = d, .ptr = l.kv.ptr + bytes, .len = l.kv.len - bytes, .borrowed = true }, fp8);
        }
        if (ratio == 4) l.indexer = try Compressor.init(d, context, ratio, c.index_head_dim, null, fp8);
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
    kv8: bool = false, // --kv-cache fp8: this sequence's rows are e4m3 with a scale each
    staging: ?D = null, // the rows an attention step selects, read back as BF16 for the sparse kernel (fp8 only)
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
    /// How many device buffers this sequence owns, in `part`'s order: the whole set a captured round must name.
    pub fn partCount(s: *const Sequence) usize {
        var n: usize = 2; // the dspark taps and main, which part() ends with
        for (s.layers) |l| {
            n += 1;
            if (l.compressor != null) n += 2;
            if (l.indexer != null) n += 3;
        }
        return n + s.draft.len;
    }
    /// The addresses a captured round bakes in: staging once, then every part base; extras only cost a dropAll.
    pub fn graphWords(s: *Sequence, out: []u64) !usize {
        const parts = s.partCount();
        const want = parts + (if (s.staging != null) @as(usize, 1) else 0);
        if (out.len < want) return error.AddressSetTooSmall;
        var i: usize = 0;
        if (s.staging) |*b| { out[i] = b.ptr; i += 1; }
        var k: usize = 0;
        while (k < parts) : (k += 1) { out[i] = s.part(k).ptr; i += 1; }
        return i;
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
    pub fn init(a: std.mem.Allocator, d: *const cuda.Driver, c: Config, context: usize, fp8: bool) !Sequence {
        const layers = try a.alloc(Layer, c.num_hidden_layers);
        errdefer a.free(layers);
        var n: usize = 0;
        errdefer for (layers[0..n]) |*l| l.free();
        for (layers, 0..) |*l, i| {
            l.* = try Layer.init(d, c, context, c.compress_ratios[i], fp8);
            n += 1;
        }
        const draft = try a.alloc(Layer, c.num_nextn_predict_layers);
        errdefer a.free(draft);
        var made: usize = 0;
        errdefer for (draft[0..made]) |*l| l.free();
        for (draft) |*l| { l.* = try Layer.init(d, c, context, 0, fp8); made += 1; }
        var taps = try D.alloc(d, c.hidden_size * c.dspark_target_layer_ids.len * 2);
        errdefer taps.free();
        var main = try D.alloc(d, c.hidden_size * 2);
        errdefer main.free();
        const journal = try a.alloc(?Snapshot, c.dspark_block_size + 1);
        @memset(journal, null);
        // A window's rows and an index's keys, back as BF16 for the sparse kernel: one buffer, every layer reuses it.
        errdefer a.free(journal);
        const staging = if (fp8) try D.alloc(d, @import("sparse_rows.zig").capacity(c, context) * c.head_dim * 2) else null;
        errdefer if (staging) |*rows| rows.free();
        return .{ .a = a, .c = c, .layers = layers, .draft = draft, .taps = taps, .main = main, .journal = journal, .kv8 = fp8, .staging = staging };
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
        if (s.staging) |*rows| rows.free();
    }
};
