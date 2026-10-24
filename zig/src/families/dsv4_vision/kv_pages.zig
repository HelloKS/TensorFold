//! The streams' KV as one pool of pages: the layout the native host plans from, and the pages it agrees at open.
const std = @import("std");
const kv = @import("core").kv_pool;
const Config = @import("config.zig").Config;
const state_plan = @import("state_plan.zig");

/// Tokens a KV page holds: a multiple of every compression ratio, so a page is whole rows of every layer.
pub const page_tokens = 512;
/// A kept state's own header, as serving.Saved: its id, the position it holds and the identity, padded to a row.
pub const saved_bytes = 48;

/// The smallest room of the ranks and whose it is, agreed once over the family's control link.
pub const Peer = struct { bytes: u64, rank: usize };

/// After the one exchange, the room to admit by: `peer`'s figure when it is the smaller, else the local one stands.
pub fn smaller(local: u64, peer: u64, rank: usize) ?Peer {
    return if (peer < local) .{ .bytes = peer, .rank = rank } else null;
}

/// One cache row of `dim` channels: the bf16 halves, or --kv-cache fp8's e4m3 bytes and their fp32 scale.
pub const rowBytes = state_plan.rowBytes; // one cache row's width, shared with the snapshot's byte ranges

/// `a` over `b` rounded up: a page's arithmetic never promises a device fewer bytes than a layer holds.
fn divUp(a: u64, b: u64) u64 {
    return (a + b - 1) / b;
}

/// The compressor ring of one layer, kv and scores: the recurrent part a page never holds.
fn ringBytes(ratio: usize, dim: usize) u64 {
    const overlap: u64 = if (ratio == 4) 2 else 1;
    return 2 * @as(u64, ratio) * (@as(u64, dim) * overlap) * overlap * 4;
}

/// What a stream holds at any length: every layer's window rows and rings, the draft rows, the taps and the journal.
pub fn fixedBytes(c: Config, fp8: bool) u64 {
    const row = rowBytes(c.head_dim, fp8);
    var bytes: u64 = 0;
    for (c.compress_ratios[0..c.num_hidden_layers]) |ratio| {
        bytes += (c.sliding_window + c.vision_max_n_token) * row;
        if (ratio == 0) continue;
        bytes += ringBytes(ratio, c.head_dim);
        if (ratio == 4) bytes += ringBytes(ratio, c.index_head_dim);
    }
    bytes += @as(u64, c.num_nextn_predict_layers) * c.sliding_window * row;
    bytes += @as(u64, c.hidden_size) * (c.dspark_target_layer_ids.len + 1) * 2;
    bytes += @as(u64, c.dspark_block_size + 1) * (state_plan.bytes(c, 0, true, fp8) catch return std.math.maxInt(u64));
    return bytes;
}

/// The bytes one token adds across every paged cache, rounded up a layer at a time: the prefixes and the index.
pub fn tokenBytes(c: Config, fp8: bool) u64 {
    var bytes: u64 = 0;
    for (c.compress_ratios[0..c.num_hidden_layers]) |ratio| {
        if (ratio == 0) continue;
        bytes += divUp(rowBytes(c.head_dim, fp8), ratio);
        if (ratio == 4) bytes += divUp(rowBytes(c.index_head_dim, fp8), ratio);
    }
    return bytes;
}

/// One kept state's slot at the longest position `at` it can hold: the whole snapshot twice over, plus its header.
pub fn snapBytes(c: Config, fp8: bool, at: usize) u64 {
    const whole = state_plan.bytes(c, at, false, fp8) catch return std.math.maxInt(u64);
    return 2 * whole + saved_bytes;
}

/// The page's bytes: a whole window's rows of every paged layer, rounded the same way the token's are.
pub fn pageBytes(c: Config, fp8: bool) u64 {
    return tokenBytes(c, fp8) * page_tokens;
}

/// TF_DSV4_KV_PAGES=1 names the pool: the whole paged path, placement included, stays off until it does.
pub fn enabled() bool {
    const named = std.c.getenv("TF_DSV4_KV_PAGES") orelse return false;
    return std.mem.eql(u8, std.mem.span(named), "1");
}

/// Where kept states go: the host's pinned memory, once the pool is named and the card cannot hold the cache.
pub fn placed(paged: bool, host_mode: bool) bool {
    return paged and host_mode;
}

/// The family's stream memory as the host's page layout: a fixed part, a token's paged bytes and a snapshot's slot.
pub fn layout(c: Config, fp8: bool, context: usize) ?kv.Layout {
    const fixed = fixedBytes(c, fp8);
    const token = tokenBytes(c, fp8);
    if (fixed == 0 or fixed == std.math.maxInt(u64) or token == 0) return null;
    return .{ .fixed_bytes = fixed, .token_bytes = token, .page_tokens = page_tokens, .snap_bytes = snapBytes(c, fp8, context) };
}

/// TF_DSV4_KV_PAGES=1 puts the family's streams on the host's page pool; null: each keeps its own window, as today.
pub fn layoutOf(c: Config, o: anytype) ?kv.Layout {
    if (!enabled()) return null;
    const fp8 = if (comptime @hasField(@TypeOf(o), "kv8")) o.kv8 else false;
    return layout(c, fp8, if (comptime @hasField(@TypeOf(o), "context")) o.context else @import("config.zig").default_context);
}

/// What the native host charges one kept state's slot at `at` tokens; 0 leaves its whole-cache accounting as it stands.
pub fn slotBytes(c: Config, fp8: bool, at: usize) u64 {
    return if (enabled()) snapBytes(c, fp8, at) else 0;
}

/// Whether the asked prompt cache fits whole beside the streams, the native host's own test for where snapshots go.
pub fn holds(l: kv.Layout, room: u64, lanes: u32, fixed: bool, cache: u64, snaps: ?usize, window: u64) bool {
    return kv.holds(l, .{ .room = room, .engine = 0, .lanes = lanes, .fixed = fixed, .cache = cache, .snaps = snaps, .window = window });
}

/// What the room pays for: the streams served, the cache's slots and copied rows, and the pages left for the streams.
pub fn plan(l: kv.Layout, room: u64, lanes: u32, fixed: bool, cache: u64, snaps: ?usize, window: u64) error{ NoStream, TooMany }!kv.Plan {
    return kv.plan(l, .{ .room = room, .engine = 0, .lanes = lanes, .fixed = fixed, .cache = cache, .snaps = snaps, .window = window });
}

/// The pool of page ids the host planned, and one lease per stream: the family's half of the paged path.
pub const Pages = struct {
    gpa: std.mem.Allocator,
    layout: kv.Layout,
    pool: kv.Pool,
    leases: []kv.Lease,
    used: []bool, // a lease that ever took a page owns its table; one that was only promised has none to free

    /// The plan's `pages` of `l`, one lease for each of the `streams` streams the host admitted.
    pub fn init(gpa: std.mem.Allocator, l: kv.Layout, streams: u32, pages: usize) !Pages {
        if (streams == 0 or pages == 0) return error.NoPages;
        const used = try gpa.alloc(bool, streams);
        errdefer gpa.free(used);
        @memset(used, false);
        const leases = try gpa.alloc(kv.Lease, streams);
        errdefer gpa.free(leases);
        @memset(leases, .{}); // a lease that never promised or took a page is empty, not whatever the allocator held
        return .{ .gpa = gpa, .layout = l, .pool = try kv.Pool.init(gpa, pages, l.pageBytes()), .leases = leases, .used = used };
    }

    pub fn deinit(p: *Pages) void {
        for (0..p.leases.len) |i| p.finish(i);
        p.gpa.free(p.leases);
        p.gpa.free(p.used);
        p.pool.deinit();
        p.* = undefined;
    }

    /// Stream `i`'s lease; the host admitted exactly `leases.len` of them.
    pub fn lease(p: *Pages, i: usize) *kv.Lease {
        return &p.leases[i];
    }

    /// Promise the pages stream `i` needs for `tokens` past its table, evicting the store's idle ones first.
    pub fn reserve(p: *Pages, i: usize, tokens: u64, window: u64) !void {
        const need = p.layout.need(tokens, 0, window) -| p.leases[i].ids.items.len;
        try p.pool.promise(p.lease(i), need);
    }

    /// Take the promised pages into the stream's table; a device table writer would see them from `from`.
    pub fn cover(p: *Pages, i: usize, tokens: u64) !void {
        try p.pool.cover(p.lease(i), p.layout.pages(tokens), null);
        p.used[i] = true;
    }

    /// The stream ended: its promise and every page it held are the pool's again.
    pub fn finish(p: *Pages, i: usize) void {
        if (!p.used[i]) {
            p.pool.unpromise(p.lease(i)); // promised and never taken: there is no table to give back
            return;
        }
        p.pool.finish(p.lease(i));
    }

    /// Pages free and promised to nobody: what the next stream's admission asks about.
    pub fn available(p: *const Pages) usize {
        return p.pool.available();
    }

    /// Whether `tokens` plus a verify window could be promised to stream `i` now, the store's idle pages counted.
    pub fn fits(p: *const Pages, i: usize, tokens: u64, window: u64) bool {
        const need = p.layout.need(tokens, 0, window) -| p.leases[i].ids.items.len;
        return p.pool.fits(need);
    }

    /// The counts an idle server logs: nothing promised, no page leased, every page free or the store's.
    pub fn census(p: *const Pages) kv.Census {
        return p.pool.census();
    }
};
