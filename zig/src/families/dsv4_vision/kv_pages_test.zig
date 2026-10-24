//! The page layout's contract, and the pool's leases over fake streams: no device, no ranks, no weights.
const std = @import("std");
const pages_mod = @import("kv_pages.zig");
const kv = @import("core").kv_pool;
const Config = @import("config.zig").Config;

fn config(a: std.mem.Allocator) !Config {
    return Config.parse(a, @import("config.zig").fixture);
}

test {
    _ = @import("kv8.zig"); // the e4m3 row codec dsv4_text.cu's q8/gather8 kernels must agree with
}

test "the page layout is a whole window's rows of every paged layer and never under-books a stream" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try config(a);
    const l = pages_mod.layout(c, false, 262144) orelse return error.MissingLayout;
    try std.testing.expectEqual(@as(u32, pages_mod.page_tokens), l.page_tokens);
    try std.testing.expectEqual(@as(usize, 0), l.page_tokens % 4); // the compressor's ring period divides a page
    try std.testing.expectEqual(@as(usize, 0), l.page_tokens % 128); // and so does the long-range one
    const stream = try @import("parallel.zig").Memory.calculate(c, 262144, 1, false);
    try std.testing.expect(l.fixed_bytes + l.pages(262144) * l.pageBytes() >= stream.total());
    try std.testing.expect(l.fixed_bytes > stream.draft_bytes); // the rings and window rows are the fixed part
    const per_token = @as(u64, 21) * (512 * 2 / 4 + 128 * 2 / 4) + @as(u64, 20) * (512 * 2 / 128);
    try std.testing.expectEqual(per_token, l.token_bytes); // 21 layers at 1 in 4, 20 at 1 in 128, the index with the 1 in 4
}

test "fp8 rows halve the paged bytes and keep the fixed part's rings" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try config(a);
    const bf16 = pages_mod.layout(c, false, 262144) orelse return error.MissingLayout;
    const fp8 = pages_mod.layout(c, true, 262144) orelse return error.MissingLayout;
    try std.testing.expect(fp8.token_bytes < bf16.token_bytes);
    try std.testing.expect(fp8.token_bytes > bf16.token_bytes / 2); // an e4m3 row carries a scale the bf16 row does not
    try std.testing.expect(fp8.fixed_bytes < bf16.fixed_bytes); // the window rows shrink with the same rows
    try std.testing.expectEqual(@as(u64, 516), pages_mod.rowBytes(512, true)); // 512 bytes and one fp32 scale
    try std.testing.expectEqual(@as(usize, 516), @import("kv8.zig").rowBytes(512)); // the CPU codec and the family's arithmetic agree on a row
    try std.testing.expectEqual(@as(u64, 1024), pages_mod.rowBytes(512, false));
    try std.testing.expect(fp8.snap_bytes < bf16.snap_bytes);
}

test "a layout the host plans from keeps the asked cache beside the streams, and refuses one it cannot" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try config(a);
    const l = pages_mod.layout(c, false, 8192) orelse return error.MissingLayout;
    const page = l.pageBytes();
    const room = l.fixed_bytes * 3 + page * 3072;
    try std.testing.expect(pages_mod.holds(l, room, 4, false, room / 2, null, 70000));
    try std.testing.expect(!pages_mod.holds(l, room, 4, false, room, null, 70000)); // nothing left for a page a stream
    const served = try pages_mod.plan(l, room, 4, false, room / 2, null, 70000);
    try std.testing.expectEqual(@as(u32, 4), served.lanes); // the engine's own among them: three more than the engine
    try std.testing.expect(served.pages > 0 and served.pages < 3072); // the cache's slots and rows came off the pages
    try std.testing.expect(served.snaps > 0); // a slot is a prompt's state at this context, so the asked cache buys some
    try std.testing.expectError(error.TooMany, pages_mod.plan(l, l.fixed_bytes + page, 4, true, 0, null, 0)); // --parallel named what the room cannot hold
    const tight = try pages_mod.plan(l, 0, 4, false, 0, null, 0); // no room at all: the engine's own sequence, and no pages
    try std.testing.expectEqual(@as(u32, 1), tight.lanes);
    try std.testing.expectEqual(@as(usize, 0), tight.pages);
}

test "fake streams promise, cover and hand back their pages without stranding one" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try config(a);
    const l = pages_mod.layout(c, false, 8192) orelse return error.MissingLayout;
    var p = try pages_mod.Pages.init(a, l, 2, 64);
    defer p.deinit();
    try std.testing.expectEqual(@as(usize, 64), p.available());
    const tokens = @as(u64, l.page_tokens) * 8;
    try p.reserve(0, tokens, 0);
    try std.testing.expectEqual(@as(usize, 56), p.available()); // promised, though nobody has taken it yet
    try p.cover(0, tokens);
    try std.testing.expectEqual(@as(usize, 8), p.lease(0).ids.items.len);
    try std.testing.expectEqual(@as(usize, 56), p.available()); // taken out of the free list, no longer promised
    try std.testing.expect(p.fits(1, tokens, 0));
    try std.testing.expect(!p.fits(1, @as(u64, l.page_tokens) * 60, 0)); // the other stream cannot have what this one holds
    p.finish(0);
    try std.testing.expectEqual(@as(usize, 64), p.available());
    const idle = p.census();
    try std.testing.expectEqual(@as(usize, 0), idle.promised);
    try std.testing.expectEqual(@as(usize, 0), idle.leased);
    try std.testing.expectEqual(@as(usize, 64), idle.free + idle.kept);
}

test "a stream past its reservation is refused, and an empty pool plans nothing" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try config(a);
    const l = pages_mod.layout(c, false, 8192) orelse return error.MissingLayout;
    try std.testing.expectError(error.NoPages, pages_mod.Pages.init(a, l, 0, 0));
    var p = try pages_mod.Pages.init(a, l, 1, 2);
    defer p.deinit();
    try std.testing.expectError(error.LaneMemory, p.reserve(0, @as(u64, l.page_tokens) * 3, 0));
    try p.reserve(0, @as(u64, l.page_tokens) * 2, 0);
    try std.testing.expectError(error.LaneMemory, p.reserve(0, @as(u64, l.page_tokens) * 3, 0)); // the same promise twice
    const one = p.census();
    try std.testing.expectEqual(@as(usize, 2), one.promised);
}

test "the paged path stays off until TF_DSV4_KV_PAGES names it" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try config(a);
    try std.testing.expectEqual(@as(?kv.Layout, null), pages_mod.layoutOf(c, .{ .context = 8192, .drafts = false }));
    try std.testing.expectEqual(@as(u64, 0), pages_mod.slotBytes(c, false, 8192)); // the host keeps its whole-cache accounting
    try std.testing.expect(pages_mod.layout(c, false, 8192) != null); // the gate is the environment, never the arithmetic
}

test "the ranks agree the room once and the host admits by the smaller of them" {
    // The single exchange serving.plan makes at open: rank 1's figure wins only when it is genuinely the smaller.
    try std.testing.expectEqual(@as(?pages_mod.Peer, null), pages_mod.smaller(1 << 30, 1 << 30, 1));
    try std.testing.expectEqual(@as(?pages_mod.Peer, null), pages_mod.smaller(1 << 30, 3 << 29, 1));
    const behind = pages_mod.smaller(1 << 30, 1 << 29, 1) orelse return error.MissingPeer;
    try std.testing.expectEqual(@as(u64, 1 << 29), behind.bytes);
    try std.testing.expectEqual(@as(usize, 1), behind.rank); // so the host's refusal can name the rank it admitted by
}

test "an fp8 layout never under-books what an fp8 stream actually needs" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try config(a);
    const l = pages_mod.layout(c, true, 262144) orelse return error.MissingLayout;
    const stream = try @import("parallel.zig").Memory.calculate(c, 262144, 1, true);
    try std.testing.expect(l.fixed_bytes + l.pages(262144) * l.pageBytes() >= stream.total());
    try std.testing.expect(l.fixed_bytes > stream.draft_bytes); // the FP32 rings still sit in the fixed part
    try std.testing.expectEqual(pages_mod.tokenBytes(c, true) * pages_mod.page_tokens, l.pageBytes()); // a page is the token's bytes, a window deep
    try std.testing.expect(l.pageBytes() < pages_mod.tokenBytes(c, false) * pages_mod.page_tokens); // the same window, narrower rows
}

test "a kept state's slot is the whole snapshot at the position it can hold, twice over with its header" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try config(a);
    const ranges = @import("state_plan.zig");
    const whole = try ranges.bytes(c, 262144, false, false);
    try std.testing.expectEqual(2 * whole + pages_mod.saved_bytes, pages_mod.snapBytes(c, false, 262144)); // what a capture writes
    const short = try ranges.bytes(c, 128, false, false);
    try std.testing.expectEqual(2 * short + pages_mod.saved_bytes, pages_mod.snapBytes(c, false, 128)); // and the same at a short prompt
    try std.testing.expect(pages_mod.snapBytes(c, false, 8192) < pages_mod.snapBytes(c, false, 262144)); // sized by the open's context
    const narrow = try ranges.bytes(c, 262144, false, true);
    try std.testing.expect(2 * narrow < 2 * whole); // an e4m3 cache keeps smaller states, and the host budgets them
}

test "the host's own fits() test decides where kept states go, both ways, and only on the paged path" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try config(a);
    const l = pages_mod.layout(c, false, 8192) orelse return error.MissingLayout;
    const room = l.fixed_bytes * 3 + l.pageBytes() * 3072;
    try std.testing.expect(pages_mod.holds(l, room, 4, false, room / 2, null, 70000)); // the card holds the asked cache: states stay there
    try std.testing.expect(pages_mod.placed(true, false) == false);
    try std.testing.expect(!pages_mod.holds(l, room, 4, false, room, null, 70000)); // the same card cannot: the host places them on the host
    try std.testing.expect(pages_mod.placed(true, true));
    try std.testing.expect(pages_mod.placed(false, true) == false); // unnamed pool: today's fixed policy stands, whatever fits() said
}

test "the cache is planned against the room both ranks agreed to, not the room rank 0 sees" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try config(a);
    const l = pages_mod.layout(c, false, 8192) orelse return error.MissingLayout;
    const local = l.fixed_bytes * 3 + l.pageBytes() * 3072;
    const asked = local / 2; // --prompt-cache-gib, asked of rank 0
    const behind = pages_mod.smaller(local, local / 2, 1) orelse return error.MissingPeer;
    try std.testing.expect(pages_mod.holds(l, local, 4, false, asked, null, 70000)); // rank 0 alone would admit this cache
    try std.testing.expect(!pages_mod.holds(l, behind.bytes, 4, false, asked, null, 70000)); // rank 1's room refuses it, so the open does too
}

test "the cache's ring hands a kept state one slot at a time and takes it back" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const states = @import("text_state.zig");
    const taken = try a.alloc(bool, 3);
    @memset(taken, false);
    var r = states.Ring{ .rows = .{ .d = undefined, .ptr = 4096, .len = 3 * 512, .borrowed = true }, .taken = taken, .bytes = 512 };
    const first = r.take() orelse return error.NoSlot;
    const second = r.take() orelse return error.NoSlot;
    const third = r.take() orelse return error.NoSlot;
    try std.testing.expectEqual(@as(usize, 0), first);
    try std.testing.expectEqual(@as(usize, 1), second);
    try std.testing.expectEqual(@as(usize, 2), third);
    const view = r.slot(second);
    try std.testing.expectEqual(@as(u64, 4096 + 512), view.ptr); // the second state's own bytes, and nobody else's
    try std.testing.expectEqual(@as(usize, 512), view.len);
    try std.testing.expect(view.borrowed); // the ring owns the memory, not the state parked in it
    try std.testing.expect(r.take() == null); // full: a state falls back to a buffer of its own, never a refused prompt
    try std.testing.expect(r.take() == null); // asking twice while full changes nothing
    r.give(first);
    const again = r.take() orelse return error.NoSlot;
    try std.testing.expectEqual(first, again); // a dropped state's slot is the next state's
    r.give(second);
    r.give(again);
    r.give(third);
    try std.testing.expectEqual(@as(usize, 0), r.take() orelse return error.NoSlot); // all three are back in circulation
}

fn fake(at: u64) @import("cuda").DeviceBuffer {
    return .{ .d = undefined, .ptr = at, .len = 64, .borrowed = true }; // addresses only: nothing here is ever read as memory
}

test "the addresses a kv8 round bakes in are the one walk a snapshot already copies" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const st = @import("text_state.zig");
    const c = try config(a);
    var layers = [_]st.Layer{.{ .kv = fake(0x1000), .compressor = .{ .kv = fake(0x1100), .scores = fake(0x1200), .cache = fake(0x1300) }, .indexer = .{ .kv = fake(0x1400), .scores = fake(0x1500), .cache = fake(0x1600) } }};
    var draft = [_]st.Layer{.{ .kv = fake(0x1700) }};
    var journal = [_]?st.Snapshot{null};
    var seq: st.Sequence = .{ .a = a, .c = c, .layers = &layers, .draft = &draft, .taps = fake(0x1800), .main = fake(0x1900), .journal = &journal, .kv8 = true, .staging = fake(0x1a00) };
    try std.testing.expectEqual(@as(usize, 9), seq.partCount()); // the layer's rows, the compressor's three, the indexer's three, the draft row, taps and main
    var words: [12]u64 = undefined;
    try std.testing.expectEqual(@as(usize, 10), try seq.graphWords(&words)); // staging once, then every part base
    try std.testing.expectEqual(@as(u64, 0x1a00), words[0]); // gather8's write target leads the set: one entry, never one per layer
    var k: usize = 0;
    while (k < 9) : (k += 1) try std.testing.expectEqual(seq.part(k).ptr, words[k + 1]); // the very walk snapshots copy, so the two can never disagree
    var few: [5]u64 = undefined;
    try std.testing.expectError(error.AddressSetTooSmall, seq.graphWords(&few)); // a short list is refused, not silently truncated
    seq.staging = null; // a bf16 sequence has no staging row: the set is exactly the parts it owns
    try std.testing.expectEqual(@as(usize, 9), try seq.graphWords(&words));
}
