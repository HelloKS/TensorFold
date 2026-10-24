// Lane-host adapter follows Bertholomus AI dsv41/native.zig, dfc8fe50 (Apache-2.0).
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const cluster = @import("cluster_backend");
const sys = @import("sys");
const serve = @import("serve/template.zig");
const model = @import("text_model.zig");
const kv = @import("core").kv_pool; // the shared page layout the native host plans this family's streams from
pub const model_type = "deepseek_v4";
pub const explains_open = true;
/// Templateless checkpoints serve the family's encoder; the shim keeps the cuda_chat boundary primitive-only.
fn chatRender(a: std.mem.Allocator, messages: []const u8, tools: []const u8, add_generation_prompt: bool, thinking: bool, effort: ?[]const u8, problem: *[]const u8, text: *[]u8, spans: *[]const [2]usize) anyerror!void {
    const r = try serve.render(a, messages, .{ .tools = tools, .add_generation_prompt = add_generation_prompt, .enable_thinking = thinking, .reasoning_effort = effort }, problem);
    text.* = r.text;
    spans.* = r.spans;
}
pub const chat = .{ .source = serve.source, .render = &chatRender };
pub const formats: []const []const u8 = &.{ "native-fp4-fp8", "exl3-inst3", "exl3-mcg", "exl3-mul1" };
pub const default_context: i64 = 262144;
pub const max_segments: u32 = 1;
pub const prompt_rows: u32 = 8;
pub const default_image_tokens: u32 = 1536;
pub const snap_grid: u32 = 8;
pub const snap_lookahead: u32 = 0;
pub const snap_bytes = @import("serving.zig").bytes;
pub const snap_save = @import("serving.zig").save;
pub const snap_restore = @import("serving.zig").restore;
pub const snap_drop = @import("serving.zig").drop;
pub const snap_ready = @import("serving.zig").ready;
pub const snap_slots = @import("serving.zig").slots; // the cache's kept-state slots, made where the host placed them
pub const snap_place = @import("serving.zig").place; // device or pinned host states, as cuda_kv_pool.fits found the cache fits
pub const attach = @import("serving.zig").attach; // the plan's streams and pages, agreed once at open (paged path only)
pub const window_rows: u32 = prompt_rows + 1; // a request's last verify window: pages past it stay unmade
pub const Options = struct { context: usize, drafts: bool, segments: usize = 1, slide: bool = false, vision: ?@import("vision_serving.zig").Limits = null, lanes: u32 = 1, lanes_fixed: bool = false, kv8: bool = false };
pub const LoneRun = *const fn (*anyopaque, *lanes.Stream, *anyopaque, *const fn (*anyopaque) void, *const fn (*anyopaque) bool, bool) anyerror!bool;
pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    stream_bytes: usize,
    free_streams: u32 = 0,
    max_lanes: u32 = 4,
    ctx: *anyopaque,
    target: *State,
    vision: ?*@import("serving.zig").Vision = null,
    deinit: *const fn (*anyopaque) void,
    lone: ?LoneRun = null,
    learns: ?*anyopaque = null,
    learn_bytes: usize = 0,
    /// The streams' paged KV as one pool (null: each keeps its own window; TF_DSV4_KV_PAGES names it).
    layout: ?kv.Layout = null,
    /// Rank 1's room when it is the smaller: the server admits streams and kept states by the smallest.
    peer_room: ?@import("kv_pages.zig").Peer = null,
    /// Kept states may go to the host's pinned memory: only the paged path ever sets it true.
    host_snaps: bool = false,
    /// One kept state's slot outside the pages; 0: the host's whole-cache accounting stands (pool unnamed).
    snap_bytes: usize = 0,
};
const State = @import("serving.zig").State;
pub fn env(name: [:0]const u8) ?[]const u8 {
    return std.mem.span(std.c.getenv(name) orelse return null);
}
pub fn cacheDir() ![]const u8 {
    return env("TF_DSV4_RANK_CACHE") orelse error.Set_TF_DSV4_RANK_CACHE;
}
pub fn master() ![4]u8 {
    return @import("link.zig").parseIp(env("TF_TP_MASTER") orelse return error.Set_TF_TP_MASTER);
}
pub fn port() !u16 {
    return std.fmt.parseInt(u16, env("TF_TP_PORT") orelse "29620", 10);
}
pub fn tensorWeightBytes(_: std.mem.Allocator, io: std.Io, dir: []const u8, _: bool) !u64 {
    return weightBytes(io, dir);
}
pub fn weightBytes(io: std.Io, dir: []const u8) u64 {
    _ = dir;
    const cache = cacheDir() catch return 0;
    const a = std.heap.page_allocator;
    const path = std.fs.path.join(a, &.{ cache, "rank0.safetensors" }) catch return 0;
    defer a.free(path);
    var f = std.Io.Dir.cwd().openFile(io, path, .{}) catch return 0;
    defer f.close(io);
    return f.length(io) catch 0;
}
pub fn open(a: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: ?[]const u8, o: Options) !Loaded {
    if (o.slide or o.segments != 1) return error.UnsupportedMode;
    if (o.lanes_fixed and o.lanes > 4) return error.ParallelPastRound; // the four request slots: before the weights load, as kolibri1/native.zig:80 words it
    if (sys.experimental.refusal("deepseek_v4", sys.getenv("TF_EXPERIMENTAL"))) |_| return error.ExperimentalRequired;
    if (!std.mem.eql(u8, env("TF_TP_WORLD") orelse "2", "2")) return error.TP2Required;
    const s = try a.create(State);
    errdefer a.destroy(s);
    const m = try model.Model.open(a, io, ctx, dir, try cacheDir(), env("TF_DSV4_KERNELS") orelse kernels, 0, try master(), try port(), o.context, o.kv8);
    errdefer m.close();
    const buffer = try a.alloc(u8, @import("control.zig").limit + 64);
    errdefer a.free(buffer);
    s.* = .{ .a = a, .m = m, .control = .{ .link = m.link, .rank = 0, .buffer = buffer }, .leader = undefined };
    s.leader = .{ .gpa = a, .x = s.control.view(), .shard = m.shard(), .prefill_rows = prompt_rows };
    const serving = @import("serving.zig");
    s.leader.prompt = .{ .ptr = s, .begin = serving.begin, .end = serving.end };
    s.leader.drafter = .{ .ptr = s, .guess = serving.unused, .guess_slot = serving.guess };
    if (o.vision) |limits| s.vision = try serving.Vision.open(a, io, m, dir, try cacheDir(), limits);
    const dspark = o.drafts and std.mem.eql(u8, env("TF_DSV4_DSPARK") orelse "0", "1");
    const width: u32 = if (dspark) @intCast(m.c.dspark_block_size + 1) else 1;
    return .{ .backend = s.leader.backend(), .facts = .{ .exact_width = width, .hidden_rows = true, .batch_rows = 24, .max_streams = 4, .copy_drafts = false, .lone_copies = false,
        .mtp = dspark, .draft_streams = dspark, .drafts = @intCast(m.c.dspark_block_size), .speculate = dspark, .speculate_early = false, .window_costs = if (m.wc.complete()) m.wc.costs() else &.{} },
        .rows = width, .stream_bytes = 0, .free_streams = 4, .ctx = s, .target = s, .vision = if (s.vision) |*v| v else null, .deinit = close,
        .layout = try @import("serving.zig").plan(s, o), .peer_room = s.peer, .host_snaps = s.layout != null, .snap_bytes = @import("kv_pages.zig").slotBytes(m.c, s.kv8, m.context) };
}
fn close(p: *anyopaque) void {
    const s: *State = @ptrCast(@alignCast(p));
    s.leader.stop() catch {};
    s.leader.deinit();
    @import("serving.zig").free(s); // the plan's leases and the cache's kept-state slots go back before the ranks close
    if (s.vision) |*v| v.free(s.a);
    s.a.free(s.control.buffer);
    s.m.close();
    s.a.destroy(s);
}
pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.ExperimentalRequired => "DeepSeek-V4 Flash Vision is experimental while whole-model gate #827 is pending: serve it with TF_EXPERIMENTAL=deepseek",
        error.TP2Required => "DeepSeek-V4 Flash Vision serves on two ranks: start tf-dsv4-worker and set TF_TP_WORLD=2 with TF_TP_MASTER and TF_TP_PORT",
        error.ParallelPastRound => "DeepSeek V4 TP2 allocates four request slots; --parallel must be 1 to 4",
        error.UnsupportedProfile => "this checkpoint's rank or context is outside the validated DeepSeek-V4 Flash Vision profile: rank 0 to 1 and a context of 1 to 262144 tokens",
        error.MemoryBudgetExceeded => "the text TP2 path does not fit the memory its budget promises: lower --parallel or --context, or raise the promised host memory",
        error.HostMemoryUnavailable => "the host's free memory cannot be read: lower --parallel or --context so the text TP2 path needs less",
        error.InvalidMemoryBudget => "the promised host memory is not a usable GiB count: a finite GiB count from 2 up",
        error.RankNotReady => "the rank-1 worker never reported ready: start tf-dsv4-worker over the same TF_TP_MASTER and TF_TP_PORT",
        error.PeerTimeout, error.PeerDown => "TP2 peer disconnected or exceeded its control timeout",
        error.RankIdentityMismatch => "TP2 peers have different checkpoint identities, code or context settings",
        error.PositionMismatch, error.RankStateInvalid => "TP2 sequence state diverged; restart both ranks",
        else => null,
    };
}
