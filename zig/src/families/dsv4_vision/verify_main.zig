//! Hardware qualification emits actual token IDs from the same native lane backend used by HTTP.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const native = @import("native.zig");
const serving = @import("serving.zig");
const Case = struct { prompt: []const u32, max_tokens: u32 = 32, temperature: f64 = 0, seed: u64 = 77, images: []const []const u8 = &.{} };
const Input = struct { prompt: []const u32, images: []const lanes.stream.Image, max_tokens: u32, sampling: ?lanes.Sampling };
const Run = struct { tokens: [][]u32, drafted: u64, accepted: u64, cached: u64 };
const Mark = struct {
    state: *serving.State,
    saved: ?*anyopaque = null,
    err: ?anyerror = null,
    fn save(p: *anyopaque, s: *lanes.Stream, at: u32) void {
        const m: *Mark = @ptrCast(@alignCast(p));
        m.saved = serving.save(m.state, s, at) catch |err| { m.err = err; return; };
    }
};
const Cancel = struct {
    stop: bool = false,
    fn check(p: *anyopaque) bool { const c: *Cancel = @ptrCast(@alignCast(p)); return c.stop; }
};
fn generate(a: std.mem.Allocator, io: std.Io, loaded: native.Loaded, inputs: []const Input, drafts: bool, reuse: lanes.stream.Reuse, cancel: bool) !Run {
    var cfg = try lanes.Config.init(a, loaded.facts, loaded.rows, loaded.rows - 1);
    defer cfg.deinit(a);
    var clock: lanes.backend.WallClock = .{ .io = io };
    var engine = lanes.Engine.init(a, &cfg, loaded.backend, clock.clock());
    defer engine.deinit();
    const streams = try a.alloc(lanes.Stream, inputs.len);
    defer a.free(streams);
    var ready: usize = 0;
    defer for (streams[0..ready]) |*s| s.deinit(a);
    var stop: Cancel = .{};
    for (inputs, streams) |input, *s| {
        s.* = try lanes.Stream.init(a, .{ .id = "qualification", .prompt = input.prompt, .images = input.images, .max_new = input.max_tokens,
            .sampling = input.sampling, .drafts = drafts, .reuse = reuse, .cancel_check = if (cancel) .{ .ptr = &stop, .check = Cancel.check } else null });
        ready += 1;
        try engine.addStream(s);
    }
    while (engine.activeCount() > 0) {
        if (cancel and streams[0].emitted().len >= 2) stop.stop = true;
        try engine.step();
    }
    const tokens = try a.alloc([]u32, streams.len);
    var result: Run = .{ .tokens = tokens, .drafted = 0, .accepted = 0, .cached = 0 };
    for (streams, tokens) |s, *ids| {
        if (s.reason == .@"error") { std.debug.print("native qualification failed: {s}\n", .{s.problem}); return error.InferenceFailed; }
        ids.* = try a.dupe(u32, s.emitted());
        result.drafted += s.drafted;
        result.accepted += s.accepted;
        result.cached += s.cached;
    }
    return result;
}
fn same(lhs: Run, rhs: Run) !void {
    if (lhs.tokens.len != rhs.tokens.len) return error.StreamCount;
    for (lhs.tokens, rhs.tokens) |a, b| if (!std.mem.eql(u32, a, b)) return error.TokenMismatch;
}
fn prepare(a: std.mem.Allocator, io: std.Io, state: *serving.State, cases: []const Case, context: usize) ![]Input {
    const out = try a.alloc(Input, cases.len);
    for (out, cases) |*input, c| {
        var ids: std.ArrayList(u32) = .empty;
        var images: std.ArrayList(lanes.stream.Image) = .empty;
        var i: usize = 0;
        for (c.prompt) |token| {
            if (state.vision != null and token == state.vision.?.marks.image) {
                if (i >= c.images.len) return error.ImageCount;
                const data = try std.Io.Dir.cwd().readFileAlloc(io, c.images[i], a, .limited(10 << 20));
                const v = &state.vision.?;
                const p = try v.prepare(a, data, 384);
                const at: u32 = @intCast(ids.items.len);
                const words = try v.position(a, p.gh, p.gw, at);
                try images.append(a, .{ .pixels = p.pixels, .gh = p.gh, .gw = p.gw, .at = at, .tokens = @intCast(words.len), .hash = p.hash });
                try ids.appendSlice(a, words);
                i += 1;
            } else try ids.append(a, token);
        }
        if (i != c.images.len or ids.items.len == 0 or ids.items.len + c.max_tokens > context) return error.FixtureContext;
        input.* = .{ .prompt = ids.items, .images = images.items, .max_tokens = c.max_tokens,
            .sampling = if (c.temperature == 0) null else .{ .seed = c.seed, .temperature = c.temperature, .top_k = 0, .top_p = 0.95 } };
    }
    return out;
}
pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 6) { std.debug.print("usage: tf-dsv4-verify MODEL FIXTURE.json CONTEXT KERNEL_DIR RESULT.json\n", .{}); return error.Usage; }
    if (!std.mem.eql(u8, native.env("TF_DSV4_DSPARK") orelse "0", "1")) {
        std.debug.print("set TF_DSV4_DSPARK=1 for explicit DSpark qualification\n", .{});
        return error.DSparkNotEnabled;
    }
    const text = try std.Io.Dir.cwd().readFileAlloc(init.io, args[2], a, .limited(64 << 20));
    const fixture = try std.json.parseFromSliceLeaky(struct { cases: []const Case }, a, text, .{});
    if (fixture.cases.len == 0 or fixture.cases.len > 4) return error.FixtureConcurrency;
    var d = try cuda.Driver.open();
    defer d.close();
    var ctx = try cuda.Context.init(&d, 0);
    defer ctx.deinit();
    const context = try std.fmt.parseInt(usize, args[3], 10);
    var vision = false;
    for (fixture.cases) |c| { if (c.images.len > 0) vision = true; }
    const loaded = try native.open(init.gpa, init.io, &ctx, args[1], args[4], .{ .context = context, .drafts = true, .vision = if (vision) .{} else null });
    var live = true;
    defer if (live) loaded.deinit(loaded.ctx);
    const state: *serving.State = @ptrCast(@alignCast(loaded.ctx));
    const inputs = try prepare(a, init.io, state, fixture.cases, context);
    const plain = try generate(a, init.io, loaded, inputs, false, .{}, false);
    _ = try serving.ready(state, 512 << 20);
    const drafted = try generate(a, init.io, loaded, inputs, true, .{}, false);
    try same(plain, drafted);
    for (inputs, 0..) |input, i| {
        const solo = try generate(a, init.io, loaded, &.{input}, true, .{}, false);
        if (!std.mem.eql(u32, solo.tokens[0], plain.tokens[i])) return error.ConcurrentMismatch;
    }
    var at: u32 = @intCast(inputs[0].prompt.len / 2);
    for (inputs[0].images) |image| if (at > image.at and at < image.at + image.tokens) { at = image.at + image.tokens; };
    if (at == 0 or at >= inputs[0].prompt.len) return error.FixtureNeedsPrefix;
    var mark: Mark = .{ .state = state };
    _ = try generate(a, init.io, loaded, inputs[0..1], true, .{ .marks = &.{at}, .hook = .{ .ptr = &mark, .at = Mark.save } }, false);
    if (mark.err) |err| return err;
    const saved = mark.saved orelse return error.SnapshotNotSaved;
    var kept = true;
    defer if (kept) serving.drop(state, saved);
    const cached = try generate(a, init.io, loaded, inputs[0..1], true, .{ .saved = saved, .at = at }, false);
    if (cached.cached != at or !std.mem.eql(u32, cached.tokens[0], plain.tokens[0])) return error.CachedMismatch;
    _ = try generate(a, init.io, loaded, inputs[0..1], true, .{}, true);
    const after = try generate(a, init.io, loaded, inputs, true, .{}, false);
    try same(after, plain);
    if (drafted.drafted == 0) return error.DSparkNotExercised;
    const identity = state.m.identity;
    const image_hits = if (state.vision) |v| v.cache.hits else 0;
    const image_misses = if (state.vision) |v| v.cache.misses else 0;
    const used = cuda.usage(false);
    const before = try ctx.memInfo();
    serving.drop(state, saved);
    kept = false;
    loaded.deinit(loaded.ctx);
    live = false;
    const receipt = try std.json.Stringify.valueAlloc(a, .{ .passed = true, .state_identity = identity, .context = context,
        .concurrency = inputs.len, .drafted = drafted.drafted, .accepted = drafted.accepted, .cached_tokens = cached.cached,
        .plain_token_ids = plain.tokens, .dspark_token_ids = drafted.tokens, .tracked = used, .after_cleanup = cuda.usage(false),
        .cuda_free_before_cleanup = before.free, .cuda_free_after_cleanup = (try ctx.memInfo()).free,
        .image_embedding_hits = image_hits, .image_embedding_misses = image_misses }, .{ .whitespace = .indent_2 });
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[5], .data = receipt });
}
