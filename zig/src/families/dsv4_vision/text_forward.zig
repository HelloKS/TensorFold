//! Single-token reference graph; prefill and decode execute identical row arithmetic.
const std = @import("std");
const cuda = @import("cuda");
const st = @import("text_state.zig");
const gpu = @import("gpu_weights.zig");
const B = @import("text_runtime.zig").B;
const rg = @import("round_graph.zig");

/// A host-dependent stretch of one row: recorded as a deferred piece and re-run live (uncaptured) on every replay.
fn Region(comptime M: type) type {
    return struct {
        const Self = @This();
        m: M,
        /// Kinds carry row indices only: live() reads the model's fresh row, so replays see this round's values.
        kind: union(enum) {
            embed: struct { at: u32 },
            attention: struct { i: usize, at: u32 },
            moe: struct { i: usize, at: u32, hashed: bool },
            absorb: struct { at: u32 },
        },

        fn live(ctx: *anyopaque, stream: cuda.Stream) anyerror!void {
            const s: *Self = @ptrCast(@alignCast(ctx));
            _ = stream; // the round's stream is the model's own, current at every replay
            switch (s.kind) {
                .embed => |e| {
                    const token = s.m.glive[e.at].token; // the model's refreshed row: a replay embeds this round's token
                    try s.m.r.buf(.ids).upload(0, std.mem.asBytes(&token));
                    const embed = try s.m.w.tensor("embed.weight");
                    try s.m.r.operators.tensor.run(.embed, s.m.stream, .{ .x = @intCast((s.m.c.hidden_size + 127) / 128), .y = 1 }, 128, .{ embed.data.ptr, s.m.r.p(.ids), s.m.r.p(.h), @as(u32, @intCast(s.m.c.hidden_size)), @as(u32, @intCast(s.m.c.vocab_size)), @as(u32, @backingInt(try gpu.kind(embed.dtype))), @as(u32, 4) });
                },
                .attention => |a| {
                    const seq = try s.m.sequence(s.m.glive[a.at].slot); // live at replay: the row's current sequence, its current length
                    var b: [80]u8 = undefined;
                    try attention(s.m, try key(&b, "layers.{d}", .{a.i}), &seq.layers[a.i], seq, seq.len, s.m.c.compress_ratios[a.i], null);
                },
                .moe => |o| {
                    var b: [80]u8 = undefined;
                    try moe(s.m, try key(&b, "layers.{d}", .{o.i}), o.hashed, s.m.glive[o.at].token >= s.m.c.vocab_size); // image-ness too: read fresh
                },
                .absorb => |a| try @import("dspark.zig").absorb(s.m, try s.m.sequence(s.m.glive[a.at].slot), (try s.m.sequence(s.m.glive[a.at].slot)).len),
            }
        }

        fn destroy(ctx: *anyopaque) void {
            const s: *Self = @ptrCast(@alignCast(ctx));
            s.m.a.destroy(s);
        }

        fn deferStretch(r: *rg.Recorder, m: M, kind: @FieldType(Self, "kind")) !void {
            const s = try m.a.create(Self);
            errdefer m.a.destroy(s);
            s.* = .{ .m = m, .kind = kind };
            try r.deferSteps(1, .{ .ctx = s, .run = live, .dtor = destroy });
        }
    };
}
pub fn key(buf: []u8, comptime pattern: []const u8, args: anytype) ![]const u8 {
    return std.fmt.bufPrint(buf, pattern, args);
}
pub fn norm(m: anytype, x: u64, w: u64, y: u64, dim: usize, rows: usize, qnorm: bool) !void {
    try m.r.text.call("norm", m.stream, @intCast(rows), 256, .{ x, w, y, @as(u32, @intCast(dim)), @as(u32, 0), @as(f32, @floatCast(m.c.rms_norm_eps)), @as(u32, @intFromBool(qnorm)) });
}
pub fn project(m: anytype, name: []const u8, x: u64, y: u64, output: cuda.dsv4_tensor.Kind) !void {
    try (try m.w.matrix(name)).run(m.r.operators.exl3, m.r.operators.native, m.r.operators.tensor, m.stream, &m.r.scratch, x, y, 1, .bf16, output);
}
pub fn rawProject(m: anytype, name: []const u8, x: u64, y: u64, k: usize, n: usize, normalized: bool, output: cuda.dsv4_tensor.Kind) !void {
    const weight = try m.w.tensor(name);
    try m.r.operators.tensor.dense(m.stream, x, weight.data.ptr, 0, y, 1, @intCast(k), @intCast(n), .bf16, try gpu.kind(weight.dtype), output, normalized, @floatCast(m.c.rms_norm_eps));
}
pub fn rope(m: anytype, x: u64, dim: usize, heads: usize, pos: u32, compressed: bool, inverse: bool) !void {
    try m.r.text.call("rope", m.stream, @intCast(heads), 128, .{ x, m.freq[@intFromBool(compressed)].ptr, pos, @as(u32, @intCast(dim)), @as(u32, @intCast(m.c.qk_rope_head_dim)), @as(u32, @intFromBool(inverse)) });
}
pub fn reduce(m: anytype, x: u64, y: u64, n: usize, cast: bool) !void {
    try m.comm.allGather(x, m.r.p(.gather), n, .f32, m.stream);
    try m.r.text.call("merge", m.stream, @intCast((n + 127) / 128), 128, .{ m.r.p(.gather), y, @as(u32, @intCast(n)), @as(u32, @intFromBool(cast)) });
}
pub fn hc(m: anytype, p: []const u8, part: []const u8) !void {
    var b: [160]u8 = undefined;
    try rawProject(m, try key(&b, "{s}.hc_{s}_fn", .{ p, part }), m.r.p(.h), m.r.p(.mixes), 4 * m.c.hidden_size, 24, true, .f32);
    const scale = try m.w.ptr(try key(&b, "{s}.hc_{s}_scale", .{ p, part }));
    const base = try m.w.ptr(try key(&b, "{s}.hc_{s}_base", .{ p, part }));
    try m.r.operators.model.sinkhorn(m.stream, m.r.p(.mixes), scale, base, m.r.p(.pre), m.r.p(.post), m.r.p(.comb), 1, @intCast(m.c.hc_sinkhorn_iters), @floatCast(m.c.hc_eps));
    try m.r.buf(.residual).copyFrom(0, m.r.p(.h), 4 * m.c.hidden_size * 2, m.stream.handle);
    try m.r.operators.model.hcPre(m.stream, m.r.p(.h), m.r.p(.pre), m.r.p(.x), 1, @intCast(m.c.hidden_size));
    try norm(m, m.r.p(.x), try m.w.ptr(try key(&b, "{s}.{s}_norm.weight", .{ p, part })), m.r.p(.norm), m.c.hidden_size, 1, false);
}
pub fn post(m: anytype) !void {
    try m.r.operators.model.hcPost(m.stream, m.r.p(.out), m.r.p(.residual), m.r.p(.post), m.r.p(.comb), m.r.p(.h), 1, @intCast(m.c.hidden_size));
}
/// One cache row's bytes: the BF16 halves, or --kv-cache fp8's e4m3 codes with the one FP32 scale they share.
pub fn rowBytes(m: anytype, dim: usize) usize {
    return @import("state_plan.zig").rowBytes(dim, m.kv8);
}
/// One row into the cache: quantized to e4m3 when the sequence's rows are, a plain copy while they are still halves.
pub fn writeRow(m: anytype, to: cuda.DeviceBuffer, at: usize, src: u64, dim: usize) !void {
    if (m.kv8) return m.r.text.call("q8", m.stream, 1, @intCast(dim), .{ src, to.ptr + at, @as(u32, @intCast(dim)) });
    return to.copyFrom(at, src, dim * 2, m.stream.handle);
}
fn compress(m: anytype, p: []const u8, cache: *st.Compressor, pos: u32, ratio: usize, dim: usize, index: bool) !void {
    var b: [160]u8 = undefined;
    const channels = dim * (if (ratio == 4) @as(usize, 2) else 1);
    try project(m, try key(&b, "{s}.wkv", .{p}), m.r.p(.norm), m.r.p(.gate), .f32);
    try project(m, try key(&b, "{s}.wgate", .{p}), m.r.p(.norm), m.r.p(.up), .f32);
    try m.r.text.call("store", m.stream, @intCast((channels + 127) / 128), 128, .{ m.r.p(.gate), m.r.p(.up), try m.w.ptr(try key(&b, "{s}.ape", .{p})), cache.kv.ptr, cache.scores.ptr, pos, @as(u32, @intCast(ratio)), @as(u32, @intCast(dim)) });
    if ((pos + 1) % ratio != 0) return;
    try m.r.text.call("pool", m.stream, @intCast((dim + 127) / 128), 128, .{ cache.kv.ptr, cache.scores.ptr, m.r.p(.kv), pos, @as(u32, @intCast(ratio)), @as(u32, @intCast(dim)) });
    try norm(m, m.r.p(.kv), try m.w.ptr(try key(&b, "{s}.norm.weight", .{p})), m.r.p(.kv), dim, 1, false);
    try rope(m, m.r.p(.kv), dim, 1, pos + 1 - @as(u32, @intCast(ratio)), true, false);
    if (index) {
        try m.r.text.call("rotate", m.stream, 1, 128, .{ m.r.p(.kv), @as(u32, 128) });
        try m.r.operators.model.qat(m.stream, m.r.p(.kv), 1, @intCast(dim), 32, true, @intCast(dim));
    } else try m.r.operators.model.qat(m.stream, m.r.p(.kv), 1, @intCast(dim - m.c.qk_rope_head_dim), 64, false, @intCast(dim));
    try writeRow(m, cache.cache, @as(usize, pos / @as(u32, @intCast(ratio))) * rowBytes(m, dim), m.r.p(.kv), dim);
}
const Score = struct { value: f32, index: i32 };
fn better(_: void, a: Score, b: Score) bool {
    return if (a.value == b.value) a.index < b.index else a.value > b.value;
}
pub fn attention(m: anytype, p: []const u8, l: *st.Layer, seq: *st.Sequence, pos: u32, ratio: usize, visible: ?@import("image_layout.zig").Span) !void {
    var b: [160]u8 = undefined;
    const dim = m.c.head_dim;
    try query(m, p, pos, ratio > 0);
    try keyValue(m, p, pos, ratio > 0);
    try writeRow(m, l.kv, @as(usize, pos % m.c.sliding_window) * rowBytes(m, dim), m.r.p(.kv), dim);
    const count = if (ratio > 0) (pos + 1) / ratio else 0;
    if (ratio > 0) {
        try compress(m, try key(&b, "{s}.attn.compressor", .{p}), &l.compressor.?, pos, ratio, dim, false);
        if (ratio == 4) try compress(m, try key(&b, "{s}.attn.indexer.compressor", .{p}), &l.indexer.?, pos, ratio, m.c.index_head_dim, true);
    }
    var indices: [4096]i32 = undefined;
    var selected: usize = 0;
    const causal_first = (@as(usize, pos) + 1) -| m.c.sliding_window;
    const first = if (visible) |span| @min(causal_first, span.start) else causal_first;
    const last = if (visible) |span| span.end else pos + 1;
    for (first..last) |i| {
        indices[selected] = @intCast(if (visible) |span| (if (i >= span.start and i < span.end) m.c.sliding_window + i - span.start else i % m.c.sliding_window) else i % m.c.sliding_window);
        selected += 1;
    }
    if (count > 0 and ratio == 4) {
        const ih = m.c.index_n_heads / 2;
        const id = m.c.index_head_dim;
        try project(m, try key(&b, "{s}.attn.indexer.wq_b", .{p}), m.r.p(.qr), m.r.p(.ixq), .bf16);
        try rope(m, m.r.p(.ixq), id, ih, pos, true, false);
        try m.r.text.call("rotate", m.stream, @intCast(ih), 128, .{ m.r.p(.ixq), @as(u32, 128) });
        try m.r.operators.model.qat(m.stream, m.r.p(.ixq), @intCast(ih), @intCast(id), 32, true, @intCast(id));
        try rawProject(m, try key(&b, "{s}.attn.indexer.weights_proj.weight", .{p}), m.r.p(.norm), m.r.p(.ixw), m.c.hidden_size, ih, false, .bf16);
        if (m.kv8) { // the index's key rows are e4m3 too: the same FP32 accumulation, read through each row's scale
            try m.r.text.call("index8", m.stream, @intCast((count + 127) / 128), 128, .{ m.r.p(.ixq), l.indexer.?.cache.ptr, m.r.p(.ixw), m.r.p(.score), @as(u32, @intCast(count)), @as(u32, @intCast(ih)), @as(u32, @intCast(id)), @as(u32, @intCast(rowBytes(m, id))) });
        } else try m.r.text.call("index", m.stream, @intCast((count + 127) / 128), 128, .{ m.r.p(.ixq), l.indexer.?.cache.ptr, m.r.p(.ixw), m.r.p(.score), @as(u32, @intCast(count)), @as(u32, @intCast(ih)), @as(u32, @intCast(id)) });
        try m.comm.allGather(m.r.p(.score), m.r.p(.gather), count, .f32, m.stream);
        try m.r.text.call("merge", m.stream, @intCast((count + 127) / 128), 128, .{ m.r.p(.gather), m.r.p(.score), @as(u32, @intCast(count)), @as(u32, 2) });
        try m.sync();
        const values = try m.a.alloc(f32, count);
        defer m.a.free(values);
        const scores = try m.a.alloc(Score, count);
        defer m.a.free(scores);
        try m.r.buf(.score).download(0, std.mem.sliceAsBytes(values));
        for (scores, values, 0..) |*v, s, i| v.* = .{ .value = s, .index = @intCast(i) };
        std.sort.heap(Score, scores, {}, better);
        for (scores[0..@min(count, m.c.index_topk)]) |s| {
            indices[selected] = @intCast(m.c.sliding_window + m.c.vision_max_n_token + @as(usize, @intCast(s.index)));
            selected += 1;
        }
    } else if (count > 0) for (0..count) |i| {
        indices[selected] = @intCast(m.c.sliding_window + m.c.vision_max_n_token + i);
        selected += 1;
    };
    try m.sync();
    try m.r.buf(.selected).upload(0, std.mem.sliceAsBytes(indices[0..selected]));
    try @import("sparse_rows.zig").run(m, seq.staging, l.kv.ptr,
        try m.w.ptr(try key(&b, "{s}.attn.attn_sink", .{p})), selected);
    try outputProjection(m, p, pos, ratio > 0);

}
pub fn query(m: anytype, p: []const u8, pos: u32, compressed: bool) !void {
    var b: [160]u8 = undefined;
    const heads = m.c.num_attention_heads / 2;
    const dim = m.c.head_dim;
    try project(m, try key(&b, "{s}.attn.wq_a", .{p}), m.r.p(.norm), m.r.p(.qr), .bf16);
    try norm(m, m.r.p(.qr), try m.w.ptr(try key(&b, "{s}.attn.q_norm.weight", .{p})), m.r.p(.qr), m.c.q_lora_rank, 1, false);
    try project(m, try key(&b, "{s}.attn.wq_b", .{p}), m.r.p(.qr), m.r.p(.q), .bf16);
    try norm(m, m.r.p(.q), 0, m.r.p(.q), dim, heads, true);
    try rope(m, m.r.p(.q), dim, heads, pos, compressed, false);
}
pub fn keyValue(m: anytype, p: []const u8, pos: u32, compressed: bool) !void {
    var b: [160]u8 = undefined;
    const dim = m.c.head_dim;
    try project(m, try key(&b, "{s}.attn.wkv", .{p}), m.r.p(.norm), m.r.p(.kv), .bf16);
    try norm(m, m.r.p(.kv), try m.w.ptr(try key(&b, "{s}.attn.kv_norm.weight", .{p})), m.r.p(.kv), dim, 1, false);
    try rope(m, m.r.p(.kv), dim, 1, pos, compressed, false);
    try m.r.operators.model.qat(m.stream, m.r.p(.kv), 1, @intCast(dim - m.c.qk_rope_head_dim), 64, false, @intCast(dim));
}
pub fn outputProjection(m: anytype, p: []const u8, pos: u32, compressed: bool) !void {
    var b: [160]u8 = undefined;
    const heads = m.c.num_attention_heads / 2;
    const dim = m.c.head_dim;
    try rope(m, m.r.p(.attn), dim, heads, pos, compressed, true);
    const groups = m.c.o_groups / 2;
    const k = heads / groups * dim;
    if (m.c.storage == .native_fp4_fp8) {
        const matrix = try m.w.matrix(try key(&b, "{s}.attn.wo_a", .{p}));
        try matrix.prepareDense(m.r.operators.native, m.stream);
        for (0..groups) |g| try m.r.operators.tensor.dense(m.stream, m.r.p(.attn) + g * k * 2, matrix.data.ptr + g * m.c.o_lora_rank * k * 2, 0, m.r.p(.grouped) + g * m.c.o_lora_rank * 2, 1, @intCast(k), @intCast(m.c.o_lora_rank), .bf16, .bf16, .bf16, false, 0);
    } else for (0..groups) |g| try project(m, try key(&b, "{s}.attn.wo_a.slice.{d}", .{ p, m.rank * groups + g }), m.r.p(.attn) + g * k * 2, m.r.p(.grouped) + g * m.c.o_lora_rank * 2, .bf16);
    try project(m, try key(&b, "{s}.attn.wo_b", .{p}), m.r.p(.grouped), m.r.p(.out), .f32);
    try reduce(m, m.r.p(.out), m.r.p(.out), m.c.hidden_size, true);
}
pub fn moe(m: anytype, p: []const u8, hashed: bool, image: bool) !void {
    var b: [160]u8 = undefined;
    try rawProject(m, try key(&b, "{s}.ffn.gate.weight", .{p}), m.r.p(.norm), m.r.p(.logits), m.c.hidden_size, 256, false, .f32);
    const table = if (hashed and !image) try m.w.ptr(try key(&b, "{s}.ffn.gate.tid2eid", .{p})) else 0;
    const bias = try m.w.ptr(try key(&b, "{s}.ffn.gate.{s}", .{p, if (image) "bias_vl" else "bias"}));
    try m.r.text.call("route", m.stream, 1, 1, .{ m.r.p(.logits), m.r.p(.ids), table, bias, m.r.p(.picks), m.r.p(.routeweights), @as(u32, @intFromBool(hashed and !image)), @as(f32, @floatCast(m.c.routed_scaling_factor)) });
    try m.sync();
    var picks: [6]u32 = undefined;
    try m.r.buf(.picks).download(0, std.mem.sliceAsBytes(&picks));
    try m.r.buf(.accum).fill8(0, m.stream.handle);
    for (0..7) |j| {
        const expert = if (j < 6) try key(&b, "{s}.ffn.experts.{d}", .{ p, picks[j] }) else try key(&b, "{s}.ffn.shared_experts", .{p});
        var w1: [160]u8 = undefined;
        var w2: [160]u8 = undefined;
        var w3: [160]u8 = undefined;
        try project(m, try key(&w1, "{s}.w1", .{expert}), m.r.p(.norm), m.r.p(.gate), .bf16);
        try project(m, try key(&w3, "{s}.w3", .{expert}), m.r.p(.norm), m.r.p(.up), .bf16);
        try m.r.operators.model.swiglu(m.stream, m.r.p(.gate), m.r.p(.up), if (j < 6) m.r.p(.routeweights) + j * 4 else 0, m.r.p(.act), 1, @intCast(m.c.moe_intermediate_size / 2), @floatCast(m.c.swiglu_limit));
        try project(m, try key(&w2, "{s}.w2", .{expert}), m.r.p(.act), m.r.p(.expert), .f32);
        try reduce(m, m.r.p(.expert), m.r.p(.expert), m.c.hidden_size, true);
        try m.r.text.call("accumulate", m.stream, @intCast((m.c.hidden_size + 127) / 128), 128, .{ m.r.p(.accum), m.r.p(.expert), @as(u32, @intCast(m.c.hidden_size)) });
    }
    try m.r.text.call("round", m.stream, @intCast((m.c.hidden_size + 127) / 128), 128, .{ m.r.p(.accum), m.r.p(.out), @as(u32, @intCast(m.c.hidden_size)) });
}
pub const Head = enum { full, partial, none };
pub fn step(m: anytype, seq: *st.Sequence, token: u32, sample: bool, logits: ?[]f32, embedded: u64) !void {
    try body(m, seq, token, embedded, if (sample) Head.full else Head.none, logits);
}
/// One row's reference program; `.partial` leaves the logits half in the logits buffer for the round's single exchange.
pub fn body(m: anytype, seq: *st.Sequence, token: u32, embedded: u64, head: Head, logits: ?[]f32) !void {
    // a Recorder defers the host-dependent stretches (re-run live each replay); device-only frames record as graphs
    return bodyRec(m, seq, token, embedded, head, logits, 0, null); // the row index is unread without a Recorder
}

/// The same program with a Recorder: four host stretches defer, the device-only frames become graph pieces.
pub fn bodyRec(m: anytype, seq: *st.Sequence, token: u32, embedded: u64, head: Head, logits: ?[]f32, at: u32, rec: ?*rg.Recorder) !void {
    const Reg = Region(@TypeOf(m));
    const pos = seq.len;
    if (pos >= m.context or (token >= m.c.vocab_size and embedded == 0)) return error.ContextOrTokenOutOfRange;
    if (rec) |r| {
        if (embedded != 0) try r.hostRead(); // a media row reached the graphed path: the plan lied, so the round goes eager
        try Reg.deferStretch(r, m, .{ .embed = .{ .at = at } });
    } else {
        try m.r.buf(.ids).upload(0, std.mem.asBytes(&token));
        if (embedded != 0) {
            try m.r.operators.tensor.run(.expand, m.stream, .{ .x = @intCast((m.c.hidden_size + 127) / 128) }, 128, .{embedded, m.r.p(.h), @as(u32, @intCast(m.c.hidden_size))});
        } else {
            const embed = try m.w.tensor("embed.weight");
            try m.r.operators.tensor.run(.embed, m.stream, .{ .x = @intCast((m.c.hidden_size + 127) / 128), .y = 1 }, 128, .{ embed.data.ptr, m.r.p(.ids), m.r.p(.h), @as(u32, @intCast(m.c.hidden_size)), @as(u32, @intCast(m.c.vocab_size)), @as(u32, @backingInt(try gpu.kind(embed.dtype))), @as(u32, 4) });
        }
    }
    for (seq.layers, 0..) |*l, i| {
        var b: [80]u8 = undefined;
        const p = try key(&b, "layers.{d}", .{i});
        try hc(m, p, "attn"); // device-only frame: its launches record into the open capture
        if (rec) |r| try Reg.deferStretch(r, m, .{ .attention = .{ .i = i, .at = at } }) else try attention(m, p, l, seq, pos, m.c.compress_ratios[i], null);
        try post(m);
        try hc(m, p, "ffn");
        if (rec) |r| try Reg.deferStretch(r, m, .{ .moe = .{ .i = i, .at = at, .hashed = i < m.c.num_hash_layers } }) else try moe(m, p, i < m.c.num_hash_layers, token >= m.c.vocab_size);
        try post(m);
        for (m.c.dspark_target_layer_ids, 0..) |layer, tap| if (layer == i) {
            try m.r.operators.tensor.run(.tap, m.stream, .{ .x = @intCast((m.c.hidden_size + 127) / 128) }, 128,
                .{ m.r.p(.h), seq.taps.ptr, @as(u32, @intCast(m.c.hidden_size)), @as(u32, @intCast(m.c.dspark_target_layer_ids.len)), @as(u32, @intCast(tap)) });
        };
    }
    if (head != .none) {
        try rawProject(m, "hc_head_fn", m.r.p(.h), m.r.p(.mixes), m.c.hidden_size * 4, 4, true, .f32);
        try m.r.operators.tensor.run(.collapse, m.stream, .{ .x = @intCast((m.c.hidden_size + 127) / 128), .y = 1 }, 128, .{ m.r.p(.h), m.r.p(.mixes), try m.w.ptr("hc_head_scale"), try m.w.ptr("hc_head_base"), m.r.p(.x), @as(u32, @intCast(m.c.hidden_size)), @as(f32, @floatCast(m.c.hc_eps)) });
        try norm(m, m.r.p(.x), try m.w.ptr("norm.weight"), m.r.p(.norm), m.c.hidden_size, 1, false);
        try project(m, "head", m.r.p(.norm), m.r.p(.logits), .f32);
        if (head == .full) {
            try m.comm.allGather(m.r.p(.logits), m.r.p(.gather), m.c.vocab_size / 2, .f32, m.stream);
            try m.sync();
            if (logits) |out| try m.r.buf(.gather).download(0, std.mem.sliceAsBytes(out));
        }
    }
    if (rec) |r| try Reg.deferStretch(r, m, .{ .absorb = .{ .at = at } }) else try @import("dspark.zig").absorb(m, seq, pos);
    if (rec == null) { // the graphed round's drain and advance belong to the step that launched it, not to a recording
        try m.sync();
        seq.len += 1;
    }
}
