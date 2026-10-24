//! Checkpoint DSpark stages propose a noncausal block; the lane core alone accepts target draws.
const std = @import("std");
const cuda = @import("cuda");
const fw = @import("text_forward.zig");
const st = @import("text_state.zig");
const Kind = cuda.dsv4_tensor.Kind;

const StageKeys = struct {
    prefix_buf: [32]u8 = undefined,
    sink_buf: [96]u8 = undefined,
    prefix_len: usize = 0,
    sink_len: usize = 0,

    fn init(keys: *StageKeys, stage: usize) !void {
        keys.prefix_len = (try fw.key(&keys.prefix_buf, "mtp.{d}", .{stage})).len;
        keys.sink_len = (try fw.key(&keys.sink_buf, "{s}.attn.attn_sink", .{keys.prefix()})).len;
    }

    fn prefix(keys: *const StageKeys) []const u8 {
        return keys.prefix_buf[0..keys.prefix_len];
    }

    fn sink(keys: *const StageKeys) []const u8 {
        return keys.sink_buf[0..keys.sink_len];
    }
};

test "DSpark attention sink formatting preserves the stage prefix for every proposal row" {
    var keys: StageKeys = .{};
    try keys.init(0);
    try std.testing.expectEqualStrings("mtp.0", keys.prefix());
    try std.testing.expectEqualStrings("mtp.0.attn.attn_sink", keys.sink());
    try keys.init(12);
    try std.testing.expectEqualStrings("mtp.12", keys.prefix());
    try std.testing.expectEqualStrings("mtp.12.attn.attn_sink", keys.sink());
}

pub fn absorb(m: anytype, seq: *st.Sequence, pos: u32) !void {
    try fw.project(m, "mtp.0.main_proj", seq.taps.ptr, m.r.p(.x), .bf16);
    try fw.norm(m, m.r.p(.x), try m.w.ptr("mtp.0.main_norm.weight"), seq.main.ptr, m.c.hidden_size, 1, false);
    for (seq.draft, 0..) |*layer, i| {
        var b: [80]u8 = undefined;
        const p = try fw.key(&b, "mtp.{d}", .{i});
        try m.r.buf(.norm).copyFrom(0, seq.main.ptr, m.c.hidden_size * 2, m.stream.handle);
        try fw.keyValue(m, p, pos, false);
        try fw.writeRow(m, layer.kv, @as(usize, pos % m.c.sliding_window) * fw.rowBytes(m, m.c.head_dim), m.r.p(.kv), m.c.head_dim);
    }
}

pub fn head(m: anytype, p: []const u8, hidden: u64, logits: []f32) !void {
    var b: [128]u8 = undefined;
    try fw.rawProject(m, try fw.key(&b, "{s}hc_head_fn", .{p}), m.r.p(.h), m.r.p(.mixes), m.c.hidden_size * 4, 4, true, .f32);
    const scale = try m.w.ptr(try fw.key(&b, "{s}hc_head_scale", .{p}));
    const base = try m.w.ptr(try fw.key(&b, "{s}hc_head_base", .{p}));
    try m.r.operators.tensor.run(.collapse, m.stream, .{ .x = @intCast((m.c.hidden_size + 127) / 128) }, 128,
        .{ m.r.p(.h), m.r.p(.mixes), scale, base, hidden, @as(u32, @intCast(m.c.hidden_size)), @as(f32, @floatCast(m.c.hc_eps)) });
    try fw.norm(m, hidden, try m.w.ptr(try fw.key(&b, "{s}norm.weight", .{p})), m.r.p(.norm), m.c.hidden_size, 1, false);
    try fw.project(m, "head", m.r.p(.norm), m.r.p(.logits), .f32);
    try m.comm.allGather(m.r.p(.logits), m.r.p(.gather), m.c.vocab_size / 2, .f32, m.stream);
    try m.sync();
    try m.r.buf(.gather).download(0, std.mem.sliceAsBytes(logits));
}

pub fn propose(m: anytype, seq: *st.Sequence, token: u32, out: []u32) !void {
    const size = m.c.dspark_block_size;
    if (seq.len == 0 or out.len > size or token >= m.c.vocab_size) return error.InvalidDraft;
    if (out.len == 0) return;
    const hbytes = m.c.hidden_size * 4 * 2;
    const qbytes = m.c.num_attention_heads / 2 * m.c.head_dim * 2;
    const kvbytes = m.c.head_dim * 2;
    const d = m.d;
    var block = try cuda.DeviceBuffer.alloc(d, size * hbytes);
    defer block.free();
    var queries = try cuda.DeviceBuffer.alloc(d, size * qbytes);
    defer queries.free();
    var keys = try cuda.DeviceBuffer.alloc(d, size * kvbytes);
    defer keys.free();
    var hidden = try cuda.DeviceBuffer.alloc(d, size * m.c.hidden_size * 2);
    defer hidden.free();
    defer m.sync() catch {};
    const embed = try m.w.tensor("embed.weight");
    const ids = try m.a.alloc(u32, size);
    defer m.a.free(ids);
    @memset(ids, @intCast(m.c.dspark_noise_token_id));
    ids[0] = token;
    try m.r.buf(.ids).upload(0, std.mem.sliceAsBytes(ids));
    try m.r.operators.tensor.run(.embed, m.stream, .{ .x = @intCast((m.c.hidden_size + 127) / 128), .y = @intCast(size) }, 128,
        .{ embed.data.ptr, m.r.p(.ids), block.ptr, @as(u32, @intCast(m.c.hidden_size)), @as(u32, @intCast(m.c.vocab_size)), @as(u32, @backingInt(try @import("gpu_weights.zig").kind(embed.dtype))), @as(u32, 4) });
    var selected: [512]i32 = undefined;
    const past = @min(seq.len, m.c.sliding_window);
    if (past + size > selected.len) return error.DraftGeometry;
    for (0..past) |i| selected[i] = @intCast(i);
    for (0..size) |i| selected[past + i] = @intCast(m.c.sliding_window + i);
    try m.sync();
    try m.r.buf(.selected).upload(0, std.mem.sliceAsBytes(selected[0 .. past + size]));
    for (seq.draft, 0..) |*layer, stage| {
        var names: StageKeys = .{};
        try names.init(stage);
        const p = names.prefix();
        const sink = try m.w.ptr(names.sink());
        for (0..size) |i| {
            try m.r.buf(.h).copyFrom(0, block.ptr + i * hbytes, hbytes, m.stream.handle);
            try fw.hc(m, p, "attn");
            try fw.query(m, p, seq.len + @as(u32, @intCast(i)), false);
            try fw.keyValue(m, p, seq.len + @as(u32, @intCast(i)), false);
            try queries.copyFrom(i * qbytes, m.r.p(.q), qbytes, m.stream.handle);
            try keys.copyFrom(i * kvbytes, m.r.p(.kv), kvbytes, m.stream.handle);
        }
        if (m.kv8) { // the block's rows join the draft cache as e4m3: one q8 a row, from the BF16 scratch the queries drew into
            for (0..size) |i| try m.r.text.call("q8", m.stream, 1, @intCast(m.c.head_dim), .{ keys.ptr + i * kvbytes, layer.kv.ptr + (m.c.sliding_window + i) * fw.rowBytes(m, m.c.head_dim), @as(u32, @intCast(m.c.head_dim)) });
        } else try layer.kv.copyFrom(m.c.sliding_window * kvbytes, keys.ptr, size * kvbytes, m.stream.handle);
        for (0..size) |i| {
            try m.r.buf(.h).copyFrom(0, block.ptr + i * hbytes, hbytes, m.stream.handle);
            try fw.hc(m, p, "attn");
            try m.r.buf(.q).copyFrom(0, queries.ptr + i * qbytes, qbytes, m.stream.handle);
            try @import("sparse_rows.zig").run(m, seq.staging, layer.kv.ptr, sink, past + size);
            try fw.outputProjection(m, p, seq.len + @as(u32, @intCast(i)), false);
            try fw.post(m);
            try fw.hc(m, p, "ffn");
            try fw.moe(m, p, false, false);
            try fw.post(m);
            try block.copyFrom(i * hbytes, m.r.p(.h), hbytes, m.stream.handle);
        }
    }
    const logits = try m.a.alloc(f32, size * m.c.vocab_size);
    defer m.a.free(logits);
    var prefix: [32]u8 = undefined;
    const last = try fw.key(&prefix, "mtp.{d}.", .{m.c.num_nextn_predict_layers - 1});
    for (0..size) |i| {
        try m.r.buf(.h).copyFrom(0, block.ptr + i * hbytes, hbytes, m.stream.handle);
        try head(m, last, hidden.ptr + i * m.c.hidden_size * 2, logits[i * m.c.vocab_size ..][0..m.c.vocab_size]);
    }
    var previous = token;
    for (0..out.len) |i| {
        var b: [128]u8 = undefined;
        const table = try m.w.matrix(try fw.key(&b, "{s}markov_head.markov_w1", .{last}));
        if (table.trellis_params != null or table.native_kind != null) return error.UnsupportedMarkovEmbedding;
        try m.r.buf(.ids).upload(0, std.mem.asBytes(&previous));
        try m.r.operators.tensor.run(.embed, m.stream, .{ .x = @intCast((m.c.dspark_markov_rank + 127) / 128) }, 128,
            .{ table.data.ptr, m.r.p(.ids), m.r.p(.gate), @as(u32, @intCast(m.c.dspark_markov_rank)), @as(u32, @intCast(m.c.vocab_size)), @as(u32, @backingInt(table.dtype)), @as(u32, 1) });
        try fw.project(m, try fw.key(&b, "{s}markov_head.markov_w2", .{last}), m.r.p(.gate), m.r.p(.logits), .f32);
        try m.comm.allGather(m.r.p(.logits), m.r.p(.gather), m.c.vocab_size / 2, .f32, m.stream);
        try m.sync();
        const bias = try m.a.alloc(f32, m.c.vocab_size);
        defer m.a.free(bias);
        try m.r.buf(.gather).download(0, std.mem.sliceAsBytes(bias));
        const row = logits[i * m.c.vocab_size ..][0..m.c.vocab_size];
        for (row, bias) |*v, value| v.* += value;
        // Proposal sampling is greedy; acceptance uses the target's keyed sampling on the lane core.
        var best: usize = 0;
        for (row, 0..) |v, j| if (v > row[best]) { best = j; };
        out[i] = @intCast(best);
        previous = out[i];
        try m.r.text.call("cast", m.stream, @intCast((m.c.hidden_size + 127) / 128), 128,
            .{ hidden.ptr + i * m.c.hidden_size * 2, m.r.p(.tmp), @as(u32, @intCast(m.c.hidden_size)), @as(u32, @backingInt(Kind.bf16)) });
        try m.r.text.call("cast", m.stream, @intCast((m.c.dspark_markov_rank + 127) / 128), 128,
            .{ m.r.p(.gate), m.r.p(.tmp) + m.c.hidden_size * 4, @as(u32, @intCast(m.c.dspark_markov_rank)), @as(u32, @backingInt(Kind.bf16)) });
        const confidence = try m.w.matrix(try fw.key(&b, "{s}confidence_head.proj", .{last}));
        try confidence.run(m.r.operators.exl3, m.r.operators.native, m.r.operators.tensor, m.stream, &m.r.scratch, m.r.p(.tmp), m.r.p(.expert), 1, .f32, .f32);
        try m.sync();
        try m.r.buf(.expert).download(0, std.mem.asBytes(&m.confidence[i]));
    }
}
