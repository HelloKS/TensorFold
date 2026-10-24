//! Image spans prefill by layer so each image query can see every row of the same image.
const std = @import("std");
const cuda = @import("cuda");
const fw = @import("text_forward.zig");
const state = @import("text_state.zig");
const Span = @import("vision_serving.zig").Span;
pub fn run(m: anytype, seq: *state.Sequence, span: Span, rows: anytype, logits: ?[]f32) !void {
    const count = span.end - span.start;
    if (seq.len != span.start or rows.len != count or count > m.c.vision_max_n_token) return error.ChunkCutsImage;
    const hbytes = m.c.hidden_size * 4 * 2;
    const taps_bytes = m.c.hidden_size * m.c.dspark_target_layer_ids.len * 2;
    var hidden = try cuda.DeviceBuffer.alloc(m.d, count * hbytes);
    defer hidden.free();
    var taps = try cuda.DeviceBuffer.alloc(m.d, count * taps_bytes);
    defer taps.free();
    defer m.sync() catch {};
    const embedding = span.embeddings.ptr + @as(usize, span.start - span.at) * m.c.hidden_size * 2;
    try m.r.operators.tensor.run(.expand, m.stream, .{ .x = @intCast((m.c.hidden_size + 127) / 128), .y = count }, 128,
        .{ embedding, hidden.ptr, @as(u32, @intCast(m.c.hidden_size)) });
    for (seq.layers, 0..) |*layer, i| {
        var b: [80]u8 = undefined;
        const p = try fw.key(&b, "layers.{d}", .{i});
        for (0..count) |j| {
            try m.r.buf(.h).copyFrom(0, hidden.ptr + j * hbytes, hbytes, m.stream.handle);
            try fw.hc(m, p, "attn");
            try fw.keyValue(m, p, span.start + @as(u32, @intCast(j)), m.c.compress_ratios[i] > 0);
            try fw.writeRow(m, layer.kv, (m.c.sliding_window + j) * fw.rowBytes(m, m.c.head_dim), m.r.p(.kv), m.c.head_dim);
        }
        for (rows, 0..) |row, j| {
            if (row.index != span.start + j or row.token < m.c.vocab_size or (row.sample and j + 1 != count)) return error.InvalidImageSpan;
            try m.r.buf(.h).copyFrom(0, hidden.ptr + j * hbytes, hbytes, m.stream.handle);
            try fw.hc(m, p, "attn");
            try fw.attention(m, p, layer, seq, row.index, m.c.compress_ratios[i], .{ .start = span.start, .end = span.end });
            try fw.post(m);
            try fw.hc(m, p, "ffn");
            try fw.moe(m, p, false, true);
            try fw.post(m);
            try hidden.copyFrom(j * hbytes, m.r.p(.h), hbytes, m.stream.handle);
        }
        for (m.c.dspark_target_layer_ids, 0..) |target, tap| if (target == i) {
            try m.r.operators.tensor.run(.tap, m.stream, .{ .x = @intCast((m.c.hidden_size + 127) / 128), .y = count }, 128,
                .{ hidden.ptr, taps.ptr, @as(u32, @intCast(m.c.hidden_size)), @as(u32, @intCast(m.c.dspark_target_layer_ids.len)), @as(u32, @intCast(tap)) });
        };
    }
    for (0..count) |j| {
        try seq.taps.copyFrom(0, taps.ptr + j * taps_bytes, taps_bytes, m.stream.handle);
        try @import("dspark.zig").absorb(m, seq, span.start + @as(u32, @intCast(j)));
    }
    if (rows[rows.len - 1].sample) {
        try m.r.buf(.h).copyFrom(0, hidden.ptr + (count - 1) * hbytes, hbytes, m.stream.handle);
        const scratch = try m.a.alloc(f32, m.c.vocab_size);
        defer m.a.free(scratch);
        try @import("dspark.zig").head(m, "", m.r.p(.x), if (logits) |out| out else scratch);
    }
    try m.sync();
    seq.len = span.end;
}
