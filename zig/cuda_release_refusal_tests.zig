//! The actual release admission refuses unavailable DeepSeek backends independently of the environment.
const std = @import("std");
const gate = @import("release_refusal");
const env = @import("policy_env");
const io = std.testing.io;
const expected = "DeepSeek-V4.1 is not served by this release's CUDA server; its whole-model gate is #827";
const experimental = "DeepSeek-V4.1 CUDA is experimental in 1.1.1 while whole-model gate #827 is pending; set TF_EXPERIMENTAL=deepseek to opt in";

/// A readable checkpoint dir; `data` is its config.json, so the gate reads the fingerprint it fails closed on.
fn checkpoint(config: []const u8) !struct { tmp: std.testing.TmpDir, path: [:0]u8 } {
    var tmp = std.testing.tmpDir(.{});
    errdefer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = config });
    return .{ .tmp = tmp, .path = try tmp.dir.realPathFileAlloc(io, ".", std.testing.allocator) };
}

const vision_config = "{\"model_type\":\"deepseek_v4\",\"vision_n_layers\":32,\"vision_patch_size\":14}";
const text_config = "{\"model_type\":\"deepseek_v4\"}";

fn check(value: ?[:0]const u8) !void {
    env.value = value;
    defer env.value = null;
    // DeepSeek-V4.1 is refused whatever the environment says and whatever its checkpoint claims.
    for ([_][]const u8{ "not-opened", "another-unread-checkpoint" }) |dir| {
        try std.testing.expectEqualStrings(expected, gate.backend(io, .{ .dir = dir, .model_type = "deepseek_v41" }) orelse "");
        try std.testing.expectEqualStrings(expected, gate.backend(io, .{ .dir = dir, .model_type = "deepseek_v41", .policy = "unsupported", .mtp_drafts = 32 }) orelse "");
    }
    // DeepSeek-V4 without the opt-in is refused with the message that names the opt-in.
    for ([_][]const u8{ "not-opened", "another-unread-checkpoint" }) |dir| {
        try std.testing.expectEqualStrings(experimental, gate.backend(io, .{ .dir = dir, .model_type = "deepseek_v4" }) orelse "");
        try std.testing.expectEqualStrings(experimental, gate.backend(io, .{ .dir = dir, .model_type = "deepseek_v4", .policy = "unsupported", .mtp_drafts = 32 }) orelse "");
    }
}

test "DeepSeek release: TF_EXPERIMENTAL unset is a constant refusal before device work" {
    try check(null);
    // the opt-in is required even for a readable checkpoint whose config names the vision tower
    var v = try checkpoint(vision_config);
    defer std.testing.allocator.free(v.path);
    defer v.tmp.cleanup();
    try std.testing.expectEqualStrings(experimental, gate.backend(io, .{ .dir = v.path, .model_type = "deepseek_v4" }) orelse "");
}

test "DeepSeek release: TF_EXPERIMENTAL=1 is the same constant refusal" {
    try check("1");
}

test "DeepSeek release: TF_EXPERIMENTAL=deepseek keeps V4.1 and every fail-closed V4 refused" {
    env.value = "deepseek";
    defer env.value = null;
    // DeepSeek-V4.1 is refused under the opt-in too, and a readable config naming the vision tower changes that nothing
    var v = try checkpoint(vision_config);
    defer std.testing.allocator.free(v.path);
    defer v.tmp.cleanup();
    try std.testing.expectEqualStrings(expected, gate.backend(io, .{ .dir = v.path, .model_type = "deepseek_v41" }) orelse "");
    // an unreadable checkpoint gives the gate no fingerprint to open on
    try std.testing.expectEqualStrings(expected, gate.backend(io, .{ .dir = "not-opened", .model_type = "deepseek_v4" }) orelse "");
    try std.testing.expectEqualStrings(expected, gate.backend(io, .{ .dir = "another-unread-checkpoint", .model_type = "deepseek_v4", .policy = "unsupported", .mtp_drafts = 32 }) orelse "");
    // a readable checkpoint that names no vision tower is the gated V4.1 text shape
    var t = try checkpoint(text_config);
    defer std.testing.allocator.free(t.path);
    defer t.tmp.cleanup();
    try std.testing.expectEqualStrings(expected, gate.backend(io, .{ .dir = t.path, .model_type = "deepseek_v4" }) orelse "");
}

test "DeepSeek release: TF_EXPERIMENTAL=deepseek admits the vision-exp checkpoint by its own config" {
    env.value = "deepseek";
    defer env.value = null;
    var v = try checkpoint(vision_config);
    defer std.testing.allocator.free(v.path);
    defer v.tmp.cleanup();
    try std.testing.expect(gate.backend(io, .{ .dir = v.path, .model_type = "deepseek_v4" }) == null);
}

test "DeepSeek release: other CUDA admission stays available" {
    env.value = "deepseek";
    defer env.value = null;
    try std.testing.expect(gate.backend(io, .{ .dir = "not-opened", .model_type = "nemotron_h" }) == null);
}
