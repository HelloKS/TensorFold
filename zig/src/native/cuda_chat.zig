//! A family renders a templateless checkpoint's chat: JSON texts in, rendered text and spans out; std types only.
const std = @import("std");
const Allocator = std.mem.Allocator;

/// JSON texts in (empty tools is none), rendered text and spans out by out params; every type here is std's.
const RenderFn = *const fn (Allocator, []const u8, []const u8, bool, bool, ?[]const u8, *[]const u8, *[]u8, *[]const [2]usize) anyerror!void;

/// The chat probe of this CUDA family registry: the one family naming `model_type` and declaring `chat`.
pub fn forRegistry(comptime registry: anytype) type {
    return struct {
        pub fn chatEncoder(model_type: []const u8) ?struct { source: []const u8, render: RenderFn } {
            inline for (registry) |F| if (@hasDecl(F, "model_type") and @hasDecl(F, "chat") and std.mem.eql(u8, F.model_type, model_type))
                return .{ .source = F.chat.source, .render = F.chat.render };
            return null;
        }
    };
}

fn render(a: Allocator, messages: []const u8, tools: []const u8, add_generation_prompt: bool, thinking: bool, effort: ?[]const u8, problem: *[]const u8, text: *[]u8, spans: *[]const [2]usize) anyerror!void {
    _ = a;
    _ = tools;
    _ = add_generation_prompt;
    _ = thinking;
    _ = effort;
    _ = problem;
    if (messages.len == 0) return error.InvalidMessages;
    text.* = @constCast("rendered");
    spans.* = &.{.{ 0, 3 }};
}

test "the chat probe answers only the family that declares one, by its own model_type" {
    const With = struct {
        pub const model_type = "with_chat";
        pub const chat = .{ .source = "the family's template", .render = &render };
    };
    const Without = struct {
        pub const model_type = "without_chat";
    };
    const probe = forRegistry(.{ With, Without });
    try std.testing.expectEqualStrings("the family's template", (probe.chatEncoder("with_chat") orelse return error.MissingChat).source);
    try std.testing.expect(probe.chatEncoder("without_chat") == null);
    try std.testing.expect(probe.chatEncoder("no_such_family") == null);
}
