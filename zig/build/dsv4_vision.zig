//! Portable checks for the DeepSeek-V4 Flash Vision model contract and operators.
const std = @import("std");

pub fn tests(b: *std.Build, core: *std.Build.Module, cuda: *std.Build.Module, all: *std.Build.Step) void {
    const family = b.createModule(.{
        .root_source_file = b.path("zig/src/families/dsv4_vision/dsv4_vision.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{ .{ .name = "core", .module = core }, .{ .name = "cuda", .module = cuda } },
    });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = family }));
    b.step("test-dsv4-vision", "DeepSeek-V4 Flash Vision host tests (no model weights or GPU)").dependOn(&run.step);
    all.dependOn(&run.step);
    const inspect = b.addExecutable(.{ .name = "tf-dsv4-inspect", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/families/dsv4_vision/inspect_main.zig"),
        .target = b.graph.host,
        .optimize = .safe,
        .link_libc = true,
        .imports = &.{.{ .name = "core", .module = core }},
    }) });
    const install = b.addInstallArtifact(inspect, .{});
    b.step("dsv4-inspect", "Build the offline DeepSeek-V4 Flash Vision checkpoint inspector").dependOn(&install.step);
    const image = b.addExecutable(.{ .name = "tf-dsv4-image", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/families/dsv4_vision/image_main.zig"),
        .target = b.graph.host,
        .optimize = .safe,
        .link_libc = true,
        .imports = &.{.{ .name = "core", .module = core }},
    }) });
    const install_image = b.addInstallArtifact(image, .{});
    if (b.graph.host.result.os.tag == .windows) image.root_module.linkSystemLibrary("kernel32", .{});
    b.step("dsv4-image", "Build the native image preprocessing oracle runner").dependOn(&install_image.step);
}

pub fn nativeTargets(b: *std.Build, target: std.Build.ResolvedTarget, core: *std.Build.Module, cuda: *std.Build.Module) void {
    const project = b.addExecutable(.{ .name = "tf-dsv4-project", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/families/dsv4_vision/project_main.zig"),
        .target = target,
        .optimize = .safe,
        .link_libc = true,
        .imports = &.{ .{ .name = "core", .module = core }, .{ .name = "cuda", .module = cuda } },
    }) });
    const install = b.addInstallArtifact(project, .{});
    b.step("dsv4-project", "Build a native real-checkpoint CUDA projection probe").dependOn(&install.step);
}
