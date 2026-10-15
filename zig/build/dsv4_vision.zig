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
    const project = b.addExecutable(.{ .name = "tf-dsv4-project-host", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/families/dsv4_vision/project_main.zig"),
        .target = b.graph.host,
        .optimize = .safe,
        .link_libc = true,
        .imports = &.{ .{ .name = "core", .module = core }, .{ .name = "cuda", .module = cuda } },
    }) });
    if (b.graph.host.result.os.tag == .windows) project.root_module.linkSystemLibrary("kernel32", .{});
    const install_project = b.addInstallArtifact(project, .{});
    b.step("dsv4-project-host", "Build a host CUDA projection probe for device development").dependOn(&install_project.step);
    const vision = visionExecutable(b, b.graph.host, core, cuda, "tf-dsv4-vision-host");
    b.step("dsv4-vision-host", "Build the host native ViT and aligner probe").dependOn(&b.addInstallArtifact(vision, .{}).step);
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
    const contracts = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/families/dsv4_vision/dsv4_vision.zig"),
        .target = target,
        .optimize = .safe,
        .link_libc = true,
        .imports = &.{ .{ .name = "core", .module = core }, .{ .name = "cuda", .module = cuda } },
    }) });
    b.step("check-dsv4-target", "Compile the model storage and cache contract tests for the target").dependOn(&contracts.step);
    const vision = visionExecutable(b, target, core, cuda, "tf-dsv4-vision");
    b.step("dsv4-vision", "Build the native ViT and aligner probe").dependOn(&b.addInstallArtifact(vision, .{}).step);
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

fn visionExecutable(b: *std.Build, target: std.Build.ResolvedTarget, core: *std.Build.Module, cuda: *std.Build.Module, name: []const u8) *std.Build.Step.Compile {
    const exe = b.addExecutable(.{ .name = name, .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/families/dsv4_vision/vision_main.zig"),
        .target = target,
        .optimize = .safe,
        .link_libc = true,
        .imports = &.{ .{ .name = "core", .module = core }, .{ .name = "cuda", .module = cuda } },
    }) });
    if (target.result.os.tag == .windows) exe.root_module.linkSystemLibrary("kernel32", .{});
    return exe;
}
