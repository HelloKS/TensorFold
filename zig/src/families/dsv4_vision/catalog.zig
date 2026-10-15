//! Bounded safetensors-header reads; the model stays on disk until individual tensors are requested.
const std = @import("std");
const st = @import("core").safetensors;
const weights = @import("weights.zig");
const Config = @import("config.zig").Config;
const Io = std.Io;

pub const Location = struct { path: []const u8, data: u64 };
pub const Catalog = struct {
    headers: st.Header = .empty,
    locations: std.StringHashMapUnmanaged(Location) = .empty,
    stored_bytes: u64 = 0,

    /// All returned names and entries live in the supplied arena.
    pub fn open(a: std.mem.Allocator, io: Io, dir: []const u8) !Catalog {
        var out: Catalog = .{};
        const index_path = try std.fs.path.join(a, &.{ dir, "model.safetensors.index.json" });
        const index_text = Io.Dir.cwd().readFileAlloc(io, index_path, a, .limited(64 << 20)) catch |e|
            if (e == error.FileNotFound) null else return e;
        var names: std.StringArrayHashMapUnmanaged(void) = .empty;
        var map: ?std.json.ObjectMap = null;
        if (index_text) |text| {
            const p = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
            if (p != .object) return error.BadIndex;
            const v = p.object.get("weight_map") orelse return error.BadIndex;
            if (v != .object or v.object.count() == 0) return error.BadIndex;
            map = v.object;
            for (v.object.values()) |name| {
                if (name != .string or !safeName(name.string)) return error.BadShardName;
                try names.put(a, name.string, {});
            }
        } else {
            var directory = try Io.Dir.cwd().openDir(io, dir, .{ .iterate = true });
            defer directory.close(io);
            var it = directory.iterate();
            while (try it.next(io)) |file| {
                if (file.kind != .file and file.kind != .sym_link) continue;
                if (safeName(file.name) and std.mem.startsWith(u8, file.name, "model"))
                    try names.put(a, try a.dupe(u8, file.name), {});
            }
        }
        if (names.count() == 0) return error.NoSafetensors;
        for (names.keys()) |name| {
            const path = try std.fs.path.join(a, &.{ dir, name });
            const previous = out.headers.count();
            try out.addFile(a, io, path);
            for (out.headers.keys()[previous..]) |key| {
                if (map) |m| {
                    const declared = m.get(key) orelse return error.UnindexedTensor;
                    if (declared != .string or !std.mem.eql(u8, declared.string, name)) return error.IndexShardMismatch;
                }
            }
        }
        if (map) |m| if (m.count() != out.headers.count()) return error.IndexTensorMissing;
        return out;
    }

    pub fn addFile(out: *Catalog, a: std.mem.Allocator, io: Io, path: []const u8) !void {
        var file = try Io.Dir.cwd().openFile(io, path, .{});
        defer file.close(io);
        const len = try file.length(io);
        var prefix: [8]u8 = undefined;
        if (try file.readPositionalAll(io, &prefix, 0) != 8) return error.BadSafetensors;
        const n = std.mem.readInt(u64, &prefix, .little);
        if (n > 64 << 20 or len < 8 or n > len - 8) return error.BadSafetensors;
        const json = try a.alloc(u8, @intCast(n));
        if (try file.readPositionalAll(io, json, 8) != n) return error.BadSafetensors;
        const header = try st.parseHeader(a, json, @intCast(len - 8 - n));
        const owned_path = try a.dupe(u8, path);
        for (header.keys(), header.values()) |key, tensor| {
            if (out.headers.contains(key)) return error.DuplicateTensor;
            try out.headers.put(a, key, tensor);
            try out.locations.put(a, key, .{ .path = owned_path, .data = n + 8 });
        }
        out.stored_bytes = try std.math.add(u64, out.stored_bytes, len);
    }

    /// Payload ownership transfers to the caller; no process-global weight mapping is kept.
    pub fn read(c: *const Catalog, a: std.mem.Allocator, io: Io, name: []const u8) !st.Tensor {
        const entry = c.headers.get(name) orelse return error.MissingTensor;
        const location = c.locations.get(name) orelse return error.MissingTensor;
        var file = try Io.Dir.cwd().openFile(io, location.path, .{});
        defer file.close(io);
        const data = try a.alloc(u8, entry.end - entry.begin);
        errdefer a.free(data);
        if (try file.readPositionalAll(io, data, location.data + entry.begin) != data.len) return error.TruncatedTensor;
        return .{ .dtype = entry.dtype, .rank = entry.rank, .shape = entry.shape, .bytes = data };
    }

    pub fn validate(c: Catalog, a: std.mem.Allocator, config: Config, problem: *[]const u8) !weights.Summary {
        return weights.validate(a, config, c.headers, problem);
    }
};

pub fn safeName(name: []const u8) bool {
    return name.len != 0 and std.mem.endsWith(u8, name, ".safetensors") and
        std.mem.indexOfAny(u8, name, "/\\:") == null and !std.mem.startsWith(u8, name, ".");
}

test "a checkpoint index cannot escape its directory" {
    for ([_][]const u8{ "../model.safetensors", "C:\\model.safetensors", "/model.safetensors", "x/model.safetensors" }) |n|
        try std.testing.expect(!safeName(n));
    try std.testing.expect(safeName("model-00001-of-00016.safetensors"));
}

test "header reader rejects missing indexed tensors and reads payloads on demand" {
    const io = std.testing.io;
    var temp = std.testing.tmpDir(.{});
    defer temp.cleanup();
    const header = "{\"image_start\":{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[0,4]}}";
    var bytes: [8 + header.len + 4]u8 = @splat(0);
    std.mem.writeInt(u64, bytes[0..8], header.len, .little);
    @memcpy(bytes[8..][0..header.len], header);
    try temp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = &bytes });
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{temp.sub_path});
    const catalog = try Catalog.open(a, io, path);
    const tensor = try catalog.read(a, io, "image_start");
    try std.testing.expectEqual(@as(usize, 4), tensor.bytes.len);
    try temp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"weight_map\":{\"image_start\":\"model.safetensors\",\"missing\":\"model.safetensors\"}}" });
    try std.testing.expectError(error.IndexTensorMissing, Catalog.open(a, io, path));
}
