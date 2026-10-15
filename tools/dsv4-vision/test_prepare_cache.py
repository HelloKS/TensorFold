"""Verify that TP slices preserve literal stored words and do not dequantize or mix Hadamard blocks."""
import unittest
import json
import struct
import tempfile
from pathlib import Path

from prepare_cache import ranges, safe_shard, selection, catalog, write_rank, sha256


class LosslessSlices(unittest.TestCase):
    def test_written_rank_is_valid_safetensors_with_literal_source_bytes(self):
        scratch = Path.cwd() / ".zig-cache"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="tf-dsv4-rank-test-", dir=scratch) as temp:
            root = Path(temp)
            header = json.dumps({"norm.weight": {"dtype": "F32", "shape": [8], "data_offsets": [0, 32]}}).encode()
            payload = bytes(range(32))
            (root / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + payload)
            (root / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {"norm.weight": "model.safetensors"}}))
            part = {"keep": True, "axis": 0, "first": 4, "count": 4}
            plan = {"items": [{"spec": {"key": "norm.weight", "kind": "float"}, "ranks": [part, part]}]}
            path = root / "rank0.safetensors"
            result = write_rank(root, path, selection(catalog(root), plan, 0))
            raw = path.read_bytes()
            size, = struct.unpack("<Q", raw[:8])
            output_header = json.loads(raw[8:8 + size])
            self.assertEqual(output_header["norm.weight"]["shape"], [4])
            self.assertEqual(raw[8 + size:], payload[16:])
            self.assertEqual(result["sha256"], sha256(path))

    def test_column_slice_selects_whole_trellis_tiles(self):
        spans, shape = ranges([8, 16, 64], 2, 1, 8, 8)
        self.assertEqual(shape, [8, 8, 64])
        words = bytes(range(256)) * 64
        selected = b"".join(words[off:off + n] for off, n in spans)
        expected = b"".join(words[(r * 16 + 8) * 128:(r + 1) * 16 * 128] for r in range(8))
        self.assertEqual(selected, expected)

    def test_matrix_input_split_slices_suh_and_preserves_svh(self):
        catalog = {
            "w.trellis": {"dtype": "I16", "shape": [16, 8, 64]},
            "w.suh": {"dtype": "F16", "shape": [256]},
            "w.svh": {"dtype": "F16", "shape": [128]},
            "w.mul1": {"dtype": "I32", "shape": []},
        }
        part = {"keep": True, "axis": 0, "first": 128, "count": 128}
        plan = {"items": [{"spec": {"kind": "matrix", "key": "w"}, "ranks": [part, part]}]}
        chosen = selection(catalog, plan, 0)
        self.assertEqual(chosen["w.trellis"]["shape"], [8, 8, 64])
        self.assertEqual(chosen["w.suh"]["spans"], [(256, 256)])
        self.assertEqual(chosen["w.svh"]["spans"], [(0, 256)])
        self.assertEqual(chosen["w.mul1"]["spans"], [(0, 4)])

    def test_rank_one_does_not_store_vision_weights(self):
        plan = {"items": [{"spec": {"key": "vision.norm.weight", "kind": "float"},
                            "ranks": [{"keep": True}, {"keep": False}]}]}
        self.assertEqual(selection({}, plan, 1), {})

    def test_shard_names_are_not_paths(self):
        self.assertTrue(safe_shard("model-00001-of-00016.safetensors"))
        for name in ["../model.safetensors", "x/model.safetensors", "C:\\model.safetensors", ""]:
            self.assertFalse(safe_shard(name))


if __name__ == "__main__":
    unittest.main()
