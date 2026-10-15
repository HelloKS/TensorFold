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
        self.assertEqual(list(chosen["w.suh"]["spans"]), [(256, 256)])
        self.assertEqual(list(chosen["w.svh"]["spans"]), [(0, 256)])
        self.assertEqual(list(chosen["w.mul1"]["spans"]), [(0, 4)])

    def test_rank_one_does_not_store_vision_weights(self):
        plan = {"items": [{"spec": {"key": "vision.norm.weight", "kind": "float"},
                            "ranks": [{"keep": True}, {"keep": False}]}]}
        self.assertEqual(selection({}, plan, 1), {})

    def test_native_payload_slices_preserve_nibbles_and_scale_bytes(self):
        tensors = {
            "expert.weight": {"dtype": "I8", "shape": [4, 128]},
            "expert.scale": {"dtype": "F8_E8M0", "shape": [4, 8]},
        }
        weight_part = {"keep": True, "axis": 1, "first": 64, "count": 64}
        scale_part = {"keep": True, "axis": 1, "first": 4, "count": 4}
        plan = {"version": 2, "payloads": [
            {"name": "expert.weight", "dtype": "i8", "shape": [4, 128], "ranks": [weight_part] * 2},
            {"name": "expert.scale", "dtype": "f8_e8m0", "shape": [4, 8], "ranks": [scale_part] * 2},
        ]}
        chosen = selection(tensors, plan, 1)
        payload = bytes(range(256)) * 2
        actual = b"".join(payload[start:start + n] for start, n in chosen["expert.weight"]["spans"])
        self.assertEqual(actual, b"".join(payload[r * 128 + 64:r * 128 + 128] for r in range(4)))
        self.assertEqual(list(chosen["expert.scale"]["spans"]), [(4, 4), (12, 4), (20, 4), (28, 4)])
        self.assertEqual(chosen["expert.weight"]["shape"], [4, 64])
        tensors["expert.scale"]["shape"] = [4, 7]
        with self.assertRaisesRegex(ValueError, "inspector/source tensor mismatch"):
            selection(tensors, plan, 1)

    def test_shard_names_are_not_paths(self):
        self.assertTrue(safe_shard("model-00001-of-00016.safetensors"))
        for name in ["../model.safetensors", "x/model.safetensors", "C:\\model.safetensors", ""]:
            self.assertFalse(safe_shard(name))

    def test_expert_row_slices_do_not_allocate_millions_of_span_tuples(self):
        import sys
        spans, shape = ranges([100000, 1024], 1, 1, 512, 512)
        self.assertEqual(shape, [100000, 512])
        self.assertEqual(spans.byte_count, 51200000)
        self.assertEqual(next(iter(spans)), (512, 512))
        self.assertLess(sys.getsizeof(spans), 256)
        with self.assertRaises(ValueError):
            ranges([2, 128], 1, -1, 0, 64)


if __name__ == "__main__":
    unittest.main()
