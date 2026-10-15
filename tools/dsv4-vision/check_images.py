"""Compare native preprocessing with the pinned official image_processor.py using Pillow and NumPy."""
import argparse
import ast
import io
import json
import math
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageOps


class Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def permute(self, *axes):
        return Tensor(self.value.transpose(axes))

    def reshape(self, *shape):
        return Tensor(self.value.reshape(shape))

    view = reshape

    def transpose(self, a, b):
        return Tensor(self.value.swapaxes(a, b))

    def __truediv__(self, other):
        return Tensor(self.value / other)

    def __sub__(self, other):
        return Tensor(self.value - other)

    def __getitem__(self, key):
        return Tensor(self.value[key.value if isinstance(key, Tensor) else key])

    def __setitem__(self, key, value):
        self.value[key] = value.value if isinstance(value, Tensor) else value

    def __ge__(self, other):
        return self.value >= other

    def to(self, dtype):
        assert dtype == "bf16"
        x = self.value.astype(np.float32).view(np.uint32)
        return Tensor(((x + 0x7FFF + ((x >> 16) & 1)) >> 16).astype(np.uint16))


def official(path):
    names = {"grid_tokens", "solve_resize_ratio", "safe_resize", "load_image", "build_image_block"}
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    selected = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
    torch = SimpleNamespace(
        from_numpy=Tensor, bfloat16="bf16", int64=np.int64,
        tensor=lambda x, dtype=None: Tensor(np.array(x, dtype=dtype)),
        arange=lambda *x: Tensor(np.arange(*x)),
        full=lambda shape, v, dtype=None: Tensor(np.full(shape, v, dtype=dtype)),
        cat=lambda values: Tensor(np.concatenate([x.value for x in values])),
    )
    namespace = dict(math=math, np=np, io=io, Image=Image, ImageOps=ImageOps, torch=torch,
                     COMPRESS_PAD_TO=4, IMAGE_START=0, IMAGE_PAD=1, IMAGE=2, IMAGE_NEW_LINE=3, IMAGE_END=4,
                     load_image_bytes=lambda record: record["data"])
    exec(compile(selected, str(path), "exec"), namespace)
    return namespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--jpeg-library", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    reference = official(args.reference)
    cfg = SimpleNamespace(**json.loads(args.config.read_text(encoding="utf-8")))
    rng = np.random.default_rng(54321)
    checks = 0
    scratch = Path.cwd() / ".zig-cache"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tf-dsv4-images-", dir=scratch) as temp:
        root = Path(temp)
        for number, (height, width, mode) in enumerate([
            (1, 1, "RGB"), (17, 29, "RGBA"), (1000, 2, "RGB"), (7, 901, "RGB"),
            (251, 167, "L"), (71, 137, "P"), (42, 42, "I;16"), (130, 141, "RGB"), (23, 31, "CMYK"),
        ]):
            channels = 4 if mode == "RGBA" else 3
            array = rng.integers(0, 256, (height, width, channels), dtype=np.uint8)
            image = Image.fromarray(array).convert(mode) if mode != "I;16" else Image.fromarray(
                rng.integers(0, 65536, (height, width), dtype=np.uint16))
            formats = [] if mode == "CMYK" else ["PNG"]
            if args.jpeg_library and mode in ("RGB", "L", "CMYK"):
                formats.append("JPEG")
            for fmt in formats:
                source = root / f"input-{number}.{fmt.lower()}"
                image.save(source, format=fmt)
                expected, vh, vw, rows, cols = reference["load_image"]({"data": source.read_bytes()}, cfg)
                for position in [0, 1, 2, 3, 127, 2047]:
                    out = root / "patches.bf16"
                    command = [str(args.binary.resolve()), str(args.config.resolve()), str(source), str(out), str(position)]
                    if args.jpeg_library:
                        command.append(str(args.jpeg_library.resolve()))
                    result = subprocess.run(command, check=True, capture_output=True)
                    native = json.loads(result.stdout)
                    assert native["geometry"]["height"] == vh * cfg.vision_patch_size
                    assert native["geometry"]["width"] == vw * cfg.vision_patch_size
                    assert native["geometry"]["rows"] == rows and native["geometry"]["cols"] == cols
                    actual = np.frombuffer(out.read_bytes(), dtype=np.uint16).reshape(expected.value.shape)
                    np.testing.assert_array_equal(actual, expected.value, err_msg=f"{fmt}/{number}/{position}")
                    types, permutation = reference["build_image_block"](rows, cols, position)
                    np.testing.assert_array_equal(native["types"], types.value)
                    np.testing.assert_array_equal(native["permutation"], permutation.value)
                    checks += 1
    text = json.dumps({"image_layout_patch_cases": checks, "patches_bit_exact": True,
                       "jpeg_checked": bool(args.jpeg_library), "model_qualified": False}, indent=2)
    print(text)
    if args.report:
        args.report.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
