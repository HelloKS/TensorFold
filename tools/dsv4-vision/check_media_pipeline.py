"""Connect native PNG/JPEG preprocessing, ViT, aligner and positioned image spans without a server."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from PIL import Image

from check_checkpoint_projection import read
from check_ops_cuda import bf16, floats
from prepare_cache import catalog


def execute(command):
    result = subprocess.run([str(s) for s in command], capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace"))
    return result.stdout


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--image-probe", type=Path, required=True)
    p.add_argument("--vision-probe", type=Path, required=True)
    p.add_argument("--kernels", type=Path, required=True)
    p.add_argument("--jpeg-library", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()
    source = args.checkpoint.resolve()
    tensors = catalog(source)
    config = json.loads((source / "config.json").read_text())
    specials = []
    for name in ["image_start", "image_pad", "image_newline", "image_end"]:
        raw = read(source, tensors, name)
        specials.append(bf16(floats(raw) if raw.dtype == np.uint16 else raw))
    scratch = Path.cwd() / ".zig-cache"
    scratch.mkdir(exist_ok=True)
    cases = []
    with tempfile.TemporaryDirectory(prefix="tf-native-media-", dir=scratch) as tmp:
        root = Path(tmp).resolve()
        y, x = np.indices((64, 64))
        rgb = np.stack([x * 4, y * 4, ((x + y) * 2) % 256], axis=-1).astype(np.uint8)
        for suffix, position in [("png", 0), ("jpg", 5)]:
            image = root / ("image." + suffix)
            Image.fromarray(rgb).save(image)
            patches, raster, span = root / "patches.bf16", root / "raster.bf16", root / "span.bf16"
            info = json.loads(execute([args.image_probe.resolve(), source / "config.json", image, patches,
                                       position, args.jpeg_library.resolve()]))
            grid = info["geometry"]
            execute([args.vision_probe.resolve(), source, patches, grid["height"] // config["vision_patch_size"],
                     grid["width"] // config["vision_patch_size"], raster, args.kernels.resolve()])
            output = json.loads(execute([args.vision_probe.resolve(), "--image", source, image, span,
                                         args.kernels.resolve(), position, args.jpeg_library.resolve()]))
            aligned = np.frombuffer(raster.read_bytes(), dtype="<u2").reshape(-1, config["hidden_size"])
            expected, image_index = [], 0
            special_for_type = {0: 0, 1: 1, 3: 2, 4: 3}
            for kind in info["types"]:
                if kind == 2:
                    expected.append(aligned[info["permutation"][image_index]])
                    image_index += 1
                else:
                    expected.append(specials[special_for_type[kind]])
            np.testing.assert_array_equal(np.frombuffer(span.read_bytes(), dtype="<u2").reshape(-1, config["hidden_size"]), np.stack(expected))
            assert output["ids"] == [config["vocab_size"] + kind for kind in info["types"]]
            assert output["span"] == dict(start=position + 3 - position % 4, end=position + len(expected))
            cases.append(dict(format=suffix, position=position, tokens=len(expected), span=output["span"], bit_exact=True))
            print(suffix, "PASS", len(expected), "tokens", flush=True)
    args.report.write_text(json.dumps(dict(passed=len(cases), cases=cases, vision_fidelity_qualified=False,
                                           language_model_executed=False, dgx_spark_executed=False), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
