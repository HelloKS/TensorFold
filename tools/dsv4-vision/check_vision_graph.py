"""Check the complete native ViT and aligner against a BF16-boundary NumPy reference using the same weights."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
import sys

import numpy as np

from check_checkpoint_projection import read
from check_ops_cuda import bf16, floats
from prepare_cache import catalog


def reference(source, tensors, config, patches, height, width, trace=None):
    def weight(key):
        raw = read(source, tensors, key)
        return floats(raw) if tensors[key]["dtype"] == "BF16" else raw

    def cast(x):
        return floats(bf16(np.asarray(x, dtype=np.float32)))

    def linear(x, key):
        value = x.astype(np.float64) @ weight(key + ".weight").astype(np.float64).T
        if key + ".bias" in tensors:
            value += weight(key + ".bias")
        return cast(value)

    def norm(x, key):
        xx = x.astype(np.float32)
        inv = np.float32(1) / np.sqrt(np.mean(xx * xx, axis=-1, keepdims=True) + np.float32(1e-6))
        return cast((xx * inv) * weight(key))

    heads = config["vision_n_heads"]
    dim = config["vision_dim"] // heads
    positions = np.stack([np.arange(height * width) // width, np.arange(height * width) % width], axis=1)
    inv = np.float32(config["vision_rope_theta"]) ** (-np.arange(0, dim // 2, 2, dtype=np.float32) / np.float32(dim // 2))
    angles = (positions[:, :, None].astype(np.float32) * inv).reshape(height * width, dim // 2)
    cosine, sine = np.cos(angles)[:, None], np.sin(angles)[:, None]

    def rope(x):
        a, b = np.split(x.astype(np.float32), 2, axis=-1)
        return cast(np.concatenate([a * cosine - b * sine, b * cosine + a * sine], axis=-1))

    x = linear(patches, "vision.patch_embed.proj")
    def observe(index):
        if trace:
            actual = floats(np.frombuffer((trace / f"layer{index}.bf16").read_bytes(), dtype="<u2")).reshape(x.shape)
            error = np.abs(actual - x)
            print("layer", index, float(error.max()), float(np.sqrt(np.mean(error ** 2))), flush=True)
    observe(0)
    for i in range(config["vision_n_layers"]):
        p = f"vision.blocks.{i}"
        normalized = norm(x, p + ".norm1.weight")
        q, k, v = [v.reshape(height * width, heads, dim) for v in np.split(linear(normalized, p + ".attn.wqkv"), 3, axis=-1)]
        q, k = [rope(v).transpose(1, 0, 2).astype(np.float64) for v in (q, k)]
        v = v.transpose(1, 0, 2).astype(np.float64)
        scores = q @ k.transpose(0, 2, 1) / np.sqrt(dim)
        probability = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probability /= probability.sum(axis=-1, keepdims=True)
        attention = cast((probability @ v).transpose(1, 0, 2).reshape(height * width, -1))
        x = cast(x + linear(attention, p + ".attn.wo"))
        normalized = norm(x, p + ".norm2.weight")
        gate, up = np.split(linear(normalized, p + ".mlp.w1"), 2, axis=-1)
        activated = cast(gate / (np.float32(1) + np.exp(-gate)))
        x = cast(x + linear(cast(activated * up), p + ".mlp.w2"))
        observe(i + 1)
    x = norm(x, "vision.norm.weight").reshape(height, width, -1)
    padded = np.pad(x, ((0, -height % 3), (0, -width % 3), (0, 0)))
    unfolded = padded.reshape((height + 2) // 3, 3, (width + 2) // 3, 3, -1).transpose(0, 2, 4, 1, 3).reshape(-1, 9 * config["vision_dim"])
    x = linear(unfolded, "aligner.w1")
    erf = np.frompyfunc(math.erf, 1, 1)(x.astype(np.float64) / np.sqrt(2)).astype(np.float64)
    x = cast(.5 * x * (1 + erf))
    return linear(x, "aligner.w2")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--probe", type=Path, required=True)
    p.add_argument("--kernels", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--trace", action="store_true")
    p.add_argument("--official-reference", type=Path)
    p.add_argument("--grid", action="append", help="patch grid such as 28x28; default retains both tiny stress cases")
    args = p.parse_args()
    source = args.checkpoint.resolve()
    config = json.loads((source / "config.json").read_text())
    tensors = catalog(source)
    rng = np.random.default_rng(22361)
    scratch = Path.cwd() / ".zig-cache"
    scratch.mkdir(exist_ok=True)
    official = None
    if args.official_reference:
        from vision_reference import Official
        official = Official(args.official_reference, source, tensors, config)
    cases = []
    with tempfile.TemporaryDirectory(prefix="tf-native-vision-", dir=scratch) as tmp:
        root = Path(tmp)
        for height, width in ([tuple(map(int, s.split("x"))) for s in args.grid] if args.grid else [(3, 4), (5, 3)]):
            patches = floats(bf16(rng.normal(0, .2, (height * width, 3 * 14 * 14))))
            inp, out = root / "input.bf16", root / "output.bf16"
            inp.write_bytes(bf16(patches).tobytes())
            trace = root / "trace"
            trace.mkdir(exist_ok=True)
            result = subprocess.run([str(args.probe.resolve()), str(source), str(inp), str(height), str(width),
                                     str(out), str(args.kernels.resolve())] + ([str(trace)] if args.trace else []), capture_output=True)
            if result.returncode:
                raise RuntimeError(result.stderr.decode(errors="replace"))
            expected = official.run(patches, height, width) if official else reference(source, tensors, config, patches, height, width, trace if args.trace else None)
            actual = floats(np.frombuffer(out.read_bytes(), dtype="<u2")).reshape(expected.shape)
            error = np.abs(actual - expected)
            matched = bool(np.all(np.isclose(actual, expected, rtol=.04, atol=.04)))
            cases.append(dict(passed=matched, grid=[height, width], layers=config["vision_n_layers"], max_abs=float(error.max()),
                              rms=float(np.sqrt(np.mean(error ** 2))), max_abs_by_column=error.max(axis=0).tolist(),
                              outside_tolerance=int(np.count_nonzero(~np.isclose(actual, expected, rtol=.04, atol=.04))),
                              max_tolerance_ratio=float(np.max(error / (.04 + .04 * np.abs(expected)))),
                              input_sha256=hashlib.sha256(inp.read_bytes()).hexdigest(),
                              output_sha256=hashlib.sha256(out.read_bytes()).hexdigest()))
            print("full vision", height, width, "PASS" if matched else "FAIL", "max_abs", float(error.max()), "rms", cases[-1]["rms"])
    report = dict(passed=sum(c["passed"] for c in cases), failed=sum(not c["passed"] for c in cases), cases=cases,
                  reference="official PyTorch CPU" if official else "independent NumPy", rtol=.04, atol=.04, vision_graph_executed=True, language_model_executed=False,
                  dgx_spark_executed=False, seed=22361, model_revision="6821d6ad3681a4b137b066b76094fa82ebd0a380")
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if report["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
