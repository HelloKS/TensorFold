"""Run the native checkpoint loader and CUDA projection probe against independent real-weight FP64 references."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from prepare_cache import catalog
from check_native_cuda import FP8, quantize, unpack
from check_ops_cuda import bf16, floats


DEFAULTS = ["layers.0.attn.wq_a", "layers.0.attn.wo_a", "layers.0.ffn.experts.0.w1",
            "layers.0.ffn.experts.0.w2", "vision.blocks.0.attn.wqkv", "vision.blocks.0.mlp.w1"]


def read(source, tensors, name):
    entry = tensors[name]
    begin, end = entry["data_offsets"]
    with (source / entry["file"]).open("rb") as stream:
        stream.seek(entry["data_start"] + begin)
        data = stream.read(end - begin)
    if len(data) != end - begin:
        raise ValueError(f"truncated tensor: {name}")
    dtype = {"BF16": "<u2", "F32": "<f4", "F8_E4M3": "u1", "I8": "u1", "F8_E8M0": "u1"}[entry["dtype"]]
    return np.frombuffer(data, dtype=dtype).reshape(entry["shape"])


def check(args):
    source = args.checkpoint.resolve()
    tensors = catalog(source)
    rng = np.random.default_rng(82145)
    scratch = Path.cwd() / ".zig-cache"
    scratch.mkdir(exist_ok=True)
    results = []
    with tempfile.TemporaryDirectory(prefix="tf-native-projection-", dir=scratch) as tmp:
        root = Path(tmp)
        for key in args.tensor or DEFAULTS:
            header = tensors[key + ".weight"]
            w = read(source, tensors, key + ".weight")
            fp4 = header["dtype"] == "I8"
            n, stored_k = w.shape
            k = stored_k * (2 if fp4 else 1)
            x = bf16(rng.normal(0, .1, (4, k)).astype(np.float32))
            inp, out = root / "input.bf16", root / "output.f32"
            inp.write_bytes(x.tobytes())
            command = [str(args.probe.resolve()), str(source), key, str(k), str(n), str(inp), str(out),
                       "4", str(args.kernels.resolve())]
            subprocess.run(command, check=True, capture_output=True)
            actual = np.frombuffer(out.read_bytes(), dtype="<f4").reshape(4, n)
            xx = floats(x).astype(np.float64)
            if header["dtype"] == "BF16":
                ww = floats(w).astype(np.float64)
            else:
                scale = read(source, tensors, key + ".scale")
                ww = unpack(w, scale, fp4)
                if key.endswith(".attn.wo_a"):
                    ww = floats(bf16(ww)).astype(np.float64)
                else:
                    q, qs = quantize(xx)
                    xx = FP8[q].astype(np.float64) * np.repeat(np.exp2(qs.astype(np.float64) - 127), 128, axis=1)
            expected = xx @ ww.T
            if key + ".bias" in tensors:
                bias = read(source, tensors, key + ".bias")
                expected += floats(bias) if bias.dtype == np.uint16 else bias
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
            columns = np.max(np.abs(actual - expected), axis=0)
            results.append(dict(tensor=key, input_shape=list(x.shape), dtype=header["dtype"],
                                weight_sha256=hashlib.sha256(w.tobytes()).hexdigest(),
                                max_abs=float(columns.max()), worst_column=int(columns.argmax()),
                                max_abs_by_column=columns.tolist()))
            print(key, "PASS", "max_abs", float(columns.max()))
    return dict(passed=len(results), projections=results, model_revision="6821d6ad3681a4b137b066b76094fa82ebd0a380",
                full_model_executed=False, dgx_spark_executed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--kernels", type=Path, required=True)
    parser.add_argument("--tensor", action="append")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = check(args)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
