"""Check compact FP8 cache gathers and sparse attention against independently decoded rows and FP64 attention."""
import argparse
import ctypes as C
import json
from pathlib import Path

import numpy as np

from check_exl3_cuda import Driver
from check_native_cuda import FP8
from check_ops_cuda import bf16, floats, u, f


def check(d, text, ops):
    rng = np.random.default_rng(241024)
    dim, heads = 512, 2
    selections = [np.array([2, 0, 1], np.int32),
                  np.r_[np.arange(3), np.arange(128, 133)].astype(np.int32),
                  np.r_[np.arange(128), np.arange(512, 2560)].astype(np.int32)]
    cases = []
    for ids in selections:
        count, source_rows = len(ids), int(ids.max()) + 1
        codes = rng.integers(0, 120, (source_rows, dim), dtype=np.uint8)
        codes |= rng.integers(0, 2, codes.shape, dtype=np.uint8) << 7
        scales = np.exp2(rng.integers(-15, -10, source_rows)).astype(np.float32)
        packed = np.empty((source_rows, dim + 4), np.uint8)
        packed[:, :dim] = codes
        packed[:, dim:] = scales.view(np.uint8).reshape(-1, 4)
        expected_rows = bf16(FP8[codes[ids]] * scales[ids, None])
        guard = np.full((count + 1, dim), 0x3F80, np.uint16)
        gathered = d.upload(guard)
        d.module = text
        d.launch("tf_dsv4_text_gather8", (count, 1, 1), (256, 1, 1),
                 d.upload(packed), d.upload(ids), gathered, u(count), u(dim), u(dim + 4))
        actual_rows = d.download(gathered, guard.shape, np.uint16)
        np.testing.assert_array_equal(actual_rows[:count], expected_rows)
        np.testing.assert_array_equal(actual_rows[count], guard[count])
        q = bf16(rng.normal(0, .1, (heads, dim)))
        sink = np.array([0, .25], np.float32)
        keys = floats(expected_rows).astype(np.float64)
        scores = floats(q).astype(np.float64) @ keys.T / np.sqrt(dim)
        maximum = np.maximum(scores.max(axis=1), sink)
        probabilities = np.exp(scores - maximum[:, None])
        probabilities /= (probabilities.sum(axis=1) + np.exp(sink - maximum))[:, None]
        expected = probabilities @ keys
        qptr, iptr, sptr = d.upload(q), d.upload(np.arange(count, dtype=np.int32)), d.upload(sink)
        out = d.empty(heads * dim * 2)
        d.module = ops
        first = None
        for _ in range(5):
            d.launch("tf_dsv4_sparse_attn", (1, heads, 1), (256, 1, 1),
                     qptr, gathered, iptr, sptr, out, u(heads), u(count), u(dim), f(1 / np.sqrt(dim)),
                     shared=(count + 256) * 4)
            actual = d.download(out, (heads, dim), np.uint16)
            if first is None:
                first = actual
            else:
                np.testing.assert_array_equal(actual, first)
        np.testing.assert_allclose(floats(first), expected, atol=2e-5, rtol=.004)
        np.testing.assert_array_equal(d.download(gathered, guard.shape, np.uint16), actual_rows)
        error = np.abs(floats(first) - expected)
        cases.append(dict(rows=count, source_rows=source_rows, passed=True,
                          repeats=5, max_abs=float(error.max()),
                          worst_per_column=error.max(axis=0).tolist(),
                          legacy_indices_out_of_bounds=bool(ids.max() >= count)))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text_fatbin", type=Path)
    parser.add_argument("ops_fatbin", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    d = Driver(args.text_fatbin, 0)
    text = d.module
    ops = C.c_void_p()
    try:
        image = C.create_string_buffer(args.ops_fatbin.read_bytes())
        d.call("cuModuleLoadData", C.byref(ops), image)
        cases = check(d, text, ops)
    finally:
        d.module = text
        if ops.value:
            d.call("cuModuleUnload", ops)
        d.close()
    args.report.write_text(json.dumps(dict(passed=len(cases), failed=0, cases=cases,
                                         nccl_executed=False, full_model_executed=False), indent=2) + "\n")
    print(f"FP8 sparse rows: {len(cases)} passed, 0 failed (five repeats each)")


if __name__ == "__main__":
    main()
