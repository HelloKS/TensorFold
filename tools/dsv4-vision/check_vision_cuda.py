"""Verify ViT operators with independent NumPy arithmetic and explicit BF16 boundaries."""
import argparse
import json
from pathlib import Path

import numpy as np

from check_exl3_cuda import Driver
from check_ops_cuda import bf16, floats, u


def cancellation(driver):
    cases = []
    for kind in ("qk", "pv"):
        rows, heads, dim = (2 if kind == "qk" else 129), 1, 64
        q, k, v = [np.zeros((rows, heads, dim), np.float32) for _ in range(3)]
        if kind == "qk":
            q[:] = 1
            q[:, :, [0, -1]] = 4096
            k[0] = 1
            k[0, 0, 0], k[0, 0, -1] = 4096, -4096
            v[0] = 1
        else:
            v[:] = 1
            v[0], v[-1] = 2 ** 20, -(2 ** 20)
        q, k, v = [bf16(x) for x in (q, k, v)]
        y = driver.empty(q.nbytes)
        driver.launch("tf_dsv4_vision_attention", (rows, heads, 1), (128, 1, 1),
                      driver.upload(q), driver.upload(k), driver.upload(v), y, u(rows), u(heads), u(dim), shared=rows * 4)
        qq, kk, vv = [floats(x).transpose(1, 0, 2).astype(np.float64) for x in (q, k, v)]
        scores = qq @ kk.transpose(0, 2, 1) / np.sqrt(dim)
        probability = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probability /= probability.sum(axis=-1, keepdims=True)
        expected = (probability @ vv).transpose(1, 0, 2)
        actual = floats(driver.download(y, q.shape, np.uint16))
        error = np.abs(actual - expected)
        cases.append(dict(operation=kind, max_abs=float(error.max()),
                          max_abs_by_column=error.max(axis=(0, 1)).tolist(),
                          passed=bool(np.array_equal(bf16(actual), bf16(expected)))))
    return cases


def check(driver):
    rng = np.random.default_rng(2251)
    passed = 0
    for kind in (0, 1, 2):
        x = bf16(rng.normal(size=(3, 1024)))
        w = rng.normal(1, .02, 1024).astype(np.float32)
        w = w if kind == 0 else w.astype(np.float16) if kind == 1 else bf16(w)
        y = driver.empty(x.nbytes)
        driver.launch("tf_dsv4_vision_norm", (3, 1, 1), (256, 1, 1),
                      driver.upload(x), driver.upload(w), y, u(1024), u(kind))
        xx = floats(x).astype(np.float64)
        ww = (floats(w) if kind == 2 else w).astype(np.float64)
        expected = xx / np.sqrt(np.mean(xx ** 2, axis=1, keepdims=True) + 1e-6) * ww
        np.testing.assert_allclose(floats(driver.download(y, x.shape, np.uint16)), expected, rtol=.004, atol=.001)
        passed += 1
    for parts in (2, 3):
        rows, dim = 3, 128
        fused = rng.integers(0, 65536, (rows, parts, dim), dtype=np.uint16)
        output = [driver.empty(rows * dim * 2) for _ in range(3)]
        driver.launch("tf_dsv4_vision_split", (1, rows, 1), (128, 1, 1),
                      driver.upload(fused), *output, u(dim), u(parts))
        for i in range(parts):
            np.testing.assert_array_equal(driver.download(output[i], (rows, dim), np.uint16), fused[:, i])
        passed += 1
    for rows in (1, 17, 129):
        heads, dim = 2, 64
        q, k, v = [bf16(rng.normal(0, .3, (rows, heads, dim))) for _ in range(3)]
        y = driver.empty(q.nbytes)
        driver.launch("tf_dsv4_vision_attention", (rows, heads, 1), (128, 1, 1),
                      driver.upload(q), driver.upload(k), driver.upload(v), y, u(rows), u(heads), u(dim), shared=rows * 4)
        qq, kk, vv = [floats(x).transpose(1, 0, 2).astype(np.float64) for x in (q, k, v)]
        scores = qq @ kk.transpose(0, 2, 1) / np.sqrt(dim)
        weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
        weights /= weights.sum(axis=-1, keepdims=True)
        expected = (weights @ vv).transpose(1, 0, 2)
        np.testing.assert_allclose(floats(driver.download(y, q.shape, np.uint16)), expected, rtol=.004, atol=.001)
        passed += 1
    gate = bf16(rng.normal(size=(3, 256)))
    up = bf16(rng.normal(size=(3, 256)))
    y = driver.empty(gate.nbytes)
    driver.launch("tf_dsv4_vision_swiglu", (2, 3, 1), (128, 1, 1), driver.upload(gate), driver.upload(up), y, u(256))
    g = floats(gate).astype(np.float64)
    expected = bf16(floats(bf16(g / (1 + np.exp(-g)))) * floats(up))
    np.testing.assert_array_equal(driver.download(y, gate.shape, np.uint16), expected)
    raster = bf16(rng.normal(size=(6, 128)))
    special = rng.normal(size=(4, 128)).astype(np.float32)
    mapping = np.array([-2, -1, 0, 3, 1, 4, 2, 5, -3, -4], dtype=np.int32)
    span = driver.empty(len(mapping) * 128 * 2)
    driver.launch("tf_dsv4_vision_span", (1, len(mapping), 1), (128, 1, 1),
                  driver.upload(raster), driver.upload(special), driver.upload(mapping), span, u(128))
    expected = np.stack([raster[index] if index >= 0 else bf16(special[-index - 1]) for index in mapping])
    np.testing.assert_array_equal(driver.download(span, expected.shape, np.uint16), expected)
    return passed + 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("fatbin")
    p.add_argument("--report", type=Path)
    args = p.parse_args()
    driver = Driver(args.fatbin, 0)
    try:
        count = check(driver)
        regressions = cancellation(driver)
        report = dict(passed=count + sum(c["passed"] for c in regressions),
                      failed=sum(not c["passed"] for c in regressions),
                      cancellation=regressions, full_model_executed=False)
    finally:
        driver.close()
    text = json.dumps(report, indent=2)
    print(text)
    if args.report:
        args.report.write_text(text + "\n", encoding="utf-8")
    if report["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
