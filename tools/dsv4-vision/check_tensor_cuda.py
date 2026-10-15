"""Check dtype-preserving V4 tensor primitives against independent NumPy references."""
import argparse
import ctypes as C
import json
from pathlib import Path

import numpy as np

from check_exl3_cuda import Driver
from check_ops_cuda import bf16, floats, u, f


def dense(driver, rng):
    count = 0
    for xkind in (0, 2):
        for wkind in (0, 1, 2):
            for normalize in (0, 1):
                rows, k, n = 3, 588, 31
                raw_x = rng.normal(size=(rows, k)).astype(np.float32)
                raw_w = rng.normal(0, 0.01, (n, k)).astype(np.float32)
                x = raw_x if xkind == 0 else bf16(raw_x)
                w = raw_w if wkind == 0 else raw_w.astype(np.float16) if wkind == 1 else bf16(raw_w)
                bias = rng.normal(0, 0.1, n).astype(np.float32)
                y = driver.empty(rows * n * 4)
                driver.launch("tf_dsv4_dense", (1, rows, 1), (128, 1, 1),
                              driver.upload(x), driver.upload(w), driver.upload(bias), y,
                              u(k), u(n), u(xkind), u(wkind), u(0), u(normalize), f(1e-6))
                xx = (x if xkind == 0 else floats(x)).astype(np.float64)
                ww = (floats(w) if wkind == 2 else w).astype(np.float64)
                expected = xx @ ww.T
                if normalize:
                    expected *= 1 / np.sqrt(np.mean(xx ** 2, axis=1, keepdims=True) + 1e-6)
                expected += bias
                np.testing.assert_allclose(driver.download(y, (rows, n), np.float32), expected, atol=1e-6, rtol=1e-5)
                count += 1
    return count


def embeddings(driver, rng):
    dim, vocab = 128, 7
    embedding = bf16(rng.normal(size=(vocab, dim)))
    ids = np.array([0, 6, 7, 11], dtype=np.uint32)
    output = driver.empty(4 * 4 * dim * 2)
    driver.launch("tf_dsv4_embed", (1, 4, 1), (128, 1, 1), driver.upload(embedding), driver.upload(ids),
                  output, u(dim), u(vocab), u(2), u(4))
    expected = np.zeros((4, 4, dim), dtype=np.uint16)
    expected[0] = embedding[0]
    expected[1] = embedding[6]
    np.testing.assert_array_equal(driver.download(output, expected.shape, np.uint16), expected)
    taps = driver.empty(4 * 3 * dim * 2)
    driver.launch("tf_dsv4_tap", (1, 4, 1), (128, 1, 1), output, taps, u(dim), u(3), u(2))
    actual = driver.download(taps, (4, 3, dim), np.uint16)
    np.testing.assert_array_equal(actual[:, 2], bf16(floats(expected).mean(axis=1)))
    return 2


def vision_rotary(driver, rng):
    rows, heads, dim = 12, 2, 64
    x = bf16(rng.normal(size=(rows, heads, dim)))
    ptr = driver.upload(x)
    driver.launch("tf_dsv4_vision_rope", (rows, heads, 1), (128, 1, 1), ptr, u(4), u(heads), u(dim), f(10000))
    actual = floats(driver.download(ptr, x.shape, np.uint16))
    positions = np.stack([np.arange(rows) // 4, np.arange(rows) % 4], axis=1)
    inv = 10000 ** (-np.arange(0, dim // 2, 2, dtype=np.float64) / (dim // 2))
    angles = (positions[:, :, None] * inv[None, None, :]).reshape(rows, dim // 2)
    a, b = np.split(floats(x).astype(np.float64), 2, axis=2)
    cosine, sine = np.cos(angles)[:, None], np.sin(angles)[:, None]
    expected = np.concatenate([a * cosine - b * sine, b * cosine + a * sine], axis=2)
    np.testing.assert_allclose(actual, expected, atol=0.008, rtol=0.005)
    return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fatbin")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    driver = Driver(args.fatbin, args.device)
    rng = np.random.default_rng(7625)
    try:
        counts = {fn.__name__: fn(driver, rng) for fn in [dense, embeddings, vision_rotary]}
    finally:
        driver.close()
    text = json.dumps({"checks": counts, "passed": sum(counts.values()), "model_qualified": False}, indent=2)
    print(text)
    if args.report:
        args.report.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
