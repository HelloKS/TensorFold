"""Numerical and state-shape checks for the source-built V4 model operators; no model weights required."""
import argparse
import ctypes as C
import json
from pathlib import Path

import numpy as np

from check_exl3_cuda import Driver


def bf16(values):
    x = np.asarray(values, dtype=np.float32).copy().view(np.uint32)
    return ((x + np.uint32(0x7FFF) + ((x >> 16) & 1)) >> 16).astype(np.uint16)


def floats(values):
    return (np.asarray(values, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def u(x):
    return C.c_uint32(x)


def f(x):
    return C.c_float(x)


def routing(d, rng):
    logits = rng.normal(0, 3, (4, 256)).astype(np.float32)
    bias = rng.normal(0, 0.1, 256).astype(np.float16)
    vl = rng.normal(0, 0.7, 256).astype(np.float32)
    ids = np.array([0, 2, 4, 8], dtype=np.uint32)
    table = np.array([rng.choice(256, 6, replace=False) for _ in range(4)], dtype=np.int64)
    ptrs = [d.upload(a) for a in [logits, ids, table, bias, vl]]
    picks, weights = d.empty(4 * 6 * 4), d.empty(4 * 6 * 4)
    original = np.sqrt(np.logaddexp(0, logits.astype(np.float64)))
    for hashed in (0, 1):
        d.launch("tf_dsv4_route", (4, 1, 1), (256, 1, 1), *ptrs, picks, weights, u(4), u(hashed), f(1.5))
        got = d.download(picks, (4, 6), np.uint32)
        for row in range(4):
            scores = original[row] + (vl if ids[row] >= 4 else bias)
            expected = table[ids[row]] if hashed and ids[row] < 4 else np.argsort(-scores)[:6]
            np.testing.assert_array_equal(got[row], expected)
            selected = original[row, expected]
            np.testing.assert_allclose(d.download(weights, (4, 6), np.float32)[row],
                                       selected / selected.sum() * 1.5, atol=2e-7, rtol=2e-6)
    return 2


def sinkhorn(d, rng):
    mixes = rng.normal(0, 2, (3, 24)).astype(np.float32)
    scale = np.array([0.8, 0.3, 1.5], dtype=np.float32)
    base = rng.normal(0, 1, 24).astype(np.float32)
    outputs = [d.empty(3 * count * 4) for count in (4, 4, 16)]
    d.launch("tf_dsv4_sinkhorn", (3, 1, 1), (32, 1, 1),
             *[d.upload(x) for x in [mixes, scale, base]], *outputs, u(20), f(1e-6))
    pre = 1 / (1 + np.exp(-(mixes[:, :4].astype(np.float64) * scale[0] + base[:4]))) + 1e-6
    post = 2 / (1 + np.exp(-(mixes[:, 4:8].astype(np.float64) * scale[1] + base[4:8])))
    matrix = (mixes[:, 8:].astype(np.float64) * scale[2] + base[8:]).reshape(3, 4, 4)
    matrix = np.exp(matrix - matrix.max(axis=2, keepdims=True))
    matrix = matrix / matrix.sum(axis=2, keepdims=True) + 1e-6
    for iteration in range(20):
        if iteration:
            matrix /= matrix.sum(axis=2, keepdims=True) + 1e-6
        matrix /= matrix.sum(axis=1, keepdims=True) + 1e-6
    for ptr, shape, expected in zip(outputs, [(3, 4), (3, 4), (3, 4, 4)], [pre, post, matrix]):
        np.testing.assert_allclose(d.download(ptr, shape, np.float32), expected, atol=5e-7, rtol=2e-6)
    residual = bf16(rng.normal(size=(3, 4, 128)))
    x = bf16(rng.normal(size=(3, 128)))
    out = d.empty(residual.nbytes)
    d.launch("tf_dsv4_hc_post", (1, 3, 1), (128, 1, 1), d.upload(x), d.upload(residual),
             outputs[1], outputs[2], out, u(128))
    expected = np.einsum("bij,bid->bjd", matrix, floats(residual).astype(np.float64))
    expected += post[:, :, None] * floats(x)[:, None, :]
    actual = floats(d.download(out, residual.shape, np.uint16))
    np.testing.assert_allclose(actual, expected, atol=0.02, rtol=0.005)
    return 2


def quantization(d, rng):
    for fp4, group in ((0, 64), (1, 32)):
        x = bf16(rng.normal(0, 3, (8, 128)))
        values = floats(x).astype(np.float64)
        groups = values.reshape(-1, group)
        scale = np.maximum(np.max(np.abs(groups), axis=1), 6 * 2.0 ** -126 if fp4 else 1e-4)
        raw = (scale.astype(np.float32) * np.float32(1 / (6 if fp4 else 448))).astype(np.float32)
        scale = 2.0 ** np.ceil(np.log2(raw))
        if fp4:
            table = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6])
        else:
            table = np.array([(m / 8) * 2 ** -6 if e == 0 else (1 + m / 8) * 2 ** (e - 7)
                              for e in range(16) for m in range(8) if e < 15 or m < 7])
        normalized = np.abs(groups) / scale[:, None]
        dist = np.abs(normalized[:, :, None] - table)
        best = dist.argmin(axis=2)
        for i, j in np.ndindex(best.shape):
            equal = np.flatnonzero(dist[i, j] == dist[i, j, best[i, j]])
            if len(equal) > 1:
                best[i, j] = next(k for k in equal if k % 2 == 0)
        expected = bf16((table[best] * np.sign(groups) * scale[:, None]).reshape(values.shape))
        ptr = d.upload(x)
        d.launch("tf_dsv4_qat", (128 // group, 8, 1), (128, 1, 1), ptr, u(128), u(group), u(fp4), u(128))
        np.testing.assert_array_equal(d.download(ptr, x.shape, np.uint16), expected)
    return 2


def attention(d, rng):
    q = bf16(rng.normal(size=(3, 2, 128)))
    kv = bf16(rng.normal(size=(9, 128)))
    indices = np.array([[0, 1, -1, -1], [1, 5, 8, -1], [2, 0, 7, 4]], dtype=np.int32)
    sink = np.array([-0.5, 1.0], dtype=np.float32)
    output = d.empty(q.nbytes)
    fn = C.c_void_p()
    d.call("cuModuleGetFunction", C.byref(fn), d.module, b"tf_dsv4_sparse_attn")
    values = [d.upload(q), d.upload(kv), d.upload(indices), d.upload(sink), output, u(2), u(4), u(128), f(128 ** -0.5)]
    args = (C.c_void_p * len(values))(*(C.addressof(x) for x in values))
    d.call("cuLaunchKernel", fn, 3, 2, 1, 256, 1, 1, (4 + 256) * 4, C.c_void_p(), args, C.c_void_p())
    d.call("cuCtxSynchronize")
    actual = floats(d.download(output, q.shape, np.uint16))
    for row, head in np.ndindex(3, 2):
        keys = floats(kv)[indices[row][indices[row] >= 0]].astype(np.float64)
        logits = keys @ floats(q)[row, head].astype(np.float64) / np.sqrt(128)
        largest = max(logits.max(), sink[head])
        probs = np.exp(logits - largest)
        probs /= probs.sum() + np.exp(sink[head] - largest)
        np.testing.assert_allclose(actual[row, head], probs @ keys, atol=0.008, rtol=0.005)
    return 1


def pooling_and_unfold(d, rng):
    values = rng.normal(size=(2, 8, 128)).astype(np.float32)
    scores = rng.normal(size=values.shape).astype(np.float32)
    output = d.empty(2 * 128 * 2)
    d.launch("tf_dsv4_pool", (1, 2, 1), (128, 1, 1), d.upload(values), d.upload(scores), output, u(8), u(128))
    weights = np.exp(scores.astype(np.float64) - scores.max(axis=1, keepdims=True))
    expected = (weights * values).sum(axis=1) / weights.sum(axis=1)
    np.testing.assert_allclose(floats(d.download(output, (2, 128), np.uint16)), expected, atol=0.008, rtol=0.005)
    x = bf16(np.arange(4 * 5 * 2).reshape(4, 5, 2))
    output = d.empty(4 * 2 * 9 * 2)
    d.launch("tf_dsv4_unfold", (1, 4, 1), (128, 1, 1), d.upload(x), output, u(4), u(5), u(2))
    padded = np.pad(x, ((0, 2), (0, 1), (0, 0)))
    expected = np.array([padded[r:r + 3, c:c + 3].transpose(2, 0, 1).reshape(-1)
                         for r in range(0, 6, 3) for c in range(0, 6, 3)])
    np.testing.assert_array_equal(d.download(output, expected.shape, np.uint16), expected)
    return 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fatbin")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    rng = np.random.default_rng(20261016)
    driver = Driver(args.fatbin, args.device)
    try:
        counts = {fn.__name__: fn(driver, rng) for fn in [routing, sinkhorn, quantization, attention, pooling_and_unfold]}
    finally:
        driver.close()
    text = json.dumps({"checks": counts, "passed": sum(counts.values()), "model_qualified": False}, indent=2)
    print(text)
    if args.report:
        args.report.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
