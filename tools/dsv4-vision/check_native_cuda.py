"""Check original E2M1/E4M3 projections against independent scalar formats and FP64 dot products."""
import argparse
import json
from pathlib import Path

import numpy as np

from check_exl3_cuda import Driver
from check_ops_cuda import bf16, floats, u


def fp8_table():
    codes = np.arange(256)
    exp, mant = (codes >> 3) & 15, codes & 7
    values = np.where(exp == 0, mant * 2.0 ** -9, (1 + mant / 8) * 2.0 ** (exp - 7))
    values = np.where(codes & 128, -values, values)
    values[(exp == 15) & (mant == 7)] = np.nan
    return values.astype(np.float32)


FP8 = fp8_table()
FP4 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)


def quantize(x):
    x = x.astype(np.float32)
    maximum = np.maximum(np.max(np.abs(x.reshape(-1, 128)), axis=1), np.float32(1e-4))
    raw = maximum * np.float32(1 / 448)
    exp = np.ceil(np.log2(raw.astype(np.float64))).astype(np.int32)
    scale = (exp + 127).astype(np.uint8).reshape(x.shape[0], -1)
    normalized = x / np.repeat(np.exp2(exp).reshape(x.shape[0], -1), 128, axis=1).astype(np.float32)
    values = np.minimum(np.abs(normalized), np.float32(448))
    high = np.searchsorted(FP8[:127], values).clip(0, 126)
    low = np.maximum(high - 1, 0)
    dh, dl = np.abs(FP8[high] - values), np.abs(FP8[low] - values)
    take_high = (dh < dl) | ((dh == dl) & (high % 2 == 0))
    codes = np.where(take_high, high, low).astype(np.uint8)
    codes |= np.signbit(normalized).astype(np.uint8) << 7
    return codes, scale


def unpack(w, scale, fp4):
    if fp4:
        codes = np.stack([w & 15, w >> 4], axis=-1).reshape(w.shape[0], -1)
        data = FP4[codes].astype(np.float64)
        factor = np.repeat(np.exp2(scale.astype(np.float64) - 127), 32, axis=1)
    else:
        data = FP8[w].astype(np.float64)
        factor = np.repeat(np.repeat(np.exp2(scale.astype(np.float64) - 127), 128, axis=0), 128, axis=1)
        factor = factor[:w.shape[0], :w.shape[1]]
    return data * factor


def decode(driver):
    ptr = driver.empty(528 * 4)
    driver.launch("tf_dsv4_native_decode", (1, 1, 1), (256, 1, 1), ptr)
    actual = driver.download(ptr, (528,), np.float32)
    np.testing.assert_array_equal(actual[:256], FP8)
    np.testing.assert_array_equal(actual[256:511], np.exp2(np.arange(255, dtype=np.float64) - 127).astype(np.float32))
    assert np.isnan(actual[511])
    np.testing.assert_array_equal(actual[512:], FP4)
    return 528


def run_case(driver, x, w, ws, bias, fp4, input_f32, output_f32):
    rows, k = x.shape
    n = w.shape[0]
    xp, wp, wsp, bp = [driver.upload(v) for v in (x, w, ws, bias)]
    q, qs = driver.empty(rows * k), driver.empty(rows * k // 128)
    y = driver.empty(rows * n * (4 if output_f32 else 2))
    driver.launch("tf_dsv4_native_quant", (k // 128, rows, 1), (128, 1, 1), xp, q, qs, u(k), u(input_f32))
    driver.launch("tf_dsv4_native_matmul", ((n + 127) // 128, rows, 1), (128, 1, 1), q, qs, wp, wsp, bp,
                  y, u(k), u(n), u(fp4), u(output_f32))
    q_actual = driver.download(q, (rows, k), np.uint8)
    s_actual = driver.download(qs, (rows, k // 128), np.uint8)
    q_expected, s_expected = quantize(x if input_f32 else floats(x))
    np.testing.assert_array_equal(q_actual, q_expected)
    np.testing.assert_array_equal(s_actual, s_expected)
    activation = FP8[q_expected].astype(np.float64) * np.repeat(np.exp2(s_expected.astype(np.float64) - 127), 128, axis=1)
    expected = activation @ unpack(w, ws, fp4).T + bias
    raw = driver.download(y, (rows, n), np.float32 if output_f32 else np.uint16)
    actual = raw if output_f32 else floats(raw)
    np.testing.assert_allclose(actual, expected, rtol=2e-5 if output_f32 else .004, atol=2e-6 if output_f32 else .002)
    return raw, np.max(np.abs(actual - expected), axis=0)


def projections(driver):
    rng = np.random.default_rng(198524)
    count, worst, worst_columns = 0, 0., []
    for fp4 in (0, 1):
        for input_f32 in (0, 1):
            for output_f32 in (0, 1):
                for k, n in [(128, 129), (512, 256), (4096, 31)]:
                    x = rng.normal(0, .1, (4, k)).astype(np.float32)
                    x[0] = 0
                    x[1, :16] = np.array([0, -0., .5, 1, 1.5, 2, 3, 4, 6, -.5, -1, -1.5, -2, -3, -4, -6]) * 2 ** -12
                    x = x if input_f32 else bf16(x)
                    if fp4:
                        w = rng.integers(0, 256, (n, k // 2), dtype=np.uint8)
                        ws = rng.integers(119, 125, (n, k // 32), dtype=np.uint8)
                    else:
                        w = rng.integers(0, 256, (n, k), dtype=np.uint8)
                        w[(w & 127) == 127] = 126
                        ws = rng.integers(110, 116, ((n + 127) // 128, k // 128), dtype=np.uint8)
                    bias = rng.normal(0, .01, n).astype(np.float32)
                    actual, columns = run_case(driver, x, w, ws, bias, fp4, input_f32, output_f32)
                    solo, _ = run_case(driver, x[2:3], w, ws, bias, fp4, input_f32, output_f32)
                    np.testing.assert_array_equal(actual[2], solo[0])
                    worst = max(worst, float(columns.max()))
                    worst_columns.append(float(columns.max()))
                    count += 1
    return dict(passed=count, max_abs=worst, worst_column_by_case=worst_columns, row_invariance=True)


def dequant(driver):
    rng = np.random.default_rng(821)
    w = rng.integers(0, 256, (256, 256), dtype=np.uint8)
    w[(w & 127) == 127] = 126
    scale = np.array([[120, 127], [128, 133]], dtype=np.uint8)
    y = driver.empty(w.size * 2)
    driver.launch("tf_dsv4_native_dequant", (w.size // 256, 1, 1), (256, 1, 1),
                  driver.upload(w), driver.upload(scale), y, u(256), u(256))
    np.testing.assert_array_equal(driver.download(y, w.shape, np.uint16), bf16(unpack(w, scale, False)))
    return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fatbin")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    driver = Driver(args.fatbin, args.device)
    try:
        report = dict(format_states=decode(driver), projections=projections(driver), wo_a_dequant=dequant(driver),
                      full_model_executed=False, dgx_spark_executed=False)
    finally:
        driver.close()
    text = json.dumps(report, indent=2)
    print(text)
    if args.report:
        args.report.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
