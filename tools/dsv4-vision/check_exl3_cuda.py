"""Check source-built EXL3 CUDA kernels against an independent scalar-format/FP64 oracle."""
import argparse
import ctypes as C
import json
import os
from pathlib import Path

import numpy as np


class Params(C.Structure):
    _fields_ = [(n, C.c_uint32) for n in (
        "k", "n", "rows", "halves", "codebook", "multiplier", "su_signs", "sv_signs", "input_f32", "output_f32"
    )] + [("tile_k_stride", C.c_uint64), ("tile_n_stride", C.c_uint64)]


class Driver:
    def __init__(self, fatbin, device):
        self.lib = C.WinDLL("nvcuda.dll") if os.name == "nt" else C.CDLL("libcuda.so.1")
        self.allocations = []
        self.context = C.c_void_p()
        self.module = C.c_void_p()
        self.call("cuInit", 0)
        self.call("cuCtxCreate_v2", C.byref(self.context), 0, device)
        try:
            image = C.create_string_buffer(Path(fatbin).read_bytes())
            self.call("cuModuleLoadData", C.byref(self.module), image)
        except BaseException:
            self.close()
            raise

    def call(self, name, *args):
        status = getattr(self.lib, name)(*args)
        if status:
            raise RuntimeError(f"{name}: CUDA error {status}")

    def close(self):
        for ptr in self.allocations:
            self.call("cuMemFree_v2", ptr)
        self.allocations.clear()
        if self.module.value:
            self.call("cuModuleUnload", self.module)
            self.module = C.c_void_p()
        if self.context.value:
            self.call("cuCtxDestroy_v2", self.context)
            self.context = C.c_void_p()

    def upload(self, array):
        ptr = self.empty(array.nbytes)
        self.call("cuMemcpyHtoD_v2", ptr, C.c_void_p(array.ctypes.data), C.c_size_t(array.nbytes))
        return ptr

    def empty(self, size):
        ptr = C.c_uint64()
        self.call("cuMemAlloc_v2", C.byref(ptr), C.c_size_t(size))
        self.allocations.append(ptr)
        return ptr

    def download(self, ptr, shape, dtype):
        out = np.empty(shape, dtype=dtype)
        self.call("cuMemcpyDtoH_v2", C.c_void_p(out.ctypes.data), ptr, C.c_size_t(out.nbytes))
        return out

    def launch(self, name, grid, block, *args, shared=0):
        fn = C.c_void_p()
        self.call("cuModuleGetFunction", C.byref(fn), self.module, name.encode())
        values = (C.c_void_p * len(args))(*(C.addressof(a) for a in args))
        self.call("cuLaunchKernel", fn, *grid, *block, shared, C.c_void_p(), values, C.c_void_p())
        self.call("cuCtxSynchronize")


MULTIPLIERS = [89226354, 0xCBAC1FED, 0x83DCD12D]


def codebook(cb):
    states = np.arange(65536, dtype=np.uint64)
    x = (states * MULTIPLIERS[cb]) & 0xFFFFFFFF
    if cb == 2:
        h = 1024 + (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + (x >> 24)
        scale = np.array([0x1EEE], dtype=np.uint16).view(np.float16)[0].astype(np.float64)
        bias = np.array([0xC931], dtype=np.uint16).view(np.float16)[0].astype(np.float64)
        return (h.astype(np.float64) * scale + bias).astype(np.float16)
    if cb == 0:
        x = (x + 64248484) & 0xFFFFFFFF
    x = (x & 0x8FFF8FFF) ^ 0x3B603B60
    lo = (x & 65535).astype(np.uint16).view(np.float16).astype(np.float64)
    hi = (x >> 16).astype(np.uint16).view(np.float16).astype(np.float64)
    return (lo + hi).astype(np.float16)


def unpack(trellis, halves, cb):
    kt, nt, _ = trellis.shape
    out = np.empty((kt * 16, nt * 16), dtype=np.float64)
    table = codebook(cb).astype(np.float64)
    ring = 128 * halves
    for kr in range(kt):
        for nc in range(nt):
            words = trellis[kr, nc].astype(np.uint32)
            pairs = words[0::2] | (words[1::2] << 16)
            bits = [(int(word) >> b) & 1 for word in pairs for b in range(31, -1, -1)]
            for p in range(256):
                end = ((p + 1) * halves - (((p + 1) % 2) if halves % 2 else 0)) // 2
                state = 0
                for b in range(16):
                    state = state * 2 + bits[(end - 16 + b) % ring]
                lane, j = divmod(p, 8)
                row = 2 * (lane % 4) + (j & 1) + 8 * ((j >> 1) & 1)
                col = lane // 4 + 8 * (j >> 2)
                out[kr * 16 + row, nc * 16 + col] = table[state]
    return out


def hadamard(x, dtype=np.float64):
    out = np.asarray(x, dtype=dtype).copy().reshape(-1, 128)
    for step in (1, 2, 4, 8, 16, 32, 64):
        for i in range(0, 128, step * 2):
            a, b = out[:, i:i + step].copy(), out[:, i + step:i + 2 * step].copy()
            out[:, i:i + step], out[:, i + step:i + 2 * step] = a + b, a - b
    return (out * dtype(1 / np.sqrt(128))).reshape(x.shape)


def check(driver):
    output = driver.empty(3 * 65536 * 2)
    driver.launch("tf_exl3_codebook_check", (768, 1, 1), (256, 1, 1), output)
    got = driver.download(output, (3, 65536), np.float16)
    for cb in range(3):
        np.testing.assert_array_equal(got[cb].view(np.uint16), codebook(cb).view(np.uint16))
    rng = np.random.default_rng(12345)
    worst = 0.0
    cases = 0
    for cb in range(3):
        for halves in ([2, 4, 6, 8, 10, 16] if cb < 2 else [2, 3, 4, 5, 6, 7, 8, 10, 16]):
            k, n, rows = 128, 256, 4
            trellis = rng.integers(0, 65536, (k // 16, n // 16, halves * 8), dtype=np.uint16)
            su = (rng.uniform(0.7, 1.3, k) * rng.choice([-1, 1], k)).astype(np.float16)
            sv = (rng.uniform(0.7, 1.3, n) * rng.choice([-1, 1], n)).astype(np.float16)
            x = rng.normal(0, 0.5, (rows, k)).astype(np.float32)
            bias = rng.normal(0, 0.1, n).astype(np.float16)
            xp, tp, up, vp, bp = [driver.upload(a) for a in [x, trellis, su, sv, bias]]
            rotated, sums, y = driver.empty(rows * k * 2), driver.empty(rows * n * 4), driver.empty(rows * n * 4)
            params = Params(k, n, rows, halves, cb, MULTIPLIERS[cb], 0, 0, 1, 1,
                            n // 16 * halves * 16, halves * 16)
            driver.launch("tf_exl3_rotate_in", (k // 128, rows, 1), (128, 1, 1), xp, up, rotated, params)
            driver.launch("tf_exl3_matmul", (n // 128, rows, 1), (128, 1, 1), rotated, tp, sums, params)
            driver.launch("tf_exl3_rotate_out", (n // 128, rows, 1), (128, 1, 1), sums, vp, bp, y, params)
            actual = driver.download(y, (rows, n), np.float32)
            xr = hadamard(x * su.astype(np.float32), np.float32).astype(np.float16)
            np.testing.assert_array_equal(driver.download(rotated, (rows, k), np.float16), xr)
            xr = xr.astype(np.float64)
            expected = hadamard(xr @ unpack(trellis, halves, cb)) * sv + bias
            error = float(np.max(np.abs(actual - expected)))
            worst = max(worst, error)
            np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=2e-4)
            params.rows = 1
            driver.launch("tf_exl3_rotate_in", (k // 128, 1, 1), (128, 1, 1), xp, up, rotated, params)
            driver.launch("tf_exl3_matmul", (n // 128, 1, 1), (128, 1, 1), rotated, tp, sums, params)
            driver.launch("tf_exl3_rotate_out", (n // 128, 1, 1), (128, 1, 1), sums, vp, bp, y, params)
            np.testing.assert_array_equal(driver.download(y, (1, n), np.float32)[0], actual[0])
            cases += 1
    return {"codebook_states": 196608, "projection_cases": cases, "worst_abs_error": worst,
            "row_invariant": True, "model_qualified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fatbin")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    driver = Driver(args.fatbin, args.device)
    try:
        report = check(driver)
    finally:
        driver.close()
    text = json.dumps(report, indent=2)
    print(text)
    if args.report:
        args.report.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
