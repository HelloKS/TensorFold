# DeepSeek-V4 Flash Vision EXL3 development

This branch adds native building blocks for DeepSeek-V4-Flash-Vision-Exp on two
DGX Sparks. **It does not yet provide an end-to-end serving backend.** The CUDA
server's family registry has not been extended to advertise this model. An
offline storage check, a successful projection probe or synthetic GPU checks
must not be reported as a qualified text/vision server.

The intended serving profile is 262,144 prompt-plus-output tokens per request,
four concurrent active requests, DSpark, prefix caching, streaming and tool
calls. The user will perform final serving validation on the two Sparks, using
a complete 4bpw checkpoint when that checkpoint is available.

## Pinned inputs

| Input | Revision | Usage |
| --- | --- | --- |
| [TensorFold upstream](https://github.com/ashhart/TensorFold) | `c7137bb31a7dad924fb541a7068422127b9a94b2` | Base engine and server |
| [Official Vision-Exp model](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp) | `6821d6ad3681a4b137b066b76094fa82ebd0a380` | Model semantics and image oracle |
| [Turboderp EXL3, 3.04bpw](https://huggingface.co/turboderp/DeepSeek-V4-Flash-Vision-Exp-exl3/tree/3.04bpw) | `1480a917a829ce4b1c6faa97a0d64195e9e05055` | Inspected storage layout |
| [Bertholomus V4.1 branch](https://github.com/bertholomus/TensorFold/tree/deepseek-v41-zig) | `dfc8fe50af303fcbdda0e839cae3a18eb53be2e1` | Selective PNG/RGB port and integration reference |

The public EXL3 config's average bpw is not a projection width. Widths are read
from each trellis tensor; the inspected checkpoint mixes 3-bit and 5-bit
projections, including quantized vision projections. It also retains redundant
original fused ViT weights. The required graph explicitly selects the split
quantized q/k/v and gate/up projections rather than silently falling back to
those original tensors.

## Implemented components

- Config validation for the 43-layer V4 text model, the 32-layer ViT and DSpark.
- A complete required-tensor graph, including image embeddings, `bias_vl`, hash
  routing tables, compressed attention, all experts and the DSpark heads.
- Bounded local safetensors-header inspection and individual tensor reads, with
  index/file consistency, duplicate-name and shape/dtype checks.
- EXL3 `inst3`, `mcg` and `mul1` CUDA projections, including per-tensor widths,
  multipliers and packed-sign scales. Source kernels need no captured kit.
- CUDA routing, Sinkhorn/mHC mixing, FP8/FP4 QAT simulation, rotary embeddings,
  sparse attention with sinks, pooling, SwiGLU and aligner unfolding primitives.
- Explicit-dtype dense projections, embeddings, HC taps/collapse and ViT rotary
  primitives, and native checkpoint-to-device matrix uploads.
- Native PNG and JPEG decoding, Pillow-compatible RGB resampling, V4 geometry,
  N-layout, BF16 patchification and image-safe chunk boundaries.
- TP2 partition rules and a lossless rank-cache writer with source/payload
  SHA256 hashes. The native reader verifies schema, config, payload and local
  tensor shapes. Vision weights are stored on rank 0 only.
- Media-aware cache-key and snapshot-boundary helpers. These helpers are not
  yet wired into the engine's prefix cache.

## Remaining integration

The complete model forward, incremental compressed attention/indexer state,
two-rank command/NCCL execution and the layer graph remain to be connected.
DSpark verification/rollback, shared lane rounds, prefix snapshot save/restore,
image-bearing HTTP requests and capability/CLI registration also remain.
CUDA Graph and RDMA optimizations are not enabled.

No `serve` command is documented for this model until those paths exist. The
provided probes are development tools, not replacements for an engine.

## Build and inspect

Use Zig 0.17.0, exactly as pinned in `.zig-version`.

```sh
zig build test-dsv4-vision test-dsv4-cuda-host dsv4-inspect dsv4-image
zig-out/bin/tf-dsv4-inspect /models/vision-exl3-4bpw
```

Inspection needs the complete checkpoint, tokenizer and config for subsequent
cache preparation. It validates shapes without loading GPU weights. The
`--catalog HEADERS.json` development option validates a merged JSON header
catalog; it cannot verify tensor payload contents or model execution.

On the inspected public 3.04bpw revision, all 146,663 indexed tensor names were
found in the sixteen shard headers. The graph validated 36,391 packed
projections, three dense projections and 1,000 raw tensors. The TP plan selects
approximately 55.954 GiB on rank 0 and 55.680 GiB on rank 1. These are **3.04bpw
storage figures**, not 4bpw memory measurements.

### Lossless rank caches

```sh
python3 tools/dsv4-vision/prepare_cache.py \
  /models/vision-exl3-4bpw /models/vision-exl3-4bpw-tp2 \
  --inspector zig-out/bin/tf-dsv4-inspect
```

Use a new, empty output directory outside the source checkpoint. Preparation
hashes the entire source before and after copying, preserves trellis words and
scales without requantization, and publishes `manifest.json` last. It never
overwrites an existing cache. An interrupted directory without a final manifest
is not a usable cache. Both ranks must use the same manifest identity.

The 256K/four-stream persistent KV/working-compressor estimate is about 6.79 GiB
per rank. It excludes weights, prefix snapshots, vision/tensor scratch,
verification backups, communication buffers and the operating-system reserve.
The future serving admission check must include every term; this estimate is
not a claim that a checkpoint and four requests fit.

### Native real-weight projection probe (Linux ARM64)

```sh
zig build fatbins -Dnvcc=nvcc -Dsm=121
zig build dsv4-project -Dtarget=aarch64-linux-gnu
zig-out/bin/tf-dsv4-project /models/vision-exl3-4bpw \
  layers.0.attn.wq_a 4096 1024 input.bf16 output.f32 1 zig-out/fatbin
```

Input is a contiguous little-endian BF16 matrix of shape `[ROWS,K]`; output is
FP32 `[ROWS,N]`. The probe loads the checkpoint's matrix and bias, widens dense
weights only where requested, and runs the source-built CUDA kernels. It is
limited to one projection and does not run a decoder.

### Independent operator and image checks

Python is used only by development checks and cache preparation. Install NumPy
and Pillow for these checks; the native image path uses a system TurboJPEG
library (`libturbojpeg.so.0` on Linux).

```sh
python3 tools/dsv4-vision/check_exl3_cuda.py zig-out/fatbin/exl3.fatbin
python3 tools/dsv4-vision/check_ops_cuda.py zig-out/fatbin/dsv4_ops.fatbin
python3 tools/dsv4-vision/check_tensor_cuda.py zig-out/fatbin/dsv4_tensor.fatbin
python3 -m unittest discover -s tools/dsv4-vision -p 'test_*.py'
python3 tools/dsv4-vision/check_images.py zig-out/bin/tf-dsv4-image \
  /models/vision-exl3-4bpw/config.json /reference/inference/image_processor.py \
  --jpeg-library /usr/lib/aarch64-linux-gnu/libturbojpeg.so.0
```

The image oracle executes the pinned official preprocessing functions using
Pillow and a minimal NumPy tensor adapter. It compares exact BF16 patch bytes,
resize geometry, image types and aligner permutation, across six placement
positions for each generated fixture, including RGB, RGBA, grayscale, palette,
16-bit PNG and CMYK JPEG.

## Recorded local validation

Local synthetic GPU validation used an RTX 5070 Ti 16GB with CUDA 13.1, not a
DGX Spark. EXL3 checks passed every state of all three codebooks (196,608 total),
21 projection cases and single-row/batched-row equality. The largest absolute
projection error against the conditioned FP64 oracle was `3.7757064e-6`.
The input Hadamard/FP16 rounding is checked separately, bit for bit.

Nine model-operator cases and fifteen tensor-primitive cases passed independent
NumPy comparisons. Eighty-four native PNG/JPEG layout/patch cases matched the
official preprocessing bytes exactly. These checks do not qualify the full
model or guarantee its final decoded tokens.

The Linux ARM64 real-weight projection probe cross-compiles. The full upstream
CUDA/server host suite currently fails to compile on Windows due to inherited
POSIX calls and file-descriptor assumptions; use Linux for that suite. The
focused family and CUDA-runtime host suites run on the local Windows host.

Final user-run model validation must cover plain/DSpark, cold/reused prefixes,
solo/four-stream replies, cancellation and rollback, image-content cache
isolation, 8K/32K/128K/256K prompts, peak memory on both ranks and served API
behavior. No tok/s or DGX Spark qualification claim has been made.

## Attribution

README source bullets, `NOTICE`, `THIRD_PARTY_NOTICES.md` and `LICENSES/` retain
the applicable credits. MiaAI-Lab's AGPL GLM deployment recipe is used only as
an operational reference; its code is not copied. The native JPEG wrapper calls
the [TurboJPEG C API](https://github.com/libjpeg-turbo/libjpeg-turbo/blob/3.1.1/src/turbojpeg.h)
and includes no libjpeg-turbo implementation or binary.
