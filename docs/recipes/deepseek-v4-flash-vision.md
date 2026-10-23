# DeepSeek-V4 Flash Vision native FP4/FP8 and EXL3 development

This branch adds native building blocks for DeepSeek-V4-Flash-Vision-Exp on two
DGX Sparks, loading either the original FP4/FP8 or EXL3 checkpoint.
An experimental native **text/vision TP=2** server and rank-1 worker are now wired
into the existing CUDA registry and lane scheduler. See
[two-Spark build and launch instructions](deepseek-v4-text-tp2.md).
DSpark, distributed prefix snapshots and HTTP vision are connected. The complete
checkpoint has not been served on two Sparks here; local operator or vision-probe
results do not qualify the complete text/vision server.

The intended serving profile is 262,144 prompt-plus-output tokens per request,
four concurrent active requests, DSpark, prefix caching, streaming and tool
calls. The original FP4/FP8 checkpoint is now the primary implementation target.
EXL3 remains supported by the development probes; the user will perform final
serving validation on the two Sparks.

## Pinned inputs

| Input | Revision | Usage |
| --- | --- | --- |
| [TensorFold upstream](https://github.com/ashhart/TensorFold) | `f4d6c4f2` (rebased from `c7137bb3`) | Base engine and server |
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
- Separate original FP4/FP8 and EXL3 storage contracts; the original graph uses
  fused vision QKV/MLP weights and a grouped FP8 `wo_a` source matrix.
- Native FP8-activation x FP4/FP8-weight CUDA reference projections and original
  UE8M0 scales, including FP8-to-BF16 `wo_a` conversion. No expert expansion or
  requantization is used; uploads stage at most 8 MiB of source bytes at a time.
- A complete ViT/aligner graph and PNG/JPEG-to-positioned-embedding-span path.
  The original-weight graph passes the four recorded full-graph comparisons
  against the official CPU reference; target-device qualification remains open.
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
- Media-aware HTTP input, positioned image rows and two-rank prefix snapshots.
  Aligner outputs share the prefix-cache budget; snapshots never cut image spans.

## Serving integration and remaining qualification

Text prefill/decode, DSpark target taps and three draft stages, two-rank
transactional snapshots and PNG/JPEG HTTP input are connected to the existing
lane scheduler. Image spans prefill by layer with bidirectional visibility
inside each image; text and compressor/indexer queries retain their original
causal rules. Vision runs on rank zero and sends BF16 embedding spans to rank one.

DSpark proposals are currently greedy. Markov and confidence heads execute,
but confidence does not change the acceptance rule. The shared lane core
checks every proposal against its own target draw, including seeded sampling.
`TF_DSV4_DSPARK=1` explicitly enables this unqualified path; `--no-drafts`
and a request's `"draft": false` select plain decoding.

Local evidence covers state byte equality after rejection/partial acceptance,
slot transfer/reset and bounded image-cache reuse on an RTX 5070 Ti, plus host
protocol and HTTP regressions. It does **not** qualify real-checkpoint text or
DSpark logits, complete image-to-answer fidelity, two-Spark NCCL execution,
256K × 4 admission, cancellation under real peer failure or served throughput.
CUDA Graphs, fused batched execution and transport optimizations await that evidence.

See [two-Spark serving and hardware qualification](deepseek-v4-text-tp2.md).
Historical receipts below describe their recorded source state only.

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
scales without requantization, and publishes `manifest.json` last. Cache schema v2 adds the storage format, partition ABI
and a verified canonical identity digest; regenerate v1 caches. The reader also
requires an exact fingerprint of the storage/splitting/loading source, normalized
for Git line endings, so a relevant code change invalidates an existing cache. It never
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

## Original FP4/FP8 checkpoint path

The reference is the original model revision in the pinned-input table.
All 48 shard headers were inspected using bounded HTTP Range reads. The index
contains 72,633 tensors and 167,811,372,792 payload bytes (156.2865 GiB).
The required graph validates 35,328 FP4 expert projections, 390 FP8 projections,
258 dense projections and 936 raw tensors. Three optional DSpark `bias_vl`
tensors are unused because the draft graph receives text-only IDs.

FP4 expert matrices are `I8[N,K/2]` containers holding low-nibble-first E2M1
values, with `F8_E8M0[N,K/32]` scales. FP8 matrices are `F8_E4M3[N,K]` with
`F8_E8M0[ceil(N/128),K/128]` scales. Activation quantization is per-row, per-128
along K, using the official clamped maximum and power-of-two scale rounding.
I8 outside the expert graph does not imply FP4. Unsupported configuration,
missing/misshaped scales, mixed EXL3 storage, NaN FP8 and NaN scale bytes are
rejected. Original dense vision weights stay BF16.

The physical TP2 plan slices weights and their scale grids together. Both
ranks retain half of every expert's intermediate dimension; vision and image
special embeddings reside on rank 0. The inspected plan uses approximately
80.39 GiB of resident weights on rank 0 and 79.52 GiB on rank 1, including the
limited `wo_a` BF16 conversion. These are header-derived byte counts, not peak
process/device measurements. The plan also reports dtype subtotals, freed FP8
scales, source-buffer overlap during conversion and host staging. Add the KV
estimate and every snapshot, scratch, communication and OS reservation before
making any 256K/four-stream admission decision.

```sh
zig build test-dsv4-vision test-dsv4-cuda-host dsv4-inspect
zig build dsv4-project dsv4-vision -Dtarget=aarch64-linux-gnu
zig build fatbins -Dnvcc=nvcc -Dsm=121
zig-out/bin/tf-dsv4-inspect /models/vision-original --plan > original-plan.json
python3 tools/dsv4-vision/prepare_cache.py \
  /models/vision-original /models/vision-original-tp2 \
  --inspector zig-out/bin/tf-dsv4-inspect
zig-out/bin/tf-dsv4-project /models/vision-original \
  layers.0.ffn.experts.0.w1 4096 2048 input.bf16 output.f32 1 zig-out/fatbin
```

The projection probe chooses modules by checkpoint format. Original weights
need `dsv4_native.fatbin` and `dsv4_tensor.fatbin`; they do not need EXL3 kernels.
It supports complete source checkpoints and validates the selected matrix.
The rank-cache reader is a family API; the probe does not accept a cache
manifest as a complete source checkpoint.

The vision probe takes BF16 channel-major patches and returns raster-ordered
BF16 aligner embeddings. Optional `TRACE_DIR` writes each block's hidden state
for teacher-forced debugging. The `--image` variant runs native image decoding,
preprocessing, the complete tower and N-layout/special-embedding assembly; its
JSON output names token IDs, image span and cache content/preprocessing hashes.
These are offline tools, not serving commands.

```sh
zig-out/bin/tf-dsv4-vision /models/vision-original patches.bf16 \
  28 28 aligned.bf16 zig-out/fatbin
zig-out/bin/tf-dsv4-vision --image /models/vision-original picture.png \
  span.bf16 zig-out/fatbin 0
```

On Windows development hosts, `dsv4-project-host` and `dsv4-vision-host` build
CUDA probes using `nvcuda.dll`. This does not add Windows server support.
The target server remains Linux ARM64/GB10. JPEG on Linux needs the system
TurboJPEG library; its optional explicit path follows `POSITION`.

### Numerical checks and corrected vision gate

```sh
python3 tools/dsv4-vision/check_native_cuda.py zig-out/fatbin/dsv4_native.fatbin \
  --report native-operators.json
python3 tools/dsv4-vision/check_checkpoint_projection.py /models/vision-original \
  --probe zig-out/bin/tf-dsv4-project --kernels zig-out/fatbin --report projections.json
python3 tools/dsv4-vision/check_vision_cuda.py zig-out/fatbin/dsv4_vision.fatbin \
  --report vision-operators.json
python3 tools/dsv4-vision/check_vision_graph.py /models/vision-original \
  --probe zig-out/bin/tf-dsv4-vision --kernels zig-out/fatbin \
  --official-reference /reference/inference/vision.py \
  --grid 28x28 --grid 24x32 --report vision-graph.json
```

PyTorch is optional and used only by the last development check; the supplied
reference file must match the pinned revision's SHA256. Without that option,
the check uses the independent NumPy reference. Tiny 3x4 and 5x3 stress grids
are the default and are retained because they exposed numerical sensitivity.
All cases write a JSON receipt, and a failed full-graph comparison exits 1.

The local GPU passed 528 format values, 24 native projection combinations,
single-row/batched-row equality, the `wo_a` dequantization check and six actual
original-weight projections. The real matrices cover FP8 attention, FP4 w1/w2,
grouped `wo_a`, fused vision QKV and fused vision MLP. Independent FP64 maximum
absolute projection errors ranged from 1.46e-8 to 1.24e-7. Vision operator tests
also cover FP32/BF16/F16 normalization, fused splits, separate Q/K/V attention,
the BF16 SiLU boundary and N-layout special-embedding assembly.

The complete 32-layer ViT and aligner pass the official PyTorch CPU comparison
using the actual original vision weights. The fixed `rtol=0.04, atol=0.04`
tolerances, seed 22361 and input generation are unchanged.

| Patch grid | Previous maximum absolute error | Corrected maximum absolute error | Result |
| --- | ---: | ---: | --- |
| 28x28 | 0.052734375 | 0.03955078125 | Pass |
| 24x32 | 0.0703125 | 0.044921875 | Pass |

The gate checks each element against `atol + rtol * abs(reference)`, so a
maximum absolute error above 0.04 can still pass. The largest tolerance ratios
are 0.948001 and 0.832127 respectively; neither case has an out-of-tolerance
element. A repeat run produced identical output hashes. The separate default
tiny-grid run also passes: 3x4 has maximum error 0.00439453125 and 5x3 has
0.023681640625.

RoPE now rounds the reciprocal frequency before multiplying by position, as
the official implementation does. Attention QK dot products and probability/V
sums use compensated FP32 accumulation to retain small terms during
cancellation. Activations and output boundaries remain BF16; weights and
normalization precision are unchanged. No tolerance was loosened.

Two independent FP64 cancellation regressions fail on the preceding kernel
and pass on the corrected kernel, matching FP64 results rounded to BF16.
Their maximum errors fall from 0.499569 to 0.000431 (QK) and from 0.007691 to
0.000121 (PV). All 12 vision operator checks and 15 tensor checks pass.
These local original-weight results do not qualify EXL3 vision or DGX Spark.

Those receipts predate the DSpark, distributed prefix snapshots and vision
API integration now implemented. Real NCCL execution, full-model numerical checks
and the user's two-Spark 256K/four-active-request qualification remain unrun.
No throughput or memory-fit result for that profile has been measured here.

The local PNG and JPEG integration check runs the complete vision graph twice
per image, once from patches and once through `--image`, and requires exact
N-layout span bytes, token IDs and positions. Both cases pass. This checks
pipeline wiring within this implementation and is separate from the official
full-graph comparison. Both cases were rerun after the arithmetic correction.

```sh
python3 tools/dsv4-vision/check_media_pipeline.py /models/vision-original \
  --image-probe zig-out/bin/tf-dsv4-image \
  --vision-probe zig-out/bin/tf-dsv4-vision --kernels zig-out/fatbin \
  --jpeg-library /usr/lib/aarch64-linux-gnu/libturbojpeg.so.0 \
  --report media-pipeline.json
```

Cache row ranges use compact strided descriptors instead of materializing
millions of small row tuples. Selection metadata peaked at approximately
30 MiB for the original checkpoint and 56 MiB for the inspected EXL3 checkpoint
in the local Python check, excluding the separately loaded headers and plan.
The native inspector and Python writer agree on every selected tensor and
rank payload byte count for both complete header catalogs.

The earlier [native checkpoint receipt](../../tools/dsv4-vision/receipts/2026-10-16-native-checkpoint.json)
records the pre-fix successes and unresolved gates separately. The new
[vision correction receipt](../../tools/dsv4-vision/receipts/2026-10-16-vision-fix.json)
records before/after errors, per-column differences, regression tests, source
hashes and the unchanged qualification limits. The
[initial vision failure](../../tools/dsv4-vision/receipts/2026-10-16-native-vision-before.json)
is retained for the normalization comparison.

The 2026-10-16 host run covered 29 family contract tests and 43 CUDA-runtime tests.
`check-dsv4-target -Dtarget=aarch64-linux-gnu` also cross-compiles the complete
family contract test executable; it does not run that executable on Windows.
The native server, projection probe and vision probe cross-build for ARM64.
That historical lean receipt recorded 527 inherited findings with zero new findings.
