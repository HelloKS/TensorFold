# DeepSeek V4 text/vision serving on two DGX Sparks

This is an experimental native TP=2 path for the pinned Vision-Exp checkpoint,
with original packed FP4/FP8 or EXL3 rank caches. Rank 0 runs the existing HTTP
server and lane scheduler; rank 1 runs `tf-dsv4-worker`. Every expert is split
along its intermediate dimension, and attention heads/groups are split in two.
NCCL gathers FP32 partials, which the ranks add in rank order.

DSpark, distributed prefix snapshots and HTTP PNG/JPEG images are now connected.
Whole-checkpoint model fidelity, two-Spark NCCL execution, 256K × 4 admission
and throughput have **not been qualified here**. Local GPU state tests and
successful cross-builds do not establish those results. The user performs the
hardware runs below with a complete, immutable checkpoint.

## Build and prepare

Use the repository's pinned Zig, CUDA 13.1 and `libnccl.so.2` satisfying the
shared runtime's minimum (2.30.0 in this checkout). JPEG needs system TurboJPEG.
No Python inference process or captured kernel kit is required.

```sh
zig build native dsv4-worker dsv4-verify dsv4-inspect dsv4-image fatbins \
  -Dtarget=aarch64-linux-gnu -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121
python3 tools/dsv4-vision/prepare_cache.py /models/dsv4-original \
  /models/dsv4-tp2 --inspector zig-out/bin/tf-dsv4-inspect
```

Both nodes need identical `manifest.json`, config and source/kernel revisions.
Each needs its own `rank0.safetensors` or `rank1.safetensors`. Rank 0 also needs
the tokenizer files. The reader checks hashes and shapes; the writer splits
original packed payload/scales losslessly. It does not requantize weights.
Regenerate a cache when the inspector reports an incompatible cache ABI.

## Serve

On both nodes, replace the reserved address with the head's fabric address:

```sh
export TF_TP_WORLD=2
export TF_TP_MASTER=192.0.2.10
export TF_TP_PORT=29620
export TF_DSV4_RANK_CACHE=/models/dsv4-tp2
export TF_DSV4_KERNELS="$PWD/zig-out/fatbin"
export TF_DSV4_DSPARK=1
export TF_EXPERIMENTAL=deepseek
```

The pinned release gate refuses `deepseek_v4` on the CUDA server unless
`TF_EXPERIMENTAL=deepseek` is set; DeepSeek-V4 CUDA is experimental while
whole-model gate [#827](https://github.com/ashhart/TensorFold/pull/827) is pending.
`TF_TP_WORLD` / `TF_TP_MASTER` / `TF_TP_PORT` here are this family's private TP2
config, not the shared `fabric/tp` namespace: `TF_TP_MASTER` is a dotted-quad only
and this path takes no `TF_TP_RANK`.

Configure `NCCL_SOCKET_IFNAME` / `NCCL_IB_HCA` for the actual CX7 interfaces.
DSpark remains opt-in until numerical and performance qualification. Unset
`TF_DSV4_DSPARK` or use `--no-drafts` to disable it; an individual request can
use `"draft": false` as the plain oracle. The initial proposer uses greedy
DSpark candidates. The lane core accepts only its own target draws, including
seeded sampling. Confidence scores do not override that rule.

On rank 1:

```sh
export TF_DSV4_WORKER_RECEIPT=validation-rank1.json
zig-out/bin/tf-dsv4-worker /models/dsv4-original 262144 "$TF_DSV4_KERNELS"
```

A peer that dies today costs up to the ~10 s control timeout (the `TF_TP` control
deadline / `PeerTimeout`). Upstream's rank contract stops a rank in ~1 s with a
named peer on `cuda.tp_watch`, which this path does not use yet; that faster peer
watch is a documented gap and a follow-up.

On rank 0:

```sh
zig-out/native/bin/tensorfold-native serve /models/dsv4-original \
  --context 262144 --parallel 4 --vision --vision-urls --prompt-cache-gib 16 \
  --host 0.0.0.0 --port 8080 --name deepseek-v4 --no-thinking
```

Omit `--vision-urls` to accept data URLs alone. HTTP image sources accept base64
or bounded public HTTPS fetching with redirect/address validation; local file
paths are refused. Chat Completions, Responses and Anthropic Messages normalize
to the existing common image path. Videos are unsupported by this family.

Image rows use native RGB processing, patch size 14 and the N-layout. Every
image occupies at most 384 rows including compression padding; four images
have a default total allowance of 1536. `--vision-image-tokens` can lower the request
allowance. Position-dependent row counts enter context admission and usage.
Image parts must be separate content blocks, for example:

```json
{"model":"deepseek-v4","messages":[{"role":"user","content":[
  {"type":"text","text":"Describe the image."},
  {"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}
]}],"max_tokens":64,"stream":true}
```

Vision runs on rank 0 and sends final BF16 spans to rank 1. Each span prefills
by layer so queries can see the entire same image; compression/indexer paths
remain causal. Chunk and snapshot boundaries do not split an image.

## Cache, rollback and memory

The normal prefix store owns snapshot identity, lookup, borrowing and eviction.
Each snapshot is published only after both ranks finish saving. Failure on
one rank discards both copies. Restore/drop operations address the same saved
ID on both ranks. Restoring copies KV, compressor, indexer, DSpark KV, target
taps and main hidden state. A failed restore stops further inference rather
than reusing uncertain state. Verify rollback restores overwritten rows in
reverse order, including partial rejection and cancellation.

`--prompt-cache-gib` charges both rank snapshots together. Up to one eighth
of that budget, capped at 64 MiB, is reserved for rank-zero aligner outputs.
Those entries use SHA256 of prepared pixels, grid, preprocessing and execution
identity and are evicted by recency. Position-specific packing is applied
again on a hit. `--prompt-cache-gib 0` disables both retained caches.

The ranks negotiate the smaller remaining memory pool before cache admission.
Admission also accounts for four active sequence states, rollback journals,
image rows, draft state, staging/scratch and OS reserve. Host memory counts
on unified-memory Sparks. An oversized context/concurrency is refused; the
server does not silently reduce the requested 256K × 4 profile.

The reference graph processes rows serially within shared lane rounds. No CUDA
Graph, fused batching or alternate RDMA optimization is enabled by this port.

## Hardware token-ID qualification

Stop the HTTP process first. Run a fresh worker with the **same context** as
the verifier. Each verifier run stops that worker when it completes. Repeat
8K, 32K, 128K and 256K at concurrency 1, 2 and 4, restarting the worker each time.
Keep `TF_DSV4_DSPARK=1` and the same rank-cache/kernel settings on both hosts.

Fixture generation needs the development-only Python `tokenizers` package.
The final profile has exactly the requested prompt-plus-output count:

```sh
mkdir -p validation
python3 tools/dsv4-vision/make_serving_fixture.py /models/dsv4-original \
  validation/256k-4.json --context 262144 --concurrency 4 --max-tokens 64
# rank 1: restart tf-dsv4-worker with context 262144 before the following command
zig-out/bin/tf-dsv4-verify /models/dsv4-original validation/256k-4.json \
  262144 "$TF_DSV4_KERNELS" validation/rank0-256k-4.json
```

For positioned images, pass up to four local validation images and the native
CPU preprocessor. These file paths belong to the CLI fixture, not HTTP inputs:

```sh
python3 tools/dsv4-vision/make_serving_fixture.py /models/dsv4-original \
  validation/images-8k-4.json --context 8192 --concurrency 4 \
  --image validation/square.png --image validation/portrait.jpg \
  --image-probe zig-out/bin/tf-dsv4-image
```

The verifier runs the same backend as HTTP and fails on token-ID differences
between plain/DSpark, shared/solo, reset/cached and after cancellation. It fails
if no DSpark proposal was exercised or if the selected prefix was not restored.
Its JSON includes emitted IDs, draft/acceptance totals, cache counts and rank-0
memory before/after cleanup. `TF_DSV4_WORKER_RECEIPT` records the matching
rank-1 memory; counters labelled tracked exclude NCCL/driver internal storage.
Retain physical memory telemetry from both hosts as well. Reset-state cold
comparisons here are not cold-process TTFT measurements.

After correctness passes, test HTTP text/image streaming, reasoning, tools and
usage in all three APIs. Compare changed-image requests against fresh cache-off
runs. Record peer shutdown and reconnect failures. Measure TTFT, prefill/decode,
NCCL time, memory and draft acceptance at least three times with the existing
benchmark clients, alternating plain/DSpark and cache off/on. Only measured
improvements should become deployment defaults. Keep the checkpoint hash,
fixture metadata, commands, receipts and stderr logs with those results.
