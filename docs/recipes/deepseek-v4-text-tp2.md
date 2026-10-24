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

`TF_DSV4_SKIP_PAYLOAD_HASH=1` skips the full rank-file SHA256 scan at startup.
Set it on each node whose cache you trust; unset it or set `0` for full checking
(the default). A warning records the skip. File size, manifest identity, config,
cache ABI and tensor layout checks still run, but same-size payload corruption
is no longer detected. This option does not skip source hashing during cache
preparation. Rebuild both binaries and the inspector after upgrading; adding
this option changes the source-based cache ABI, so prepare a new cache in an
empty directory with the rebuilt inspector before using it.

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

A peer that dies on the default path costs up to the ~10 s control timeout
(the `TF_TP` control deadline / `PeerTimeout`). The `TF_DSV4_WATCH=1` opt-in
below wires this path to upstream's rank contract — a named peer on
`cuda.tp_watch` that stops a rank in ~1 s — which `tf-dsv4-worker` does not
use by default yet; the two-Spark peer-kill qualification pass below covers
it before it becomes the deployment default.

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

## Performance opt-ins (default off)

The default path keeps the reference semantics: serial rows within shared lane
rounds, NCCL for every rank exchange, bf16 caches, per-sequence KV windows and
host-side snapshots. Each setting below opts into a faster path — except
`TF_DSV4_WC`, which is on by default and documented here for its opt-out — and
none is hardware-qualified yet, so the qualification pass below must measure
them before any becomes a deployment default.

`TF_DSV4_BATCH=1` — set on both ranks. Several streams' rows share one verify
and one FP32 rank exchange per lane round instead of running serially. Both
ranks derive the round plan identically from the broadcast rows, and the wire
is unchanged: no new command kinds, so mixed-version ranks still speak.
`TF_DSV4_ROWS_CAP` caps a round at a bucketed row count (default 24);
`TF_DSV4_SERIAL=1` forces the serial path as the fail-safe. Batched and serial
runs are token-identical in the host fixtures.

`TF_DSV4_KV_PAGES=1` — set on both ranks. The streams' KV comes from one page
pool on the shared core page layout: streams beyond the engine's own window
take pages as needed, kept prompt states become budget-sized slots, and when
the asked prompt cache does not fit beside the streams the states go to pinned
host memory instead of refusing. Unset, each stream keeps its own full window
and the startup accounting reads exactly as before.

`--kv-cache fp8` on rank 0 with `TF_DSV4_KV8=1` on rank 1 — KV rows are stored
as e4m3 codes with one FP32 scale per row (516 bytes per 512-wide row, about
half the cache bytes); the sparse gather dequantizes selected rows into a
per-sequence BF16 staging buffer. The cache width is part of the rank identity
digest: a pair that disagrees on it refuses with `RankIdentityMismatch` rather
than mixing row shapes. The fp8 kernels compile in the family's kernel section
and their nvcc verification belongs to the qualification pass.

`TF_DSV4_WC` — defaults ON; set `TF_DSV4_WC=0` on both ranks for the static
admission arithmetic. At open the family measures per-row-width forward costs
(two warm plus one timed pass per width on scratch slots, at most six widths
with DSpark) and feeds the lane scheduler measured costs instead of estimates,
the same boot-time calibration the other CUDA lane families run by default.
Any failed measurement warns and falls back to the static arithmetic — boot is
never blocked. The setting is part of the rank identity digest: a measuring
rank and a silent rank refuse each other rather than admitting lanes under
different costs.

`TF_DSV4_P2P=1` — set on both ranks (rank 0 server and rank 1 worker). After
the NCCL bootstrap the ranks exchange staging handles over the command link
(the `TFP2P2` offer and `P2PA` ack); the FP32 expert partial all-add then runs
over CUDA IPC staging — two sequence-parity slots and a mailbox with stream
memop doorbells — instead of an NCCL all-gather and local add. Requires peer
access in both directions and stream memop support; any decline, missing
symbol, failed mapping or vote timeout falls back to NCCL on both ranks,
quietly, at open. Every round within the 24-row cap fits the staging; larger
exchanges fall back on both ranks together. Mixed versions fail loud: an old
follower meeting the new offer refuses at startup (upgrade both ranks
together), while a new follower behind an old leader waits at most 2 s and
then uses NCCL. A mid-stream P2P failure poisons the communicator rather than
rerouting one round — a split sequence is communicator death.

`TF_DSV4_ROCE=1` — set on both ranks, and `TF_ROCE` must satisfy the shared
RoCE requirement (`cuda/roce.zig`; interface selection via `TF_ROCE_HCA` /
`NCCL_IB_HCA`). The probe frame carries a roce vote; on agreement the first
all-gather lazily brings up the RoCE group, where the hardware truth votes
inside the shared transport's NCCL swap and both ranks reach that decision
together. A one-rank env split aborts loudly by design — there is no silent
half-RDMA. Either side declining (no HCA, unmet `TF_ROCE`) leaves both ranks
on the typed `ncclAllGather`, byte-identical to the default.

`TF_DSV4_WATCH=1` — set on both ranks. Opens the shared `cuda.tp_watch`
liveness lines **before** the NCCL communicator is initialized, at the master
host with the command port plus one (derived from the link's bind address),
so any rank's abort or death stops every rank in about one second — through
an error record or the line close — instead of waiting on the socket
deadline; a clean deinit is an expected close and stops nobody. Requires the
derived port to be free on the rank-0 host. Unset: no lines, no environment
writes, no watch threads, no added time, and the peer-death semantics above
are exactly as before.

`TF_DSV4_GRAPHS=1` — set on both ranks. Captures the batched decode round
(only rounds the batched path can fuse and that take no samples — sampled and
DSpark speculative rounds always stay eager) keyed by row and context bucket,
cut at the NCCL collective: one capture per shape, replayed on hits, a failed
capture falls to eager permanently. The replay is collective-identical to the
eager round — same single all-gather, same sync — so ranks never diverge in
rank order and the flag needs no identity byte. Any address the round bakes
(KV rows, exchange registers, the fp8 staging buffer, the identity digest)
entering a new epoch drops every graph and recaptures, and any communicator
poison drops the cache too; per-round values re-run live, never baked.
Tuning knobs (defaults chosen for the two-Spark shape): `TF_DSV4_GRAPHS_MAX`
captured shapes, `TF_DSV4_GRAPH_BUCKET` / `TF_DSV4_GRAPH_BUCKET_GROW` shape
bucketing, `TF_DSV4_GRAPH_ROWS_MAX`, `TF_DSV4_GRAPH_FLOOR=hold|shed` memory
policy with `TF_DSV4_GRAPH_FLOOR_GIB` / `TF_DSV4_GRAPH_COST_MIB` bounds.
Unset: zero captures, zero fingerprint cost, the exact pre-flag program.
Capture and replay execution semantics are compile-verified only on this
build host — the qualification pass below owns the device half.

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

By default the port keeps the reference semantics: rows run serially within
shared lane rounds and NCCL gathers every round. The opt-ins above add batched
rounds and the P2P all-add; CUDA Graph replay and an RDMA transport remain off.

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
