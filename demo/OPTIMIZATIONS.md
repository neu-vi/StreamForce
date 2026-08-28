# Inference optimizations — what they need, what they cost

Every optimization below is **ON by default**. This file records what each one requires, what
it is worth, and how to turn it off. Numbers are steady-state medians on one H200, rolling
forcing at 480x832, `RF_BLOCK=3` (one block = 3 latent frames = 12 video frames), measured
headless (no Flask, no captioner) so the web stack does not blur the comparison.

## Headline

| | block gap | gen fps |
|---|---|---|
| before any of this | 1418 ms | 8.5 |
| **single GPU, all optimizations** | **519 ms** | **23.1** |
| **two GPUs, all optimizations** | **321 ms** | **37.3** |

The demo itself lands lower than the headless number (~28 fps on two GPUs) because JPEG
encoding and base64 framing for the browser cost CPU on top. That is a transport problem, not
a generation one — see *Not done yet*.

## Requirements

| | needs |
|---|---|
| extra GPU | **optional** — a 2nd GPU is used automatically if visible; 1 GPU works and is still ~2.7x faster than before |
| packages | nothing to install beyond the existing env. `flash_attn` (FA2) and `flash_attn_interface` (FA3) are both already present; cuDNN ships with torch |
| GPU arch | channels_last_3d + FA3 both want Hopper-class (H100/H200). On older cards FA3 is absent and the code falls back to FA2 on its own |
| VRAM | generator ~28 GB, VAE ~2-13 GB, captioner ~60 GB. On one GPU without the captioner: ~40 GB |

## The optimizations

### 1. Async VAE decode — `WAN_ASYNC_VAE=1` (default)

Decode runs on a worker thread with its own CUDA stream instead of inline in the denoising
loop, so a block's decode overlaps the next window's denoising instead of adding to it.

- **Needs**: nothing.
- **Worth**: 1418 -> 976 ms on one GPU. On one GPU it recovers launch bubbles only, since decode
  and denoising still contend for the same SMs (concurrently: generator 191 -> 346 ms/call,
  decode 806 -> 1088 ms). With the VAE on its own GPU the overlap is nearly perfect.
- **Numerics**: bit-identical (verified, same sha256 over 165 frames).
- **Off**: `WAN_ASYNC_VAE=0`.

The VAE decoder carries a causal cache across chunks, so chunks must be decoded strictly in
order — that is why it is a single worker thread and not a pool. Do not widen it.

It also hands over **each latent's 4 frames as they are decoded** rather than a whole 12-frame
block (`decode_latent_chunk(..., on_partial=...)`). The loop was always per-latent — the causal
cache forces it — so this is a delivery change only: bit-identical output, unchanged throughput,
but a third the burst size, and one latent's decode (~74 ms) instead of a block's (~222 ms)
before frames start arriving.
  It is what lets this demo hold a single-digit frame buffer; see the pacing section of
  [README.md](README.md).

### 2. VAE on its own GPU — `VAE_DEVICE=auto` (default)

`auto` puts the VAE on `cuda:1` when two or more GPUs are visible, otherwise leaves it beside
the generator. Decode is the larger half of a block, so on one GPU it competes with the
generator for SMs; on two it does not.

- **Needs**: a 2nd visible GPU. Falls back silently and correctly to single-GPU.
- **Worth**: 519 -> 321 ms (+62%). Block gap also becomes far steadier (854-874 ms vs
  876-1563 ms at an earlier stage of this work), and ttff improves.
- **Cost**: ~0.45 MB of latents copied per block. Negligible.
- **Numerics**: bit-identical.
- **Off / pin**: `VAE_DEVICE=same` (single-GPU mode), or `VAE_DEVICE=cuda:3` to choose.

### 3. VAE `channels_last_3d` — `VAE_CHANNELS_LAST=1` (default)

The single biggest win. In **bf16 + contiguous**, PyTorch cannot hand a 3D convolution to cuDNN
and falls back to `aten::slow_conv_dilated3d`, an im2col path that materializes the whole
convolution window:

```
bf16 contiguous       -> aten::slow_conv_dilated3d   0.39 ms
bf16 channels_last_3d -> aten::cudnn_convolution     0.11 ms
fp32 contiguous       -> aten::cudnn_convolution     0.27 ms   (fp32 never falls back)
```

Converting the 62 Conv3d weights to `channels_last_3d` is enough — PyTorch's conv dispatch
takes the cuDNN path when *either* input or weight is channels_last, so no activation plumbing
is needed.

- **Needs**: nothing (cuDNN ships with torch).
- **Worth**: VAE decode **809 -> 225 ms (3.6x)** for a 3-latent chunk. Kernel launches for one
  decode drop **238,442 -> 3,245** and its CPU time 1297 -> 212 ms — the 78k `copy_` / 157k
  `select` / 78k `fill_` seen in profiles were `slow_conv_dilated3d`'s internal im2col loop.
- **Numerics**: **not bit-identical** — a different conv algorithm. Measured on real VAE weights
  with identical latents: decode output differs by at most **0.039 on a [-1,1] range (~5/255
  levels)**, mean 0.15/255. cuDNN accumulates in fp32, so it is likely the more accurate of the
  two. Visually indistinguishable.
- **Off**: `VAE_CHANNELS_LAST=0`.

### 4. Generator `channels_last_3d` — `GEN_CHANNELS_LAST=1` (default)

The same fallback hits the generator's two patch embeddings:

```
model.patch_embedding          k=(1,2,2) s=(1,2,2) in=48 out=3072
control_model.patch_embedding  k=(1,2,2) s=(1,2,2) in=48 out=3072
```

(`input_hint_block` is `conv_nd(2, ...)` — 1x1 **Conv2d** — and is unaffected.)

- **Needs**: nothing.
- **Worth**: 432 -> 367 ms (+18%). One generator forward drops from **93,746 to 9,558** kernel
  launches and 572 -> 418 ms of GPU time; two convs buy that much because the im2col launch
  overhead goes with them.
- **Numerics**: **not bit-identical**, same reason as #3. These sit at the very front of the
  network so the difference propagates through all 45 layers; not separately quantified.
- **Off**: `GEN_CHANNELS_LAST=0`.

### 5. FlashAttention 3 — automatic

`wan/modules/attention.py` prefers FA3 when `flash_attn_interface` is importable. It used to
crash with `unflatten: ... dim 0 (24)` because upstream does `...flash_attn_varlen_func(...)[0]`,
assuming the old `(out, softmax_lse)` tuple return; current builds return the tensor directly,
so `[0]` sliced off row 0. Now both return shapes are handled.

- **Needs**: `flash_attn_interface` (already installed) and Hopper-class hardware. Without it
  the code takes FA2 by itself.
- **Worth**: 347 -> 321 ms (+7.9%). Attention is 23% of the generator's GPU time (47.3 ms), and
  FA3 is 1.87x FA2 on the real shape (q=4680, k=8190, 24 heads, dim 128): 1.339 -> 0.717 ms.
- **Numerics**: **not bit-identical** — different kernel. Single-op difference vs FA2 is
  `4.88e-4`, i.e. bf16 rounding.
- **Off**: no env knob. Pass `fa_version=2`, or set `FLASH_ATTN_3_AVAILABLE = False` in
  `wan/modules/attention.py`.

### 6. Host-side KV-cache indices — always on

`CausalWanSelfAttention` used to read its cache bookkeeping back out of GPU tensors:
`kv_cache["local_end_index"].item()`, 8 times per attention module. Those values are Python
ints the host itself had just written with `.fill_()`, so it was a pure round trip — **227
`.item()` calls and 68 ms of `cudaStreamSynchronize` per generator forward, for 0.44 ms of
actual transfer**. They are now mirrored as ints (the tensors are still updated, so any other
reader is unaffected).

- **Worth in eager: nothing** (367 -> 365 ms, inside noise). The 68 ms was the CPU *waiting on a
  GPU-bound pipeline* — a symptom, not a cause. Kept because it is free, and because it is what
  turned `torch.compile` from −12% into +2.8% (see *Tried and rejected* below) — so it is the
  prerequisite if anyone revisits that.
- **Numerics**: bit-identical (integer bookkeeping).

## Single-GPU mode

Nothing needs to be passed — `VAE_DEVICE=auto` detects one GPU and keeps the VAE beside the
generator. Everything else applies unchanged, so single-GPU is still ~2.7x the original:

| single GPU | block gap | gen fps |
|---|---|---|
| async decode off | 601 ms | 20.0 |
| **all optimizations (default)** | **519 ms** | **23.1** |

The gap to the two-GPU number (321 ms) is SM contention between decode and denoising, which
only a second GPU removes. If you have a spare GPU, use it; if not, this is the ceiling.

## Verifying numerics

Three optimizations change results because they select different kernels: VAE channels_last,
generator channels_last, and FA3. Each was checked at the level of the op it changes (see
above), but **the full pipeline was not validated against the real checkpoint** — the 61 GB
weight file could not be loaded alongside a running instance. If output quality matters, look
at a clip with the defaults, then compare against:

```bash
VAE_CHANNELS_LAST=0 GEN_CHANNELS_LAST=0 ./this demo/run.sh
```

Whole-pipeline diffs measured with *random* weights are meaningless here: the denoising loop is
chaotic without trained weights, so a 5/255 decode difference amplified to 133/255 over 45
layers x 4 steps. Trained weights contract toward the data manifold; expect far less.

## Tried and rejected: `torch.compile`

**This one is not pending — it was implemented, measured, and deliberately left out.** Recording
it so nobody spends the day re-deriving the same answer.

The pre-flight said it should work: `compile_headroom_analyzer.py` on an eager trace put the
**fusible ceiling at 23.5%** of the generator's GPU time (pointwise 21% + norm/reduction 3%;
gemm is 37% and attention 23%, neither of which fusion can touch) and returned
`COMPILE LIKELY WORTH IT`.

It did not. Integrated the standard way — `torch.compile(block, dynamic=True)` on all 45
`CausalWanAttentionBlock`, with `torch._dynamo.disable` on the untraceable flash-attention call
so the norm/proj/FFN around it still fuse — a strict same-conditions A/B gave:

| | block gap | gen fps | ttff |
| --- | --- | --- | --- |
| eager | 365 ms | 32.9 | 2.3 s |
| compiled, first attempt | 411 ms | 29.2 | 48.8 s (warm inductor cache) |
| compiled, after the KV-index fix (#6) | 355 ms | 33.8 | 102.8 s |

So: **a 12% regression at first, and +2.8% at best** — for ~100 s of startup compile every
launch. `TORCH_LOGS=graph_breaks,recompiles` explains it: **49 graph breaks and 42 steady-state
recompiles** remain, where a healthy compiled loop has only the breaks you placed on purpose and
zero recompiles. Removing the KV-cache `.item()` reads (#6) is what turned −12% into +2.8%, which
confirms the mechanism.

The remaining blocker is `frame_seqlen = math.prod(grid_sizes[0][1:]).item()` in
`CausalWanSelfAttention.forward`. `grid_sizes` is a **CPU** tensor, so this costs no GPU sync —
it purely stops Dynamo, and the unbacked symint it produces spawns 7 data-dependent
`u0*u1*u2 < 0` guard failures. Fixing it means threading `frame_seqlen` through as a Python int
from the four `grid_sizes` construction sites (`causal_model.py:899/1050/1251/1410`). That is a
real refactor of shared model code, and even if it landed perfectly the payoff is bounded by the
23.5% ceiling. Judged not worth it; revisit only if the transport work below lands and generation
becomes the binding constraint again.

## Not done yet

Ordered by remaining value.

1. **Transport.** With generation at ~37 fps headless, JPEG (74 KB/frame at q85) + base64
   (+33%) on the CPU is what holds the demo to ~28 fps. **Generation is now faster than the
   wire, so this is the only item left that changes what the browser actually shows.**
   A bounded send queue that drops the oldest frame
   (`--frame_queue_max 12`, trading frame rate for latency); it has not been ported here. Quick
   partial mitigation: `JPEG_Q=60 WIRE_SCALE=0.75`.
2. **CUDA Graph.** Not attempted — a judgement call, not a measurement. Whole-stack capture is
   where the win is, and it wants shape stability. Rolling forcing's window is variable-length
   at the start and end of a clip (`causal_model` raises `"should be full window, first window,
   or last window"`), and the 42 recompiles above are independent evidence that shapes move.
   Precondition not met.
3. **The two `torch.cat` per attention module.** The rolling-forcing path materializes K and V
   as `cat([anchor, window, new])` — ~50 MB each, ~4.5 GB per forward, `aten::cat` 16.3 ms
   (7.8% of generator GPU time). Could write into a preallocated buffer, or split the attention
   and combine via log-sum-exp. Not attempted.

## Scope

All of this is wired into **`this demo`** only. `this demo`
The async decoder lives in
`pipeline/streaming_vae_decoder.py` and could be wired into self-forcing the same
way, but has not been.
