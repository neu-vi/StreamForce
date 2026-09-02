# StreamForce — interactive demo

The browser demo for [StreamForce](../README.md). Generate from a single image and
change the force while it is generating.

**Rolling forcing + server-side generation pacing.** It adds one thing over plain rolling forcing: the generator **holds at a block boundary**
while it is more than `--pace_high_frames` video frames ahead of the frame the browser reports
showing, and resumes once the gap falls to `--pace_low_frames`.

Run: `./demo/run.sh` (port **5013**). `PACE_HIGH=0` disables
pacing, giving the original behaviour for an A/B.

## Why

With the optimized generator at ~30 fps and the client displaying frames as they arrive, the
generator finishes a 501-frame clip long before the viewer has watched it. A force change is
applied at the generation frontier, so it lands far ahead of what you are looking at -- the
`lands N frames ahead` figure in the log. Pacing bounds that gap.

## Where it happens

**No backend or pipeline change.** The backend's `_on_decoded_chunk` already ends with

```python
on_chunk(chunk_frames, block_index)
if should_stop(): raise GenerationStopped(...)
while should_pause(): time.sleep(0.02)      # <-- pacing lands here
```

so extending `_should_pause()` in `app.py` is enough. The hold happens **after a block has been
emitted, before the next window is denoised** -- block granularity, 3 latents = 12 video frames.
Manual pause still wins, and `should_stop` is checked inside the wait loop, so Stop can never
deadlock against a hold.

## Where a force change lands (two boundaries)

The write target used to come from `last_decoded_block` -- the *decode* frontier -- so the reported
"first affected video frame" was optimistic by however far denoising led decoding. In one measured
run it said **frame 129** when the earliest still-influenceable frame was **~273** and the effect
was actually visible around **300**.

The hook is called once per window in order, so the window index is just the call count, and the
pipeline's schedule follows from it (`start_block = w - steps + 1`, `end_block = min(nb-1, w)`):

* **partial** from block `w - steps + 1` -- the oldest block still in flight. Blocks part-way
  through denoising take the new force for their *remaining* steps only.
* **full** from block `w` -- entering the window now on its first step, so every one of its steps
  sees the new force.

Both are logged, and `full` is always the later one:

```
[force] APPLIED id=6 mode=wind (build 3 ms) | partial from block 23 (latent 69, video frame 273)
        | FULL from block 26 (latent 78, video frame 309) | viewer on frame 107 -> 166 ahead
        partial, 202 ahead full
```

Replaying the real run that reported `frame 129` (window index 26) through this gives partial 273
/ full 309 -- and 309 is where the change was actually seen.

## The viewer-buffer floor — why the watermark alone stutters

The watermark paces `denoised - shown`, which is the right frontier for *force reachability* (see
above). It is the wrong one for *smoothness*, and on its own it stutters badly.

`denoised` leads `emitted` by the rolling window's depth — 4 blocks in flight, so ~3 blocks or
**36 video frames**, and that lead is structural, not a decode backlog. So a paced gap of 45-60
sits on top of a browser holding only `emitted - shown` = **11-23 frames**, well under a second at
16 fps. Generation then arrives in 12-frame bursts separated by ~1 s holds, and the queue keeps
running dry.

Measured with a headless viewer that paints on a fixed 16 fps clock and never drops (so a tick
with nothing to paint is exactly the stutter you would see), 501-frame clip, force reversal
requested at t=12 s:

| | stall events | force lands ahead (partial / full) |
| --- | --- | --- |
| `PACE_HIGH=0` (no pacing) | 1 (startup only) | 213 / 249 frames — 13 s at 16 fps |
| watermark only, no floor | **15** | 16 / 52 |
| **watermark + floor 18** | **1** (startup only) | **29 / 65** |

So the floor removes the stutter outright and still keeps a force change **7x** closer than not
pacing at all.

`--pace_min_buffer` (default **18**, `PACE_MIN_BUFFER=`) is the number of frames the browser must
still have queued before generation is allowed to hold. Below it the hold is released immediately:
only new frames can refill the queue, so holding there is exactly backwards. It subsumes the older
`shown >= emitted` deadlock breaker, which only fired once the queue had already hit zero — too
late to prevent the stall it was meant to catch.

In practice the floor becomes the binding controller: in the run above, all 19 holds were released
by the floor and none by the low watermark. That is the intended operating point — hold generation
as long as possible, release the instant the viewer is at risk of running dry.

**Tuning.** The floor is the latency: a force change lands at the first block boundary past the
emitted frontier, so partial latency tracks the floor plus the delivery granularity.

The VAE decoder now hands over **each latent's 4 frames as they are ready** instead of a whole
12-frame block (`decode_latent_chunk(..., on_partial=...)`). Its loop was always per-latent —
Wan's causal VAE carries `feat_cache` across frames, so they can only be decoded in order — and
holding them to the end of a block was purely a delivery choice. Returning them as they land cut
the burst size to a third and took a block's decode (~222 ms) out of the refill latency in favour
of one latent's (~74 ms). Output is bit-identical (same sha256 over 165 frames) and end-to-end
throughput is unchanged; only the arrival pattern differs.

That is what makes a small floor safe. Measured, same 501-frame clip and force reversal at t=12 s:

| delivery | floor | stall events | force ahead (partial) |
| --- | --- | --- | --- |
| whole block (12 frames) | 18 | 1 | 29 |
| whole block (12 frames) | 12 | 2 | 24 |
| whole block (12 frames) | 0 | 15 | 16 |
| per latent (4 frames) | 18 | 2 | 16 |
| per latent (4 frames) | 12 | 2 | 10 |
| **per latent (4 frames)** | **8** | **1** | **6** |

Per-latent delivery roughly halves the latency at every floor, and lets the floor drop to
single digits without the stalls coming back. The default is **10** — 6-10 frames ahead, well
under a second — chosen with headroom over the ~6.6-frame refill latency measured locally
(200 ms to notice + ~107 ms for one latent of generation + 74 ms decode + wire).

**Raise it on a slow or remote link.** Those numbers are from a local websocket; RTT and wire
jitter add directly to the refill latency, and the floor has to cover them. If the stream
stutters, `PACE_MIN_BUFFER=18` restores the old headroom.

**`RF_BLOCK` does not help — measured, don't retry.** A smaller block quantizes the landing point
more finely and makes each generation burst smaller, so it looks like it should allow a lower
floor. It does not move the curve, only your position on it: RF_BLOCK=1 at floor 18 landed 26
frames ahead against 29 for RF_BLOCK=3 — three frames, in exchange for shrinking the attention
sink from 3 latents to 1, which is what anchors the rolling window. Per-latent *delivery* is the
change that actually moved it, and it costs nothing.

**The client says when it is low, rather than being asked.** `client_stats` is polled at 200 ms,
which became the largest single term in the refill latency once decode was per-latent (~107 ms for
one latent of generation, ~74 ms to decode it). The browser now raises `buffer_low` the moment its
undrawn depth crosses below the floor — edge-triggered, one signal per dip, re-armed at floor+2 —
and the server releases the hold on the spot. The floor is pushed to the client in
`default_config`. The polled tick still carries `buffered` as the fallback. In a default run 12 of
12 releases came from the signal and none from the fallback.

The server also paces on the **browser's own** count of undrawn frames rather than
`emitted - shown`, which counts frames still on the wire and so overstates what can be painted —
exactly the error that lets a hold starve the viewer.

**Hysteresis on the buffer, not just the gap.** Entering a hold at exactly the floor leaves one
frame before the release fires — 60 ms at 16 fps. `--pace_buffer_margin` (default: same as the
floor) requires `floor + margin` to *start* holding, while the release stays at the floor. Without
it a floor of 5 stalled 13 times while releasing correctly every time.

### Where it ended up

Defaults are `PACE_HIGH=45 PACE_LOW=40 PACE_MIN_BUFFER=6` (hold from a buffer of 12 down to 6).
Three consecutive runs at the defaults: **2 stall events (startup only), force change landing 6
frames — 0.4 s — ahead of the viewer.**

| | stall events | force ahead (partial) |
| --- | --- | --- |
| no pacing at all | 1 | 213 |
| pacing as first written | 15 | 16 |
| + viewer-buffer floor | 1 | 29 |
| + per-latent decode delivery | 1 | 10 |
| **+ event-driven signal and buffer hysteresis** | **1-2** | **6** |

**Read these as single samples.** Each row is one force change in one run, and the first
generation after a launch lands noticeably further ahead than steady state (23 frames against 6
in otherwise identical runs) while allocator and caches warm. The repeats above are the steady
state; treat differences of a few frames between rows as noise.

**Raise the floor on a slow or remote link.** All of this is a local websocket; RTT and wire
jitter add straight to the refill latency, and the floor has to cover them. `PACE_MIN_BUFFER=18`
restores the conservative headroom.

## Watermarks

| | default | why |
|---|---|---|
| `--pace_high_frames` | **48** | 4 blocks = one full rolling window = ~3 s of video at 16 fps. The generator cannot usefully be closer than a window; beyond this, force changes start landing far away. |
| `--pace_low_frames` | **24** | 2 blocks. The gap between the two is the hysteresis -- a single threshold would make the generator chatter on/off every block. |

`PACE_HIGH` / `PACE_LOW` in `run.sh`. `pace_low` is clamped below `pace_high` automatically.

Two guards, both failing **open** (generate rather than hang):

* the browser has not reported a displayed frame yet (`client_displayed < 0`)
* its last report is older than 3 s (`_PACING_STALE_S`) -- a closed tab or wedged socket must not
  freeze generation

## What it does NOT do

* never drops or skips a frame
* never rolls back or regenerates an already-generated frame
* does not change force-update semantics
* does not touch the readiness-driven client playback, frame ordering or epoch handling.
  `flask_static/app.js`, `flask_frontend/index.html`, `arrow_overlay.py` and the backend are
  **byte-identical** to the unpaced folder; the whole experiment is in `app.py`.

## Gallery presets

The left panel opens with a **Gallery** grid of preset cases from `assets/`. Clicking one loads,
in this order: the image (normalised to 832x480 exactly as an upload is), the mode, the four force
boxes (**Angle**, **Magnitude**, **Anchor X/Y**), the prompt, and a per-run clip cap. Then press
Start.

The force values go **straight into the browser's input boxes** and are applied with the same
`applyForceInputsToState()` the boxes use themselves, so a preset is indistinguishable from typing
those numbers by hand. Anchors are canvas pixels (832x480, y down); wind ignores the anchor
because its origin is pinned.

`assets/gallery.json` is the manifest and is re-read on every request -- edit values or add cases
without restarting. Images live beside it and are served from `/assets/<file>`.

**Clip length: presets are capped at 45 latents (177 video frames, ~11 s at 16 fps); uploading
your own image is unchanged and runs the full configured length** (`LATENTS`, default 126). This
exists because output degrades past roughly 2-3x the 81-frame training horizon -- see
`.claude/notes/long-horizon-force-artifact.md`.

A preset is **pristine** until you touch the force. Pristine means it runs as the curated case:
capped clip, and the force is **frozen for the duration of the run** (the four boxes and the mode
control are disabled, and dragging on the canvas is refused). Editing the force **before Start** --
a box, the mode, or a drag -- drops the preset: the highlight clears and the run reverts to the
configured full length, exactly like an upload. So the gallery gives a fixed, reproducible clip,
and one edit turns it into an ordinary interactive run.

`shown frame` uses the *current run's* length as its denominator, so it reads `137 / 177` for a
preset and `412 / 501` after you switch back to your own image. The browser keeps the server's
full length separately (`fullClipFrames`) and restores it on upload.

How the cap is plumbed, and why it is done this way:

* the browser sends `latents` in the `start` payload only for presets; an upload omits it
* `app.py` clamps it to `[1, DEMO_NUM_LATENT_FRAMES]` and rounds down to a whole
  `num_frame_per_block` (the pipeline asserts `num_frames % 3 == 0` on the image-conditioned path)
* it reaches the pipeline as a **call argument** (`generate_segment_streaming(...,
  num_latent_frames=...)` -> `inference_rolling_forcing_stream(num_frames=...)`), never through
  `DemoBackendConfig` -- `num_latent_frames` is part of the backend cache key in `_build_backend`,
  so routing it through config would reload the 61 GB checkpoint on every preset
* shortening needs no reallocation: `noise`/`output` are allocated per call, and the rolling
  KV cache is sized by the attention window (24 latents), not by clip length
* `max_output_frames` and the condition-signal length follow the per-run value, so the clip stops
  cleanly and the hint is exactly as long as the run

Socket.IO server and streaming generation, plus Qwen3-VL
prompt writing from `demo/captioner.py`.

```bash
conda activate streamforce
cd <your checkout of StreamForce>

CHECKPOINT=<PATH_TO_DISTILLED_STUDENT_CKPT> \
  ./demo/run.sh
ssh -L 5010:localhost:5010 <host>
```

`--checkpoint` is **required** — there is no default, so the demo cannot silently run the wrong
weights. `DEVICE=cuda:1` moves the model off GPU 0.

Default port: 5013 —
so all three can run at once if the GPUs allow.

## Writing the prompt

A VLM fills the prompt box for you, and **which** VLM call it makes depends on the force mode —
because the two modes need different text.

### Wind: automatic captioning

Wind acts on the whole scene, so a static description of it plus the force vector is everything
the generator needs. Choose an image and the caption fills the box by itself:

1. the browser normalises the image to 832x480 JPEG and sends it via `set_input` (v6's flow)
2. the server sees a new `reference_image_data`, spawns a thread and emits `caption_status`
   `{state:"working", kind:"caption"}` — the dot above the prompt box turns amber
3. Qwen3-VL captions it, the server emits `caption_ready {caption, took_s, kind:"caption"}`, the
   box fills and the dot turns green

The captioning instruction asks for the subject, appearance and setting, and explicitly **not**
motion — because motion is what the force signal is supposed to specify.

### Point: your hint, expanded

A point force acts on **one object**, so the prompt has to name it — and a caption of the image
cannot know which one you mean. So point mode does not auto-caption. Instead:

1. upload an image in point mode and the server emits `caption_status {state:"need_hint"}`; the
   bar reads *"name the object the force should move, then press Generate"*
2. type a short phrase in the **object** box above the prompt — *"the red vase"* is enough — and
   press **Generate** (or Enter). The browser emits `expand_prompt {object_hint}`; the image is
   already on the session, so it is not re-sent
3. `ImageCaptioner.expand_prompt` grounds your phrase in the image and writes the full prompt:
   the object named with its real colour, material and position, an unseen external force nudging
   it, and the rest of the scene explicitly left undisturbed. That is the shape of the point
   prompts in `assets/gallery.json`, i.e. what the generator was trained on
4. the server emits `caption_ready {caption, took_s, kind:"expand"}` and the box fills

The hint is a *starting point*, not a constraint: edit the generated prompt however you like, or
ignore the box entirely and write the whole prompt by hand.

### Details worth knowing

* **An automatic caption never clobbers what you typed.** It is only written if the box is empty
  or still holds the previous VLM output. Otherwise you get *"caption ready but the prompt was
  edited; left as-is"*.
* **Generate does overwrite the box**, edits included — you asked for it by pressing the button,
  and pressing it again is the way back, so nothing is lost for good.
* **A new upload drops a prompt the VLM wrote**, since it described the previous image, but keeps
  one you typed by hand.
* **Superseded requests are discarded.** A per-session sequence number means a slow result for an
  image or hint you already replaced is dropped rather than pasted over the newer one.
* **It never blocks the socket.** Both calls run on their own thread; the VLM takes a second or two.
* **An empty prompt does not block Start.** You get a warning in the bar above the box and the run
  goes ahead.
* **Disable it** with `CAPTION_MODEL="" ./demo/run.sh`, which turns both paths off and hides the
  object box, leaving the prompt fully manual. Or point `CAPTION_MODEL` at a local path or a
  different VLM — anything `transformers` can load through `AutoModelForVision2Seq` works.
* **The default is `Qwen/Qwen3-VL-8B-Instruct`** (~18 GB in bf16). It is pulled from the Hub on
  first run, so the first launch with the VLM on is slow.
* **Gallery presets ship a finished prompt** and leave the object box empty; there is nothing to
  expand.

`CUDA_VISIBLE_DEVICES` defaults to `0,1`: the model takes one card, and the captioner loads with
`device_map=auto` before it, so the second card keeps the VLM from crowding the generator. Pin it
explicitly with `CAPTION_DEVICE=cuda:1 DEVICE=cuda:0` if you prefer.

## Layout

`demo/`'s arrangement: header with model + transport pills, a left control panel (image, object
hint + prompt, mode, force fields, seed, start/stop) and a right panel with the canvas and a stats
row. The object hint row is shown in point mode only.

The canvas keeps **v6's interaction contract** exactly, because the server derives
force/angle/x_pos/y_pos from it:

* **Point** — click to place the application point, drag to set direction and length. The dashed
  ring is the region the conditioning signal actually covers (`gaussian_blob > 0.1`, radius
  `20·sqrt(-2·ln 0.1)` ≈ 43 px — note v6's point blob is radius 20 on the *actual* 480x832
  canvas, with no rescaling).
* **Wind** — the origin is pinned at `(width-100, 100)`; drag anywhere to aim it.
* **Releasing the drag** is what emits `change_force`. Editing the numeric angle/magnitude/anchor
  fields does the same.

`payload_text` is still v6's `{mode, canvas_width, canvas_height, anchor, vector_raw}`, and the
playback buffer with its lag control (`playbackFps 15`, `MAX_BUFFER_LAG_FRAMES 36` → skip to 8)
is copied verbatim.

## Saving the video

**Save mp4** writes the run to `demo/demo_outputs/<timestamp>/` and the browser
downloads it immediately; the link stays on the page to fetch again.

What gets saved is what the **model produced at full 480x832** -- `--wire_scale` and
`--jpeg_quality` shrink only the copy streamed to the browser, in the sender, long after the
frames have been kept. So you can stream at 624x360/q60 to keep the demo responsive and still
save a full-resolution clip.

* Frames are retained as they are decoded, so **Save works on a stopped or still-running run**
  too -- the result is marked `partial` in the UI and in `meta.json`.
* `--max_save_frames` (default 1200) caps how many are held. Each is a full-resolution uint8 RGB
  frame, ~1.2 MB at 480x832, so 1200 is about 1.4 GB.
* `--save_fps` (default 16) is the frame rate written into the file.
* Each folder also gets a `meta.json` with the prompt, seed, force payload, frame count and
  whether the run was partial.
* **`burn the force arrow into the saved video`** (checkbox) draws the arrow onto every frame,
  exactly as the browser draws it -- just the arrow;
  the blob-footprint ring and white anchor handle point mode used to draw are gone from both. The force can change mid-run, so each frame gets the force
  that was *actually acting on it* -- the arrow changes on the same frame the model first felt
  the new force, not when you dragged it. Written as `video_arrows.mp4` /`meta_arrows.json`
  rather than `video.mp4` / `meta.json`.
  Cost is ~0.8 s for a full 501-frame clip, on top of the same encode as before.
* The choice is made **at save time, not at generation time**. Frames and the force timeline are
  both kept after the run ends, so you can press Save twice -- once with the box ticked, once
  without -- and get both versions of the same run without regenerating. Each press writes its
  own timestamped folder.

### Which frame does a mid-run arrow change land on?

The first video frame that could possibly show it. A force update is picked up at the top of the
block loop in `pipeline/rolling_forcing_streaming_inference.py`, which overwrites the hint from
`current_start_frame` onward, so the block starting at that latent is the first one denoised
under the new force. With `num_frame_per_block = 3` and the causal VAE (latent 0 -> 1 frame, then
4 each), block *b* starts at video frame `12b - 3`: 0, 9, 21, 33, ...

So `video_first` is the causally correct instant -- not when you dragged. The *visible* motion
change trails it by a few frames, because a physical response ramps up rather than switching
instantly; the arrow marks when the model was first told, which is the only well-defined point.
* `meta.json` records the whole `force_timeline` -- every force the run used and the first video
  frame it acted on -- whether or not arrows were burned in.

Files are served by `GET /download/<folder>/<file>` as an attachment, restricted to paths inside
the output directory. The server path shown after a save wraps (`.path` uses
`overflow-wrap: anywhere`): no browser inserts a line break at `/`, so a long path used to spill
out past the panel border instead of wrapping inside it.

## Point-force conditioning

The blob mask is **frozen at frame 0**, matching the trainer
(`trainer/wan_controlnet_distillation.py`), every `inference_*.py` and the ODE-pair generators,
all of which do:

```python
gaussian_blob = gaussian_blob[:, 0:1]      # "do not let the blob move"
hint = hint * (gaussian_blob > 1e-1)
```

The demos were missing that line and masked each frame with the *moving* blob, so a point force's
conditioning drifted away from what the checkpoint was trained on -- differing on 99 of 100 frames
by up to 2.9% of the frame area.

Because the mask is frozen and channels 1-3 are constant in time, the prepared hint is **one
spatial pattern**, for point exactly as for wind. So a point change builds **2 frames** instead of
the whole remainder of the clip (2 is the minimum: the loader computes `t = frame/(num_frames-1)`),
and one frame is broadcast across the remaining latents. This is exact, not an approximation --
frame 0's blob sits at `(x_pos, y_pos)` regardless of `num_frames`, and the resulting hint and
`masked_latent` are bit-identical to the old full-length build.

It also removes the stutter on every point change: that rebuild was ~490 frames, about **1 s of
CPU and ~2.7 GB of GPU allocation on the generation thread**, at a block boundary, before the
block could be denoised. Wind was always 1 frame and ~1 ms, which is why only point stuttered.

The initial build at Start is still full length. It is a one-time cost before the first block, and
collapses to the same single pattern -- worth reducing later, but it is not what caused the hitch.

## Force delivery panel

| tile | source | meaning |
|---|---|---|
| `you set` | the canvas arrow | what the UI currently holds |
| `applied at` | `force_change_debug.video_start_frame` | first video frame produced under the new force |
| `shown frame uses` | client timeline vs the displayed frame index | what the frame on screen is conditioned on |
| `changes` | count of applied changes | how many landed |

`shown frame uses` is derived entirely on the client: it keeps `{video_first, label}` entries and
picks the last one whose `video_first` is `<=` the displayed frame index -- the same rule
`burn_arrows` uses, so this readout always agrees with the arrow burned into the saved video. It
is seeded at Start with the force the run begins under, so it is populated from the first frame.

Watch it against **`shown frame`** in the Stream panel: when that index reaches `applied at`,
`shown frame uses` flips. Both are the generator's own frame index, taken from the frame header,
so they are directly comparable -- as is `first affected video frame N` in the server log.

Two things this replaced:

* `model is using` was dead markup. It came over from `demo/`, whose `streaming_engine.py`
  published `model_control`/`model_control_block`; the v6 server has no equivalent, so nothing
  ever wrote to the element and it showed `-` forever.
* `applied at` was scraped out of the prose status line with a regex on `latent=.. video=..`.
  Rewording that log message silently blanked the readout, so the server now sends
  `video_start_frame` as a structured field.

`shown / buffered` in the Stream panel became **`shown frame`** (the frame index, with the clip
length as denominator). The old value was `cursor / frameBuffer.length` -- a buffer position, in
different units from `applied at`, and it resets to 0 on an epoch purge. `buffer lag` already
carries the backlog those two numbers implied.

## Debug readouts

Three panels in the page and a matching set of log prefixes on the server — see the table in
`README` below, or just start it and watch. Server prefixes: `[gen]`, `[ttff]`, `[stream]`,
`[force]`, `[client]`, `[captioner]`, `[browser:*]`.

| panel | fields |
| --- | --- |
| **Force delivery** | you set · model is using · applied at (latent + first affected frame) · changes |
| **Stream** | state · shown/buffered · block · recv fps · draw fps · buffer lag · KB/frame · gen fps · sent fps · MB/s out · server queue · behind · queue wait · ttff gen/sent |
| **Model** | device · clip · block size · vram · transport · weights (`generator_ema` vs `generator`) |

## Caveats

Same as the rebuild: `socket.io.min.js` comes from a CDN (loaded by your browser, so it needs
internet on your laptop — the only local copies on this machine are 2.1.2 and too old to use);
and attention only reads back 84 latents, so the tail of a 501-frame clip drifts from the input
image regardless of the force.

Memory is now a single stack rather than two: base **Wan2.2-TI2V-5B** DiT + 15-layer ControlNet
(~30 GB from the checkpoint's `generator_ema`), one umt5-xxl text encoder (~11 GB), one VAE, and
~34 GB of KV cache at 126 latents — roughly **75 GB on one GPU**, versus the rebuild's two such
stacks. Plus the captioner wherever `device_map=auto` puts it.

