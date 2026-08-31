/* Force-control demo front end over the Socket.IO server in app.py.
 *
 * The protocol and the semantics are v6's, unchanged, and deliberately so:
 *
 *   up    set_input {prompt, mode, payload_text, reference_image_data?}   (also polled every 500 ms)
 *         start     {prompt, mode, payload_text, reference_image_data, seed}
 *         stop      {}
 *         change_force {mode, payload_text, displayed_frame_index, displayed_block_index}
 *   down  status, default_config, model_loaded, frame_ready, restart_cutover,
 *         caption_ready / caption_status (added here)
 *
 * `payload_text` keeps v6's shape -- canvas size, anchor, raw vector -- because the server's
 * `parse_ui_force_payload` derives force/angle/x_pos/y_pos from exactly that. The playback
 * buffer, its lag control and the client-side image normalisation are v6's too, copied as-is.
 * What changed is only the presentation.
 */

const $ = (id) => document.getElementById(id);

const canvas = $("forceCanvas");
const ctx = canvas.getContext("2d");
const modeSel = $("mode");
const promptEl = $("prompt");
// No <img> and no data URL per frame: frames arrive as raw JPEG bytes and are decoded with
// createImageBitmap, which runs off the main thread. `baseImageDataUrl` is now produced only
// when something actually needs a data URL (Start, set_reference_frame) -- see frameDataUrl().
let liveBitmap = null;
// Whether `liveBitmap` holds a generated frame or the uploaded still. v6 kept updating
// `baseImageDataUrl` on every displayed frame, so pressing Start again continued from the frame
// you were looking at; with frames now arriving as bitmaps that flag is what preserves it. Start
// also reads it to tell a resume from a first run -- see the `continuing` rule there.
let liveBitmapIsGenerated = false;
const seedInputEl = $("seedInput");
const modeButtons = Array.from(document.querySelectorAll("#modeSeg button"));
const angleInputEl = $("angleInput");
const magInputEl = $("magInput");
const anchorXInputEl = $("anchorXInput");
const anchorYInputEl = $("anchorYInput");
const anchorXWrapEl = $("anchorXWrap");
const anchorYWrapEl = $("anchorYWrap");

// ---- v6 constants: must match interactive_demo_socketio_app.py / the force adapter ----------
const MAX_LEN = 80;                     // MAX_POINT_FORCE_LEN == MAX_WIND_FORCE_LEN == 80
const WIND_ANCHOR_RIGHT_OFFSET = 100;   // where the wind origin is pinned
const WIND_ANCHOR_TOP_OFFSET = 100;
// ---- v6 playback constants ------------------------------------------------------------------
// Fixed playback clock: paint at most PLAYBACK_FPS frames per second. Readiness still WAKES the
// painter (a decode completing arms the next paint), but the clock sets the ceiling, so the
// effective rate is min(PLAYBACK_FPS, generation, transport, decode, display refresh).
// A paint is never issued more often than PLAYBACK_INTERVAL_MS after the previous one; frames are
// still never dropped, so a faster generator just builds backlog instead of speeding up display.
const PLAYBACK_FPS = 16;
const PLAYBACK_INTERVAL_MS = 1000 / PLAYBACK_FPS;
//
// Decoding IS paced: at most MAX_DECODE_IN_FLIGHT at once, always starting the lowest-index
// undecoded frame. Unbounded decode meant a post-stall burst launched hundreds of concurrent
// createImageBitmap calls (~1.6 MB of bitmap each) and could starve the one frame that actually
// blocks display while finishing later ones.
const MAX_DECODE_IN_FLIGHT = 6;
// v6 also had MAX_BUFFER_LAG_FRAMES / TARGET_BUFFER_LAG_FRAMES: once the buffer held more than
// 36 undrawn frames it jumped the cursor forward and threw away everything it skipped -- up to
// 28 frames at once, which is the "jumps past a few frames" you see while the saved mp4 (written
// server-side) stays smooth. Frames are never dropped here, so that is gone. The cost is
// latency: after a network stall the backlog is played out in full, so the picture runs behind
// until it drains. The server logs the backlog as `inflight` on every [stream] line.

const socket = io({ transports: ["websocket", "polling"], upgrade: true });

let baseImageDataUrl = "";
let dragging = false;
let imageProcessing = false;
let hasUploadedImage = false;
let force = { anchor: { x: canvas.width / 2, y: canvas.height / 2 }, vec: { dx: 0, dy: 0 } };
// Nothing is drawn for a point force until you actually place one. The blob ring and the white
// handle used to be painted at the canvas centre on load, which showed a force that was not
// there (magnitude defaults to 0).
let pointPlaced = false;

let frameBuffer = [];
// Pacing floor pushed by the server. When the undrawn depth crosses below it we say so at once
// rather than waiting for the 200 ms client_stats tick -- that lag was the largest single term
// in how long the server takes to notice it must resume generating. Edge-triggered: one signal
// per dip, re-armed when the buffer recovers.
let paceMinBuffer = 0;
let bufferLowSignalled = false;
let cursor = 0;
let paintPending = false;
let lastPaintMs = 0;         // performance.now() of the last painted frame (playback clock)
let decodeInFlight = 0;
const latencyMs = [];          // receive -> visible, per painted frame
let lastDisplayedFrameIndex = -1;
let lastDisplayedBlockIndex = -1;
let currentEpoch = -1;
let acceptIncomingFrames = true;
let generating = false;
let clipFrames = 0;
// The server's configured full length, kept separately so a gallery preset's shorter cap can be
// undone. Without this, `shown frame` kept a preset's 177 denominator on a later full-length
// upload run and read e.g. "412 / 177".
let fullClipFrames = 0;

// What the frame on screen is conditioned on. Entries are {video_first, label}, sorted, and the
// answer is the last one whose video_first <= the displayed frame index -- the same rule
// burn_arrows uses server-side, so this readout and the saved video's arrow always agree.
let forceTimeline = [];
let pendingForceLabels = [];            // emitted, awaiting the server's seq
let forceLabelBySeq = {};               // seq -> label, until the change is applied
let lastShownUses = null;

// Set by a gallery preset, sent with `start` so only preset runs are capped. null => the
// normal upload flow, which keeps the server's configured maximum.
let galleryLatents = null;
let galleryActiveId = null;
// A preset that has been loaded and NOT touched. Pristine => it runs as the curated case: capped
// clip, and the force is fixed for the duration of the run. Editing the force before Start makes
// it an ordinary custom run at the configured full length.
let galleryPristine = false;
let applyingPreset = false;       // so applyPreset's own setMode/apply calls don't self-invalidate

let lastAutoCaption = "";               // so a later caption never clobbers text you typed
const recvTimes = [], drawTimes = [];

// -------------------------------------------------------------------------- small helpers

function message(text, kind = "") {
  const el = $("message");
  el.textContent = text;
  el.className = `message ${kind}`;
}
const setStatus = message;

function setText(id, v) { const el = $(id); if (el) el.textContent = v; }

function noteTime(arr) {
  arr.push(performance.now());
  while (arr.length > 120) arr.shift();
}
function fpsOf(arr) {
  // Pruned by time as well as count, so it decays to 0 when frames stop rather than freezing.
  const now = performance.now();
  while (arr.length && now - arr[0] > 4000) arr.shift();
  if (arr.length < 2) return 0;
  const span = (arr[arr.length - 1] - arr[0]) / 1000;
  return span > 0.1 ? (arr.length - 1) / span : 0;
}

function setStartEnabled() {
  $("startBtn").disabled = imageProcessing || !hasUploadedImage;
}

// -------------------------------------------------------------------------- force state (v6)

function vecToAngleDeg(dx, dy) {
  const len = Math.hypot(dx, dy);
  if (len < 1e-6) return 0;
  return ((Math.atan2(-dy, dx) * 180) / Math.PI + 360) % 360;
}

function angleDegToUnit(angleDeg) {
  const rad = (angleDeg * Math.PI) / 180.0;
  return { ux: Math.cos(rad), uy: -Math.sin(rad) };
}

function clampVec(dx, dy) {
  const len = Math.sqrt(dx * dx + dy * dy);
  if (len < 1e-6) return { dx: 0, dy: 0 };
  const s = Math.min(1, MAX_LEN / len);
  return { dx: dx * s, dy: dy * s };
}

function clampAnchorPoint() {
  force.anchor.x = Math.max(0, Math.min(canvas.width, force.anchor.x));
  force.anchor.y = Math.max(0, Math.min(canvas.height, force.anchor.y));
}

function windAnchor() {
  return { x: canvas.width - WIND_ANCHOR_RIGHT_OFFSET, y: WIND_ANCHOR_TOP_OFFSET };
}

function syncForceInputsFromState() {
  const v = clampVec(force.vec.dx, force.vec.dy);
  force.vec = v;
  angleInputEl.value = vecToAngleDeg(v.dx, v.dy).toFixed(1);
  magInputEl.value = Math.max(0, Math.min(1, Math.hypot(v.dx, v.dy) / MAX_LEN)).toFixed(3);
  anchorXInputEl.value = force.anchor.x.toFixed(1);
  anchorYInputEl.value = force.anchor.y.toFixed(1);
  const showAnchor = modeSel.value === "point";
  anchorXWrapEl.style.display = showAnchor ? "" : "none";
  anchorYWrapEl.style.display = showAnchor ? "" : "none";
}

function applyForceInputsToState() {
  if (forceLocked()) { syncForceInputsFromState(); return; }
  invalidatePreset();
  const angle = parseFloat(angleInputEl.value || "0");
  const magRaw = parseFloat(magInputEl.value || "0");
  const magNorm = Number.isFinite(magRaw) ? Math.max(0, Math.min(1, magRaw)) : 0;
  const mag = magNorm * MAX_LEN;
  let ax = parseFloat(anchorXInputEl.value || String(force.anchor.x));
  let ay = parseFloat(anchorYInputEl.value || String(force.anchor.y));
  if (!Number.isFinite(ax)) ax = force.anchor.x;
  if (!Number.isFinite(ay)) ay = force.anchor.y;
  if (modeSel.value === "wind") {
    force.anchor = windAnchor();
  } else {
    force.anchor = { x: ax, y: ay };
    clampAnchorPoint();
  }
  const unit = angleDegToUnit(Number.isFinite(angle) ? angle : 0);
  force.vec = { dx: unit.ux * mag, dy: unit.uy * mag };
  if (modeSel.value === "point") pointPlaced = true;   // typing coordinates counts as placing
  syncForceInputsFromState();
  draw();
  applyForceChange();
}

function resetForceStateForMode(mode) {
  force.anchor = mode === "wind" ? windAnchor() : { x: canvas.width / 2, y: canvas.height / 2 };
  force.vec = { dx: 0, dy: 0 };
  pointPlaced = false;         // switching modes clears the force, so clear the drawing too
  syncForceInputsFromState();
  draw();
}

function setMode(mode) {
  if (forceLocked()) return;
  invalidatePreset();
  modeSel.value = mode;
  modeButtons.forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
  resetForceStateForMode(mode);
  applyForceChange();
}

/** v6's payload shape, verbatim -- the server derives force/angle/x_pos/y_pos from it. */
function payloadText() {
  return JSON.stringify({
    mode: modeSel.value,
    canvas_width: canvas.width,
    canvas_height: canvas.height,
    anchor: { x: force.anchor.x, y: force.anchor.y },
    vector_raw: { dx: force.vec.dx, dy: force.vec.dy },
  });
}

// -------------------------------------------------------------------------- drawing

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (liveBitmap) {
    ctx.drawImage(liveBitmap, 0, 0, canvas.width, canvas.height);
  } else {
    ctx.fillStyle = "#0d0f12";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }

  const ax = force.anchor.x;
  const ay = force.anchor.y;
  const v = clampVec(force.vec.dx, force.vec.dy);
  const ex = ax + v.dx;
  const ey = ay + v.dy;
  const len = Math.hypot(v.dx, v.dy);

  // Point mode used to also draw the blob-footprint ring and a white handle at the anchor.
  // Both are gone: they were decoration (nothing here drags the handle), and with magnitude
  // defaulting to 0 they painted a force at the canvas centre that did not exist.
  if (len < 1e-4 || (modeSel.value === "point" && !pointPlaced)) return;
  const ux = v.dx / len;
  const uy = v.dy / len;
  const nx = -uy;
  const ny = ux;

  const head = (tx, ty, scale = 1.0) => {
    const h = 12 * scale;
    const ang = Math.atan2(v.dy, v.dx);
    ctx.beginPath();
    ctx.moveTo(tx, ty);
    ctx.lineTo(tx - h * Math.cos(ang - Math.PI / 7), ty - h * Math.sin(ang - Math.PI / 7));
    ctx.lineTo(tx - h * Math.cos(ang + Math.PI / 7), ty - h * Math.sin(ang + Math.PI / 7));
    ctx.closePath();
    ctx.fillStyle = "#ffe14d";
    ctx.fill();
  };

  ctx.strokeStyle = "#ffe14d";
  ctx.lineWidth = 2.8;
  ctx.lineCap = "round";

  if (modeSel.value === "wind") {
    // Three offset strands, matching v6's rendering of a pinned wind vector.
    for (const off of [-14, 0, 14]) {
      const sx = ax + nx * off, sy = ay + ny * off;
      const tx = ex + nx * off, ty = ey + ny * off;
      const c1x = sx + ux * (len * 0.34) + nx * (off * -0.15 + 3);
      const c1y = sy + uy * (len * 0.34) + ny * (off * -0.15 + 3);
      const c2x = sx + ux * (len * 0.70) + nx * (off * 0.10 - 3);
      const c2y = sy + uy * (len * 0.70) + ny * (off * 0.10 - 3);
      ctx.beginPath();
      ctx.moveTo(sx, sy);
      ctx.bezierCurveTo(c1x, c1y, c2x, c2y, tx, ty);
      ctx.stroke();
      head(tx, ty, 0.62);
    }
  } else {
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(ex, ey);
    ctx.stroke();
    head(ex, ey, 1.0);
  }

  // Grab handle at the tip, so it reads as draggable when idle.
  if (!generating) {
    ctx.fillStyle = "#ffe14d";
    ctx.strokeStyle = "rgba(0,0,0,0.55)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(ex, ey, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }
}

// -------------------------------------------------------------------------- pointer (v6 logic)

function pos(evt) {
  const r = canvas.getBoundingClientRect();
  return {
    x: ((evt.clientX - r.left) / r.width) * canvas.width,
    y: ((evt.clientY - r.top) / r.height) * canvas.height,
  };
}

/** What the UI currently holds, for the "you set" tile. */
function describeForce() {
  const angle = Math.round(parseFloat(angleInputEl.value || "0"));
  const mag = parseFloat(magInputEl.value || "0").toFixed(2);
  const head = modeSel.value === "wind"
    ? "wind"
    : `point (${Math.round(force.anchor.x)},${Math.round(force.anchor.y)})`;
  return `${head} ${angle}° ${mag}`;
}

function applyForceChange() {
  setText("s-ui-force", describeForce());
  // Snapshot what was set NOW: the server assigns the seq, and by the time it comes back the
  // arrow may have moved on. FIFO order holds -- one socket, and the server answers each
  // change_force synchronously.
  pendingForceLabels.push(describeForce());
  socket.emit("change_force", {
    mode: modeSel.value,
    payload_text: payloadText(),
    displayed_frame_index: lastDisplayedFrameIndex,
    displayed_block_index: lastDisplayedBlockIndex,
  });
}

canvas.addEventListener("pointerdown", (e) => {
  if (!hasUploadedImage) return;
  if (forceLocked()) return;
  invalidatePreset();
  const p = pos(e);
  dragging = true;
  if (modeSel.value === "point") {
    force.anchor = { x: p.x, y: p.y };
    force.vec = { dx: 0, dy: 0 };
    pointPlaced = true;
  } else {
    force.anchor = windAnchor();
    force.vec = { dx: p.x - force.anchor.x, dy: p.y - force.anchor.y };
  }
  canvas.setPointerCapture(e.pointerId);
  syncForceInputsFromState();
  draw();
});

canvas.addEventListener("pointermove", (e) => {
  if (!dragging) return;
  const p = pos(e);
  force.vec = { dx: p.x - force.anchor.x, dy: p.y - force.anchor.y };
  syncForceInputsFromState();
  draw();
});

canvas.addEventListener("pointerup", () => {
  if (!dragging) return;
  dragging = false;
  applyForceChange();          // v6: the release is what commits the change
});
window.addEventListener("pointerup", () => { dragging = false; });

modeButtons.forEach((b) => { b.onclick = () => setMode(b.dataset.mode); });
[angleInputEl, magInputEl, anchorXInputEl, anchorYInputEl].forEach((el) =>
  el.addEventListener("change", applyForceInputsToState));

// -------------------------------------------------------------------------- playback (v6)

/** Retire a buffer slot: free its pixels, and tell a still-running decode not to bother.
 *
 * `gone` matters because a slot can leave the buffer while its `createImageBitmap` is still in
 * flight. Without the flag that decode would resolve onto an orphaned object and leak a bitmap.
 * The one on screen is spared: `liveBitmap` is still referenced by the canvas.
 */
function releaseSlot(f) {
  if (!f) return;
  f.gone = true;                 // also tells an in-flight decode to discard its result
  if (f.bitmap && f.bitmap !== liveBitmap && f.bitmap.close) f.bitmap.close();
  f.bitmap = null;
  f.bytes = null;                // an undecoded slot holds the raw JPEG; release it too
}

function clearPlaybackState(resetEpoch) {
  frameBuffer.forEach(releaseSlot);
  frameBuffer = [];
  cursor = 0;
  lastDisplayedFrameIndex = -1;
  lastDisplayedBlockIndex = -1;
  recvTimes.length = 0;
  drawTimes.length = 0;
  if (resetEpoch) currentEpoch = -1;
}

/** Start decodes, lowest index first, up to the concurrency limit.
 *
 * Lowest-index-first matters: `cursor` can only advance when ITS frame is decoded, so letting the
 * browser's decoder pool work on later frames while the head waits stalls display even though
 * plenty of frames are ready.
 */
function pumpDecodes() {
  for (let i = cursor; i < frameBuffer.length && decodeInFlight < MAX_DECODE_IN_FLIGHT; i++) {
    const s = frameBuffer[i];
    if (s.gone || s.decoding || s.bitmap || s.failed || !s.bytes) continue;
    s.decoding = true;
    decodeInFlight += 1;
    createImageBitmap(new Blob([s.bytes], { type: "image/jpeg" }))
      .then((bmp) => {
        if (s.gone) { if (bmp.close) bmp.close(); }
        else { s.bitmap = bmp; s.bytes = null; }
      })
      .catch(() => { framesFailed += 1; s.failed = true; s.bytes = null; })
      .finally(() => {
        decodeInFlight -= 1;
        s.decoding = false;
        pumpDecodes();      // free capacity -> start the next frame
        schedulePaint();    // the head may just have become ready
      });
  }
}

/** Ask for one paint on the next compositor frame. Idempotent. */
function schedulePaint() {
  if (paintPending) return;
  paintPending = true;
  // Hold off until the clock allows the next frame. rAF still does the actual painting so we
  // never paint twice inside one compositor frame.
  const wait = PLAYBACK_INTERVAL_MS - (performance.now() - lastPaintMs);
  if (wait > 0) setTimeout(() => requestAnimationFrame(paintFrame), wait);
  else requestAnimationFrame(paintFrame);
}

/** Paint at most one frame, in order, and only if it is ready.
 *
 * If the head is still decoding we simply return WITHOUT re-arming: the decode's `finally` calls
 * schedulePaint(), so readiness is what wakes us; the clock only bounds how often.
 */
function maybeSignalBufferLow() {
  if (paceMinBuffer <= 0) return;
  const depth = frameBuffer.length - cursor;
  if (depth < paceMinBuffer) {
    if (!bufferLowSignalled) {
      bufferLowSignalled = true;
      try {
        socket.emit("buffer_low", { displayed: lastDisplayedFrameIndex, buffered: depth });
      } catch (e) {}
    }
  } else if (depth >= paceMinBuffer + 2) {
    // small hysteresis so a buffer sitting exactly on the floor does not re-arm every paint
    bufferLowSignalled = false;
  }
}

function paintFrame() {
  paintPending = false;
  // A failed decode has no pixels to show; step over it without spending a paint. This is not
  // dropping -- the frame never existed as an image.
  while (cursor < frameBuffer.length && frameBuffer[cursor].failed) cursor++;
  if (cursor >= frameBuffer.length) return;
  const f = frameBuffer[cursor];
  if (!f.bitmap) return;                    // head still decoding -> wait to be woken
  cursor++;
  lastPaintMs = performance.now();          // advance the playback clock
  lastDisplayedFrameIndex = typeof f.frame_index === "number" ? f.frame_index : cursor - 1;
  lastDisplayedBlockIndex =
    typeof f.block_index === "number" ? f.block_index : lastDisplayedBlockIndex;
  if (liveBitmap && liveBitmap.close) liveBitmap.close();   // release the previous decode
  liveBitmap = f.bitmap;
  liveBitmapIsGenerated = true;
  noteTime(drawTimes);
  maybeSignalBufferLow();
  if (f.t_arrived) {
    latencyMs.push(performance.now() - f.t_arrived);
    while (latencyMs.length > 120) latencyMs.shift();
  }
  draw();
  updateShownUses();
  pumpDecodes();     // cursor moved: new decode candidates inside the window
  schedulePaint();   // more may be ready -- next one no sooner than the clock allows
}

// Count the frame when it is actually painted, not when .src is assigned. Assigning a ~100 KB
// data URL only *starts* a decode; if the decode takes longer than the 66 ms playback tick the
// next assignment supersedes it and that frame is never drawn at all. Timing the assignment
// therefore over-reported "draw fps" and hid exactly that.


// -------------------------------------------------------------------------- socket events

socket.on("connect", () => {
  $("link-status").textContent = "connected";
  $("link-status").className = "pill pill-ok";
  try {
    $("link-status").textContent = socket.io.engine.transport.name;
    setText("s-transport", socket.io.engine.transport.name);
  } catch (e) {}
});
socket.on("disconnect", () => {
  $("link-status").textContent = "disconnected";
  $("link-status").className = "pill pill-err";
});
try {
  socket.io.engine.on("upgrade", () => {
    $("link-status").textContent = socket.io.engine.transport.name;
    setText("s-transport", socket.io.engine.transport.name);
  });
} catch (e) {}

socket.on("status", (m) => {
  const text = m.message || "status";
  // The server never told the client a run had ended, so `generating` stayed true after a
  // natural finish -- which also left the state tile reading "generating" forever.
  if (/Generation (finished|stopped)/i.test(text)) {
    generating = false;
    setForceControlsEnabled(true);
  }
  if (m.kind === "error" && saveInFlight) { saveInFlight = false; $("saveBtn").disabled = false; }
  message(text, m.kind === "error" ? "error" : m.kind === "warning" ? "" : "ok");
  // `applied at` comes from force_change_debug's structured video_start_frame now, not from
  // scraping this prose line.
});

socket.on("default_config", (d) => {
  if (d && Number.isFinite(d.seed)) seedInputEl.value = d.seed;
  if (d && Number.isFinite(d.pace_min_buffer)) paceMinBuffer = d.pace_min_buffer;
  if (d && d.captioner_model) {
    setCaption(d.captioner_ready ? "idle" : "loading", d.captioner_ready
      ? "captioner ready — upload an image to fill this in"
      : `captioner loading (${String(d.captioner_model).split("/").pop()})`);
  } else {
    setCaption("off", "auto-caption disabled");
  }
});

socket.on("model_loaded", (d) => {
  const pill = $("model-status");
  if (!d || !d.ok) {
    const loading = d && d.loading;
    pill.textContent = loading ? "loading model…" : "model load failed";
    pill.className = loading ? "pill pill-wait" : "pill pill-err";
    if (!loading) message((d && d.error) || "model preload failed", "error");
    return;
  }
  const m = d.metadata || {};
  pill.textContent = "ready";
  pill.className = "pill pill-ok";
  fullClipFrames = Number(m.num_video_frames || 0);
  clipFrames = galleryLatents ? (galleryLatents - 1) * 4 + 1 : fullClipFrames;
  setText("s-device", m.device || "–");
  setText("s-clip", `${m.num_latent_frames} lat / ${m.num_video_frames} fr`);
  setText("s-blocksize", m.num_frame_per_block
    ? `${m.num_frame_per_block} lat = ${m.num_frame_per_block * 4} fr`
    : "–");
  setText("s-vram", m.vram || "–");
  setText("s-weights", m.use_ema ? "generator_ema" : "generator");
  setStartEnabled();
});

socket.on("perf", (p) => {
  const io_ = (p && p.io_stats) || {};
  const live = (p && p.live) || {};
  setText("s-genfps", live.gen_fps != null ? live.gen_fps.toFixed(1) : "–");
  setText("s-mbps", live.mb_per_s != null ? live.mb_per_s.toFixed(2) : "–");
  // Generation only. The delivered figure is still logged server-side as `[ttff] first frame
  // reached the browser after ...`, along with everything else this panel used to show.
  const g = Number(io_.ttff_generated_ms || 0);
  setText("s-ttff", g ? `${(g / 1000).toFixed(2)} s` : "–");
  if (live.vram) setText("s-vram", live.vram);
  if (live.transport) setText("s-transport", live.transport);
});

// One of these per force change: 'requested' when it arrives, 'completed' once the generator
// has actually picked it up.
let forceChanges = 0;
socket.on("force_change_debug", (d) => {
  if (!d) return;
  if (d.event === "requested") {
    setText("s-applied", "queued…");
    if (pendingForceLabels.length) forceLabelBySeq[d.seq] = pendingForceLabels.shift();
  } else if (d.event === "completed") {
    forceChanges += 1;
    setText("s-changes", String(forceChanges));
    const at = Number(d.video_start_frame);
    const full = Number(d.full_video_frame);
    if (Number.isFinite(at) && at >= 0) {
      // Two boundaries: `at` is where the new hint first reaches a block (which may already be
      // part-way through denoising, so it responds partially), `full` is the first block whose
      // every denoising step saw it. `full` is what you actually see change.
      setText("s-applied", Number.isFinite(full) && full > at
        ? `frame ${at} → full ${full}`
        : `frame ${at}`);
      const label = forceLabelBySeq[d.seq];
      if (label) {
        forceTimeline.push({ video_first: at, label });
        forceTimeline.sort((a, b) => a.video_first - b.video_first);
      }
    }
    delete forceLabelBySeq[d.seq];
  }
});

/** Refresh `shown frame uses` for whatever frame is on screen. */
function updateShownUses() {
  let label = null;
  for (const e of forceTimeline) {
    if (e.video_first <= lastDisplayedFrameIndex) label = e.label; else break;
  }
  if (label !== lastShownUses) {
    lastShownUses = label;
    setText("s-shown-uses", label || "–");
  }
}

// Wire format: 12-byte little-endian header (frame_index, block_index, epoch) then the JPEG.
socket.on("frame_ready", (buf) => {
  if (!acceptIncomingFrames) return;
  const bytes = buf instanceof ArrayBuffer ? buf : (buf && buf.buffer) || null;
  if (!bytes || bytes.byteLength < 13) return;
  const head = new DataView(bytes, 0, 12);
  const frame_index = head.getUint32(0, true);
  const block_index = head.getUint32(4, true);
  const e = head.getUint32(8, true);
  // Epochs only ever increase (the server bumps stream_epoch on Start), so take the newest one
  // seen and reject anything older. The previous test pinned currentEpoch to whatever arrived
  // FIRST: a frame still sitting in the tunnel from the last run would set it to the old epoch,
  // and then every frame of the new run failed `e !== currentEpoch` and the picture froze.
  if (e < currentEpoch) return;
  if (e > currentEpoch) {
    currentEpoch = e;
    // Discard leftovers from the superseded run only -- not frames of the current one.
    frameBuffer.forEach((fr) => { if (fr.epoch !== e) releaseSlot(fr); });
    frameBuffer = frameBuffer.filter((fr) => fr.epoch === e);
    cursor = 0;
  }

  noteTime(recvTimes);
  framesReceived += 1;

  // Reserve the slot NOW, while we are still in arrival order.
  //
  // Decoding runs off the main thread, and `createImageBitmap` promises resolve in
  // decode-completion order, not creation order -- JPEG decode time depends on the image. So
  // pushing the frame AFTER the await put the buffer out of order, and playback walks the buffer
  // by position, so frames were painted out of order too: N+1, then N, then N+2, which looked
  // like the subject moving backwards and then jumping forwards. The saved mp4 was always clean
  // because it is written server-side, before any of this.
  //
  // The raw JPEG is kept on the slot and decoded by pumpDecodes(), not awaited here: decoding
  // every arrival immediately is what made a post-stall burst launch hundreds of concurrent
  // decodes. `t_arrived` is what makes receive-to-visible latency measurable.
  frameBuffer.push({
    frame_index, block_index, epoch: e,
    bytes: new Uint8Array(bytes, 12),      // a view; no copy
    bitmap: null, failed: false, gone: false, decoding: false,
    t_arrived: performance.now(),
  });
  pumpDecodes();
  schedulePaint();
});

socket.on("restart_cutover", (p) => {
  const cut = Number(p?.cut_frame_index ?? -1);
  const nextEpoch = Number(p?.epoch ?? -1);
  if (Number.isFinite(nextEpoch) && nextEpoch >= 0) currentEpoch = nextEpoch;
  if (!Number.isFinite(cut) || cut < 0) return;
  frameBuffer = frameBuffer.filter((fr) => {
    const idx = typeof fr.frame_index === "number" ? fr.frame_index : -1;
    const keep = idx >= 0 && idx < cut;
    if (!keep) releaseSlot(fr);
    return keep;
  });
  if (cursor > frameBuffer.length) cursor = frameBuffer.length;
  message(`cutover applied at frame ${cut}`, "ok");
});

// ---- captioning ----------------------------------------------------------------------------

function setCaption(state, text) {
  const bar = $("caption-bar");
  bar.className = "caption-bar" + (state === "working" ? " working" : state === "ready" ? " ready" : state === "error" ? " err" : "");
  setText("caption-text", text);
}

socket.on("caption_status", (d) => {
  if (d && d.state === "working") setCaption("working", "captioning the image…");
  else if (d && d.state === "error") setCaption("error", `caption failed: ${d.error || "?"}`);
});

socket.on("caption_ready", (d) => {
  const caption = (d && d.caption) || "";
  if (!caption) return;
  const current = promptEl.value.trim();
  // Only fill an empty box, or replace a caption we put there ourselves -- never clobber
  // a prompt that was typed by hand.
  if (!current || current === lastAutoCaption.trim()) {
    promptEl.value = caption;
    lastAutoCaption = caption;
    setCaption("ready", `auto-captioned in ${(d.took_s || 0).toFixed(1)}s — edit freely`);
    pushInput(false);
  } else {
    setCaption("ready", "caption ready but the prompt was edited; left as-is");
  }
});

// -------------------------------------------------------------------------- lifecycle

/** A JPEG data URL of what is on screen, made only when something asks for one.
 *
 * The server's `set_input` / `start` / `set_reference_frame` take `reference_image_data` as a
 * data URL, and a Start that RESUMES a part-watched clip continues from the frame you were
 * looking at. Rendering one per frame was the expensive part; one per click costs nothing.
 */
function frameDataUrl() {
  if (!liveBitmap || !liveBitmapIsGenerated) return baseImageDataUrl;
  const off = document.createElement("canvas");
  off.width = canvas.width;
  off.height = canvas.height;
  off.getContext("2d").drawImage(liveBitmap, 0, 0, off.width, off.height);
  return off.toDataURL("image/jpeg", 0.9);
}

function pushInput(includeImage) {
  socket.emit("set_input", {
    prompt: promptEl.value,
    mode: modeSel.value,
    payload_text: payloadText(),
    reference_image_data: includeImage ? baseImageDataUrl : undefined,
  });
}

$("startBtn").onclick = () => {
  if (imageProcessing) return message("image is still processing", "error");
  if (!hasUploadedImage || !baseImageDataUrl) return message("choose an image first", "error");
  const startSeed = parseInt(seedInputEl.value || "0", 10);
  if (!Number.isFinite(startSeed)) return message("seed must be an integer", "error");
  acceptIncomingFrames = true;
  generating = true;
  setForceControlsEnabled(!galleryPristine);
  forceChanges = 0;
  // Per-RUN counters. The server compares these against its own per-run sent_frames, which it
  // resets on start, so letting these accumulate across runs made `inflight` go negative by the
  // length of every previous run.
  framesReceived = 0;
  framesFailed = 0;
  setText("s-changes", "0");
  setText("s-applied", "–");
  // The clip is frames 0..clipFrames-1 and that is the whole budget.
  //   stopped part-way  -> RESUME: carry on from the frame on screen, keeping its numbering,
  //                        the saved clip and the force timeline, up to the last frame.
  //   watched to the end -> NEW CLIP: back to frame 0 and the uploaded image.
  // Read BEFORE clearPlaybackState(), which resets lastDisplayedFrameIndex to -1.
  clipFrames = galleryLatents ? (galleryLatents - 1) * 4 + 1 : fullClipFrames;
  const continuing = liveBitmapIsGenerated && lastDisplayedFrameIndex >= 0
    && lastDisplayedFrameIndex < clipFrames - 1;
  const contFrom = continuing ? lastDisplayedFrameIndex : -1;
  const segStart = continuing ? contFrom + 1 : 0;
  // The segment's first frame onwards uses what is set right now, so the readout is populated
  // from the first frame instead of staying blank until the first change. Earlier segments'
  // entries are kept: they still label the frames already on the timeline.
  forceTimeline = (continuing ? forceTimeline.filter((e) => e.video_first < segStart) : [])
    .concat([{ video_first: segStart, label: describeForce() }]);
  pendingForceLabels = [];
  forceLabelBySeq = {};
  lastShownUses = null;
  setText("s-shown-uses", "–");
  setText("s-ui-force", describeForce());
  clearPlaybackState(true);
  pushInput(true);
  socket.emit("start", {
    prompt: promptEl.value,
    mode: modeSel.value,
    payload_text: payloadText(),
    // Resuming continues from the frame on screen, so the picture carries on where it stopped.
    // A finished clip starts over from the upload -- see `continuing` above.
    reference_image_data: continuing ? frameDataUrl() : baseImageDataUrl,
    // Tells the server this is a continuation and which frame it continues from. The server
    // cannot infer it: generation runs ahead of playback, so its own frame counter points past
    // the frame being sent back as the reference.
    continue_from_last: continuing,
    continue_from_frame: contFrom,
    seed: startSeed,
    // Only present for gallery presets; the server falls back to its configured maximum.
    latents: galleryLatents || undefined,
  });
};

let saveInFlight = false;

$("saveBtn").onclick = () => {
  saveInFlight = true;
  $("saveBtn").disabled = true;
  const arrows = $("saveArrows").checked;
  setText("save-line", arrows ? "burning arrows and writing the mp4 on the server…"
                              : "writing the mp4 on the server…");
  socket.emit("save_video", { arrows });
};

socket.on("save_ready", (d) => {
  saveInFlight = false;
  $("saveBtn").disabled = false;
  if (!d || !d.url) return;
  const mb = (d.size_mb || 0).toFixed(1);
  const secs = (d.seconds || 0).toFixed(1);
  // Trigger the download without navigating away from the page.
  const a = document.createElement("a");
  a.href = d.url;
  a.download = "";
  document.body.appendChild(a);
  a.click();
  a.remove();
  const el = $("save-line");
  el.innerHTML = `saved ${d.frames} frames (${secs}s, ${mb} MB)` +
    (d.arrows ? ` with arrows, ${d.force_segments || 1} force segment(s)` : "") +
    (d.partial ? " <b>[partial - run still going]</b>" : "") +
    ` &middot; <a class="link" href="${d.url}" download>download again</a>` +
    `<span class="path">${d.path}</span>`;
  message(`saved ${d.frames} frames to ${d.path}`, "ok");
});

$("stopBtn").onclick = () => {
  socket.emit("stop");
  generating = false;
  setForceControlsEnabled(true);   // don't rely on the status message to unfreeze
  message("stop requested — waiting for shown to catch up to buffered", "ok");
};

/** Adopt an already-decoded image as the reference frame.
 *
 * Everything downstream (Start, force overlay, save) reads `baseImageDataUrl` / `liveBitmap`, so
 * the gallery has to reproduce exactly what the upload path produces -- normalised to the canvas
 * resolution and re-encoded as JPEG so the Socket.IO payload stays small.
 */
function adoptReferenceImage(img) {
  const off = document.createElement("canvas");
  off.width = canvas.width;
  off.height = canvas.height;
  off.getContext("2d").drawImage(img, 0, 0, off.width, off.height);
  baseImageDataUrl = off.toDataURL("image/jpeg", 0.9);
  if (liveBitmap && liveBitmap.close) liveBitmap.close();
  liveBitmap = null;
  liveBitmapIsGenerated = false;
  createImageBitmap(off)
    .then((b) => { liveBitmap = b; liveBitmapIsGenerated = false; draw(); })
    .catch(() => message("could not display the image", "error"));
  $("canvas-empty").classList.add("hidden");
  imageProcessing = false;
  hasUploadedImage = true;
  setStartEnabled();
}

// -------------------------------------------------------------------------- gallery presets

/** Enable/disable the force controls. Used to freeze them while a pristine preset runs. */
function setForceControlsEnabled(on) {
  [angleInputEl, magInputEl, anchorXInputEl, anchorYInputEl].forEach((el) => { el.disabled = !on; });
  modeButtons.forEach((b) => { b.disabled = !on; });
}

/** The force was edited before Start: this is no longer the curated case, so drop the cap. */
function invalidatePreset() {
  if (applyingPreset || !galleryPristine) return false;
  galleryPristine = false;
  galleryLatents = null;
  clipFrames = fullClipFrames;
  markGalleryActive(null);
  message("force edited — this run is no longer the gallery case, full length restored", "ok");
  return true;
}

/** True if a force edit should be refused because a pristine preset is mid-run. */
function forceLocked() {
  if (generating && galleryPristine) {
    message("this gallery case runs with a fixed force — edit the force before Start to interact");
    return true;
  }
  return false;
}

function markGalleryActive(id) {
  galleryActiveId = id;
  document.querySelectorAll("#galleryGrid .gallery-item").forEach((el) => {
    el.classList.toggle("active", el.dataset.id === id);
  });
}

/** Load a preset: image, prompt, mode, the four force boxes, and the clip cap. */
function applyPreset(item) {
  acceptIncomingFrames = false;
  generating = false;
  clearPlaybackState(true);
  imageProcessing = true;
  hasUploadedImage = false;
  setStartEnabled();
  message(`loading “${item.title}”…`);

  const img = new Image();
  img.onload = () => {
    applyingPreset = true;
    setForceControlsEnabled(true);
    adoptReferenceImage(img);

    // Order matters: setMode() calls resetForceStateForMode(), which zeroes the force and clears
    // pointPlaced. So switch mode FIRST, then fill the boxes, then apply them.
    setMode(item.mode);
    angleInputEl.value = String(item.angle);
    magInputEl.value = String(item.magnitude);
    if (item.mode === "point") {
      anchorXInputEl.value = String(item.anchor_x);
      anchorYInputEl.value = String(item.anchor_y);
    }
    // The inputs listen for `change`, which assigning .value does not fire -- so apply by hand.
    // This also sets pointPlaced, without which the arrow would not be drawn.
    applyForceInputsToState();

    // Assigning .value does not fire the textarea's `change` either. Leaving lastAutoCaption
    // alone means caption_ready treats this as hand-typed and will not overwrite it.
    promptEl.value = item.prompt;
    setCaption("ready", "preset prompt — edit freely");

    applyingPreset = false;
    galleryPristine = true;
    galleryLatents = Number(item.latents) || null;
    if (galleryLatents) clipFrames = (galleryLatents - 1) * 4 + 1;
    setText("fileNameLabel", item.file);
    markGalleryActive(item.id);
    // Sync prompt/mode/force to the server without asking for a caption: pushInput(true) would
    // start the captioner on an image we already have a prompt for.
    pushInput(false);
    const secs = galleryLatents ? ((galleryLatents - 1) * 4 + 1) / 16 : 0;
    message(`“${item.title}” ready — ${item.mode} force set` +
            (galleryLatents ? `, clip capped to ${galleryLatents} latents (~${secs.toFixed(0)}s)` : "") +
            " — press Start", "ok");
  };
  img.onerror = () => {
    imageProcessing = false; hasUploadedImage = false; setStartEnabled();
    message(`could not load ${item.file}`, "error");
  };
  img.src = `/assets/${item.file}`;
}

fetch("/assets/gallery.json")
  .then((r) => r.json())
  .then((doc) => {
    const items = (doc && doc.items) || [];
    const grid = $("galleryGrid");
    grid.innerHTML = "";
    items.forEach((item) => {
      // A <button> for keyboard access. It must NOT live inside #modeSeg: modeButtons is
      // captured once at load from that container and every member is treated as a mode switch.
      const b = document.createElement("button");
      b.type = "button";
      b.className = "gallery-item";
      b.dataset.id = item.id;
      b.title = `${item.mode} · angle ${item.angle}° · magnitude ${item.magnitude}` +
                (item.mode === "point" ? ` · anchor ${item.anchor_x},${item.anchor_y}` : "");
      const im = document.createElement("img");
      im.src = `/assets/${item.file}`;
      im.alt = item.title;
      const badge = document.createElement("span");
      badge.className = "g-badge";
      badge.textContent = item.mode;
      const lab = document.createElement("span");
      lab.className = "g-label";
      lab.textContent = item.title;
      b.append(im, badge, lab);
      b.onclick = () => applyPreset(item);
      grid.appendChild(b);
    });
    if (!items.length) message("no presets found in assets/gallery.json", "error");
  })
  .catch(() => message("could not load assets/gallery.json", "error"));

$("pickImageBtn").onclick = () => $("imageUpload").click();

$("imageUpload").addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;
  setText("fileNameLabel", file.name || "selected file");
  // Your own image runs at the server's configured length; only presets are capped.
  galleryLatents = null;
  galleryPristine = false;
  clipFrames = fullClipFrames;      // back to the server's configured length
  setForceControlsEnabled(true);
  markGalleryActive(null);
  acceptIncomingFrames = false;
  generating = false;
  clearPlaybackState(true);
  imageProcessing = true;
  hasUploadedImage = false;
  setStartEnabled();
  message("processing image…");
  const reader = new FileReader();
  reader.onload = () => {
    const img = new Image();
    img.onload = () => {
      adoptReferenceImage(img);
      resetForceStateForMode(modeSel.value);
      message("image ready — press Start", "ok");
      pushInput(true);          // this is what triggers auto-captioning server-side
    };
    img.onerror = () => {
      imageProcessing = false; hasUploadedImage = false; setStartEnabled();
      message("failed to decode the image", "error");
    };
    img.src = reader.result;
  };
  reader.onerror = () => {
    imageProcessing = false; hasUploadedImage = false; setStartEnabled();
    message("failed to read the file", "error");
  };
  reader.readAsDataURL(file);
});

// -------------------------------------------------------------------------- stats tick

let framesReceived = 0;
let framesFailed = 0;

// Tell the server what actually arrived. `emit()` on the server side only proves a packet was
// queued inside engine.io (an unbounded queue), so `emitted - received` is the only honest
// measure of how far behind this browser is.
setInterval(() => {
  try {
    socket.emit("client_stats", {
      received: framesReceived,
      recv_fps: Number(fpsOf(recvTimes).toFixed(2)),
      // Painted, not arrived. Reporting only one of these made "not arriving" and "arriving
      // but not painted" indistinguishable in the server log -- two very different faults.
      draw_fps: Number(fpsOf(drawTimes).toFixed(2)),
      failed: framesFailed,
      displayed: lastDisplayedFrameIndex,
      // The server prefers this over `emitted - shown`, which also counts frames still in flight.
      buffered: frameBuffer.length - cursor,
    });
  } catch (e) {}
  // 200 ms, not 1000: the server paces generation off `displayed`, so a 1 s-stale value made the
  // watermark overshoot badly -- it held until the viewer had fully caught up (gap 0) instead of
  // releasing at the low mark. The payload is a few tens of bytes.
}, 200);

setInterval(() => {
  setText("s-state", generating ? "generating" : hasUploadedImage ? "ready" : "idle");
  // The generator's own frame index, straight from the frame header -- the same coordinate as
  // `applied at`. A buffer position would be in different units, and would reset to 0 on an
  // epoch purge.
  setText("s-shown", lastDisplayedFrameIndex < 0
    ? "–"
    : `${lastDisplayedFrameIndex}` + (clipFrames ? ` / ${clipFrames}` : ""));
  // The server keeps the frames, so Save stays available after a run ends or is stopped.
  // Only ever enables it here; the click handler owns disabling while a write is in flight.
  if (framesReceived > 0 && !saveInFlight) $("saveBtn").disabled = false;
  setText("s-block", lastDisplayedBlockIndex >= 0 ? String(lastDisplayedBlockIndex) : "–");
  setText("s-recv", fpsOf(recvTimes).toFixed(1));
  // Now meaningful: with no artificial clock, draw ~= recv means the client keeps up, and
  // draw << recv means decode/render is genuinely the bottleneck.
  setText("s-draw", fpsOf(drawTimes).toFixed(1));
  setText("s-lag", String(Math.max(0, frameBuffer.length - cursor)) +
                   (decodeInFlight ? ` (${decodeInFlight} dec)` : ""));
  // Receive -> visible. The number to minimise; neither fps figure captures it.
  if (latencyMs.length) {
    const v = [...latencyMs].sort((a, b) => a - b);
    setText("s-latency", `${v[v.length >> 1].toFixed(0)} ms`);
  }
}, 500);

resetForceStateForMode(modeSel.value);
setStartEnabled();
// v6 re-sent prompt/mode/force every 500 ms. That is now redundant: `change_force` carries
// payload_text itself, `start` carries everything, and the prompt is only read at Start. Two
// messages a second on a link that is already the bottleneck is not free, so it is gone --
// explicit pushes on the events that actually change something replace it.
promptEl.addEventListener("change", () => pushInput(false));
