from __future__ import annotations

import argparse
import struct
import json
import os
import threading
import time
import traceback
from queue import Empty, Queue
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit

import sys

# Rebuild layout: every module lives in this one directory, which sits directly under the
# repo root. (In the original v6 tree this file was at <repo>/scripts/interactive_demo/v6/,
# hence the parents[3] / parents[1] indices.)
DEMO_ROOT = Path(__file__).resolve().parent
REPO_ROOT = DEMO_ROOT.parent
for _p in (REPO_ROOT, DEMO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from interactive_demo_backend_rolling_forcing import (
    DemoBackendConfig,
    GenerationStopped,
    InteractiveDemoBackend,
)
from interactive_demo_force_adapter import ForceAdapterConfig, build_model_condition_signal, parse_ui_force_payload

from captioner import ImageCaptioner


FORCE_MIN = 0.0
FORCE_MAX = 1.0
MAX_POINT_FORCE_LEN = 80.0
MAX_WIND_FORCE_LEN = 80.0
# Rolling forcing rolls its KV cache (sink + sliding window) instead of holding every latent,
# so the run length is bounded by the RoPE table rather than by cache capacity.
MAX_SEGMENT_LATENT_FRAMES = 1021


@dataclass
class SessionRuntime:
    sid: str
    prompt: str = ""
    mode: str = "point"
    payload_text: str = ""

    reference_frame: Optional[np.ndarray] = None
    current_display_frame: Optional[np.ndarray] = None

    generating: bool = False
    pause_requested: bool = False
    stop_requested: bool = False
    restart_requested: bool = False
    pending_change_target_block: Optional[int] = None
    latest_decoded_block: int = -1
    rolling_window_blocks: int = 1
    stream_epoch: int = 0

    frame_counter: int = 0
    worker: Optional[threading.Thread] = None
    sender_worker: Optional[threading.Thread] = None
    frame_queue: Queue = field(default_factory=lambda: Queue(maxsize=512))
    sender_stop: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)
    model_loaded: bool = False
    config_path: Optional[str] = None
    checkpoint_path: Optional[str] = None
    output_dir: Optional[str] = None
    height: Optional[int] = None
    width: Optional[int] = None
    use_ema: Optional[bool] = None
    seed: Optional[int] = None
    num_latent_frames: Optional[int] = None
    loading_model: bool = False
    decoded_block_last_frame: Dict[int, np.ndarray] = field(default_factory=dict)
    decoded_block_last_frame_index: Dict[int, int] = field(default_factory=dict)
    force_change_seq: int = 0
    applied_force_seq: int = 0     # mirror of the worker's closure, for logging only
    # Diagnostics for "a force change was requested but never applied, with no error anywhere".
    # If update_calls stops climbing, the pipeline stopped asking. If it climbs while
    # update_seq_seen == update_applied_seen, the guard is rejecting -- i.e. force_change_seq is
    # not what on_change_force appeared to set.
    # Video frame the denoising loop is working on. Distinct from frame_counter, which tracks
    # what the (async) VAE decoder has EMITTED -- the two can be hundreds of frames apart.
    denoise_frontier: int = -1
    # Smallest gap the rolling window can ever reach: the blocks it is refining are denoised but
    # not emitted until they go clean, so `den - shown` has a hard floor. Watermarks below it make
    # the release condition unreachable -> permanent hold.
    pace_floor: int = 0
    update_calls: int = 0
    update_seq_seen: int = -1
    update_applied_seen: int = -1
    pending_force_change_debug: list[Dict[str, Any]] = field(default_factory=list)
    io_stats: Dict[str, float] = field(
        default_factory=lambda: {
            "queued_frames": 0.0,
            "sent_frames": 0.0,
            "queue_wait_ms": 0.0,
            "encode_ms": 0.0,
            "emit_ms": 0.0,
            "chunk_cb_ms": 0.0,
            # added for observability (no change to the send policy)
            "sent_bytes": 0.0,
            "ttff_generated_ms": 0.0,
            "ttff_sent_ms": 0.0,
        }
    )
    generation_started_at: float = 0.0
    transport: str = "?"
    # What the browser says it has actually received. `emitted - received` is the only
    # end-to-end measure of in-flight backlog that does not depend on library internals.
    client_received: int = 0
    client_recv_fps: float = 0.0
    client_draw_fps: float = 0.0
    inflight_warned: bool = False
    # Per-run clip cap, set by a gallery preset. None => use the configured maximum, which is
    # what the normal upload flow always does.
    run_latents: Optional[int] = None
    # Generation pacing (hysteresis watermark). `pacing_holding` is the latch: it turns on above
    # the high watermark and off below the low one, so the generator does not chatter around a
    # single threshold.
    pacing_holding: bool = False
    pacing_holds: int = 0
    pacing_held_s: float = 0.0
    pacing_hold_started: float = 0.0
    client_displayed: int = -1      # frame index actually on screen, as reported by the browser
    client_buffered: int = -1       # frames received but not yet painted, as reported by the
                                    # browser. `emitted - shown` counts frames still in
                                    # flight, so it overstates what the viewer can paint.
    # Full-resolution frames kept for saving. These are the frames as generated, BEFORE the
    # wire downscale/JPEG in the sender, so a saved mp4 is unaffected by --wire_scale and
    # --jpeg_quality. Accumulated in the chunk callback so Stop still leaves something to save.
    saved_frames: list = field(default_factory=list)
    # [{"video_first": int, "ui": {...}}] -- which force governed which frames, so a save
    # can burn in the arrow that was actually acting at each frame.
    force_timeline: list = field(default_factory=list)
    saving: bool = False
    client_reported_at: float = 0.0


APP = Flask(
    __name__,
    template_folder=str(Path(__file__).resolve().parent / "flask_frontend"),
    static_folder=str(DEMO_ROOT / "flask_static"),
    # Without this Flask derives the URL prefix from the folder's basename and would serve the
    # assets at /flask_static/..., while index.html asks for /static/... . v6's page was a single
    # self-contained file, so it never needed the static route at all.
    static_url_path="/static",
)
APP.config["SECRET_KEY"] = "interactive_force_demo"
SOCKET = SocketIO(
    APP,
    cors_allowed_origins="*",
    async_mode="threading",
    max_http_buffer_size=50 * 1024 * 1024,
)

_SESSIONS: Dict[str, SessionRuntime] = {}
_SESSIONS_LOCK = threading.Lock()

_BACKENDS: Dict[str, InteractiveDemoBackend] = {}
_BACKEND_KEYS: Dict[str, Tuple[Any, ...]] = {}
_BACKEND_METADATA: Dict[str, Dict[str, Any]] = {}
_BACKEND_LOAD_ERROR: Optional[str] = None
_BACKEND_LOCK = threading.Lock()

# Auto-captioner (Qwen3-VL). Optional: with --caption_model "" the demo behaves exactly like
# an earlier variant, and the prompt box stays manual.
CAPTIONER: Optional[ImageCaptioner] = None
_CAPTION_SEQ: Dict[str, int] = {}
_CAPTION_LOCK = threading.Lock()


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _captioner_ready() -> bool:
    return CAPTIONER is not None and CAPTIONER.ready


def _caption_async(sid: str, data_url: str) -> None:
    """Caption a freshly uploaded reference image and push the text to that client.

    Runs on its own thread: the VLM takes seconds and this is triggered from a Socket.IO
    handler, which must not block. Each upload bumps a per-session sequence number so a slow
    caption from a superseded image is discarded instead of overwriting a newer one.
    """
    with _CAPTION_LOCK:
        _CAPTION_SEQ[sid] = _CAPTION_SEQ.get(sid, 0) + 1
        seq = _CAPTION_SEQ[sid]

    def _work() -> None:
        if not _captioner_ready():
            return
        SOCKET.emit("caption_status", {"state": "working"}, to=sid)
        try:
            from io import BytesIO
            import base64 as _b64

            from PIL import Image as _Image

            raw = data_url.split(",", 1)[1] if "," in data_url else data_url
            pil = _Image.open(BytesIO(_b64.b64decode(raw))).convert("RGB")
            started = time.perf_counter()
            caption = CAPTIONER.caption(pil)
            took = time.perf_counter() - started
        except Exception as exc:
            _log(f"[captioner] failed: {exc}")
            SOCKET.emit("caption_status", {"state": "error", "error": str(exc)}, to=sid)
            return
        with _CAPTION_LOCK:
            if _CAPTION_SEQ.get(sid) != seq:
                _log("[captioner] discarding caption for a superseded image")
                return
        _log(f"[captioner] {took:.1f}s: {caption[:110]}")
        SOCKET.emit("caption_ready", {"caption": caption, "took_s": took}, to=sid)

    threading.Thread(target=_work, name="caption", daemon=True).start()


def _safe_json_parse(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    return json.loads(text)


def _parse_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    return int(value)


def _session(sid: str) -> SessionRuntime:
    with _SESSIONS_LOCK:
        if sid not in _SESSIONS:
            _SESSIONS[sid] = SessionRuntime(sid=sid)
        return _SESSIONS[sid]


def _build_backend(
    mode: str,
    config_path: str,
    checkpoint_path: str,
    output_dir: str,
    height: int,
    width: int,
    use_ema: bool,
    num_latent_frames: int,
    device: Optional[str] = None,
    vae_device: Optional[str] = None,
    vae_channels_last: bool = False,
    gen_channels_last: bool = False,
) -> InteractiveDemoBackend:
    global _BACKEND_LOAD_ERROR
    mode_key = str(mode)
    key = (
        mode_key,
        config_path,
        checkpoint_path,
        output_dir,
        int(height),
        int(width),
        bool(use_ema),
        int(num_latent_frames),
        str(device) if device is not None else "",
        str(vae_device) if vae_device is not None else "",
        bool(vae_channels_last),
        bool(gen_channels_last),
    )
    with _BACKEND_LOCK:
        if mode_key in _BACKENDS and _BACKEND_KEYS.get(mode_key) == key:
            return _BACKENDS[mode_key]

        cfg = DemoBackendConfig(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            height=height,
            width=width,
            num_latent_frames=int(num_latent_frames),
            use_ema=use_ema,
            seed=0,
            device=device,
            vae_device=vae_device,
            vae_channels_last=vae_channels_last,
            gen_channels_last=gen_channels_last,
            rolling_forcing_block_frames=int(APP.config.get("DEMO_RF_BLOCK_FRAMES", 3)),
            rolling_forcing_max_frames=int(APP.config.get("DEMO_RF_MAX_FRAMES", 21)),
            rolling_forcing_cache_frames=int(APP.config.get("DEMO_RF_CACHE_FRAMES", 96)),
        )
        backend = InteractiveDemoBackend(cfg)
        metadata = backend.load()
        _BACKENDS[mode_key] = backend
        _BACKEND_KEYS[mode_key] = key
        _BACKEND_METADATA[mode_key] = metadata
        _BACKEND_LOAD_ERROR = None
        return backend


# One checkpoint serves both modes, so there is a single backend. `mode` still matters -- it
# selects the point-blob vs all-ones-mask force signal in the adapter -- but it no longer
# selects a model. (The original v6 loaded a separate wind and point checkpoint on cuda:0 and
# cuda:1; an earlier variant did that.)
BACKEND_KEY = "shared"


def _resolve_vae_device(requested: Optional[str], gen_device: Optional[str]) -> Optional[str]:
    """Pick the VAE's GPU. 'auto' takes a second GPU when one is visible, else stays put.

    Decode is the larger half of a block, so on two GPUs it should not share SMs with the
    generator. On one GPU it still runs on its own stream (async decode), which recovers the
    launch bubbles but not the SM contention -- see OPTIMIZATIONS.md.
    """
    req = (requested or "auto").strip().lower()
    if req in ("same", "none", ""):
        return None
    if req != "auto":
        return requested
    if not torch.cuda.is_available():
        return None
    n = torch.cuda.device_count()
    if n < 2:
        print(f"[rf] vae_device=auto: only {n} GPU visible -> VAE shares the generator's GPU")
        return None
    gen_idx = 0
    if gen_device and ":" in str(gen_device):
        try:
            gen_idx = int(str(gen_device).split(":")[1])
        except ValueError:
            gen_idx = 0
    vae_idx = 1 if gen_idx == 0 else 0
    print(f"[rf] vae_device=auto: {n} GPUs visible -> VAE on cuda:{vae_idx}")
    return f"cuda:{vae_idx}"


def _preload_backend(
    config_path: str,
    checkpoint_path: str,
    output_dir: str,
    height: int,
    width: int,
    use_ema: bool,
    num_latent_frames: int,
    device: Optional[str],
    vae_device: Optional[str] = None,
    vae_channels_last: bool = False,
    gen_channels_last: bool = False,
) -> None:
    global _BACKEND_LOAD_ERROR
    try:
        _build_backend(
            mode=BACKEND_KEY,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            height=height,
            width=width,
            use_ema=use_ema,
            num_latent_frames=num_latent_frames,
            device=device,
            vae_device=vae_device,
            vae_channels_last=vae_channels_last,
            gen_channels_last=gen_channels_last,
        )
    except Exception as exc:
        _BACKEND_LOAD_ERROR = str(exc)
        raise


def _backend_ready() -> bool:
    with _BACKEND_LOCK:
        return BACKEND_KEY in _BACKENDS and _BACKEND_LOAD_ERROR is None


def _backend_for_mode(sess: SessionRuntime) -> InteractiveDemoBackend:
    """The one loaded backend, whatever the mode is."""
    with _BACKEND_LOCK:
        backend = _BACKENDS.get(BACKEND_KEY)
    if backend is None:
        raise RuntimeError("Backend is not loaded")
    return backend


def _decode_data_url_image(data_url: str) -> np.ndarray:
    import base64
    from io import BytesIO
    from PIL import Image

    if not data_url:
        raise ValueError("Empty image payload")
    if "," in data_url:
        _, b64 = data_url.split(",", 1)
    else:
        b64 = data_url
    raw = base64.b64decode(b64)
    img = Image.open(BytesIO(raw)).convert("RGB")
    return np.array(img, dtype=np.uint8)


def _build_condition(
    backend: InteractiveDemoBackend, sess: SessionRuntime,
    num_video_frames: Optional[int] = None,
) -> tuple[Dict[str, Any], Any]:
    return _build_condition_from_payload(backend, sess.mode, sess.payload_text,
                                         num_video_frames=num_video_frames)


def _build_condition_from_payload(
    backend: InteractiveDemoBackend,
    mode: str,
    payload_text: str,
    num_video_frames: Optional[int] = None,
    frame_start_video: int = 0,
    total_video_frames: Optional[int] = None,
) -> tuple[Dict[str, Any], Any]:
    payload = _safe_json_parse(payload_text)
    if not payload:
        raise ValueError("No force payload set")
    payload["mode"] = mode

    adapter_cfg = ForceAdapterConfig(
        min_force=FORCE_MIN,
        max_force=FORCE_MAX,
        max_point_force_len=MAX_POINT_FORCE_LEN,
        max_wind_force_len=MAX_WIND_FORCE_LEN,
    )
    converted = parse_ui_force_payload(payload, adapter_cfg)
    condition_signal = build_model_condition_signal(
        force_payload=converted,
        height=backend.cfg.height,
        width=backend.cfg.width,
        num_video_frames=int(num_video_frames or backend.num_video_frames),
        cfg=adapter_cfg,
        frame_start=int(frame_start_video),
        total_video_frames=int(total_video_frames) if total_video_frames is not None else None,
    )
    return converted, condition_signal


def _latent_to_video_frame_index(latent_index: int) -> int:
    idx = int(max(0, latent_index))
    if idx == 0:
        return 0
    return 1 + (idx - 1) * 4


# How stale the browser's last report may be before pacing gives up and lets generation run.
# Failing OPEN matters: if the client stops reporting (tab closed, socket wedged) a closed-fail
# policy would hold generation forever.
_PACING_STALE_S = 3.0


def _pacing_state(sess: SessionRuntime) -> Tuple[int, int, int, int]:
    """(shown, denoised, emitted, gap) in video-frame indices. Caller holds sess.lock.

    The gap that matters is denoised-ahead-of-shown, not emitted-ahead-of-shown: once a latent is
    denoised its pixels are fixed, so a force change can no longer reach it. Since the VAE decode
    moved to a worker thread those two frontiers can be hundreds of frames apart.
    """
    emitted = int(sess.frame_counter) - 1
    denoised = int(sess.denoise_frontier)
    lead = denoised if denoised >= 0 else emitted
    shown = int(sess.client_displayed)
    return shown, denoised, emitted, (lead - shown if shown >= 0 else -1)


def _pacing_should_hold(sess: SessionRuntime) -> bool:
    """Hysteresis watermark on (frontier - shown). Caller holds sess.lock.

    Hold generation when it is more than HIGH frames ahead of what the browser is showing, and
    keep holding until the gap falls to LOW. Nothing is dropped and nothing is regenerated -- the
    generator simply waits, which is why this reuses the existing `should_pause` path rather than
    adding a second mechanism.
    """
    high = int(APP.config.get("DEMO_PACE_HIGH", 0))
    low = int(APP.config.get("DEMO_PACE_LOW", 0))
    if high <= 0:
        return False
    shown, denoised, emitted, gap = _pacing_state(sess)
    frontier = denoised if denoised >= 0 else emitted
    # Lift the watermarks above the structural floor, or the latch can never clear.
    floor = int(sess.pace_floor)
    if floor > 0:
        low = max(low, floor + 5)
        high = max(high, low + 5)
    # Nothing to pace against until the browser has reported a displayed frame, and never trust a
    # stale report.
    fresh = sess.client_reported_at and (time.time() - sess.client_reported_at) <= _PACING_STALE_S
    if shown < 0 or not fresh:
        if sess.pacing_holding:
            sess.pacing_holding = False
            sess.pacing_held_s += max(0.0, time.time() - sess.pacing_hold_started)
            _log(f"[pace] RESUME (client feedback {'stale' if shown >= 0 else 'absent'}) "
                 f"shown={shown} frontier={frontier}")
        return False
    # What keeps playback smooth is the viewer's own queue, and that is NOT the gap being paced.
    # `denoised` leads `emitted` by the rolling window's depth -- 3 blocks, ~36 frames, structural
    # -- so a gap of 45-60 can sit on top of a browser holding only 11-23 frames. Measured: 15
    # mid-playback stalls with the watermark alone. Never hold below the floor; only new frames
    # can refill it, so holding there is exactly backwards. Subsumes the old shown >= emitted
    # deadlock breaker, which only fired once the queue had already hit zero.
    min_buffer = int(APP.config.get("DEMO_PACE_MIN_BUFFER", 0))
    # Prefer the browser's own count of undrawn frames. `emitted - shown` also counts frames
    # still on the wire, so it overstates what the viewer can actually paint -- exactly the
    # error that lets a hold starve it.
    viewer_buffer = sess.client_buffered if sess.client_buffered >= 0 else (emitted - shown)
    if sess.pacing_holding and viewer_buffer < max(min_buffer, 1):
        sess.pacing_holding = False
        sess.pacing_held_s += max(0.0, time.time() - sess.pacing_hold_started)
        _log(f"[pace] RESUME (viewer buffer {viewer_buffer} < floor {max(min_buffer, 1)}: "
             f"shown={shown} emit={emitted}; gap={gap} still above low={low}, but only new "
             f"frames can refill it)")
        return False
    # Hysteresis on the buffer, not just on the gap. Entering a hold at exactly the floor leaves
    # one frame of margin before the release fires -- 60 ms at 16 fps -- which is why a floor of 5
    # stalled 13 times while releasing correctly. Only start holding with a working buffer; keep
    # holding until it falls to the floor.
    hold_entry = min_buffer + int(APP.config.get("DEMO_PACE_BUFFER_MARGIN", min_buffer))
    if not sess.pacing_holding and gap > high and viewer_buffer >= hold_entry:
        sess.pacing_holding = True
        sess.pacing_hold_started = time.time()
        sess.pacing_holds += 1
        _log(f"[pace] HOLD  gap={gap} > high={high} | shown={shown} frontier={frontier} "
             f"buffer={viewer_buffer}>={hold_entry} (hold #{sess.pacing_holds})")
    elif sess.pacing_holding and gap <= low:
        sess.pacing_holding = False
        held = max(0.0, time.time() - sess.pacing_hold_started)
        sess.pacing_held_s += held
        _log(f"[pace] RESUME gap={gap} <= low={low} | shown={shown} frontier={frontier} "
             f"(held {held * 1000:.0f} ms, {sess.pacing_held_s:.1f}s total)")
    return sess.pacing_holding


def _emit_force_change_completion_upto(
    sid: str, sess: SessionRuntime, applied_seq: int, video_start_frame: int = -1,
    full_video_frame: int = -1,
) -> None:
    """Tell the browser a queued force change has been picked up by the generator.

    `video_start_frame` is the first video frame produced under the new force. The panel needs it
    as a structured field: it used to be scraped out of the prose status line with a regex, so
    rewording that log message silently blanked the readout.
    """
    completed_records: list[Dict[str, Any]] = []
    with sess.lock:
        remaining: list[Dict[str, Any]] = []
        completion_last_frame_index = int(sess.frame_counter - 1) if sess.frame_counter > 0 else -1
        completion_latest_decoded_block = int(sess.latest_decoded_block)
        for rec in sess.pending_force_change_debug:
            if int(rec.get("seq", -1)) <= int(applied_seq):
                payload = dict(rec)
                payload.update(
                    {
                        "event": "completed",
                        "completed_at_ms": int(time.time() * 1000),
                        "completed_last_frame_index": int(completion_last_frame_index),
                        "completed_latest_decoded_block": int(completion_latest_decoded_block),
                        "video_start_frame": int(video_start_frame),
                        # First frame whose every denoising step saw the new force. Always >=
                        # video_start_frame: the blocks already in the rolling window take it
                        # only for their remaining steps.
                        "full_video_frame": int(full_video_frame),
                    }
                )
                req_ms = payload.get("requested_at_ms")
                if isinstance(req_ms, int):
                    payload["latency_ms"] = int(max(0, payload["completed_at_ms"] - req_ms))
                completed_records.append(payload)
            else:
                remaining.append(rec)
        sess.pending_force_change_debug = remaining
    for payload in completed_records:
        SOCKET.emit("force_change_debug", payload, to=sid)


def _emit_status(sid: str, message: str, kind: str = "info") -> None:
    SOCKET.emit("status", {"message": message, "kind": kind}, to=sid)


def _reset_io_stats(sess: SessionRuntime) -> None:
    sess.io_stats = {
        "queued_frames": 0.0,
        "sent_frames": 0.0,
        "queue_wait_ms": 0.0,
        "encode_ms": 0.0,
        "emit_ms": 0.0,
        "chunk_cb_ms": 0.0,
        "sent_bytes": 0.0,
        "ttff_generated_ms": 0.0,
        "ttff_sent_ms": 0.0,
    }


def _eio_queue_depth(sid: str) -> int:
    """Packets queued inside engine.io for this client but not yet written to the socket.

    THIS is the real send backlog. `SOCKET.emit()` ends in `engineio.socket.Socket.send()`,
    which does `self.queue.put(pkt)` on an *unbounded* queue and returns -- so it never blocks
    and never reports pressure. The app-level `frame_queue` therefore drains instantly and reads
    0/0 even while the client is minutes behind, which is exactly what the first version of this
    instrumentation showed. -1 means the queue could not be reached.
    """
    try:
        eio_sid = SOCKET.server.manager.eio_sid_from_sid(sid, "/")
        sock = SOCKET.server.eio.sockets.get(eio_sid)
        return int(sock.queue.qsize()) if sock is not None else -1
    except Exception:
        return -1


def _client_transport(sid: str) -> str:
    """'websocket' or 'polling'. A client left on polling pays a round trip per frame."""
    try:
        return str(SOCKET.server.transport(sid))
    except Exception:
        return "?"


def _vram(device: str) -> str:
    try:
        import torch as _t

        if not str(device).startswith("cuda") or not _t.cuda.is_available():
            return "n/a"
        idx = int(str(device).split(":")[1]) if ":" in str(device) else 0
        return (f"{_t.cuda.memory_allocated(idx) / 1024 ** 3:.1f}/"
                f"{_t.cuda.memory_reserved(idx) / 1024 ** 3:.1f} GiB alloc/reserved")
    except Exception:
        return "n/a"


def _drain_frame_queue(sess: SessionRuntime) -> None:
    # Drop queued stale frames so cutover can take effect immediately.
    while True:
        try:
            sess.frame_queue.get_nowait()
            sess.frame_queue.task_done()
        except Empty:
            break


def _clear_generation_state(sess: SessionRuntime) -> None:
    sess.pause_requested = False
    sess.restart_requested = False
    sess.pending_change_target_block = None
    sess.latest_decoded_block = -1
    sess.decoded_block_last_frame.clear()
    sess.decoded_block_last_frame_index.clear()
    sess.frame_counter = 0
    sess.current_display_frame = None if sess.reference_frame is None else sess.reference_frame.copy()
    sess.pending_force_change_debug.clear()
    _reset_io_stats(sess)
    _drain_frame_queue(sess)


def _start_sender_if_needed(sess: SessionRuntime) -> None:
    if sess.sender_worker is not None and sess.sender_worker.is_alive():
        return
    sess.sender_stop = False

    def _sender_loop() -> None:
        while True:
            with sess.lock:
                stop = sess.sender_stop
            if stop and sess.frame_queue.empty():
                break
            try:
                queued_at, frame, frame_idx, block_index, epoch = sess.frame_queue.get(timeout=0.1)
            except Empty:
                continue

            try:
                # Every generated frame is sent. There is deliberately no backpressure gate
                # here: on a slow link the browser falls behind rather than losing frames.
                t0 = time.perf_counter()
                import cv2 as _cv2

                quality = int(APP.config.get("DEMO_JPEG_QUALITY", 85))
                scale = float(APP.config.get("DEMO_WIRE_SCALE", 1.0))
                if scale > 0.0 and abs(scale - 1.0) > 1e-3:
                    h, w = frame.shape[:2]
                    frame = _cv2.resize(
                        frame,
                        (max(2, int(w * scale) // 2 * 2), max(2, int(h * scale) // 2 * 2)),
                        interpolation=_cv2.INTER_AREA,
                    )
                # Raw JPEG bytes, not a base64 data URL inside JSON. base64 cost a flat 33% of
                # the wire for nothing, and JSON then had to escape a ~100 KB string on the way
                # out and parse it on the way in. The 12-byte header carries what the JSON dict
                # used to, so the whole frame is one binary socket.io attachment.
                ok, enc = _cv2.imencode(
                    ".jpg", _cv2.cvtColor(frame, _cv2.COLOR_RGB2BGR),
                    [int(_cv2.IMWRITE_JPEG_QUALITY), quality],
                )
                if not ok:
                    continue
                payload = (
                    struct.pack("<III", int(frame_idx) & 0xFFFFFFFF,
                                int(block_index) & 0xFFFFFFFF, int(epoch) & 0xFFFFFFFF)
                    + enc.tobytes()
                )
                t1 = time.perf_counter()
                SOCKET.emit("frame_ready", payload, to=sess.sid)
                t2 = time.perf_counter()
                with sess.lock:
                    sess.io_stats["sent_frames"] += 1.0
                    sess.io_stats["queue_wait_ms"] += (t0 - queued_at) * 1000.0
                    sess.io_stats["encode_ms"] += (t1 - t0) * 1000.0
                    sess.io_stats["emit_ms"] += (t2 - t1) * 1000.0
                    sess.io_stats["sent_bytes"] += float(len(payload))
                    global _LAST_WIRE_KB
                    _LAST_WIRE_KB = len(payload) / 1024.0
                    if sess.io_stats["ttff_sent_ms"] == 0.0 and sess.generation_started_at:
                        sess.io_stats["ttff_sent_ms"] = (time.time() - sess.generation_started_at) * 1000.0
                        _log(f"[ttff] first frame reached the browser after "
                             f"{sess.io_stats['ttff_sent_ms'] / 1000.0:.2f}s "
                             f"({len(payload) / 1024.0:.0f} KB on the wire)")
            finally:
                sess.frame_queue.task_done()

    sess.sender_worker = threading.Thread(target=_sender_loop, daemon=True)
    sess.sender_worker.start()


STATS_INTERVAL_S = 2.0


def _start_stats_monitor(sid: str, sess: SessionRuntime, device: str) -> None:
    """One [stream] line every 2 s while generating. Observational only."""

    def _loop() -> None:
        last_sent = last_queued = last_bytes = 0.0
        last_t = time.time()
        while True:
            time.sleep(STATS_INTERVAL_S)
            transport = _client_transport(sid)
            eio_depth = _eio_queue_depth(sid)
            with sess.lock:
                if not sess.generating:
                    break
                sess.transport = transport
                st = dict(sess.io_stats)
                depth = sess.frame_queue.qsize()
                client_recv = int(sess.client_received)
                client_fps = float(sess.client_recv_fps)
                pace_shown, pace_den, pace_emit, pace_gap = _pacing_state(sess)
                pace_hold = bool(sess.pacing_holding)
                upd_calls = int(sess.update_calls)
                upd_seq, upd_applied = int(sess.update_seq_seen), int(sess.update_applied_seen)
                draw_fps = float(sess.client_draw_fps)
                client_age = time.time() - sess.client_reported_at if sess.client_reported_at else -1.0
            now = time.time()
            dt = max(now - last_t, 1e-6)
            gen_fps = (st["queued_frames"] - last_queued) / dt
            snd_fps = (st["sent_frames"] - last_sent) / dt
            mbps = (st["sent_bytes"] - last_bytes) / dt / 1024.0 / 1024.0
            last_queued, last_sent, last_bytes, last_t = (
                st["queued_frames"], st["sent_frames"], st["sent_bytes"], now)
            wait = (st["queue_wait_ms"] / st["sent_frames"]) if st["sent_frames"] else 0.0
            behind = int(st["queued_frames"] - st["sent_frames"])
            # `sent` is enqueue rate, NOT delivery -- engine.io's queue is unbounded, so emit
            # always succeeds. `drawn` (reported by the browser) and `inflight` are the honest
            # end-to-end numbers; `eio` is the packet backlog inside engine.io.
            raw_inflight = int(st["sent_frames"]) - client_recv
            # Receiving more than was sent is impossible within a run, so a negative value means
            # the two counters disagree about which run they are counting -- report it rather
            # than printing a negative backlog.
            if raw_inflight < -1 and not sess.inflight_warned:
                sess.inflight_warned = True
                _log(f"[stream] WARNING: inflight={raw_inflight} (sent={int(st['sent_frames'])}, "
                     f"client reported received={client_recv}); counters are out of step, "
                     f"treating as 0")
            inflight = max(0, raw_inflight)
            stale = " client-silent" if client_age < 0 or client_age > 5 else ""
            _log(
                f"[stream] gen {gen_fps:4.1f} | queued {snd_fps:4.1f} | "
                f"recv {client_fps:4.1f} | drawn {draw_fps:4.1f} fps "
                f"| {mbps:4.2f} MB/s enqueued | appq {depth:3d} | eio {eio_depth:4d} | "
                f"inflight {inflight:5d} | shown {pace_shown:4d} den {pace_den:4d} "
                f"emit {pace_emit:4d} "
                f"gap {pace_gap:4d}{' HOLD' if pace_hold else ''} | "
                f"upd {upd_calls:4d} seq {upd_seq}>{upd_applied} | "
                f"{transport}{stale} | vram {_vram(device)}"
            )
            if inflight > 60:
                # `inflight / client_fps` is meaningless until the client has reported a rate;
                # clamping the divisor to 0.1 turned "no data yet" into "~800s of lag".
                lag = f"~{inflight / client_fps:.0f}s" if client_fps > 0.5 else "unknown yet"
                _log(
                    # recv ~0 means the bytes are not reaching the browser at all; recv high
                    # with drawn low would instead mean the tab is not painting what it gets.
                    f"[stream] {'NOT-ARRIVING' if client_fps < 1.0 else 'DELIVERY-LIMITED'}: "
                    f"{inflight} frames emitted, browser receiving {client_fps:.1f} fps "
                    f"(lag {lag}). Note eio={eio_depth}: they are not stuck in engine.io but in "
                    f"the socket/tunnel buffers. Nothing is dropped, so the picture falls "
                    f"further behind rather than losing frames."
                )
            SOCKET.emit(
                "perf",
                {
                    "io_stats": st,
                    "live": {
                        "gen_fps": gen_fps, "sent_fps": snd_fps, "mb_per_s": mbps,
                        "queue_depth": depth, "behind_frames": behind,
                        "transport": transport, "vram": _vram(device),
                        "eio_queue": eio_depth, "inflight": inflight,
                        "client_fps": client_fps,
                        "draw_fps": draw_fps,
                    },
                    "runtime_update_logs": [],
                },
                to=sid,
            )
            if snd_fps > 0.1 and gen_fps > snd_fps * 1.25:
                _log(f"[stream] WIRE-LIMITED: generating {gen_fps:.1f} fps, shipping "
                     f"{snd_fps:.1f} fps. The queue holds up to 512 frames, so the browser falls "
                     f"behind rather than dropping -- the queue is bounded below by --pace_min_buffer.")

    threading.Thread(target=_loop, name=f"stats-{sid[:6]}", daemon=True).start()


def _generation_worker(sid: str) -> None:
    sess = _session(sid)
    try:
        seed = int(APP.config["DEMO_SEED"] if sess.seed is None else sess.seed)
        backend = _backend_for_mode(sess)

        _emit_status(sid, "Generation started")
        with sess.lock:
            _reset_io_stats(sess)
            sess.generation_started_at = time.time()
            _start_sender_if_needed(sess)
        _log(
            f"[gen] start | mode={sess.mode} | seed={seed} | "
            f"{int(backend.cfg.num_latent_frames)} latents -> {int(backend.num_video_frames)} frames "
            f"| device={backend.device} | vram {_vram(str(backend.device))}"
        )
        _log(f"[gen] prompt: {str(sess.prompt)[:110]!r}")
        _start_stats_monitor(sid, sess, str(backend.device))
        with sess.lock:
            try:
                sess.rolling_window_blocks = int(len(backend.pipeline.denoising_step_list))
            except Exception:
                sess.rolling_window_blocks = 1
        with sess.lock:
            run_latents = sess.run_latents
        block = max(1, int(backend.pipeline.num_frame_per_block))
        if run_latents:
            # The pipeline asserts num_frames % num_frame_per_block == 0 on this path.
            run_latents = max(block, (int(run_latents) // block) * block)
            run_video_frames = (run_latents - 1) * 4 + 1
            _log(f"[gen] clip capped to {run_latents} latents = {run_video_frames} video frames "
                 f"(max {int(backend.cfg.num_latent_frames)})")
        else:
            run_video_frames = int(backend.num_video_frames)
        max_output_frames = int(run_video_frames)
        # (steps-1) blocks are always in flight inside the rolling window: denoised, not yet
        # emitted. `den - shown` can never fall below that, so the watermarks must clear it.
        _steps = max(1, len(backend.pipeline.denoising_step_list))
        _blk = max(1, int(backend.pipeline.num_frame_per_block))
        with sess.lock:
            sess.pace_floor = (_steps - 1) * _blk * 4 + 1
        _pace_hi = int(APP.config.get("DEMO_PACE_HIGH", 0))
        _pace_lo = int(APP.config.get("DEMO_PACE_LOW", 0))
        if _pace_hi > 0 and _pace_lo <= sess.pace_floor:
            _log(f"[pace] watermarks raised: the rolling window keeps {_steps - 1} blocks "
                 f"denoised-but-unemitted, so gap cannot go below {sess.pace_floor}. "
                 f"Using low={sess.pace_floor + 5} high={max(_pace_hi, sess.pace_floor + 10)} "
                 f"instead of low={_pace_lo} high={_pace_hi}")
        absorbed_upto = -1        # highest seq already reported as baked into an initial signal

        while True:
            with sess.lock:
                if sess.stop_requested:
                    break
                if sess.reference_frame is None:
                    _emit_status(sid, "No reference frame set", "error")
                    break

            converted, condition_signal = _build_condition(
                backend, sess, num_video_frames=run_video_frames)
            with sess.lock:
                pending_before = int(sess.force_change_seq)
                applied_force_seq = pending_before
                sess.force_timeline = [
                    {"video_first": 0, "ui": dict(converted.get("ui") or {}),
                     "seq": applied_force_seq, "mode": str(sess.mode)}
                ]
                sess.applied_force_seq = applied_force_seq
            # A request that arrives just before the signal is (re)built is baked into it rather
            # than applied mid-run, so it never reaches _get_condition_signal_update. Without
            # this, `applied at` sat on "queued..." forever with no error anywhere: the change had
            # taken effect but nothing said so.
            if pending_before > absorbed_upto:
                _log(f"[force] ABSORBED id<={pending_before} into the initial signal "
                     f"(built at frame 0); nothing to apply mid-run")
                _emit_force_change_completion_upto(sid, sess, pending_before,
                                                   video_start_frame=0, full_video_frame=0)
                absorbed_upto = pending_before

            def _on_chunk(chunk_frames: np.ndarray, block_index: int) -> None:
                cb_start = time.perf_counter()
                n = int(chunk_frames.shape[0])
                cutover_now = False
                cut_frame_idx = -1
                target_block = -1
                enqueue_epoch = 0
                with sess.lock:
                    if sess.stop_requested:
                        return
                    start_idx = int(sess.frame_counter)
                    remaining = int(max_output_frames - start_idx)
                    if remaining <= 0:
                        sess.stop_requested = True
                        return
                    emit_n = int(min(n, remaining))
                    cap = int(APP.config.get("DEMO_MAX_SAVE_FRAMES", 1200))
                    if emit_n > 0 and len(sess.saved_frames) < cap:
                        room = cap - len(sess.saved_frames)
                        for i in range(min(emit_n, room)):
                            sess.saved_frames.append(chunk_frames[i].copy())
                    if sess.io_stats["ttff_generated_ms"] == 0.0 and emit_n > 0 and sess.generation_started_at:
                        sess.io_stats["ttff_generated_ms"] = (time.time() - sess.generation_started_at) * 1000.0
                        _log(f"[ttff] first frame generated after "
                             f"{sess.io_stats['ttff_generated_ms'] / 1000.0:.2f}s")
                    sess.frame_counter += emit_n
                    enqueue_epoch = int(sess.stream_epoch)
                    sess.io_stats["queued_frames"] += float(emit_n)
                    if emit_n > 0:
                        sess.current_display_frame = chunk_frames[emit_n - 1].copy()
                        bidx = int(block_index)
                        sess.latest_decoded_block = max(sess.latest_decoded_block, bidx)
                        sess.decoded_block_last_frame[bidx] = chunk_frames[emit_n - 1].copy()
                        sess.decoded_block_last_frame_index[bidx] = int(start_idx + emit_n - 1)
                        min_keep = sess.latest_decoded_block - 64
                        stale = [k for k in sess.decoded_block_last_frame.keys() if k < min_keep]
                        for k in stale:
                            del sess.decoded_block_last_frame[k]
                        stale_idx = [k for k in sess.decoded_block_last_frame_index.keys() if k < min_keep]
                        for k in stale_idx:
                            del sess.decoded_block_last_frame_index[k]

                        if sess.pending_change_target_block is not None and bidx >= int(sess.pending_change_target_block):
                            target = int(sess.pending_change_target_block)
                            ref = sess.decoded_block_last_frame.get(target, chunk_frames[emit_n - 1].copy())
                            cut_frame_idx = int(sess.decoded_block_last_frame_index.get(target, start_idx + emit_n - 1))
                            sess.reference_frame = ref.copy()
                            sess.current_display_frame = ref.copy()
                            # Reuse frame index range after cutover so new frames replace
                            # old buffered future frames in the frontend timeline.
                            sess.frame_counter = cut_frame_idx
                            sess.restart_requested = True
                            sess.pause_requested = False
                            sess.stop_requested = False
                            sess.pending_change_target_block = None
                            sess.stream_epoch += 1
                            enqueue_epoch = int(sess.stream_epoch)
                            cutover_now = True
                            target_block = target
                            _drain_frame_queue(sess)
                    if emit_n < n:
                        sess.stop_requested = True
                if cutover_now:
                    SOCKET.emit(
                        "restart_cutover",
                        {
                            "cut_frame_index": int(cut_frame_idx),
                            "target_block_index": int(target_block),
                            "epoch": int(enqueue_epoch),
                        },
                        to=sid,
                    )
                    # Do not enqueue stale frames from old run after cutover.
                    return
                for i in range(emit_n):
                    frame = chunk_frames[i].copy()
                    sess.frame_queue.put((time.perf_counter(), frame, start_idx + i, int(block_index), int(enqueue_epoch)))
                cb_end = time.perf_counter()
                with sess.lock:
                    sess.io_stats["chunk_cb_ms"] += (cb_end - cb_start) * 1000.0

            def _should_stop() -> bool:
                with sess.lock:
                    return sess.stop_requested or sess.restart_requested

            def _should_pause(denoised_frame: Optional[int] = None) -> bool:
                # Called from the per-window hook on the GENERATION thread, so returning True
                # actually holds denoising. `denoised_frame` is where the window loop is -- the
                # frontier that decides whether a force change can still land.
                with sess.lock:
                    if denoised_frame is not None:
                        sess.denoise_frontier = int(denoised_frame)
                    if sess.pause_requested:
                        return True
                    return _pacing_should_hold(sess)

            def _get_condition_signal_update(current_start_latent_frame: int,
                                             full_start_latent: Optional[int] = None) -> Optional[Dict[str, Any]]:
                nonlocal applied_force_seq
                with sess.lock:
                    sess.update_calls += 1
                    seq = int(sess.force_change_seq)
                    sess.update_seq_seen = seq
                    sess.update_applied_seen = int(applied_force_seq)
                    if seq <= applied_force_seq:
                        return None
                    mode_snapshot = str(sess.mode)
                    payload_snapshot = str(sess.payload_text)
                video_start = _latent_to_video_frame_index(int(current_start_latent_frame))
                blk = max(1, int(backend.pipeline.num_frame_per_block))
                full_latent = (int(full_start_latent) if full_start_latent is not None
                               else int(current_start_latent_frame))
                full_video = _latent_to_video_frame_index(full_latent)
                signal_frame_start = 0
                signal_total_frames: Optional[int] = None
                if mode_snapshot == "wind":
                    # Wind control is time-constant; one frame is enough and will be broadcast.
                    signal_video_frames = 1
                else:
                    # Point force needs only the blob at frame 0. `_prepare_hint` freezes the mask
                    # there ("do not let the blob move") and channels 1-3 are constant in time, so
                    # the prepared hint is one spatial pattern for both modes.
                    #
                    # This used to rebuild the whole remainder of the clip -- up to ~490 frames,
                    # ~1 s of CPU and ~2.7 GB of GPU allocation on the generation thread, which is
                    # what made generation stutter on every point change. It then needed 2 frames
                    # rather than 1, because the trajectory builder computes `t = frame /
                    # (num_frames - 1)` and 1 divides by zero. The adapter now draws the single
                    # blob directly, so 1 is fine -- and `compress_time` subsampled both 1 and 2
                    # down to one latent anyway, so the conditioning is unchanged.
                    signal_video_frames = 1
                t0 = time.perf_counter()
                try:
                    converted, cond = _build_condition_from_payload(
                        backend,
                        mode_snapshot,
                        payload_snapshot,
                        num_video_frames=signal_video_frames,
                        frame_start_video=signal_frame_start,
                        total_video_frames=signal_total_frames,
                    )
                except Exception as exc:
                    # This used to tell only the browser, so a dropped force change left no trace
                    # in the log at all.
                    _log(f"[force] IGNORED id={seq} mode={mode_snapshot}: {exc!r}")
                    _emit_status(sid, f"Ignored force update {seq}: {exc}", "warning")
                    applied_force_seq = seq
                    return None
                build_ms = (time.perf_counter() - t0) * 1000.0
                applied_force_seq = seq
                with sess.lock:
                    sess.applied_force_seq = seq
                    sess.force_timeline.append(
                        {"video_first": int(video_start), "ui": dict(converted.get("ui") or {}),
                         "seq": int(seq), "mode": mode_snapshot}
                    )
                    # Use what the BROWSER says is on screen. Deriving it from
                    # `frame_counter - frame_queue.qsize()` assumed the app queue held the
                    # backlog; it does not -- engine.io and the TCP buffers do -- so that
                    # estimate reported the generator's position, not the viewer's, and made
                    # every change look like it landed 1 frame ahead.
                    shown = int(sess.client_displayed)
                    fps = float(sess.client_recv_fps)
                ahead = (video_start - shown) if shown >= 0 else -1
                behind_s = f", ~{ahead / fps:.0f}s of video away" if ahead > 0 and fps > 0.5 else ""
                ahead_full = (full_video - shown) if shown >= 0 else -1
                _log(
                    f"[force] APPLIED id={seq} mode={mode_snapshot} (build {build_ms:.0f} ms) | "
                    f"partial from block {int(current_start_latent_frame) // blk} "
                    f"(latent {int(current_start_latent_frame)}, video frame {video_start}) | "
                    f"FULL from block {full_latent // blk} "
                    f"(latent {full_latent}, video frame {full_video}) | "
                    f"viewer on frame {shown if shown >= 0 else '?'} -> "
                    f"{ahead if ahead >= 0 else '?'} ahead partial, "
                    f"{ahead_full if ahead_full >= 0 else '?'} ahead full{behind_s}"
                )
                _emit_force_change_completion_upto(sid, sess, applied_force_seq,
                                                   video_start_frame=int(video_start),
                                                   full_video_frame=int(full_video))
                _emit_status(
                    sid,
                    "Force update "
                    f"id={seq} prepared at latent={int(current_start_latent_frame)} "
                    f"(mode={mode_snapshot}, video={video_start}, signal_video_frames={signal_video_frames}) "
                    f"build={build_ms:.1f}ms",
                )
                return {
                    "seq": int(seq),
                    "mode": mode_snapshot,
                    "condition_signal": cond,
                    "video_start_frame": int(video_start),
                    "build_ms": float(build_ms),
                }

            try:
                result = backend.generate_segment_streaming(
                    reference_image=sess.reference_frame,
                    prompt=sess.prompt,
                    condition_signal=condition_signal,
                    seed=seed,
                    on_chunk=_on_chunk,
                    should_stop=_should_stop,
                    should_pause=_should_pause,
                    get_condition_signal_update=_get_condition_signal_update,
                    num_latent_frames=run_latents,
                )
                with sess.lock:
                    if result["frames"].shape[0] > 0:
                        sess.reference_frame = result["frames"][-1].copy()
                        sess.current_display_frame = sess.reference_frame.copy()
                    io_stats = dict(sess.io_stats)
                SOCKET.emit(
                    "perf",
                    {
                        "stream_metrics": result.get("stream_metrics", {}),
                        "io_stats": io_stats,
                        "runtime_update_logs": result.get("runtime_update_logs", []),
                    },
                    to=sid,
                )
                for log in result.get("runtime_update_logs", []):
                    _emit_status(
                        sid,
                        "Force update "
                        f"id={int(log.get('seq', -1))} "
                        f"applied@block={int(log.get('block_index', -1))} "
                        f"latent_start={int(log.get('latent_start_frame', -1))} "
                        f"video_start={int(log.get('video_start_frame', -1))} "
                        f"updated_latents={int(log.get('updated_latent_frames', -1))} "
                        f"build={float(log.get('build_ms', 0.0)):.1f}ms "
                        f"prepare={float(log.get('prepare_ms', 0.0)):.1f}ms "
                        f"total={float(log.get('total_ms', 0.0)):.1f}ms",
                    )
                _emit_status(sid, f"Generation finished (max {max_output_frames} frames)")
                break

            except GenerationStopped:
                with sess.lock:
                    if sess.restart_requested:
                        sess.restart_requested = False
                        sess.pending_change_target_block = None
                        sess.pause_requested = False
                        sess.stop_requested = True
                        _emit_status(sid, "Restart is disabled in fixed-length mode; start a new run instead")
                        break

                    if sess.stop_requested:
                        break

                    # Pause path: remain in same generation loop, waiting for resume.
                    _emit_status(sid, "Paused")
                    while True:
                        with sess.lock:
                            if sess.stop_requested:
                                break
                            if sess.restart_requested:
                                break
                            if not sess.pause_requested:
                                break
                        time.sleep(0.05)
                    continue

        _emit_status(sid, "Generation stopped")
    except Exception as exc:
        _emit_status(sid, f"Generation error: {exc}", "error")
        traceback.print_exc()
    finally:
        with sess.lock:
            sess.generating = False
            sess.sender_stop = True
            sess.worker = None


@APP.route("/")
def index():
    return render_template("index.html")


@SOCKET.on("connect")
def on_connect():
    sid = request.sid
    sess = _session(sid)
    sess.config_path = APP.config["DEMO_CONFIG_PATH"]
    sess.output_dir = APP.config["DEMO_OUTPUT_DIR"]
    sess.height = APP.config["DEMO_HEIGHT"]
    sess.width = APP.config["DEMO_WIDTH"]
    sess.seed = APP.config["DEMO_SEED"]
    sess.use_ema = APP.config["DEMO_USE_EMA"]
    sess.num_latent_frames = APP.config["DEMO_NUM_LATENT_FRAMES"]
    with _BACKEND_LOCK:
        sess.model_loaded = BACKEND_KEY in _BACKENDS and _BACKEND_LOAD_ERROR is None
    _log(f"[client] connected sid={sid[:8]} transport={_client_transport(sid)}")
    emit("status", {"message": "Connected", "kind": "info"})
    emit(
        "default_config",
        {
            "height": sess.height,
            "width": sess.width,
            "seed": sess.seed,
            "use_ema": sess.use_ema,
            "num_latent_frames": sess.num_latent_frames,
            "captioner_ready": _captioner_ready(),
            "captioner_model": APP.config.get("DEMO_CAPTION_MODEL", ""),
            # So the browser can raise `buffer_low` the moment it crosses this, instead of the
            # server waiting up to 200 ms for the next polled client_stats tick.
            "pace_min_buffer": int(APP.config.get("DEMO_PACE_MIN_BUFFER", 0)),
        },
    )
    with _BACKEND_LOCK:
        if _BACKEND_LOAD_ERROR:
            emit(
                "model_loaded",
                {"ok": False, "error": _BACKEND_LOAD_ERROR},
            )
        elif BACKEND_KEY not in _BACKENDS:
            # Still preloading. Without this branch the `else` below indexes _BACKENDS and
            # raises KeyError *inside the connect handler*, which makes the whole socket.io
            # connection fail -- so opening the page before the model finished loading left the
            # client unable to connect at all, with only a traceback on the server.
            emit("model_loaded", {"ok": False, "loading": True,
                                  "error": "models are still loading"})
        else:
            meta = dict(_BACKEND_METADATA.get(BACKEND_KEY, {}))
            emit(
                "model_loaded",
                {
                    "ok": True,
                    "metadata": {
                        "height": sess.height,
                        "width": sess.width,
                        # The backend may have clamped this to what the KV cache holds.
                        "num_latent_frames": int(meta.get("num_latent_frames", sess.num_latent_frames)),
                        "num_video_frames": int(meta.get("num_video_frames", 0)),
                        "rolling_window_blocks": int(len(_BACKENDS[BACKEND_KEY].pipeline.denoising_step_list)),
                        "device": str(meta.get("device", "n/a")),
                        "checkpoint_path": APP.config["DEMO_CHECKPOINT_PATH"],
                        "config_path": APP.config["DEMO_CONFIG_PATH"],
                        "use_ema": bool(APP.config.get("DEMO_USE_EMA", True)),
                        "num_frame_per_block": int(_BACKENDS[BACKEND_KEY].pipeline.num_frame_per_block),
                        "vram": _vram(str(meta.get("device", ""))),
                    },
                },
            )


# --------------------------------------------------------------------------------------------
# Transport self-test. Streams synthetic frames of the same size as real ones, with no model
# involved, so "can this connection carry the stream?" can be answered separately from "is the
# demo slow?". Open /selftest from wherever you are and read the numbers.
# --------------------------------------------------------------------------------------------

_SELFTEST_PAYLOAD: Dict[int, bytes] = {}
_SELFTEST_STOP: Dict[str, threading.Event] = {}
# What the real sender last put on the wire, so the test defaults to the real frame size
# instead of a number someone has to guess.
_LAST_WIRE_KB: float = 0.0


def _selftest_payload(kb: int) -> bytes:
    """A frame body of exactly `kb` kilobytes, built once and reused.

    BINARY, not a base64 data URL: the real sender emits
    `struct.pack("<III", ...) + jpeg_bytes` and this must match it, or the test measures a
    different transport than the demo uses. base64 alone would inflate every byte by 4/3 and
    would travel as an engine.io *text* packet rather than a WebSocket binary frame.

    Only the byte count matters -- the page counts bytes and never decodes -- so incompressible
    random bytes are used at the requested size.
    """
    if kb not in _SELFTEST_PAYLOAD:
        import os as _os

        _SELFTEST_PAYLOAD[kb] = _os.urandom(max(1024, int(kb) * 1024) - 12)
    return _SELFTEST_PAYLOAD[kb]


@SOCKET.on("selftest_stop")
def on_selftest_stop():
    """Stop emitting now. Frames already handed to the socket keep arriving -- that drain is
    itself the measurement of how much the buffers were holding."""
    ev = _SELFTEST_STOP.get(request.sid)
    if ev is not None:
        ev.set()
        _log(f"[selftest] sid={request.sid[:8]} stop requested")


@SOCKET.on("selftest_start")
def on_selftest_start(data):
    """Blast synthetic frames at a fixed rate until the duration elapses or Stop is pressed."""
    sid = request.sid
    fps = float((data or {}).get("fps", 12.0))
    seconds = min(float((data or {}).get("seconds", 10.0)), 300.0)
    kb = int((data or {}).get("kb", 20))
    body = _selftest_payload(kb)
    # Starting a new test abandons any test still running for this sid, so settings can be
    # switched without waiting. The old thread holds its own Event, so swapping the dict entry
    # cancels it cleanly.
    previous = _SELFTEST_STOP.get(sid)
    if previous is not None:
        previous.set()
    stop = threading.Event()
    _SELFTEST_STOP[sid] = stop
    token = int((data or {}).get("token", 0)) & 0xFFFFFFFF
    frame_bytes = len(body) + 12
    _log(f"[selftest] sid={sid[:8]} {fps:.0f} fps x {seconds:.0f}s @ {frame_bytes/1024:.0f} KB "
         f"= {fps * frame_bytes / 1024 / 1024:.2f} MB/s requested (binary, same framing as "
         f"frame_ready)")

    def _run() -> None:
        sent = 0
        t0 = time.time()
        next_at = t0
        stopped = False
        while time.time() - t0 < seconds:
            if stop.is_set():
                stopped = True
                break
            now = time.time()
            if now >= next_at:
                # Byte-for-byte the real sender's framing: 12-byte little-endian header
                # (frame index, block index, epoch) followed by the body.
                # The run token rides in the header's second field (the real stream's
                # block index), so the page can discard frames still draining from an
                # abandoned run instead of counting them against the new one.
                SOCKET.emit(
                    "selftest_frame",
                    struct.pack("<III", sent & 0xFFFFFFFF, token, 0) + body,
                    to=sid,
                )
                sent += 1
                next_at += 1.0 / max(fps, 0.1)
            else:
                time.sleep(0.002)
        elapsed = time.time() - t0
        SOCKET.emit("selftest_done", {"emitted": sent,
                                      "kb": frame_bytes / 1024.0,
                                      "seconds": elapsed,
                                      "stopped": stopped,
                                      "token": token}, to=sid)
        _log(f"[selftest] sid={sid[:8]} {'STOPPED' if stopped else 'finished'} after "
             f"{elapsed:.1f}s; emitted {sent} frames "
             f"({sent * frame_bytes / 1024 / 1024:.1f} MB); "
             f"eio queue still holding {_eio_queue_depth(sid)}")

    threading.Thread(target=_run, name="selftest", daemon=True).start()


def _selftest_context() -> Dict[str, Any]:
    """Template vars for the self-test page.

    Default operating point is fixed at 15 fps x 50 KB. The size the demo actually put on the
    wire is offered as an extra preset instead of overriding the default, so the starting point
    never moves between reloads.
    """
    measured = int(round(_LAST_WIRE_KB)) if _LAST_WIRE_KB > 0 else 0
    if measured > 0:
        preset = (f'<button class="st-mini" type="button" data-p="{measured},15">'
                  f'{measured} @ 15 (real frames)</button>')
    else:
        preset = '<span class="hint">(stream once for a preset at the real frame size)</span>'
    return {
        "default_fps": 15,
        "default_kb": 50,
        "measured_preset": preset,
        "reset_label": "Reset link",
        "reset_hint": ("<b>Reset link</b> drops and remakes the WebSocket, which discards buffered "
                       "data so the next measurement starts from an empty pipe."),
    }


@APP.route("/assets/<path:filename>")
def demo_assets(filename):
    """Gallery preset images and gallery.json, from `assets/` inside this demo folder.

    Read fresh on every request, so gallery.json can be edited without a restart.
    """
    from flask import send_from_directory

    return send_from_directory(str(DEMO_ROOT / "assets"), filename)


@APP.route("/selftest")
def selftest_page():
    return render_template("selftest.html", **_selftest_context())





@SOCKET.on("save_video")
def on_save_video(data):
    """Write the frames of the current/last run to an mp4 and hand back a download link.

    Saves what the model produced at full 480x832, not what went over the wire -- the
    wire_scale/jpeg_quality reduction happens later, in the sender, so it never touches this.
    """
    sid = request.sid
    sess = _session(sid)
    with sess.lock:
        if sess.saving:
            emit("status", {"message": "A save is already running", "kind": "warning"})
            return
        n = len(sess.saved_frames)
        if n == 0:
            emit("status", {"message": "Nothing to save yet - press Start first", "kind": "warning"})
            return
        sess.saving = True
        frames = list(sess.saved_frames)          # snapshot; generation may still be running
        timeline = [dict(e) for e in sess.force_timeline]
        control_text = str(sess.payload_text)
        prompt = str(sess.prompt)
        seed = int(sess.seed if sess.seed is not None else 0)
        still_generating = bool(sess.generating)

    fps = float(APP.config.get("DEMO_SAVE_FPS", 16.0))
    label = str((data or {}).get("label", "")).strip()
    arrows = bool((data or {}).get("arrows", False))

    def _work() -> None:
        import numpy as _np

        from interactive_demo_utils import write_video_uint8

        try:
            out_dir = APP.config["DEMO_OUTPUT_DIR"]
            stamp = time.strftime("%Y%m%d-%H%M%S")
            name = f"{stamp}{('-' + label) if label.isalnum() else ''}"
            folder = os.path.join(out_dir, name)
            os.makedirs(folder, exist_ok=True)
            filename = "video_arrows.mp4" if arrows else "video.mp4"
            path = os.path.join(folder, filename)
            t0 = time.perf_counter()
            to_write = frames
            if arrows:
                from arrow_overlay import burn_arrows

                t_burn = time.perf_counter()
                to_write = burn_arrows(frames, timeline)
                _log(f"[save] burned arrows into {len(to_write)} frames in "
                     f"{(time.perf_counter() - t_burn) * 1000:.0f} ms "
                     f"({len(timeline)} force segment(s))")
            write_video_uint8(path, _np.stack(to_write), fps=int(round(fps)))
            took = time.perf_counter() - t0
            size_mb = os.path.getsize(path) / 1024 / 1024
            with open(os.path.join(folder, f"meta{'_arrows' if arrows else ''}.json"), "w") as fh:
                json.dump(
                    {
                        "prompt": prompt,
                        "seed": seed,
                        "force_payload": control_text,
                        "arrows_burned_in": arrows,
                        # Every force the run used, and the first video frame it acted on.
                        "force_timeline": [
                            {"video_first": int(e.get("video_first", 0)),
                             "seq": int(e.get("seq", 0)),
                             "mode": e.get("mode"),
                             "ui": e.get("ui")}
                            for e in timeline
                        ],
                        "frames": len(frames),
                        "fps": fps,
                        "seconds": len(frames) / max(fps, 1e-6),
                        "partial": still_generating,
                        "checkpoint_path": APP.config.get("DEMO_CHECKPOINT_PATH"),
                        "config_path": APP.config.get("DEMO_CONFIG_PATH"),
                        "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                    fh,
                    indent=2,
                )
            _log(f"[save] {path} - {len(frames)} frames, {size_mb:.1f} MB, {took:.1f}s"
                 + (" (partial: still generating)" if still_generating else ""))
            SOCKET.emit(
                "save_ready",
                {
                    "url": f"/download/{name}/{filename}",
                    "path": path,
                    "frames": len(frames),
                    "seconds": len(frames) / max(fps, 1e-6),
                    "size_mb": size_mb,
                    "partial": still_generating,
                    "arrows": arrows,
                    "force_segments": len(timeline),
                },
                to=sid,
            )
        except Exception as exc:
            _log(f"[save] FAILED: {exc}")
            SOCKET.emit("status", {"message": f"Save failed: {exc}", "kind": "error"}, to=sid)
        finally:
            with sess.lock:
                sess.saving = False

    _log(f"[save] writing {n} frames at {fps:g} fps"
         + (f" with arrows ({len(timeline)} force segment(s))" if arrows else "")
         + (" (run still in progress)" if still_generating else ""))
    emit("status", {"message": f"Saving {n} frames…", "kind": "info"})
    threading.Thread(target=_work, name="save", daemon=True).start()


@APP.route("/download/<path:relpath>")
def download(relpath):
    """Serve a saved file as an attachment, so the browser downloads rather than plays it."""
    from flask import abort, send_from_directory

    root = os.path.abspath(APP.config["DEMO_OUTPUT_DIR"])
    target = os.path.abspath(os.path.join(root, relpath))
    # Keep the path inside the output directory.
    if not target.startswith(root + os.sep) or not os.path.isfile(target):
        abort(404)
    return send_from_directory(os.path.dirname(target), os.path.basename(target),
                               as_attachment=True)


@SOCKET.on("buffer_low")
def on_buffer_low(data):
    """The browser crossed below the pacing floor -- release any hold now, not in up to 200 ms.

    `client_stats` is polled at 200 ms, which was the largest single term left in the refill
    latency (against ~107 ms for one latent of generation and ~74 ms for its decode). This is the
    same information, edge-triggered: the client raises it once per dip below the floor.
    """
    sid = request.sid
    sess = _session(sid)
    with sess.lock:
        d = data or {}
        shown = int(d.get("displayed", -1))
        if shown > sess.client_displayed:
            sess.client_displayed = shown
        sess.client_buffered = int(d.get("buffered", -1))
        sess.client_reported_at = time.time()
        if sess.pacing_holding:
            sess.pacing_holding = False
            sess.pacing_held_s += max(0.0, time.time() - sess.pacing_hold_started)
            _log(f"[pace] RESUME (client signalled buffer {sess.client_buffered} below floor; "
                 f"shown={sess.client_displayed})")


@SOCKET.on("client_stats")
def on_client_stats(data):
    """The browser reporting what it has actually received. See `_eio_queue_depth`."""
    sid = request.sid
    sess = _session(sid)
    with sess.lock:
        sess.client_received = int((data or {}).get("received", 0))
        sess.client_recv_fps = float((data or {}).get("recv_fps", 0.0))
        sess.client_draw_fps = float((data or {}).get("draw_fps", 0.0))
        sess.client_displayed = int((data or {}).get("displayed", -1))
        sess.client_buffered = int((data or {}).get("buffered", -1))
        sess.client_reported_at = time.time()


@SOCKET.on("disconnect")
def on_disconnect():
    sid = request.sid
    _log(f"[client] disconnected sid={sid[:8]}")
    sess = _session(sid)
    with sess.lock:
        sess.stop_requested = True
        sess.pause_requested = False
        sess.restart_requested = False
        sess.sender_stop = True


@SOCKET.on("set_input")
def on_set_input(data):
    sid = request.sid
    sess = _session(sid)
    new_image_data_url = ""
    with sess.lock:
        image_updated = False
        requested_stop_for_new_image = False
        if "prompt" in data:
            sess.prompt = str(data.get("prompt") or "")
        if "mode" in data:
            sess.mode = str(data.get("mode") or "point")
        if "payload_text" in data:
            sess.payload_text = str(data.get("payload_text") or "")
        if data.get("reference_image_data"):
            try:
                sess.reference_frame = _decode_data_url_image(data["reference_image_data"])
                sess.current_display_frame = sess.reference_frame.copy()
                image_updated = True
                new_image_data_url = str(data["reference_image_data"])
                if sess.generating:
                    sess.stop_requested = True
                    sess.sender_stop = True
                    _clear_generation_state(sess)
                    requested_stop_for_new_image = True
            except Exception as exc:
                emit("status", {"message": f"Invalid uploaded image payload: {exc}", "kind": "error"})
                return
    # Outside the lock: captioning spawns a thread and emits on its own.
    if image_updated and _captioner_ready():
        _caption_async(sid, new_image_data_url)
    if requested_stop_for_new_image:
        emit("status", {"message": "New image selected: stopped current generation and cleared state. Press Start to restart.", "kind": "info"})
    elif image_updated:
        emit("status", {"message": "Input updated (new reference image set)", "kind": "info"})
    else:
        emit("status", {"message": "Input updated", "kind": "info"})


@SOCKET.on("set_reference_frame")
def on_set_reference_frame(data):
    sid = request.sid
    sess = _session(sid)
    ref = data.get("reference_image_data")
    if not ref:
        emit("status", {"message": "Empty reference frame payload", "kind": "warning"})
        return
    try:
        img = _decode_data_url_image(ref)
        with sess.lock:
            sess.reference_frame = img
            sess.current_display_frame = img.copy()
        emit("status", {"message": "Updated reference frame from paused display frame", "kind": "info"})
    except Exception as exc:
        emit("status", {"message": f"Failed to set reference frame: {exc}", "kind": "error"})


@SOCKET.on("start")
def on_start(data):
    sid = request.sid
    sess = _session(sid)

    with sess.lock:
        if sess.generating:
            emit("status", {"message": "Already generating", "kind": "warning"})
            return

        if data.get("prompt"):
            sess.prompt = str(data.get("prompt") or "")
        if data.get("mode"):
            sess.mode = str(data.get("mode") or "point")
        if data.get("payload_text"):
            sess.payload_text = str(data.get("payload_text") or "")
        # Gallery presets cap their own clip; an upload sends nothing and gets the full length.
        want_latents = _parse_optional_int(data.get("latents"))
        cap = int(APP.config.get("DEMO_NUM_LATENT_FRAMES", MAX_SEGMENT_LATENT_FRAMES))
        if want_latents is not None and want_latents > 0:
            sess.run_latents = max(1, min(int(want_latents), cap))
        else:
            sess.run_latents = None
        start_seed = _parse_optional_int(data.get("seed"))
        if start_seed is not None:
            sess.seed = int(start_seed)
        if data.get("reference_image_data"):
            try:
                sess.reference_frame = _decode_data_url_image(data["reference_image_data"])
                sess.current_display_frame = sess.reference_frame.copy()
            except Exception as exc:
                emit("status", {"message": f"Invalid uploaded image payload: {exc}", "kind": "error"})
                return

        with _BACKEND_LOCK:
            backends_ready = (BACKEND_KEY in _BACKENDS and _BACKEND_LOAD_ERROR is None)
        sess.model_loaded = backends_ready
        if not backends_ready:
            emit("status", {"message": f"Models are not ready: {_BACKEND_LOAD_ERROR or 'unknown load state'}", "kind": "error"})
            return
        if sess.reference_frame is None:
            emit("status", {"message": "Upload image first", "kind": "error"})
            return
        if not sess.payload_text:
            emit("status", {"message": "Set force input first", "kind": "error"})
            return

        sess.stop_requested = False
        sess.pause_requested = False
        sess.restart_requested = False
        sess.pending_change_target_block = None
        sess.latest_decoded_block = -1
        sess.decoded_block_last_frame.clear()
        sess.decoded_block_last_frame_index.clear()
        sess.pending_force_change_debug.clear()
        sess.stream_epoch += 1
        sess.sender_stop = False
        sess.inflight_warned = False
        sess.pacing_holding = False
        sess.pacing_holds = 0
        sess.pacing_held_s = 0.0
        sess.update_calls = 0
        sess.update_seq_seen = -1
        sess.update_applied_seen = -1
        sess.denoise_frontier = -1
        sess.client_received = 0        # per-run, like io_stats; the browser resets its copy too
        sess.client_displayed = -1
        sess.client_buffered = -1
        sess.saved_frames = []          # a new run replaces what was available to save
        sess.force_timeline = []        # re-seeded by the worker once the signal is built
        sess.generating = True
        sess.frame_counter = 0
        _reset_io_stats(sess)
        _start_sender_if_needed(sess)
        sess.worker = threading.Thread(target=_generation_worker, args=(sid,), daemon=True)
        sess.worker.start()


@SOCKET.on("pause")
def on_pause():
    sid = request.sid
    sess = _session(sid)
    with sess.lock:
        sess.pause_requested = True
    emit("status", {"message": "Pause requested", "kind": "info"})


@SOCKET.on("stop")
def on_stop():
    sid = request.sid
    sess = _session(sid)
    with sess.lock:
        sess.stop_requested = True
        sess.pause_requested = False
        sess.restart_requested = False
    emit("status", {"message": "Stop requested", "kind": "info"})


@SOCKET.on("resume")
def on_resume(data):
    sid = request.sid
    sess = _session(sid)
    with sess.lock:
        if data.get("mode"):
            sess.mode = str(data.get("mode") or sess.mode)
        if data.get("payload_text"):
            sess.payload_text = str(data.get("payload_text") or sess.payload_text)
        sess.pause_requested = False
    emit("status", {"message": "Resumed", "kind": "info"})


@SOCKET.on("restart")
def on_restart(data):
    sid = request.sid
    sess = _session(sid)
    with sess.lock:
        if data.get("mode"):
            sess.mode = str(data.get("mode") or sess.mode)
        if data.get("payload_text"):
            sess.payload_text = str(data.get("payload_text") or sess.payload_text)
        sess.restart_requested = True
        # force current streaming call to break at next decoded chunk
        sess.stop_requested = False
    emit("status", {"message": "Restart requested", "kind": "info"})


@SOCKET.on("change_force")
def on_change_force(data):
    sid = request.sid
    sess = _session(sid)
    immediate_debug = None
    with sess.lock:
        mode = sess.mode
        if data.get("mode"):
            sess.mode = str(data.get("mode") or sess.mode)
            mode = sess.mode
        payload_text = sess.payload_text
        if data.get("payload_text"):
            sess.payload_text = str(data.get("payload_text") or sess.payload_text)
            payload_text = sess.payload_text

        parsed_payload = None
        try:
            parsed_payload = _safe_json_parse(payload_text)
        except Exception:
            parsed_payload = None

        sess.force_change_seq += 1
        seq = int(sess.force_change_seq)
        latest_block = int(sess.latest_decoded_block)
        request_last_frame_index = int(sess.decoded_block_last_frame_index.get(latest_block, sess.frame_counter - 1))
        record = {
            "seq": seq,
            "event": "requested",
            "requested_at_ms": int(time.time() * 1000),
            "mode": mode,
            "force_payload": parsed_payload,
            "force_payload_text": payload_text,
            "requested_displayed_frame_index": int(data.get("displayed_frame_index", -1)),
            "requested_displayed_block_index": int(data.get("displayed_block_index", -1)),
            "requested_latest_decoded_block": latest_block,
            "requested_last_frame_index": request_last_frame_index,
            "generating_when_requested": bool(sess.generating),
        }
        if sess.generating:
            sess.pending_force_change_debug.append(record)
            immediate_debug = dict(record)
        else:
            immediate_debug = dict(record)
            immediate_debug["event"] = "completed"
            immediate_debug["completed_at_ms"] = int(time.time() * 1000)
            immediate_debug["completed_last_frame_index"] = request_last_frame_index
            immediate_debug["completed_latest_decoded_block"] = latest_block
            req_ms = immediate_debug.get("requested_at_ms")
            if isinstance(req_ms, int):
                immediate_debug["latency_ms"] = int(max(0, immediate_debug["completed_at_ms"] - req_ms))

    # The request carries the browser's displayed index at the instant of the drag, which is
    # fresher than the periodic client_stats report. Pacing and the "lands N frames ahead" figure
    # both read sess.client_displayed, so adopt it here rather than waiting for the next tick.
    _req_shown = int(data.get("displayed_frame_index", -1))
    if _req_shown >= 0:
        with sess.lock:
            if _req_shown > sess.client_displayed:
                sess.client_displayed = _req_shown
                sess.client_reported_at = time.time()
    _log(
        f"[force] requested id={seq} mode={mode} "
        f"watching frame={int(data.get('displayed_frame_index', -1))} "
        f"(generator applied up to id={int(sess.applied_force_seq)}) "
        f"block={int(data.get('displayed_block_index', -1))} "
        f"generating={bool(immediate_debug and immediate_debug.get('generating_when_requested'))}"
    )
    emit(
        "status",
        {
            "message": f"Force signal updated (id={seq}; applying during current generation without restart)",
            "kind": "info",
        },
    )
    if immediate_debug is not None:
        emit("force_change_debug", immediate_debug)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5006)
    parser.add_argument("--config_path", default="configs/dmd_everything.yaml")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the single generator checkpoint, used for BOTH point and wind modes. "
             "Relative paths are resolved against the repo root.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device for the model, e.g. cuda:0 (default: cuda if available, else cpu).",
    )
    parser.add_argument(
        "--vae_device",
        default="auto",
        help="GPU for the VAE decoder. 'auto' (default) = cuda:1 when two or more GPUs are "
             "visible, else the generator's GPU. 'same' pins it beside the generator. Decode is "
             "the larger half of a block, so giving it its own GPU stops it competing for SMs; "
             "the latents handed across are ~0.45 MB. See OPTIMIZATIONS.md.",
    )
    parser.add_argument(
        "--vae_channels_last", dest="vae_channels_last", action="store_true", default=True,
        help="ON BY DEFAULT. Run the VAE's Conv3d in channels_last_3d so cuDNN handles them "
             "instead of the aten::slow_conv_dilated3d im2col fallback (809 -> 225 ms per "
             "3-latent decode). Not bit-identical: ~5/255 max on the decode.",
    )
    parser.add_argument(
        "--no_vae_channels_last", dest="vae_channels_last", action="store_false",
        help="Disable the VAE channels_last_3d fix (bit-identical decode, ~3.6x slower).",
    )
    parser.add_argument(
        "--gen_channels_last", dest="gen_channels_last", action="store_true", default=True,
        help="ON BY DEFAULT. Same channels_last_3d fix for the generator's two patch_embedding "
             "Conv3d, which hit the same im2col fallback.",
    )
    parser.add_argument(
        "--no_gen_channels_last", dest="gen_channels_last", action="store_false",
        help="Disable the generator channels_last_3d fix.",
    )
    parser.add_argument("--output_dir", default="demo/demo_outputs")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_latent_frames", type=int, default=126)
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument(
        "--save_fps",
        type=float,
        default=16.0,
        help="Frame rate written into saved mp4s (16 matches the training rate).",
    )
    parser.add_argument(
        "--max_save_frames",
        type=int,
        default=1200,
        help="Cap on frames retained for saving. Each is a full-resolution uint8 RGB frame "
             "(~1.2 MB at 480x832), so 1200 is about 1.4 GB.",
    )
    parser.add_argument(
        "--rolling_forcing_block_frames",
        type=int,
        default=3,
        help="Latent frames per block, and the size of the attention sink that is pinned at the "
             "front of the KV cache. RollingForcing's default is 3.",
    )
    parser.add_argument(
        "--rolling_forcing_max_frames",
        type=int,
        default=21,
        help="Latent frames the attention window reads back over (max_attention_size = this x "
             "frame_seq_length). RollingForcing's default is 21.",
    )
    parser.add_argument(
        "--rolling_forcing_cache_frames",
        type=int,
        default=96,
        help="Passed through to the pipeline as `rolling_forcing_cache_frames`. Note the pipeline "
             "currently hardcodes its cache at 24 latent frames (sink 3 + window 21), so this is "
             "vestigial unless that changes.",
    )
    parser.add_argument(
        "--pace_high_frames",
        type=int,
        default=48,
        help="Generation pacing: hold generation while it is more than this many video frames "
             "ahead of the frame the browser reports showing. 0 disables pacing entirely "
             "(original behaviour). Default 48 = 4 blocks = one rolling window.",
    )
    parser.add_argument(
        "--pace_min_buffer",
        type=int,
        default=6,
        help="Frames the browser must still have queued before generation is allowed to hold. "
             "The paced gap is measured on the DENOISE frontier, which leads the emitted one by "
             "the rolling window's depth (~36 frames), so without this floor the watermark can "
             "starve the viewer. Slightly more than one 12-frame block. 0 disables the floor.",
    )
    parser.add_argument(
        "--pace_buffer_margin",
        type=int,
        default=-1,
        help="Extra frames above --pace_min_buffer required before a hold may START. Gives the "
             "floor hysteresis: entering at exactly the floor leaves one frame before the "
             "release fires. -1 (default) means use --pace_min_buffer, i.e. hold from 2x the "
             "floor down to the floor.",
    )
    parser.add_argument(
        "--pace_low_frames",
        type=int,
        default=24,
        help="Resume generation once the gap falls to this. Must be < --pace_high_frames; the "
             "gap between them is the hysteresis that stops the generator chattering.",
    )
    parser.add_argument(
        "--jpeg_quality",
        type=int,
        default=85,
        help="Wire JPEG quality. A 480x832 frame is ~74 KB as a base64 data URL at q85 and "
             "~41 KB at q60, i.e. 1.16 vs 0.64 MB/s at 16 fps.",
    )
    parser.add_argument(
        "--wire_scale",
        type=float,
        default=1.0,
        help="Downscale applied only to frames sent to the browser (1.0 = off). Generation "
             "resolution is unchanged; 0.75 roughly halves the bytes.",
    )
    parser.add_argument(
        "--caption_model",
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="HuggingFace model ID or local path for the VLM captioner, used to fill the prompt "
             "box automatically when an image is uploaded. Pass an empty string to disable.",
    )
    parser.add_argument(
        "--caption_device",
        default="auto",
        help="device_map for the captioner (default: auto). It is loaded before the two "
             "diffusion models, so 'auto' will spread it across whatever is free.",
    )
    args = parser.parse_args()

    APP.config["DEMO_CONFIG_PATH"] = args.config_path
    APP.config["DEMO_CHECKPOINT_PATH"] = args.checkpoint
    APP.config["DEMO_SAVE_FPS"] = float(args.save_fps)
    APP.config["DEMO_MAX_SAVE_FRAMES"] = int(args.max_save_frames)
    APP.config["DEMO_JPEG_QUALITY"] = int(args.jpeg_quality)
    APP.config["DEMO_WIRE_SCALE"] = float(args.wire_scale)
    print(f"[v6] wire: jpeg_quality={args.jpeg_quality} wire_scale={args.wire_scale} "
          f"frame dropping=never (browser falls behind instead)")
    APP.config["DEMO_RF_BLOCK_FRAMES"] = args.rolling_forcing_block_frames
    APP.config["DEMO_RF_MAX_FRAMES"] = args.rolling_forcing_max_frames
    APP.config["DEMO_RF_CACHE_FRAMES"] = args.rolling_forcing_cache_frames
    APP.config["DEMO_PACE_HIGH"] = int(args.pace_high_frames)
    APP.config["DEMO_PACE_LOW"] = int(min(args.pace_low_frames, max(0, args.pace_high_frames - 1)))
    APP.config["DEMO_PACE_MIN_BUFFER"] = int(max(0, args.pace_min_buffer))
    APP.config["DEMO_PACE_BUFFER_MARGIN"] = int(max(0, args.pace_buffer_margin)) \
        if args.pace_buffer_margin >= 0 else int(max(0, args.pace_min_buffer))
    APP.config["DEMO_OUTPUT_DIR"] = args.output_dir
    APP.config["DEMO_HEIGHT"] = args.height
    APP.config["DEMO_WIDTH"] = args.width
    APP.config["DEMO_SEED"] = args.seed
    APP.config["DEMO_NUM_LATENT_FRAMES"] = min(int(args.num_latent_frames), MAX_SEGMENT_LATENT_FRAMES)
    APP.config["DEMO_USE_EMA"] = not args.no_ema

    print(f"[v6] config     {args.config_path}")
    print(f"[v6] checkpoint {args.checkpoint}  -> {args.device or 'cuda'}")
    APP.config["DEMO_CAPTION_MODEL"] = args.caption_model

    # The captioner is loaded in the background so it is usually ready by the time the first
    # image is uploaded -- but NOT concurrently with the diffusion preload below. The captioner
    # goes through accelerate's device_map="auto", whose init_empty_weights() context manager
    # patches parameter allocation process-wide (it is not thread-local): a diffusion model built
    # in another thread while it is active gets meta-device parameters and dies on .to(device)
    # with "Cannot copy out of meta tensor". So the thread is started only after preload finishes.
    def _start_captioner() -> None:
        if not args.caption_model:
            print("[v6] auto-captioning disabled (--caption_model \"\")")
            return
        global CAPTIONER
        CAPTIONER = ImageCaptioner(model_path=args.caption_model, device=args.caption_device)

        def _load_captioner() -> None:
            try:
                CAPTIONER.load()
            except Exception:
                import traceback

                traceback.print_exc()

        threading.Thread(target=_load_captioner, name="load-captioner", daemon=True).start()

    print(f"[rf] rolling forcing: sink/block {args.rolling_forcing_block_frames} latents, "
          f"window {args.rolling_forcing_max_frames} latents")
    if args.pace_high_frames > 0:
        print(f"[pace] viewer-buffer floor {APP.config['DEMO_PACE_MIN_BUFFER']} frames "
              f"(~{APP.config['DEMO_PACE_MIN_BUFFER'] / 16.0:.1f}s): never hold below it")
        print(f"[pace] generation pacing ON: hold above {args.pace_high_frames} frames ahead of "
              f"the viewer, resume at {APP.config['DEMO_PACE_LOW']} "
              f"(~{args.pace_high_frames / 16.0:.1f}s / {APP.config['DEMO_PACE_LOW'] / 16.0:.1f}s "
              f"of video at 16 fps)")
    else:
        print("[pace] generation pacing OFF -- the generator runs as far ahead as it can")
    print("[v6] Preloading the model (one checkpoint, shared by point and wind)...")
    try:
        _preload_backend(
            config_path=args.config_path,
            checkpoint_path=args.checkpoint,
            device=args.device,
            vae_device=_resolve_vae_device(args.vae_device, args.device),
            vae_channels_last=bool(args.vae_channels_last),
            gen_channels_last=bool(args.gen_channels_last),
            output_dir=args.output_dir,
            height=args.height,
            width=args.width,
            use_ema=not args.no_ema,
            num_latent_frames=min(int(args.num_latent_frames), MAX_SEGMENT_LATENT_FRAMES),
        )
        print("[v6] Model preload complete")
    except Exception as exc:
        print(f"[v6] Model preload failed: {exc}")

    _start_captioner()

    # async_mode is "threading", so flask-socketio serves through Werkzeug and refuses to
    # start without this opt-in. This is a single-user local demo, not a production server.
    SOCKET.run(APP, host=args.host, port=args.port, debug=False, allow_unsafe_werkzeug=True)


if __name__ == "__main__":
    main()
