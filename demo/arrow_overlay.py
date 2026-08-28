"""Burn the force arrow into saved frames, drawn exactly as the browser draws it.

The browser's force payload is in canvas pixels and the canvas is the model's own
resolution (832x480), so the arrow can be replayed onto the decoded frames with no
coordinate conversion at all -- every constant below is copied from `draw()` in
flask_static/app.js, which is itself pinned to v6's rendering.

Frames are RGB uint8 (that is how the generator hands them over; the wire path converts
to BGR later, in the sender), so the colours here are RGB and cv2 just sees channels.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np

# ---- copied from flask_static/app.js --------------------------------------------------
MAX_LEN = 80.0
ARROW_RGB = (255, 225, 77)        # #ffe14d
# The arrow is rasterised at SS times the frame's resolution and area-downscaled into a coverage
# mask, then composited. Drawing straight onto the frame with cv2.LINE_AA gave a much harder edge
# (~60 intermediate levels vs ~150) and snapped the endpoints to whole pixels, which reads as
# jagged -- especially once a player upscales an 832x480 file. Only the arrow's bounding box is
# supersampled, so the cost is a few ms per frame.
SS = 4
WIND_STRAND_OFFSETS = (-14.0, 0.0, 14.0)
HEAD_LEN = 12.0
HEAD_SPREAD = math.pi / 7.0


def _clamped_vector(ui: Dict[str, Any]) -> tuple[float, float]:
    """Prefer the already-clamped vector the adapter produced; fall back to clamping."""
    vec = ui.get("vector_clamped") or ui.get("vector_raw") or {}
    dx = float(vec.get("dx", 0.0))
    dy = float(vec.get("dy", 0.0))
    length = math.hypot(dx, dy)
    if length > MAX_LEN and length > 1e-6:
        s = MAX_LEN / length
        dx, dy = dx * s, dy * s
    return dx, dy


def _head(mask: np.ndarray, tx: float, ty: float, ang: float, scale: float, s: float) -> None:
    h = HEAD_LEN * scale * s
    pts = np.array(
        [
            [tx, ty],
            [tx - h * math.cos(ang - HEAD_SPREAD), ty - h * math.sin(ang - HEAD_SPREAD)],
            [tx - h * math.cos(ang + HEAD_SPREAD), ty - h * math.sin(ang + HEAD_SPREAD)],
        ],
        dtype=np.float32,
    )
    cv2.fillConvexPoly(mask, np.round(pts).astype(np.int32), 255, lineType=cv2.LINE_AA)


def _bezier(p0, c1, c2, p3, steps: int = 24) -> np.ndarray:
    t = np.linspace(0.0, 1.0, steps, dtype=np.float32)[:, None]
    mt = 1.0 - t
    pts = (
        (mt ** 3) * np.asarray(p0, dtype=np.float32)
        + 3 * (mt ** 2) * t * np.asarray(c1, dtype=np.float32)
        + 3 * mt * (t ** 2) * np.asarray(c2, dtype=np.float32)
        + (t ** 3) * np.asarray(p3, dtype=np.float32)
    )
    return np.round(pts).astype(np.int32)


def draw_force_arrow(frame: np.ndarray, ui: Dict[str, Any]) -> np.ndarray:
    """Draw one force arrow onto `frame` in place. `ui` is the payload's `ui` dict."""
    if not ui:
        return frame
    h_img, w_img = frame.shape[:2]
    canvas_w = max(1.0, float(ui.get("canvas_width") or w_img))
    canvas_h = max(1.0, float(ui.get("canvas_height") or h_img))
    # One factor: the canvas and the frame share an aspect ratio in this app.
    s = min(w_img / canvas_w, h_img / canvas_h)

    anchor = ui.get("anchor") or {}
    ax = float(anchor.get("x", 0.0)) * s
    ay = float(anchor.get("y", 0.0)) * s
    dx, dy = _clamped_vector(ui)
    dx, dy = dx * s, dy * s
    ex, ey = ax + dx, ay + dy
    length = math.hypot(dx, dy)
    mode = str(ui.get("mode", "point"))
    line_w = max(1, int(round(2.8 * s * SS)))

    # Only the arrow is drawn. The blob-footprint ring and the white anchor handle that point
    # mode used to get are gone from the canvas too, so nothing is drawn here either -- and a
    # zero-magnitude force (an unplaced one) leaves the frame untouched.
    if length < 1e-4:
        return frame
    ux, uy = dx / length, dy / length
    nx, ny = -uy, ux
    ang = math.atan2(dy, dx)

    # Bounding box of everything about to be drawn, padded for stroke width and arrowhead.
    xs = [ax, ex]
    ys = [ay, ey]
    if mode == "wind":
        for off_raw in WIND_STRAND_OFFSETS:
            off = off_raw * s
            xs += [ax + nx * off, ex + nx * off]
            ys += [ay + ny * off, ey + ny * off]
    pad = 2.8 * s + HEAD_LEN * s + 3.0
    x0 = max(0, int(math.floor(min(xs) - pad)))
    y0 = max(0, int(math.floor(min(ys) - pad)))
    x1 = min(w_img, int(math.ceil(max(xs) + pad)))
    y1 = min(h_img, int(math.ceil(max(ys) + pad)))
    if x1 <= x0 or y1 <= y0:
        return frame

    # Rasterise at SS resolution, in box-local coordinates so the mask stays small.
    mask = np.zeros(((y1 - y0) * SS, (x1 - x0) * SS), np.uint8)
    def T(px: float, py: float) -> tuple[float, float]:
        return ((px - x0) * SS, (py - y0) * SS)

    if mode == "wind":
        for off_raw in WIND_STRAND_OFFSETS:
            off = off_raw * s
            sx, sy = ax + nx * off, ay + ny * off
            tx, ty = ex + nx * off, ey + ny * off
            c1 = (sx + ux * (length * 0.34) + nx * (off * -0.15 + 3 * s),
                  sy + uy * (length * 0.34) + ny * (off * -0.15 + 3 * s))
            c2 = (sx + ux * (length * 0.70) + nx * (off * 0.10 - 3 * s),
                  sy + uy * (length * 0.70) + ny * (off * 0.10 - 3 * s))
            cv2.polylines(mask, [_bezier(T(sx, sy), T(*c1), T(*c2), T(tx, ty), steps=48)],
                          False, 255, line_w, cv2.LINE_AA)
            hx, hy = T(tx, ty)
            _head(mask, hx, hy, ang, 0.62, s * SS)
    else:
        sx, sy = T(ax, ay)
        tx, ty = T(ex, ey)
        cv2.line(mask, (int(round(sx)), int(round(sy))), (int(round(tx)), int(round(ty))),
                 255, line_w, cv2.LINE_AA)
        _head(mask, tx, ty, ang, 1.0, s * SS)

    # Area-downscale the mask to get per-pixel coverage, then alpha-composite the arrow colour.
    alpha = cv2.resize(mask, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
    a = (alpha.astype(np.float32) / 255.0)[..., None]
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    colour = np.array(ARROW_RGB, np.float32)
    frame[y0:y1, x0:x1] = np.clip(roi * (1.0 - a) + colour * a, 0, 255).astype(np.uint8)
    return frame


def burn_arrows(
    frames: Sequence[np.ndarray],
    timeline: List[Dict[str, Any]],
    frame_offset: int = 0,
) -> List[np.ndarray]:
    """Copy `frames` with each one's own force arrow burned in.

    `timeline` is [{"video_first": int, "ui": {...}}, ...] in application order: entry i
    governs every frame from its `video_first` until the next entry starts. That is what
    makes a mid-run force change visible in the file -- the arrow turns on the same frame
    the model first felt it.
    """
    out: List[np.ndarray] = []
    ordered = sorted(
        [e for e in (timeline or []) if e.get("ui")],
        key=lambda e: int(e.get("video_first", 0)),
    )
    cursor = 0
    current: Optional[Dict[str, Any]] = None
    for i, frame in enumerate(frames):
        idx = frame_offset + i
        while cursor < len(ordered) and int(ordered[cursor].get("video_first", 0)) <= idx:
            current = ordered[cursor]
            cursor += 1
        copy = np.ascontiguousarray(frame.copy())
        if current is not None:
            draw_force_arrow(copy, current["ui"])
        out.append(copy)
    return out
