from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch

from utils.forceprompt_data.controlnet_datasets import (
    load_controlnet_signal_wind_force,
    load_point_force_mask,
)

# NOTE on blob scale. The training loaders draw the mask on a fixed 480x720 canvas and then
# resize_for_crop() it up to the target, which magnifies it. This file used to build straight
# at (height, width) and skip that magnification: at 480x832 it produced a mask covering 1.44%
# of the frame where training used 1.93% (effective radius 42.8 px vs 49.6 px), i.e. ~34% too
# small. Since the mask gates both how much of the force field reaches the model and how much
# of the first frame becomes `masked_latent`, the demo was conditioning point forces on a
# tighter region than any checkpoint was trained on. `load_point_force_mask` now scales the
# radius by max(H/480, W/720) so drawing at the target resolution matches -- which keeps the
# demo's position semantics ("where you click is where the blob lands") intact.


@dataclass
class ForceAdapterConfig:
    min_force: float = 0.0
    max_force: float = 1.0
    max_point_force_len: float = 220.0
    max_wind_force_len: float = 160.0
    unified_prompt: bool = True


def _clamp_vector(dx: float, dy: float, max_len: float) -> Tuple[float, float, float, float]:
    raw_len = math.sqrt(dx * dx + dy * dy)
    if raw_len < 1e-8:
        return 0.0, 0.0, 0.0, 0.0
    scale = min(1.0, max_len / raw_len)
    cdx, cdy = dx * scale, dy * scale
    return cdx, cdy, raw_len, math.sqrt(cdx * cdx + cdy * cdy)


def _angle_from_canvas_vector(dx: float, dy: float) -> float:
    # Model convention matches dataset utility: angle=0 points right, +CCW, y-axis inverted.
    return float((math.degrees(math.atan2(-dy, dx)) + 360.0) % 360.0)


def _normalized_force(clamped_len: float, max_len: float, min_force: float, max_force: float) -> float:
    ratio = 0.0 if max_len <= 0 else max(0.0, min(1.0, clamped_len / max_len))
    return float(min_force + ratio * (max_force - min_force))


def parse_ui_force_payload(
    payload: Dict[str, Any],
    cfg: ForceAdapterConfig,
) -> Dict[str, Any]:
    mode = payload.get("mode", "point")
    anchor = payload.get("anchor", {})
    vector_raw = payload.get("vector_raw", {})

    ax = float(anchor.get("x", 0.0))
    ay = float(anchor.get("y", 0.0))
    dx = float(vector_raw.get("dx", 0.0))
    dy = float(vector_raw.get("dy", 0.0))

    max_len = cfg.max_point_force_len if mode == "point" else cfg.max_wind_force_len
    cdx, cdy, raw_len, clamped_len = _clamp_vector(dx, dy, max_len)

    angle = _angle_from_canvas_vector(cdx, cdy) if clamped_len > 0 else 0.0
    force = _normalized_force(clamped_len, max_len, cfg.min_force, cfg.max_force)

    canvas_w = int(payload.get("canvas_width", 1))
    canvas_h = int(payload.get("canvas_height", 1))
    x_pos = max(0.0, min(1.0, ax / max(1, canvas_w)))
    y_pos = max(0.0, min(1.0, 1.0 - (ay / max(1, canvas_h))))

    model = {
        "force": force,
        "angle": angle,
        "x_pos": x_pos,
        "y_pos": y_pos,
        "mode": mode,
    }
    ui = {
        "mode": mode,
        "canvas_width": canvas_w,
        "canvas_height": canvas_h,
        "anchor": {"x": ax, "y": ay},
        "vector_raw": {"dx": dx, "dy": dy, "length": raw_len},
        "vector_clamped": {"dx": cdx, "dy": cdy, "length": clamped_len, "max": max_len},
    }
    return {"ui": ui, "model": model}


def build_model_condition_signal(
    force_payload: Dict[str, Any],
    height: int,
    width: int,
    num_video_frames: int,
    cfg: ForceAdapterConfig,
    frame_start: int = 0,
    total_video_frames: int | None = None,
) -> torch.Tensor:
    mode = force_payload["model"]["mode"]
    force = force_payload["model"]["force"]
    angle = force_payload["model"]["angle"]

    wind_signal = load_controlnet_signal_wind_force(
        force=force,
        angle=angle,
        num_frames=num_video_frames,
        height=height,
        width=width,
        min_force=cfg.min_force,
        max_force=cfg.max_force,
    )

    if int(frame_start) != 0:
        # `frame_start` would only mean something if the blob still travelled; it is static,
        # and app.py hardcodes 0. Be loud if that ever changes.
        raise ValueError(
            f"frame_start={frame_start} requested, but the point-force mask is static; "
            "there is no trajectory to phase-shift"
        )

    if mode == "point":
        point_mask = load_point_force_mask(
            x_pos=force_payload["model"]["x_pos"],
            y_pos=force_payload["model"]["y_pos"],
            num_frames=num_video_frames,
            height=height,
            width=width,
        )
    else:
        # Wind is a global field, so its "where" mask is the whole frame.
        point_mask = torch.ones_like(wind_signal[:, :1])
    return torch.cat([point_mask, wind_signal], dim=1)


def summarize_condition_signal(signal: torch.Tensor) -> Dict[str, Any]:
    return {
        "shape": list(signal.shape),
        "dtype": str(signal.dtype),
        "min": float(signal.min().item()),
        "max": float(signal.max().item()),
        "mean": float(signal.mean().item()),
    }
