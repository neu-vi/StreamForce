from __future__ import annotations

import base64
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.io import write_video


def ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def resize_rgb_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if image is None:
        raise ValueError("Input image is None")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 image, got shape={image.shape}")
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def image_to_data_url(
    image: np.ndarray,
    image_format: str = "PNG",
    jpeg_quality: int = 85,
) -> str:
    arr = image.astype(np.uint8, copy=False)
    fmt = image_format.strip().upper()
    if fmt in {"JPG", "JPEG"}:
        ok, enc = cv2.imencode(".jpg", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        if not ok:
            raise ValueError("Failed to JPEG-encode frame")
        b64 = base64.b64encode(enc.tobytes()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"
    ok, enc = cv2.imencode(".png", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError("Failed to PNG-encode frame")
    b64 = base64.b64encode(enc.tobytes()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def write_video_uint8(path: str, frames: np.ndarray, fps: int = 16) -> str:
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected [T,H,W,3] frames, got shape={frames.shape}")
    ensure_dir(os.path.dirname(path) or ".")
    tensor = torch.from_numpy(frames.astype(np.uint8))
    write_video(path, tensor, fps=fps)
    return path


def image_to_model_tensor(image: np.ndarray) -> torch.Tensor:
    # [H,W,C] uint8 -> [1,1,C,H,W] float32
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).contiguous().float()
    return tensor
