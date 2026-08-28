"""Auto-caption uploaded images with Qwen3-VL."""

from __future__ import annotations

import threading
import time

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class ImageCaptioner:
    """Loads a Qwen3-VL model once and captions PIL images on demand."""

    def __init__(self, model_path: str = "Qwen/Qwen3-VL-8B-Instruct", device: str = "auto"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.processor = None
        self._lock = threading.Lock()

    def load(self) -> None:
        log(f"[captioner] loading {self.model_path} on {self.device} ...")
        started = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        if self.device == "auto":
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_path, torch_dtype=torch.bfloat16, device_map="auto",
            )
        else:
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_path, torch_dtype=torch.bfloat16,
            ).to(self.device)
        self.model.eval()
        log(f"[captioner] loaded in {time.perf_counter() - started:.1f}s")

    @property
    def ready(self) -> bool:
        return self.model is not None and self.processor is not None

    @torch.inference_mode()
    def caption(self, pil_image: Image.Image, max_new_tokens: int = 256) -> str:
        """Return a short description of the image content."""
        if not self.ready:
            raise RuntimeError("captioner not loaded yet")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {
                        "type": "text",
                        "text": (
                            "Describe the content of this image in one or two sentences. "
                            "Focus on the main subject, its appearance, and the setting. "
                            "Do not describe any motion or how things might move."
                        ),
                    },
                ],
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text],
            images=[pil_image],
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        # Strip the input tokens from the output.
        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        caption = self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

        with self._lock:
            log(f"[captioner] caption: {caption[:120]}{'…' if len(caption) > 120 else ''}")
        return caption
