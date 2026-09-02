"""Describe uploaded images with Qwen3-VL: whole-scene captions, or point-force prompts."""

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
    def _run(self, pil_image: Image.Image, instruction: str, max_new_tokens: int) -> str:
        """Ask the VLM `instruction` about `pil_image` and return the reply."""
        if not self.ready:
            raise RuntimeError("captioner not loaded yet")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": instruction},
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
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

    def caption(self, pil_image: Image.Image, max_new_tokens: int = 256) -> str:
        """Return a short description of the image content.

        Used for wind force, where the force acts on the whole scene and a static description
        plus the force vector is all the generator needs.
        """
        caption = self._run(
            pil_image,
            "Describe the content of this image in one or two sentences. "
            "Focus on the main subject, its appearance, and the setting. "
            "Do not describe any motion or how things might move.",
            max_new_tokens,
        )
        with self._lock:
            log(f"[captioner] caption: {caption[:120]}{'…' if len(caption) > 120 else ''}")
        return caption

    def expand_prompt(
        self, pil_image: Image.Image, object_hint: str, max_new_tokens: int = 320,
    ) -> str:
        """Expand a short phrase naming an object into a full point-force prompt.

        Point force acts on ONE object, so the prompt has to name it -- a plain caption cannot,
        which is why this takes a hint from the user. The requested shape mirrors the point
        prompts in assets/gallery.json, i.e. what the generator was trained on: the object is
        named and grounded in the image, an unseen external force nudges it, and the rest of the
        scene is explicitly left undisturbed.
        """
        hint = " ".join(object_hint.split())
        if not hint:
            raise ValueError("object_hint is empty")

        prompt = self._run(
            pil_image,
            "You write prompts for a video generation model that simulates a physical force "
            "pushing a single object in an image.\n\n"
            f'The user wants to move this object: "{hint}"\n\n'
            "Find that object in the image and write ONE prompt describing the moment the force "
            "acts on it. Requirements:\n"
            "- Name the object using what you can actually see in the image, so it is "
            "unambiguous: keep the user's words but add its real colour, material and position "
            "(for example \"the white mug\" or \"the dark red vase on the low wooden table\").\n"
            "- Say that an unseen external force nudges it, and describe how it moves in "
            "response -- sliding, tilting or shifting across the surface it rests on.\n"
            "- Mention the immediate surroundings, then state that the rest of the scene stays "
            "still and undisturbed.\n"
            "- Describe only this one object moving. Nothing else in the scene moves.\n"
            "- Two or three sentences of flowing prose. Output the prompt itself and nothing "
            "else: no preamble, no quotation marks, no bullet points.",
            max_new_tokens,
        )
        with self._lock:
            log(f"[captioner] expanded \"{hint}\" -> {prompt[:110]}{'…' if len(prompt) > 110 else ''}")
        return prompt
