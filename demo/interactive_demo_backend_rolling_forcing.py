from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.rolling_forcing_streaming_inference import (
    ControlRollingForcingStreamingInferencePipeline,
)
from utils.misc import compress_time

DEMO_ROOT = Path(__file__).resolve().parent
if str(DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEMO_ROOT))

from interactive_demo_force_adapter import summarize_condition_signal
from interactive_demo_utils import image_to_model_tensor, resize_rgb_image


def _resolve_repo_path(path: str) -> str:
    """Accept paths relative to the repo root as well as to the current directory.

    v6 passed `configs/...` and `logs/...` straight to `open()`, so it only worked when started
    from the repo root. Resolving against REPO_ROOT as a fallback makes the launch directory
    irrelevant.
    """
    if not path:
        return path
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return str(candidate)
    from_repo = REPO_ROOT / candidate
    return str(from_repo) if from_repo.exists() else str(candidate)


# `_initialize_kv_cache`'s fallback allocation when `local_attn_size == -1`: 34320 tokens,
# which at frame_seq_length=390 is only 88 latent frames.
DEFAULT_KV_CACHE_TOKENS = 34320


# `_initialize_kv_cache` in the rolling-forcing pipeline allocates
# `24 * frame_seq_length` = sink (3 latent frames) + working window (21), following the
# original RollingForcing design. It is not sized from the clip length, which is precisely why
# this path can run indefinitely at flat memory.
ROLLING_FORCING_CACHE_FRAMES = 24


class GenerationStopped(Exception):
    pass


@dataclass
class DemoBackendConfig:
    config_path: str
    checkpoint_path: str
    output_dir: str
    height: int = 480
    width: int = 832
    num_latent_frames: int = 126
    seed: int = 0
    use_ema: bool = True
    device: Optional[str] = None
    # Put the VAE on its own GPU so its decode stops competing with the generator for SMs.
    # None keeps it beside the generator.
    vae_device: Optional[str] = None
    # channels_last_3d on the VAE's Conv3d weights. Without it PyTorch cannot hand these convs
    # to cuDNN and falls back to aten::slow_conv_dilated3d (an im2col path): measured 809 -> 225
    # ms per 3-latent decode, a 3.6x win on what is ~62% of a block. It picks a different conv
    # algorithm, so the decode is not bit-identical -- measured max 0.039 on a [-1,1] output,
    # i.e. ~5/255 levels, mean 0.15/255.
    vae_channels_last: bool = False
    # Same fix for the generator's Conv3d -- the controlnet's input_hint_block runs on the
    # full-resolution hint, so it hit the same im2col fallback (36.9 ms of vol2col plus ~32k
    # elementwise launches per forward).
    gen_channels_last: bool = False
    # Rolling-forcing knobs, matching inference_causal_rolling_forcing.py's defaults.
    rolling_forcing_block_frames: int = 3
    rolling_forcing_max_frames: int = 21
    rolling_forcing_cache_frames: int = 96


class InteractiveDemoBackend:
    def __init__(self, cfg: DemoBackendConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.pipeline: Optional[ControlRollingForcingStreamingInferencePipeline] = None

        self.num_video_frames: int = 0

    def load(self) -> Dict[str, Any]:
        self.cfg.config_path = _resolve_repo_path(self.cfg.config_path)
        self.cfg.checkpoint_path = _resolve_repo_path(self.cfg.checkpoint_path)
        if not os.path.exists(self.cfg.config_path):
            raise FileNotFoundError(f"Config not found: {self.cfg.config_path}")
        if not os.path.exists(self.cfg.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.cfg.checkpoint_path}")

        torch.set_grad_enabled(False)
        torch.manual_seed(self.cfg.seed)

        config = OmegaConf.load(self.cfg.config_path)
        default_config = OmegaConf.load(_resolve_repo_path("configs/default_config.yaml"))
        config = OmegaConf.merge(default_config, config)
        config.gradient_checkpointing = False
        self.num_video_frames = (int(self.cfg.num_latent_frames) - 1) * 4 + 1

        # ---- turn rolling forcing on --------------------------------------------------
        # Exactly how inference_causal_rolling_forcing.py does it (lines 128-133): inject into
        # `generator_model_kwargs` at runtime, so `CausalWanControlNetWrapper` passes the flags
        # down to CausalControlledWanModel/CausalControlNet, whose CausalWanSelfAttention then
        # takes the rolling-forcing branch. Nothing outside this folder is modified.
        if "generator_model_kwargs" not in config:
            config.generator_model_kwargs = {}
        config.generator_model_kwargs.rolling_forcing_attention = True
        config.generator_model_kwargs.rolling_forcing_block_frames = int(self.cfg.rolling_forcing_block_frames)
        config.generator_model_kwargs.rolling_forcing_max_frames = int(self.cfg.rolling_forcing_max_frames)
        config.rolling_forcing_cache_frames = int(self.cfg.rolling_forcing_cache_frames)

        self.pipeline = ControlRollingForcingStreamingInferencePipeline(config, device=self.device)

        try:
            state_dict = torch.load(self.cfg.checkpoint_path, map_location="cpu", mmap=True)
        except Exception:
            # mmap needs a local, seekable, new-format zip archive; fall back if it is not.
            state_dict = torch.load(self.cfg.checkpoint_path, map_location="cpu")
        state_key = "generator_ema" if self.cfg.use_ema else "generator"
        if state_key not in state_dict:
            raise KeyError(f"Missing '{state_key}' in checkpoint")

        generator_state = state_dict[state_key]
        cleaned = {}
        for n, p in generator_state.items():
            cleaned[n.replace("_fsdp_wrapped_module.", "").replace("_checkpoint_wrapped_module.", "").replace("_orig_mod.", "")] = p

        self.pipeline.generator.load_state_dict(cleaned)
        # Drop the host-side copies before moving to GPU, so two models can be loaded one after
        # the other without holding both checkpoints in RAM.
        del cleaned, generator_state, state_dict
        self.pipeline = self.pipeline.to(device=self.device, dtype=torch.bfloat16)

        # Hand the VAE its own GPU if asked. Only the decode moves; the generator stays put and
        # the per-block latent handed across is ~0.45 MB.
        if self.cfg.vae_device:
            vae_dev = torch.device(self.cfg.vae_device)
            self.pipeline.vae = self.pipeline.vae.to(device=vae_dev)
            self.pipeline.vae_device = vae_dev
            print(f"[rf] VAE on {vae_dev}, generator on {self.device}")
        else:
            self.pipeline.vae_device = None

        if self.cfg.vae_channels_last:
            n = 0
            for m in self.pipeline.vae.model.modules():
                if isinstance(m, torch.nn.Conv3d) and m.weight.dim() == 5:
                    m.weight.data = m.weight.data.to(memory_format=torch.channels_last_3d)
                    n += 1
            print(f"[rf] VAE: {n} Conv3d weights -> channels_last_3d (cuDNN instead of "
                  f"slow_conv_dilated3d)")

        if self.cfg.gen_channels_last:
            g = 0
            for m in self.pipeline.generator.modules():
                if isinstance(m, torch.nn.Conv3d) and m.weight.dim() == 5:
                    m.weight.data = m.weight.data.to(memory_format=torch.channels_last_3d)
                    g += 1
            print(f"[rf] generator: {g} Conv3d weights -> channels_last_3d")

        # ---- KV cache ---------------------------------------------------------------------
        # Nothing to size here, and that is the point. The rolling-forcing pipeline allocates a
        # fixed `24 * frame_seq_length` cache -- a 3-latent-frame attention sink plus a
        # 21-frame working window -- and evicts, so memory is flat in clip length instead of
        # growing with it. (this demo had to enlarge an append-only cache to fit the whole
        # clip: ~34 GiB at 126 latents, and a hard crash past 88.)
        frame_seq = int(self.pipeline.frame_seq_length)
        attn = self._rolling_forcing_attention_config()
        cache_tokens = ROLLING_FORCING_CACHE_FRAMES * frame_seq
        gib = cache_tokens * 24 * 128 * 2 * 2 * int(self.pipeline.num_transformer_blocks) * 2 / (1024 ** 3)
        if not attn.get("enabled"):
            raise RuntimeError(
                "rolling forcing did not reach the attention modules -- refusing to run, because "
                "the cache is sized for a 24-frame window and the append-only path would overrun "
                "it almost immediately"
            )
        print(
            f"[rf] rolling forcing ON: sink {attn['block_frames']} latent frames, window "
            f"{attn['max_frames']} frames ({attn['max_attention_size']} tokens), "
            f"frame_length {attn['frame_length']}, {attn['modules']} attention modules",
            flush=True,
        )
        print(
            f"[rf] KV cache {cache_tokens} tokens = {ROLLING_FORCING_CACHE_FRAMES} latent frames "
            f"(~{gib:.1f} GiB, eviction ON) on {self.device}; clip length is not bounded by the "
            f"cache -- only by the RoPE table",
            flush=True,
        )
        print(
            f"[rf] generating {self.cfg.num_latent_frames} latents = {self.num_video_frames} "
            f"video frames",
            flush=True,
        )

        return {
            "device": str(self.device),
            "height": self.cfg.height,
            "width": self.cfg.width,
            "num_latent_frames": self.cfg.num_latent_frames,
            "num_video_frames": self.num_video_frames,
            "inference_mode": "self_forcing_stream",
            "checkpoint": self.cfg.checkpoint_path,
            "config": self.cfg.config_path,
        }

    def _rolling_forcing_attention_config(self) -> Dict[str, Any]:
        """Read back what the attention modules actually ended up with.

        Worth checking rather than assuming: the flags travel config -> wrapper -> from_pretrained
        -> attention module, and a silent failure anywhere in that chain would leave the
        append-only path running against a 24-frame cache.
        """
        info: Dict[str, Any] = {"enabled": False, "modules": 0}
        for module in self.pipeline.generator.modules():
            # `rolling_forcing_attention` alone is not enough to identify an attention module:
            # the container models (CausalControlledWanModel, CausalControlNet) carry the same
            # flag so they can pass it down to their blocks, but only CausalWanSelfAttention
            # derives `frame_length` / `block_length` / `max_attention_size` from it. Requiring
            # those is what distinguishes the module that actually implements the branch.
            if not all(
                hasattr(module, name)
                for name in ("rolling_forcing_attention", "frame_length", "block_length", "max_attention_size")
            ):
                continue
            if not bool(module.rolling_forcing_attention):
                continue
            info["modules"] += 1
            info["enabled"] = True
            info["block_frames"] = int(module.rolling_forcing_block_frames)
            info["max_frames"] = int(module.rolling_forcing_max_frames)
            info["frame_length"] = int(module.frame_length)
            info["max_attention_size"] = int(module.max_attention_size)
            info["block_length"] = int(module.block_length)
        return info


    def _prepare_hint(self, hint_full: torch.Tensor, image_tensor: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if hint_full.shape[2] < 2:
            raise ValueError(f"Expected at least 2 hint channels, got {hint_full.shape}")

        gaussian_blob = hint_full[:, :, 0:1]
        hint = hint_full[:, :, 1:]
        # "do not let the blob move" -- the mask is the blob at frame 0, broadcast over every
        # frame. The trainer (wan_controlnet_distillation.py), every inference_*.py and the ODE
        # pair generators all do this; the demos did not, so a point force's mask travelled with
        # the blob and the conditioning drifted away from what the checkpoint was trained on.
        gaussian_blob = gaussian_blob[:, 0:1]
        hint = hint * (gaussian_blob > 1e-1)

        image = image_tensor.to(device=hint.device)
        image = image.permute(0, 2, 1, 3, 4) / 255.0
        image = image * 2.0 - 1.0
        masked_image = (image + 1.0) / 2.0 * gaussian_blob.permute(0, 2, 1, 3, 4)
        masked_image = masked_image * 2.0 - 1.0
        masked_image = masked_image.permute(0, 2, 1, 3, 4)
        masked_latent = compress_time(masked_image, self.num_video_frames, method="subsample")

        hint = compress_time(hint, self.num_video_frames, method="subsample")
        return hint, masked_latent

    def generate_segment_streaming(
        self,
        reference_image: np.ndarray,
        prompt: str,
        condition_signal: torch.Tensor,
        seed: Optional[int] = None,
        on_chunk: Optional[Callable[[np.ndarray, int], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
        should_pause: Optional[Callable[[], bool]] = None,
        get_condition_signal_update: Optional[Callable[[int], Optional[Dict[str, Any]]]] = None,
        num_latent_frames: Optional[int] = None,
    ) -> Dict[str, Any]:
        if self.pipeline is None:
            raise RuntimeError("Backend is not loaded. Click 'Load/Reload Model' first.")
        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        # Per-run clip length. Only ever SHORTER than the configured maximum, which is what makes
        # this safe with no reload: `noise`/`output` are allocated per call from num_frames, and
        # the rolling-forcing KV cache is sized by the attention window (24 latents), not by clip
        # length. It must stay a call argument -- routing it through DemoBackendConfig would change
        # the backend cache key in app.py and reload the checkpoint on every gallery preset.
        run_latents = int(self.cfg.num_latent_frames)
        if num_latent_frames is not None:
            block = max(1, int(self.pipeline.num_frame_per_block))
            want = max(block, min(int(num_latent_frames), run_latents))
            # The pipeline asserts num_frames % num_frame_per_block == 0 on the image-conditioned
            # path, so round down to a whole block.
            run_latents = (want // block) * block
        run_video_frames = (run_latents - 1) * 4 + 1
        if run_latents != int(self.cfg.num_latent_frames):
            print(f"[gen] clip capped for this run: {run_latents} latents = {run_video_frames} "
                  f"video frames (max {self.cfg.num_latent_frames})", flush=True)

        image = resize_rgb_image(reference_image, width=self.cfg.width, height=self.cfg.height)
        image_tensor = image_to_model_tensor(image).to(device=self.device, dtype=torch.float32)

        hint_full = condition_signal.unsqueeze(0).to(device=self.device, dtype=torch.bfloat16)
        hint, masked_latent = self._prepare_hint(hint_full=hint_full, image_tensor=image_tensor)

        if masked_latent is not None:
            masked_latent = masked_latent.to(device=self.device, dtype=torch.bfloat16)
        hint = hint.to(device=self.device, dtype=torch.bfloat16)

        runtime_update_logs: list[Dict[str, Any]] = []
        # Highest block index the decoder has emitted. Everything above it is still inside the
        # rolling denoising window and can still absorb a new force.
        last_decoded_block = -1

        def _get_runtime_control_update(block_index: int, current_start_latent_frame: int,
                                        full_start_latent: Optional[int] = None) -> Optional[Dict[str, Optional[torch.Tensor]]]:
            nonlocal hint, masked_latent
            if get_condition_signal_update is None:
                return None
            t_update0 = time.perf_counter()
            try:
                update_info = get_condition_signal_update(
                    int(current_start_latent_frame),
                    int(full_start_latent) if full_start_latent is not None else None)
                if update_info is None:
                    return None

                updated_condition = update_info.get("condition_signal")
                if updated_condition is None:
                    return None
                video_start = int(update_info.get("video_start_frame", 0))
                seq = int(update_info.get("seq", -1))
                mode = str(update_info.get("mode", ""))
                build_ms = float(update_info.get("build_ms", 0.0))

                t_prepare0 = time.perf_counter()
                updated_hint_full = updated_condition.unsqueeze(0).to(device=self.device, dtype=torch.bfloat16)
                if mode == "wind":
                    # Wind update only needs hint channels; masked_latent is image-only and unchanged.
                    gaussian_blob = updated_hint_full[:, :, 0:1][:, 0:1]
                    updated_hint = updated_hint_full[:, :, 1:] * (gaussian_blob > 1e-1)
                    updated_masked_latent = None
                else:
                    updated_hint, updated_masked_latent = self._prepare_hint(hint_full=updated_hint_full, image_tensor=image_tensor)
                    if updated_masked_latent is not None:
                        updated_masked_latent = updated_masked_latent.to(device=self.device, dtype=torch.bfloat16)
                updated_hint = updated_hint.to(device=self.device, dtype=torch.bfloat16)
                hint_len = int(hint.shape[1])
                start_latent = max(0, min(int(current_start_latent_frame), hint_len))
                # With the blob frozen and channels 1-3 constant in time, the prepared hint is a
                # single spatial pattern for BOTH modes -- so one frame is broadcast over every
                # remaining latent, and point no longer needs a per-frame trajectory.
                if int(updated_hint.shape[1]) >= 1:
                    replace_latent = max(0, hint_len - start_latent)
                else:
                    replace_latent = 0
                if replace_latent > 0:
                    hint[:, start_latent:start_latent + replace_latent] = updated_hint[:, :1].expand(
                        -1, replace_latent, -1, -1, -1
                    )
                    if mode != "wind" and updated_masked_latent is not None:
                        # Under direct_add, masked_latent is a SINGLE standing frame ("this is
                        # the reference frame"), expanded across every frame of every block
                        # inside the model -- it is not time-indexed. The original
                        # [start_latent : start_latent + replace_latent] assignment was
                        # therefore an empty write whenever start_latent > 0, so a point-force
                        # change reached the hint while the mask stayed on the OLD blob: two
                        # contradictory conditions. Replace it wholesale instead.
                        masked_latent = updated_masked_latent
                t_update1 = time.perf_counter()
                prepare_ms = (t_update1 - t_prepare0) * 1000.0
                total_ms = (t_update1 - t_update0) * 1000.0
                runtime_update_logs.append(
                    {
                        "seq": int(seq),
                        "mode": mode,
                        "block_index": int(block_index),
                        "latent_start_frame": int(start_latent),
                        "video_start_frame": int(video_start),
                        "updated_latent_frames": int(replace_latent),
                        "build_ms": float(build_ms),
                        "prepare_ms": float(prepare_ms),
                        "total_ms": float(total_ms),
                    }
                )
                return {"hint": hint, "masked_latent": masked_latent}
            except Exception as exc:
                total_ms = (time.perf_counter() - t_update0) * 1000.0
                runtime_update_logs.append(
                    {
                        "seq": -1,
                        "mode": "unknown",
                        "block_index": int(block_index),
                        "latent_start_frame": int(current_start_latent_frame),
                        "video_start_frame": -1,
                        "updated_latent_frames": 0,
                        "build_ms": 0.0,
                        "prepare_ms": 0.0,
                        "total_ms": float(total_ms),
                        "error": str(exc),
                    }
                )
                print(f"[v6 runtime update] failed at block={block_index} latent_start={current_start_latent_frame}: {exc}")
                return None

        def _on_decoded_chunk(chunk_tensor: torch.Tensor, block_index: int) -> None:
            nonlocal last_decoded_block
            last_decoded_block = max(last_decoded_block, int(block_index))
            if on_chunk is None:
                return
            chunk_np = chunk_tensor[0].permute(0, 2, 3, 1).detach().cpu().numpy()
            chunk_frames = (chunk_np * 255.0).clip(0, 255).astype(np.uint8)
            on_chunk(chunk_frames, block_index)
            if should_stop is not None and should_stop():
                raise GenerationStopped("Stopped by user request.")
            # The pause/pacing wait used to live here. Since the VAE decode moved to a worker
            # thread this callback no longer runs on the generation thread, so waiting here
            # throttled DECODING while denoising raced ahead -- which is what made pacing pace
            # the wrong thing. It now lives in the per-window hook below.

        window_calls = 0

        def _get_runtime_control_update_rf() -> Optional[Dict[str, Optional[torch.Tensor]]]:
            """Called once per window, in order, ON THE GENERATION THREAD.

            Two things depend on that, and both used to be wrong because they read
            `last_decoded_block` -- which the async VAE worker updates, so it tracks DECODING, not
            denoising. Pacing then drove the two frontiers apart and the error became large.

            The window index is simply the call count, from which the whole schedule follows
            (mirrors the pipeline: start_block = w - steps + 1, end_block = min(nb - 1, w)):

              * block `w` is entering the window now, on its FIRST denoising step, so it and
                everything after it see the new force for all of their steps -> fully affected.
              * blocks `w-1 .. w-steps+1` are 1..steps-1 steps in, so they can only take the new
                force for their remaining steps -> partially affected.
              * blocks below that are finished; writing their hint is a no-op.

            So the write starts at the oldest still-in-flight block (most responsive) and the
            caller is told both boundaries.
            """
            nonlocal window_calls
            window_index = window_calls
            window_calls += 1

            # Pacing / manual pause, on the generation thread: this is what actually holds
            # denoising. Stop is checked inside so it can never deadlock against a hold.
            if should_pause is not None:
                steps_blk = int(self.pipeline.num_frame_per_block)
                den_first = min(window_index, max(0, run_latents // steps_blk - 1)) * steps_blk
                den_frame = (den_first - 1) * 4 + 1 if den_first > 0 else 0
                while should_pause(den_frame):
                    if should_stop is not None and should_stop():
                        raise GenerationStopped("Stopped by user request.")
                    time.sleep(0.02)

            steps = len(self.pipeline.denoising_step_list)
            blk = int(self.pipeline.num_frame_per_block)
            num_blocks = max(1, run_latents // blk)
            partial_block = max(0, window_index - steps + 1)      # oldest still in flight
            full_block = min(num_blocks - 1, window_index)        # first with ALL steps affected
            return _get_runtime_control_update(
                window_index, partial_block * blk, full_start_latent=full_block * blk)

        t0 = time.perf_counter()
        with torch.no_grad():
            video, stream_metrics = self.pipeline.inference_rolling_forcing_stream(
                image=image_tensor,
                text_prompts=[prompt],
                num_frames=run_latents,
                return_latents=False,
                hint=hint,
                masked_latent=masked_latent,
                on_decoded_chunk=_on_decoded_chunk,
                # The rolling-forcing hook takes no arguments (the self-forcing one takes
                # block_index and current_start_frame), so the adapter supplies them.
                get_runtime_control_update=_get_runtime_control_update_rf,
            )
        t1 = time.perf_counter()

        video_np = rearrange(video, "b t c h w -> b t h w c").detach().cpu().numpy()
        frames = (video_np[0] * 255.0).clip(0, 255).astype(np.uint8)

        self.pipeline.vae.model.clear_cache()

        total_s = max(t1 - t0, 1e-8)
        stream_metrics = dict(stream_metrics)
        stream_metrics["backend_total_ms"] = total_s * 1000.0

        return {
            "video_path": None,
            "frames": frames,
            "num_frames": int(frames.shape[0]),
            "signal_summary": summarize_condition_signal(condition_signal),
            "hint_summary": summarize_condition_signal(hint_full[0].float().cpu()),
            "stream_metrics": stream_metrics,
            "runtime_update_logs": runtime_update_logs,
        }
