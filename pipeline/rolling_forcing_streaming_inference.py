from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from pipeline.rolling_forcing_inference import ControlRollingForcingInferencePipeline
from pipeline.streaming_vae_decoder import _AsyncStreamingVAEChunkDecoder


class ControlRollingForcingStreamingInferencePipeline(ControlRollingForcingInferencePipeline):
    """Rolling-forcing inference with incremental decode callbacks.

    This keeps the core rolling-forcing denoising logic but decodes/streams blocks
    as soon as they reach the clean timestep (i.e. when they effectively leave the
    denoising window schedule).
    """

    def inference_rolling_forcing_stream(
        self,
        image: torch.Tensor,
        text_prompts,
        num_frames: int,
        return_latents: bool = False,
        profile: bool = False,
        hint: Optional[torch.Tensor] = None,
        masked_latent: Optional[torch.Tensor] = None,
        on_decoded_chunk: Optional[Callable[[torch.Tensor, int], None]] = None,
        get_runtime_control_update: Optional[Callable[[], Optional[Dict[str, Optional[torch.Tensor]]]]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        del profile

        if torch.cuda.is_available():
            torch.cuda.synchronize(device=self.generator.model.device)
        stream_start = time.perf_counter()

        image = image.to(self.generator.model.device, dtype=torch.bfloat16)
        image = image.permute(0, 2, 1, 3, 4)
        image = image / 255.0
        image = image * 2.0 - 1.0
        # encode_to_latent takes its device from the input, so send the image to whichever
        # GPU the VAE sits on, then bring the latent back to the generator.
        _vae_dev = getattr(self, "vae_device", None)
        _img = image if _vae_dev is None else image.to(_vae_dev)
        initial_latent = self.vae.encode_to_latent(_img).to(device=self.generator.model.device, dtype=torch.bfloat16)

        noise = torch.randn(
            [
                image.shape[0],
                num_frames,
                initial_latent.shape[2],
                initial_latent.shape[3],
                initial_latent.shape[4],
            ],
            device=self.generator.model.device,
            dtype=torch.bfloat16,
        )

        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block

        conditional_dict = self.text_encoder(text_prompts=text_prompts)

        output = torch.zeros(
            [batch_size, num_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype,
        )

        if self.kv_cache1 is None:
            self._initialize_kv_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device)
            self._initialize_crossattn_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device)
            self._initialize_kv_cache2(batch_size=batch_size, dtype=noise.dtype, device=noise.device)
            self._initialize_crossattn_cache2(batch_size=batch_size, dtype=noise.dtype, device=noise.device)
        else:
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False
                self.crossattn_cache2[block_index]["is_init"] = False
            for block_index in range(len(self.kv_cache1)):
                self.kv_cache1[block_index]["global_end_index"] = torch.tensor([0], dtype=torch.long, device=noise.device)
                self.kv_cache1[block_index]["local_end_index"] = torch.tensor([0], dtype=torch.long, device=noise.device)
                self.kv_cache2[block_index]["global_end_index"] = torch.tensor([0], dtype=torch.long, device=noise.device)
                self.kv_cache2[block_index]["local_end_index"] = torch.tensor([0], dtype=torch.long, device=noise.device)

        if self.initial_latent_to_cahce:
            timestep = torch.zeros([batch_size, 1], device=noise.device, dtype=torch.int64)
            self.generator(
                noisy_image_or_video=initial_latent[:, :1],
                conditional_dict=conditional_dict,
                timestep=timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                kv_cache2=self.kv_cache2,
                crossattn_cache2=self.crossattn_cache2,
                current_start=0,
                hint=hint[:, :1] if self.first_frame_control and hint is not None else None,
                masked_latent=masked_latent[:, :1] if self.first_frame_control and masked_latent is not None else None,
            )

        num_denoising_steps = len(self.denoising_step_list)
        rolling_window_length_blocks = num_denoising_steps

        window_start_blocks = []
        window_end_blocks = []
        window_num = num_blocks + rolling_window_length_blocks - 1
        for window_index in range(window_num):
            start_block = max(0, window_index - rolling_window_length_blocks + 1)
            end_block = min(num_blocks - 1, window_index)
            window_start_blocks.append(start_block)
            window_end_blocks.append(end_block)

        noisy_cache = torch.zeros_like(output)

        shared_timestep = torch.ones(
            [batch_size, rolling_window_length_blocks * self.num_frame_per_block],
            device=noise.device,
            dtype=torch.float32,
        )
        for index, current_timestep in enumerate(reversed(self.denoising_step_list)):
            start = index * self.num_frame_per_block
            end = (index + 1) * self.num_frame_per_block
            shared_timestep[:, start:end] *= current_timestep

        # Decoding runs on a worker thread + private CUDA stream, so a block's ~805 ms of VAE
        # decode overlaps the next window's ~460 ms of denoising instead of adding to it.
        chunk_decoder = _AsyncStreamingVAEChunkDecoder(
            self.vae, device=noise.device, on_chunk=on_decoded_chunk,
            vae_device=getattr(self, "vae_device", None))
        emitted_blocks = set()

        for window_index in range(window_num):
            start_block = window_start_blocks[window_index]
            end_block = window_end_blocks[window_index]

            if get_runtime_control_update is not None:
                runtime_update = get_runtime_control_update()
                if runtime_update is not None:
                    if "hint" in runtime_update:
                        hint = runtime_update["hint"]
                    if "masked_latent" in runtime_update:
                        masked_latent = runtime_update["masked_latent"]

            current_start_frame = start_block * self.num_frame_per_block
            current_end_frame = (end_block + 1) * self.num_frame_per_block
            current_num_frames = current_end_frame - current_start_frame

            if (
                current_num_frames == rolling_window_length_blocks * self.num_frame_per_block
                or current_start_frame == 0
            ):
                noisy_input = torch.cat(
                    [
                        noisy_cache[:, current_start_frame : current_end_frame - self.num_frame_per_block],
                        noise[:, current_end_frame - self.num_frame_per_block : current_end_frame],
                    ],
                    dim=1,
                )
            else:
                noisy_input = noisy_cache[:, current_start_frame:current_end_frame]

            if current_num_frames == rolling_window_length_blocks * self.num_frame_per_block:
                current_timestep = shared_timestep.clone()
            elif current_start_frame == 0:
                current_timestep = shared_timestep[:, -current_num_frames:].clone()
            elif current_end_frame == num_frames:
                current_timestep = shared_timestep[:, :current_num_frames].clone()
            else:
                raise ValueError("current_num_frames should be full window, first window, or last window")

            if current_start_frame == 0:
                noisy_input[:, :1] = initial_latent[:, :1]
                current_timestep[:, :1] = 0

            hint_input = None
            if hint is not None:
                hint_input = hint[:, current_start_frame:current_end_frame]

            masked_latent_input = masked_latent if masked_latent is not None else None

            _, denoised_pred = self.generator(
                noisy_image_or_video=noisy_input,
                conditional_dict=conditional_dict,
                timestep=current_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                kv_cache2=self.kv_cache2,
                crossattn_cache2=self.crossattn_cache2,
                current_start=current_start_frame * self.frame_seq_length,
                hint=hint_input,
                masked_latent=masked_latent_input,
            )

            output[:, current_start_frame:current_end_frame] = denoised_pred

            # Update noisy cache for blocks that are not yet clean.
            with torch.no_grad():
                for block_idx in range(start_block, end_block + 1):
                    local_start = (block_idx - start_block) * self.num_frame_per_block
                    local_end = local_start + self.num_frame_per_block

                    block_timestep_values = current_timestep[:, local_start:local_end].reshape(-1).float()
                    non_zero_values = block_timestep_values[block_timestep_values > 0]
                    if non_zero_values.numel() > 0:
                        block_timestep = non_zero_values.float().mean().item()
                    else:
                        block_timestep = block_timestep_values.float().mean().item()

                    matches = torch.abs(self.denoising_step_list.float() - block_timestep) < 1e-4
                    matched_indices = torch.nonzero(matches, as_tuple=True)[0]
                    if matched_indices.numel() == 0:
                        nearest_idx = torch.argmin(torch.abs(self.denoising_step_list.float() - block_timestep)).item()
                        nearest_step = self.denoising_step_list[nearest_idx].item()
                        if abs(float(nearest_step) - float(block_timestep)) < 1e-2:
                            matched_indices = torch.tensor([nearest_idx], device=noise.device)
                        else:
                            raise RuntimeError(
                                f"Cannot map block timestep {block_timestep} to denoising_step_list "
                                f"{self.denoising_step_list.detach().cpu().tolist()}"
                            )

                    block_timestep_index = int(matched_indices[0].item())

                    # Emit decoded block immediately once clean.
                    if block_timestep_index == len(self.denoising_step_list) - 1 and block_idx not in emitted_blocks:
                        # Hand it off and keep denoising; the worker decodes and emits.
                        chunk_decoder.submit(denoised_pred[:, local_start:local_end], block_idx)
                        emitted_blocks.add(block_idx)
                        continue

                    if block_timestep_index == len(self.denoising_step_list) - 1:
                        continue

                    next_timestep = self.denoising_step_list[block_timestep_index + 1].to(noise.device)
                    updated_noisy = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep
                        * torch.ones([batch_size * current_num_frames], device=noise.device, dtype=torch.long),
                    ).unflatten(0, denoised_pred.shape[:2])

                    noisy_cache[
                        :,
                        block_idx * self.num_frame_per_block : (block_idx + 1) * self.num_frame_per_block,
                    ] = updated_noisy[:, local_start:local_end]

            # Update KV cache with clean context for the oldest block of this window.
            with torch.no_grad():
                context_denoised_pred = denoised_pred[:, : self.num_frame_per_block].clone()
                context_timestep = torch.ones_like(current_timestep[:, : self.num_frame_per_block]) * self.args.context_noise

                if current_start_frame == 0:
                    context_denoised_pred[:, :1] = initial_latent[:, :1]

                context_hint = None
                if hint_input is not None:
                    context_hint = hint_input[:, : self.num_frame_per_block]

                self.generator(
                    noisy_image_or_video=context_denoised_pred,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    kv_cache2=self.kv_cache2,
                    crossattn_cache2=self.crossattn_cache2,
                    current_start=current_start_frame * self.frame_seq_length,
                    hint=context_hint,
                    masked_latent=masked_latent_input,
                    updating_cache=True,
                )

        # Safety fallback in case any clean block wasn't emitted due numeric edge case.
        for block_idx in range(num_blocks):
            if block_idx in emitted_blocks:
                continue
            s = block_idx * self.num_frame_per_block
            e = s + self.num_frame_per_block
            chunk_decoder.submit(output[:, s:e], block_idx)
            emitted_blocks.add(block_idx)

        # Every submitted block must be decoded before the results are read.
        chunk_decoder.join()
        decoded_chunks = chunk_decoder.chunks
        total_decoded_frames = chunk_decoder.frames
        first_frame_ts = chunk_decoder.first_frame_ts
        first_ready_chunk_frames = chunk_decoder.first_chunk_frames

        if decoded_chunks:
            video = torch.cat(decoded_chunks, dim=1)
        else:
            video = torch.empty((batch_size, 0, 3, image.shape[3], image.shape[4]), device=image.device)

        stream_end = time.perf_counter()
        total_s = max(stream_end - stream_start, 1e-8)
        ttff_ms = None if first_frame_ts is None else (first_frame_ts - stream_start) * 1000.0

        after_first_s = None
        fps_after_first = None
        fps_after_first_chunk = None
        if first_frame_ts is not None and stream_end > first_frame_ts:
            after_first_s = stream_end - first_frame_ts
            fps_after_first = max(total_decoded_frames - 1, 0) / after_first_s
            fps_after_first_chunk = max(total_decoded_frames - first_ready_chunk_frames, 0) / after_first_s

        metrics = {
            "mode": "rolling_forcing_stream",
            "decoded_frames": int(total_decoded_frames),
            "ttff_ms": ttff_ms,
            "total_ms": total_s * 1000.0,
            "fps_end_to_end": float(total_decoded_frames) / total_s,
            "fps_after_first_frame": fps_after_first,
            "fps_after_first_chunk": fps_after_first_chunk,
            "seconds_after_first_frame": after_first_s,
            "num_latent_frames": int(num_frames),
            "num_frame_per_block": int(self.num_frame_per_block),
        }

        chunk_decoder.clear()
        self.vae.model.clear_cache()

        if return_latents:
            return (video, output), metrics
        return video, metrics
