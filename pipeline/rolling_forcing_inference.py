from typing import List, Optional

import torch

from utils.wan_wrapper import CausalWanControlNetWrapper, WanTextEncoder, WanVAEWrapper


class ControlRollingForcingInferencePipeline(torch.nn.Module):
    def __init__(
        self,
        args,
        device,
        generator=None,
        text_encoder=None,
        vae=None,
    ):
        super().__init__()
        self.generator = (
            CausalWanControlNetWrapper(
                **getattr(args, "generator_model_kwargs", {}),
                is_causal=True,
                training=False,
                meta_init=False,
            )
            if generator is None
            else generator
        )
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper() if vae is None else vae

        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat(
                (self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32))
            )
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_transformer_blocks = 30
        self.frame_seq_length = 390

        self.kv_cache1 = None
        self.kv_cache2 = None
        self.crossattn_cache = None
        self.crossattn_cache2 = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = args.independent_first_frame
        self.local_attn_size = self.generator.model.local_attn_size

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block
        self.first_frame_control = getattr(args, "first_frame_control", True)
        self.initial_latent_to_cahce = getattr(args, "initial_latent_to_cahce", False)
        self.rolling_forcing_cache_frames = getattr(args, "rolling_forcing_cache_frames", 96)

    def inference_rolling_forcing(
        self,
        image: torch.Tensor,
        text_prompts: List[str],
        num_frames: int,
        return_latents: bool = False,
        profile: bool = False,
        hint: Optional[torch.Tensor] = None,
        masked_latent: Optional[torch.Tensor] = None,
        long_generation: bool = False,
    ) -> torch.Tensor:
        del long_generation  # Not used in rolling-forcing mode.

        image = image.to(self.generator.model.device, dtype=torch.bfloat16)
        image = image.permute(0, 2, 1, 3, 4)
        image = image / 255.0
        image = image * 2.0 - 1.0
        initial_latent = self.vae.encode_to_latent(image).to(
            device=self.generator.model.device, dtype=torch.bfloat16
        )

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
        if not self.independent_first_frame or (
            self.independent_first_frame and initial_latent is not None
        ):
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block

        num_output_frames = num_frames
        conditional_dict = self.text_encoder(text_prompts=text_prompts)

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype,
        )

        if profile:
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            vae_start = torch.cuda.Event(enable_timing=True)
            vae_end = torch.cuda.Event(enable_timing=True)
            block_times = []
            block_start = torch.cuda.Event(enable_timing=True)
            block_end = torch.cuda.Event(enable_timing=True)
            init_start.record()

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
                self.kv_cache1[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device
                )
                self.kv_cache1[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device
                )
                self.kv_cache2[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device
                )
                self.kv_cache2[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device
                )

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
                masked_latent=masked_latent[:, :1]
                if self.first_frame_control and masked_latent is not None
                else None,
            )

        if profile:
            init_end.record()
            torch.cuda.synchronize()
            diffusion_start.record()

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

        for window_index in range(window_num):
            if profile:
                block_start.record()

            print("window_index:", window_index)
            start_block = window_start_blocks[window_index]
            end_block = window_end_blocks[window_index]
            print(f"start_block: {start_block}, end_block: {end_block}")

            current_start_frame = start_block * self.num_frame_per_block
            current_end_frame = (end_block + 1) * self.num_frame_per_block
            current_num_frames = current_end_frame - current_start_frame

            if (
                current_num_frames
                == rolling_window_length_blocks * self.num_frame_per_block
                or current_start_frame == 0
            ):  # full window or still in the first window
                noisy_input = torch.cat(
                    [
                        noisy_cache[
                            :,
                            current_start_frame : current_end_frame
                            - self.num_frame_per_block,
                        ],  # put the cached noisy frames into the window to take up the blocks before the new block enters
                        noise[
                            :,
                            current_end_frame
                            - self.num_frame_per_block : current_end_frame,
                        ],  # attach pure noise for the new block entering the window
                    ],
                    dim=1,
                )
            else:
                noisy_input = noisy_cache[:, current_start_frame:current_end_frame] # only at the end of the process, where the window is not full again

            if current_num_frames == rolling_window_length_blocks * self.num_frame_per_block:
                current_timestep = shared_timestep.clone()
            elif current_start_frame == 0:
                current_timestep = shared_timestep[:, -current_num_frames:].clone()
            elif current_end_frame == num_frames:
                current_timestep = shared_timestep[:, :current_num_frames].clone()
            else:
                raise ValueError(
                    "current_num_frames should be full window, first window, or last window"
                )

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

            with torch.no_grad():
                for block_idx in range(start_block, end_block + 1):
                    local_start = (block_idx - start_block) * self.num_frame_per_block
                    local_end = local_start + self.num_frame_per_block

                    block_timestep_values = current_timestep[:, local_start:local_end].reshape(-1).float()
                    # I2V first block can be [0, t, t]; map block timestep using non-zero value.
                    non_zero_values = block_timestep_values[block_timestep_values > 0]
                    if non_zero_values.numel() > 0:
                        block_timestep = non_zero_values.float().mean().item()
                    else:
                        block_timestep = block_timestep_values.float().mean().item()
                    matches = torch.abs(self.denoising_step_list.float() - block_timestep) < 1e-4
                    matched_indices = torch.nonzero(matches, as_tuple=True)[0]
                    if matched_indices.numel() == 0:
                        nearest_idx = torch.argmin(
                            torch.abs(self.denoising_step_list.float() - block_timestep)
                        ).item()
                        nearest_step = self.denoising_step_list[nearest_idx].item()
                        if abs(float(nearest_step) - float(block_timestep)) < 1e-2:
                            matched_indices = torch.tensor([nearest_idx], device=noise.device)
                        else:
                            block_timestep_values_list = [
                                float(v) for v in block_timestep_values.detach().cpu().tolist()
                            ]
                            raise RuntimeError(
                                "Cannot map block timestep "
                                f"{block_timestep} to denoising_step_list "
                                f"{self.denoising_step_list.detach().cpu().tolist()} "
                                f"(block timestep values: {block_timestep_values_list})"
                            )

                    if matched_indices.numel() == 0:
                        raise RuntimeError(f"Cannot map block timestep {block_timestep} to denoising_step_list")

                    block_timestep_index = int(matched_indices[0].item())
                    if block_timestep_index == len(self.denoising_step_list) - 1:
                        continue

                    next_timestep = self.denoising_step_list[block_timestep_index + 1].to(noise.device)

                    updated_noisy = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep
                        * torch.ones(
                            [batch_size * current_num_frames],
                            device=noise.device,
                            dtype=torch.long,
                        ),
                    ).unflatten(0, denoised_pred.shape[:2])

                    noisy_cache[
                        :,
                        block_idx * self.num_frame_per_block : (block_idx + 1) * self.num_frame_per_block,
                    ] = updated_noisy[:, local_start:local_end]

            with torch.no_grad():
                context_denoised_pred = denoised_pred[:, : self.num_frame_per_block].clone()
                context_timestep = (
                    torch.ones_like(current_timestep[:, : self.num_frame_per_block])
                    * self.args.context_noise
                )

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

            if profile:
                block_end.record()
                torch.cuda.synchronize()
                block_times.append(block_start.elapsed_time(block_end))

        if profile:
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)
            vae_start.record()

        video = self.vae.decode_to_pixel(output, use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        if profile:
            vae_end.record()
            torch.cuda.synchronize()
            vae_time = vae_start.elapsed_time(vae_end)
            total_time = init_time + diffusion_time + vae_time

            print("Profiling results:")
            print(
                f"  - Initialization/caching time: {init_time:.2f} ms "
                f"({100 * init_time / total_time:.2f}%)"
            )
            print(
                f"  - Diffusion generation time: {diffusion_time:.2f} ms "
                f"({100 * diffusion_time / total_time:.2f}%)"
            )
            for i, block_time in enumerate(block_times):
                print(
                    f"    - Window {i} generation time: {block_time:.2f} ms "
                    f"({100 * block_time / diffusion_time:.2f}% of diffusion)"
                )
            print(f"  - VAE decoding time: {vae_time:.2f} ms ({100 * vae_time / total_time:.2f}%)")
            print(f"  - Total time: {total_time:.2f} ms")

        if return_latents:
            return video, output
        return video

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []
        # if self.local_attn_size != -1:
        #     kv_cache_size = self.local_attn_size * self.frame_seq_length
        # else:
        #     kv_cache_size = 34320
        # kv_cache_size = max(
        #     kv_cache_size, self.rolling_forcing_cache_frames * self.frame_seq_length
        # )
        kv_cache_size = 24 * self.frame_seq_length  # following original RollingForcing design

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append(
                {
                    "k": torch.zeros([batch_size, kv_cache_size, 24, 128], dtype=dtype, device=device),
                    "v": torch.zeros([batch_size, kv_cache_size, 24, 128], dtype=dtype, device=device),
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                }
            )

        self.kv_cache1 = kv_cache1

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append(
                {
                    "k": torch.zeros([batch_size, 512, 24, 128], dtype=dtype, device=device),
                    "v": torch.zeros([batch_size, 512, 24, 128], dtype=dtype, device=device),
                    "is_init": False,
                }
            )
        self.crossattn_cache = crossattn_cache

    def _initialize_kv_cache2(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache2 = []
        # if self.local_attn_size != -1:
        #     kv_cache_size = self.local_attn_size * self.frame_seq_length
        # else:
        #     kv_cache_size = 34320
        # kv_cache_size = max(
        #     kv_cache_size, self.rolling_forcing_cache_frames * self.frame_seq_length
        # )
        kv_cache_size = 24 * self.frame_seq_length  # following original RollingForcing design

        for _ in range(self.num_transformer_blocks):
            kv_cache2.append(
                {
                    "k": torch.zeros([batch_size, kv_cache_size, 24, 128], dtype=dtype, device=device),
                    "v": torch.zeros([batch_size, kv_cache_size, 24, 128], dtype=dtype, device=device),
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                }
            )

        self.kv_cache2 = kv_cache2

    def _initialize_crossattn_cache2(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache2 = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache2.append(
                {
                    "k": torch.zeros([batch_size, 512, 24, 128], dtype=dtype, device=device),
                    "v": torch.zeros([batch_size, 512, 24, 128], dtype=dtype, device=device),
                    "is_init": False,
                }
            )
        self.crossattn_cache2 = crossattn_cache2
