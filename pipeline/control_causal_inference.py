from typing import List, Optional
import torch

from utils.wan_wrapper import WanTextEncoder, WanVAEWrapper, CausalWanControlNetWrapper


class ControlCausalInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None
    ):
        super().__init__()
        # Step 1: Initialize all models
        self.generator = CausalWanControlNetWrapper(
            **getattr(args, "generator_model_kwargs", {}), is_causal=True, training=False, meta_init=False) if generator is None else generator
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper() if vae is None else vae

        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
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

    # def inference(
    #     self,
    #     noise: torch.Tensor,
    #     text_prompts: List[str],
    #     initial_latent: Optional[torch.Tensor] = None,
    #     return_latents: bool = False,
    #     profile: bool = False,
    #     hint: Optional[torch.Tensor] = None,
    #     masked_latent: Optional[torch.Tensor] = None,
    #     long_generation: bool = False,
    # ) -> torch.Tensor:
    #     """
    #     Perform inference on the given noise and text prompts.
    #     Inputs:
    #         noise (torch.Tensor): The input noise tensor of shape
    #             (batch_size, num_output_frames, num_channels, height, width).
    #         text_prompts (List[str]): The list of text prompts.
    #         initial_latent (torch.Tensor): The initial latent tensor of shape
    #             (batch_size, num_input_frames, num_channels, height, width).
    #             If num_input_frames is 1, perform image to video.
    #             If num_input_frames is greater than 1, perform video extension.
    #         return_latents (bool): Whether to return the latents.
    #     Outputs:
    #         video (torch.Tensor): The generated video tensor of shape
    #             (batch_size, num_output_frames, num_channels, height, width).
    #             It is normalized to be in the range [0, 1].
    #     """
    #     pass

    def inference(
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
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            initial_latent (torch.Tensor): The initial latent tensor of shape
                (batch_size, num_input_frames, num_channels, height, width).
                If num_input_frames is 1, perform image to video.
                If num_input_frames is greater than 1, perform video extension.
            return_latents (bool): Whether to return the latents.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                It is normalized to be in the range [0, 1].
        """
        image = image.to(self.generator.model.device, dtype=torch.bfloat16)
        image = image.permute(0, 2, 1, 3, 4)
        image = image / 255.0
        image = image * 2.0 - 1.0
        initial_latent = self.vae.encode_to_latent(image).to(device=self.generator.model.device, dtype=torch.bfloat16)

        noise = torch.randn(
            [1, num_frames, initial_latent.shape[2], initial_latent.shape[3], initial_latent.shape[4]],
            device=self.generator.model.device,
            dtype=torch.bfloat16
        )   # [C, F, H, W]


        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        # num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        # num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        num_input_frames = 0
        num_output_frames = num_frames
        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Set up profiling if requested
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

        # Step 1: Initialize KV cache to all zeros
        if self.kv_cache1 is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
            self._initialize_kv_cache2(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
            self._initialize_crossattn_cache2(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
        else:
            # reset cross attn cache
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False
                self.crossattn_cache2[block_index]["is_init"] = False
            # reset kv cache
            for block_index in range(len(self.kv_cache1)):
                self.kv_cache1[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache1[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache2[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache2[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)

        # Step 2: Cache context feature
        current_start_frame = 0
        cache_start_frame = 0
        if self.initial_latent_to_cahce:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            self.generator(
                noisy_image_or_video=initial_latent[:, :1],
                conditional_dict=conditional_dict,
                timestep=timestep * 0,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                kv_cache2=self.kv_cache2,
                crossattn_cache2=self.crossattn_cache2,
                current_start=current_start_frame * self.frame_seq_length,
                hint=hint[:, :1] if self.first_frame_control and hint is not None else None,
                masked_latent=masked_latent[:, :1] if self.first_frame_control and masked_latent is not None else None,
            )
            cache_start_frame += 1

        if profile:
            init_end.record()
            torch.cuda.synchronize()
            diffusion_start.record()

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        for block_index, current_num_frames in enumerate(all_num_frames):
            if profile:
                block_start.record()

            noisy_input = noise[
                :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]
            if hint is not None:
                hint_input = hint[
                    :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]
            else:
                hint_input = None

            if masked_latent is not None:
                # masked_latent_input = masked_latent[
                #     :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]
                masked_latent_input = masked_latent
            else:
                masked_latent_input = None

            # Step 3.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                print(f"current_timestep: {current_timestep}")
                # set current timestep
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep

                if block_index == 0:
                    noisy_input[:, :1] = initial_latent[:, :1]  # store the initial latent
                    timestep[:, :1] = 0
                    print("first block timestep: ", timestep)

                if index < len(self.denoising_step_list) - 1:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        kv_cache2=self.kv_cache2,
                        crossattn_cache2=self.crossattn_cache2,
                        current_start=cache_start_frame * self.frame_seq_length,
                        hint=hint_input,
                        masked_latent=masked_latent_input,
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # for getting real output
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        kv_cache2=self.kv_cache2,
                        crossattn_cache2=self.crossattn_cache2,
                        current_start=cache_start_frame * self.frame_seq_length,
                        hint=hint_input,
                        masked_latent=masked_latent_input,
                    )

            # Step 3.2: record the model's output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred
        
            if long_generation:
                # if (current_start_frame - num_input_frames) % 3 == 0 and (current_start_frame - num_input_frames) > 0:
                if cache_start_frame > 6:
                    # clean cache to avoid degradation: https://derewah.dev/projects/self-forcing-endless
                    self._initialize_kv_cache(batch_size=1, dtype=noise.dtype, device=noise.device)
                    self._initialize_kv_cache2(batch_size=1, dtype=noise.dtype, device=noise.device)
                    self._initialize_crossattn_cache(batch_size=1, dtype=noise.dtype, device=noise.device)
                    self._initialize_crossattn_cache2(batch_size=1, dtype=noise.dtype, device=noise.device)
                    cache_start_frame = 0
                    self.vae.model.clear_cache()

            # Step 3.3: rerun with timestep zero to update KV cache using clean context
            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                kv_cache2=self.kv_cache2,
                crossattn_cache2=self.crossattn_cache2,
                current_start=cache_start_frame * self.frame_seq_length,
                hint=hint_input,
                masked_latent=masked_latent_input,
            )

            if profile:
                block_end.record()
                torch.cuda.synchronize()
                block_time = block_start.elapsed_time(block_end)
                block_times.append(block_time)

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames
            cache_start_frame += current_num_frames

        if profile:
            # End diffusion timing and synchronize CUDA
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)
            vae_start.record()

        # Step 4: Decode the output
        # if initial_latent is not None:
        #     # output = output[:, 1:]
        #     output = output[:, :-1] # remove the last latent frame and add in the initial latent frame
        video = self.vae.decode_to_pixel(output, use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        if profile:
            # End VAE timing and synchronize CUDA
            vae_end.record()
            torch.cuda.synchronize()
            vae_time = vae_start.elapsed_time(vae_end)
            total_time = init_time + diffusion_time + vae_time

            print("Profiling results:")
            print(f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)")
            print(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)")
            for i, block_time in enumerate(block_times):
                print(f"    - Block {i} generation time: {block_time:.2f} ms ({100 * block_time / diffusion_time:.2f}% of diffusion)")
            print(f"  - VAE decoding time: {vae_time:.2f} ms ({100 * vae_time / total_time:.2f}%)")
            print(f"  - Total time: {total_time:.2f} ms")

        if return_latents:
            return video, output
        else:
            return video

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []
        if self.local_attn_size != -1:
            # Use the local attention size to compute the KV cache size
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            # Use the default KV cache size
            # kv_cache_size = 32760
            kv_cache_size = 34320

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, kv_cache_size, 24, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 24, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 24, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 24, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache

    def _initialize_kv_cache2(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache2 = []
        if self.local_attn_size != -1:
            # Use the local attention size to compute the KV cache size
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            # Use the default KV cache size
            # kv_cache_size = 32760
            kv_cache_size = 34320

        for _ in range(self.num_transformer_blocks):
            kv_cache2.append({
                "k": torch.zeros([batch_size, kv_cache_size, 24, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 24, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache2 = kv_cache2  # always store the clean cache

    def _initialize_crossattn_cache2(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache2 = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache2.append({
                "k": torch.zeros([batch_size, 512, 24, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 24, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache2 = crossattn_cache2
