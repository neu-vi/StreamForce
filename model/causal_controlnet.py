import torch.nn.functional as F
from typing import Tuple, Optional
from einops import rearrange
import torch.distributed as dist
import torch

from model.base import BaseModel
from utils.loss import get_denoising_loss
from utils.wan_wrapper import CausalWanControlNetWrapper, WanTextEncoder, WanVAEWrapper, WanControlNetWrapper
from pipeline import SelfForcingControlCausalTrainingPipeline
from torchvision.io import write_video
import numpy as np
import os
from utils.forceprompt_data.data_utils import (
    add_aesthetic_wind_force_prompt_to_video,
    add_aesthetic_wind_force_change_prompt_to_video,
    add_aesthetic_point_force_prompt_to_video,
    add_aesthetic_point_force_change_prompt_to_video,
)

class CausalWanControlNet(BaseModel):
    def __init__(self, args, device):
        super().__init__(args, device)
        self.denoising_loss_func = get_denoising_loss(args.denoising_loss_type)()
        # Noise augmentation in teacher forcing, we add small noise to clean context latents
        self.noise_augmentation_max_timestep = getattr(args, "noise_augmentation_max_timestep", 0)

        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.same_step_across_blocks = getattr(args, "same_step_across_blocks", True)
        self.num_training_frames = getattr(args, "num_training_frames", 21)
        self.first_frame_control = getattr(args, "first_frame_control", True)

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        if self.independent_first_frame:
            self.generator.model.independent_first_frame = True
        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()
            self.fake_score.enable_gradient_checkpointing()

        self.inference_pipeline: SelfForcingControlCausalTrainingPipeline = None

        self.num_train_timestep = args.num_train_timestep
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)
        if hasattr(args, "real_guidance_scale"):
            self.real_guidance_scale = args.real_guidance_scale
            self.fake_guidance_scale = args.fake_guidance_scale
        else:
            self.real_guidance_scale = args.guidance_scale
            self.fake_guidance_scale = 0.0
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)
        self.ts_schedule = getattr(args, "ts_schedule", True)
        self.ts_schedule_max = getattr(args, "ts_schedule_max", False)
        self.min_score_timestep = getattr(args, "min_score_timestep", 0)

        if getattr(self.scheduler, "alphas_cumprod", None) is not None:
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        else:
            self.scheduler.alphas_cumprod = None
        
        # classifier free guidance
        self.negative_prompt = getattr(args, "negative_prompt", None)
        self.guide_scale = getattr(args, "guide_scale", 5.0)
        self.debug_video = getattr(args, "debug_video", True)
        self.enable_teacher_controlnet=getattr(args, "enable_teacher_controlnet", True)

        self.debug_video_save_path = getattr(args, "debug_video_save_path", "./debug_video/")
        if not os.path.exists(self.debug_video_save_path):
            os.makedirs(self.debug_video_save_path, exist_ok=True)
        if not os.path.exists(os.path.join(self.debug_video_save_path, "train_generator")):
            os.makedirs(os.path.join(self.debug_video_save_path, "train_generator"), exist_ok=True)

        self.video_save_step = 50
        self.arrow_info = None

    def _initialize_models(self, args, device):
        self.real_model_name = getattr(args, "real_model_name", "Wan2.2-TI2V-5B")
        self.fake_model_name = getattr(args, "fake_model_name", "Wan2.2-TI2V-5B")
        self.generator = CausalWanControlNetWrapper(**getattr(args, "generator_model_kwargs", {}), is_causal=True)
        self.generator.model.requires_grad_(True)
        self.generator.control_model.requires_grad_(True)

        self.vae = WanVAEWrapper()
        self.vae.requires_grad_(False)

        self.text_encoder = WanTextEncoder()
        self.text_encoder.requires_grad_(False)

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

        self.real_score = WanControlNetWrapper(model_name=self.real_model_name, **getattr(args, "model_kwargs", {}), is_causal=False)
        self.real_score.model.requires_grad_(False)
        self.real_score.control_model.requires_grad_(False)

        self.fake_score = WanControlNetWrapper(model_name=self.fake_model_name, **getattr(args, "model_kwargs", {}), is_causal=False)
        self.fake_score.model.requires_grad_(True)
        self.fake_score.control_model.requires_grad_(True)

    def _active_real_score(self):
        """
        The teacher used as the real score in the DMD gradient. Subclasses that
        keep more than one teacher (see CausalWanControlNetDualTeacher) override
        this to route per batch.
        """
        return self.real_score

    def _real_score_debug_suffix(self) -> str:
        """Tag appended to the real-score debug video filename."""
        return ""

    def _compute_kl_grad(
        self, noisy_image_or_video: torch.Tensor,
        estimated_clean_image_or_video: torch.Tensor,
        timestep: torch.Tensor,
        conditional_dict: dict, unconditional_dict: dict,
        normalization: bool = True,
        hint: torch.Tensor = None,
        masked_latent: torch.Tensor = None,
        train_step: int = 0,
        arrow_info: dict = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the KL grad (eq 7 in https://arxiv.org/abs/2311.18828).
        Input:
            - noisy_image_or_video: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - estimated_clean_image_or_video: a tensor with shape [B, F, C, H, W] representing the estimated clean image or video.
            - timestep: a tensor with shape [B, F] containing the randomly generated timestep.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - normalization: a boolean indicating whether to normalize the gradient.
        Output:
            - kl_grad: a tensor representing the KL grad.
            - kl_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        # Step 1: Compute the fake score
        _, pred_fake_image_cond = self.fake_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=timestep,
            hint=hint,
            masked_latent=masked_latent,
        )

        if dist.get_rank() == 0 and self.debug_video and train_step % self.video_save_step == 0:
            videos = pred_fake_image_cond.detach().to(self.device, dtype=self.dtype)
            videos = self.vae.decode_to_pixel(videos, use_cache=False)
            videos = (videos * 0.5 + 0.5).clamp(0, 1)
            videos = rearrange(videos, "b t c h w -> b t h w c")
            videos = videos.data.cpu().numpy()
            videos = self.annotate_video_with_arrow_info(videos, arrow_info)
            video_id = train_step
            write_video(f"{self.debug_video_save_path}/train_generator/fake_score_output_{video_id}.mp4", videos, fps=16)

        pred_fake_image = pred_fake_image_cond

        # Step 2: Compute the real score
        # We compute the conditional and unconditional prediction
        # and add them together to achieve cfg (https://arxiv.org/abs/2207.12598)
        real_score = self._active_real_score()

        _, pred_real_image_cond = real_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=timestep,
            hint=hint,
            masked_latent=masked_latent,
        )

        _, pred_real_image_uncond = real_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=unconditional_dict,
            timestep=timestep,
            hint=hint,
            masked_latent=masked_latent,
        )

        pred_real_image = pred_real_image_cond + (
            pred_real_image_cond - pred_real_image_uncond
        ) * self.real_guidance_scale

        if dist.get_rank() == 0 and self.debug_video and train_step % self.video_save_step == 0:
            videos = pred_real_image.detach().to(self.device, dtype=self.dtype)
            videos = self.vae.decode_to_pixel(videos, use_cache=False)
            videos = (videos * 0.5 + 0.5).clamp(0, 1)
            videos = rearrange(videos, "b t c h w -> b t h w c")
            videos = videos.data.cpu().numpy()
            videos = self.annotate_video_with_arrow_info(videos, arrow_info)
            video_id = train_step
            write_video(f"{self.debug_video_save_path}/train_generator/real_score_output_{video_id}{self._real_score_debug_suffix()}.mp4", videos, fps=16)

        # Step 3: Compute the DMD gradient (DMD paper eq. 7).
        grad = (pred_fake_image - pred_real_image)

        # TODO: Change the normalizer for causal teacher
        if normalization:
            # Step 4: Gradient normalization (DMD paper eq. 8).
            p_real = (estimated_clean_image_or_video - pred_real_image)
            normalizer = torch.abs(p_real).mean(dim=[1, 2, 3, 4], keepdim=True)
            grad = grad / normalizer
        grad = torch.nan_to_num(grad)

        return grad, {
            "dmdtrain_gradient_norm": torch.mean(torch.abs(grad)).detach(),
            "timestep": timestep.detach()
        }

    def compute_distribution_matching_loss(
        self,
        image_or_video: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        gradient_mask: Optional[torch.Tensor] = None,
        denoised_timestep_from: int = 0,
        denoised_timestep_to: int = 0,
        hint: torch.Tensor = None,
        masked_latent: torch.Tensor = None,
        train_step: int = 0,
        arrow_info: dict = None,
        initial_latent: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the DMD loss (eq 7 in https://arxiv.org/abs/2311.18828).
        Input:
            - image_or_video: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - gradient_mask: a boolean tensor with the same shape as image_or_video indicating which pixels to compute loss .
        Output:
            - dmd_loss: a scalar tensor representing the DMD loss.
            - dmd_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        original_latent = image_or_video

        batch_size, num_frame = image_or_video.shape[:2]

        with torch.no_grad():
            # Step 1: Randomly sample timestep based on the given schedule and corresponding noise
            min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
            max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.num_train_timestep
            timestep = self._get_timestep(
                min_timestep,
                max_timestep,
                batch_size,
                num_frame,
                self.num_frame_per_block,
                uniform_timestep=True
            )

            # TODO:should we change it to `timestep = self.scheduler.timesteps[timestep]`?
            if self.timestep_shift > 1:
                timestep = self.timestep_shift * \
                    (timestep / 1000) / \
                    (1 + (self.timestep_shift - 1) * (timestep / 1000)) * 1000
            timestep = timestep.clamp(self.min_step, self.max_step)

            noise = torch.randn_like(image_or_video)
            noisy_latent = self.scheduler.add_noise(
                image_or_video.flatten(0, 1),
                noise.flatten(0, 1),
                timestep.flatten(0, 1)
            ).detach().unflatten(0, (batch_size, num_frame))

            timestep[:, :1] = 0
            noisy_latent[:, :1] = initial_latent[:, :1]

            # Step 2: Compute the KL grad
            grad, dmd_log_dict = self._compute_kl_grad(
                noisy_image_or_video=noisy_latent,
                estimated_clean_image_or_video=original_latent,
                timestep=timestep,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                hint=hint,
                masked_latent=masked_latent,
                train_step=train_step,
                arrow_info=arrow_info,
            )

        if gradient_mask is not None:
            dmd_loss = 0.5 * F.mse_loss(original_latent.double(
            )[gradient_mask], (original_latent.double() - grad.double()).detach()[gradient_mask], reduction="mean")
        else:
            dmd_loss = 0.5 * F.mse_loss(original_latent.double(
            ), (original_latent.double() - grad.double()).detach(), reduction="mean")
        return dmd_loss, dmd_log_dict

    def _run_generator(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        noise: torch.Tensor=None,
        initial_latent: torch.tensor = None,
        hint: torch.tensor = None,
        masked_latent: torch.tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int]:   # needs more modification
        """
        Run the generator.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - noise: a tensor containing the noise [B, F, C, H, W].
            - initial_latent: a tensor containing the initial latents [B, F, C, H, W].
            - hint: a tensor containing the hint [B, F, C, H, W].
            - masked_latent: a tensor containing the masked latent [B, F, C, H, W].
        Output:
            - x0_pred: a tensor containing the predicted x0 [B, F, C, H, W].
            - gradient_mask: a tensor containing the gradient mask [B, F, C, H, W].
            - denoised_timestep_from: an integer representing the starting timestep for denoising.
            - denoised_timestep_to: an integer representing the ending timestep for denoising.
        """
        # Step 1: Sample noise and backward simulate the generator's input
        assert getattr(self.args, "backward_simulation", True), "Backward simulation needs to be enabled"
        if initial_latent is not None:
            conditional_dict["initial_latent"] = initial_latent
        if self.args.i2v:
            noise_shape = [image_or_video_shape[0], image_or_video_shape[1] - 1, *image_or_video_shape[2:]]
        else:
            noise_shape = image_or_video_shape.copy()

        # During training, the number of generated frames should be uniformly sampled from
        # [21, self.num_training_frames], but still being a multiple of self.num_frame_per_block
        min_num_frames = 20 if self.args.independent_first_frame else 21
        max_num_frames = self.num_training_frames - 1 if self.args.independent_first_frame else self.num_training_frames
        assert max_num_frames % self.num_frame_per_block == 0
        assert min_num_frames % self.num_frame_per_block == 0
        max_num_blocks = max_num_frames // self.num_frame_per_block
        min_num_blocks = min_num_frames // self.num_frame_per_block
        num_generated_blocks = torch.randint(min_num_blocks, max_num_blocks + 1, (1,), device=self.device)
        dist.broadcast(num_generated_blocks, src=0)
        num_generated_blocks = num_generated_blocks.item()
        num_generated_frames = num_generated_blocks * self.num_frame_per_block
        if self.args.independent_first_frame and initial_latent is None:
            num_generated_frames += 1
            min_num_frames += 1
        # Sync num_generated_frames across all processes
        noise_shape[1] = num_generated_frames

        if noise is None:
            noise = torch.randn(noise_shape,
                              device=self.device, dtype=self.dtype)
        
        noise = noise[:, :num_generated_frames, ...]

        pred_image_or_video, denoised_timestep_from, denoised_timestep_to, last_timestep, output_flow = self._consistency_backward_simulation(
            noise=noise,
            hint=hint,
            masked_latent=masked_latent,
            **conditional_dict,
        )
        # Slice last 21 frames
        if pred_image_or_video.shape[1] > 21:
            with torch.no_grad():
                # Reencode to get image latent
                latent_to_decode = pred_image_or_video[:, :-20, ...]
                # Deccode to video
                pixels = self.vae.decode_to_pixel(latent_to_decode)
                frame = pixels[:, -1:, ...].to(self.dtype)
                frame = rearrange(frame, "b t c h w -> b c t h w")
                # Encode frame to get image latent
                image_latent = self.vae.encode_to_latent(frame).to(self.dtype)
            pred_image_or_video_last_21 = torch.cat([image_latent, pred_image_or_video[:, -20:, ...]], dim=1)
            # pred_image_or_video_last_21 = pred_image_or_video[:, :21, ...]    # modify slice to directly return the first 21 frames
        else:
            pred_image_or_video_last_21 = pred_image_or_video

        if num_generated_frames != min_num_frames:
            # Currently, we do not use gradient for the first chunk, since it contains image latents
            gradient_mask = torch.ones_like(pred_image_or_video_last_21, dtype=torch.bool)
            if self.args.independent_first_frame:
                gradient_mask[:, :1] = False
            else:
                gradient_mask[:, :self.num_frame_per_block] = False
        else:
            gradient_mask = None

        pred_image_or_video_last_21 = pred_image_or_video_last_21.to(self.dtype)
        return pred_image_or_video_last_21, gradient_mask, denoised_timestep_from, denoised_timestep_to

    def _consistency_backward_simulation(
        self,
        noise: torch.Tensor,
        hint: torch.Tensor = None,
        masked_latent: torch.Tensor = None,
        **conditional_dict: dict,
    ) -> torch.Tensor:
        """
        Simulate the generator's input from noise to avoid training/inference mismatch.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Here we use the consistency sampler (https://arxiv.org/abs/2303.01469)
        Input:
            - noise: a tensor sampled from N(0, 1) with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
        Output:
            - output: a tensor with shape [B, T, F, C, H, W].
            T is the total number of timesteps. output[0] is a pure noise and output[i] and i>0
            represents the x0 prediction at each timestep.
        """
        if self.inference_pipeline is None:
            self._initialize_inference_pipeline()

        return self.inference_pipeline.inference_with_trajectory(
            noise=noise, hint=hint, masked_latent=masked_latent, **conditional_dict
        )

    def _initialize_inference_pipeline(self):
        """
        Lazy initialize the inference pipeline during the first backward simulation run.
        Here we encapsulate the inference code with a model-dependent outside function.
        We pass our FSDP-wrapped modules into the pipeline to save memory.
        """
        self.inference_pipeline = SelfForcingControlCausalTrainingPipeline(
            denoising_step_list=self.denoising_step_list,
            scheduler=self.scheduler,
            generator=self.generator,
            num_frame_per_block=self.num_frame_per_block,
            independent_first_frame=self.args.independent_first_frame,
            same_step_across_blocks=self.args.same_step_across_blocks,
            last_step_only=self.args.last_step_only,
            num_max_frames=self.num_training_frames,
            context_noise=self.args.context_noise,
            first_frame_control=self.first_frame_control,
        )

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
        hint: torch.Tensor = None,
        masked_latent: torch.Tensor = None,
        train_step: int = 0,
        arrow_info: dict = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and compute the DMD loss.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
        """

        noise = torch.randn(image_or_video_shape, device=self.device, dtype=self.dtype)
        batch_size, num_frame = image_or_video_shape[:2]

        full_steps = self.scheduler.timesteps
        # if self.input_clean_latent and clean_latent is not None:
        #     rank = dist.get_rank() if dist.is_initialized() else 0
        #     if rank == 0:
        #         idx = torch.randint(0, full_steps.numel(), (1,), device=self.device)
        #     else:
        #         idx = torch.empty((1,), dtype=torch.long, device=self.device)

        #     dist.broadcast(idx, src=0)
        #     idx = int(idx[0])
        #     self.denoising_step_list_choose = full_steps[idx:]
        #     timestep = self.denoising_step_list_choose[0].repeat(batch_size, num_frame).to(dtype=self.dtype, device=self.device)
        #     noisy_latents = self.scheduler.add_noise(
        #         clean_latent.flatten(0, 1),
        #         noise.flatten(0, 1),
        #         timestep.flatten(0, 1)
        #     ).unflatten(0, (batch_size, num_frame))

        # else:
        self.denoising_step_list_choose = full_steps
        noisy_latents = noise
        x0_pred, gradient_mask, denoised_timestep_from, denoised_timestep_to = self._run_generator(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            noise=noisy_latents,
            hint=hint,
            masked_latent=masked_latent,
            # TODO
            initial_latent=initial_latent, # if self.args.i2v else None,
        )

        assert x0_pred.shape[1] == 21, "x0_pred should have 21 frames"
        
        if dist.get_rank() == 0 and self.debug_video and train_step % self.video_save_step == 0:
            print("save generator debug video train_step: ", train_step)
            # if initial_latent is not None:
            #     videos = x0_pred[:, 1:].detach()
            #     # videos = x0_pred[:, :-1].detach()
            # else:
            #     videos = x0_pred.detach()
            videos = x0_pred.detach()
            videos = self.vae.decode_to_pixel(videos, use_cache=False)
            videos = (videos * 0.5 + 0.5).clamp(0, 1)
            videos = rearrange(videos, "b t c h w -> b t h w c")
            videos = videos.data.cpu().numpy()
            video_id = train_step
            videos = self.annotate_video_with_arrow_info(videos, arrow_info)
            write_video(f"{self.debug_video_save_path}/train_generator/generator_output_{video_id}.mp4", videos, fps=16)
        
        dmd_loss, dmd_log_dict = self.compute_distribution_matching_loss(
            image_or_video=x0_pred,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            gradient_mask=gradient_mask,
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to,
            hint=hint,
            masked_latent=masked_latent,
            train_step=train_step,
            arrow_info=arrow_info,
            initial_latent=initial_latent,
        )
        return dmd_loss, dmd_log_dict

    def critic_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
        hint: torch.Tensor = None,
        masked_latent: torch.Tensor = None,
        train_step: int = 0,
        arrow_info: dict = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the critic loss.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
        """
        with torch.no_grad():
            noise = torch.randn(image_or_video_shape, device=self.device, dtype=self.dtype)
            batch_size, num_frame = image_or_video_shape[:2]

            full_steps = self.scheduler.timesteps
            # if self.input_clean_latent:
            #     rank = dist.get_rank() if dist.is_initialized() else 0
            #     if rank == 0:
            #         idx = torch.randint(0, full_steps.numel(), (1,), device=self.device)
            #     else:
            #         idx = torch.empty((1,), dtype=torch.long, device=self.device)

            #     dist.broadcast(idx, src=0)
            #     idx = int(idx[0])
            #     self.denoising_step_list_choose = full_steps[idx:]
            #     timestep = self.denoising_step_list_choose[0].repeat(batch_size, num_frame).to(dtype=self.dtype, device=self.device)
            #     noisy_latents = self.scheduler.add_noise(
            #         clean_latent.flatten(0, 1),
            #         noise.flatten(0, 1),
            #         timestep.flatten(0, 1)
            #     ).unflatten(0, (batch_size, num_frame))

            #     # keep the 0th frame clean?
            #     if initial_latent is not None:
            #         noisy_latents[:, 0] = initial_latent[:, 0]
            # else:
            self.denoising_step_list_choose = full_steps
            noisy_latents = noise

            x0_pred, gradient_mask, denoised_timestep_from, denoised_timestep_to = self._run_generator(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                noise=noisy_latents,
                hint=hint,
                masked_latent=masked_latent,
                initial_latent=initial_latent, # if self.args.i2v else None,
            )

            if dist.get_rank() == 0 and self.debug_video and train_step % self.video_save_step == 0:
                print("save critic debug video train_step: ", train_step)
                # if initial_latent is not None:
                #     videos = x0_pred[:, 1:].detach()
                #     # videos = x0_pred[:, :-1].detach()
                # else:
                #     videos = x0_pred.detach()
                videos = x0_pred.detach()
                videos = self.vae.decode_to_pixel(videos, use_cache=False)
                videos = (videos * 0.5 + 0.5).clamp(0, 1)
                videos = rearrange(videos, "b t c h w -> b t h w c")
                videos = videos.data.cpu().numpy()
                videos = self.annotate_video_with_arrow_info(videos, arrow_info)
                video_id = train_step
                write_video(f"{self.debug_video_save_path}/generator_output_{video_id}.mp4", videos, fps=16)
        
        min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
        max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.num_train_timestep
        critic_timestep = self._get_timestep(
            min_timestep,
            max_timestep,
            image_or_video_shape[0],
            image_or_video_shape[1],
            self.num_frame_per_block,
            uniform_timestep=True
        )
        
        if self.timestep_shift > 1:
            critic_timestep = self.timestep_shift * \
                (critic_timestep / 1000) / (1 + (self.timestep_shift - 1) * (critic_timestep / 1000)) * 1000

        critic_timestep = critic_timestep.clamp(self.min_step, self.max_step)

        critic_noise = torch.randn_like(x0_pred)
        noisy_x0_pred = self.scheduler.add_noise(
            x0_pred.flatten(0, 1),
            critic_noise.flatten(0, 1),
            critic_timestep.flatten(0, 1)
        ).unflatten(0, image_or_video_shape[:2])

        noisy_x0_pred[:, :1] = initial_latent[:, :1]
        critic_timestep[:, :1] = 0

        _, pred_fake_image = self.fake_score(
            noisy_image_or_video=noisy_x0_pred,
            conditional_dict=conditional_dict,
            timestep=critic_timestep,
            hint=hint,
            masked_latent=masked_latent,
        )

        if dist.get_rank() == 0 and self.debug_video and train_step % self.video_save_step == 0:
            videos = pred_fake_image.detach().to(self.device, dtype=self.dtype)
            videos = self.vae.decode_to_pixel(videos, use_cache=False)
            videos = (videos * 0.5 + 0.5).clamp(0, 1)
            videos = rearrange(videos, "b t c h w -> b t h w c")
            videos = videos.data.cpu().numpy()
            videos = self.annotate_video_with_arrow_info(videos, arrow_info)
            video_id = train_step
            write_video(f"{self.debug_video_save_path}/fake_score_output_{video_id}.mp4", videos, fps=16)

        # Step 3: Compute the denoising loss for the fake critic
        if self.args.denoising_loss_type == "flow":
            from utils.wan_wrapper import WanDiffusionWrapper
            flow_pred = WanDiffusionWrapper._convert_x0_to_flow_pred(
                scheduler=self.scheduler,
                x0_pred=pred_fake_image.flatten(0, 1),
                xt=noisy_x0_pred.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1)
            )
            pred_fake_noise = None
        else:
            flow_pred = None
            pred_fake_noise = self.scheduler.convert_x0_to_noise(
                x0=pred_fake_image.flatten(0, 1),
                xt=noisy_x0_pred.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1)
            ).unflatten(0, image_or_video_shape[:2])

        denoising_loss = self.denoising_loss_func(
            x=x0_pred.flatten(0, 1),
            x_pred=pred_fake_image.flatten(0, 1),
            noise=critic_noise.flatten(0, 1),
            noise_pred=pred_fake_noise,
            alphas_cumprod=self.scheduler.alphas_cumprod,
            timestep=critic_timestep.flatten(0, 1),
            flow_pred=flow_pred
        )

        # Step 5: Debugging Log
        critic_log_dict = {
            "critic_timestep": critic_timestep.detach()
        }

        return denoising_loss, critic_log_dict

    def annotate_video_with_arrow_info(self, videos, arrow_info):
        if arrow_info is not None:
            if "force_1" in arrow_info: # change force situation
                if "x_pos_1" in arrow_info: # point force change situation
                    videos = add_aesthetic_point_force_change_prompt_to_video(
                        video=videos[0],
                        force_1=arrow_info["force_1"],
                        angle_1=arrow_info["angle_1"],
                        x_pos_1=arrow_info["x_pos_1"],
                        y_pos_1=1 -arrow_info["y_pos_1"],
                        force_2=arrow_info["force_2"],
                        angle_2=arrow_info["angle_2"],
                        x_pos_2=arrow_info["x_pos_1"],  # use the same position for the second force
                        y_pos_2=1 -arrow_info["y_pos_1"],
                        idx=videos.shape[1]//2,
                        num_frames_with_signal=videos.shape[1],
                    )
                else: # wind force change situation
                    videos = add_aesthetic_wind_force_change_prompt_to_video(
                        video=videos[0],
                        force_1=arrow_info["force_1"],
                        angle_1=arrow_info["angle_1"],
                        force_2=arrow_info["force_2"],
                        angle_2=arrow_info["angle_2"],
                        idx=videos.shape[1]//2,
                        num_frames_with_signal=videos.shape[1],
                    )
            else:
                if "x_pos" in arrow_info: # point force situation
                    videos = add_aesthetic_point_force_prompt_to_video(
                        video=videos[0],
                        force=arrow_info["force"],
                        angle=arrow_info["angle"],
                        x_pos=arrow_info["x_pos"],
                        y_pos=1 - arrow_info["y_pos"],
                        num_frames_with_signal=videos.shape[1],
                    )
                else: # wind force situation
                    videos = add_aesthetic_wind_force_prompt_to_video(
                        video=videos[0],
                        force=arrow_info["force"],
                        angle=arrow_info["angle"],
                        num_frames_with_signal=videos.shape[1],
                    )
        else:
            videos = (videos[0] * 255.0).astype(np.uint8)
        return videos