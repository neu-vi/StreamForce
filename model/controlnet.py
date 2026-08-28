from typing import Tuple, List, Optional
import torch.distributed as dist
import torch

from model.base import BaseModel
from utils.loss import get_denoising_loss
from utils.wan_wrapper import WanControlNetWrapper, WanTextEncoder, WanVAEWrapper
from wan.utils.utils import masks_like
from tqdm import tqdm

class WanControlNet(BaseModel):
    def __init__(self, args, device):
        super().__init__(args, device)
        self.denoising_loss_func = get_denoising_loss(args.denoising_loss_type)()
        # Noise augmentation in teacher forcing, we add small noise to clean context latents
        self.noise_augmentation_max_timestep = getattr(args, "noise_augmentation_max_timestep", 0)

        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.first_frame_control = getattr(args, "first_frame_control", True)

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        if self.independent_first_frame:
            self.generator.model.independent_first_frame = True
        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()

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

    def _initialize_models(self, args, device):
        self.generator = WanControlNetWrapper(**getattr(args, "model_kwargs", {}), is_causal=False)   # currently on a bidirectional model
        self.generator.control_model.requires_grad_(True)

        self.vae = WanVAEWrapper()
        self.vae.requires_grad_(False)

        self.text_encoder = WanTextEncoder()
        self.text_encoder.requires_grad_(False)

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

        self.inference_scheduler = self.generator.inference_scheduler
        self.inference_scheduler.timesteps = self.inference_scheduler.timesteps.to(device)

    def _run_generator(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        noise: torch.Tensor=None,
        initial_latent: torch.tensor = None,
        hint: torch.tensor = None,
        masked_latent: torch.tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_frame = image_or_video_shape[:2]

        last_timestep = self.denoising_step_list_choose[0]
        timestep = last_timestep.repeat(batch_size, num_frame).to(dtype=self.dtype, device=self.device)
        timestep[: 0] = 0.0 # initial frame is clean

        flow_pred, x0_pred = self.generator(
            noisy_image_or_video=noise,
            conditional_dict=conditional_dict,
            timestep=timestep,
            hint=hint,
            masked_latent=masked_latent,
            # clean_x=initial_latent if self.args.i2v else None,
        )

        denoised_timestep_from, denoised_timestep_to = None, None
        return x0_pred, denoised_timestep_from, denoised_timestep_to, last_timestep, flow_pred

    def inference(
        self,
        image: torch.Tensor,
        input_prompts: List[str],
        num_frames: int = 21,
        return_latents: bool = False,
        hint: Optional[torch.Tensor] = None,
        masked_latent: Optional[torch.Tensor] = None,
        control_weight: float = 1.0,
    ) -> torch.Tensor:
        # this inference code skips the definition of WanControlNetWrapper, and directly utilizes the ControlledWanModel and ControlNet
        model = self.generator.model # ControlledWanModel
        control_model = self.generator.control_model # ControlNet

        # process the image
        image = image.to(self.device)
        image = image.permute(0, 2, 1, 3, 4) # from [B, F, C, H, W] to [B, C, F, H, W], B=F=1
        image = image / 255.0
        image = image * 2.0 - 1.0
        initial_latent = self.vae.encode_to_latent(image).to(device=self.device)
        z = [initial_latent[:, 0].permute(1, 0, 2, 3)]

        # process the prompts
        context = [self.text_encoder(input_prompts)["prompt_embeds"][0]]
        context = [t.to(self.device) for t in context]

        context_null = [self.text_encoder([self.negative_prompt])["prompt_embeds"][0]]
        context_null = [t.to(self.device) for t in context_null]
        
        noise = torch.randn(
            [initial_latent.shape[2], num_frames, initial_latent.shape[3], initial_latent.shape[4]],
            device=self.device,
        )
        latent = noise
        mask1, mask2 = masks_like([noise], zero=True)
        # mask2 = [torch.ones_like(mask2[0])]
        latent = (1.0 - mask2[0]) * z[0] + mask2[0] * latent
        latent = latent.to(device=self.device, dtype=self.dtype)

        # get the sequence length
        seq_len = (num_frames * initial_latent.shape[3] * initial_latent.shape[4]) // (2 * 2)
        
        # setup the scheduler
        self.inference_scheduler.set_timesteps(50, device=self.device, shift=5.0)
        timesteps = self.inference_scheduler.timesteps

        latents_list = [] if return_latents else None
        for t in tqdm(timesteps):
            latent_model_input = latent.unsqueeze(0)
            timestep = [t]
            timestep = torch.stack(timestep).to(model.device)

            temp_ts = (mask2[0][0][:, ::2, ::2] * timestep).flatten()
            temp_ts = torch.cat(
                [temp_ts, temp_ts.new_ones(seq_len - temp_ts.size(0)) * timestep]
            )
            timestep = temp_ts.unsqueeze(0)

            controls = control_model(
                latent_model_input, t=timestep, context=context, seq_len=seq_len, hint=hint, 
                masked_latent=masked_latent,
            )
            controls = [c * control_weight for c in controls]

            noise_pred = model(
                latent_model_input,
                t=timestep,
                context=context,
                seq_len=seq_len,
                control=controls,
            )[0]

            noise_pred_null = model(
                latent_model_input,
                t=timestep,
                context=context_null,
                seq_len=seq_len,
                control=controls,
            )[0]

            noise_pred = noise_pred_null + 5.0 * (noise_pred - noise_pred_null)

            temp_x0 = self.inference_scheduler.step(
                noise_pred.unsqueeze(0),
                t,
                latent.unsqueeze(0),
                return_dict=False,
            )[0]

            latent = temp_x0.squeeze(0)
            latent = (1.0 - mask2[0]) * z[0] + mask2[0] * latent
            latent = latent.to(device=self.device, dtype=self.dtype)

            if return_latents:
                latents_list.append(latent)

        video = self.vae.decode_to_pixel(latent.unsqueeze(0).permute(0, 2, 1, 3, 4))
        video = (video * 0.5 + 0.5).clamp(0, 1)

        if return_latents:
            return video, latents_list
        else:
            return video          

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
        hint: torch.Tensor = None,
        masked_latent: torch.Tensor = None,
        wind_mag: torch.Tensor = None,
        wind_dir: torch.Tensor = None,  
    ) -> Tuple[torch.Tensor, dict]:

        noise = torch.randn_like(clean_latent)
        batch_size, num_frame = image_or_video_shape[:2]

        full_steps = self.scheduler.timesteps
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            idx = torch.randint(0, full_steps.numel(), (1,), device=self.device)
        else:
            idx = torch.empty((1,), dtype=torch.long, device=self.device)

        dist.broadcast(idx, src=0)
        idx = int(idx[0])
        self.denoising_step_list_choose = full_steps[idx:]
        timestep = self.denoising_step_list_choose[0].repeat(batch_size, num_frame).to(dtype=self.dtype, device=self.device)
        noisy_latents = self.scheduler.add_noise(
            clean_latent.flatten(0, 1),
            noise.flatten(0, 1),
            timestep.flatten(0, 1)
        ).unflatten(0, (batch_size, num_frame))

        # keep the 0th frame clean?
        if initial_latent is not None:
            noisy_latents[:, 0] = initial_latent[:, 0]

        x0_pred, denoised_timestep_from, denoised_timestep_to, last_timestep, flow_pred = self._run_generator(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            noise=noisy_latents,
            hint=hint,
            masked_latent=masked_latent,
            # initial_latent=initial_latent if self.args.i2v else None,
        )
        training_target = self.scheduler.training_target(clean_latent, noise, last_timestep)

        flow_pred_actual = flow_pred[:, 1:, ...]

        loss_per_frame = torch.nn.functional.mse_loss(
            flow_pred_actual.float(), training_target[:, 1:, ...].float(), reduction='none'
        ).mean(dim=(2, 3, 4))

        weight = self.scheduler.training_weight(last_timestep.to(dtype=self.dtype, device=self.device)).repeat(batch_size, num_frame - 1)
        loss = (loss_per_frame * weight).mean()

        log_dict = {
            "x0": clean_latent.detach(),
            "x0_pred": x0_pred.detach()
        }
        return loss, log_dict