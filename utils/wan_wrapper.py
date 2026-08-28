import types
from typing import List, Optional
import torch
from torch import nn

from utils.scheduler import SchedulerInterface, FlowMatchScheduler
from wan.modules.tokenizers import HuggingfaceTokenizer
from wan.modules.model import ControlNet, ControlledWanModel
from wan.modules.vae2_2 import _video_vae
from wan.modules.t5 import umt5_xxl
from wan.modules.causal_model import CausalControlNet, CausalControlledWanModel
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from accelerate import init_empty_weights
import inspect

def load_cfg_as_dict(cfg):
    # Handle HF / OmegaConf / dataclass / plain dict
    if hasattr(cfg, "to_dict"):
        return cfg.to_dict()
    try:
        # OmegaConf
        from omegaconf import OmegaConf
        if OmegaConf.is_config(cfg):
            return OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        pass
    if hasattr(cfg, "__dict__"):
        return dict(cfg.__dict__)
    assert isinstance(cfg, dict), "Unsupported config type"
    return cfg


def filter_kwargs_for_ctor(cls, kwargs):
    sig = inspect.signature(cls.__init__)
    allowed = set(p.name for p in sig.parameters.values() if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY))
    return {k: v for k, v in kwargs.items() if k in allowed}

class WanTextEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.text_encoder = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=torch.float32,
            device=torch.device('cpu')
        ).eval().requires_grad_(False)
        self.text_encoder.load_state_dict(
            torch.load("wan_models/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth",
                       map_location='cpu', weights_only=False)
        )

        self.tokenizer = HuggingfaceTokenizer(
            name="wan_models/Wan2.2-TI2V-5B/google/umt5-xxl/", seq_len=512, clean='whitespace')

    @property
    def device(self):
        # Use module parameter device to support multi-GPU placement.
        try:
            return next(self.text_encoder.parameters()).device
        except StopIteration:
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, text_prompts: List[str]) -> dict:
        ids, mask = self.tokenizer(
            text_prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)

        for u, v in zip(context, seq_lens):
            u[v:] = 0.0  # set padding to 0.0

        return {
            "prompt_embeds": context
        }

class WanVAEWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        mean = torch.tensor(
            [
                -0.2289,
                -0.0052,
                -0.1323,
                -0.2339,
                -0.2799,
                0.0174,
                0.1838,
                0.1557,
                -0.1382,
                0.0542,
                0.2813,
                0.0891,
                0.1570,
                -0.0098,
                0.0375,
                -0.1825,
                -0.2246,
                -0.1207,
                -0.0698,
                0.5109,
                0.2665,
                -0.2108,
                -0.2158,
                0.2502,
                -0.2055,
                -0.0322,
                0.1109,
                0.1567,
                -0.0729,
                0.0899,
                -0.2799,
                -0.1230,
                -0.0313,
                -0.1649,
                0.0117,
                0.0723,
                -0.2839,
                -0.2083,
                -0.0520,
                0.3748,
                0.0152,
                0.1957,
                0.1433,
                -0.2944,
                0.3573,
                -0.0548,
                -0.1681,
                -0.0667,
            ],
        )
        std = torch.tensor(
            [
                0.4765,
                1.0364,
                0.4514,
                1.1677,
                0.5313,
                0.4990,
                0.4818,
                0.5013,
                0.8158,
                1.0344,
                0.5894,
                1.0901,
                0.6885,
                0.6165,
                0.8454,
                0.4978,
                0.5759,
                0.3523,
                0.7135,
                0.6804,
                0.5833,
                1.4146,
                0.8986,
                0.5659,
                0.7069,
                0.5338,
                0.4889,
                0.4917,
                0.4069,
                0.4999,
                0.6866,
                0.4093,
                0.5709,
                0.6065,
                0.6415,
                0.4944,
                0.5726,
                1.2042,
                0.5458,
                1.6887,
                0.3971,
                1.0600,
                0.3943,
                0.5537,
                0.5444,
                0.4089,
                0.7468,
                0.7744,
            ],
        )
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

        # init model
        self.model = _video_vae(
            pretrained_path="wan_models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth",
            z_dim=48,
            temperal_downsample=[False, True, True],
        ).eval().requires_grad_(False)

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        # pixel: [batch_size, num_channels, num_frames, height, width]
        device, dtype = pixel.device, pixel.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        output = [
            self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
            for u in pixel
        ]
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def decode_to_pixel(self, latent: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        # from [batch_size, num_frames, num_channels, height, width]
        # to [batch_size, num_channels, num_frames, height, width]
        zs = latent.permute(0, 2, 1, 3, 4)
        if use_cache:
            assert latent.shape[0] == 1, "Batch size must be 1 when using cache"

        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        if use_cache:
            decode_function = self.model.cached_decode
        else:
            decode_function = self.model.decode

        output = []
        for u in zs:
            output.append(decode_function(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0))
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output

class WanDiffusionWrapper(torch.nn.Module):
    def __init__(
            self,
            model_name="Wan2.1-T2V-1.3B",
            timestep_shift=8.0,
            is_causal=False,
            local_attn_size=-1,
            sink_size=0,
    ):
        super().__init__()

        # Subclasses build their own towers; this base only owns the scheduler.
        # For non-causal diffusion, all frames share the same timestep
        self.uniform_timestep = not is_causal

        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)

        self.seq_len = 32760  # [1, 21, 16, 60, 104]
        self.post_init()

    def enable_gradient_checkpointing(self) -> None:
        self.model.enable_gradient_checkpointing()

    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        """
        # use higher precision for calculations
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device), [flow_pred, xt,
                                                        self.scheduler.sigmas,
                                                        self.scheduler.timesteps]
        )

        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    @staticmethod
    def _convert_x0_to_flow_pred(scheduler, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert x0 prediction to flow matching's prediction.
        x0_pred: the x0 prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = (x_t - x_0) / sigma_t
        """
        # use higher precision for calculations
        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(x0_pred.device), [x0_pred, xt,
                                                      scheduler.sigmas,
                                                      scheduler.timesteps]
        )
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)


    def get_scheduler(self) -> SchedulerInterface:
        """
        Update the current scheduler with the interface's static method
        """
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise, scheduler)
        scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0, scheduler)
        scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0, scheduler)
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        """
        A few custom initialization steps that should be called after the object is created.
        Currently, the only one we have is to bind a few methods to scheduler.
        We can gradually add more methods here if needed.
        """
        self.get_scheduler()

class WanControlNetWrapper(WanDiffusionWrapper):
    def __init__(
            self,             
            model_name="Wan2.2-TI2V-5B",
            timestep_shift=8.0,
            is_causal=False,
            local_attn_size=-1,
            sink_size=0,
            dropout=0.0,
            controlnet_layers=None,
            control_weight=1.0,
            model_ckpt=None,
            training=True,
    ):
        super().__init__(
            model_name=model_name,
            timestep_shift=timestep_shift,
            is_causal=is_causal,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
        )
        self.model = ControlledWanModel.from_pretrained(
                f"wan_models/{model_name}/", local_attn_size=local_attn_size, sink_size=sink_size)

        self.control_model = ControlNet.from_pretrained(
                f"wan_models/{model_name}/", ignore_mismatched_sizes=True, local_attn_size=local_attn_size, sink_size=sink_size, strict=False, low_cpu_mem_usage=False, device_map=None, dropout=dropout, controlnet_layers=controlnet_layers)
        self.control_weight = control_weight
        self._init_zeros()

        if model_ckpt is not None and training:
            print(f"Loading pretrained generator from {model_ckpt}")
            state_dict = torch.load(model_ckpt, map_location="cpu")
            if "generator_ema" in state_dict:
                state_dict = state_dict["generator_ema"]
            elif "generator" in state_dict:
                state_dict = state_dict["generator"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            # self.model.generator.load_state_dict(state_dict, strict=True)
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace("model.", "")
                new_state_dict[new_key] = value
            del state_dict

            self.model.load_state_dict(
                new_state_dict, strict=True
            )
            self.control_model.load_state_dict(
                new_state_dict, strict=False,
            )
            del new_state_dict

        # use wan2.2 standard scheduler
        self.inference_scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=1000,
            shift=1,
            use_dynamic_shifting=False
        )

        # self.seq_len = 1911 # [1, 21, 48, 14, 26]
        self.seq_len = 8190

    def _init_zeros(self):
        nn.init.xavier_uniform_(self.control_model.input_hint_block[0].weight)
        nn.init.xavier_uniform_(self.control_model.input_hint_block[3].weight)
        nn.init.constant_(self.control_model.input_hint_block[0].bias, 0)
        nn.init.constant_(self.control_model.input_hint_block[3].bias, 0)
        for n, p in self.control_model.named_parameters():
            if "zero" in n:
                nn.init.constant_(p, 0)

    def enable_gradient_checkpointing(self) -> None:
        self.model.enable_gradient_checkpointing()
        self.control_model.enable_gradient_checkpointing()

    def generate_control(
        self, 
        prompt_embeds: torch.Tensor,
        input_timestep: torch.Tensor,
        noisy_image_or_video: torch.Tensor,
        timestep: torch.Tensor, kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        clean_x: Optional[torch.Tensor] = None,
        aug_t: Optional[torch.Tensor] = None,
        cache_start: Optional[int] = None,
        hint: Optional[torch.Tensor] = None,
        masked_latent: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # X0 prediction
        # modify back to the input format to match Wan2.2 WanModel standard input format
        x = noisy_image_or_video.permute(0, 2, 1, 3, 4)
        controls = self.control_model(
            x=x,
            t=input_timestep, 
            context=prompt_embeds,
            seq_len=self.seq_len,
            hint=hint,
            masked_latent=masked_latent,
        )

        return controls

    def forward(
        self,
        noisy_image_or_video: torch.Tensor, conditional_dict: dict, 
        timestep: torch.Tensor, kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        clean_x: Optional[torch.Tensor] = None,
        aug_t: Optional[torch.Tensor] = None,
        cache_start: Optional[int] = None,
        hint: Optional[torch.Tensor] = None,
        masked_latent: Optional[torch.Tensor] = None,
        scheduler_override: Optional[SchedulerInterface] = None,
    ) -> torch.Tensor:
        prompt_embeds = conditional_dict["prompt_embeds"]

        # [B, F] -> [B]
        if self.uniform_timestep:
            input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep

        input_timestep = timestep
        control = None
        if hint is not None:
            controls = self.generate_control(
                prompt_embeds=prompt_embeds,
                input_timestep=input_timestep,
                noisy_image_or_video=noisy_image_or_video,
                timestep=timestep,
                hint=hint,
                masked_latent=masked_latent,
                # clean_x=clean_x,
            )
            scale = self.control_weight
            control = [c * scale for c in controls]

        # modify back to the input format to match Wan2.2 WanModel standard input format
        x = noisy_image_or_video.permute(0, 2, 1, 3, 4)
        flow_pred = self.model(
            x=x,
            t=input_timestep,
            context=prompt_embeds,
            seq_len=self.seq_len,
            control=control,
            # y=clean_x.permute(0, 2, 1, 3, 4) if clean_x is not None else None,
        )[0]
        flow_pred = flow_pred.unsqueeze(0).permute(0, 2, 1, 3, 4)

        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1),
        ).unflatten(0, flow_pred.shape[:2])

        return flow_pred, pred_x0

    def get_scheduler(self) -> SchedulerInterface:
        """
        Update the current scheduler with the interface's static method
        """
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise, scheduler)
        scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0, scheduler)
        scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0, scheduler)
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        """
        A few custom initialization steps that should be called after the object is created.
        Currently, the only one we have is to bind a few methods to scheduler.
        We can gradually add more methods here if needed.
        """
        self.get_scheduler()

class CausalWanControlNetWrapper(WanDiffusionWrapper):
    def __init__(
            self,             
            model_name="Wan2.2-TI2V-5B",
            timestep_shift=8.0,
            is_causal=True,
            local_attn_size=-1,
            sink_size=0,
            dropout=0.0,
            controlnet_layers=None,
            control_weight=1.0,
            model_ckpt=None,
            training=True,
            meta_init=True,
            rolling_forcing_attention=False,
            rolling_forcing_block_frames=3,
            rolling_forcing_max_frames=21,
    ):
        assert is_causal, "CausalWanControlNetWrapper should be used with is_causal=True"
        super().__init__(
            model_name=model_name,
            timestep_shift=timestep_shift,
            is_causal=is_causal,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
        )
        if meta_init:
            extra_kwargs = dict(
                local_attn_size=local_attn_size,
                sink_size=sink_size,
                rolling_forcing_attention=rolling_forcing_attention,
                rolling_forcing_block_frames=rolling_forcing_block_frames,
                rolling_forcing_max_frames=rolling_forcing_max_frames,
            )

            # Build on 'meta' so we don't allocate memory yet
            with init_empty_weights():
                base_cfg = CausalControlledWanModel.load_config(f"wan_models/{model_name}")
                cfg = load_cfg_as_dict(base_cfg)
                cfg.update(filter_kwargs_for_ctor(CausalControlledWanModel, extra_kwargs))
                self.model = CausalControlledWanModel.from_config(cfg)

            with init_empty_weights():
                base_cfg = CausalControlNet.load_config(f"wan_models/{model_name}")
                cfg = load_cfg_as_dict(base_cfg)
                # controlnet-specific extras you want passed to __init__
                control_extras = dict(
                    local_attn_size=local_attn_size,
                    sink_size=sink_size,
                    rolling_forcing_attention=rolling_forcing_attention,
                    rolling_forcing_block_frames=rolling_forcing_block_frames,
                    rolling_forcing_max_frames=rolling_forcing_max_frames,
                    dropout=dropout,
                    controlnet_layers=controlnet_layers,
                )
                cfg.update(filter_kwargs_for_ctor(CausalControlNet, control_extras))
                self.control_model = CausalControlNet.from_config(cfg)

            self.control_weight = control_weight
        else:
            self.model = CausalControlledWanModel.from_pretrained(
                    f"wan_models/{model_name}/",
                    local_attn_size=local_attn_size,
                    sink_size=sink_size,
                    rolling_forcing_attention=rolling_forcing_attention,
                    rolling_forcing_block_frames=rolling_forcing_block_frames,
                    rolling_forcing_max_frames=rolling_forcing_max_frames,
            )

            self.control_model = CausalControlNet.from_pretrained(
                    f"wan_models/{model_name}/", ignore_mismatched_sizes=True, local_attn_size=local_attn_size, sink_size=sink_size, rolling_forcing_attention=rolling_forcing_attention, rolling_forcing_block_frames=rolling_forcing_block_frames, rolling_forcing_max_frames=rolling_forcing_max_frames, strict=False, low_cpu_mem_usage=False, device_map=None, dropout=dropout, controlnet_layers=controlnet_layers)
            self.control_weight = control_weight
            self._init_zeros()

    def _init_zeros(self):
        nn.init.xavier_uniform_(self.control_model.input_hint_block[0].weight)
        nn.init.xavier_uniform_(self.control_model.input_hint_block[3].weight)
        nn.init.constant_(self.control_model.input_hint_block[0].bias, 0)
        nn.init.constant_(self.control_model.input_hint_block[3].bias, 0)
        for n, p in self.control_model.named_parameters():
            if "zero" in n:
                nn.init.constant_(p, 0)

    def enable_gradient_checkpointing(self) -> None:
        self.model.enable_gradient_checkpointing()
        self.control_model.enable_gradient_checkpointing()

    def generate_control(
        self, 
        prompt_embeds: torch.Tensor,
        input_timestep: torch.Tensor,
        noisy_image_or_video: torch.Tensor,
        timestep: torch.Tensor, kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        clean_x: Optional[torch.Tensor] = None,
        aug_t: Optional[torch.Tensor] = None,
        cache_start: Optional[int] = None,
        hint: Optional[torch.Tensor] = None,
        masked_latent: Optional[torch.Tensor] = None,
        updating_cache: Optional[bool] = False,
    ) -> torch.Tensor:
        # X0 prediction
        # modify back to the input format to match Wan2.2 WanModel standard input format
        # x = [noisy_image_or_video.permute(0, 2, 1, 3, 4).squeeze(0)]
        x = noisy_image_or_video.permute(0, 2, 1, 3, 4)
        if kv_cache is not None:
            controls = self.control_model(
                x=x,
                t=input_timestep, context=prompt_embeds,
                seq_len=self.seq_len,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                cache_start=cache_start,
                hint=hint,
                masked_latent=masked_latent,
                updating_cache=updating_cache,
            )
        else:
            if clean_x is not None:
                # teacher forcing
                controls = self.control_model(
                    x=x,
                    t=input_timestep, context=prompt_embeds,
                    seq_len=self.seq_len,
                    clean_x=clean_x.permute(0, 2, 1, 3, 4),
                    aug_t=aug_t,
                    hint=hint,
                    masked_latent=masked_latent,
                )
            else:
                controls = self.control_model(
                    x=x,
                    t=input_timestep, context=prompt_embeds,
                    seq_len=self.seq_len,
                    hint=hint,
                    masked_latent=masked_latent,
                )

        return controls

    def forward(
        self,
        noisy_image_or_video: torch.Tensor, conditional_dict: dict,
        timestep: torch.Tensor, kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        kv_cache2: Optional[List[dict]] = None,
        crossattn_cache2: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        clean_x: Optional[torch.Tensor] = None,
        aug_t: Optional[torch.Tensor] = None,
        cache_start: Optional[int] = None,
        hint: Optional[torch.Tensor] = None,
        masked_latent: Optional[torch.Tensor] = None,
        updating_cache: Optional[bool] = False,
    ) -> torch.Tensor:
        prompt_embeds = conditional_dict["prompt_embeds"]

        # [B, F] -> [B]
        if self.uniform_timestep:
            input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep

        control = None
        if hint is not None:
            control = self.generate_control(
                prompt_embeds, 
                input_timestep,
                noisy_image_or_video,
                timestep,
                kv_cache2,
                crossattn_cache2,
                current_start,
                clean_x,
                aug_t,
                cache_start,
                hint,
                masked_latent,
                updating_cache,
            )
            scale = self.control_weight
            control = [c * scale for c in control]
        # try:
        #     torch.cuda.memory._dump_snapshot(f"mem_record.pickle")
        # except Exception as e:
        #     print(f"Failed to capture memory snapshot {e}")
        # X0 prediction
        # modify back to the input format to match Wan2.2 WanModel standard input format
        # x = [noisy_image_or_video.permute(0, 2, 1, 3, 4).squeeze(0)]
        x = noisy_image_or_video.permute(0, 2, 1, 3, 4)
        if kv_cache is not None:
            flow_pred = self.model(
                x=x,
                t=input_timestep, context=prompt_embeds,
                seq_len=self.seq_len,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                cache_start=cache_start,
                control=control,
                updating_cache=updating_cache,
            ).permute(0, 2, 1, 3, 4)
        else:
            if clean_x is not None:
                # teacher forcing
                flow_pred = self.model(
                    x=x,
                    t=input_timestep, context=prompt_embeds,
                    seq_len=self.seq_len,
                    clean_x=clean_x.permute(0, 2, 1, 3, 4),
                    aug_t=aug_t,
                    control=control,
                ).permute(0, 2, 1, 3, 4)
            else:
                flow_pred = self.model(
                    x=x,
                    t=input_timestep, context=prompt_embeds,
                    seq_len=self.seq_len,
                    control=control,
                ).permute(0, 2, 1, 3, 4)

        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1)
        ).unflatten(0, flow_pred.shape[:2])

        return flow_pred, pred_x0

    def get_scheduler(self) -> SchedulerInterface:
        """
        Update the current scheduler with the interface's static method
        """
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise, scheduler)
        scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0, scheduler)
        scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0, scheduler)
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        """
        A few custom initialization steps that should be called after the object is created.
        Currently, the only one we have is to bind a few methods to scheduler.
        We can gradually add more methods here if needed.
        """
        self.get_scheduler()
