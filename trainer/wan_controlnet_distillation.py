import gc
import logging

from model import CausalWanControlNet
from utils.forceprompt_data.controlnet_datasets import (
    ForcePromptingDataset_WindForce,
    ForcePromptingDataset_PointForce,
    ForcePromptingDataset_WindForce_ChangeForce,
    ForcePromptingDataset_PointForce_ChangeForce,
)
from utils.forceprompt_data.data_utils import (
    collate_fn_ForcePromptingDataset_WindForce,
    collate_fn_ForcePromptingDataset_PointForce,
    collate_fn_ForcePromptingDataset_WindForce_ChangeForce,
    collate_fn_ForcePromptingDataset_PointForce_ChangeForce,
)
from utils.misc import set_seed, compress_time, cycle
import torch.distributed as dist
from omegaconf import OmegaConf
import torch
import wandb
import time
import os

from utils.distributed import EMA_FSDP, barrier, fsdp_wrap, fsdp_state_dict, launch_distributed_job

class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = config.get("start_step", 0)

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.disable_wandb = config.disable_wandb
        self.all_forces = config.get("all_forces", False)
        self.gpu_memory_efficient = config.get("gpu_memory_efficient", False)
        self.remove_carnation = config.get("remove_carnation", False)
        self.dchange_no_change_ratio = config.get("dchange_no_change_ratio", 1.0)
        self.point_diverse_split_label = config.get("point_diverse_split_label", "train")
        self.wind_diverse_split_label = config.get("wind_diverse_split_label", "train")

        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        if self.is_main_process and not self.disable_wandb:
            wandb.login(host=config.wandb_host, key=config.wandb_key)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                mode="online",
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir
            )

        self.output_path = config.logdir

        self.model = CausalWanControlNet(config, device=self.device)

        # 7. (If resuming) Load the model and optimizer, lr_scheduler, ema's statedicts
        has_critic = False
        if getattr(config, "generator_ckpt", False):
            if dist.get_rank() == 0:
                print(f"[Rank 0] Loading pretrained generator from {config.generator_ckpt}")
                state_dict = torch.load(config.generator_ckpt, map_location="cpu", mmap=True)
                self.model.generator.to_empty(device="cpu")
                self.model.generator.load_state_dict(
                    state_dict["generator"], strict=True
                )
                if "critic" in state_dict:
                    self.model.fake_score.to_empty(device="cpu")
                    self.model.fake_score.load_state_dict(
                        state_dict["critic"], strict=True
                    )
                    has_critic = True
                del state_dict
                if dist.get_rank() == 0:
                    print("[Rank 0] Finished loading generator checkpoint")
            # Barrier after each checkpoint load to keep ranks synchronized
            if dist.is_initialized():
                dist.barrier()
                
        if getattr(config, "real_score_controlnet_model_ckpt_path", False):
            if dist.get_rank() == 0:
                print(f"[Rank 0] Loading pretrained real_score_controlnet from {config.real_score_controlnet_model_ckpt_path}")
                # Only rank 0 reads the .pt; others wait at barrier
                state_dict = torch.load(config.real_score_controlnet_model_ckpt_path, map_location="cpu", mmap=True)
                if "generator_ema" in state_dict:
                    state_dict = state_dict["generator_ema"]
                elif "generator" in state_dict:
                    state_dict = state_dict["generator"]
                elif "model" in state_dict:
                    state_dict = state_dict["model"]
                print("[Rank 0] state_dict load end")
                self.model.real_score.to_empty(device="cpu")
                self.model.real_score.load_state_dict(
                    state_dict, strict=True
                )
                if not has_critic:
                    self.model.fake_score.to_empty(device="cpu")
                    self.model.fake_score.load_state_dict(
                        state_dict, strict=True
                    )
                del state_dict
                print("[Rank 0] Finished loading real_score_controlnet checkpoint")
            # Barrier after each checkpoint load to keep ranks synchronized
            if dist.is_initialized():
                dist.barrier()

        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy,
            sync_module_states=True,
        )

        self.model.real_score = fsdp_wrap(
            self.model.real_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.real_score_fsdp_wrap_strategy,
            sync_module_states=True,
            cpu_offload=True if self.gpu_memory_efficient else False,
        )

        self.model.fake_score = fsdp_wrap(
            self.model.fake_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.fake_score_fsdp_wrap_strategy,
            sync_module_states=True,
            cpu_offload=True if self.gpu_memory_efficient else False,
        )

        if self.gpu_memory_efficient:
            self.model.text_encoder = fsdp_wrap(
                self.model.text_encoder,
                sharding_strategy=config.sharding_strategy,
                mixed_precision=config.mixed_precision,
                wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
                cpu_offload=True,
            )
        else:
            self.model.text_encoder = self.model.text_encoder.to(device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        if not config.no_visualize or config.load_raw_video:
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        self.critic_optimizer = torch.optim.AdamW(  # fake_score
            [param for param in self.model.fake_score.parameters()
             if param.requires_grad],
            lr=config.lr_critic if hasattr(config, "lr_critic") else config.lr,
            betas=(config.beta1_critic, config.beta2_critic),
            weight_decay=config.weight_decay
        )

        # Step 3: Initialize the dataloader
        dataset_point_synthetic = None
        dataset_point_synthetic_change = None
        dataset_point_diverse = None
        dataset_point_diverse_change = None
        dataset_wind_synthetic = None
        dataset_wind_synthetic_change = None
        dataset_wind_diverse = None
        dataset_wind_diverse_change = None

        dataset_point_synthetic = ForcePromptingDataset_PointForce(
            video_root_dir="datasets/point-force/train/point_force_23000",
            csv_path="datasets/point-force/train/point_force_23000.csv",
            image_size=(480, 832),
            stride=(1, 3),
            sample_n_frames=81,
            remove_carnation=self.remove_carnation,
        )
        dataset_point_synthetic_change = ForcePromptingDataset_PointForce_ChangeForce(
            video_root_dir="datasets/point-force-change-force/videos",
            csv_path="datasets/point-force-change-force/point-force-change.csv",
            image_size=(480, 832),
            stride=(1, 3),
            sample_n_frames=81,
            is_validation_dataset=False,
        )
        dataset_point_diverse = ForcePromptingDataset_PointForce(
            video_root_dir="datasets/point-force-diverse/local_force_pexels",
            csv_path="datasets/point-force-diverse/point_force_generated_8K.csv",
            image_size=(480, 832),
            stride=(1, 3),
            sample_n_frames=81,
            is_validation_dataset=True,
            split_label=self.point_diverse_split_label,
        )
        dataset_point_diverse_change = ForcePromptingDataset_PointForce_ChangeForce(
            video_root_dir="datasets/point-force-diverse/local_force_pexels",
            csv_path="datasets/point-force-diverse/point_force_generated_8K_change_force_v1.csv",
            image_size=(480, 832),
            stride=(1, 3),
            sample_n_frames=81,
            is_validation_dataset=True,
            split_label=self.point_diverse_split_label,
        )

        dataset_wind_synthetic = ForcePromptingDataset_WindForce(
            video_root_dir="datasets/wind-force/train/wind_force_15359",
            csv_path="datasets/wind-force/train/wind_force_15359.csv",
            image_size=(480, 832),
            stride=(1, 3),
            sample_n_frames=81,
        )
        dataset_wind_synthetic_change = ForcePromptingDataset_WindForce_ChangeForce(
            video_root_dir="datasets/wind-force-change/wind_force_change_15000",
            csv_path="datasets/wind-force-change/wind_force_change_15000.csv",
            image_size=(480, 832),
            stride=(1, 3),
            sample_n_frames=81,
            is_validation_dataset=False,
        )
        dataset_wind_diverse = ForcePromptingDataset_WindForce(
            video_root_dir="datasets/wind-force-diverse-16K-filtered_5835/after_qwen_filtered_cropped",
            csv_path="datasets/wind-force-diverse-16K-filtered_5835/image_augment_wind.csv",
            image_size=(480, 832),
            stride=(1, 3),
            sample_n_frames=81,
            is_validation_dataset=True,
            split_label=self.wind_diverse_split_label,
        )
        dataset_wind_diverse_change = ForcePromptingDataset_WindForce_ChangeForce(
            video_root_dir="datasets/wind-force-diverse-16K-filtered_5835/after_qwen_filtered_cropped",
            csv_path="datasets/wind-force-diverse-16K-filtered_5835/image_augment_wind_change.csv",
            image_size=(480, 832),
            stride=(1, 3),
            sample_n_frames=81,
            is_validation_dataset=True,
            split_label=self.wind_diverse_split_label,
        )

        self.dataset_point_synthetic = None
        self.dataset_point_synthetic_change = None
        self.dataset_point_diverse = None
        self.dataset_point_diverse_change = None
        self.dataset_wind_synthetic = None
        self.dataset_wind_synthetic_change = None
        self.dataset_wind_diverse = None
        self.dataset_wind_diverse_change = None
        self.dataloader_point_synthetic = None
        self.dataloader_point_synthetic_change = None
        self.dataloader_point_diverse = None
        self.dataloader_point_diverse_change = None
        self.dataloader_wind_synthetic = None
        self.dataloader_wind_synthetic_change = None
        self.dataloader_wind_diverse = None
        self.dataloader_wind_diverse_change = None

        def _get_collate_fn(dataset_obj):
            if isinstance(dataset_obj, ForcePromptingDataset_PointForce_ChangeForce):
                return collate_fn_ForcePromptingDataset_PointForce_ChangeForce
            if isinstance(dataset_obj, ForcePromptingDataset_PointForce):
                return collate_fn_ForcePromptingDataset_PointForce
            if isinstance(dataset_obj, ForcePromptingDataset_WindForce_ChangeForce):
                return collate_fn_ForcePromptingDataset_WindForce_ChangeForce
            return collate_fn_ForcePromptingDataset_WindForce

        def _build_dataloader(dataset_obj):
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset_obj, shuffle=True, drop_last=True
            )
            return cycle(
                torch.utils.data.DataLoader(
                    dataset_obj,
                    batch_size=config.batch_size,
                    sampler=sampler,
                    num_workers=1,
                    collate_fn=_get_collate_fn(dataset_obj),
                )
            )

        if dataset_point_synthetic is not None:
            self.dataset_point_synthetic = dataset_point_synthetic
            self.dataloader_point_synthetic = _build_dataloader(dataset_point_synthetic)
        if dataset_point_synthetic_change is not None:
            self.dataset_point_synthetic_change = dataset_point_synthetic_change
            self.dataloader_point_synthetic_change = _build_dataloader(dataset_point_synthetic_change)
        if dataset_point_diverse is not None:
            self.dataset_point_diverse = dataset_point_diverse
            self.dataloader_point_diverse = _build_dataloader(dataset_point_diverse)
        if dataset_point_diverse_change is not None:
            self.dataset_point_diverse_change = dataset_point_diverse_change
            self.dataloader_point_diverse_change = _build_dataloader(dataset_point_diverse_change)
        if dataset_wind_synthetic is not None:
            self.dataset_wind_synthetic = dataset_wind_synthetic
            self.dataloader_wind_synthetic = _build_dataloader(dataset_wind_synthetic)
        if dataset_wind_synthetic_change is not None:
            self.dataset_wind_synthetic_change = dataset_wind_synthetic_change
            self.dataloader_wind_synthetic_change = _build_dataloader(dataset_wind_synthetic_change)
        if dataset_wind_diverse is not None:
            self.dataset_wind_diverse = dataset_wind_diverse
            self.dataloader_wind_diverse = _build_dataloader(dataset_wind_diverse)
        if dataset_wind_diverse_change is not None:
            self.dataset_wind_diverse_change = dataset_wind_diverse_change
            self.dataloader_wind_diverse_change = _build_dataloader(dataset_wind_diverse_change)

        if dist.is_initialized():
            dist.barrier()

        if dist.get_rank() == 0:
            if dataset_point_synthetic is not None:
                print("DATASET POINT SYNTHETIC SIZE %d" % len(dataset_point_synthetic))
            if dataset_point_synthetic_change is not None:
                print("DATASET POINT SYNTHETIC CHANGE SIZE %d" % len(dataset_point_synthetic_change))
            if dataset_point_diverse is not None:
                print("DATASET POINT DIVERSE SIZE %d" % len(dataset_point_diverse))
            if dataset_point_diverse_change is not None:
                print("DATASET POINT DIVERSE CHANGE SIZE %d" % len(dataset_point_diverse_change))
            if dataset_wind_synthetic is not None:
                print("DATASET WIND SYNTHETIC SIZE %d" % len(dataset_wind_synthetic))
            if dataset_wind_synthetic_change is not None:
                print("DATASET WIND SYNTHETIC CHANGE SIZE %d" % len(dataset_wind_synthetic_change))
            if dataset_wind_diverse is not None:
                print("DATASET WIND DIVERSE SIZE %d" % len(dataset_wind_diverse))
            if dataset_wind_diverse_change is not None:
                print("DATASET WIND DIVERSE CHANGE SIZE %d" % len(dataset_wind_diverse_change))

        self.force_dataloader_map = {
            "diverse": {
                "point": {
                    "no_change": self.dataloader_point_diverse,
                    "change": self.dataloader_point_diverse_change,
                },
                "wind": {
                    "no_change": self.dataloader_wind_diverse,
                    "change": self.dataloader_wind_diverse_change,
                },
            },
            "synthetic": {
                "point": {
                    "no_change": self.dataloader_point_synthetic,
                    "change": self.dataloader_point_synthetic_change,
                },
                "wind": {
                    "no_change": self.dataloader_wind_synthetic,
                    "change": self.dataloader_wind_synthetic_change,
                },
            },
        }
        self.force_dataset_map = {
            "diverse": {
                "point": {
                    "no_change": self.dataset_point_diverse,
                    "change": self.dataset_point_diverse_change,
                },
                "wind": {
                    "no_change": self.dataset_wind_diverse,
                    "change": self.dataset_wind_diverse_change,
                },
            },
            "synthetic": {
                "point": {
                    "no_change": self.dataset_point_synthetic,
                    "change": self.dataset_point_synthetic_change,
                },
                "wind": {
                    "no_change": self.dataset_wind_synthetic,
                    "change": self.dataset_wind_synthetic_change,
                },
            },
        }

        # Create iterators - with num_workers=0, this should not hang

        ##############################################################################################################
        # 6. Set up EMA parameter containers
        rename_param = (
            lambda name: name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        self.name_to_trainable_params = {}
        for n, p in self.model.generator.named_parameters():
            if not p.requires_grad:
                continue

            renamed_n = rename_param(n)
            self.name_to_trainable_params[renamed_n] = p
        ema_weight = config.ema_weight
        self.generator_ema = None
        if (ema_weight is not None) and (ema_weight > 0.0):
            print(f"Setting up EMA with weight {ema_weight}")
            self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)

        ##############################################################################################################
        if dist.is_initialized():
            dist.barrier()

        # Let's delete EMA params for early steps to save some computes at training and inference
        if self.step < config.ema_start_step:
            self.generator_ema = None

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.previous_time = None

    def save(self):
        print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(
            self.model.generator)

        if self.config.ema_start_step < self.step:
            state_dict = {
                "generator": generator_state_dict,
                "generator_ema": self.generator_ema.state_dict(),
            }
        else:
            state_dict = {
                "generator": generator_state_dict,
            }

        if self.is_main_process:
            os.makedirs(os.path.join(self.output_path,
                        f"checkpoint_model_{self.step:06d}"), exist_ok=True)
            torch.save(state_dict, os.path.join(self.output_path,
                       f"checkpoint_model_{self.step:06d}", "model.pt"))
            print("Model saved to", os.path.join(self.output_path,
                  f"checkpoint_model_{self.step:06d}", "model.pt"))


    def train_one_step(self, batch, train_generator=False, max_force=None, min_force=None):
        self.model.eval()  # prevent any randomness (e.g. dropout)

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get the input
        text_prompts = batch["prompts"]
        hint = batch["controlnet_videos"]
        has_change_fields = ("angle_1" in batch and "force_1" in batch)
        if has_change_fields:
            # condition on the first force; the second is encoded in the hint's time axis
            angle_1 = angle = batch["angle_1"]
            force_1 = force = batch["force_1"]
            angle_2 = batch["angle_2"]
            force_2 = batch["force_2"]
        else:
            angle = batch["angle"]
            force = batch["force"]

        arrow_info = None
        if max_force is not None and min_force is not None:
            if has_change_fields:
                if "x_pos_1" in batch and len(batch["x_pos_1"]) > 0:
                    arrow_info = {
                        "x_pos_1": batch["x_pos_1"][0],
                        "y_pos_1": batch["y_pos_1"][0],
                        "x_pos_2": batch["x_pos_2"][0],
                        "y_pos_2": batch["y_pos_2"][0],
                        "force_1": (force_1[0] - min_force) / (max_force - min_force),
                        "angle_1": angle_1[0],
                        "force_2": (force_2[0] - min_force) / (max_force - min_force),
                        "angle_2": angle_2[0],
                    }
                else:
                    arrow_info = {
                        "force_1": (force_1[0] - min_force) / (max_force - min_force),
                        "angle_1": angle_1[0],
                        "force_2": (force_2[0] - min_force) / (max_force - min_force),
                        "angle_2": angle_2[0],
                    }
            else:
                if "x_pos" in batch and len(batch["x_pos"]) > 0:
                    arrow_info = {
                        "x_pos": batch["x_pos"][0],
                        "y_pos": batch["y_pos"][0],
                        "force": (force[0] - min_force) / (max_force - min_force),
                        "angle": angle[0],
                    }
                else:
                    arrow_info = {
                        "force": (force[0] - min_force) / (max_force - min_force),
                        "angle": angle[0],
                    }
            
        batch_size = len(text_prompts)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            with torch.inference_mode():
                if not self.config.load_raw_video:  # precomputed latent
                    clean_latent = batch["ode_latent"][:, -1].to(
                        device=self.device, dtype=self.dtype)
                elif batch["controlnet_videos"][0] is not None:  # encode raw video to latent
                    with torch.no_grad():
                        frames = batch["videos"].to(
                            device=self.device, dtype=self.dtype)
                        # [batch_size, num_frames, num_channels, height, width]
                        frames = frames.permute(0, 2, 1, 3, 4)
                        frames = frames / 255.0
                        frames = frames * 2.0 - 1.0
                            # Encode the input image as the first latent
                        first_image = frames[:, :, :1]
                        image_noise_sigma = torch.normal(mean=-3.0, std=0.5, size=(1,), device=first_image.device)
                        image_noise_sigma = torch.exp(image_noise_sigma).to(dtype=first_image.dtype)
                        noisy_images = first_image + torch.randn_like(first_image) * image_noise_sigma[:, None, None, None, None]

                        initial_latent = self.model.vae.encode_to_latent(noisy_images).to(device=self.device, dtype=self.dtype)
                        clean_latent = None
                        
                        # The force hint is 4 channels: [blob mask, magnitude, cos, sin].
                        # The mask is frozen at frame 0 and gates both the force field and the
                        # masked first frame that rides alongside it.
                        hint = hint.to(device=self.device, dtype=self.dtype)
                        gaussian_blob = hint[:, 0:1, 0:1]
                        hint = hint[:, :, 1:] * (gaussian_blob > 1e-1)
                        masked_image = (noisy_images + 1.0) / 2.0 * gaussian_blob.permute(0, 2, 1, 3, 4)
                        masked_image = masked_image * 2.0 - 1.0
                        masked_image = masked_image.permute(0, 2, 1, 3, 4)
                        masked_latent = compress_time(masked_image, 81, method="subsample")
                        hint = compress_time(hint, 81, method="subsample")
                    # assert hint.shape[1] == clean_latent.shape[1]   # hint latents and the gt latents should have the same number of frames
                else:
                    hint = None
                    initial_latent = None
                    masked_latent = None
                    clean_latent = None

                image_or_video_shape = list(self.config.image_or_video_shape)
                image_or_video_shape[0] = batch_size

                conditional_dict = self.model.text_encoder(
                    text_prompts=text_prompts)

                if not getattr(self, "unconditional_dict", None):
                    unconditional_dict = self.model.text_encoder(
                        text_prompts=[self.config.negative_prompt] * batch_size)
                    unconditional_dict = {k: v.detach()
                                          for k, v in unconditional_dict.items()}
                    self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
                else:
                    unconditional_dict = self.unconditional_dict

        if train_generator:
            generator_loss, log_dict = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                initial_latent=initial_latent,
                hint=hint,
                masked_latent=masked_latent.clone() if masked_latent is not None else None,
                train_step=self.step,
                arrow_info=arrow_info,
            )

            generator_loss.backward()
            
            generator_grad_norm = self.model.generator.clip_grad_norm_(
                self.max_grad_norm_generator)

            log_dict.update(
                {"generator_loss": generator_loss,
                "generator_grad_norm": generator_grad_norm}
            )
            
            return log_dict

        else:
            critic_loss, log_dict = self.model.critic_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                initial_latent=initial_latent,
                hint=hint,
                masked_latent=masked_latent.clone() if masked_latent is not None else None,
                train_step=self.step,
                arrow_info=arrow_info,
            )

            critic_loss.backward()

            critic_grad_norm = self.model.fake_score.clip_grad_norm_(
                self.max_grad_norm_critic)

            log_dict.update(
                {"critic_loss": critic_loss,
                "critic_grad_norm": critic_grad_norm}
            )
            
            return log_dict

    def _sample_distributed_choice(self, weights):
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            choice = torch.multinomial(weights, 1).to(device=self.device)
        else:
            choice = torch.empty((1,), dtype=torch.long, device=self.device)
        if dist.is_initialized():
            dist.broadcast(choice, src=0)
        return int(choice[0])

    def _source_has_any_loader(self, source_name):
        source_loaders = self.force_dataloader_map[source_name]
        return any(
            source_loaders[force_name][change_name] is not None
            for force_name in source_loaders
            for change_name in source_loaders[force_name]
        )

    def _force_type_has_any_loader(self, source_name, force_name):
        loaders = self.force_dataloader_map[source_name][force_name]
        return loaders["no_change"] is not None or loaders["change"] is not None

    def _select_source_name(self):
        source_names = [
            source_name
            for source_name in self.force_dataloader_map
            if self._source_has_any_loader(source_name)
        ]
        if not source_names:
            raise RuntimeError("No force dataloader is available.")
        if len(source_names) == 1:
            return source_names[0]

        source_weight_map = {
            "diverse": float(getattr(self.config, "diverse_synthetic_data_ratio", 1.0)),
            "synthetic": 1.0,
        }
        data_choice_weights = torch.tensor(
            [source_weight_map.get(source_name, 1.0) for source_name in source_names],
            device=self.device,
        )
        source_choice = self._sample_distributed_choice(data_choice_weights)
        return source_names[source_choice]

    def _select_force_type_name(self, source_name):
        has_point = self._force_type_has_any_loader(source_name, "point")
        has_wind = self._force_type_has_any_loader(source_name, "wind")
        if has_point and has_wind:
            local_global_force_weights = torch.tensor(
                [float(self.config.dlocal_global_force_ratio), 1.0],
                device=self.device,
            )
            force_choice = self._sample_distributed_choice(local_global_force_weights)
            return "point" if force_choice == 0 else "wind"
        if has_point:
            return "point"
        if has_wind:
            return "wind"
        raise RuntimeError(f"No point/wind dataloader is available for source={source_name}.")

    def _select_change_name(self, source_name, force_name):
        force_loaders = self.force_dataloader_map[source_name][force_name]
        has_change = force_loaders["change"] is not None
        has_no_change = force_loaders["no_change"] is not None
        if has_change and has_no_change:
            change_no_change_weights = torch.tensor(
                [float(self.dchange_no_change_ratio), 1.0],
                device=self.device,
            )
            change_choice = self._sample_distributed_choice(change_no_change_weights)
            return "change" if change_choice == 0 else "no_change"
        if has_no_change:
            return "no_change"
        if has_change:
            return "change"
        raise RuntimeError(
            f"No change/no_change dataloader is available for source={source_name}, force={force_name}."
        )

    def _get_next_force_loader_and_dataset(self):
        source_name = self._select_source_name()
        force_name = self._select_force_type_name(source_name)
        change_name = self._select_change_name(source_name, force_name)

        loader = self.force_dataloader_map[source_name][force_name][change_name]
        dataset = self.force_dataset_map[source_name][force_name][change_name]
        if loader is None or dataset is None:
            raise RuntimeError(
                f"Selected force loader/dataset is None: source={source_name}, force={force_name}, change={change_name}."
            )
        return loader, dataset, source_name, force_name, change_name

    def train(self):
        start_step = self.step
        while True:
            (
                dataloader,
                selected_dataset,
                source_name,
                force_name,
                change_name,
            ) = self._get_next_force_loader_and_dataset()
            if self.is_main_process:
                print(f"use {source_name} {force_name} {change_name} force")

            max_force = selected_dataset.max_force
            min_force = selected_dataset.min_force

            if self.is_main_process:
                print("step: ", self.step)
            TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0
            if TRAIN_GENERATOR:
                batch = next(dataloader)
                generator_log_dict = self.train_one_step(batch, train_generator=True, max_force=max_force, min_force=min_force)

                self.generator_optimizer.step()
                self.generator_optimizer.zero_grad(set_to_none=True)
                if self.generator_ema is not None:
                    self.generator_ema.update(self.model.generator)

            batch = next(dataloader)
            critic_log_dict = self.train_one_step(batch, train_generator=False, max_force=max_force, min_force=min_force)

            self.critic_optimizer.step()
            self.critic_optimizer.zero_grad(set_to_none=True)
            # Increment the step since we finished gradient update
            self.step += 1

            # Create EMA params (if not already created)
            if (self.step >= self.config.ema_start_step) and \
                    (self.generator_ema is None) and (self.config.ema_weight > 0):
                self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)

            # Save the model
            if (not self.config.no_save) and (self.step - start_step) > 0 and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            # Logging
            if self.is_main_process:
                wandb_loss_dict = {}
                if TRAIN_GENERATOR:
                    wandb_loss_dict.update(
                        {
                            "generator_loss": generator_log_dict["generator_loss"].item(),
                            "generator_grad_norm": generator_log_dict["generator_grad_norm"].item(),
                            "dmdtrain_gradient_norm": generator_log_dict["dmdtrain_gradient_norm"].item()
                        }
                    )

                wandb_loss_dict.update(
                    {
                        "critic_loss": critic_log_dict["critic_loss"].item(),
                        "critic_grad_norm": critic_log_dict["critic_grad_norm"].item()
                    }
                )

                if not self.disable_wandb:
                    wandb.log(wandb_loss_dict, step=self.step)
                print(wandb_loss_dict)

            if self.step % self.config.gc_interval == 0:
                if dist.get_rank() == 0:
                    logging.info("DistGarbageCollector: Running GC.")
                gc.collect()
                torch.cuda.empty_cache()

            barrier()
            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    print("per iteration time: ", current_time - self.previous_time)
                    self.previous_time = current_time
            
