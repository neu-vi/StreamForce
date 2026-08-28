import gc
import logging

from model import WanControlNet
from utils.forceprompt_data.controlnet_datasets import (
    ForcePromptingDataset_WindForce,
    ForcePromptingDataset_PointForce,
    ForcePromptingDataset_WindForce_ChangeForce,
    ForcePromptingDataset_PointForce_ChangeForce,
)
from utils.forceprompt_data.data_utils import (
    collate_fn_ForcePromptingDataset_WindForce,
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
from einops import rearrange
import numpy as np
from torchvision.io import write_video

from utils.distributed import EMA_FSDP, barrier, fsdp_wrap, fsdp_state_dict, launch_distributed_job

class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.disable_wandb = config.disable_wandb
        self.use_change_force = config.get("use_change_force", False)
        self.remove_carnation = config.get("remove_carnation", False)
        self.dlocal_global_force_ratio = config.get("dlocal_global_force_ratio", 1) # local:global = self.dlocal_global_force_ratio:1
        self.dchange_no_change_ratio = config.get("dchange_no_change_ratio", 1)
        self.randomize_change_point = config.get("randomize_change_point", False)
        self.min_change_ratio = config.get("min_change_ratio", 0.3)
        self.max_change_ratio = config.get("max_change_ratio", 0.7)

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

        # Step 2: Initialize the model and optimizer
        self.model = WanControlNet(config, device=self.device)
        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy
        )
        
        if not config.no_visualize or config.load_raw_video:
            self.model.text_encoder = self.model.text_encoder.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)
        
        trainable_param = list(filter(lambda p: p.requires_grad, self.model.generator.control_model.parameters()))

        self.generator_optimizer = torch.optim.AdamW(
            trainable_param,
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        # Step 3: Initialize the dataloader
        point_video_root_dir = config.get("point_video_root_dir", "datasets/point-force/train/point_force_23000")
        point_csv_path = config.get("point_csv_path", "datasets/point-force/train/point_force_23000.csv")
        wind_video_root_dir = config.get("wind_video_root_dir", "datasets/wind-force-remove-pole/wind_force_remove_pole_14999")
        wind_csv_path = config.get("wind_csv_path", "datasets/wind-force-remove-pole/wind_force_remove_pole_14999.csv")
        point_change_video_root_dir = config.get("point_change_video_root_dir", "datasets/point-force-change-force/videos")
        point_change_csv_path = config.get("point_change_csv_path", "datasets/point-force-change-force/point-force-change.csv")
        wind_change_video_root_dir = config.get("wind_change_video_root_dir", "datasets/wind-force-change/wind_force_change_15000")
        wind_change_csv_path = config.get("wind_change_csv_path", "datasets/wind-force-change/wind_force_change_15000.csv")

        point_dataset = ForcePromptingDataset_PointForce(
            video_root_dir=point_video_root_dir,
            csv_path=point_csv_path,
            image_size=(224, 416),
            stride=(1, 3),
            sample_n_frames=81,
            remove_carnation=self.remove_carnation,
        )
        wind_dataset = ForcePromptingDataset_WindForce(
            video_root_dir=wind_video_root_dir,
            csv_path=wind_csv_path,
            # video_root_dir="datasets/wind-force/train/wind_force_15359",
            # csv_path="datasets/wind-force/train/wind_force_15359.csv",
            image_size=(224, 416),
            stride=(1, 3),
            sample_n_frames=81,
        )
        # dataset = torch.utils.data.ConcatDataset([point_dataset, wind_dataset])
        print("all dataset!!!!!!")
        if self.use_change_force:
            point_change_dataset = ForcePromptingDataset_PointForce_ChangeForce(
                video_root_dir=point_change_video_root_dir,
                csv_path=point_change_csv_path,
                image_size=(224, 416),
                stride=(1, 3),
                sample_n_frames=81,
                randomize_change_point=self.randomize_change_point,
                min_change_ratio=self.min_change_ratio,
                max_change_ratio=self.max_change_ratio,
            )
            wind_change_dataset = ForcePromptingDataset_WindForce_ChangeForce(
                video_root_dir=wind_change_video_root_dir,
                csv_path=wind_change_csv_path,
                image_size=(224, 416),
                stride=(1, 3),
                sample_n_frames=81,
                randomize_change_point=self.randomize_change_point,
                min_change_ratio=self.min_change_ratio,
                max_change_ratio=self.max_change_ratio,
            )
        sampler_point = torch.utils.data.distributed.DistributedSampler(point_dataset, shuffle=True, drop_last=True)
        sampler_wind = torch.utils.data.distributed.DistributedSampler(wind_dataset, shuffle=True, drop_last=True)
        if self.use_change_force:
            sampler_point_change = torch.utils.data.distributed.DistributedSampler(point_change_dataset, shuffle=True, drop_last=True)
            sampler_wind_change = torch.utils.data.distributed.DistributedSampler(wind_change_dataset, shuffle=True, drop_last=True)
        dataloader_point = torch.utils.data.DataLoader(
            point_dataset,
            batch_size=config.batch_size,
            sampler=sampler_point,
            collate_fn=collate_fn_ForcePromptingDataset_WindForce,
            num_workers=4,
        )
        dataloader_wind = torch.utils.data.DataLoader(
            wind_dataset,
            batch_size=config.batch_size,
            sampler=sampler_wind,
            collate_fn=collate_fn_ForcePromptingDataset_WindForce,
            num_workers=4,
        )
        if self.use_change_force:
            dataloader_point_change = torch.utils.data.DataLoader(
                point_change_dataset,
                batch_size=config.batch_size,
                sampler=sampler_point_change,
                collate_fn=collate_fn_ForcePromptingDataset_PointForce_ChangeForce,
                num_workers=4,
            )
            dataloader_wind_change = torch.utils.data.DataLoader(
                wind_change_dataset,
                batch_size=config.batch_size,
                sampler=sampler_wind_change,
                collate_fn=collate_fn_ForcePromptingDataset_WindForce_ChangeForce,
                num_workers=4,
            )
        if dist.get_rank() == 0:
            print("DATASET POINT SIZE %d" % len(point_dataset))
            print("DATASET WIND SIZE %d" % len(wind_dataset))
            if self.use_change_force:
                print("DATASET POINT CHANGE SIZE %d" % len(point_change_dataset))
                print("DATASET WIND CHANGE SIZE %d" % len(wind_change_dataset))
        self.dataloader_point = cycle(dataloader_point)
        self.dataloader_wind = cycle(dataloader_wind)
        if self.use_change_force:
            self.dataloader_point_change = cycle(dataloader_point_change)
            self.dataloader_wind_change = cycle(dataloader_wind_change)
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

        if self.step < config.ema_start_step:
            self.generator_ema = None

        self.max_grad_norm = 10.0
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
    

    def train_one_step(self, batch):
        self.log_iters = 1

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get the input
        text_prompts = batch["prompts"]
        hint = batch["controlnet_videos"]
        if batch.get("angle_1", None) is not None:
            # change-force batch: condition on the first force
            angle = batch["angle_1"]
            force = batch["force_1"]
        else:
            angle = batch["angle"]
            force = batch["force"]
            
        batch_size = len(text_prompts)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            with torch.inference_mode():
                if not self.config.load_raw_video:  # precomputed latent
                    clean_latent = batch["ode_latent"][:, -1].to(
                        device=self.device, dtype=self.dtype)
                else:  # encode raw video to latent
                    frames = batch["videos"].to(
                        device=self.device, dtype=self.dtype)
                    with torch.no_grad():
                        # [batch_size, num_frames, num_channels, height, width]
                        frames = frames.permute(0, 2, 1, 3, 4)
                        frames = frames / 255.0
                        frames = frames * 2.0 - 1.0
                        clean_latent = self.model.vae.encode_to_latent(
                            frames).to(device=self.device, dtype=self.dtype)    # GT latent for training
                            # Encode the input image as the first latent
                        first_image = frames[:, :, :1]
                        # image_noise_sigma = torch.normal(mean=-3.0, std=0.5, size=(1,), device=first_image.device)
                        # image_noise_sigma = torch.exp(image_noise_sigma).to(dtype=first_image.dtype)
                        # noisy_images = first_image + torch.randn_like(first_image) * image_noise_sigma[:, None, None, None, None]
                        noisy_images = first_image  # try no noise version

                        initial_latent = self.model.vae.encode_to_latent(noisy_images).to(device=self.device, dtype=self.dtype)
                        
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

                assert hint.shape[1] == clean_latent.shape[1]   # hint latents and the gt latents should have the same number of frames

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
            
            # Step 3: Train the generator
            generator_loss, log_dict = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                initial_latent=initial_latent,
                hint=hint,
                masked_latent=masked_latent.clone() if masked_latent is not None else None,
                wind_mag=force,
                wind_dir=angle,
            )

        self.generator_optimizer.zero_grad()
        generator_loss.backward()
        generator_grad_norm = self.model.generator.clip_grad_norm_(
            self.max_grad_norm)

        if (
            torch.isnan(generator_loss) or
            torch.isinf(generator_loss) or
            torch.isnan(generator_grad_norm) or
            torch.isinf(generator_grad_norm)
        ):
            print(f"[Warning] NaN/Inf detected at step {self.step}. Skipping update.")
            self.generator_optimizer.zero_grad()
            return None, {}

        self.generator_optimizer.step()
        
        self.step += 1

        if self.generator_ema is not None:
            self.generator_ema.update(self.model.generator)
        # Create EMA params (if not already created)
        if (self.step >= self.config.ema_start_step) and \
                (self.generator_ema is None) and (self.config.ema_weight > 0):
            self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)

        wandb_loss_dict = {
            "generator_loss": generator_loss.item(),
            "generator_grad_norm": generator_grad_norm.item(),
        }

        # Step 4: Logging
        if self.is_main_process:
            if not self.disable_wandb:
                wandb.log(wandb_loss_dict, step=self.step)
            print(wandb_loss_dict)

        if self.step % self.config.gc_interval == 0:
            if dist.get_rank() == 0:
                logging.info("DistGarbageCollector: Running GC.")
            gc.collect()
        return generator_loss, log_dict


    def generate_video(self, pipeline, prompts, image=None):
        batch_size = len(prompts)
        sampled_noise = torch.randn(
            [batch_size, 21, 16, 60, 104], device="cuda", dtype=self.dtype
        )
        video, _ = pipeline.inference(
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=True
        )
        current_video = video.permute(0, 1, 3, 4, 2).cpu().numpy() * 255.0
        return current_video

    def save_video(self, latents, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        video = self.model.vae.decode_to_pixel(latents.to(device=self.device, dtype=self.dtype), use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        video = rearrange(video, 'b t c h w -> b t h w c').cpu()
        video = video.cpu().numpy() * 255.0
        video = video.astype(np.uint8)
        write_video(path, video[0], fps=16)
        return video

    def train(self):
        while True:
            data_choice_weights = torch.tensor([float(self.dlocal_global_force_ratio), 1.0])
            change_no_change_weights = torch.tensor([float(self.dchange_no_change_ratio), 1.0])
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank == 0:
                force_choice = torch.multinomial(data_choice_weights, 1).to(device=self.device)
            else:
                force_choice = torch.empty((1,), dtype=torch.long, device=self.device)
            dist.broadcast(force_choice, src=0)
            force_choice = int(force_choice[0])
            if force_choice == 0:
                # point force
                if self.use_change_force:
                    point_choice_weights = change_no_change_weights
                    if rank == 0:
                        point_choice = torch.multinomial(point_choice_weights, 1).to(device=self.device)
                    else:
                        point_choice = torch.empty((1,), dtype=torch.long, device=self.device)
                    dist.broadcast(point_choice, src=0)
                    point_choice = int(point_choice[0])
                    if point_choice == 0:
                        # point change force
                        if self.is_main_process:
                            print("use point change force")
                        batch = next(self.dataloader_point_change)
                    else:
                        # point force
                        if self.is_main_process:
                            print("use point force")
                        batch = next(self.dataloader_point)
                else:
                    # point force
                    if self.is_main_process:
                        print("use point force")
                    batch = next(self.dataloader_point)
            else:
                # wind force
                if self.use_change_force:
                    if rank == 0:
                        change_no_change_choice = torch.multinomial(change_no_change_weights, 1).to(device=self.device)
                    else:
                        change_no_change_choice = torch.empty((1,), dtype=torch.long, device=self.device)
                    dist.broadcast(change_no_change_choice, src=0)
                    change_no_change_choice = int(change_no_change_choice[0])
                    if change_no_change_choice == 0:
                        # wind change force
                        if self.is_main_process:
                            print("use wind change force")
                        batch = next(self.dataloader_wind_change)
                    else:
                        # wind force
                        if self.is_main_process:
                            print("use wind force")
                        batch = next(self.dataloader_wind)
                else:
                    # wind force
                    if self.is_main_process:
                        print("use wind force")
                    batch = next(self.dataloader_wind)
            if self.is_main_process:
                print(f"Training step: {self.step}")
            generator_loss, log_dict = self.train_one_step(batch)
            if generator_loss is None:
                continue
            if (not self.config.no_save) and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                # save video
                if self.is_main_process:
                    x0 = log_dict['x0']
                    x0_pred = log_dict['x0_pred']
                    self.save_video(x0_pred, os.path.join(self.output_path,
                            f"saved_videos/x0_pred_{self.step:06d}.mp4"))
                    self.save_video(x0, os.path.join(self.output_path,
                            f"saved_videos/x0_{self.step:06d}.mp4"))

                torch.cuda.empty_cache()
                # if self.step == 10000:
                #     exit()

            barrier()
            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time
