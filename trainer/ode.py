from torch.utils.data import Dataset
import numpy as np
import torch
import torch.distributed as dist
import glob
import os
import csv
import time
from utils.distributed import launch_distributed_job, barrier, fsdp_wrap, fsdp_state_dict
from collections import defaultdict
from utils.misc import set_seed, compress_time, cycle
from torchvision.io import write_video
from einops import rearrange
from model import ODERegression
from utils.forceprompt_data.controlnet_datasets import (
    ForcePromptingDataset_WindForce,
    ForcePromptingDataset_PointForce,
    ForcePromptingDataset_WindForce_ChangeForce,
    ForcePromptingDataset_PointForce_ChangeForce,
)


def _load_allowed_row_indices(csv_path, split_label, split_column="is_train_val"):
    if split_label is None:
        return None
    target = str(split_label).strip().lower()
    allowed = set()
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or split_column not in reader.fieldnames:
            raise ValueError(
                f"{csv_path} must contain split column '{split_column}' when split filtering is enabled."
            )
        for row_idx, row in enumerate(reader):
            value = str(row.get(split_column, "")).strip().lower()
            if value == target:
                allowed.add(row_idx)
    if not allowed:
        raise ValueError(f"No rows matched split '{split_label}' in {csv_path}.")
    return allowed


def _filter_pt_files_by_row_indices(files, allowed_indices):
    if allowed_indices is None:
        return files
    filtered = []
    for file_path in files:
        stem = os.path.splitext(os.path.basename(file_path))[0]
        try:
            row_idx = int(stem)
        except ValueError:
            continue
        if row_idx in allowed_indices:
            filtered.append(file_path)
    if not filtered:
        raise ValueError("No .pt files matched the requested split row indices.")
    return filtered


class DiverseWindForceODERegressionDataset(Dataset):
    def __init__(self, root_path, split_label=None):
        self.root_path = root_path
        self.files = sorted(glob.glob(os.path.join(self.root_path, "*.pt")))
        allowed_indices = _load_allowed_row_indices(
            "datasets/wind-force-diverse-16K-filtered_5835/image_augment_wind.csv",
            split_label=split_label,
        )
        self.files = _filter_pt_files_by_row_indices(self.files, allowed_indices)

        wind_dataset = ForcePromptingDataset_WindForce(
            video_root_dir="datasets/wind-force-diverse-16K-filtered_5835/after_qwen_filtered_cropped",
            csv_path="datasets/wind-force-diverse-16K-filtered_5835/image_augment_wind.csv",
            image_size=(480, 832),
            stride=(1, 3),
            sample_n_frames=81,
            is_validation_dataset=True,
            split_label=None,
        )
        self.dataset = wind_dataset

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        data_dict = torch.load(file, weights_only=False)
        real_idx = data_dict['diverse_index']
        latents = data_dict['latents_list']

        batch_data = self.dataset[real_idx]

        return {
            'file_id': batch_data['file_id'],
            'video': batch_data['video'],
            'caption': batch_data['caption'],
            'controlnet_video': batch_data['controlnet_video'],
            'force': batch_data['force'],
            'angle': batch_data['angle'],
            'x_pos': batch_data['x_pos'] if 'x_pos' in batch_data else None,
            'y_pos': batch_data['y_pos'] if 'y_pos' in batch_data else None,
            'force_type': batch_data['force_type'],
            'latents_list': latents,
        }

class DiverseWindForceChangeODERegressionDataset(Dataset):
    def __init__(self, root_path, split_label=None):
        self.root_path = root_path
        self.files = sorted(glob.glob(os.path.join(self.root_path, "*.pt")))
        allowed_indices = _load_allowed_row_indices(
            "datasets/wind-force-diverse-16K-filtered_5835/image_augment_wind_change.csv",
            split_label=split_label,
        )
        self.files = _filter_pt_files_by_row_indices(self.files, allowed_indices)

        wind_force_change_dataset = ForcePromptingDataset_WindForce_ChangeForce(
            video_root_dir="datasets/wind-force-diverse-16K-filtered_5835/after_qwen_filtered_cropped",
            csv_path="datasets/wind-force-diverse-16K-filtered_5835/image_augment_wind_change.csv",
            image_size=(480, 832),
            stride=(1, 3),
            sample_n_frames=81,
            is_validation_dataset=True,
            split_label=None,
        )
        self.dataset = wind_force_change_dataset

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        data_dict = torch.load(file, weights_only=False)
        real_idx = data_dict['diverse_index']
        latents = data_dict['latents_list']

        batch_data = self.dataset[real_idx]

        return {
            'file_id': batch_data['file_id'],
            'video': batch_data['video'],
            'caption': batch_data['caption'],
            'controlnet_video': batch_data['controlnet_video'],
            'force': batch_data['force_1'], # wind_force_change -> force_1
            'angle': batch_data['angle_1'], # wind_force_change -> angle_1
            'x_pos': batch_data['x_pos'] if 'x_pos' in batch_data else None,
            'y_pos': batch_data['y_pos'] if 'y_pos' in batch_data else None,
            'force_type': batch_data['force_type'],
            'latents_list': latents,
        }

class SyntheticWindForceODERegressionDataset(Dataset):
    def __init__(self, root_path):
        self.root_path = root_path
        self.files = sorted(glob.glob(os.path.join(self.root_path, "*.pt")))
        wind_dataset = ForcePromptingDataset_WindForce(
            video_root_dir="datasets/wind-force/train/wind_force_15359",
            csv_path="datasets/wind-force/train/wind_force_15359.csv",
            image_size=(480, 832),
            stride=(1, 3),
            sample_n_frames=81,
            is_validation_dataset=False,
        )
        self.dataset = wind_dataset

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        data_dict = torch.load(file, weights_only=False)
        real_idx = data_dict['synthetic_index']
        latents = data_dict['latents_list']

        batch_data = self.dataset[real_idx]

        return {
            'file_id': batch_data['file_id'],
            'video': batch_data['video'],
            'caption': batch_data['caption'],
            'controlnet_video': batch_data['controlnet_video'],
            'force': batch_data['force'],
            'angle': batch_data['angle'],
            'x_pos': batch_data['x_pos'] if 'x_pos' in batch_data else None,
            'y_pos': batch_data['y_pos'] if 'y_pos' in batch_data else None,
            'force_type': batch_data['force_type'],
            'latents_list': latents,
        }

class SyntheticWindForceChangeODERegressionDataset(Dataset):
    def __init__(self, root_path):
        self.root_path = root_path
        self.files = sorted(glob.glob(os.path.join(self.root_path, "*.pt")))
        
        wind_force_change_dataset = ForcePromptingDataset_WindForce_ChangeForce(
            video_root_dir="datasets/wind-force-change/wind_force_change_15000",
            csv_path="datasets/wind-force-change/wind_force_change_15000.csv",
            image_size=(480, 832),
            stride=(1, 3),
            sample_n_frames=81,
        )
        self.dataset = wind_force_change_dataset

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        data_dict = torch.load(file, weights_only=False)
        real_idx = data_dict['synthetic_index']
        latents = data_dict['latents_list']

        batch_data = self.dataset[real_idx]

        return {
            'file_id': batch_data['file_id'],
            'video': batch_data['video'],
            'caption': batch_data['caption'],
            'controlnet_video': batch_data['controlnet_video'],
            'force': batch_data['force_1'], # wind_force_change -> force_1
            'angle': batch_data['angle_1'], # wind_force_change -> angle_1
            'x_pos': batch_data['x_pos'] if 'x_pos' in batch_data else None,
            'y_pos': batch_data['y_pos'] if 'y_pos' in batch_data else None,
            'force_type': batch_data['force_type'],
            'latents_list': latents,
        }

class DiversePointForceODERegressionDataset(Dataset):
    def __init__(self, root_path, split_label=None):
        self.root_path = root_path
        self.files = sorted(glob.glob(os.path.join(self.root_path, "*.pt")))
        allowed_indices = _load_allowed_row_indices(
            "datasets/point-force-diverse/point_force_generated_8K.csv",
            split_label=split_label,
        )
        self.files = _filter_pt_files_by_row_indices(self.files, allowed_indices)
        point_dataset = ForcePromptingDataset_PointForce(
            video_root_dir="datasets/point-force-diverse/local_force_pexels",
            csv_path="datasets/point-force-diverse/point_force_generated_8K.csv",
            image_size=(480, 832),
            stride=(1, 3),
            sample_n_frames=81,
            is_validation_dataset=True,
            split_label=None,
        )
        self.dataset = point_dataset

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        data_dict = torch.load(file, weights_only=False)
        real_idx = data_dict['diverse_index']
        latents = data_dict['latents_list']

        batch_data = self.dataset[real_idx]

        return {
            'file_id': batch_data['file_id'],
            'video': batch_data['video'],
            'caption': batch_data['caption'],
            'controlnet_video': batch_data['controlnet_video'],
            'force': batch_data['force'],
            'angle': batch_data['angle'],
            'x_pos': batch_data['x_pos'] if 'x_pos' in batch_data else None,
            'y_pos': batch_data['y_pos'] if 'y_pos' in batch_data else None,
            'force_type': batch_data['force_type'],
            'latents_list': latents,
        }

class DiversePointForceChangeODERegressionDataset(Dataset):
    def __init__(self, root_path, split_label=None):
        self.root_path = root_path
        self.files = sorted(glob.glob(os.path.join(self.root_path, "*.pt")))
        allowed_indices = _load_allowed_row_indices(
            "datasets/point-force-diverse/point_force_generated_8K_change_force_v1.csv",
            split_label=split_label,
        )
        self.files = _filter_pt_files_by_row_indices(self.files, allowed_indices)
        point_force_change_dataset = ForcePromptingDataset_PointForce_ChangeForce(
            video_root_dir="datasets/point-force-diverse/local_force_pexels",
            csv_path="datasets/point-force-diverse/point_force_generated_8K_change_force_v1.csv",
            image_size=(480, 832),
            stride=(1, 3),
            sample_n_frames=81,
            is_validation_dataset=True,
            split_label=None,
        )
        self.dataset = point_force_change_dataset

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        data_dict = torch.load(file, weights_only=False)
        real_idx = data_dict['diverse_index']
        latents = data_dict['latents_list']

        batch_data = self.dataset[real_idx]

        return {
            'file_id': batch_data['file_id'],
            'video': batch_data['video'],
            'caption': batch_data['caption'],
            'controlnet_video': batch_data['controlnet_video'],
            'force': batch_data['force_1'], # point force change -> force_1
            'angle': batch_data['angle_1'], # point force change -> angle_1
            'x_pos': batch_data['x_pos_1'], # point force change -> x_pos_1
            'y_pos': batch_data['y_pos_1'], # point force change -> y_pos_1
            'force_type': batch_data['force_type'],
            'latents_list': latents,
        }

class SyntheticPointForceODERegressionDataset(Dataset):
    def __init__(self, root_path):
        self.root_path = root_path
        self.files = sorted(glob.glob(os.path.join(self.root_path, "*.pt")))

        point_dataset = ForcePromptingDataset_PointForce(
            video_root_dir="datasets/point-force/train/point_force_23000",
            csv_path="datasets/point-force/train/point_force_23000.csv",
            image_size=(480, 832),
            stride=(1, 3),
            sample_n_frames=81,
            is_validation_dataset=False,
            remove_carnation=True,
        )

        self.dataset = point_dataset

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        data_dict = torch.load(file, weights_only=False)
        real_idx = data_dict['synthetic_index']
        latents = data_dict['latents_list']

        batch_data = self.dataset[real_idx]

        return {
            'file_id': batch_data['file_id'],
            'video': batch_data['video'],
            'caption': batch_data['caption'],
            'controlnet_video': batch_data['controlnet_video'],
            'force': batch_data['force'],
            'angle': batch_data['angle'],
            'x_pos': batch_data['x_pos'] if 'x_pos' in batch_data else None,
            'y_pos': batch_data['y_pos'] if 'y_pos' in batch_data else None,
            'force_type': batch_data['force_type'],
            'latents_list': latents,
        }

class SyntheticPointForceChangeODERegressionDataset(Dataset):
    def __init__(self, root_path):
        self.root_path = root_path
        self.files = sorted(glob.glob(os.path.join(self.root_path, "*.pt")))

        point_force_change_dataset = ForcePromptingDataset_PointForce_ChangeForce(
            video_root_dir="datasets/point-force-change-force/videos",
            csv_path="datasets/point-force-change-force/point-force-change.csv",
            image_size=(480, 832),
            stride=(1, 3),
            sample_n_frames=81,
            is_validation_dataset=False,
        )
        self.dataset = point_force_change_dataset

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        data_dict = torch.load(file, weights_only=False)
        real_idx = data_dict['synthetic_index']
        latents = data_dict['latents_list']

        batch_data = self.dataset[real_idx]

        return {
            'file_id': batch_data['file_id'],
            'video': batch_data['video'],
            'caption': batch_data['caption'],
            'controlnet_video': batch_data['controlnet_video'],
            'force': batch_data['force_1'],
            'angle': batch_data['angle_1'],
            'x_pos': batch_data['x_pos_1'],
            'y_pos': batch_data['y_pos_1'],
            'force_type': batch_data['force_type'],
            'latents_list': latents,
        }


def collate_fn(examples):
    videos = [example["video"] for example in examples]
    prompts = [example["caption"] for example in examples]
    controlnet_videos = [example["controlnet_video"] for example in examples]
    file_ids = [example["file_id"] for example in examples]

    forces = [example["force"] for example in examples]
    angles = [example["angle"] for example in examples]
    x_poss = [example["x_pos"] for example in examples if "x_pos" in example]
    y_poss = [example["y_pos"] for example in examples if "y_pos" in example]
    force_types = [example["force_type"] for example in examples]

    latents_list = [example["latents_list"][0] for example in examples]
    latents_list = torch.stack(latents_list)
    latents_list = latents_list.to(memory_format=torch.contiguous_format).float()

    if videos[0] is not None:
        videos = torch.stack(videos)
        videos = videos.to(memory_format=torch.contiguous_format).float()

        # nate added this
        first_frames = videos[:, 0]
        first_frames = first_frames.to(memory_format=torch.contiguous_format).float()

        controlnet_videos = torch.stack(controlnet_videos)
        controlnet_videos = controlnet_videos.to(memory_format=torch.contiguous_format).float()
    else:
        first_frames = None
        videos = None
        controlnet_videos = None

    return {
        "file_ids" : file_ids,
        "first_frames" : first_frames,
        "videos": videos,
        "prompts": prompts,
        "controlnet_videos": controlnet_videos,
        "force": forces,
        "angle": angles,
        "x_pos": x_poss,
        "y_pos": y_poss,
        "force_type": force_types,
        "latents_list": latents_list,
    }

class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = self.config.get("start_step", 0)

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.disable_wandb = config.disable_wandb
        self.use_change_force = config.get("use_change_force", False)
        self.all_forces = config.get("all_forces", False)
        self.gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        self.output_path = config.logdir

        # Step 2: Initialize the model and optimizer

        assert config.distribution_loss == "ode", "Only ODE loss is supported for ODE training"
        self.model = ODERegression(config, device=self.device)

        if getattr(config, "generator_ckpt", False):
            print(f"Loading pretrained generator from {config.generator_ckpt}")
            ckpt_load_start = time.time()
            state_dict = torch.load(config.generator_ckpt, map_location="cpu")[
                'generator']
            print(f"torch.load of generator_ckpt took {time.time() - ckpt_load_start:.2f}s")

            # The checkpoint holds two sibling towers: "model.*" (the base DiT) and
            # "control_model.*" (the ControlNet). Route them separately.
            #
            # The previous version flattened
            # both with key.replace("model.", ""), which also rewrites the "model."
            # *inside* "control_model." -- e.g. control_model.zero_convs.0.weight
            # became control_zero_convs.0.weight, matching nothing. strict=False then
            # dropped every ControlNet tensor without a warning, so the control branch
            # started from its constructor state: a fresh xavier input_hint_block and
            # zero_convs pinned to exactly 0, i.e. no force conditioning at all,
            # whatever checkpoint was named here.
            base_state_dict = {}
            control_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("control_model."):
                    control_state_dict[key[len("control_model."):]] = value
                elif key.startswith("model."):
                    base_state_dict[key[len("model."):]] = value
                else:
                    # Bare checkpoints (raw Wan weights) carry no tower prefix.
                    base_state_dict[key] = value
            del state_dict

            if not control_state_dict:
                # Base-only checkpoint: seed the control tower from the base weights,
                # which is what CausalControlNet.from_pretrained would do anyway. Its
                # ControlNet-specific tensors stay at their zero/xavier init and will
                # be reported as missing below.
                print("  checkpoint has no control_model.* keys; seeding the control "
                      "tower from the base weights")
                control_state_dict = base_state_dict

            self._load_tower(self.model.generator.model,
                             base_state_dict, "generator.model")
            self._load_tower(self.model.generator.control_model,
                             control_state_dict, "generator.control_model")
            del base_state_dict, control_state_dict
            print(f"Loading pretrained generator from {config.generator_ckpt} took {time.time() - ckpt_load_start:.2f}s total")

        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy,
        )
        self.model.text_encoder = self.model.text_encoder.to(device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)
        self.model.vae = self.model.vae.to(device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        trainable_param = list(filter(lambda p: p.requires_grad, self.model.generator.parameters()))
        self.generator_optimizer = torch.optim.AdamW(
            trainable_param,
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        # Step 3: Initialize the dataloader

        wind_synthetic_force_dataset = None
        wind_synthetic_force_change_dataset = None
        point_synthetic_force_dataset = None
        point_synthetic_force_change_dataset = None
        wind_diverse_force_dataset = None
        wind_diverse_force_change_dataset = None
        point_diverse_force_dataset = None
        point_diverse_force_change_dataset = None

        if getattr(config, "wind_synthetic_force_data_path", False):
            wind_synthetic_force_dataset = SyntheticWindForceODERegressionDataset(config.wind_synthetic_force_data_path)
            sampler_wind_synthetic_force = torch.utils.data.distributed.DistributedSampler(
                wind_synthetic_force_dataset, shuffle=True, drop_last=True)
            dataloader_wind_synthetic_force = torch.utils.data.DataLoader(
                wind_synthetic_force_dataset, batch_size=config.batch_size, sampler=sampler_wind_synthetic_force, num_workers=0, collate_fn=collate_fn)

        if getattr(config, "wind_synthetic_force_change_data_path", False):
            wind_synthetic_force_change_dataset = SyntheticWindForceChangeODERegressionDataset(config.wind_synthetic_force_change_data_path)
            sampler_wind_synthetic_force_change = torch.utils.data.distributed.DistributedSampler(
                wind_synthetic_force_change_dataset, shuffle=True, drop_last=True)
            dataloader_wind_synthetic_force_change = torch.utils.data.DataLoader(
                wind_synthetic_force_change_dataset, batch_size=config.batch_size, sampler=sampler_wind_synthetic_force_change, num_workers=0, collate_fn=collate_fn)

        if getattr(config, "point_synthetic_force_data_path", False):
            point_synthetic_force_dataset = SyntheticPointForceODERegressionDataset(config.point_synthetic_force_data_path)
            sampler_point_synthetic_force = torch.utils.data.distributed.DistributedSampler(
                point_synthetic_force_dataset, shuffle=True, drop_last=True)
            dataloader_point_synthetic_force = torch.utils.data.DataLoader(
                point_synthetic_force_dataset, batch_size=config.batch_size, sampler=sampler_point_synthetic_force, num_workers=0, collate_fn=collate_fn)

        if getattr(config, "point_synthetic_force_change_data_path", False):
            point_synthetic_force_change_dataset = SyntheticPointForceChangeODERegressionDataset(config.point_synthetic_force_change_data_path)
            sampler_point_synthetic_force_change = torch.utils.data.distributed.DistributedSampler(
                point_synthetic_force_change_dataset, shuffle=True, drop_last=True)
            dataloader_point_synthetic_force_change = torch.utils.data.DataLoader(
                point_synthetic_force_change_dataset, batch_size=config.batch_size, sampler=sampler_point_synthetic_force_change, num_workers=0, collate_fn=collate_fn)

        wind_diverse_split_label = getattr(config, "wind_diverse_split_label", "train")
        if getattr(config, "wind_diverse_force_data_path", False):
            wind_diverse_force_dataset = DiverseWindForceODERegressionDataset(
                config.wind_diverse_force_data_path,
                split_label=wind_diverse_split_label,
            )
            sampler_wind_diverse_force = torch.utils.data.distributed.DistributedSampler(
                wind_diverse_force_dataset, shuffle=True, drop_last=True)
            dataloader_wind_diverse_force = torch.utils.data.DataLoader(
                wind_diverse_force_dataset, batch_size=config.batch_size, sampler=sampler_wind_diverse_force, num_workers=0, collate_fn=collate_fn)

        if getattr(config, "wind_diverse_force_change_data_path", False):
            wind_diverse_force_change_dataset = DiverseWindForceChangeODERegressionDataset(
                config.wind_diverse_force_change_data_path,
                split_label=wind_diverse_split_label,
            )
            sampler_wind_diverse_force_change = torch.utils.data.distributed.DistributedSampler(
                wind_diverse_force_change_dataset, shuffle=True, drop_last=True)
            dataloader_wind_diverse_force_change = torch.utils.data.DataLoader(
                wind_diverse_force_change_dataset, batch_size=config.batch_size, sampler=sampler_wind_diverse_force_change, num_workers=0, collate_fn=collate_fn)

        point_diverse_split_label = getattr(config, "point_diverse_split_label", "train")
        if getattr(config, "point_diverse_force_data_path", False):
            point_diverse_force_dataset = DiversePointForceODERegressionDataset(
                config.point_diverse_force_data_path,
                split_label=point_diverse_split_label,
            )
            sampler_point_diverse_force = torch.utils.data.distributed.DistributedSampler(
                point_diverse_force_dataset, shuffle=True, drop_last=True)
            dataloader_point_diverse_force = torch.utils.data.DataLoader(
                point_diverse_force_dataset, batch_size=config.batch_size, sampler=sampler_point_diverse_force, num_workers=0, collate_fn=collate_fn)

        if getattr(config, "point_diverse_force_change_data_path", False):
            point_diverse_force_change_dataset = DiversePointForceChangeODERegressionDataset(
                config.point_diverse_force_change_data_path,
                split_label=point_diverse_split_label,
            )
            sampler_point_diverse_force_change = torch.utils.data.distributed.DistributedSampler(
                point_diverse_force_change_dataset, shuffle=True, drop_last=True)
            dataloader_point_diverse_force_change = torch.utils.data.DataLoader(
                point_diverse_force_change_dataset, batch_size=config.batch_size, sampler=sampler_point_diverse_force_change, num_workers=0, collate_fn=collate_fn)

        self.dataloader_wind_synthetic_force = None
        self.dataloader_point_synthetic_force = None
        self.dataloader_wind_diverse_force = None
        self.dataloader_point_diverse_force = None
        self.dataloader_wind_synthetic_force_change = None
        self.dataloader_point_synthetic_force_change = None
        self.dataloader_wind_diverse_force_change = None
        self.dataloader_point_diverse_force_change = None
        if wind_synthetic_force_dataset is not None:
            if dist.get_rank() == 0:
                print("WIND SYNTHETIC FORCE DATASET SIZE %d" % len(wind_synthetic_force_dataset))
            self.dataloader_wind_synthetic_force = cycle(dataloader_wind_synthetic_force)
        if point_synthetic_force_dataset is not None:
            if dist.get_rank() == 0:
                print("POINT SYNTHETIC FORCE DATASET SIZE %d" % len(point_synthetic_force_dataset))
            self.dataloader_point_synthetic_force = cycle(dataloader_point_synthetic_force)
        if wind_diverse_force_dataset is not None:
            if dist.get_rank() == 0:
                print("WIND DIVERSE FORCE DATASET SIZE %d" % len(wind_diverse_force_dataset))
            self.dataloader_wind_diverse_force = cycle(dataloader_wind_diverse_force)
        if point_diverse_force_dataset is not None:
            if dist.get_rank() == 0:
                print("POINT DIVERSE FORCE DATASET SIZE %d" % len(point_diverse_force_dataset))
            self.dataloader_point_diverse_force = cycle(dataloader_point_diverse_force)
            
        if wind_synthetic_force_change_dataset is not None:
            if dist.get_rank() == 0:
                print("WIND SYNTHETIC FORCE CHANGE DATASET SIZE %d" % len(wind_synthetic_force_change_dataset))
            self.dataloader_wind_synthetic_force_change = cycle(dataloader_wind_synthetic_force_change)
        if point_synthetic_force_change_dataset is not None:
            if dist.get_rank() == 0:
                print("POINT SYNTHETIC FORCE CHANGE DATASET SIZE %d" % len(point_synthetic_force_change_dataset))
            self.dataloader_point_synthetic_force_change = cycle(dataloader_point_synthetic_force_change)
        if wind_diverse_force_change_dataset is not None:
            if dist.get_rank() == 0:
                print("WIND DIVERSE FORCE CHANGE DATASET SIZE %d" % len(wind_diverse_force_change_dataset))
            self.dataloader_wind_diverse_force_change = cycle(dataloader_wind_diverse_force_change)
        if point_diverse_force_change_dataset is not None:
            if dist.get_rank() == 0:
                print("POINT DIVERSE FORCE CHANGE DATASET SIZE %d" % len(point_diverse_force_change_dataset))
            self.dataloader_point_diverse_force_change = cycle(dataloader_point_diverse_force_change)
        self.force_dataloader_map = {
            "diverse": {
                "point": {
                    "no_change": self.dataloader_point_diverse_force,
                    "change": self.dataloader_point_diverse_force_change,
                },
                "wind": {
                    "no_change": self.dataloader_wind_diverse_force,
                    "change": self.dataloader_wind_diverse_force_change,
                },
            },
            "synthetic": {
                "point": {
                    "no_change": self.dataloader_point_synthetic_force,
                    "change": self.dataloader_point_synthetic_force_change,
                },
                "wind": {
                    "no_change": self.dataloader_wind_synthetic_force,
                    "change": self.dataloader_wind_synthetic_force_change,
                },
            },
        }

        self.max_grad_norm = 10.0
        self.previous_time = None

        self.image_or_video_shape = list(self.config.image_or_video_shape)
        num_latent_frames = self.image_or_video_shape[1]
        num_video_frames = (num_latent_frames - 1) * 4 + 1  # block size is 4
        self.video_shape = (1, num_video_frames, 3, self.image_or_video_shape[3] * 16, self.image_or_video_shape[4] * 16)
        self.hint_shape = (1, num_video_frames, 4, self.image_or_video_shape[3] * 16, self.image_or_video_shape[4] * 16)

    @staticmethod
    def _load_tower(module, state_dict, name):
        """Load one tower of the generator and report what actually matched.

        Non-strict on purpose: the base tower has all 30 blocks while the ControlNet
        keeps only the first `controlnet_layers`, so leftover checkpoint keys are
        expected. Matching *nothing* is not -- that is the failure mode that used to
        silently reset the whole control branch, so make it fatal instead.
        """
        own_keys = set(module.state_dict().keys())
        ckpt_keys = set(state_dict.keys())
        matched = own_keys & ckpt_keys
        missing = sorted(own_keys - ckpt_keys)
        print(f"  {name}: matched {len(matched)}/{len(own_keys)} tensors "
              f"({len(ckpt_keys - own_keys)} checkpoint tensors unused)")
        if missing:
            print(f"  {name}: {len(missing)} tensors kept their initial values, "
                  f"e.g. {missing[:5]}")
        if not matched:
            raise RuntimeError(
                f"{name}: no checkpoint tensor matched any parameter -- refusing to "
                "train from a silently re-initialised module. Check the key prefixes "
                "in the checkpoint."
            )
        module.load_state_dict(state_dict, strict=False)

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

    def save(self):
        print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(
            self.model.generator)
        state_dict = {
            "generator": generator_state_dict
        }

        if self.is_main_process:
            os.makedirs(os.path.join(self.output_path,
                        f"checkpoint_model_{self.step:06d}"), exist_ok=True)
            torch.save(state_dict, os.path.join(self.output_path,
                       f"checkpoint_model_{self.step:06d}", "model.pt"))
            print("Model saved to", os.path.join(self.output_path,
                  f"checkpoint_model_{self.step:06d}", "model.pt"))

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
        if len(source_names) == 1:
            return source_names[0]
        if not source_names:
            raise RuntimeError("No force dataloader is available.")

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
        force_names = [
            force_name
            for force_name in self.force_dataloader_map[source_name]
            if self._force_type_has_any_loader(source_name, force_name)
        ]
        if len(force_names) == 1:
            return force_names[0]
        if not force_names:
            raise RuntimeError(f"No force-type dataloader is available for source={source_name}.")
        if "point" in force_names and "wind" in force_names and len(force_names) == 2:
            local_global_force_weights = torch.tensor(
                [float(self.config.dlocal_global_force_ratio), 1.0],
                device=self.device,
            )
            force_choice = self._sample_distributed_choice(local_global_force_weights)
            return "point" if force_choice == 0 else "wind"
        force_choice = self._sample_distributed_choice(
            torch.ones(len(force_names), device=self.device)
        )
        return force_names[force_choice]

    def _select_change_name(self, source_name, force_name):
        force_loaders = self.force_dataloader_map[source_name][force_name]
        has_change = force_loaders["change"] is not None
        has_no_change = force_loaders["no_change"] is not None
        if has_change and has_no_change:
            change_no_change_weights = torch.tensor(
                [float(getattr(self.config, "dchange_no_change_ratio", 1.0)), 1.0],
                device=self.device,
            )
            change_choice = self._sample_distributed_choice(change_no_change_weights)
            return "change" if change_choice == 0 else "no_change"
        preferred = "change" if self.use_change_force else "no_change"
        fallback = "no_change" if preferred == "change" else "change"
        if force_loaders[preferred] is not None:
            return preferred
        if force_loaders[fallback] is not None:
            if self.is_main_process:
                print(
                    f"missing {source_name}/{force_name}/{preferred} dataloader, "
                    f"fallback to {fallback}"
                )
            return fallback
        raise RuntimeError(
            f"No change/no_change dataloader is available for source={source_name}, force={force_name}."
        )

    def _get_next_force_batch(self):
        source_name = self._select_source_name()
        force_name = self._select_force_type_name(source_name)
        change_name = self._select_change_name(source_name, force_name)
        loader = self.force_dataloader_map[source_name][force_name][change_name]
        if loader is None:
            raise RuntimeError(
                f"Selected force dataloader is None: source={source_name}, force={force_name}, change={change_name}."
            )
        if self.is_main_process:
            print(f"use {source_name} {force_name} {change_name} force")
        return next(loader)

    def train_one_step(self):
        self.model.eval()  # prevent any randomness (e.g. dropout)

        # Step 1: Get the next batch of data
        batch = self._get_next_force_batch()
        ode_latent = batch["latents_list"].to(
            device=self.device, dtype=self.dtype)

        is_force_data = "controlnet_videos" in batch and batch["controlnet_videos"] is not None
        text_prompts = batch["prompts"]

        if is_force_data:
            hint = batch["controlnet_videos"]
        else:
            hint = torch.zeros(self.hint_shape, device=self.device, dtype=self.dtype)
        batch_size = len(text_prompts)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            with torch.inference_mode():
                if not self.config.load_raw_video:  # precomputed latent
                    clean_latent = batch["ode_latent"][:, -1].to(
                        device=self.device, dtype=self.dtype)
                else:  # encode raw video to latent
                    if is_force_data:
                        frames = batch["videos"].to(
                            device=self.device, dtype=self.dtype)
                    else:
                        frames = torch.zeros(self.video_shape, device=self.device, dtype=self.dtype)
                    with torch.no_grad():
                        # [batch_size, num_frames, num_channels, height, width]
                        frames = frames.permute(0, 2, 1, 3, 4)
                        frames = frames / 255.0
                        frames = frames * 2.0 - 1.0
                        first_image = frames[:, :, :1]
                        # image_noise_sigma = torch.normal(mean=-3.0, std=0.5, size=(1,), device=first_image.device)
                        # image_noise_sigma = torch.exp(image_noise_sigma).to(dtype=first_image.dtype)
                        # noisy_images = first_image + torch.randn_like(first_image) * image_noise_sigma[:, None, None, None, None]
                        noisy_images = first_image
                        if is_force_data:
                            # clean_latent = self.model.vae.encode_to_latent(
                            #     frames).to(device=self.device, dtype=self.dtype)    # GT latent for training
                            clean_latent = None
                                # Encode the input image as the first latent
                            initial_latent = self.model.vae.encode_to_latent(noisy_images).to(device=self.device, dtype=self.dtype)
                        else:
                            clean_latent = None
                            initial_latent = None
                        
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
            ode_latent=ode_latent,
            image_or_video_shape=self.image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=initial_latent,
            hint=hint,
            masked_latent=masked_latent.clone() if masked_latent is not None else None,
        )

        unnormalized_loss = log_dict["unnormalized_loss"]
        timestep = log_dict["timestep"]

        if self.world_size > 1:
            gathered_unnormalized_loss = torch.zeros(
                [self.world_size, *unnormalized_loss.shape],
                dtype=unnormalized_loss.dtype, device=self.device)
            gathered_timestep = torch.zeros(
                [self.world_size, *timestep.shape],
                dtype=timestep.dtype, device=self.device)

            dist.all_gather_into_tensor(
                gathered_unnormalized_loss, unnormalized_loss)
            dist.all_gather_into_tensor(gathered_timestep, timestep)
        else:
            gathered_unnormalized_loss = unnormalized_loss
            gathered_timestep = timestep

        loss_breakdown = defaultdict(list)
        stats = {}

        for index, t in enumerate(timestep):
            loss_breakdown[str(int(t.item()) // 250 * 250)].append(
                unnormalized_loss[index].item())

        for key_t in loss_breakdown.keys():
            stats["loss_at_time_" + key_t] = sum(loss_breakdown[key_t]) / \
                len(loss_breakdown[key_t])

        generator_loss = generator_loss / self.gradient_accumulation_steps

        generator_loss.backward()
        generator_grad_norm = self.model.generator.clip_grad_norm_(
            self.max_grad_norm)

        if (self.step + 1) % self.gradient_accumulation_steps == 0:
            self.generator_optimizer.step()
            self.generator_optimizer.zero_grad()

        VISUALIZE = self.step % self.config.log_video_iters == 0
        if VISUALIZE and not self.config.no_visualize and self.is_main_process:
            # Visualize the input, output, and ground truth
            input = log_dict["input"]
            output = log_dict["output"]
            ground_truth = ode_latent[:, -1]

            self.save_video(input, os.path.join(self.output_path, f"saved_videos/input_{self.step:06d}.mp4"))
            self.save_video(output, os.path.join(self.output_path, f"saved_videos/output_{self.step:06d}.mp4"))
            self.save_video(ground_truth, os.path.join(self.output_path, f"saved_videos/ground_truth_{self.step:06d}.mp4"))

        # Step 4: Logging
        if self.is_main_process:
            wandb_loss_dict = {
                "generator_loss": generator_loss.item(),
                "generator_grad_norm": generator_grad_norm.item(),
                **stats
            }
            print(wandb_loss_dict)

    def train(self):
        start_step = self.step
        while True:
            if self.is_main_process:
                print(f"Training step: {self.step}")
            self.train_one_step()
            if (not self.config.no_save) and self.step % self.config.log_iters == 0 and self.step > start_step:
                self.save()
                torch.cuda.empty_cache()

            barrier()
            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    print(f"per iteration time: {current_time - self.previous_time}")
                    self.previous_time = current_time

            self.step += 1
