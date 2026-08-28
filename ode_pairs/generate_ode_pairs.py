"""Stage 2: generate ODE solution pairs from a trained bidirectional teacher.

Runs the teacher over one force dataset and saves, per sample, five points along its
denoising trajectory. Stage 3 (`trainer/ode.py`) regresses the causal student onto those.

One `--scenario` per data source. Stage 3 wants all eight, so this is normally run once
per scenario.

    torchrun --nproc_per_node=8 ode_pairs/generate_ode_pairs.py \
        --scenario point_diverse \
        --config_path configs/finetune_bidirectional_teacher.yaml \
        --checkpoint_path <PATH_TO_TEACHER_CKPT>

See ode_pairs/README.md.
"""

import sys
from pathlib import Path

# Run from the repository root: `python ode_pairs/generate_ode_pairs.py ...`. Python puts
# *this file's* directory on sys.path, not the root, so the project imports below would not
# resolve without this. Config and dataset paths still resolve against the working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from einops import rearrange
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, SequentialSampler
from torchvision.io import write_video
from tqdm import tqdm

from model import WanControlNet
from utils.forceprompt_data.controlnet_datasets import (
    ForcePromptingDataset_PointForce,
    ForcePromptingDataset_PointForce_ChangeForce,
    ForcePromptingDataset_WindForce,
    ForcePromptingDataset_WindForce_ChangeForce,
)
from utils.forceprompt_data.data_utils import (
    add_aesthetic_point_force_change_prompt_to_video,
    add_aesthetic_point_force_prompt_to_video,
    add_aesthetic_wind_force_change_prompt_to_video,
    add_aesthetic_wind_force_prompt_to_video,
    collate_fn_ForcePromptingDataset_PointForce_ChangeForce,
    collate_fn_ForcePromptingDataset_WindForce,
    collate_fn_ForcePromptingDataset_WindForce_ChangeForce,
)
from utils.misc import compress_time, set_seed


@dataclass(frozen=True)
class Scenario:
    """One force data source, and everything that differs about generating pairs from it."""

    dataset_cls: type
    video_root_dir: str
    csv_path: str
    #: image-mode (a still per sample) rather than video-mode (ground-truth clip)
    is_validation_dataset: bool
    #: key `trainer/ode.py` reads back out of the .pt -- see its ODERegressionDataset classes
    index_key: str
    #: which force-arrow overlay to burn into the preview mp4
    overlay: str


_DIVERSE_WIND_ROOT = "datasets/wind-force-diverse-16K-filtered_5835/after_qwen_filtered_cropped"
_DIVERSE_POINT_ROOT = "datasets/point-force-diverse/local_force_pexels"

SCENARIOS = {
    # ---- synthetic: rendered clips, ground-truth video available -------------------
    "point_synthetic": Scenario(
        ForcePromptingDataset_PointForce,
        "datasets/point-force/train/point_force_23000",
        "datasets/point-force/train/point_force_23000.csv",
        is_validation_dataset=False, index_key="synthetic_index", overlay="point",
    ),
    "point_change_synthetic": Scenario(
        ForcePromptingDataset_PointForce_ChangeForce,
        "datasets/point-force-change-force/videos",
        "datasets/point-force-change-force/point-force-change.csv",
        is_validation_dataset=False, index_key="synthetic_index", overlay="point_change",
    ),
    "wind_synthetic": Scenario(
        ForcePromptingDataset_WindForce,
        "datasets/wind-force/train/wind_force_15359",
        "datasets/wind-force/train/wind_force_15359.csv",
        is_validation_dataset=False, index_key="synthetic_index", overlay="wind",
    ),
    "wind_change_synthetic": Scenario(
        ForcePromptingDataset_WindForce_ChangeForce,
        "datasets/wind-force-change/wind_force_change_15000",
        "datasets/wind-force-change/wind_force_change_15000.csv",
        is_validation_dataset=False, index_key="synthetic_index", overlay="wind_change",
    ),
    # ---- diverse: real photographs, so image mode; the teacher invents the motion ---
    "point_diverse": Scenario(
        ForcePromptingDataset_PointForce,
        _DIVERSE_POINT_ROOT,
        "datasets/point-force-diverse/point_force_generated_8K.csv",
        is_validation_dataset=True, index_key="diverse_index", overlay="point",
    ),
    "point_change_diverse": Scenario(
        ForcePromptingDataset_PointForce_ChangeForce,
        _DIVERSE_POINT_ROOT,
        "datasets/point-force-diverse/point_force_generated_8K_change_force_v1.csv",
        is_validation_dataset=True, index_key="diverse_index", overlay="point_change",
    ),
    "wind_diverse": Scenario(
        ForcePromptingDataset_WindForce,
        _DIVERSE_WIND_ROOT,
        "datasets/wind-force-diverse-16K-filtered_5835/image_augment_wind.csv",
        is_validation_dataset=True, index_key="diverse_index", overlay="wind",
    ),
    "wind_change_diverse": Scenario(
        ForcePromptingDataset_WindForce_ChangeForce,
        _DIVERSE_WIND_ROOT,
        "datasets/wind-force-diverse-16K-filtered_5835/image_augment_wind_change.csv",
        is_validation_dataset=True, index_key="diverse_index", overlay="wind_change",
    ),
}

_COLLATE = {
    ForcePromptingDataset_PointForce_ChangeForce: collate_fn_ForcePromptingDataset_PointForce_ChangeForce,
    ForcePromptingDataset_WindForce_ChangeForce: collate_fn_ForcePromptingDataset_WindForce_ChangeForce,
}


def collate_for(dataset_cls):
    # The no-change point and wind datasets share one collate -- it is the generic
    # no-change schema, despite the WindForce name.
    return _COLLATE.get(dataset_cls, collate_fn_ForcePromptingDataset_WindForce)


def burn_overlay(kind, video, batch, num_video_frames):
    """Draw the force arrow(s) on the decoded preview clip."""
    if kind == "point":
        return add_aesthetic_point_force_prompt_to_video(
            video, batch["force"][0], batch["angle"][0],
            batch["x_pos"][0], 1 - batch["y_pos"][0],
            num_frames_with_signal=num_video_frames,
        )
    if kind == "wind":
        return add_aesthetic_wind_force_prompt_to_video(
            video, batch["force"][0], batch["angle"][0],
            num_frames_with_signal=num_video_frames,
        )
    if kind == "point_change":
        # both forces act at the same point, so force 2 reuses force 1's coordinates
        return add_aesthetic_point_force_change_prompt_to_video(
            video, batch["force_1"][0], batch["angle_1"][0],
            batch["x_pos_1"][0], 1 - batch["y_pos_1"][0],
            batch["force_2"][0], batch["angle_2"][0],
            batch["x_pos_1"][0], 1 - batch["y_pos_1"][0],
            idx=num_video_frames // 2, num_frames_with_signal=num_video_frames,
        )
    if kind == "wind_change":
        return add_aesthetic_wind_force_change_prompt_to_video(
            video, batch["force_1"][0], batch["angle_1"][0],
            batch["force_2"][0], batch["angle_2"][0],
            idx=num_video_frames // 2, num_frames_with_signal=num_video_frames,
        )
    raise ValueError(f"unknown overlay kind {kind!r}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--scenario", required=True, choices=sorted(SCENARIOS),
                        help="Which force data source to generate pairs from")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Config of the teacher being run (e.g. the stage-1 finetune config)")
    parser.add_argument("--checkpoint_path", type=str,
                        help="Teacher checkpoint. Without it the raw Wan weights are used.")
    parser.add_argument("--output_folder", type=str, default=None,
                        help="Where the .pt pairs go. Defaults to force_ode/<scenario>.")
    parser.add_argument("--video_root_dir", type=str, default=None,
                        help="Override the scenario's clip/image folder")
    parser.add_argument("--csv_path", type=str, default=None,
                        help="Override the scenario's CSV")
    parser.add_argument("--num_latent_frames", type=int, default=21,
                        help="Number of the latent frames used for denoising")
    parser.add_argument("--height", type=int, default=480, help="Height of the video")
    parser.add_argument("--width", type=int, default=832, help="Width of the video")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--control_weight", type=float, default=1.0, help="Control weight")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of dataloader workers")
    parser.add_argument("--no_preview", action="store_true",
                        help="Skip decoding and writing the arrow-overlay preview mp4s")
    return parser.parse_args()


def setup_distributed(seed: int):
    if dist.is_available() and "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl", device_id=torch.device(f"cuda:{local_rank}")
        )
        rank, world_size = dist.get_rank(), dist.get_world_size()
        device = torch.device(f"cuda:{local_rank}")
        set_seed(seed + rank)
    else:
        rank, world_size = 0, 1
        device = torch.device("cuda")
        set_seed(seed)
    return device, rank, world_size


def build_dataset(scenario: Scenario, args, config, num_video_frames):
    root = args.video_root_dir or scenario.video_root_dir
    csv = args.csv_path or scenario.csv_path

    # `remove_carnation` is forwarded everywhere, but only bites on one source:
    # `point_force_23000` is 12k `background_*` clips plus 11k `carnation_*` ones, and is the
    # only dataset with carnation rows at all. Only ForcePromptingDataset_PointForce
    # implements the filter; the other classes accept the kwarg and ignore it.
    return scenario.dataset_cls(
        video_root_dir=root,
        csv_path=csv,
        image_size=(args.height, args.width),
        stride=(1, 3),
        sample_n_frames=num_video_frames,
        is_validation_dataset=scenario.is_validation_dataset,
        remove_carnation=config.get("remove_carnation", False),
    ), root, csv


def main():
    args = parse_args()
    scenario = SCENARIOS[args.scenario]
    output_folder = args.output_folder or os.path.join("force_ode", args.scenario)

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device, rank, world_size = setup_distributed(args.seed)
    dtype = torch.bfloat16

    config = OmegaConf.merge(
        OmegaConf.load("configs/default_config.yaml"), OmegaConf.load(args.config_path)
    )
    config.model_kwargs.control_weight = args.control_weight
    config.gradient_checkpointing = False

    num_latent_frames = args.num_latent_frames
    num_video_frames = (num_latent_frames - 1) * 4 + 1  # block size is 4
    # using wan2.2 vae statistics
    seq_len = (num_latent_frames * (args.height // 16) * (args.width // 16)) // (2 * 2)
    image_or_video_shape = (1, num_latent_frames, 48, args.height // 16, args.width // 16)

    model = WanControlNet(config, device=device)
    if args.checkpoint_path:
        state_dict = torch.load(args.checkpoint_path, map_location="cpu")["generator"]
        rename = (
            lambda n: n.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        model.generator.to_empty(device="cpu")
        model.generator.load_state_dict(
            {rename(n): p for n, p in state_dict.items()}, strict=False
        )
        del state_dict
        if rank == 0:
            print("Finished loading generator checkpoint")
        if dist.is_initialized():
            dist.barrier()

    model.text_encoder = model.text_encoder.to(device=device, dtype=dtype)
    model.vae = model.vae.to(device=device, dtype=dtype)
    model = model.to(device=device, dtype=dtype)

    full_dataset, root, csv = build_dataset(scenario, args, config, num_video_frames)
    dataset_len = len(full_dataset)
    if rank == 0:
        print(f"scenario     : {args.scenario}")
        print(f"source       : {root}")
        print(f"csv          : {csv}")
        print(f"samples      : {dataset_len}")
        print(f"output       : {output_folder}")
    dataset_indices = list(range(dataset_len))
    dataset = torch.utils.data.Subset(full_dataset, dataset_indices[rank::world_size])

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        sampler=SequentialSampler(dataset),
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=collate_for(scenario.dataset_cls),
    )

    video_folder = output_folder + "_videos"
    if rank == 0:
        os.makedirs(output_folder, exist_ok=True)
        if not args.no_preview:
            os.makedirs(video_folder, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    for local_i, batch in tqdm(
        enumerate(dataloader), disable=(rank != 0), total=len(dataloader)
    ):
        ode_global_idx = dataset.indices[local_i]  # 0 ~ dataset_len-1
        output_path = os.path.join(output_folder, f"{ode_global_idx}.pt")
        if os.path.exists(output_path):
            continue

        if isinstance(batch, list):
            batch = batch[0]
        elif not isinstance(batch, dict):
            raise ValueError(f"Unexpected batch type: {type(batch)}")

        text_prompts = batch["prompts"]
        batch_size = len(text_prompts)

        hint = batch["controlnet_videos"].to(device=device, dtype=dtype)
        frames = batch["videos"].to(device=device, dtype=dtype)
        frames = frames.permute(0, 2, 1, 3, 4) / 255.0 * 2.0 - 1.0
        first_image = frames[:, :, :1]
        initial_latent = model.vae.encode_to_latent(first_image).to(device=device, dtype=dtype)

        # The force hint is 4 channels: [blob mask, magnitude, cos, sin]. The mask is frozen
        # at frame 0 and gates both the force field and the masked first frame beside it.
        gaussian_blob = hint[:, 0:1, 0:1]
        hint = hint[:, :, 1:] * (gaussian_blob > 1e-1)
        masked_image = (first_image + 1.0) / 2.0 * gaussian_blob.permute(0, 2, 1, 3, 4)
        masked_image = (masked_image * 2.0 - 1.0).permute(0, 2, 1, 3, 4)
        masked_latent = compress_time(masked_image, num_video_frames, method="subsample")
        hint = compress_time(hint, num_video_frames, method="subsample")

        conditional_dict = model.text_encoder(text_prompts=text_prompts)
        unconditional_dict = {
            k: v.detach()
            for k, v in model.text_encoder(
                text_prompts=[config.negative_prompt] * batch_size
            ).items()
        }

        latents = torch.randn(image_or_video_shape, device=device, dtype=dtype)
        latents[:, 0] = initial_latent[:, 0]  # keep the first frame clean

        latents_list = []
        inference_scheduler = model.inference_scheduler
        inference_scheduler.set_timesteps(num_inference_steps=48, device=device, shift=5.0)

        for t in tqdm(inference_scheduler.timesteps, disable=(rank != 0), leave=False):
            timestep = torch.ones([batch_size, seq_len], device=device, dtype=torch.float32) * t
            timestep[:, : seq_len // num_latent_frames] = 0.0  # initial frame is clean
            latents_list.append(latents)

            x = latents.permute(0, 2, 1, 3, 4)
            controls = model.generator.control_model(
                x, t=timestep, context=conditional_dict["prompt_embeds"],
                seq_len=seq_len, hint=hint, masked_latent=masked_latent,
            )
            controls = [c * args.control_weight for c in controls]

            noise_pred = model.generator.model(
                x, t=timestep, context=conditional_dict["prompt_embeds"],
                seq_len=seq_len, control=controls,
            )
            noise_pred_null = model.generator.model(
                x, t=timestep, context=unconditional_dict["prompt_embeds"],
                seq_len=seq_len, control=controls,
            )
            noise_pred = noise_pred_null + 5.0 * (noise_pred - noise_pred_null)

            temp_x0 = inference_scheduler.step(noise_pred, t, x, return_dict=False)[0]
            latents = temp_x0.permute(0, 2, 1, 3, 4)
            latents[:, 0] = initial_latent[:, 0]

        latents_list.append(latents)
        latents_list = torch.stack(latents_list, dim=1)
        latents_list = latents_list[:, [0, 12, 24, 36, -1]]

        if not args.no_preview:
            video = model.vae.decode_to_pixel(latents_list[:, -1])
            video = (video * 0.5 + 0.5).clamp(0, 1)
            video = rearrange(video, "b t c h w -> b t h w c").data.cpu().numpy()
            video_with_arrow = burn_overlay(
                scenario.overlay, video[0], batch, num_video_frames
            )
            write_video(
                os.path.join(video_folder, f"{ode_global_idx}.mp4"),
                video_with_arrow, fps=16,
            )

        torch.save(
            {
                scenario.index_key: dataset_indices[ode_global_idx],
                "latents_list": latents_list.cpu().detach(),
            },
            output_path,
        )

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
