import argparse
import torch
import os
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision.io import write_video
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler, Sampler

# controlnet + wan2.2 model
from pipeline.rolling_forcing_inference import ControlRollingForcingInferencePipeline

# force prompt datasets and utils
from utils.forceprompt_data.controlnet_datasets import (
    ForcePromptingDataset_WindForce,
    ForcePromptingDataset_PointForce,
    ForcePromptingDataset_PointForce_ChangeForce,
    ForcePromptingDataset_WindForce_ChangeForce,
)
from utils.forceprompt_data.data_utils import (
    collate_fn_ForcePromptingDataset_WindForce,
    collate_fn_ForcePromptingDataset_PointForce,
    collate_fn_ForcePromptingDataset_PointForce_ChangeForce,
    collate_fn_ForcePromptingDataset_WindForce_ChangeForce,
    add_aesthetic_wind_force_prompt_to_video,
    add_aesthetic_wind_force_change_prompt_to_video,
    add_aesthetic_point_force_prompt_to_video,
    add_aesthetic_point_force_change_prompt_to_video,
)
from utils.misc import set_seed, compress_time
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="Path to the config file")
parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint folder")
parser.add_argument("--output_folder", type=str, help="Output folder")
parser.add_argument(
    "--num_latent_frames",
    type=int,
    default=21,
    help="Number of the latent frames used for denoising",
)
parser.add_argument("--height", type=int, default=480, help="Height of the video")
parser.add_argument("--width", type=int, default=832, help="Width of the video")
parser.add_argument(
    "--use_ema", action="store_true", help="Whether to use EMA parameters"
)
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--control_weight", type=float, default=1.0, help="Control weight")
parser.add_argument(
    "--force_type",
    choices=["point_force", "wind_force", "point_force_change", "wind_force_change"],
    help="Which controlnet to use",
)
parser.add_argument(
    "--no_arrow",
    action="store_true",
    help="Skip add_aesthetic_* overlay and save raw generated video frames",
)
parser.add_argument(
    "--rolling_forcing_attention",
    action="store_true",
    default=True,
    help="Enable rolling-forcing-specific attention/cache update branch.",
)
parser.add_argument(
    "--rolling_forcing_block_frames",
    type=int,
    default=3,
    help="Number of latent frames treated as one rolling-forcing block in attention.",
)
parser.add_argument(
    "--rolling_forcing_max_frames",
    type=int,
    default=21,
    help="Max number of latent frames retained by rolling-forcing attention working cache.",
)
parser.add_argument(
    "--rolling_forcing_cache_frames",
    type=int,
    default=96,
    help="Allocated KV-cache capacity (latent frames) for rolling-forcing inference.",
)
args = parser.parse_args()
assert args.rolling_forcing_attention, "inference_causal_rolling_forcing.py must run with rolling forcing enabled."

# Initialize distributed inference
if "LOCAL_RANK" in os.environ:
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()
    set_seed(args.seed + local_rank)
else:
    device = torch.device("cuda")
    local_rank = 0
    world_size = 1
    set_seed(args.seed)

torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load("configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)
if "generator_model_kwargs" not in config:
    config.generator_model_kwargs = {}
config.generator_model_kwargs.rolling_forcing_attention = args.rolling_forcing_attention
config.generator_model_kwargs.rolling_forcing_block_frames = args.rolling_forcing_block_frames
config.generator_model_kwargs.rolling_forcing_max_frames = args.rolling_forcing_max_frames
config.rolling_forcing_cache_frames = args.rolling_forcing_cache_frames
config.model_kwargs.control_weight = args.control_weight
config.gradient_checkpointing = False

num_latent_frames = args.num_latent_frames
num_video_frames = (num_latent_frames - 1) * 4 + 1  # block size is 4
seq_len = (num_latent_frames * (args.height // 16) * (args.width // 16)) // (2 * 2)  # using wan2.2 vae statistics

pipeline = ControlRollingForcingInferencePipeline(config, device=device)

if args.checkpoint_path:
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    state_dict_generator = state_dict["generator_ema" if args.use_ema else "generator"]
    rename_param = (
        lambda name: name.replace("_fsdp_wrapped_module.", "")
        .replace("_checkpoint_wrapped_module.", "")
        .replace("_orig_mod.", "")
    )
    name_to_trainable_params = {}
    for n, p in state_dict_generator.items():
        renamed_n = rename_param(n)
        name_to_trainable_params[renamed_n] = p
    pipeline.generator.load_state_dict(name_to_trainable_params)
    del state_dict
    del name_to_trainable_params

pipeline = pipeline.to(device=device, dtype=torch.bfloat16)

# ---- sample data -------------------------------------------------------------------
# The repo ships six cases in assets/samples -- three point-force and three wind-force
# stills, the same ones the interactive demo offers as gallery presets. That is all the
# offline inference entry points read; nothing here depends on the private benchmark
# mounts the paper was evaluated on. See assets/samples/README.md.
SAMPLE_ROOT = "assets/samples"
_SAMPLE_SETS = {
    "point_force": (
        ForcePromptingDataset_PointForce, "point_force.csv",
        collate_fn_ForcePromptingDataset_PointForce,
    ),
    "wind_force": (
        ForcePromptingDataset_WindForce, "wind_force.csv",
        collate_fn_ForcePromptingDataset_WindForce,
    ),
    "point_force_change": (
        ForcePromptingDataset_PointForce_ChangeForce, "point_force_change.csv",
        collate_fn_ForcePromptingDataset_PointForce_ChangeForce,
    ),
    "wind_force_change": (
        ForcePromptingDataset_WindForce_ChangeForce, "wind_force_change.csv",
        collate_fn_ForcePromptingDataset_WindForce_ChangeForce,
    ),
}

_dataset_cls, _csv_name, collate_fn = _SAMPLE_SETS[args.force_type]
dataset = _dataset_cls(
    video_root_dir=os.path.join(SAMPLE_ROOT, "images"),
    csv_path=os.path.join(SAMPLE_ROOT, _csv_name),
    image_size=(args.height, args.width),
    stride=(1, 3),
    sample_n_frames=num_video_frames,
    is_validation_dataset=True,
)

# need to overwrite to values in the training dataset
dataset.min_force = 0.0
dataset.max_force = 1.0

num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts}")

if dist.is_initialized():
    class SubsetSequentialSampler(Sampler):
        def __init__(self, indices):
            self.indices = indices

        def __iter__(self):
            return iter(self.indices)

        def __len__(self):
            return len(self.indices)

    sampler_indices = list(range(local_rank, len(dataset), world_size))
    sampler = SubsetSequentialSampler(sampler_indices)
else:
    sampler = SequentialSampler(dataset)
    sampler_indices = None
dataloader = DataLoader(
    dataset,
    batch_size=1,
    sampler=sampler,
    num_workers=4,
    drop_last=False,
    collate_fn=collate_fn,
)

def prepare_hint(hint, image):
    image = image.to(device=hint.device)
    image = image.permute(0, 2, 1, 3, 4)
    image = image / 255.0
    image = image * 2.0 - 1.0

    # The force hint is 4 channels: [blob mask, magnitude, cos, sin]. The mask is frozen at
    # frame 0 and gates both the force field and the masked first frame beside it.
    gaussian_blob = hint[:, 0:1, 0:1]
    hint = hint[:, :, 1:] * (gaussian_blob > 1e-1)
    masked_image = (image + 1.0) / 2.0 * gaussian_blob.permute(0, 2, 1, 3, 4)
    masked_image = masked_image * 2.0 - 1.0
    masked_image = masked_image.permute(0, 2, 1, 3, 4)
    masked_latent = compress_time(masked_image, 81, method="subsample")
    return hint, masked_latent

# Create output directory (only on main process to avoid race conditions)
if local_rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)

if dist.is_initialized():
    dist.barrier()

for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    idx = sampler_indices[i] if sampler_indices is not None else i

    output_path = os.path.join(args.output_folder, f"{idx}.mp4")
    if os.path.exists(output_path):
        continue

    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]  # First (and only) item in the batch

    prompt = batch["prompts"][0]  # Get caption from batch
    hint = batch["controlnet_videos"]
    hint = hint.to(device=device, dtype=torch.bfloat16)
    image = batch["videos"][:, :1]
    # hint = torch.zeros_like(hint)
    hint, masked_latent = prepare_hint(hint, image)
    hint = hint.to(device=device, dtype=torch.bfloat16)
    masked_latent = masked_latent.to(device=device, dtype=torch.bfloat16)
    prompts = [prompt]

    hint = compress_time(hint, num_video_frames, method="subsample")

    video = pipeline.inference_rolling_forcing(
        image=image,
        text_prompts=prompts,
        num_frames=num_latent_frames,
        return_latents=False,
        hint=hint,
        masked_latent=masked_latent,
        # control_weight=args.control_weight,
    )
    video = rearrange(video, "b t c h w -> b t h w c").cpu()

    # Final output video
    video = video.data.cpu().numpy()

    # Clear VAE cache
    pipeline.vae.model.clear_cache()

    # Save the video if the current prompt is not a dummy prompt
    model_type = "regular" if not args.use_ema else "ema"
    min_force = dataset.min_force
    max_force = dataset.max_force
    if batch.get("force_1", None) is not None:
        normalized_force_1 = (batch["force_1"][0] - min_force) / (max_force - min_force)
        normalized_force_2 = (batch["force_2"][0] - min_force) / (max_force - min_force)
    else:
        normalized_force = (batch["force"][0] - min_force) / (max_force - min_force)
    if args.no_arrow:
        video_with_force_prompt_aesthetic = (video[0].copy() * 255).astype(np.uint8)
    elif args.force_type == "point_force":
        video_with_force_prompt_aesthetic = add_aesthetic_point_force_prompt_to_video(
            video[0], normalized_force, batch["angle"][0], batch["x_pos"][0], 1 - batch["y_pos"][0], num_frames_with_signal=num_video_frames,
        )
    elif args.force_type == "wind_force":
        video_with_force_prompt_aesthetic = add_aesthetic_wind_force_prompt_to_video(
            video[0], normalized_force, batch["angle"][0], num_frames_with_signal=num_video_frames,
        )
    elif args.force_type == "wind_force_change":
        video_with_force_prompt_aesthetic = add_aesthetic_wind_force_change_prompt_to_video(
            video[0], normalized_force_1, batch["angle_1"][0], normalized_force_2, batch["angle_2"][0], idx=num_video_frames//2, num_frames_with_signal=num_video_frames,
        )
    elif args.force_type == "point_force_change":
        if batch["change_at"][0] is not None:
            start_indice = int(batch["change_at"][0] * num_video_frames)
        else:
            start_indice = num_video_frames//2
        video_with_force_prompt_aesthetic = add_aesthetic_point_force_change_prompt_to_video(
            video[0], normalized_force_1, batch["angle_1"][0], batch["x_pos_1"][0], 1 - batch["y_pos_1"][0], normalized_force_2, batch["angle_2"][0], batch["x_pos_1"][0], 1 - batch["y_pos_1"][0], idx=start_indice, num_frames_with_signal=num_video_frames,
        )
    else:
        video_with_force_prompt_aesthetic = (video[0].copy() * 255).astype(np.uint8)
    output_path = os.path.join(args.output_folder, f"{idx}.mp4")
    write_video(output_path, video_with_force_prompt_aesthetic, fps=16)
