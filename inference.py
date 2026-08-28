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
from model import WanControlNet

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
    "--random_change_point",
    action="store_true",
    help="For *_change controls, sample a random change frame when change_at is not provided.",
)
parser.add_argument(
    "--text_only_inference",
    action="store_true",
    help="Run baseline text-only inference (hint and masked_latent are None, control_weight=0).",
)
parser.add_argument(
    "--no_arrow",
    action="store_true",
    help="Skip add_aesthetic_* overlay and save raw generated video frames",
)
args = parser.parse_args()

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
dtype = torch.bfloat16

config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load("configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)
effective_control_weight = 0.0 if args.text_only_inference else args.control_weight
config.model_kwargs.control_weight = effective_control_weight

num_latent_frames = args.num_latent_frames
num_video_frames = (num_latent_frames - 1) * 4 + 1  # block size is 4
seq_len = (num_latent_frames * (args.height // 16) * (args.width // 16)) // (2 * 2)  # using wan2.2 vae statistics

model = WanControlNet(config, device=device)

if args.checkpoint_path:
    print(f"Loading checkpoint from {args.checkpoint_path}")
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    state_dict_generator = state_dict["generator"]
    rename_param = (
        lambda name: name.replace("_fsdp_wrapped_module.", "")
        .replace("_checkpoint_wrapped_module.", "")
        .replace("_orig_mod.", "")
    )
    name_to_trainable_params = {}
    for n, p in state_dict_generator.items():
        renamed_n = rename_param(n)
        name_to_trainable_params[renamed_n] = p
    model.generator.to_empty(device="cpu")
    model.generator.load_state_dict(name_to_trainable_params)
    del state_dict
    del name_to_trainable_params

model = model.to(device=device, dtype=dtype)

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

class SubsetSequentialSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


if dist.is_initialized():
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

# Create output directory (only on main process to avoid race conditions)
if local_rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)

if dist.is_initialized():
    dist.barrier()

used_prompts_path = os.path.join(args.output_folder, "used_prompts.txt")


def _strip_trailing_non_alpha(text):
    stripped = text
    while stripped and stripped[-1].lower() not in "abcdefghijklmnopqrstuvwxyz":
        stripped = stripped[:-1]
    return stripped if stripped else text


def _angle_to_direction(angle, prefix=""):
    if (0 <= angle <= 22.5) or (337.5 <= angle <= 360):
        direction = "to the right"
    elif 22.5 <= angle <= 67.5:
        direction = "upwards and to the right"
    elif 67.5 <= angle <= 112.5:
        direction = "upwards"
    elif 112.5 <= angle <= 157.5:
        direction = "upwards and to the left"
    elif 157.5 <= angle <= 202.5:
        direction = "to the left"
    elif 202.5 <= angle <= 247.5:
        direction = "downwards and to the left"
    elif 247.5 <= angle <= 292.5:
        direction = "downwards"
    elif 292.5 <= angle <= 337.5:
        direction = "downwards and to the right"
    else:
        raise ValueError(f"Invalid angle: {angle}")
    return f"{prefix}{direction}" if prefix else direction


def _point_force_description(force):
    if 0 <= force <= 0.25:
        return "the object is moved not very forcefully"
    elif 0.25 < force <= 0.75:
        return "the object is moved forcefully"
    elif 0.75 < force <= 1.0:
        return "the object is moved very forcefully"
    else:
        raise ValueError(f"Invalid force value: {force}")


def _wind_force_description(force):
    if 0 <= force <= 0.125:
        return "the wind is very soft"
    elif 0.125 < force <= 0.4:
        return "the wind is soft"
    elif 0.4 < force <= 0.6667:
        return "the wind is medium strength"
    elif 0.6667 < force <= 1.0:
        return "the wind is very strong"
    else:
        raise ValueError(f"Invalid force value: {force}")


def get_text_prompt_point_force(prompt, force, angle):
    force_str = _point_force_description(force)
    angle_str = _angle_to_direction(angle)
    prompt_without_end_punctuation = _strip_trailing_non_alpha(prompt)
    return f"{prompt_without_end_punctuation}. {force_str.capitalize()}, {angle_str}"


def get_text_prompt_wind_force(prompt, force, angle):
    force_str = _wind_force_description(force)
    angle_str = _angle_to_direction(angle, prefix="blowing ")
    prompt_without_end_punctuation = _strip_trailing_non_alpha(prompt)
    return f"{prompt_without_end_punctuation}. {force_str.capitalize()}, {angle_str}"


def get_text_prompt_point_force_change(prompt, force_1, angle_1, force_2, angle_2):
    force_str_1 = _point_force_description(force_1)
    force_str_2 = _point_force_description(force_2)
    angle_str_1 = _angle_to_direction(angle_1)
    angle_str_2 = _angle_to_direction(angle_2)
    prompt_without_end_punctuation = _strip_trailing_non_alpha(prompt)
    return (
        f"{prompt_without_end_punctuation}. "
        f"First, {force_str_1}, {angle_str_1}; then, {force_str_2}, {angle_str_2}"
    )


def get_text_prompt_wind_force_change(prompt, force_1, angle_1, force_2, angle_2):
    force_str_1 = _wind_force_description(force_1)
    force_str_2 = _wind_force_description(force_2)
    angle_str_1 = _angle_to_direction(angle_1, prefix="blowing ")
    angle_str_2 = _angle_to_direction(angle_2, prefix="blowing ")
    prompt_without_end_punctuation = _strip_trailing_non_alpha(prompt)
    return (
        f"{prompt_without_end_punctuation}. "
        f"First, {force_str_1}, {angle_str_1}; then, {force_str_2}, {angle_str_2}"
    )


def append_force_prompt(prompt, batch, force_type, min_force, max_force):
    if not args.text_only_inference:
        return prompt

    if max_force == min_force:
        return prompt

    if force_type in ["point_force", "point_force_diverse"]:
        normalized_force = (batch["force"][0] - min_force) / (max_force - min_force)
        return get_text_prompt_point_force(prompt, normalized_force, batch["angle"][0])

    if force_type == "point_force_change":
        if "force_1" in batch and "force_2" in batch and "angle_1" in batch and "angle_2" in batch:
            normalized_force_1 = (batch["force_1"][0] - min_force) / (max_force - min_force)
            normalized_force_2 = (batch["force_2"][0] - min_force) / (max_force - min_force)
            return get_text_prompt_point_force_change(
                prompt, normalized_force_1, batch["angle_1"][0], normalized_force_2, batch["angle_2"][0]
            )
        if "force" in batch and "angle" in batch:
            normalized_force = (batch["force"][0] - min_force) / (max_force - min_force)
            return get_text_prompt_point_force(prompt, normalized_force, batch["angle"][0])
        return prompt

    if force_type in ["wind_force", "wind_force_diverse"]:
        normalized_force = (batch["force"][0] - min_force) / (max_force - min_force)
        return get_text_prompt_wind_force(prompt, normalized_force, batch["angle"][0])

    if force_type == "wind_force_change":
        if "force_1" in batch and "force_2" in batch and "angle_1" in batch and "angle_2" in batch:
            normalized_force_1 = (batch["force_1"][0] - min_force) / (max_force - min_force)
            normalized_force_2 = (batch["force_2"][0] - min_force) / (max_force - min_force)
            return get_text_prompt_wind_force_change(
                prompt, normalized_force_1, batch["angle_1"][0], normalized_force_2, batch["angle_2"][0]
            )
        if "force" in batch and "angle" in batch:
            normalized_force = (batch["force"][0] - min_force) / (max_force - min_force)
            return get_text_prompt_wind_force(prompt, normalized_force, batch["angle"][0])
        return prompt

    return prompt


def _parse_change_at(change_at_value):
    if change_at_value is None:
        return None
    try:
        if isinstance(change_at_value, torch.Tensor):
            if change_at_value.numel() == 0:
                return None
            change_at_value = change_at_value.flatten()[0].item()
        elif isinstance(change_at_value, (list, tuple)):
            if len(change_at_value) == 0:
                return None
            change_at_value = change_at_value[0]
        change_at_value = float(change_at_value)
        if np.isnan(change_at_value):
            return None
        return change_at_value
    except (TypeError, ValueError):
        return None


def _choose_change_index(batch, num_frames, use_random_when_missing=False):
    # Priority: explicit change_at from dataset -> random (optional) -> middle fallback.
    change_at = _parse_change_at(batch.get("change_at"))
    if change_at is not None:
        change_at = max(0.0, min(1.0, change_at))
        return min(int(change_at * num_frames), num_frames - 1)

    if use_random_when_missing and num_frames > 2:
        return int(np.random.randint(1, num_frames - 1))

    return num_frames // 2


def do_original_wan_inference(idx, model, batch):
    # this part only uses WanControlNet().genertor.model -> ControlledWanModel with no control input -> same as using Wan2.2-TI2V-5B WanModel for inference
    prompt = batch["prompts"][0]  # Get caption from batch
    prompts = [prompt]
    image = batch["videos"][:, :1]
    image = image.to(device=model.device)
    image = image.permute(0, 2, 1, 3, 4)
    image = image / 255.0
    image = image * 2.0 - 1.0

    initial_latent = model.vae.encode_to_latent(
        image.to(model.device, model.generator.model.dtype)
    ).to(device=model.device, dtype=model.generator.model.dtype)
    z = [initial_latent[:, 0].permute(1, 0, 2, 3)]
    noise = torch.randn(
        [48, num_latent_frames, args.height // 16, args.width // 16],
        device=model.device,
        dtype=model.generator.model.dtype,
    ) # [48, 21, 28, 52] for default

    latent = noise
    from wan.utils.utils import masks_like

    mask1, mask2 = masks_like([noise], zero=True)
    latent = (1.0 - mask2[0]) * z[0] + mask2[0] * latent

    context = [model.text_encoder(prompts)["prompt_embeds"][0]]
    context = [t.to(model.device, model.generator.model.dtype) for t in context]

    context_null = [
        model.text_encoder(
            "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
        )["prompt_embeds"][0]
    ]
    context_null = [
        t.to(model.device, model.generator.model.dtype) for t in context_null
    ]

    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

    sample_scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=1000, shift=1, use_dynamic_shifting=False
    )
    sample_scheduler.set_timesteps(50, device=model.device, shift=5.0)

    timesteps = sample_scheduler.timesteps

    # timesteps = model.scheduler.timesteps
    from tqdm import tqdm

    for t in tqdm(timesteps):
        latent_model_input = [latent.to(model.device)]
        timestep = [t]
        timestep = torch.stack(timestep).to(model.device, model.generator.model.dtype)

        temp_ts = (mask2[0][0][:, ::2, ::2] * timestep).flatten()
        temp_ts = torch.cat(
            [temp_ts, temp_ts.new_ones(seq_len - temp_ts.size(0)) * timestep]
        )
        timestep = temp_ts.unsqueeze(0)

        noise_pred = model.generator.model(
            latent_model_input, t=timestep, context=context, seq_len=seq_len
        )[0]

        noise_pred_null = model.generator.model(
            latent_model_input, t=timestep, context=context_null, seq_len=seq_len
        )[0]

        noise_pred = noise_pred_null + 5.0 * (noise_pred - noise_pred_null)

        temp_x0 = sample_scheduler.step(
            noise_pred.unsqueeze(0),
            t,
            latent.unsqueeze(0),
            return_dict=False,
        )[0]

        latent = temp_x0.squeeze(0)
        latent = (1.0 - mask2[0]) * z[0] + mask2[0] * latent

    video = model.vae.decode_to_pixel(latent.unsqueeze(0).permute(0, 2, 1, 3, 4))
    video = (video * 0.5 + 0.5).clamp(0, 1)

    # Final output video
    return video

def prepare_hint(hint, image):
    image = image.to(device=hint.device, dtype=dtype)
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


def log_used_prompt(output_path, prompt):
    prompt_one_line = " ".join(str(prompt).splitlines()).strip()
    with open(used_prompts_path, "a", encoding="utf-8") as f:
        f.write(f"{os.path.basename(output_path)}: {prompt_one_line}\n")

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
    hint = hint.to(device=device, dtype=dtype)
    image = batch["videos"][:, :1].to(device=device, dtype=dtype)
    current_force_type = args.force_type
    min_force = dataset.min_force
    max_force = dataset.max_force
    prompt = append_force_prompt(
        prompt=prompt,
        batch=batch,
        force_type=current_force_type,
        min_force=min_force,
        max_force=max_force,
    )
    if args.text_only_inference:
        hint = None
        masked_latent = None
    else:
        hint, masked_latent = prepare_hint(hint, image)
    prompts = [prompt]

    if hint is not None:
        hint = compress_time(hint, num_video_frames, method="subsample")

    if not args.text_only_inference:
        video = model.inference(
            image=image,
            input_prompts=prompts,
            num_frames=num_latent_frames,
            return_latents=False, 
            hint=hint,
            masked_latent=masked_latent,
            control_weight=effective_control_weight,
        )
    else:
        video = do_original_wan_inference(idx, model, batch)
    
    video = rearrange(video, "b t c h w -> b t h w c").cpu()

    # Final output video
    video = video.data.cpu().numpy()

    # Clear VAE cache
    model.vae.model.clear_cache()

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
        start_indice = _choose_change_index(
            batch,
            num_video_frames,
            use_random_when_missing=args.random_change_point,
        )
        video_with_force_prompt_aesthetic = add_aesthetic_wind_force_change_prompt_to_video(
            video[0], normalized_force_1, batch["angle_1"][0], normalized_force_2, batch["angle_2"][0], idx=start_indice, num_frames_with_signal=num_video_frames,
        )
    elif args.force_type == "point_force_change":
        start_indice = _choose_change_index(
            batch,
            num_video_frames,
            use_random_when_missing=args.random_change_point,
        )
        video_with_force_prompt_aesthetic = add_aesthetic_point_force_change_prompt_to_video(
            video[0], normalized_force_1, batch["angle_1"][0], batch["x_pos_1"][0], 1 - batch["y_pos_1"][0], normalized_force_2, batch["angle_2"][0], batch["x_pos_1"][0], 1 - batch["y_pos_1"][0], idx=start_indice, num_frames_with_signal=num_video_frames,
        )
    output_path = os.path.join(args.output_folder, f"{idx}.mp4")
    write_video(output_path, video_with_force_prompt_aesthetic, fps=16)
    if args.text_only_inference:
        log_used_prompt(output_path, prompts[0])
