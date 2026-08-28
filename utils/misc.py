import numpy as np
import random
import torch
from einops import rearrange


def set_seed(seed: int, deterministic: bool = False):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.

    Args:
        seed (`int`):
            The seed to set.
        deterministic (`bool`, *optional*, defaults to `False`):
            Whether to use deterministic algorithms where available. Can slow down training.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)


def merge_dict_list(dict_list):
    if len(dict_list) == 1:
        return dict_list[0]

    merged_dict = {}
    for k, v in dict_list[0].items():
        if isinstance(v, torch.Tensor):
            if v.ndim == 0:
                merged_dict[k] = torch.stack([d[k] for d in dict_list], dim=0)
            else:
                merged_dict[k] = torch.cat([d[k] for d in dict_list], dim=0)
        else:
            # for non-tensor values, we just copy the value from the first item
            merged_dict[k] = v
    return merged_dict


def compress_time(x, num_frames, method="avg_pool1d"):
    # x = rearrange(x, '(b f) c h w -> b f c h w', f=num_frames)
    batch_size, frames, channels, height, width = x.shape
    x = rearrange(x, 'b f c h w -> (b h w) c f')
        
    if x.shape[-1] % 2 == 1:
        x_first, x_rest = x[..., 0], x[..., 1:]
        if x_rest.shape[-1] > 0:
            if method == "subsample":
                x_rest = x_rest[..., ::4]
            elif method == "avg_pool1d":
                x_rest = torch.nn.functional.avg_pool1d(x_rest, kernel_size=4, stride=4)

        x = torch.cat([x_first[..., None], x_rest], dim=-1)
    else:
        if method == "subsample":
            x = x[..., ::4]
        elif method == "avg_pool1d":
            x = torch.nn.functional.avg_pool1d(x, kernel_size=4, stride=4)
    x = rearrange(x, '(b h w) c f -> b f c h w', b=batch_size, h=height, w=width)
    return x


def cycle(dl):
    """Repeat a dataloader forever."""
    while True:
        for data in dl:
            yield data
