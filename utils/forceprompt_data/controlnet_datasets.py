import os
import glob
import random


import torch
import math
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
from decord import VideoReader
from torch.utils.data.dataset import Dataset
from utils.forceprompt_data.data_utils import filter_df_by_split_label
# from controlnet_aux import CannyDetector, HEDdetector

def unpack_mm_params(p):
    if isinstance(p, (tuple, list)):
        return p[0], p[1]
    elif isinstance(p, (int, float)):
        return p, p
    raise Exception(f'Unknown input parameter type.\nParameter: {p}.\nType: {type(p)}')


def resize_for_crop(image, min_h, min_w):
    img_h, img_w = image.shape[-2:]
    
    # Calculate the scaling coefficients
    h_coef = min_h / img_h
    w_coef = min_w / img_w
    
    if img_h >= min_h and img_w >= min_w:
        # Both dimensions are larger, scale down to the minimum required size
        coef = max(h_coef, w_coef)
    elif img_h <= min_h and img_w <= min_w:
        # Both dimensions are smaller, scale up to the minimum required size
        coef = max(h_coef, w_coef)
    else:
        # Mixed case - one dimension is larger, one is smaller
        # Scale up to ensure both dimensions meet the minimum
        coef = max(h_coef, w_coef)
    
    # Calculate new dimensions
    out_h = int(img_h * coef)
    out_w = int(img_w * coef)
    
    # Ensure dimensions are at least the minimum required
    # This handles cases where rounding down during int conversion drops below minimum
    if out_h < min_h:
        # print(f"out_h < min_h: {out_h} < {min_h}")
        out_h = min_h
    if out_w < min_w:
        # print(f"out_w < min_w: {out_w} < {min_w}")
        out_w = min_w
    
    resized_image = transforms.functional.resize(image, (out_h, out_w), antialias=True)
    return resized_image


def load_controlnet_signal_wind_force(force, angle, num_frames=49, num_channels=3, height=480, width=720, min_force=0.0, max_force=1.0):

    controlnet_signal = torch.zeros((num_frames, num_channels, height, width)) # (49, 3, 480, 720)

    # first channel gets wind_speed
    controlnet_signal[:, 0] = -1 + 2*(force-min_force)/(max_force-min_force)

    # second channel gets cos(wind_angle)
    controlnet_signal[:, 1] = math.cos(angle * torch.pi / 180.0)

    # third channel gets sin(wind_angle)
    controlnet_signal[:, 2] = math.sin(angle * torch.pi / 180.0)
        
    return controlnet_signal

def get_gaussian_blob(x, y, radius=10, amplitude=1.0, shape=(3, 480, 720), device=None):
        """
        Create a tensor containing a Gaussian blob at the specified location.
        
        Args:
            x (int): x-coordinate of the blob center
            y (int): y-coordinate of the blob center
            radius (int, optional): Radius of the Gaussian blob. Defaults to 10.
            amplitude (float, optional): Maximum intensity of the blob. Defaults to 1.0.
            shape (tuple, optional): Shape of the output tensor (channels, height, width). Defaults to (3, 480, 720).
            device (torch.device, optional): Device to create the tensor on. Defaults to None.
        
        Returns:
            torch.Tensor: Tensor of shape (channels, height, width) containing the Gaussian blob
        """
        num_channels, height, width = shape
        
        # Create a new tensor filled with zeros
        blob_tensor = torch.zeros(shape, device=device)
        
        # Create coordinate grids
        y_grid, x_grid = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing='ij'
        )
        
        # Calculate squared distance from (x, y)
        squared_dist = (x_grid - x) ** 2 + (y_grid - y) ** 2
        
        # Create Gaussian blob using the squared distance
        gaussian = amplitude * torch.exp(-squared_dist / (2.0 * radius ** 2))
        
        # Add the Gaussian blob to all channels
        for c in range(num_channels):
            blob_tensor[c] = gaussian
        
        return blob_tensor


# The canvas the training signal is drawn on. The loaders never pass height/width, so the
# blob is drawn at this size and resize_for_crop()d up to the target afterwards -- which
# magnifies it. A caller that draws straight at the target resolution (the demo) has to
# scale the radius to match, hence `scale` below.
_BLOB_CANVAS_H, _BLOB_CANVAS_W = 480, 720
_BLOB_RADIUS_PX = 20.0


def load_point_force_mask(x_pos, y_pos, num_frames=49, height=_BLOB_CANVAS_H, width=_BLOB_CANVAS_W):
    """Static Gaussian blob marking *where* a local force is applied.

    Magnitude and direction live in the wind-style channels (see
    `load_controlnet_signal_wind_force`); this channel is only a spatial mask, which is what
    lets point and wind share one representation.

    This used to be a blob travelling along the force vector -- the original Force-Prompting
    representation -- but every reader freezes it at frame 0 ("do not let the blob move") and
    broadcasts it over the clip, so the trajectory was built and thrown away. Frame 0 is
    bit-identical to this and independent of force/angle.
    """
    scale = max(height / _BLOB_CANVAS_H, width / _BLOB_CANVAS_W)
    blob = get_gaussian_blob(
        x=x_pos * width,
        y=(1 - y_pos) * height,
        radius=_BLOB_RADIUS_PX * scale,
        amplitude=1.0,
        shape=(1, height, width),
    )
    return blob.unsqueeze(0).expand(num_frames, -1, -1, -1)


class BaseClass(Dataset):
    def __init__(
            self, 
            video_root_dir,
            image_size=(320, 512), 
            stride=(1, 2), 
            sample_n_frames=25,
            remove_carnation=False,
        ):
        self.height, self.width = unpack_mm_params(image_size)
        self.stride_min, self.stride_max = unpack_mm_params(stride)
        self.video_root_dir = video_root_dir
        self.sample_n_frames = sample_n_frames
        
        self.length = 0
        
        self.remove_carnation = remove_carnation

    def load_pixel_values_image(self, image_path):

        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        # image = image.resize((self.width, self.height), Image.LANCZOS)
        np_image = np.array(image) # (480, 720, 3)
        pixel_values = torch.from_numpy(np_image).permute(2, 0, 1).contiguous().unsqueeze(0) # (1, 3, 480, 720)
        pixel_values = pixel_values.repeat(self.sample_n_frames, 1, 1, 1)
        # pixel_values = pixel_values / 127.5 - 1

        return pixel_values
        
    def __len__(self):
        return self.length
        
    def get_batch(self, idx):
        raise Exception('Get batch method is not realized.')

    def __getitem__(self, idx):
        raise Exception('Get item method is not realized.')


class ForcePromptingDataset_PointForce(BaseClass):
    def __init__(
        self,
        csv_path,
        is_validation_dataset=False,
        split_label=None,
        split_column="is_train_val",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.is_validation_dataset = is_validation_dataset

        if is_validation_dataset:
            self.media_type = "image"
            blob_ext =  "*.png"
        else:
            self.media_type = "video"
            blob_ext = "*.mp4"
            
        file_paths = glob.glob(os.path.join(self.video_root_dir, blob_ext))
        file_names = set([os.path.basename(x) for x in file_paths]) # list of videos or images...
        self.df = pd.read_csv(csv_path)

        # only keep the rows in the csv whose videos we can find
        self.df['checked'] = self.df[self.media_type].map(lambda x, files=file_names: int(x in files))
        self.df = self.df[self.df['checked'] == True]
        self.df = filter_df_by_split_label(
            self.df,
            split_label=split_label,
            split_column=split_column,
            csv_path=csv_path,
        )

        self.min_force = float(self.df["force"].min())
        self.max_force = float(self.df["force"].max())

        self.length = self.df.shape[0]

        if self.remove_carnation:
            self.df = self.df[~self.df[self.media_type].str.contains('carnation')]
            self.length = self.df.shape[0]

    def get_batch(self, idx):

        item = self.df.iloc[idx]
        caption = item['caption']
        file_name = item[self.media_type]
        force = item['force']
        angle = item['angle']
        file_path = os.path.join(self.video_root_dir, file_name)

        if self.media_type == "image":
            pixel_values = self.load_pixel_values_image(file_path) # (1, 3, 480, 720) of torch.float32 in [-1, 1]
            file_id = file_name.split(".png")[0]
            x_pos = item["coordx"] / item["width"]
            y_pos = item["coordy"] / item["height"]

        elif self.media_type == "video":
            pixel_values = self.load_pixel_values_video(file_path) # (49, 3, 480, 720) of torch.float32 in [-1, 1]
            # tensor_to_video_ffmpeg(0.5 + pixel_values/2, "pixel_values.mp4", fps=10)
            file_id = file_name.split(".mp4")[0]

            # AUTOMATIC RAMDOM CROPPING PROCEDURE, but only for the carnation...
            if file_id.startswith("carnation"):
                default_height = 480
                default_width = 720
                crop_zoom_amount =  np.random.uniform(1.0, 1.3) # 1.0 means no zoom; 1.3 means zoom in 1.3x
                new_width = int(item["width"] / crop_zoom_amount) - int(item["width"] / crop_zoom_amount) % 2
                new_height = int(item["height"] / crop_zoom_amount) - int(item["height"] / crop_zoom_amount) % 2

                num_tries = 0
                while num_tries < 100:
                    new_origin_x_pos = int(np.random.uniform(0, item["width"] - item["width"]/crop_zoom_amount))
                    new_origin_y_pos = int(np.random.uniform(0, item["height"] - item["height"]/crop_zoom_amount))

                    if item["coordx"] in range(new_origin_x_pos + 50, new_origin_x_pos+new_width - 50) and item["coordy"] in range(new_origin_y_pos + 50, new_origin_y_pos+new_height - 50):
                        num_tries = 100
                    num_tries += 1
                pixel_values = pixel_values[:, :, item["height"] - (new_origin_y_pos+new_height):item["height"] - new_origin_y_pos, new_origin_x_pos:new_origin_x_pos+new_width]
                pixel_values = resize_for_crop(pixel_values, default_height, default_width) # (49, 3, 480, 720)

                # tensor_to_video_ffmpeg(0.5 + new_pixel_values/2, "pixel_values_new.mp4", fps=10)
                new_x_pos = (new_width / item["width"]) * (item["coordx"] + new_origin_x_pos)
                new_y_pos = (new_height / item["height"]) * (item["coordy"] - new_origin_y_pos)

                new_r = (crop_zoom_amount / 2) * math.sqrt((item["coordx"] - new_origin_x_pos)**2 + (item["coordy"] - new_origin_y_pos)**2)
                new_theta = math.atan((item["coordy"] - new_origin_y_pos) / (item["coordx"] - new_origin_x_pos))

                new_x_pos = int(new_r * math.cos(new_theta))
                new_y_pos = default_height - int(new_r * math.sin(new_theta))

                x_pos = new_x_pos / default_width
                y_pos = 1 - new_y_pos / default_height
            
            else:
                x_pos = item["coordx"] / item["width"]
                y_pos = item["coordy"] / item["height"]

            # new_pixel_values_with_blob = torch.clip(new_pixel_values + 10* self.get_gaussian_blob(x=new_x_pos, y=new_y_pos, radius=10, amplitude=1.0, shape=(3, 480, 720)), max=1.0)
            # tensor_to_video_ffmpeg(0.5 + new_pixel_values_with_blob/2, "pixel_values_new_with_blob.mp4", fps=10)

        controlnet_signal = torch.cat([
            load_point_force_mask(x_pos, y_pos, num_frames=self.sample_n_frames),
            load_controlnet_signal_wind_force(force, angle, num_frames=self.sample_n_frames, min_force=self.min_force, max_force=self.max_force),
        ], dim=1)

        return pixel_values, caption, controlnet_signal, force, angle, x_pos, y_pos, file_id

    def __getitem__(self, idx):
        while True:
            try:
                pixel_values, caption, controlnet_signal, force, angle, x_pos, y_pos, file_id = self.get_batch(idx)
                # video, caption, controlnet_video = self.get_batch(idx)
                break
            except Exception as e:
                print("EXCEPTION HERE!!!", e) # this prints 'text' incessantly
                idx = random.randint(0, self.length - 1)
            
        pixel_values = [
            resize_for_crop(x, self.height, self.width) for x in [pixel_values]
        ][0]
        original_width, original_height = pixel_values.shape[3], pixel_values.shape[2]
        pixel_values = [
            transforms.functional.center_crop(x, (self.height, self.width)) for x in [pixel_values]
        ][0]
        x_pos = ((x_pos * original_width) - (original_width - self.width) / 2) / self.width
        y_pos = ((y_pos * original_height) - (original_height - self.height) / 2) / self.height

        controlnet_signal = resize_for_crop(controlnet_signal, self.height, self.width)
        controlnet_signal = transforms.functional.center_crop(controlnet_signal, (self.height, self.width))

        data = {
            'file_id' : file_id,
            'video': pixel_values, 
            'caption': caption, 
            'controlnet_video': controlnet_signal,
            'force': force,
            'angle': angle,
            'x_pos': x_pos,
            'y_pos': y_pos,
            'force_type': 'point_force',
        }
        return data

    def load_pixel_values_video(self, video_path):

        video_reader = VideoReader(video_path)
        n_available = len(video_reader)

        # Start from the frame where the force is actually applied. Clamp the
        # start (and the number of frames read) to what the mp4 actually has, so
        # sample_n_frames larger than the clip length (e.g. num_latent_frames=42
        # -> 165 frames on a ~120-frame clip) no longer indexes out of bounds.
        start = min(10, max(0, n_available - 1))
        n_take = min(self.sample_n_frames, n_available - start)
        indices = np.arange(start, start + n_take, dtype=int)

        # Get the selected frames
        np_video = video_reader.get_batch(indices).asnumpy() # (49, 960, 1440, 3)
        pixel_values = torch.from_numpy(np_video).permute(0, 3, 1, 2).contiguous() # (49, 3, 960, 1440) of uint8 in [0, 255]

        # Pad by repeating the last available frame when the clip is shorter than
        # sample_n_frames. For I2V inference only the first frame is used as the
        # init image, but downstream shapes still expect sample_n_frames frames.
        if n_take < self.sample_n_frames:
            pad_n = self.sample_n_frames - n_take
            last = pixel_values[-1:].expand(pad_n, *pixel_values.shape[1:])
            pixel_values = torch.cat([pixel_values, last], dim=0)
        # pixel_values = pixel_values / 127.5 - 1 # (49, 3, 960, 1440) of torch.float32 in [-1, 1]
        del video_reader

        return pixel_values


class ForcePromptingDataset_PointForce_ChangeForce(BaseClass):
    def __init__(
        self,
        csv_path,
        is_validation_dataset=False,
        randomize_change_point=False,
        min_change_ratio=0.3,
        max_change_ratio=0.7,
        split_label=None,
        split_column="is_train_val",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.is_validation_dataset = is_validation_dataset
        self.apply_first_force_at = 10 # frame index to apply the first force
        self.apply_second_force_at = 60 # frame index to apply the second force
        self.randomize_change_point = randomize_change_point
        self.min_change_ratio = min_change_ratio
        self.max_change_ratio = max_change_ratio

        if is_validation_dataset:
            self.media_type = "image"
            blob_ext =  "*.png"
        else:
            self.media_type = "video"
            blob_ext = "*.mp4"
            
        file_paths = glob.glob(os.path.join(self.video_root_dir, blob_ext))
        file_names = set([os.path.basename(x) for x in file_paths]) # list of videos or images...
        self.df = pd.read_csv(csv_path)

        # only keep the rows in the csv whose videos we can find
        self.df['checked'] = self.df[self.media_type].map(lambda x, files=file_names: int(x in files))
        self.df = self.df[self.df['checked'] == True]
        self.df = filter_df_by_split_label(
            self.df,
            split_label=split_label,
            split_column=split_column,
            csv_path=csv_path,
        )

        self.min_force = min(float(self.df["force1"].min()), float(self.df["force2"].min()))
        self.max_force = max(float(self.df["force1"].max()), float(self.df["force2"].max()))

        self.length = self.df.shape[0]

    def _clamp_split_index(self, split_index):
        min_split = max(1, int(round(self.sample_n_frames * self.min_change_ratio)))
        max_split = min(
            self.sample_n_frames - 1,
            int(round(self.sample_n_frames * self.max_change_ratio)),
        )
        if min_split > max_split:
            min_split, max_split = 1, self.sample_n_frames - 1
        return int(np.clip(split_index, min_split, max_split))

    def _sample_window_and_split(self, change_at=None):
        if change_at is not None:
            split_index = int(change_at * self.sample_n_frames)
            split_index = self._clamp_split_index(split_index)
            window_start = self.apply_second_force_at - split_index
            return max(0, window_start), split_index

        if self.randomize_change_point:
            min_split = max(1, int(round(self.sample_n_frames * self.min_change_ratio)))
            max_split = min(
                self.sample_n_frames - 1,
                int(round(self.sample_n_frames * self.max_change_ratio)),
            )
            if min_split > max_split:
                min_split, max_split = 1, self.sample_n_frames - 1
            split_index = random.randint(min_split, max_split)
        else:
            split_index = self.sample_n_frames // 2

        split_index = self._clamp_split_index(split_index)
        window_start = self.apply_second_force_at - split_index
        return max(0, window_start), split_index

    def get_batch(self, idx):
        item = self.df.iloc[idx]
        caption = item['caption']
        file_name = item[self.media_type]
        force_1 = item['force1']
        angle_1 = item['angle1']
        force_2 = item['force2']
        angle_2 = item['angle2']
        file_path = os.path.join(self.video_root_dir, file_name)

        change_at = None
        if 'change_at' in self.df.columns:
            change_at = item['change_at']
        window_start, split_index = self._sample_window_and_split(change_at=change_at)

        if self.media_type == "image":
            pixel_values = self.load_pixel_values_image(file_path) # (1, 3, 480, 720) of torch.float32 in [-1, 1]
            file_id = file_name.split(".png")[0]
            x_pos_1 = item["coordx1"] / item["width"]
            y_pos_1 = item["coordy1"] / item["height"]
            x_pos_2 = item["coordx2"] / item["width"]
            y_pos_2 = item["coordy2"] / item["height"]

        elif self.media_type == "video":
            pixel_values = self.load_pixel_values_video(file_path, start_indice=window_start) # (49, 3, 480, 720) of torch.float32 in [-1, 1]
            # tensor_to_video_ffmpeg(0.5 + pixel_values/2, "pixel_values.mp4", fps=10)
            file_id = file_name.split(".mp4")[0]

            x_pos_1 = item["coordx1"] / item["width"]
            y_pos_1 = item["coordy1"] / item["height"]
            x_pos_2 = item["coordx2"] / item["width"]
            y_pos_2 = item["coordy2"] / item["height"]

        # The mask marks where force 1 is applied; both forces act at the same point.
        controlnet_signal = torch.cat([
            load_point_force_mask(x_pos_1, y_pos_1, num_frames=self.sample_n_frames),
            self.load_controlnet_signal_wind(force_1, force_2, angle_1, angle_2, num_frames=self.sample_n_frames, split_index=split_index),
        ], dim=1)
            
        return pixel_values, caption, controlnet_signal, force_1, angle_1, x_pos_1, y_pos_1, force_2, angle_2, x_pos_2, y_pos_2, file_id, change_at


    def __getitem__(self, idx):
        while True:
            try:
                pixel_values, caption, controlnet_signal, force_1, angle_1, x_pos_1, y_pos_1, force_2, angle_2, x_pos_2, y_pos_2, file_id, change_at = self.get_batch(idx)
                # video, caption, controlnet_video = self.get_batch(idx)
                break
            except Exception as e:
                print("EXCEPTION HERE!!!", e) # this prints 'text' incessantly
                idx = random.randint(0, self.length - 1)
            
        pixel_values = [
            resize_for_crop(x, self.height, self.width) for x in [pixel_values]
        ][0]
        original_width, original_height = pixel_values.shape[3], pixel_values.shape[2]
        pixel_values = [
            transforms.functional.center_crop(x, (self.height, self.width)) for x in [pixel_values]
        ][0]
        x_pos_1 = ((x_pos_1 * original_width) - (original_width - self.width) / 2) / self.width
        y_pos_1 = ((y_pos_1 * original_height) - (original_height - self.height) / 2) / self.height
        x_pos_2 = ((x_pos_2 * original_width) - (original_width - self.width) / 2) / self.width
        y_pos_2 = ((y_pos_2 * original_height) - (original_height - self.height) / 2) / self.height

        controlnet_signal = resize_for_crop(controlnet_signal, self.height, self.width)
        controlnet_signal = transforms.functional.center_crop(controlnet_signal, (self.height, self.width))

        data = {
            'file_id' : file_id,
            'video': pixel_values, 
            'caption': caption, 
            'controlnet_video': controlnet_signal,
            'force_1': force_1,
            'angle_1': angle_1,
            'x_pos_1': x_pos_1,
            'y_pos_1': y_pos_1,
            'force_2': force_2,
            'angle_2': angle_2,
            'x_pos_2': x_pos_2,
            'y_pos_2': y_pos_2,
            'force_type': 'point_force_change',
            'change_at': change_at,
        }
        return data

    def load_controlnet_signal_wind(self, force_1, force_2, angle_1, angle_2, num_frames=49, num_channels=3, height=480, width=720, split_index=0):
        
        controlnet_signal = torch.zeros((num_frames, num_channels, height, width))

        half = self._clamp_split_index(split_index)
        controlnet_signal[:half, 0] = -1 + 2*(force_1-self.min_force)/(self.max_force-self.min_force)
        controlnet_signal[:half, 1] = math.cos(angle_1 * torch.pi / 180.0)
        controlnet_signal[:half, 2] = math.sin(angle_1 * torch.pi / 180.0)

        controlnet_signal[half:, 0] = -1 + 2*(force_2-self.min_force)/(self.max_force-self.min_force)
        controlnet_signal[half:, 1] = math.cos(angle_2 * torch.pi / 180.0)
        controlnet_signal[half:, 2] = math.sin(angle_2 * torch.pi / 180.0)

        return controlnet_signal

    def load_pixel_values_video(self, video_path, start_indice=20):

        video_reader = VideoReader(video_path)

        # if "carnation" in video_path:
        #     indices = np.array([2*i for i in range(self.sample_n_frames)], dtype=int)
        # else:
        indices = np.array([i for i in range(start_indice, start_indice + self.sample_n_frames)], dtype=int) # align sampled video window with force-change split
        
        # Get the selected frames
        np_video = video_reader.get_batch(indices).asnumpy() # (49, 960, 1440, 3)
        pixel_values = torch.from_numpy(np_video).permute(0, 3, 1, 2).contiguous() # (49, 3, 960, 1440) of uint8 in [0, 255]
        # pixel_values = pixel_values / 127.5 - 1 # (49, 3, 960, 1440) of torch.float32 in [-1, 1]
        del video_reader

        return pixel_values

    def get_gaussian_blob(self, x, y, radius=10, amplitude=1.0, shape=(3, 480, 720), device=None):
        """
        Create a tensor containing a Gaussian blob at the specified location.
        
        Args:
            x (int): x-coordinate of the blob center
            y (int): y-coordinate of the blob center
            radius (int, optional): Radius of the Gaussian blob. Defaults to 10.
            amplitude (float, optional): Maximum intensity of the blob. Defaults to 1.0.
            shape (tuple, optional): Shape of the output tensor (channels, height, width). Defaults to (3, 480, 720).
            device (torch.device, optional): Device to create the tensor on. Defaults to None.
        
        Returns:
            torch.Tensor: Tensor of shape (channels, height, width) containing the Gaussian blob
        """
        num_channels, height, width = shape
        
        # Create a new tensor filled with zeros
        blob_tensor = torch.zeros(shape, device=device)
        
        # Create coordinate grids
        y_grid, x_grid = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing='ij'
        )
        
        # Calculate squared distance from (x, y)
        squared_dist = (x_grid - x) ** 2 + (y_grid - y) ** 2
        
        # Create Gaussian blob using the squared distance
        gaussian = amplitude * torch.exp(-squared_dist / (2.0 * radius ** 2))
        
        # Add the Gaussian blob to all channels
        for c in range(num_channels):
            blob_tensor[c] = gaussian
        
        return blob_tensor

class ForcePromptingDataset_WindForce(BaseClass):

    def __init__(
        self,
        csv_path,
        is_validation_dataset=False,
        split_label=None,
        split_column="is_train_val",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.is_validation_dataset = is_validation_dataset

        if is_validation_dataset:
            self.media_type = "image"
            blob_ext =  "*.png"
        else:
            self.media_type = "video"
            blob_ext = "*.mp4"

        file_paths = glob.glob(os.path.join(self.video_root_dir, blob_ext))
        file_names = set([os.path.basename(x) for x in file_paths]) # list of videos or images...
        self.df = pd.read_csv(csv_path)

        # only keep the rows in the csv whose videos we can find
        self.df['checked'] = self.df[self.media_type].map(lambda x, files=file_names: int(x in files))
        self.df = self.df[self.df['checked'] == True]
        self.df = filter_df_by_split_label(
            self.df,
            split_label=split_label,
            split_column=split_column,
            csv_path=csv_path,
        )

        self.min_force = float(self.df["wind_speed"].min())
        self.max_force = float(self.df["wind_speed"].max())

        self.length = self.df.shape[0]

    def get_batch(self, idx):

        item = self.df.iloc[idx]
        caption = item['caption']
        file_name = item[self.media_type]
        force = item['wind_speed']
        angle = item['wind_angle']
        file_path = os.path.join(self.video_root_dir, file_name)

        if self.media_type == "image":
            pixel_values = self.load_pixel_values_image(file_path) # (1, 3, 480, 720) of torch.float32 in [-1, 1]
            file_id = file_name.split(".png")[0]
        elif self.media_type == "video":
            pixel_values = self.load_pixel_values_video(file_path) # (49, 3, 480, 720) of torch.float32 in [-1, 1]
            file_id = file_name.split(".mp4")[0]

        controlnet_signal = load_controlnet_signal_wind_force(
            force, angle, height=self.height, width=self.width, num_frames=self.sample_n_frames, min_force=self.min_force, max_force=self.max_force
        )
        # Wind is a global field, so its "where" mask is the whole frame.
        controlnet_signal_point = torch.ones_like(controlnet_signal[:, :1])
        controlnet_signal = torch.cat([controlnet_signal_point, controlnet_signal], dim=1)

        return pixel_values, caption, controlnet_signal, force, angle, file_id

    def __getitem__(self, idx):
        while True:
            try:
                pixel_values, caption, controlnet_signal, force, angle, file_id = self.get_batch(idx)
                # video, caption, controlnet_video = self.get_batch(idx)
                break
            except Exception as e:
                print(e) # this prints 'text' incessantly
                idx = random.randint(0, self.length - 1)
            
        pixel_values = [
            resize_for_crop(x, self.height, self.width) for x in [pixel_values]
        ][0]
        pixel_values = [
            transforms.functional.center_crop(x, (self.height, self.width)) for x in [pixel_values]
        ][0]
        data = {
            'file_id' : file_id,
            'video': pixel_values, 
            'caption': caption, 
            'controlnet_video': controlnet_signal,
            'force': force,
            'angle': angle,
            'force_type': 'wind_force',
        }
        return data

    def load_pixel_values_video(self, video_path):

        video_reader = VideoReader(video_path)
        # if random.uniform(0, 1) < 0.5:
        indices = np.array([i for i in range(self.sample_n_frames)], dtype=int)
        # else:
        #     indices = np.array([2*i for i in range(self.sample_n_frames)], dtype=int)

        # Get the selected frames
        np_video = video_reader.get_batch(indices).asnumpy() # (49, 480, 720, 3)
        pixel_values = torch.from_numpy(np_video).permute(0, 3, 1, 2).contiguous() # (49, 3, 480, 720) of uint8 in [0, 255]
        # pixel_values = pixel_values / 127.5 - 1 # (49, 3, 480, 720) of torch.float32 in [-1, 1]
        del video_reader

        return pixel_values

    
from utils.forceprompt_data.data_utils import TRANSFORM_MODES

# This is the dataset for the changable wind force during the inference stage
class ForcePromptingDataset_WindForce_ChangeForce_Inference(ForcePromptingDataset_WindForce):
    def __init__(self, csv_path, is_validation_dataset=False, multi_change=False, *args, **kwargs):
        super().__init__(csv_path, is_validation_dataset=is_validation_dataset, *args, **kwargs)

        self.multi_change = multi_change
    
    # Change force dataset - The second half of the control signal video will be changed from the original setting

    def load_controlnet_signal(self, force, angle, idx, num_frames=49, num_channels=3, height=480, width=720):
        controlnet_signal = torch.zeros((num_frames, num_channels, height, width))

        if not self.multi_change:
            half = num_frames // 2

            # first half of the control signal video is the same as the original setting
            norm_force = -1 + 2*(force-self.min_force)/(self.max_force-self.min_force)
            angle_rad = angle * torch.pi / 180.0
            controlnet_signal[:half, 0] = norm_force
            controlnet_signal[:half, 1] = math.cos(angle_rad)
            controlnet_signal[:half, 2] = math.sin(angle_rad)

            # second half of the control signal video is the transformed setting
            mode_id = idx % len(TRANSFORM_MODES)
            rot, fscale = TRANSFORM_MODES[mode_id]

            mod_angle = (angle + rot) % 360
            mod_force = force * fscale

            norm_force = -1 + 2*(mod_force-self.min_force)/(self.max_force-self.min_force)
            angle_rad = mod_angle * torch.pi / 180.0
            controlnet_signal[half:, 0] = norm_force
            controlnet_signal[half:, 1] = math.cos(angle_rad)
            controlnet_signal[half:, 2] = math.sin(angle_rad)
        
        else:
            # change 2 times
            first = num_frames // 3
            second = 2 * first

            norm_force = -1 + 2*(force-self.min_force)/(self.max_force-self.min_force)
            angle_rad = angle * torch.pi / 180.0
            controlnet_signal[:first, 0] = norm_force
            controlnet_signal[:first, 1] = math.cos(angle_rad)
            controlnet_signal[:first, 2] = math.sin(angle_rad)

            # first change
            mode_id = idx % len(TRANSFORM_MODES)
            rot_1, fscale_1 = TRANSFORM_MODES[mode_id]

            mod_angle_1 = (angle + rot_1) % 360
            mod_force_1 = force * fscale_1

            norm_force_1 = -1 + 2*(mod_force_1-self.min_force)/(self.max_force-self.min_force)
            angle_rad_1 = mod_angle_1 * torch.pi / 180.0
            controlnet_signal[first:second, 0] = norm_force_1
            controlnet_signal[first:second, 1] = math.cos(angle_rad_1)
            controlnet_signal[first:second, 2] = math.sin(angle_rad_1)

            # second change
            mode_id = (idx + 1) % len(TRANSFORM_MODES)
            rot_2, fscale_2 = TRANSFORM_MODES[mode_id]

            mod_angle_2 = (angle + rot_2) % 360
            mod_force_2 = force * fscale_2

            norm_force_2 = -1 + 2*(mod_force_2-self.min_force)/(self.max_force-self.min_force)
            angle_rad_2 = mod_angle_2 * torch.pi / 180.0
            controlnet_signal[second:, 0] = norm_force_2
            controlnet_signal[second:, 1] = math.cos(angle_rad_2)
            controlnet_signal[second:, 2] = math.sin(angle_rad_2)
        
        return controlnet_signal
    
    def get_batch(self, idx):
        item = self.df.iloc[idx]
        caption = item['caption']
        file_name = item[self.media_type]
        force = item['wind_speed']
        angle = item['wind_angle']
        file_path = os.path.join(self.video_root_dir, file_name)

        if self.media_type == "image":
            pixel_values = self.load_pixel_values_image(file_path)
            file_id = file_name.split(".png")[0]
        else:
            pixel_values = self.load_pixel_values_video(file_path)
            file_id = file_name.split(".mp4")[0]

        controlnet_signal = self.load_controlnet_signal(force, angle, idx, height=self.height, width=self.width, num_frames=self.sample_n_frames)

        # Wind is a global field, so its "where" mask is the whole frame.
        controlnet_signal_point = torch.ones_like(controlnet_signal[:, :1])
        controlnet_signal = torch.cat([controlnet_signal_point, controlnet_signal], dim=1)

        return pixel_values, caption, controlnet_signal, force, angle, file_id

class ForcePromptingDataset_WindForce_ChangeForce(BaseClass):
    def __init__(
        self,
        csv_path,
        is_validation_dataset=False,
        randomize_change_point=False,
        min_change_ratio=0.3,
        max_change_ratio=0.7,
        split_label=None,
        split_column="is_train_val",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.is_validation_dataset = is_validation_dataset
        self.apply_second_force_at = 80
        self.randomize_change_point = randomize_change_point
        self.min_change_ratio = min_change_ratio
        self.max_change_ratio = max_change_ratio
        # try:
        #     assert self.is_validation_dataset == False
        # except:
        #     raise ValueError("This dataset is only for training")

        if is_validation_dataset:
            self.media_type = "image"
            blob_ext =  "*.png"
        else:
            self.media_type = "video"
            blob_ext = "*.mp4"

        file_paths = glob.glob(os.path.join(self.video_root_dir, blob_ext))
        file_names = set([os.path.basename(x) for x in file_paths]) # list of videos or images...
        self.df = pd.read_csv(csv_path)

        # only keep the rows in the csv whose videos we can find
        self.df['checked'] = self.df[self.media_type].map(lambda x, files=file_names: int(x in files))
        self.df = self.df[self.df['checked'] == True]
        self.df = filter_df_by_split_label(
            self.df,
            split_label=split_label,
            split_column=split_column,
            csv_path=csv_path,
        )

        self.min_force = min(float(self.df["wind_speed_1"].min()), float(self.df["wind_speed_2"].min()))
        self.max_force = max(float(self.df["wind_speed_1"].max()), float(self.df["wind_speed_2"].max()))

        self.length = self.df.shape[0]

    def _clamp_split_index(self, split_index):
        min_split = max(1, int(round(self.sample_n_frames * self.min_change_ratio)))
        max_split = min(
            self.sample_n_frames - 1,
            int(round(self.sample_n_frames * self.max_change_ratio)),
        )
        if min_split > max_split:
            min_split, max_split = 1, self.sample_n_frames - 1
        return int(np.clip(split_index, min_split, max_split))

    def _sample_window_and_split(self, change_at=None):
        if change_at is not None:
            split_index = int(change_at * self.sample_n_frames)
            split_index = self._clamp_split_index(split_index)
            window_start = self.apply_second_force_at - split_index
            return max(0, window_start), split_index

        if self.randomize_change_point:
            min_split = max(1, int(round(self.sample_n_frames * self.min_change_ratio)))
            max_split = min(
                self.sample_n_frames - 1,
                int(round(self.sample_n_frames * self.max_change_ratio)),
            )
            if min_split > max_split:
                min_split, max_split = 1, self.sample_n_frames - 1
            split_index = random.randint(min_split, max_split)
        else:
            split_index = self.sample_n_frames // 2

        split_index = self._clamp_split_index(split_index)
        window_start = self.apply_second_force_at - split_index
        return max(0, window_start), split_index

    def get_batch(self, idx):

        item = self.df.iloc[idx]
        caption = item['caption']
        file_name = item[self.media_type]
        force_1 = item['wind_speed_1']
        force_2 = item['wind_speed_2']
        angle_1 = item['wind_angle_1']
        angle_2 = item['wind_angle_2']
        file_path = os.path.join(self.video_root_dir, file_name)

        change_at = None
        if 'change_at' in self.df.columns:
            change_at = item['change_at']
        window_start, split_index = self._sample_window_and_split(change_at=change_at)

        # we can also consider using some randomness here
        # start_indice = random.randint(0, 160 - self.sample_n_frames)

        if self.media_type == "image":
            pixel_values = self.load_pixel_values_image(file_path) # (1, 3, 480, 720) of torch.float32 in [-1, 1]
            file_id = file_name.split(".png")[0]
        elif self.media_type == "video":
            pixel_values = self.load_pixel_values_video(file_path, start_indice=window_start) # (49, 3, 480, 720) of torch.float32 in [-1, 1]
            file_id = file_name.split(".mp4")[0]

        controlnet_signal = self.load_controlnet_signal(
            force_1, force_2, angle_1, angle_2, height=self.height, width=self.width, num_frames=self.sample_n_frames, split_index=split_index
        )   # we need to pass in start_indice to built the correct controlnet signal -> matches with the selected frames of the video

        # Wind is a global field, so its "where" mask is the whole frame.
        controlnet_signal_point = torch.ones_like(controlnet_signal[:, :1])
        controlnet_signal = torch.cat([controlnet_signal_point, controlnet_signal], dim=1)

        return pixel_values, caption, controlnet_signal, force_1, force_2, angle_1, angle_2, file_id

    def __getitem__(self, idx):
        while True:
            try:
                pixel_values, caption, controlnet_signal, force_1, force_2, angle_1, angle_2, file_id = self.get_batch(idx)
                # video, caption, controlnet_video = self.get_batch(idx)
                break
            except Exception as e:
                print(e) # this prints 'text' incessantly
                idx = random.randint(0, self.length - 1)
            
        pixel_values = [
            resize_for_crop(x, self.height, self.width) for x in [pixel_values]
        ][0]
        pixel_values = [
            transforms.functional.center_crop(x, (self.height, self.width)) for x in [pixel_values]
        ][0]
        data = {
            'file_id' : file_id,
            'video': pixel_values, 
            'caption': caption, 
            'controlnet_video': controlnet_signal,
            'force_1': force_1,
            'force_2': force_2,
            'angle_1': angle_1,
            'angle_2': angle_2,
            'force_type': 'wind_force_change',
        }
        return data

    def load_pixel_values_video(self, video_path, start_indice=0):

        video_reader = VideoReader(video_path)
        # if random.uniform(0, 1) < 0.5:
        # indices = np.array([i for i in range(self.sample_n_frames)], dtype=int)
        # else:
        #     indices = np.array([2*i for i in range(self.sample_n_frames)], dtype=int)

        indices = np.array([i for i in range(start_indice, start_indice + self.sample_n_frames)])


        # Get the selected frames
        np_video = video_reader.get_batch(indices).asnumpy() # (49, 480, 720, 3)
        pixel_values = torch.from_numpy(np_video).permute(0, 3, 1, 2).contiguous() # (49, 3, 480, 720) of uint8 in [0, 255]
        # pixel_values = pixel_values / 127.5 - 1 # (49, 3, 480, 720) of torch.float32 in [-1, 1]
        del video_reader

        return pixel_values

    def load_controlnet_signal(self, force_1, force_2, angle_1, angle_2, num_frames=49, num_channels=3, height=480, width=720, split_index=0):
        
        controlnet_signal = torch.zeros((num_frames, num_channels, height, width))

        half = self._clamp_split_index(split_index)
        controlnet_signal[:half, 0] = -1 + 2*(force_1-self.min_force)/(self.max_force-self.min_force)
        controlnet_signal[:half, 1] = math.cos(angle_1 * torch.pi / 180.0)
        controlnet_signal[:half, 2] = math.sin(angle_1 * torch.pi / 180.0)

        controlnet_signal[half:, 0] = -1 + 2*(force_2-self.min_force)/(self.max_force-self.min_force)
        controlnet_signal[half:, 1] = math.cos(angle_2 * torch.pi / 180.0)
        controlnet_signal[half:, 2] = math.sin(angle_2 * torch.pi / 180.0)

        return controlnet_signal


