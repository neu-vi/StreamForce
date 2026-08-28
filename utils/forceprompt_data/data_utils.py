import torch
import cv2
import math
import numpy as np


def filter_df_by_split_label(df, split_label=None, split_column="is_train_val", csv_path=None):
    if split_label is None:
        return df

    if split_column not in df.columns:
        where = f" in {csv_path}" if csv_path else ""
        raise ValueError(
            f"Requested split='{split_label}' but column '{split_column}' was not found{where}."
        )

    normalized = df[split_column].astype(str).str.strip().str.lower()
    target = str(split_label).strip().lower()
    filtered = df[normalized == target]

    if filtered.empty:
        where = f" in {csv_path}" if csv_path else ""
        raise ValueError(
            f"Split filter produced 0 rows for split='{split_label}' using column '{split_column}'{where}."
        )

    return filtered


def collate_fn_ForcePromptingDataset_PointForce(examples): 

    videos = [example["video"] for example in examples]
    prompts = [example["caption"] for example in examples]
    controlnet_videos = [example["controlnet_video"] for example in examples]
    file_ids = [example["file_id"] for example in examples]

    forces = [example["force"] for example in examples]
    angles = [example["angle"] for example in examples]
    x_poss = [example["x_pos"] for example in examples]
    y_poss = [example["y_pos"] for example in examples]
    force_types = [example["force_type"] for example in examples]
    
    videos = torch.stack(videos)
    videos = videos.to(memory_format=torch.contiguous_format).float()

    # nate added this
    first_frames = videos[:, 0]
    first_frames = first_frames.to(memory_format=torch.contiguous_format).float()

    controlnet_videos = torch.stack(controlnet_videos)
    controlnet_videos = controlnet_videos.to(memory_format=torch.contiguous_format).float()

    return {
        "file_ids" : file_ids,
        "first_frames" : first_frames,
        "videos": videos,
        "prompts": prompts,
        "controlnet_videos": controlnet_videos,
        "force": forces,
        "angle": angles,
        "x_pos" : x_poss,
        "y_pos" : y_poss,
        "force_type": force_types,
    }


def collate_fn_ForcePromptingDataset_PointForce_ChangeForce(examples):

    videos = [example["video"] for example in examples]
    prompts = [example["caption"] for example in examples]
    controlnet_videos = [example["controlnet_video"] for example in examples]
    file_ids = [example["file_id"] for example in examples]
    change_ats = [example["change_at"] for example in examples if "change_at" in example]

    force_1s = [example["force_1"] for example in examples]
    force_2s = [example["force_2"] for example in examples]
    angle_1s = [example["angle_1"] for example in examples]
    angle_2s = [example["angle_2"] for example in examples]
    x_pos_1s = [example["x_pos_1"] for example in examples]
    y_pos_1s = [example["y_pos_1"] for example in examples]
    x_pos_2s = [example["x_pos_2"] for example in examples]
    y_pos_2s = [example["y_pos_2"] for example in examples]
    force_types = [example["force_type"] for example in examples]
    videos = torch.stack(videos)
    videos = videos.to(memory_format=torch.contiguous_format).float()

    # nate added this
    first_frames = videos[:, 0]
    first_frames = first_frames.to(memory_format=torch.contiguous_format).float()

    controlnet_videos = torch.stack(controlnet_videos)
    controlnet_videos = controlnet_videos.to(memory_format=torch.contiguous_format).float()

    return {
        "file_ids" : file_ids,
        "first_frames" : first_frames,
        "videos": videos,
        "prompts": prompts,
        "controlnet_videos": controlnet_videos,
        "force_1": force_1s,
        "force_2": force_2s,
        "angle_1": angle_1s,
        "angle_2": angle_2s,
        "x_pos_1": x_pos_1s,
        "y_pos_1": y_pos_1s,
        "x_pos_2": x_pos_2s,
        "y_pos_2": y_pos_2s,
        "force_type": force_types,
        "change_at": change_ats,
    }

def collate_fn_ForcePromptingDataset_WindForce(examples): 

    videos = [example["video"] for example in examples]
    prompts = [example["caption"] for example in examples]
    controlnet_videos = [example["controlnet_video"] for example in examples]
    file_ids = [example["file_id"] for example in examples]

    forces = [example["force"] for example in examples]
    angles = [example["angle"] for example in examples]
    x_poss = [example["x_pos"] for example in examples if "x_pos" in example]
    y_poss = [example["y_pos"] for example in examples if "y_pos" in example]
    force_types = [example["force_type"] for example in examples]

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
    }

# This is used for the training stage of the wind force change dataset
def collate_fn_ForcePromptingDataset_WindForce_ChangeForce(examples): 

    videos = [example["video"] for example in examples]
    prompts = [example["caption"] for example in examples]
    controlnet_videos = [example["controlnet_video"] for example in examples]
    file_ids = [example["file_id"] for example in examples]

    force_1s = [example["force_1"] for example in examples]
    force_2s = [example["force_2"] for example in examples]
    angle_1s = [example["angle_1"] for example in examples]
    angle_2s = [example["angle_2"] for example in examples]
    force_types = [example["force_type"] for example in examples]
    videos = torch.stack(videos)
    videos = videos.to(memory_format=torch.contiguous_format).float()

    # nate added this
    first_frames = videos[:, 0]
    first_frames = first_frames.to(memory_format=torch.contiguous_format).float()

    controlnet_videos = torch.stack(controlnet_videos)
    controlnet_videos = controlnet_videos.to(memory_format=torch.contiguous_format).float()

    return {
        "file_ids" : file_ids,
        "first_frames" : first_frames,
        "videos": videos,
        "prompts": prompts,
        "controlnet_videos": controlnet_videos,
        "force_1": force_1s,
        "force_2": force_2s,
        "angle_1": angle_1s,
        "angle_2": angle_2s,
        "force_type": force_types,
    }


TRANSFORM_MODES = [
    (0,   0.5),
    (90,  1.0),
    (90,  0.5),
    (180, 1.0),
    (180, 0.5),
    (270, 1.0),
    (270, 0.5),
]

def add_aesthetic_point_force_prompt_to_video(video, force, angle, x_pos, y_pos, circle_radius=20, num_frames_with_signal=1):
    """
    Annotate the first frame of a video with a white circle and directional yellow arrow.
    
    Parameters:
    -----------
    video : numpy.ndarray
        Video array with shape (num_frames, height, width, channels), values in [0,1]
    force : float
        Value in [0,1] that determines the length of the arrow
    angle : float
        Value in [0,360] that determines the direction of the arrow
    x_pos : float
        Horizontal position in [0,1] (will be scaled to pixel coordinates)
    y_pos : float
        Vertical position in [0,1] (will be scaled to pixel coordinates)
    
    Returns:
    --------
    numpy.ndarray
        Modified video with annotations on the first frame
    """
    # Create a copy of the video to avoid modifying the original
    result_video = (video.copy() * 255).astype(np.uint8)
    
    # Get the dimensions of the video
    num_frames, height, width, channels = video.shape
    
    # Convert the position from [0,1] range to pixel coordinates
    center_x = int(x_pos * width)
    center_y = int(y_pos * height)
    
    # Convert angle from degrees to radians
    angle_rad = math.radians(angle)
    
    # Calculate the arrow endpoint
    arrow_length = 10 + 90 * force # min force in dataset, corresponidng to 0, should have some positive length...
    end_x = int(center_x + arrow_length * math.cos(angle_rad))
    end_y = int(center_y - arrow_length * math.sin(angle_rad))

    for i in range(num_frames_with_signal):
        # Convert the first frame to uint8 format (0-255) for OpenCV
        this_frame = result_video[i]
        
        # Draw a white circle with radius 10 pixels and thickness 2 pixels
        cv2.circle(this_frame, (center_x, center_y), circle_radius, (255, 255, 255), 2)
        
        # Draw a yellow arrow
        cv2.arrowedLine(this_frame, (center_x, center_y), (end_x, end_y), (0, 255, 255), 2, tipLength=0.3)
        
        # Convert the frame back to [0,1] range
    
        result_video[i] = this_frame # / 255.0
    
    return result_video


def add_aesthetic_point_force_change_prompt_to_video(
    video,
    force_1,
    angle_1,
    x_pos_1,
    y_pos_1,
    force_2,
    angle_2,
    x_pos_2,
    y_pos_2,
    idx,    # frame index to apply the second force
    circle_radius=20,
    num_frames_with_signal=1,
):
    # Create a copy of the video to avoid modifying the original
    result_video = (video.copy() * 255).astype(np.uint8)
    
    # Get the dimensions of the video
    num_frames, height, width, channels = video.shape
    
    # Convert the position from [0,1] range to pixel coordinates
    center_x_1 = int(x_pos_1 * width)
    center_y_1 = int(y_pos_1 * height)
    center_x_2 = int(x_pos_2 * width)
    center_y_2 = int(y_pos_2 * height)

    # Convert angle from degrees to radians
    angle_rad_1 = math.radians(angle_1)
    angle_rad_2 = math.radians(angle_2)
    
    # Calculate the arrow endpoint
    arrow_length_1 = 10 + 90 * force_1 # min force in dataset, corresponidng to 0, should have some positive length...
    arrow_length_2 = 10 + 90 * force_2 # min force in dataset, corresponidng to 0, should have some positive length...
    end_x_1 = int(center_x_1 + arrow_length_1 * math.cos(angle_rad_1))
    end_y_1 = int(center_y_1 - arrow_length_1 * math.sin(angle_rad_1))
    end_x_2 = int(center_x_2 + arrow_length_2 * math.cos(angle_rad_2))
    end_y_2 = int(center_y_2 - arrow_length_2 * math.sin(angle_rad_2))

    for i in range(num_frames_with_signal):
        frame = result_video[i]
        cv2.circle(frame, (center_x_1, center_y_1), circle_radius, (255, 255, 255), 2)
        cv2.arrowedLine(frame, (center_x_1, center_y_1), (end_x_1, end_y_1), (0, 255, 255), 2, tipLength=0.3)
        if i >= idx:
            cv2.circle(frame, (center_x_2, center_y_2), circle_radius, (255, 255, 255), 2)
            cv2.arrowedLine(frame, (center_x_2, center_y_2), (end_x_2, end_y_2), (0, 255, 255), 2, tipLength=0.3)
        result_video[i] = frame

    return result_video

# Update max_arrow_length properly to 90 to account for forward distance
def add_aesthetic_wind_force_prompt_to_video(
    video,
    force,
    angle,
    num_frames_with_signal=1,
    base_periods=1,
    periods_per_0_1_force=1,
    wave_amplitude=2,
    extra_straight_length=20,
    arrowhead_length=7,
    forward_distance=6
):
    result_video = (video.copy() * 255).astype(np.uint8)
    num_frames, height, width, channels = video.shape

    arrowhead_base = int(arrowhead_length * (2 / math.sqrt(3)))

    min_arrow_length = 30
    max_arrow_length = 90  # final correct value

    arrow_length = min_arrow_length + force * (max_arrow_length - min_arrow_length)
    periods = base_periods + int(force * 10) * periods_per_0_1_force

    angle_rad = math.radians(angle)
    dir_x = math.cos(angle_rad)
    dir_y = -math.sin(angle_rad)
    perp_x = -dir_y
    perp_y = dir_x

    base_x = width - 100
    base_y = 100

    for i in range(min(num_frames_with_signal, num_frames)):
        frame = result_video[i]

        for j in range(3):
            offset = (j - 1) * 20
            start_x = base_x + offset * perp_x
            start_y = base_y + offset * perp_y

            points = []
            num_points = 100
            squiggly_part_length = arrow_length - extra_straight_length
            squiggly_end_t = squiggly_part_length / arrow_length

            for k in range(num_points):
                t = k / (num_points - 1)
                if t < squiggly_end_t:
                    main_x = start_x + dir_x * t * arrow_length
                    main_y = start_y + dir_y * t * arrow_length
                    squiggle = math.sin(t * periods * 2 * math.pi) * wave_amplitude
                    squiggle_x = main_x + perp_x * squiggle
                    squiggle_y = main_y + perp_y * squiggle
                else:
                    straight_progress = (t - squiggly_end_t) / (1 - squiggly_end_t)
                    main_x = start_x + dir_x * (squiggly_part_length + straight_progress * extra_straight_length)
                    main_y = start_y + dir_y * (squiggly_part_length + straight_progress * extra_straight_length)
                    squiggle_x = main_x
                    squiggle_y = main_y

                points.append((int(squiggle_x), int(squiggle_y)))

            for p in range(len(points) - 1):
                cv2.line(frame, points[p], points[p + 1], (0, 255, 255), 2)

            tip = points[-1]
            tip_forward_x = tip[0] + forward_distance * dir_x
            tip_forward_y = tip[1] + forward_distance * dir_y
            tip_point = (int(tip_forward_x), int(tip_forward_y))

            base_center_x = tip[0] - arrowhead_length * dir_x
            base_center_y = tip[1] - arrowhead_length * dir_y

            left_base_x = int(base_center_x + (arrowhead_base / 2) * -dir_y)
            left_base_y = int(base_center_y + (arrowhead_base / 2) * dir_x)

            right_base_x = int(base_center_x - (arrowhead_base / 2) * -dir_y)
            right_base_y = int(base_center_y - (arrowhead_base / 2) * dir_x)

            cv2.line(frame, (left_base_x, left_base_y), tip_point, (0, 255, 255), 2)
            cv2.line(frame, (right_base_x, right_base_y), tip_point, (0, 255, 255), 2)

        result_video[i] = frame # / 255.0

    return result_video


def add_aesthetic_wind_force_change_prompt_to_video_transform_set(
    video,
    force,
    angle,
    idx,
    num_frames_with_signal=1,
    base_periods=1,
    periods_per_0_1_force=1,
    wave_amplitude=2,
    extra_straight_length=20,
    arrowhead_length=7,
    forward_distance=6,
    multi_change=False
):
    mode_id = idx % len(TRANSFORM_MODES)
    rot, fscale = TRANSFORM_MODES[mode_id]

    if multi_change:
        rot1, fscale1 = rot, fscale
        rot2, fscale2 = TRANSFORM_MODES[(idx + 1) % len(TRANSFORM_MODES)]

    result_video = (video.copy() * 255).astype(np.uint8)
    num_frames, height, width, channels = video.shape

    arrowhead_base = int(arrowhead_length * (2 / math.sqrt(3)))

    min_arrow_length = 30
    max_arrow_length = 90  # final correct value

    periods = base_periods + int(force * 10) * periods_per_0_1_force

    base_x = width - 100
    base_y = 100

    num_annotation_frames = min(num_frames_with_signal, num_frames)
    half = num_annotation_frames // 2
    if multi_change:
        first = num_frames // 3
        second = 2 * first

    for i in range(min(num_frames_with_signal, num_frames)):
        frame = result_video[i]

        if multi_change:
            if i < first:
                this_angle = angle
                this_fscale = 1
            elif i < second:
                this_angle = (angle + rot1) % 360
                this_fscale = fscale1
            else:
                this_angle = (angle + rot2) % 360
                this_fscale = fscale2
        else:
            this_angle = angle if i < half else (angle + rot) % 360
            this_fscale = 1 if i < half else fscale

        arrow_length = min_arrow_length + force * (max_arrow_length - min_arrow_length) * this_fscale
        angle_rad = math.radians(this_angle)
        dir_x = math.cos(angle_rad)
        dir_y = -math.sin(angle_rad)
        perp_x = -dir_y
        perp_y =  dir_x

        for j in range(3):
            offset = (j - 1) * 20
            start_x = base_x + offset * perp_x
            start_y = base_y + offset * perp_y

            points = []
            num_points = 100
            squiggly_part_length = arrow_length - extra_straight_length
            squiggly_end_t = squiggly_part_length / arrow_length

            for k in range(num_points):
                t = k / (num_points - 1)
                if t < squiggly_end_t:
                    main_x = start_x + dir_x * t * arrow_length
                    main_y = start_y + dir_y * t * arrow_length
                    squiggle = math.sin(t * periods * 2 * math.pi) * wave_amplitude
                    squiggle_x = main_x + perp_x * squiggle
                    squiggle_y = main_y + perp_y * squiggle
                else:
                    straight_progress = (t - squiggly_end_t) / (1 - squiggly_end_t)
                    main_x = start_x + dir_x * (squiggly_part_length + straight_progress * extra_straight_length)
                    main_y = start_y + dir_y * (squiggly_part_length + straight_progress * extra_straight_length)
                    squiggle_x = main_x
                    squiggle_y = main_y

                points.append((int(squiggle_x), int(squiggle_y)))

            for p in range(len(points) - 1):
                cv2.line(frame, points[p], points[p + 1], (0, 255, 255), 2)

            tip = points[-1]
            tip_forward_x = tip[0] + forward_distance * dir_x
            tip_forward_y = tip[1] + forward_distance * dir_y
            tip_point = (int(tip_forward_x), int(tip_forward_y))

            base_center_x = tip[0] - arrowhead_length * dir_x
            base_center_y = tip[1] - arrowhead_length * dir_y

            left_base_x = int(base_center_x + (arrowhead_base / 2) * -dir_y)
            left_base_y = int(base_center_y + (arrowhead_base / 2) * dir_x)

            right_base_x = int(base_center_x - (arrowhead_base / 2) * -dir_y)
            right_base_y = int(base_center_y - (arrowhead_base / 2) * dir_x)

            cv2.line(frame, (left_base_x, left_base_y), tip_point, (0, 255, 255), 2)
            cv2.line(frame, (right_base_x, right_base_y), tip_point, (0, 255, 255), 2)

        result_video[i] = frame

    return result_video


def add_aesthetic_wind_force_change_prompt_to_video(
    video,
    force_1,
    angle_1,
    force_2,
    angle_2,
    idx,
    num_frames_with_signal=1,
    base_periods=1,
    periods_per_0_1_force=1,
    wave_amplitude=2,
    extra_straight_length=20,
    arrowhead_length=7,
    forward_distance=6
):
    result_video = (video.copy() * 255).astype(np.uint8)
    num_frames, height, width, channels = video.shape

    arrowhead_base = int(arrowhead_length * (2 / math.sqrt(3)))

    min_arrow_length = 30
    max_arrow_length = 90  # final correct value

    arrow_length_1 = min_arrow_length + force_1 * (max_arrow_length - min_arrow_length)
    arrow_length_2 = min_arrow_length + force_2 * (max_arrow_length - min_arrow_length)

    angle_rad_1 = math.radians(angle_1)
    angle_rad_2 = math.radians(angle_2)

    dir_x_1 = math.cos(angle_rad_1)
    dir_y_1 = -math.sin(angle_rad_1)
    dir_x_2 = math.cos(angle_rad_2)
    dir_y_2 = -math.sin(angle_rad_2)
    perp_x_1 = -dir_y_1
    perp_y_1 = dir_x_1
    perp_x_2 = -dir_y_2
    perp_y_2 = dir_x_2

    periods_1 = base_periods + int(force_1 * 10) * periods_per_0_1_force
    periods_2 = base_periods + int(force_2 * 10) * periods_per_0_1_force

    base_x = width - 100
    base_y = 100

    for i in range(min(num_frames_with_signal, num_frames)):
        frame = result_video[i]

        if i < idx:
            arrow_length = arrow_length_1
            angle_rad = angle_rad_1
            dir_x = dir_x_1
            dir_y = dir_y_1
            perp_x = perp_x_1
            perp_y = perp_y_1
            periods = periods_1
        else:
            arrow_length = arrow_length_2
            angle_rad = angle_rad_2
            dir_x = dir_x_2
            dir_y = dir_y_2
            perp_x = perp_x_2
            perp_y = perp_y_2
            periods = periods_2

        for j in range(3):
            offset = (j - 1) * 20
            start_x = base_x + offset * perp_x
            start_y = base_y + offset * perp_y

            points = []
            num_points = 100
            squiggly_part_length = arrow_length - extra_straight_length
            squiggly_end_t = squiggly_part_length / arrow_length

            for k in range(num_points):
                t = k / (num_points - 1)
                if t < squiggly_end_t:
                    main_x = start_x + dir_x * t * arrow_length
                    main_y = start_y + dir_y * t * arrow_length
                    squiggle = math.sin(t * periods * 2 * math.pi) * wave_amplitude
                    squiggle_x = main_x + perp_x * squiggle
                    squiggle_y = main_y + perp_y * squiggle
                else:
                    straight_progress = (t - squiggly_end_t) / (1 - squiggly_end_t)
                    main_x = start_x + dir_x * (squiggly_part_length + straight_progress * extra_straight_length)
                    main_y = start_y + dir_y * (squiggly_part_length + straight_progress * extra_straight_length)
                    squiggle_x = main_x
                    squiggle_y = main_y

                points.append((int(squiggle_x), int(squiggle_y)))

            for p in range(len(points) - 1):
                cv2.line(frame, points[p], points[p + 1], (0, 255, 255), 2)

            tip = points[-1]
            tip_forward_x = tip[0] + forward_distance * dir_x
            tip_forward_y = tip[1] + forward_distance * dir_y
            tip_point = (int(tip_forward_x), int(tip_forward_y))

            base_center_x = tip[0] - arrowhead_length * dir_x
            base_center_y = tip[1] - arrowhead_length * dir_y

            left_base_x = int(base_center_x + (arrowhead_base / 2) * -dir_y)
            left_base_y = int(base_center_y + (arrowhead_base / 2) * dir_x)

            right_base_x = int(base_center_x - (arrowhead_base / 2) * -dir_y)
            right_base_y = int(base_center_y - (arrowhead_base / 2) * dir_x)

            cv2.line(frame, (left_base_x, left_base_y), tip_point, (0, 255, 255), 2)
            cv2.line(frame, (right_base_x, right_base_y), tip_point, (0, 255, 255), 2)

        result_video[i] = frame

    return result_video
