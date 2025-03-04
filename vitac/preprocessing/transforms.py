"""
transforms.py

Default video/image transforms for Voltron preprocessing and training. Provides utilities for defining different scale
and crop transformations on a dataset-specific basis.

There are two key desiderata we ensure with the transforms:
    - Aspect Ratio --> We *never* naively reshape images in a way that distorts the aspect ratio; we crop instead!
    - Minimum Size --> We *never* upsample images; processing strictly reduces dimensionality!
"""
from functools import partial
from typing import Any, Callable, List, Tuple

import torch
import numpy as np
from PIL import Image, ImageOps
from torchvision.transforms import Compose, ConvertImageDtype, Lambda, Normalize, Resize


# Simple Identity Function --> needs to be top-level/pickleable for mp/distributed.spawn()
def identity(x: torch.Tensor) -> torch.Tensor:
    return x.float()
def scaled_crop(
    target_resolution: int ,
    frames: List[Image.Image],
    type: str = "right"
) -> List[Image.Image]:
    # Assert width >= height and height >= target_resolution
    orig_w, orig_h = frames[0].size
    assert orig_w >= orig_h >= target_resolution

    # Compute scale factor --> just a function of height and target_resolution\
    scale_factor = target_resolution / orig_h
    for idx in range(len(frames)):
        frames[idx] = ImageOps.scale(frames[idx], factor=scale_factor)
        if type == "center":
            left = (frames[idx].size[0] - target_resolution) // 2
        elif type == "right":
            left = frames[idx].size[0] - target_resolution
        else:
            raise TypeError("No such crop Type!")
        frames[idx] = frames[idx].crop((left, 0, left + target_resolution, target_resolution))

    # Return "scaled and squared" images
    return frames

class PreprocessTactile:
    @staticmethod
    def origin(
        tactile_data: List[List[List[float]]]
    ) ->  List[List[List[float]]]:
        tactile_data = np.array(tactile_data)
        return tactile_data.tolist()

    @staticmethod
    def event(
        tactile_data: List[List[List[float]]]
    ) -> List[List[List[float]]]:
        tactile_data = np.array(tactile_data)
        for i in range(len(tactile_data) - 1)[::-1]:
            tactile_data[i+1] = tactile_data[i+1] - tactile_data[i]
        return tactile_data.tolist()

    @staticmethod
    def binary(
        threshold: float,
        tactile_data: List[List[List[float]]]
    ) -> List[List[List[int]]]:
        tactile_data = np.array(tactile_data)
        tactile_1_idx = tactile_data > threshold
        tactile_data[tactile_1_idx] = 1
        tactile_data[~tactile_1_idx] = 0

        return tactile_data.tolist()

    @staticmethod
    def normalize_global(
        tactile_data_input: List[List[List[float]]]
    ) -> List[List[List[int]]]:
        tactile_data = np.array(tactile_data_input)
        # acquire the max value of the whole tactile data
        tactile_serials_max = np.max(tactile_data)
        # normalize the tactile data
        tactile_data = tactile_data / tactile_serials_max

        return tactile_data.tolist()

    @staticmethod
    def normalize_self(
        ignore_threshold: float,        
        tactile_data_input: List[List[List[float]]]
    ) -> List[List[List[int]]]:
        tactile_data = np.array(tactile_data_input)
        # accquire the max and min value of each sensor
        tactile_serials_max_array = np.max(tactile_data, axis=0)
        tactile_serials_min_array = np.min(tactile_data, axis=0)
        tactile_serials_delta_array = tactile_serials_max_array - tactile_serials_min_array
        
        tactile_0_idx = tactile_serials_delta_array < ignore_threshold
        tactile_serials_max_array[tactile_0_idx] = 1.0
        
        tactile_data = tactile_data - tactile_serials_min_array
        
        tactile_data = tactile_data / tactile_serials_max_array
        
        tactile_0_idx_full = np.array([tactile_0_idx.tolist()] * tactile_data.shape[0])
        tactile_data[tactile_0_idx_full] = 0.0

        return tactile_data.tolist()

def get_preprocess_transform(
    dataset_name: str, preprocess_resolution: int
) -> Callable[[List[Image.Image]], List[Image.Image]]:
    """Returns a transform that extracts square crops of `preprocess_resolution` from videos (as [T x H x W x C])."""
    if dataset_name in {'ViTacReal', 'BottleCap'}:
        return partial(scaled_crop, preprocess_resolution, type="center")
    else:
        raise ValueError(f"Preprocessing transform for dataset `{dataset_name}` is not defined!")

def get_online_transform(
    dataset_name: str, model_arch: str, online_resolution: int, normalization: Tuple[Any, Any]
) -> Compose:
    """Returns an "online" torchvision Transform to be applied during training (batching/inference)."""
    if dataset_name in {"ViTacReal", "BottleCap"}:
        return Compose(
            [
                Resize((online_resolution, online_resolution), antialias=True),
                ConvertImageDtype(torch.float),
                Normalize(mean=normalization[0], std=normalization[1]),
            ]
        )
    else:
        raise ValueError(f"Online Transforms for Dataset `{dataset_name}` not implemented!")

class tactile_transform:
    @staticmethod
    def no_operation(raw_tac: Any) -> Any:
        return raw_tac
    @staticmethod
    def tactile_flatten_torch(raw_tac: List[List[float]]) -> torch.Tensor:
        """
        E1, D1, C1, B1, A1,
        E2, D2, C2, B2, A2,
        ...
        E4, D4, C4, B4, A4      --->>    E1, D1, C1, B1, A1, E2, ...A1, E3, ... E4, D4, C4, B4, A4
        """
        torch_tac = torch.Tensor(raw_tac)
        flatten_tac = torch_tac.flatten()
        return flatten_tac
def get_online_tactile_transform(
    dataset_name: str, model_arch: str
) -> Callable[[List[List[float]]], torch.Tensor]: # return type is a function
    """Returns an "online" tactile Transform to be applied during training (batching/inference)."""
    if dataset_name == "ViTacReal":
        if model_arch == "vtt-cond" or model_arch == "vtt-condv2":
            return tactile_transform.tactile_flatten_torch
        elif model_arch == "v-cond":
            return tactile_transform.no_operation
        else:
            raise ValueError(f"Online Tactile Transforms for Dataset `{dataset_name}` with arch {model_arch} is not implemented!")
    elif dataset_name == "BottleCap":
        if model_arch in {"v-repic", "vtt-reall", "t-retac"}:
            return tactile_transform.tactile_flatten_torch
        else:
            raise ValueError(
                f"Online Tactile Transforms for Dataset `{dataset_name}` with arch {model_arch} is not implemented!")
    else:
        raise ValueError(f"Online Tactile Transforms for Dataset `{dataset_name}` not implemented!")