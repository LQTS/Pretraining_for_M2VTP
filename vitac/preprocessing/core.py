"""
utils.py

Preprocessing utilities, including dry-run and single-video (single-example) processing. This file effectively defines
the "atomic" logic (take one video --> extract all frames, etc.), while the `process.py` functions invoke each unit
in a multiprocessing pool.
"""
import copy
import glob
import json
import logging
import os
import shutil
import time
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import av
import h5py
import numpy as np
import pandas as pd
from hurry.filesize import alternative, size
from PIL import Image
from rich.progress import track
from tqdm import tqdm

# Grab Logger
overwatch = logging.getLogger(__file__)
logging.getLogger("libav").setLevel(logging.ERROR)


# === General Utilities ===


# Videos are saved as `train_dir/{vid}/{vid}_idx={i}.jpg || if `relpath` then *relative path* `{split}/{vid}/...
def get_path(save_dir: Path, v: str, i: int, relpath: bool = False) -> str:
    return str((save_dir if not relpath else Path(save_dir.name)) / v / f"{v}_idx={i}.jpg")

def get_videos_path_ViTacReal(save_dir: Path, v: str, i: int, relpath: bool = False) -> str:
    
    return str((save_dir if not relpath else Path('/'.join(save_dir.parts[-2:]))) / v / f"{v}_idx={i}.jpg")

def get_tactiles_path_ViTacReal(save_dir: Path, v: str, i: int, relpath: bool = False) -> str:
    return str((save_dir if not relpath else Path('/'.join(save_dir.parts[-2:]))) / "_method_" / v / f"{v}_idx={i}.pkl")


# === Atomic "Processing" Steps ===
def process_tactile_clip(
    path: Path,  # raw data path
    save: Path,  # save data to {} path
    rewrite: bool,
    tac_trig_thre: float, # judge whether tactile is triggered
    preprocess_transform_list: List[Tuple[str, Callable[[List[List[float]]], List[List[float]]]]],  # a list of tactile data process functions
    item         # input parameters
) -> Tuple[Optional[str], Optional[int]]:

    # preprocess all tactile data and record the triggered frames

    vid = item
    # open tactile file
    with open(str(Path(path) / "tactile" / f"{vid}.pkl"), "rb") as f:
        tactile_data_input = pickle.load(f)
    # record which tactile frame is triggered
    tac_triggered_frames_dict = dict(on=[], off=[])
    for idx, tactile_clip in enumerate(tactile_data_input):
        np_tac_clip = np.array(tactile_clip)
        tac_triggered_frames_dict["on" if np.any(np_tac_clip >= tac_trig_thre) else \
                                  "off"].append(idx)
    tac_triggered_frames_return = {
        str(vid): tac_triggered_frames_dict
    }
    # according to preprocess_transform_list, preprocess tactile data and save
    handle_frames = 0
    for type, pre_func in preprocess_transform_list:
        
        save_dir = save / type / str(vid)
        # Short-Circuit | Rewrite
        if os.path.exists(str(save_dir)):
            if rewrite:
                shutil.rmtree(str(save_dir))
            else: # Short-Circuit
                overwatch.info(f"type {type} exists, continue!")
                continue
        os.makedirs(save_dir)
        overwatch.info(f"tactile processing... process type is {type}")
        # preprocess
        tactile_data = pre_func(tactile_data_input)
        for idx, tactile_clip in enumerate(tactile_data):
            save_path = save_dir / f"{vid}_idx={idx}.pkl"
            with open(str(save_path), "wb") as f:
                pickle.dump(tactile_clip, f)
        handle_frames = len(tactile_data)
    # after processing, return the vid and the triggered frames
    return vid, handle_frames, tac_triggered_frames_return


def process_clip(
    name: str,
    path: Path,
    save: Path,
    preprocess_transform: Callable[[List[Image.Image]], List[Image.Image]],
    rewrite: bool,
    item: str,
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Processes a single video clip and extracts/serializes all frames (as jpeg), returning the registry contents."""
    if name in {'ViTacReal', "BottleCap"}:
        vid, lang = item
        # Read imgs
        imgs_dir = Path(path) / "videos" / f"{vid}"
        imgs_path = glob.glob(str(imgs_dir / "*.png"), recursive=False)
        assert len(imgs_path) > 1, "no img file in imgs_path!"
        # sort the imgs_path by the frame index
        imgs_path = sorted(imgs_path, key=lambda f: int(f.split('=')[-1][:-4]))
        imgs, registration = [Image.open(img_path) for img_path in imgs_path], {"language": lang,"n_frames": 0}

        # Book-Keeping
        registration["n_frames"] = len(imgs)

        # Short Circuit | Rewrite
        save_dir = save / vid
        if rewrite: # Rewrite, empty entire directory
            if os.path.exists(save_dir):
                shutil.rmtree(str(save_dir))
        else:  # Short Circuit --> Writes are Expensive!
            if len(glob.glob1(save_dir, "*.jpg")) == len(imgs):
                return vid, registration
        os.makedirs(save_dir, exist_ok=True)

        # Apply `preprocess_transform` --> write individual frames, register, and move on!
        imgs = preprocess_transform(imgs)
        for idx in range(len(imgs)):
            imgs[idx].save(get_path(save, vid, idx))

        # Return title & registration
        return vid, registration
    else:
        raise ValueError(f"Clip Processing for Dataset `{name}` is not implemented!")


# ruff: noqa: C901
def serialize_epoch(
    index_dir: Path,
    registry: Dict[str, Any],
    vid_dir: Tuple[Path, Path],
    batch_formats: Tuple[Tuple[str, Tuple[str, ...]], ...],
    do_initial: bool,
    do_final: bool,
    initial_final_alpha: float,
    n_int: int,
    data_modality: List[str],
    multimodel_input: dict,
    epoch: int,
    is_validation: bool = False
) -> Tuple[int, int, Optional[Set[str]]]:

    index_file = "validation-batches.json" if is_validation else f"train-epoch={epoch}-batches.json"
    index_hdf5 = "validation-batches.hdf5" if is_validation else f"train-epoch={epoch}-batches.hdf5"

    # Short-Circuit
    if all([(index_dir / key / index_file).exists() for key, _ in batch_formats]):
        return -1, -1, None

    # Random seed is inherited from parent process... we want new randomness w/ each process
    np.random.seed((os.getpid() * int(time.time())) % 123456789)

    # Create Tracking Variables
    unique_states, batches = set(), {b: [] for b, _ in batch_formats}

    # unzip multimodelinput
    if "tactile" in data_modality:
        tac_triggered = multimodel_input["tac_triggered"]["val" if is_validation else "train"]

    # Iterate through Registry --> Note we're using `tqdm` instead of `track` here because of `position` feature!
    for vid in tqdm(registry.keys(), desc=f"Epoch {epoch}", total=len(registry), position=epoch):
        if "tactile" not in data_modality:
            # The initial/final states are sampled from the first [0, \alpha) and final 1-\alpha, 1] percent of the video
            n_frames = registry[vid]["n_frames"]
            initial_idx, final_idx = 0, n_frames - 1
            if do_initial:
                initial_idx = np.random.randint(0, np.around(n_frames * initial_final_alpha))

            if do_final:
                final_idx = np.random.randint(np.around(n_frames * (1 - initial_final_alpha)), n_frames)

            # Assertion --> initial_idx < final_idx - len(state_elements)
            assert initial_idx < final_idx - n_int, "Initial & Final are too close... no way to sample!"

            # Assume remaining elements are just random "interior" states --> sort to get ordering!
            sampled_idxs = np.random.choice(np.arange(initial_idx + 1, final_idx), size=n_int, replace=False)
            sampled_idxs = sorted(list(sampled_idxs))

        else:
            ## sample frames from video idx=vid
            n_frames = registry[vid]["n_frames"]

            assert str(vid) in list(tac_triggered.keys()), \
                f"vid \"{vid}\" not found in tac triggered for " + ("val" if is_validation else "train")
            tac_on_list, tac_off_list = tac_triggered[vid]["on"], tac_triggered[vid]["off"]
            assert len(tac_on_list) > 3, \
                f"tactile on triggered clips num for video \"{vid}\" is less than 3!"

            sampled_idxs_tmp = tac_on_list + tac_off_list
            sampled_idxs_tmp = sorted(sampled_idxs_tmp)
            cut = int(0.1 * sampled_idxs_tmp.__len__())
            sampled_idxs_clip = sampled_idxs_tmp[cut:-cut]
            sampled_idxs = tac_on_list if len(tac_on_list) > len(sampled_idxs_clip) else sampled_idxs_clip
            n_frames = sampled_idxs.__len__()


            initial_idx = sampled_idxs[0]
            final_idx = sampled_idxs[-1]
            sampled_idxs = sampled_idxs[1:-1]

        # Compile full-set "batch"
        # 
        # create batch content manager，and generate batch content
        batch_content_manager = batch_content(data_modality, vid_dir, vid, [initial_idx, *sampled_idxs] + [final_idx], n_frames)
        # only one batch content
        key, elements = batch_formats[0]
        retrieved_states_len = len(batch_content_manager.retrieved_states)
        for idx in range(retrieved_states_len):
            batches[key].append(batch_content_manager.get_batch_content(elements, idx=idx))

    # Write JSON Index directly to disk...
    for key in batches:
        with open(index_dir / key / index_file, "w") as f:
            json.dump(batches[key], f)

    # Write HDF5 Index directly to disk...
    for key, elements in batch_formats:
        n_states = elements[-1] + int("initial" in elements) + int("final" in elements)

        # Create HDF5 File
        df = pd.DataFrame(batches[key])
        h5 = h5py.File(index_dir / key / index_hdf5, "w")
        for k in ["vid", "n_frames"]:
            h5.create_dataset(k, data=df[k].values)

        # Handle "state(s)" --> (image path strings) --> add leading dimension (`n_states`)
        if n_states == 1:
            dfs = df["state"].apply(pd.Series)
            h5.create_dataset("states", data=dfs.values)

        else:
            dfs = df["states"].apply(pd.Series)
            h5.create_dataset("states", data=dfs.values)

        if "tactile" in df.columns:
            dfs = df["tactile"].apply(pd.Series)
            h5.create_dataset("tactile", data=dfs.values)

        # Close HDF5 File
        h5.close()

    return epoch, len(batches["only_in"]), unique_states

class batch_content:
    def __init__(self,
                 dataset_modality:List[str],
                 data_dir:Dict[str, Path],
                 frame_id:str,
                 sampled_frames_list:List[str],
                 n_frames:int):
        self.dataset_modality = dataset_modality
        self.vid = frame_id
        self.n_frames = n_frames
        self.video_align_modality = []
        assert "vision" in dataset_modality, "Vision Modality is necessary in Batch_Content"
        self.retrieved_states = [get_videos_path_ViTacReal(data_dir["vision"], frame_id, x, relpath=True) for x in
                                 sampled_frames_list]
        if "tactile" in dataset_modality:
            self.video_align_modality.append(
                ("tactile", [get_tactiles_path_ViTacReal(data_dir["tactile"], frame_id, x, relpath=True) for x in
                                      sampled_frames_list])
            )

    def get_batch_content(self, batch_format:list, idx:int=-1):
        result = {"vid": self.vid, "states": [], "n_frames": self.n_frames}
        sum_video_clip_num = batch_format[-1] + int("initial" in batch_format) + int("final" in batch_format)
        if sum_video_clip_num == 1:
            assert idx >= 0, f'something go wrong with get_batch_content'
            # add one video frame
            result["state"] = self.retrieved_states[idx]
            for dm, clips in self.video_align_modality:
                result[dm] = clips[idx]
        elif sum_video_clip_num == 2 and "initial" in batch_format:
            assert idx >= 0, f'something go wrong with get_batch_content'
            # add two video frame
            result["states"] = [self.retrieved_states[0], self.retrieved_states[idx]]
            for dm, clips in self.video_align_modality:
                result[dm] = [clips[0], clips[idx]]
        elif sum_video_clip_num == 5:
            result["states"] = self.retrieved_states
            for dm, clips in self.video_align_modality:
                result[dm] = clips
        else:
            raise ValueError(f"Batch_format '{batch_format}' is not supported in Batch Content Manager!")

        return result