
import json
import logging
import multiprocessing as mp
import os
import shutil
import random
from functools import partial
from pathlib import Path
from typing import Tuple, List, Dict

import torch
from rich.progress import track
from transformers import AutoTokenizer

from vitac.preprocessing.core import process_clip, process_tactile_clip, serialize_epoch
from vitac.preprocessing.transforms import get_preprocess_transform
from vitac.preprocessing.transforms import PreprocessTactile

# Grab Logger
overwatch = logging.getLogger(__file__)

def set_registry_path(artifact_path, name):
    if name in {"ViTacReal", "BottleCap"}:
        t_dir, v_dir = Path(artifact_path) / name / "train" / "videos", Path(artifact_path) / name / "val" / "videos"
    else:
        raise AttributeError(f'Dataset {name} is unknown!')
    t_registry, v_registry = t_dir.parent / "registry.json", v_dir.parent / "registry.json"
    return t_dir, t_registry, v_dir, v_registry

def extract_frames(
    name: str,
    path: str,
    artifact_path: str,
    preprocess_resolution: int,
    train_val_ratio: float,
    rewrite: bool = False,
) -> Tuple[Path, Path, Path, Path]:
    """Phase I: Extract and serialize *all frames* from video clips; uses multiprocessing to parallelize."""
    overwatch.info(f"Phase 1 Preprocessing :: Extracting Frames for Dataset `{name}`")

    # Overview of Return Values:
    #   `t_registry` and `v_registry` =>> store mappings of "video id" -> {metadata}
    #   `t_dir` and `v_dir` =>> store "processed data" (extracted frames)
    t_dir, t_registry, v_dir, v_registry = set_registry_path(artifact_path, name)

    # Short-Circuit
    if t_registry.exists() and v_registry.exists():
        return t_registry, v_registry, t_dir, v_dir

    # Setup / Book-Keeping
    os.makedirs(t_dir, exist_ok=True)
    os.makedirs(v_dir, exist_ok=True)

    # Retrieve "pre-serialization" frame transform --> we scale down video frames (*while preserving aspect ratios*)
    #   and center crop each frame to `(preprocess_resolution, preprocess_resolution)`; saves on disk space (by a lot!)
    preprocess_transform = get_preprocess_transform(name, preprocess_resolution=preprocess_resolution)

    # Switch on dataset (`name`)
    if name in {"ViTacReal", "BottleCap"}:
        # split the dataset into train and validation
        with open(Path(path) / "labels/labels.json", "r") as f:
            annotations = json.load(f)
        random.shuffle(annotations)
        all_num = len(annotations)
        n_val_videos = int(all_num / train_val_ratio)
        train_annotations = annotations[:-n_val_videos]
        val_annotations = annotations[-n_val_videos:]
        train_ids, train_lang = [str(int(x["id"])) for x in train_annotations], [x["label"] for x in train_annotations]
        val_ids, val_lang = [str(int(x["id"])) for x in val_annotations], [x["label"] for x in val_annotations]
    else:
        raise ValueError(f"Language/Metadata Extraction Pipeline for Dataset `{name}` not implemented!")

    # Otherwise =>> Iterate through all videos, dump all frames subject to the following structure:
    #   |-> .../processed/something-something-v2/
    #       |-> <split>/
    #           |-> <video-id>/frames<0..k>.jpg
    #
    # We'll build a single metadata file with a mapping <video-id> : ("language", n_frames)
    #   > To speed up serialization, we'll use a multiprocessing.Pool and max out CPU workers
    with mp.Pool(mp.cpu_count()) as pool:
        for k, save, vids, langs in [("train", t_dir, train_ids, train_lang), ("val", v_dir, val_ids, train_lang)]:
            overwatch.info(f"\tWriting `{k}` videos to disk...")

            # Spawn!
            process_fn, registration = partial(process_clip, name, Path(path), save, preprocess_transform, rewrite), {}
            for key, value in track(
                pool.imap_unordered(process_fn, zip(vids, langs)),
                total=len(vids),
                transient=True,
            ):
                if key is not None:
                    registration[key] = value

            # Write Registration to Disk
            with open(t_registry if k == "train" else v_registry, "w") as f:
                json.dump(registration, f)

    # Return Paths to Registry & Extract Directories...
    return t_registry, v_registry, t_dir, v_dir


def preprocess_language(
    name: str,
    train_registry: Path,
    val_registry: Path,
    artifact_path: str,
    max_lang_len: int,
    language_model: str,
    hf_cache: str,
    rewrite: bool
) -> Path:
    """Phase II: Iterate through Language Captions/Narrations and Normalize/Tokenize (truncate/pad to max length)."""
    overwatch.info(f"Phase 2 Preprocessing :: Normalizing & Tokenizing Language for Dataset `{name}`")
    t_index, v_index = train_registry.parent / "index.pt", val_registry.parent / "index.pt"
    t_json, v_json = train_registry.parent / "index.json", val_registry.parent / "index.json"
    index_dir = Path(artifact_path) / name / "index"
    os.makedirs(index_dir, exist_ok=True)

    # Short-Circuit | Rewrite
    if rewrite: # Rewrite
        if t_index.exists():
            os.remove(str(t_index))
            os.remove(str(v_index))
        if t_json.exists():
            os.remove(str(t_json))
            os.remove(str(v_json))
        if (index_dir / "train-language-index.json").exists():
            os.remove(str(index_dir / "train-language-index.json"))
            os.remove(str(index_dir / "val-language-index.json"))
            os.remove(str(index_dir / "train-language-index.pt"))
            os.remove(str(index_dir / "val-language-index.pt"))
    else: #Short-Circuit
        if (index_dir / "train-language-index.json").exists() and (index_dir / "val-language-index.json").exists():
            return index_dir

    # Grab Language --> retain metadata for building index structures!
    with open(train_registry, "r") as f:
        train_metadata = json.load(f)
        train = [(vid, train_metadata[vid]["language"], train_metadata[vid]) for vid in train_metadata]

    with open(val_registry, "r") as f:
        val_metadata = json.load(f)
        val = [(vid, val_metadata[vid]["language"], val_metadata[vid]) for vid in val_metadata]

    # Assemble *all* language
    language = [x[1] for x in train + val]

    # Build AutoTokenizer (from `language_model` identifier)
    tokenizer = AutoTokenizer.from_pretrained(hf_cache)

    # If `max_lang_len` not specified, dump some statistics to compute...
    if max_lang_len == -1:
        # Naively tokenizes and pads to the "maximum length" of _all_ language... long tail is a problem!
        encoded_language = tokenizer(language, return_tensors="pt", padding=True)
        lengths = encoded_language["attention_mask"].sum(dim=1)

        # Compute a histogram of lengths
        hist = lengths.float().histc(bins=lengths.max()).int()
        overwatch.info(f"Histogram: {hist.numpy().tolist()}")
        raise AssertionError("Compute max length and update dataset configuration!")

    # Otherwise, we've already set the maximum length, so let's use it!
    overwatch.info(f"\tTokenizing all language in dataset to maximum length `{max_lang_len}`")
    encoded_language = tokenizer(
        language, return_tensors="pt", max_length=max_lang_len, truncation=True, padding="max_length"
    )
    input_ids, attention_mask = encoded_language["input_ids"], encoded_language["attention_mask"]
    train_input_ids, train_attention_mask = input_ids[: len(train)], attention_mask[: len(train)]
    val_input_ids, val_attention_mask = input_ids[len(train):], attention_mask[len(train):]

    # Assertion, just to sanity check
    assert len(val_input_ids) == len(val_attention_mask) == len(val), "Something went wrong tokenizing language..."

    # Compute `index.pt` contents
    overwatch.info("\tAssembling `train` and `val` index structures...")
    train_pt = {
        train[i][0]: {**train[i][2], **{"input_ids": train_input_ids[i], "attention_mask": train_attention_mask[i]}}
        for i in range(len(train))
    }
    val_pt = {
        val[i][0]: {**val[i][2], **{"input_ids": val_input_ids[i], "attention_mask": val_attention_mask[i]}}
        for i in range(len(val))
    }

    # Additionally dump JSON versions of the same --> downstream interpretability, XLA
    overwatch.info("JSONifying both Train and Validation Language")
    train_json, val_json = {}, {}
    for vid in track(train_pt, description="Train Language :: ", transient=True):
        train_json[vid] = {
            "language": train_pt[vid]["language"],
            "n_frames": train_pt[vid]["n_frames"],
            "input_ids": train_pt[vid]["input_ids"].numpy().tolist(),
            "attention_mask": train_pt[vid]["attention_mask"].numpy().tolist(),
        }

    for vid in track(val_pt, description="Validation Language :: ", transient=True):
        val_json[vid] = {
            "language": val_pt[vid]["language"],
            "n_frames": val_pt[vid]["n_frames"],
            "input_ids": val_pt[vid]["input_ids"].numpy().tolist(),
            "attention_mask": val_pt[vid]["attention_mask"].numpy().tolist(),
        }

    # Dump Structures...
    overwatch.info(f"Saving Torch indices to `{t_index}` and `{v_index}` respectively...")
    torch.save(train_pt, t_index)
    torch.save(val_pt, v_index)

    overwatch.info(f"Saving JSON indices to `{t_json}` and `{v_json}` respectively...")
    with open(t_json, "w") as f:
        json.dump(train_json, f)

    with open(v_json, "w") as f:
        json.dump(val_json, f)

    # Pull relevant files out into their own `index` directory...
    shutil.copy(t_json, index_dir / "train-language-index.json")
    shutil.copy(v_json, index_dir / "val-language-index.json")
    shutil.copy(t_index, index_dir / "train-language-index.pt")
    shutil.copy(v_index, index_dir / "val-language-index.pt")
    return index_dir

def extract_tactile_data(
    name: str,
    train_registry: Path,
    val_registry: Path,
    tac_trig_thre: float,
    path: str,
    artifact_path: str,
    rewrite:bool=False,
    debug_mode:bool=False
) -> Tuple[Path, Path, Dict[str, List[str]]]:
    """
    Phase 3 :: Extract & Normalize & Binary Tactile Data, serialize Frames align to Video Clips
    --> get `registry` (index files) for train and validation
    """
    overwatch.info(f"Phase 3 Preprocessing :: Extracting Tactile Frames for Dataset `{name}`")
    # extract to which dir
    t_dir, v_dir = Path(artifact_path) / name / "train" / "tactile", Path(artifact_path) / name / "val" / "tactile"

    # Rewrite
    if rewrite:
        if os.path.exists(str(t_dir / "tactile_registry.json")):
            os.remove(str(t_dir / "tactile_registry.json"))
            os.remove(str(v_dir / "tactile_registry.json"))
    # create dir
    os.makedirs(t_dir, exist_ok=True)
    os.makedirs(v_dir, exist_ok=True)

    # Grab train $ val registry
    with open(train_registry, "r") as f:
        train_metadata = json.load(f)
        train_ids = list(train_metadata.keys())

    with open(val_registry, "r") as f:
        val_metadata = json.load(f)
        val_ids = list(val_metadata.keys())

    # Processs
    tac_triggered_dict = {"train": dict(), "val": dict()}
    with mp.Pool(mp.cpu_count()) as pool:
        for k, save, vids, metadata in [("train", t_dir, train_ids, train_metadata), ("val", v_dir, val_ids, val_metadata)]:
            overwatch.info(f"\tWriting `{k}` tactile clips to disk...")
            preprocess_transforms_list = [
                # ('event',         PreprocessTactile.event),
                # ('normalize_global_0_3', PreprocessTactile.normalize_global),
                # ('normalize_self_0_3', partial(PreprocessTactile.normalize_self, 0.3)),
                ('binary_0_3',      partial(PreprocessTactile.binary, 0.3)),
                # ('raw',           PreprocessTactile.origin)
            ]
            process_fn = partial(process_tactile_clip, Path(path), save, rewrite, tac_trig_thre, preprocess_transforms_list)
            tactile_registry = dict()
            frames_num_align_list = list()
            if not debug_mode:
                for key, processed_frames, vid_trig in track(
                        pool.imap_unordered(process_fn, vids),
                        total=len(vids),
                        transient=True,
                ):
                    tac_triggered_dict[k].update(vid_trig) 
                    tactile_registry[str(key)] = [processed_frames, metadata[key]["n_frames"]]
                    if processed_frames > 0:
                        frames_num_align_list.append(processed_frames == metadata[key]["n_frames"])
            else:
                for vid in vids:
                    vid, processed_frames, vid_trig = process_fn(vid)
                    tac_triggered_dict[k].update(vid_trig)  
                    tactile_registry[str(vid)] = [processed_frames, metadata[vid]["n_frames"]]
                    if processed_frames > 0:
                        frames_num_align_list.append(processed_frames == metadata[vid]["n_frames"])
            
            if not all(frames_num_align_list):
                overwatch.error(f"{k} tactile clips cannot match with videos clips!")
                raise ValueError(f"{k} tactile clips cannot match with videos clips!")
            
            save_trigger_json = json.dumps(tac_triggered_dict[k], indent=2)
            save_trigger_json_path = t_dir / "tactile_triggered.json" if k == "train" else \
                                     v_dir / "tactile_triggered.json"
            with open(str(save_trigger_json_path), "w", newline='\n') as f:
                f.write(save_trigger_json)
            
            save_registry_json = json.dumps(tactile_registry, indent=2)
            save_registry_json_path = t_dir / "tactile_registry.json" if k == "train" else \
                                      v_dir / "tactile_registry.json"
            with open(str(save_registry_json_path), "w", newline='\n') as f:
                f.write(save_registry_json)
    return t_dir, v_dir, tac_triggered_dict

def unify_batches(
    name: str,
    train_registry: Path,
    val_registry: Path,
    train_dir: Dict[str, Path],
    val_dir: Dict[str, Path],
    index_dir: Path,
    multimodel_input: dict,
    data_modality: List[str],
    batch_formats: Tuple[Tuple[str, Tuple[str, ...]], ...],
    max_epochs: int = 400,
    initial_final_alpha: float = 0.2,
    rewrite:bool = False
) -> None:
    """Phase III: Assemble "Data-Locked" Batches for *all models* for *all epochs* for consistency!"""
    overwatch.info(f"Phase 3 Preprocessing :: Assembling *Data-Locked* Batches for Dataset `{name}`")

    # Load Registries
    with open(train_registry, "r") as f:
        train_registrations = json.load(f)

    with open(val_registry, "r") as f:
        val_registrations = json.load(f)

    # batch_format
    full_set_inputs, b_keys, unique_states = set(), set(), set()
    batch_formats_union = []
    do_initial, do_final = False, False
    for name, format in batch_formats:
        # add the type of input
        full_set_inputs.update(set(format))
        b_keys.add(name)
        
        union_format = []
        state_num = 0
        for element in format:
            if element == "state_initial":
                union_format.append("initial")
                do_initial = True
            elif element == "state_final":
                union_format.append("final")
                do_final = True
            elif "state_" in element:
                state_num += 1
            else:
                union_format.append(element)
        union_format.append(state_num)

        batch_formats_union.append(
            [name, union_format]
        )

    # Rewrite
    if rewrite:
        for b in b_keys:
            if os.path.exists(str(index_dir / b)):
                shutil.rmtree(str(index_dir / b))

    # Check if we have initial and final states
    state_elements = [s for s in full_set_inputs if "state_" in s]
    n_int = len(state_elements) - 2 if ("state_initial" in state_elements and "state_final" in state_elements) else 0

    # Serialize Epochs
    overwatch.info("\tSerializing Epochs to JSON --> Storing mapping of Epoch -> Image Paths")
    for b in b_keys:
        os.makedirs(index_dir / b, exist_ok=True)

    # We only write the Validation Epoch once --> held constant across *all* of training!
    overwatch.info("\tWriting Validation Epoch to Disk")
    val_epoch_idx, _, uniq_s = serialize_epoch(
        index_dir,
        val_registrations,
        val_dir,
        batch_formats_union,
        do_initial,
        do_final,
        initial_final_alpha,
        n_int,
        data_modality,
        multimodel_input,
        epoch=0,
        is_validation=True
    )

    # Update Trackers...
    if val_epoch_idx != -1:
        unique_states |= uniq_s

    # Compute length of epochs --> CPU Count should be no higher...
    epochs, n_frames_per_epoch = list(range(max_epochs)), -1

    # Parallelize Train Epoch Serialization
    overwatch.info("\tPlacing the Train Registry into Shared Memory")
    manager = mp.Manager()
    mg_registry = manager.dict(train_registrations)

    # Multiprocess --> the memory demands here are a bit higher, so limit workers by factor of 4
    with mp.Pool(mp.cpu_count() // 4) as pool:
        overwatch.info("\tWriting Train Batches per Epoch to Disk")
        precompute_fn = partial(
            serialize_epoch,
            index_dir,
            mg_registry,
            train_dir,
            batch_formats_union,
            do_initial,
            do_final,
            initial_final_alpha,
            n_int,
            data_modality,
            multimodel_input
        )
        for epoch_idx, n_frames, uniq_s in pool.imap_unordered(precompute_fn, epochs):
            if epoch_idx == -1:
                continue

            # Update Trackers
            unique_states |= uniq_s
            n_frames_per_epoch = n_frames

    # Dump Statistics (Note :: Only makes sense on "initial" computation --> uninterrupted!)
    overwatch.info(f"Train Uniqueness: {len(unique_states)} States & {len(mg_registry)} Utterances")
    overwatch.info(f"Final Statistics :: 1 Epoch has ~ {n_frames_per_epoch} Frames...")
