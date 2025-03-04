"""
preprocess.py

Centralized script for preprocessing various video/vision-language datasets for GPU pretraining, using a multi-stage,
multiprocessing approach.

Run as a standalone script, *prior* to calling `pretrain.py` =>> mostly because we want to preprocess the data once, as
a fixed cost.
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from vitac.conf import DatasetConfig, BottleCapConfig
from vitac.overwatch import OverwatchRich
from vitac.preprocessing import extract_frames, preprocess_language, extract_tactile_data, unify_batches
from vitac.util import set_global_seed

# Grab Logger
overwatch = logging.getLogger(__file__)

@dataclass
class PreprocessingConfig:
    # fmt: off
    defaults: List[Any] = field(default_factory=lambda: ["_self_",
                                                         {"dataset": "BottleCap"},
                                                         {"override hydra/job_logging": "overwatch_rich"}
                                                         ])
    hydra: Dict[str, Any] = field(
        default_factory=lambda: {"run": {"dir": "./runs/preprocessing/${now:%m-%d}/dataset-${dataset.name}"}}
    )
    # Generated data modalities
    data_modality: List[Any] = field(default_factory=lambda:["vision",
                                                             "tactile"])

    # Whether to overwrite existing files
    if_rewrite: bool = True

    # Threshold to determine if tactile data is triggered
    tac_trigger_threshold: float = 0.3

    # Debug mode (single process instead of multiprocessing)
    if_debug: bool = False

    # Command Line Arguments
    seed: int = 21                                  # Random Seed (for reproducibility)

    # Composable / Structured Arguments
    dataset: DatasetConfig = MISSING                # Dataset(s) for pretraining/preprocessing
    # fmt: on


# Hydra Setup :: Retrieve ConfigStore (Singleton) & Register Components
cs = ConfigStore.instance()
cs.store(group="dataset", name="BottleCap", node=BottleCapConfig)
cs.store(group="hydra/job_logging", name="overwatch_rich", node=OverwatchRich)
cs.store(name="config", node=PreprocessingConfig)

@hydra.main(config_path=None, config_name="config")
def preprocess(cfg: PreprocessingConfig) -> None:
    overwatch.info("Preprocessing :: Running Phases for Frame Extraction, Language Compilation, and Batching...")

    # Set Randomness
    set_global_seed(cfg.seed)
    # Vision modality is necessary in this version
    assert "vision" in cfg.data_modality, "Vision modality is necessary in this Version"

    unify_t_dir, unify_v_dir = dict(), dict()
    multimodel_input = dict()
    # Phase 1 :: Serialize Frames from Video Clips --> get `registry` (index files) for train and validation
    train_registry, val_registry, train_dir, val_dir = extract_frames(
        cfg.dataset.name,
        path=cfg.dataset.path,
        artifact_path=cfg.dataset.artifact_path,
        preprocess_resolution=cfg.dataset.preprocess_resolution,
        train_val_ratio=cfg.dataset.train_val_ratio,
        rewrite=cfg.if_rewrite,
    )
    unify_t_dir["vision"] = train_dir
    unify_v_dir["vision"] = val_dir

    # Phase 2 :: Normalize & Tokenize Language --> create `index.pt` and `index.json` files
    if "language" in cfg.data_modality:
        index_dir = preprocess_language(
            cfg.dataset.name,
            train_registry,
            val_registry,
            artifact_path=cfg.dataset.artifact_path,
            max_lang_len=cfg.dataset.max_lang_len,
            language_model=cfg.dataset.language_model,
            hf_cache=cfg.dataset.hf_cache,
            rewrite=cfg.if_rewrite
        )
        unify_t_dir["language"] = index_dir
        unify_v_dir["language"] = index_dir
    else:
        index_dir = Path(cfg.dataset.artifact_path) / cfg.dataset.name / "index"

    # Phase 3 :: Extract & Normalize & Binary Tactile Data, serialize Frames align to Video Clips
    #            --> get `registry` (index files) for train and validation
    if "tactile" in cfg.data_modality:
        (
            unify_t_dir["tactile"],
            unify_v_dir["tactile"],
            multimodel_input["tac_triggered"]
        ) = extract_tactile_data(
            cfg.dataset.name,
            train_registry,
            val_registry,
            tac_trig_thre=cfg.tac_trigger_threshold,
            path=cfg.dataset.path,
            artifact_path=cfg.dataset.artifact_path,
            rewrite=cfg.if_rewrite,
            debug_mode=cfg.if_debug
        )

    # Phase 4 :: Assemble "Data-Locked" Batch Sets for Various Models (e.g., for single-frame/dual-frame/quintet)

    unify_batches(
        cfg.dataset.name,
        train_registry,
        val_registry,
        unify_t_dir,
        unify_v_dir,
        index_dir,
        multimodel_input,
        data_modality=cfg.data_modality,
        batch_formats=cfg.dataset.batch_formats,
        max_epochs=cfg.dataset.max_epochs,
        initial_final_alpha=cfg.dataset.initial_final_alpha,
        rewrite=cfg.if_rewrite
    )

    overwatch.info("Preprocessing Complete!")


if __name__ == "__main__":
    preprocess()
