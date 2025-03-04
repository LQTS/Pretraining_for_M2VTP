"""
instantiate.py

Simple wrapping script for instantiating a core Voltron/reproduction model and configuring the torch.Optimizer for DDP
pretraining. Meant to be modular and extensible!
"""
from typing import Callable, Tuple

import torch.nn as nn
from torch.optim import Optimizer

from vitac.conf import DatasetConfig, ModelConfig

import json

from .core.v_repic import V_RePic
from .core.vtt_reall import VTT_ReAll
from .core.t_retac import T_ReTac
from pathlib import Path



def get_model_optimizer(
    model_cfg: ModelConfig, dataset_cfg: DatasetConfig, model_json_save_dir: Path,
    if_save_model_cfg: bool=True
) -> Tuple[nn.Module, Optimizer, Callable[[int, float], float]]:
    """Switch on `model_cfg.arch` --> instantiate the correct nn.Module and Optimizer (on CPU/default device)."""
    # create input dict
    coder_dict = {
        "encoder_depth": int(model_cfg.encoder_depth),
        "encoder_embed_dim": int(model_cfg.encoder_embed_dim),
        "encoder_n_heads": int(model_cfg.encoder_n_heads),
        "decoder_depth": int(model_cfg.decoder_depth),
        "decoder_embed_dim": int(model_cfg.decoder_embed_dim),
        "decoder_n_heads": int(model_cfg.decoder_n_heads),
    }
    train_cfg_param = {
        "optimizer": str(model_cfg.optimizer),
        "schedule": str(model_cfg.schedule),
        "base_lr": float(model_cfg.base_lr),
        "min_lr": float(model_cfg.min_lr),
        "effective_bsz": int(model_cfg.effective_bsz),
        "betas": (model_cfg.betas[0], model_cfg.betas[1]),
        "weight_decay": float(model_cfg.weight_decay),
        "mlp_ratio": float(model_cfg.mlp_ratio),
        "norm_pixel_loss": bool(model_cfg.norm_pixel_loss),
        "use_cls_token": bool(model_cfg.use_cls_token)
    }
    dataset_cfg_param = {
        "resolution": int(dataset_cfg.resolution),
        "in_channels": 3,           
        "warmup_epochs": int(dataset_cfg.warmup_epochs),
        "max_epochs": int(dataset_cfg.max_epochs),

    }
    model_selection = {
        "vtt-reall": VTT_ReAll,
        "v-repic": V_RePic,
        "t-retac": T_ReTac,
    }
    input_cfg_param_img_set = {
        "patch_size": int(model_cfg.patch_size),
        "mask_ratio": float(model_cfg.mask_ratio)
    }
    if model_cfg.arch not in {"v-repic"}:
        input_cfg_param_tac_set = {
            "tactile_dim": int(model_cfg.tactile_dim),
            "tactile_en_dim": int(model_cfg.tactile_en_dim),
            "tactile_type": str(model_cfg.tactile_type.split('_')[0]),
            "tactile_emb_type": str(model_cfg.tactile_emb_type),
            "tactile_mask_ratio": float(model_cfg.tactile_mask_ratio)
        }
    input_cfg_param = dict()
    # Data-Locked Reproductions
    if model_cfg.arch == "v-repic":
        input_cfg_param.update(input_cfg_param_img_set)


    elif model_cfg.arch == "t-retac":
        input_cfg_param.update(input_cfg_param_tac_set)

    elif model_cfg.arch == "vtt-reall":
        input_cfg_param.update(input_cfg_param_img_set)
        input_cfg_param.update(input_cfg_param_tac_set)
        input_cfg_param["tactile_set_to_ZERO"] = bool(model_cfg.tactile_set_to_ZERO)
        input_cfg_param["img_weight"] = float(model_cfg.img_weight)
        input_cfg_param["tac_weight"] = float(model_cfg.tac_weight)

    else:
        raise ValueError(f"Input Config for Model Architecture `{model_cfg.arch}` is not implemented!")

    # set model
    assert model_cfg.arch in list(model_selection.keys()), \
          f"Model selection for Model Arch {model_cfg.arch} is not implemented!"
    model = model_selection[model_cfg.arch](
        encoder_decoder_cfg=coder_dict,
        train_cfg=train_cfg_param,
        input_cfg=input_cfg_param,
        dataset_cfg=dataset_cfg_param
    )

    # save Input Parameters as Json Files
    if if_save_model_cfg:
        model_json_save_dir = str(model_json_save_dir)
        if len(model_json_save_dir) > 5 and model_json_save_dir[-5:] == ".json":
            input_param_dict = {
                "encoder_decoder_cfg": coder_dict,
                "train_cfg": train_cfg_param,
                "input_cfg": input_cfg_param,
                "dataset_cfg": dataset_cfg_param
            }
            json_str = json.dumps(input_param_dict, indent=2)
            with open(model_json_save_dir, 'w') as f:
                f.write(json_str)

    # Configure Optimizer --> on same device (CPU)
    optimizer, update_lr = model.configure_optimizer()

    return model, optimizer, update_lr
