"""
models.py

Base Hydra Structured Config for defining various pretraining model architectures and appropriate configurations. Uses a
simple single inheritance structure.
"""
from dataclasses import dataclass
from typing import Tuple, List, Any

from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path
from omegaconf import MISSING
from dataclasses import dataclass, field



@dataclass
class ModelConfig:
    arch: str = MISSING
    identifier: str = MISSING
    remarks: str = MISSING

    # Dataset Modality
    data_modality: str = MISSING
    data_formats: List[Any] = MISSING

    # Default Vision Transformer Configuration
    patch_size: int = 16
    mlp_ratio: float = 4.0

    # Effective batch size --> total number of examples before gradient update
    effective_bsz: int = MISSING

    # Number of examples one can safely fit on an accelerator w/ this model!
    device_bsz: int = MISSING  # For backwards compatibility, only use device_bsz for XLA/TPU pretraining...
    native_bsz: int = MISSING  # For backwards compatibility, define a separate `native_bsz`...


# @Data-Locked Reproductions --- Encompasses MVP (MAE) + R3M


# MVP (Base Masked Autoencoder)
@dataclass
class MVPConfig(ModelConfig):
    arch: str = "v-mvp"
    identifier: str = MISSING

    # Dataset Modality
    data_modality: str = "state"

    # Base MAE Parameters
    mask_ratio: float = 0.75

    # Architecture Parameters
    encoder_depth: int = MISSING
    encoder_embed_dim: int = MISSING
    encoder_n_heads: int = MISSING

    decoder_depth: int = MISSING
    decoder_embed_dim: int = MISSING
    decoder_n_heads: int = MISSING

    # MAE Loss/Objective Configuration
    norm_pixel_loss: bool = True
    effective_bsz: int = 1024
    device_bsz: int = MISSING
    native_bsz: int = MISSING

    # Optimization Parameters
    optimizer: str = "adamw"
    schedule: str = "linear-warmup+cosine-decay"
    base_lr: float = 1.5e-4
    min_lr: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.05


@dataclass
class MVPSmallConfig(MVPConfig):
    identifier = "r-mvp"

    # Architecture Parameters -- should match ViT Small Architecture to the letter!
    #   Note: Small is defined in TIMM:
    #       > https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py#L683
    encoder_depth = 12
    encoder_embed_dim = 384
    encoder_n_heads = 6

    decoder_depth = 6
    decoder_embed_dim = 192
    decoder_n_heads = 6

    # Number of examples one can safely fit on an accelerator w/ this model!
    #   > TPU-v3: max of 128 per device.
    device_bsz = 128
    native_bsz = 128


# R3M Models --> Just different visual encoders, roughly following the above!
@dataclass
class R3MConfig(ModelConfig):
    arch: str = "v-r3m"
    identifier: str = MISSING

    # Dataset Modality
    data_modality: str = "quintet+language"

    # ViT Architecture Parameters
    depth: int = MISSING
    embed_dim: int = MISSING
    n_heads: int = MISSING

    # Effective Batch Size
    effective_bsz: int = 1024

    # Language Model Parameters
    language_model: str = "distilbert-base-uncased"
    hf_cache: str = to_absolute_path("data/hf-cache")
    language_dim: int = 768
    vocab_size: int = 30522
    reward_dim: int = 1024

    # Loss/Objective Configuration
    lang_reward_weight: float = 1.0
    tcn_weight: float = 1.0
    l1_weight: float = 1e-5
    l2_weight: float = 1e-5
    n_negatives: int = 3
    device_bsz: int = MISSING
    native_bsz: int = MISSING

    # Optimization Parameters
    optimizer: str = "adam"
    schedule: str = "linear-warmup+cosine-decay"
    lr: float = 1e-4
    min_lr: float = 0.0


@dataclass
class R3MSmallConfig(R3MConfig):
    identifier = "r-r3m-vit"

    # Architecture Parameters -- should match ViT Small Architecture to the letter!
    #   Note: Small is defined in TIMM:
    #       > https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py#L683
    depth = 12
    embed_dim = 384
    n_heads = 6

    # Device Batch Size
    device_bsz = 32
    native_bsz = 128


# R3M -- ResNet50 Encoder (instead of ViT)
@dataclass
class ResNet3MConfig(ModelConfig):
    arch: str = "v-rn3m"
    identifier: str = MISSING

    # Dataset Modality
    data_modality: str = "quintet+language"

    # Effective Batch Size
    effective_bsz: int = 1024

    # Architecture Parameters
    fc_dim: int = MISSING

    # Language Model Parameters
    language_model: str = "distilbert-base-uncased"
    hf_cache: str = to_absolute_path("data/hf-cache")
    language_dim: int = 768
    vocab_size: int = 30522
    reward_dim: int = 1024

    # Loss/Objective Configuration
    lang_reward_weight: float = 1.0
    tcn_weight: float = 1.0
    l1_weight: float = 1e-5
    l2_weight: float = 1e-5
    n_negatives: int = 3
    device_bsz: int = MISSING
    native_bsz: int = MISSING

    # Optimization Parameters
    optimizer: str = "adam"
    lr: float = 1e-4


class RN3M50Config(ResNet3MConfig):
    identifier = "r-r3m-rn50"

    # Architecture Parameters
    fc_dim = 2048

    # Device Batch Size
    device_bsz = 32
    native_bsz = 128


# @Voltron Models -- VCond, VDual, VGen


# VCond -- Single Frame + Language Conditioning
@dataclass
class VCondConfig(ModelConfig):
    arch: str = "v-cond"
    identifier: str = MISSING

    # Dataset Modality
    data_modality: str = "state+language"

    # Base MAE Parameters
    mask_ratio: float = 0.75

    # Base Language Parameters --> full sentence dropout only...
    language_model: str = "distilbert-base-uncased"
    hf_cache: str = "/remote-home/share/lqt/touch_vision_manipulation/data/hf-cache" # to_absolute_path("data/hf-cache")
    language_dim: int = 768
    vocab_size: int = 30522
    lang_dropout: float = MISSING

    # Architecture Parameters
    encoder_depth: int = MISSING
    encoder_embed_dim: int = MISSING
    encoder_n_heads: int = MISSING

    decoder_depth: int = MISSING
    decoder_embed_dim: int = MISSING
    decoder_n_heads: int = MISSING

    use_cls_token: bool = True

    # MAE Loss/Objective Configuration
    norm_pixel_loss: bool = True
    effective_bsz: int = 32 # 1024
    device_bsz: int = MISSING
    native_bsz: int = MISSING

    # Optimization Parameters
    optimizer: str = "adamw"
    schedule: str = "linear-warmup+cosine-decay"
    base_lr: float = 1.5e-4
    min_lr: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.05


@dataclass
class VCondSmallConfig(VCondConfig):
    identifier = "v-cond"

    # No language dropout...
    lang_dropout = 0.0

    # Architecture Parameters -- should match ViT Small Architecture to the letter!
    #   Note: Small is defined in TIMM:
    #       > https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py#L683
    encoder_depth = 12
    encoder_embed_dim = 384
    encoder_n_heads = 6

    decoder_depth = 6
    decoder_embed_dim = 192
    decoder_n_heads = 6

    # Number of examples one can safely fit on an accelerator w/ this model!
    #   > TPU-v3: max of 128 per device
    #   > GPU w/ 32G of RAM: max of 128 per device!
    device_bsz = 8#128
    native_bsz = 8#128


@dataclass
class VCondBaseConfig(VCondConfig):
    identifier = "v-cond-base"

    # No language dropout...
    lang_dropout = 0.0

    # Architecture Parameters -- should match ViT Base Architecture to the letter!
    #   Note: Base is defined in TIMM & Original MAE Repository:
    #       > https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py#L723
    #       > https://github.com/facebookresearch/mae/blob/main/models_mae.py#L223
    encoder_depth = 12
    encoder_embed_dim = 768
    encoder_n_heads = 12

    decoder_depth = 8
    decoder_embed_dim = 512
    decoder_n_heads = 16

    # Number of examples one can safely fit on an accelerator w/ this model!
    #   > TPU-v3: max of 128 per device!
    #   > GPU w/ 32G of RAM: max of 128 per device!
    device_bsz = 128
    native_bsz = 128


# VDual - Dual Frame (0th Frame + Kth frame) + Language Conditioning
@dataclass
class VDualConfig(ModelConfig):
    arch: str = "v-dual"
    identifier: str = MISSING

    # Dataset Modality
    data_modality: str = "state+ok"

    # Base MAE Parameters
    mae_weight: float = 1.0
    mask_ratio: float = 0.75

    # Base Language Parameters --> full sentence dropout only...
    language_model: str = "distilbert-base-uncased"
    hf_cache: str = to_absolute_path("data/hf-cache")
    language_dim: int = 768
    vocab_size: int = 30522
    lang_dropout: float = MISSING

    # Architecture Parameters
    encoder_depth: int = MISSING
    encoder_embed_dim: int = MISSING
    encoder_n_heads: int = MISSING

    decoder_depth: int = MISSING
    decoder_embed_dim: int = MISSING
    decoder_n_heads: int = MISSING

    use_cls_token: bool = True

    # MAE Loss/Objective Configuration -- Cut effective batch size since we see 12-25x contexts per batch example!
    norm_pixel_loss: bool = True
    effective_bsz: int = 1024
    device_bsz: int = MISSING
    native_bsz: int = MISSING

    # Optimization Parameters
    optimizer: str = "adamw"
    schedule: str = "linear-warmup+cosine-decay"
    base_lr: float = 1.5e-4
    min_lr: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.05


@dataclass
class VDualSmallConfig(VDualConfig):
    identifier = "v-dual"

    # No language dropout...
    lang_dropout = 0.0

    # Architecture Parameters -- should match ViT Small Architecture to the letter!
    #   Note: Small is defined in TIMM:
    #       > https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py#L683
    encoder_depth = 12
    encoder_embed_dim = 384
    encoder_n_heads = 6

    decoder_depth = 6
    decoder_embed_dim = 192
    decoder_n_heads = 6

    # Number of examples one can safely fit on an accelerator w/ this model!
    #   > TPU-v3: max of 128 per device!
    #   > GPU w/ 32G of RAM: max of 128 per device!
    device_bsz = 128
    native_bsz = 128


@dataclass
class VDualBaseConfig(VDualConfig):
    identifier = "v-dual-base"

    # No language dropout...
    lang_dropout = 0.0

    # Architecture Parameters -- should match ViT Base Architecture to the letter!
    #   Note: Base is defined in TIMM & Original MAE Repository:
    #       > https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py#L723
    #       > https://github.com/facebookresearch/mae/blob/main/models_mae.py#L223
    encoder_depth = 12
    encoder_embed_dim = 768
    encoder_n_heads = 12

    decoder_depth = 8
    decoder_embed_dim = 512
    decoder_n_heads = 16

    # Number of examples one can safely fit on an accelerator w/ this model!
    #   > TPU-v3: max of 128 per device!
    #   > GPU w/ 32G of RAM: max of 64 per device!
    device_bsz = 128
    native_bsz = 64


# VGen - Dual Frame with Language Conditioning AND Language Generation
@dataclass
class VGenConfig(ModelConfig):
    arch: str = "v-gen"
    identifier: str = MISSING

    # Dataset Modality
    data_modality: str = "state+ok"

    # Base MAE & LM Parameters --> LM Weight is set such that mae & lang loss are ~same order of magnitude
    mae_weight: float = 1.0
    lm_weight: float = 0.5
    mask_ratio: float = 0.75
    gen_ratio: float = MISSING

    # Base Language Parameters
    language_model: str = "distilbert-base-uncased"
    hf_cache: str = to_absolute_path("data/hf-cache")
    language_dim: int = 768
    vocab_size: int = 30522

    # Architecture Parameters
    encoder_depth: int = MISSING
    encoder_embed_dim: int = MISSING
    encoder_n_heads: int = MISSING

    decoder_depth: int = MISSING
    decoder_embed_dim: int = MISSING
    decoder_n_heads: int = MISSING

    use_cls_token: bool = True

    # MAE Loss/Objective Configuration -- Cut effective batch size since we see 12-25x contexts per batch example!
    norm_pixel_loss: bool = True
    effective_bsz: int = 1024
    device_bsz: int = MISSING
    native_bsz: int = MISSING

    # Optimization Parameters
    optimizer: str = "adamw"
    schedule: str = "linear-warmup+cosine-decay"
    base_lr: float = 1.5e-4
    min_lr: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.05


@dataclass
class VGen50SmallConfig(VGenConfig):
    identifier = "v-gen"

    # LM Parameters --> control % of examples that are for "language generation" (no conditioning)
    gen_ratio = 0.50

    # Architecture Parameters -- should match ViT Small Architecture to the letter!
    #   Note: Small is defined in TIMM:
    #       > https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py#L683
    encoder_depth = 12
    encoder_embed_dim = 384
    encoder_n_heads = 6

    decoder_depth = 6
    decoder_embed_dim = 192
    decoder_n_heads = 6

    # Number of examples one can safely fit on an accelerator w/ this model!
    #   > TPU-v3: max of 64 per device!
    #   > GPU w/ 32G of RAM: max of 64 per device!
    device_bsz = 64
    native_bsz = 64


@dataclass
class VGen50BaseConfig(VGenConfig):
    identifier = "v-gen-base"

    # LM Parameters --> control % of examples that are for "language generation" (no conditioning)
    gen_ratio = 0.50

    # Architecture Parameters -- should match ViT Base Architecture to the letter!
    #   Note: Base is defined in TIMM & Original MAE Repository:
    #       > https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py#L723
    #       > https://github.com/facebookresearch/mae/blob/main/models_mae.py#L223
    encoder_depth = 12
    encoder_embed_dim = 768
    encoder_n_heads = 12

    decoder_depth = 8
    decoder_embed_dim = 512
    decoder_n_heads = 16

    # Number of examples one can safely fit on an accelerator w/ this model!
    #   > TPU-v3: max of 32 per device!
    #   > GPU w/ 32G of RAM: max of 32 per device!
    device_bsz = 32
    native_bsz = 32

@dataclass
class VRePicBasicConfig(ModelConfig):
    arch: str = "v-repic"
    identifier: str = MISSING

    # Dataset Modality
    data_modality: str = "only_in"
    data_formats: List[Any] = field(default_factory=lambda: [
        "vision"
    ])

    # Base MAE Parameters
    mask_ratio: float = 0.15

    # Architecture Parameters
    encoder_depth: int = MISSING
    encoder_embed_dim: int = MISSING
    encoder_n_heads: int = MISSING

    decoder_depth: int = MISSING
    decoder_embed_dim: int = MISSING
    decoder_n_heads: int = MISSING

    use_cls_token: bool = True

    # tactile type
    tactile_type: str = MISSING

    # MAE Loss/Objective Configuration
    norm_pixel_loss: bool = True
    effective_bsz: int = 32  
    device_bsz: int = MISSING
    native_bsz: int = MISSING

    # Optimization Parameters
    optimizer: str = "adamw"
    schedule: str = "linear-warmup+cosine-decay"
    base_lr: float = 1.5e-4
    min_lr: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.05

@dataclass
class VRePicConfig(VRePicBasicConfig):
    # identifier
    identifier = "v-repic"
    remarks = "base"

    # Architecture Parameters -- should match ViT Small Architecture to the letter!
    #   Note: Small is defined in TIMM:
    #       > https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py#L683
    encoder_depth = 12
    encoder_embed_dim = 384
    encoder_n_heads = 6

    decoder_depth = 6
    decoder_embed_dim = 192
    decoder_n_heads = 6

    # tactile type (necessary although not use)
    tactile_type: str = 'normalize_global_0_3'

    # Number of examples one can safely fit on an accelerator w/ this model!
    #   > TPU-v3: max of 128 per device
    #   > GPU w/ 32G of RAM: max of 128 per device!
    device_bsz = 8  # 128 # effectie_bsz / nums_work
    native_bsz = 8  # 128 # effectie_bsz / nums_work

    # mask ratio
    mask_ratio = 0.15

@dataclass
class VTTReAllBasicConfig(ModelConfig):
    arch: str = "vtt-reall"
    identifier: str = MISSING

    # Dataset Modality
    data_modality: str = "only_in"
    data_formats: List[Any] = field(default_factory=lambda: [
        "vision",
        "tactile"
    ])

    # Base MAE Parameters
    mask_ratio: float = 0.75

    # Architecture Parameters
    encoder_depth: int = MISSING
    encoder_embed_dim: int = MISSING
    encoder_n_heads: int = MISSING

    decoder_depth: int = MISSING
    decoder_embed_dim: int = MISSING
    decoder_n_heads: int = MISSING

    use_cls_token: bool = True

    # MAE Loss/Objective Configuration
    norm_pixel_loss: bool = True
    effective_bsz: int = 32  
    device_bsz: int = MISSING
    native_bsz: int = MISSING

    # Optimization Parameters
    optimizer: str = "adamw"
    schedule: str = "linear-warmup+cosine-decay"
    base_lr: float = 1.5e-4
    min_lr: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.05

@dataclass
class VTTReAllConfig(VTTReAllBasicConfig):
    # identifier
    identifier = "vtt-reall"
    remarks = ""

    # Architecture Parameters -- should match ViT Small Architecture to the letter!
    #   Note: Small is defined in TIMM:
    #       > https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py#L683
    encoder_depth = 12
    encoder_embed_dim = 384
    encoder_n_heads = 6

    decoder_depth = 6
    decoder_embed_dim = 192
    decoder_n_heads = 6

    # Base Tactile Parameters
    tactile_dim: int = 20
    tactile_en_dim: int = 768
    tactile_type: str = 'normalize_global_0_3'
    tactile_set_to_ZERO: bool = False
    tactile_emb_type: str = "flatten"
    tactile_mask_ratio: float = 0.75

    # loss weight
    img_weight: float = 0.6
    tac_weight: float = 0.4

    # Number of examples one can safely fit on an accelerator w/ this model!
    #   > TPU-v3: max of 128 per device
    #   > GPU w/ 32G of RAM: max of 128 per device!
    device_bsz = 8  # 128 # effectie_bsz / nums_work
    native_bsz = 8  # 128 # effectie_bsz / nums_work

    # mask ratio
    mask_ratio = 0.75

@dataclass
class TReTacBasicConfig(ModelConfig):
    arch: str = "t-retac"
    identifier: str = MISSING

    # Dataset Modality
    data_modality: str = "only_in"
    data_formats: List[Any] = field(default_factory=lambda: [
        "vision",
        "tactile"
    ])

    # Architecture Parameters
    encoder_depth: int = MISSING
    encoder_embed_dim: int = MISSING
    encoder_n_heads: int = MISSING

    decoder_depth: int = MISSING
    decoder_embed_dim: int = MISSING
    decoder_n_heads: int = MISSING

    use_cls_token: bool = True

    # MAE Loss/Objective Configuration
    norm_pixel_loss: bool = True
    effective_bsz: int = 32  
    device_bsz: int = MISSING
    native_bsz: int = MISSING

    # Optimization Parameters
    optimizer: str = "adamw"
    schedule: str = "linear-warmup+cosine-decay"
    base_lr: float = 1.5e-4
    min_lr: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.05

@dataclass
class TReTacConfig(TReTacBasicConfig):
    # identifier
    identifier = "t-retac"
    remarks = ""

    # Architecture Parameters -- should match ViT Small Architecture to the letter!
    #   Note: Small is defined in TIMM:
    #       > https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py#L683
    encoder_depth = 12
    encoder_embed_dim = 384
    encoder_n_heads = 6

    decoder_depth = 6
    decoder_embed_dim = 192
    decoder_n_heads = 6

    # Base Tactile Parameters
    tactile_dim: int = 20
    tactile_en_dim: int = 384
    tactile_type: str = 'normalize_global_0_3'
    tactile_emb_type: str = "flatten"

    # mask ratio
    mask_ratio: float = 0.75
    tactile_mask_ratio: float = 0.75

    # Number of examples one can safely fit on an accelerator w/ this model!
    #   > TPU-v3: max of 128 per device
    #   > GPU w/ 32G of RAM: max of 128 per device!
    device_bsz = 8  # 128 # effectie_bsz / nums_work
    native_bsz = 8  # 128 # effectie_bsz / nums_work

@dataclass
class VT20TReAll_TMR05_Bin_FT_Config(VTTReAllConfig):
    identifier = "vt20t-reall-tmr05-bin-ft"
    remarks = "mr075-fr-bce-norelu"

    use_cls_token = False
    mask_ratio = 0.75
    tactile_mask_ratio = 0.5

    tactile_set_to_ZERO = False
    tactile_type = "binary_0_3"
    tactile_emb_type = "separate"
    img_weight = 1
    tac_weight = 100

@dataclass
class T20ReTac_TMR05_Bin_FT_Config(TReTacConfig):
    identifier = "t20-retac-tmr05-bin-ft"
    remarks = "mse"

    use_cls_token = False
    tactile_mask_ratio = 0.5

    tactile_type = "binary_0_3"
    tactile_emb_type = "separate"

@dataclass
class VRePic_Bin_FT_Config(VRePicConfig):
    identifier = "v-repic-bin-ft"
    remarks = "mr075-fr-mse"

    use_cls_token = False
    mask_ratio = 0.75

    tactile_type = "binary_0_3"


cs = ConfigStore.instance()

cs.store(group="model", name="t-retac", node=TReTacConfig)
cs.store(group="model", name="v-repic", node=VRePicConfig)
cs.store(group="model", name="vtt-reall", node=VTTReAllConfig)

cs.store(group="model", name="t20-retac-tmr05-bin-ft", node=T20ReTac_TMR05_Bin_FT_Config)
cs.store(group="model", name="vt20t-reall-tmr05-bin-ft", node=VT20TReAll_TMR05_Bin_FT_Config)
cs.store(group="model", name="v-repic-bin-ft", node=VRePic_Bin_FT_Config)