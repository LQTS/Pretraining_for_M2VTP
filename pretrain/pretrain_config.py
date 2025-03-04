
import os

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
print("root_dir: ", root_dir)

# Set Defaults (Hydra w/ Structured Configs)
DEFAULTS = [
    "_self_",
    {"model": "v-cond"},
    {"dataset": "sth-sth-v2"},
    {"accelerator": "torchrun"},
    {"tracking": "voltron-tracking"},
    {"override hydra/job_logging": "overwatch_rich"},
]


VT20T_ReAll_TMR05_Bin_FT_BottleCap = [
    "_self_",
    {"model": "vt20t-reall-tmr05-bin-ft"},
    {"dataset": "BottleCap"},
    {"accelerator": "torchone"},
    {"tracking": "vitac-track"},
    {"override hydra/job_logging": "overwatch_rich"},
]

T20_ReTac_TMR05_Bin_FT_BottleCap = [
    "_self_",
    {"model": "t20-retac-tmr05-bin-ft"},
    {"dataset": "BottleCap"},
    {"accelerator": "torchone"},
    {"tracking": "vitac-track"},
    {"override hydra/job_logging": "overwatch_rich"},
]


V_RePic_Bin_BottleCap = [
    "_self_",
    {"model": "v-repic-bin-ft"},
    {"dataset": "BottleCap"},
    {"accelerator": "torchone"},
    {"tracking": "vitac-track"},
    {"override hydra/job_logging": "overwatch_rich"},
]

class Pretrain_Config:
    model_dataset = VT20T_ReAll_TMR05_Bin_FT_BottleCap ###################


    hydra_ = {
        "run": {"dir": "./runs/binary/train/${model.identifier}+dataset-${dataset.name}"}
    }

    resume = False  ###################

    # load model(checkpoint) to finetone
    if_finetone = True  ###################
    finetone_model_path = root_dir + "data/model_cache/v-cond+vit-small+sth-sth+epoch-400.pt"###################
