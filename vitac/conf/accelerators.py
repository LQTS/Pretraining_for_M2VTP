"""
accelerator.py

Base Hydra Structured Configs for defining various accelerator schemes. Uses a simple single inheritance structure.
"""
import os
from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

# === Vanilla Accelerators (Deprecated; mostly for XLA code) ===


@dataclass
class AcceleratorConfig:
    accelerator: str = MISSING
    num_accelerators: int = MISSING
    num_workers: int = MISSING
    enable_distributed: bool = True


# === GPU Default Config --> just set `num_workers`; `torchrun` takes care of the rest! ===
#   > Note :: Defaults to 1 GPU if WORLD_SIZE not set (e.g., not running with `torchrun`)

@dataclass
class TorchOneConfig(AcceleratorConfig):
    accelerator = "gpu"
    num_accelerators = 1
    num_workers = 8
    enable_distributed: bool = False

@dataclass
class TorchMultiConfig(AcceleratorConfig):
    accelerator = "gpu"
    num_accelerators = int(os.environ["WORLD_SIZE"] if "WORLD_SIZE" in os.environ else 1)
    num_workers = 8
    enable_distributed: bool = True

# Create a configuration group `accelerator` and populate with the above...
cs = ConfigStore.instance()

cs.store(group="accelerator", name="torchone", node=TorchOneConfig)
cs.store(group="accelerator", name="torchmulti", node=TorchMultiConfig)