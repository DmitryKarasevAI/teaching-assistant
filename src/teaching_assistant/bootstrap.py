from __future__ import annotations

from hydra import initialize, compose
from omegaconf import OmegaConf

from .config_schema import Config


def load_cfg() -> Config:
    with initialize(config_path="../../configs", version_base=None):
        cfg = compose(config_name="config")

    base = OmegaConf.structured(Config())
    merged = OmegaConf.merge(base, cfg)
    return OmegaConf.to_object(merged)
