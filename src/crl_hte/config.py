"""
Config loading and CLI-override layer (OmegaConf-based, no Hydra).

Behaviour
---------
1. Load `default.yaml` (canonical center config).
2. If a slice config path is provided, merge it on top.
3. Apply any CLI overrides (dot notation: `encoder.phi_dim=10 sweep.alphas=[0,0.5]`).
4. Resolve, freeze, return as a plain dict.

Usage
-----
    from crl_hte.config import load_config
    cfg = load_config(
        default_yaml="configs/default.yaml",
        slice_yaml="configs/slices/alpha_sweep.yaml",
        overrides=["encoder.phi_dim=10", "n_trials=3"],
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from omegaconf import DictConfig, OmegaConf


def load_config(
    default_yaml: str,
    slice_yaml: Optional[str] = None,
    overrides: Optional[Iterable[str]] = None,
) -> DictConfig:
    cfg = OmegaConf.load(default_yaml)
    if slice_yaml:
        slice_cfg = OmegaConf.load(slice_yaml)
        cfg = OmegaConf.merge(cfg, slice_cfg)
    if overrides:
        cli_cfg = OmegaConf.from_dotlist(list(overrides))
        cfg = OmegaConf.merge(cfg, cli_cfg)
    OmegaConf.resolve(cfg)
    return cfg


def to_container(cfg: DictConfig) -> dict:
    return OmegaConf.to_container(cfg, resolve=True)


def to_yaml_str(cfg: DictConfig) -> str:
    return OmegaConf.to_yaml(cfg, resolve=True)


def find_default_config() -> Path:
    """Locate `configs/default.yaml` relative to package root."""
    here = Path(__file__).resolve()
    # post_rebuttals/src/crl_hte/config.py -> post_rebuttals/configs/default.yaml
    candidate = here.parent.parent.parent / "configs" / "default.yaml"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"default.yaml not found at expected path {candidate}")
