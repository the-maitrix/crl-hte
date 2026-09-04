"""
Hash-keyed filesystem cache for trained encoders.

Why
---
Across slices, the same encoder is requested many times:
    - alpha sweep:  encoder is pretrained on H, H is invariant in alpha → reuse
    - delta sweep:  encoder is pretrained on H, H is invariant in delta → reuse
    - learner sweep: encoder is unchanged across (T, X, DML)
We avoid retraining whenever the encoder-relevant inputs match.

Cache key
---------
A deterministic SHA256 hash over the JSON-serialised dict of:
    (method, dgp_seed, encoder_seed, x_dim, z_dim, s_dim, z_s_fraction,
     phi_dim, arch, num_observational, encoder_historical_seed,
     lr, batch_size, epochs_obs,
     early_stopping_patience, early_stopping_min_delta,
     lambda_loss_s, vib_beta, infonce_temperature, proj_dim,
     sigma_S, sigma_Y, lib_version)

`lib_version` is bumped manually if the encoder code changes meaningfully.

Storage
-------
- Neural encoders: torch.save the state_dict + small metadata json side-by-side.
- Sklearn objects: joblib.dump the whole estimator.
- Atomic writes: write to .tmp then os.rename — safe against concurrent
  SLURM array tasks on a shared filesystem.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import joblib
import torch


CACHE_VERSION = "v4"  # key records the historical draw used to fit the encoder


def _hash_dict(d: Dict) -> str:
    payload = json.dumps(d, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:32]


def encoder_cache_key(method: str, cfg: Dict, dgp_seed: int, encoder_seed: int,
                      arch: Optional[str] = None, phi_dim_override: Optional[int] = None) -> str:
    arch = arch or cfg.get("arch", "medium")
    phi_dim = phi_dim_override if phi_dim_override is not None else cfg["phi_dim"]
    payload = {
        "method": method,
        "dgp_seed": dgp_seed,
        "encoder_seed": encoder_seed,
        "x_dim": cfg["x_dim"],
        "z_dim": cfg["z_dim"],
        "s_dim": cfg["s_dim"],
        "z_s_fraction": cfg.get("z_s_fraction", 0.5),
        "phi_dim": phi_dim,
        "arch": arch,
        "num_observational": cfg["num_observational"],
        "encoder_historical_seed": cfg.get("encoder_historical_seed"),
        "lr": cfg["lr"],
        "batch_size": cfg["batch_size"],
        "epochs_obs": cfg["epochs_obs"],
        "early_stopping_patience": cfg["early_stopping_patience"],
        "early_stopping_min_delta": cfg["early_stopping_min_delta"],
        "val_frac": float(cfg.get("val_frac", 0.0)),
        "select_best_epoch": bool(cfg.get("select_best_epoch", False)),
        "lambda_loss_s": cfg.get("lambda_loss_s"),
        "vib_beta": cfg.get("vib_beta"),
        "infonce_temperature": cfg.get("infonce_temperature"),
        "proj_dim": cfg.get("proj_dim"),
        "sigma_S": cfg["sigma_S"],
        "sigma_Y": cfg["sigma_Y"],
        "treatment_effect": cfg.get("treatment_effect"),
        "x_to_z_nonlinearity": cfg.get("x_to_z_nonlinearity", "linear"),
        "z_to_s_nonlinearity": cfg.get("z_to_s_nonlinearity", "linear"),
        "s_to_y_nonlinearity": cfg.get("s_to_y_nonlinearity", "linear"),
        "lib_version": CACHE_VERSION,
    }
    return _hash_dict(payload)


@dataclass
class CachePaths:
    base: Path

    def torch_path(self, key: str) -> Path:
        return self.base / f"{key}.pt"

    def sklearn_path(self, key: str) -> Path:
        return self.base / f"{key}.joblib"

    def meta_path(self, key: str) -> Path:
        return self.base / f"{key}.json"


def make_cache(base_dir: str | os.PathLike) -> CachePaths:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    return CachePaths(base=base)


def _atomic_write_bytes(path: Path, write_fn) -> None:
    # Disambiguate the .tmp by pid + SLURM_ARRAY_TASK_ID so concurrent array
    # tasks computing the same cache key don't corrupt each other's partial
    # writes. os.replace is atomic; the .tmp is per-writer.
    pid = os.getpid()
    task = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
    tmp = path.with_suffix(f"{path.suffix}.{task}.{pid}.tmp")
    write_fn(tmp)
    os.replace(tmp, path)


def save_torch(cache: CachePaths, key: str, state_dict: Dict, meta: Dict) -> None:
    _atomic_write_bytes(cache.torch_path(key), lambda p: torch.save(state_dict, p))
    _atomic_write_bytes(cache.meta_path(key),
                        lambda p: p.write_text(json.dumps(meta, indent=2)))


def load_torch(cache: CachePaths, key: str):
    p = cache.torch_path(key)
    if not p.exists():
        return None, None
    sd = torch.load(p, map_location="cpu")
    meta = json.loads(cache.meta_path(key).read_text()) if cache.meta_path(key).exists() else {}
    return sd, meta


def save_sklearn(cache: CachePaths, key: str, obj, meta: Dict) -> None:
    _atomic_write_bytes(cache.sklearn_path(key), lambda p: joblib.dump(obj, p))
    _atomic_write_bytes(cache.meta_path(key),
                        lambda p: p.write_text(json.dumps(meta, indent=2)))


def load_sklearn(cache: CachePaths, key: str):
    p = cache.sklearn_path(key)
    if not p.exists():
        return None, None
    obj = joblib.load(p)
    meta = json.loads(cache.meta_path(key).read_text()) if cache.meta_path(key).exists() else {}
    return obj, meta


def cached(cache: CachePaths, key: str) -> bool:
    return (cache.torch_path(key).exists() or cache.sklearn_path(key).exists())


# ── Diagnostics cache ────────────────────────────────────────────────────────
# Diagnostic outputs (sufficiency-i / sufficiency-ii) are expensive — minutes to
# tens-of-minutes per method on x_dim=1000 — and are deterministic given the
# encoder, the trial seed, and the (α, δ) used to generate the experimental
# pool. Cache them so a crash anywhere downstream doesn't waste that compute.

def diag_cache_key(encoder_key: str, kind: str, trial_seed: int,
                   alpha: Optional[float] = None,
                   delta: Optional[float] = None) -> str:
    """Hash key for a diagnostic output. `kind` is e.g. 'suff_i' or 'suff_ii'."""
    payload = {
        "encoder_key": encoder_key,
        "kind": kind,
        "trial_seed": int(trial_seed),
        "alpha": None if alpha is None else float(alpha),
        "delta": None if delta is None else float(delta),
        "lib_version": CACHE_VERSION,
    }
    return "diag_" + _hash_dict(payload)


def save_diag(cache: CachePaths, key: str, data: Dict) -> None:
    path = cache.base / f"{key}.json"
    _atomic_write_bytes(path, lambda p: p.write_text(json.dumps(data, indent=2)))


def load_diag(cache: CachePaths, key: str) -> Optional[Dict]:
    path = cache.base / f"{key}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())
