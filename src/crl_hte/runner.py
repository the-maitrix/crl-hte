"""
Slice runner — the engine for the post-rebuttal experiments.

A *slice* is a set of axis specifications. The runner enumerates the Cartesian
product, trains each unique encoder once (cached), evaluates at every
(sample_size × learner × y_type × alpha × delta) cell, and emits a long-format
parquet file.

Encoder reuse rules
-------------------
The encoder is pretrained on the historical sample H, where T=0 always. So:
    - alpha (sufficiency-(ii)) lives in E only → encoder INVARIANT in alpha.
    - delta (surrogacy) lives in E only       → encoder INVARIANT in delta.
The runner therefore loops alpha/delta as the INNER axis, reusing each encoder.

The encoder DOES depend on:
    method, phi_dim, arch, z_dim, z_s_fraction, x_dim, s_dim, encoder_seed,
    dgp_seed, num_observational, training hyperparameters.

Output schema (parquet)
-----------------------
    slice_name (str)        — name of this slice
    method (str)
    phi_dim (int)
    arch (str)              — 'small'/'medium'/'large'
    z_dim (int)
    z_s_fraction (float)
    x_dim (int)
    s_dim (int)
    num_observational (int) — historical representation-training sample size
    alpha (float)
    delta (float)
    n (int)                 — experimental sample size
    learner (str)           — 't_learner'/'x_learner'/'dml_learner'
    y_type (str)            — 'true_y' or 'h_y'
    finetune (str)          — 'frozen' / 'ftS' / 'ftY' / 'ftSY'
    trial_seed (int)
    encoder_key (str)       — cache hash, useful for joins
    epochs_run (int)
    mse, pehe, pehe_norm, spearman_rho, pearson_r,
    abs_regret, norm_regret, auuc,
    topk_10_value, topk_10_norm, topk_20_value, topk_20_norm,
    topk_50_value, topk_50_norm
    # diagnostics (NaN unless run)
    diag_r2_phi_s, diag_r2_phi_s_x, diag_r2_gap,
    diag_r2_xs_resid_mean, diag_r2_xs_resid_max
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

from . import dgp as dgp_mod
from . import encoders as enc_mod
from . import cate_learners as cate_mod
from . import metrics as metrics_mod
from . import diagnostics as diag_mod
from . import cache as cache_mod


# ── Result row construction ──────────────────────────────────────────────────

_NA = float("nan")
_DIAG_KEYS = (
    "diag_r2_phi_s", "diag_r2_phi_s_x", "diag_r2_gap",
    "diag_r2_xs_resid_mean", "diag_r2_xs_resid_max",
)


def _empty_row(slice_name: str, method: str, phi_dim: int, arch: str,
               z_dim: int, z_s_fraction: float, x_dim: int, s_dim: int,
               alpha: float, delta: float, n: int, learner: str, y_type: str,
               finetune: str, trial_seed: int, dgp_seed: int,
               encoder_key: str, epochs_run: int,
               cate_regressor: str = "histgbr",
               lambda_loss_s: float = _NA,
               num_observational: int = 0) -> Dict:
    row = {
        "slice_name": slice_name, "method": method,
        "phi_dim": phi_dim, "arch": arch,
        "z_dim": z_dim, "z_s_fraction": z_s_fraction,
        "x_dim": x_dim, "s_dim": s_dim,
        "alpha": alpha, "delta": delta, "n": n,
        "learner": learner, "y_type": y_type, "finetune": finetune,
        "trial_seed": trial_seed, "dgp_seed": dgp_seed,
        "encoder_key": encoder_key, "epochs_run": epochs_run,
        "lambda_loss_s": lambda_loss_s,
        "num_observational": num_observational,
        "cate_regressor": cate_regressor,
        "error": None,    # populated only on the failure path
    }
    for k in _DIAG_KEYS:
        row[k] = _NA
    return row


# ── Slice config helpers ─────────────────────────────────────────────────────

@dataclass
class SliceSpec:
    """Resolved slice config — used to enumerate trial × axis combos."""
    name: str
    n_trials: int
    sample_sizes: List[int]
    methods: List[str]
    learners: List[str]
    y_types: List[str]
    alphas: List[float]
    deltas: List[float]
    z_dims: List[int]
    z_s_fractions: List[float]
    historical_sizes: List[int]
    phi_dims: List[int]
    archs: List[str]
    lambda_loss_s: List[float]
    finetune: bool
    finetune_methods: List[str]
    finetune_modes: List[str]   # subset of {ftS, ftY, ftSY}
    finetune_t1_only: bool      # restrict finetune data to T=1 rows (ftS-style protocol)
    finetune_split_frac: float  # fraction of E held out for finetune (rest is eval)
    finetune_xfit_folds: int    # K for K-fold cross-fitted finetune; 1 = single-fold (no xfit)
    diagnostics: bool


def slice_from_cfg(cfg) -> SliceSpec:
    """Resolve a (default + slice + overrides) cfg into a SliceSpec.

    Expected keys (with defaults):
        slice.name, slice.n_trials, sweep.*
    """
    sweep = cfg.sweep
    return SliceSpec(
        name=cfg.slice.name,
        n_trials=int(cfg.slice.n_trials),
        sample_sizes=list(sweep.sample_sizes),
        methods=list(sweep.methods),
        learners=list(sweep.learners),
        y_types=list(sweep.y_types),
        alphas=list(sweep.alpha),
        deltas=list(sweep.delta),
        z_dims=list(sweep.z_dim),
        z_s_fractions=list(sweep.z_s_fraction),
        historical_sizes=list(sweep.get(
            "num_observational", [int(cfg.dgp.num_observational)])),
        phi_dims=list(sweep.phi_dim),
        archs=list(sweep.arch),
        lambda_loss_s=list(sweep.get(
            "lambda_loss_s", [float(cfg.encoder.lambda_loss_s)])),
        finetune=bool(sweep.get("finetune", False)),
        finetune_methods=list(sweep.get("finetune_methods", [])),
        finetune_modes=list(sweep.get("finetune_modes", [])),
        finetune_t1_only=bool(sweep.get("finetune_t1_only", False)),
        finetune_split_frac=float(sweep.get("finetune_split_frac", 0.0)),
        finetune_xfit_folds=int(sweep.get("finetune_xfit_folds", 1)),
        diagnostics=bool(sweep.get("diagnostics", True)),
    )


def _flatten_dgp_cfg(cfg, z_dim: int, z_s_fraction: float,
                     num_observational: int, phi_dim: int,
                     arch: str, lambda_loss_s: float) -> Dict:
    """Flatten the OmegaConf dgp/encoder sections into the dict that older
    modules expect (one nested level)."""
    base = {
        "x_dim": int(cfg.dgp.x_dim),
        "z_dim": int(z_dim),
        "s_dim": int(cfg.dgp.s_dim),
        "y_dim": int(cfg.dgp.get("y_dim", 1)),
        "z_s_fraction": float(z_s_fraction),
        "num_observational": int(num_observational),
        "num_experimental": int(cfg.dgp.num_experimental),
        "sigma_S": float(cfg.dgp.sigma_S),
        "sigma_Y": float(cfg.dgp.sigma_Y),
        "treatment_effect": str(cfg.dgp.get("treatment_effect", "linear")),
        "x_to_z_nonlinearity": str(cfg.dgp.get("x_to_z_nonlinearity", "linear")),
        "z_to_s_nonlinearity": str(cfg.dgp.get("z_to_s_nonlinearity", "linear")),
        "s_to_y_nonlinearity": str(cfg.dgp.get("s_to_y_nonlinearity", "linear")),
        "phi_dim": int(phi_dim),
        "arch": str(arch),
        "batch_size": int(cfg.encoder.batch_size),
        "lr": float(cfg.encoder.lr),
        "epochs_obs": int(cfg.encoder.epochs_obs),
        "early_stopping_patience": int(cfg.encoder.early_stopping_patience),
        "early_stopping_min_delta": float(cfg.encoder.early_stopping_min_delta),
        "val_frac": float(cfg.encoder.get("val_frac", 0.0)),
        "select_best_epoch": bool(cfg.encoder.get("select_best_epoch", False)),
        "lambda_loss_s": float(lambda_loss_s),
        "vib_beta": float(cfg.encoder.vib_beta),
        "infonce_temperature": float(cfg.encoder.infonce_temperature),
        "proj_dim": int(cfg.encoder.proj_dim),
        "lambda_finetune_y": float(cfg.encoder.get("lambda_finetune_y", 1.0)),
        "finetune_lr": float(cfg.encoder.get("finetune_lr", 1e-3)),
        "finetune_epochs": int(cfg.encoder.get("finetune_epochs", 100)),
        "finetune_batch_size": int(cfg.encoder.get("finetune_batch_size", 64)),
        "finetune_freeze_backbone": bool(cfg.encoder.get("finetune_freeze_backbone", False)),
    }
    return base


# ── Encoder fetch with cache ─────────────────────────────────────────────────

def _instantiate_module(method: str, cfg: Dict) -> torch.nn.Module:
    """Re-instantiate the torch module skeleton (no weights). Required to load state_dict."""
    arch = cfg["arch"]
    if method == "mi_vib":
        return enc_mod.StochasticEncoder(cfg["x_dim"], cfg["phi_dim"], arch=arch)
    if method == "autoencoder":
        return enc_mod._AutoencoderModule(cfg["x_dim"], cfg["phi_dim"], arch=arch)
    if method in {"encoder_pred", "encoder_no_s", "encoder_no_y",
                  "mi_mine", "mi_infonce",
                  "mi_mine_cond", "mi_infonce_cond"}:
        return enc_mod.Encoder(cfg["x_dim"], cfg["phi_dim"], arch=arch)
    raise ValueError(f"_instantiate_module not implemented for: {method}")


def _torch_to_fitted(method: str, mod: torch.nn.Module, meta: Dict) -> enc_mod.FittedEncoder:
    if method == "mi_vib":
        return enc_mod.FittedEncoder(
            name=method, encode=enc_mod._neural_encode_fn(mod, "stochastic"),
            is_neural=True, epochs_run=int(meta.get("epochs_run", 0)),
            torch_module=mod,
        )
    if method == "autoencoder":
        mod.eval()
        enc_net = mod.encoder_net
        def encode(X):
            with torch.no_grad():
                return enc_net(torch.tensor(X, dtype=torch.float32)).numpy()
        return enc_mod.FittedEncoder(
            name=method, encode=encode, is_neural=True,
            epochs_run=int(meta.get("epochs_run", 0)),
            torch_module=mod,
        )
    return enc_mod.FittedEncoder(
        name=method, encode=enc_mod._neural_encode_fn(mod),
        is_neural=True, epochs_run=int(meta.get("epochs_run", 0)),
        torch_module=mod,
    )


def get_or_train_encoder(
    method: str, obs: Dict, cfg: Dict, device: torch.device,
    encoder_seed: int, dgp_seed: int,
    cache: cache_mod.CachePaths,
) -> Tuple[enc_mod.FittedEncoder, str]:
    """Returns (FittedEncoder, cache_key). Reuses cached weights if available."""
    key = cache_mod.encoder_cache_key(
        method, cfg, dgp_seed=dgp_seed, encoder_seed=encoder_seed,
        arch=cfg["arch"], phi_dim_override=cfg["phi_dim"],
    )

    # Sklearn-based methods: fit (cheap) but still cache for consistency
    if method in {"pca", "pls", "ica"}:
        obj, _ = cache_mod.load_sklearn(cache, key)
        if obj is None:
            fitted = enc_mod.fit_method(method, obs, cfg, device, encoder_seed)
            cache_mod.save_sklearn(cache, key, fitted.state["sklearn"],
                                   meta={"method": method, "key": key})
            return fitted, key
        # Wrap loaded sklearn back into FittedEncoder
        fitted = enc_mod.FittedEncoder(
            name=method,
            encode=lambda X, _o=obj: _o.transform(X).astype(np.float32),
            state={"sklearn": obj},
        )
        return fitted, key

    if method in {"raw_x", "baseline_xs", "surrogate_index", "constant_ate"}:
        return enc_mod.fit_method(method, obs, cfg, device, encoder_seed), key

    # Neural methods
    if method in enc_mod.NEURAL_METHODS:
        sd, meta = cache_mod.load_torch(cache, key)
        if sd is not None:
            mod = _instantiate_module(method, cfg)
            mod.load_state_dict(sd)
            return _torch_to_fitted(method, mod, meta), key

        fitted = enc_mod.fit_method(method, obs, cfg, device, encoder_seed)
        # Save state_dict + small meta
        if fitted.torch_module is not None:
            cache_mod.save_torch(
                cache, key,
                fitted.torch_module.state_dict(),
                meta={
                    "method": method, "key": key,
                    "epochs_run": int(fitted.epochs_run),
                    "phi_dim": cfg["phi_dim"], "arch": cfg["arch"],
                    "x_dim": cfg["x_dim"], "z_dim": cfg["z_dim"],
                    "encoder_seed": encoder_seed, "dgp_seed": dgp_seed,
                },
            )
        return fitted, key

    raise KeyError(f"Unknown method: {method}")


# ── Per-trial evaluation ─────────────────────────────────────────────────────

def _build_features(method: str, X: np.ndarray, S: np.ndarray,
                    fitted: enc_mod.FittedEncoder) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (cate_features, h_features). For most methods, cate_features=phi
    and h_features=[phi, S]. raw_x and baseline_xs are special.
    """
    if method == "raw_x":
        # No S concatenation
        return X.astype(np.float32), X.astype(np.float32)
    if method in {"surrogate_index", "constant_ate"}:
        return X.astype(np.float32), np.c_[X, S].astype(np.float32)
    if method == "baseline_xs":
        feats = np.c_[X, S].astype(np.float32)
        return feats, feats
    phi = fitted.encode(X)
    h_feats = np.c_[phi, S].astype(np.float32)
    return phi.astype(np.float32), h_feats


def _train_h_model(method: str, fitted: enc_mod.FittedEncoder,
                   obs: Dict, seed: int) -> RandomForestRegressor:
    """Train h(phi, S) -> Y on observational data (for h-Y CATE outcome)."""
    if method == "raw_x":
        h_feats = obs["X"]
    elif method in {"baseline_xs", "surrogate_index", "constant_ate"}:
        h_feats = np.c_[obs["X"], obs["S"]]
    else:
        phi_obs = fitted.encode(obs["X"])
        h_feats = np.c_[phi_obs, obs["S"]]
    return RandomForestRegressor(
        n_estimators=50, random_state=seed, n_jobs=-1
    ).fit(h_feats, obs["Y"].flatten())


def _row_metrics(row: Dict, tau_hat: np.ndarray, tau_true: np.ndarray) -> Dict:
    metrics = metrics_mod.compute_all(tau_hat, tau_true)
    row.update(metrics)
    return row


def _method_y_types(method: str, configured: List[str]) -> List[str]:
    """Return the outcome sources a method actually evaluates."""
    if method == "surrogate_index":
        return ["h_y"]
    if method == "constant_ate":
        return ["true_y"]
    return configured


AXIS_INVARIANT_METHODS = {"raw_x", "surrogate_index"}


def _axis_invariant_fit_key(*, method: str, dgp_seed: int, trial_seed: int,
                            z_dim: int, z_s_fraction: float, alpha: float,
                            delta: float, num_observational: Optional[int],
                            n: int, learner: str, y_type: str,
                            regressor: str) -> Tuple:
    """Key for baselines that do not depend on representation sweep axes."""
    if method not in AXIS_INVARIANT_METHODS:
        raise ValueError(f"{method!r} is not axis-invariant")
    return (
        method, dgp_seed, trial_seed, z_dim, float(z_s_fraction), float(alpha),
        float(delta), num_observational, n, learner, y_type, regressor,
    )


def _historical_draw_seed(mode: str, *, dgp_seed: int,
                          trial_seed: int) -> int:
    """Choose the RNG owner for the historical sample.

    ``dgp`` is the clean design used by new sweeps: one H pool per DGP draw.
    ``trial`` reproduces the camera-ready main table, whose h-model was fit on
    a fresh H sample for each experimental trial (while the cached encoder was
    shared across trials).
    """
    if mode == "dgp":
        return int(dgp_seed)
    if mode == "trial":
        return int(trial_seed)
    raise ValueError(
        f"Unknown slice.historical_seed_mode={mode!r}; expected 'dgp' or 'trial'."
    )


# ── Main runner ──────────────────────────────────────────────────────────────

def run_slice(cfg, output_dir: Path, cache_dir: Path,
              device: Optional[torch.device] = None,
              progress: bool = True,
              pretrain_only: bool = False) -> pd.DataFrame:
    """Run one slice end-to-end. Writes parquet to `output_dir/results.parquet`.

    If `pretrain_only`, train and cache encoders for every structural cell
    in the slice and return an empty DataFrame — no CATE eval, no parquet.
    Used to populate the cache from a GPU job before a CPU eval array.

    Returns the DataFrame.
    """
    spec = slice_from_cfg(cfg)
    cache = cache_mod.make_cache(cache_dir)
    device = device or enc_mod.get_device(cfg.runtime.get("device"))

    cate_cfg = cfg.get("cate", {}) if hasattr(cfg, "get") else {}
    cate_mode = str(cate_cfg.get("regressor", "histgbr"))
    cate_n_iterations = int(cate_cfg.get("n_iterations", 40))
    cate_threshold_n = int(cate_cfg.get("threshold_n", 1000))
    cate_threshold_below = str(cate_cfg.get("threshold_below", "gbr"))
    cate_threshold_above = str(cate_cfg.get("threshold_at_or_above", "histgbr"))

    def _resolve_regressor(ss: int) -> str:
        if cate_mode in ("gbr", "histgbr"):
            return cate_mode
        if cate_mode == "threshold":
            return cate_threshold_below if ss < cate_threshold_n else cate_threshold_above
        raise ValueError(f"Unknown cate.regressor mode: {cate_mode!r}")

    print(f"[runner] device = {device}")
    print(f"[runner] slice = {spec.name}")
    print(f"[runner] n_trials = {spec.n_trials}, sample_sizes = {spec.sample_sizes}")
    print(f"[runner] alphas = {spec.alphas}, deltas = {spec.deltas}")
    print(f"[runner] z_dims = {spec.z_dims}, z_s_fractions = {spec.z_s_fractions}")
    print(f"[runner] historical_sizes = {spec.historical_sizes}")
    print(f"[runner] phi_dims = {spec.phi_dims}, archs = {spec.archs}")
    print(f"[runner] lambda_loss_s = {spec.lambda_loss_s}")
    print(f"[runner] methods ({len(spec.methods)}) = {spec.methods}")
    if cate_mode == "threshold":
        print(f"[runner] cate regressor = threshold "
              f"(n<{cate_threshold_n}: {cate_threshold_below}, "
              f"n>={cate_threshold_n}: {cate_threshold_above})  "
              f"n_iterations = {cate_n_iterations}")
    else:
        print(f"[runner] cate regressor = {cate_mode}  n_iterations = {cate_n_iterations}")

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []
    dgp_summary_rows: List[Dict] = []
    historical_pool_cache: Dict[Tuple, Dict[str, np.ndarray]] = {}
    axis_invariant_h_cache: Dict[Tuple, RandomForestRegressor] = {}
    axis_invariant_tau_cache: Dict[Tuple, np.ndarray] = {}
    save_tau_arrays = bool(cfg.slice.get("save_tau_arrays", False))
    historical_seed_mode = str(cfg.slice.get("historical_seed_mode", "dgp"))
    if historical_seed_mode not in {"dgp", "trial"}:
        raise ValueError(
            "slice.historical_seed_mode must be either 'dgp' or 'trial'"
        )
    print(f"[runner] historical_seed_mode = {historical_seed_mode}")
    configured_encoder_h_seed = cfg.slice.get("encoder_historical_seed", None)
    if configured_encoder_h_seed is not None:
        configured_encoder_h_seed = int(configured_encoder_h_seed)
    print(f"[runner] encoder_historical_seed = "
          f"{configured_encoder_h_seed if configured_encoder_h_seed is not None else 'same as H'}")

    # ── Seed plan ─────────────────────────────────────────────────────────────
    # `slice.dgp_seeds` (list[int])   outer DGP-randomness axis. Default [42],
    #   matches the rebuttal scripts' FIXED_SEED=42. Use [42,43,44,45,46] for a
    #   5×5 study (5 DGP draws × n_trials data samples = 5×n_trials rows).
    # `slice.n_trials` (int)          inner data-sample trials per DGP seed.
    # `slice.shared_encoder_seed` (int|null)  if set, use this constant torch seed
    #   for every encoder init (matches FIXED_SEED=42 in originals); if null,
    #   derive a deterministic per-trial encoder seed.
    raw_dgp_seeds = cfg.slice.get("dgp_seeds", None)
    if raw_dgp_seeds is None:
        # Backward-compat with old `shared_dgp_seed`. None ⇒ default to [42].
        legacy = cfg.slice.get("shared_dgp_seed", 42)
        raw_dgp_seeds = [int(legacy)] if legacy is not None else [None]
    dgp_seeds = [int(s) if s is not None else None for s in list(raw_dgp_seeds)]

    shared_encoder_seed = cfg.slice.get("shared_encoder_seed", None)
    if shared_encoder_seed is not None:
        shared_encoder_seed = int(shared_encoder_seed)

    # The OUTER loop now also iterates over dgp_seeds. Within a dgp_seed, the
    # structural axes (z_dim, z_s_fraction, phi_dim, arch) nest, then data trials.
    outer_iter = list(product(
        dgp_seeds,                    # NEW: DGP-draw axis
        range(spec.n_trials),         # data-sample trial
        spec.z_dims,
        spec.z_s_fractions,
        spec.historical_sizes,
        spec.phi_dims,
        spec.archs,
        spec.lambda_loss_s,
    ))
    print(f"[runner] dgp_seeds = {dgp_seeds}  (× n_trials={spec.n_trials} data samples each)")

    # Track which dgp_seeds we've already dumped params for (avoid re-saving).
    dumped_dgp_seeds = set()

    n_outer = len(outer_iter)
    t_outer_start = time.time()
    for outer_idx, (
            dgp_seed_cfg, trial_idx, z_dim, z_s_fraction, num_observational,
            phi_dim, arch, lambda_loss_s,
    ) in enumerate(
            tqdm(outer_iter, desc=f"slice={spec.name}", disable=not progress)):
        cell_t0 = time.time()
        print(f"[runner] cell {outer_idx + 1}/{n_outer}  trial={trial_idx} "
              f"z={z_dim} zsf={z_s_fraction} N_H={num_observational} "
              f"φ={phi_dim} {arch} "
              f"λ={lambda_loss_s}", flush=True)
        trial_seed = int(cfg.slice.base_seed) + trial_idx
        # If dgp_seed_cfg is None → re-roll per trial; otherwise fixed across trials
        dgp_seed = (trial_seed if dgp_seed_cfg is None else dgp_seed_cfg)
        encoder_seed = (shared_encoder_seed if shared_encoder_seed is not None
                        else trial_seed + 7919)

        # Flatten cfg for legacy modules
        flat = _flatten_dgp_cfg(cfg, z_dim=z_dim, z_s_fraction=z_s_fraction,
                                num_observational=num_observational,
                                phi_dim=phi_dim, arch=arch,
                                lambda_loss_s=lambda_loss_s)

        # ── Build DGP and H/E pools ──────────────────────────────────────────
        params = dgp_mod.init_dgp_params(flat, dgp_seed=dgp_seed)

        # Dump DGP params once per (dgp_seed, z_dim, z_s_fraction) for inspection.
        dump_key = (dgp_seed, z_dim, float(z_s_fraction))
        if dump_key not in dumped_dgp_seeds:
            np.savez(
                output_dir / f"dgp_params_seed{dgp_seed}_z{z_dim}_zsf{z_s_fraction}.npz",
                W_xz=params.W_xz,
                beta_0=params.beta_0, beta_1=params.beta_1,
                gamma_0=params.gamma_0, gamma_z=params.gamma_z,
                gamma_y_norm=params.gamma_y_norm,
                w_sy=params.w_sy, w_bypass=params.w_bypass,
                z_indices_for_s=params.z_indices_for_s,
                z_indices_for_y=params.z_indices_for_y,
                rel_s=params.rel_s, rel_y=params.rel_y,
            )
            dumped_dgp_seeds.add(dump_key)

        # New sweeps fix H within a DGP seed. The paper run opts into the
        # camera-ready behavior, where each trial refits h on a fresh H draw;
        # the encoder remains shared through its legacy cache key.
        max_historical = max(spec.historical_sizes)
        historical_seed = _historical_draw_seed(
            historical_seed_mode,
            dgp_seed=dgp_seed,
            trial_seed=trial_seed,
        )
        obs_pool_key = (
            historical_seed, z_dim, float(z_s_fraction), max_historical,
        )
        if obs_pool_key not in historical_pool_cache:
            rng_obs = np.random.default_rng(historical_seed * 31 + 1)
            historical_pool_cache[obs_pool_key] = dgp_mod.generate_observational(
                max_historical, params, flat, rng_obs
            )
        obs_pool = historical_pool_cache[obs_pool_key]
        obs = {name: values[:num_observational]
               for name, values in obs_pool.items()}

        # The camera-ready array pretrained one shared encoder on trial 1000's
        # H draw, then refit only h on each trial's H. Make that provenance
        # explicit so a missing cache cannot silently train on the wrong H.
        encoder_historical_seed = (
            historical_seed if configured_encoder_h_seed is None
            else configured_encoder_h_seed
        )
        encoder_pool_key = (
            encoder_historical_seed, z_dim, float(z_s_fraction), max_historical,
        )
        if encoder_pool_key not in historical_pool_cache:
            rng_encoder_obs = np.random.default_rng(
                encoder_historical_seed * 31 + 1
            )
            historical_pool_cache[encoder_pool_key] = (
                dgp_mod.generate_observational(
                    max_historical, params, flat, rng_encoder_obs
                )
            )
        encoder_pool = historical_pool_cache[encoder_pool_key]
        encoder_obs = {
            name: values[:num_observational]
            for name, values in encoder_pool.items()
        }
        flat["encoder_historical_seed"] = encoder_historical_seed

        # Pre-allocate experimental pool — outcomes regenerated per (alpha, delta).
        # The X→Z nonlinearity is applied here so Z_pool reflects the chosen DGP.
        rng_exp_X = np.random.default_rng(trial_seed * 31 + 2)
        N_max = flat["num_experimental"]
        X_pool = rng_exp_X.normal(0, 1, (N_max, flat["x_dim"])).astype(np.float32)
        Z_pool = dgp_mod._latent_from_x(X_pool, params, flat)
        T_pool_rng = np.random.default_rng(trial_seed * 31 + 3)
        T_pool = T_pool_rng.binomial(1, 0.5, (N_max, 1)).astype(np.float32)
        # nested subsamples (large -> small)
        perm = np.random.default_rng(trial_seed * 31 + 4).permutation(N_max)
        idxs = {ss: perm[:ss] for ss in sorted(spec.sample_sizes, reverse=True)}

        # ── Train each requested method (cached) ─────────────────────────────
        fitted_by_method: Dict[str, Tuple[enc_mod.FittedEncoder, str]] = {}
        for method in spec.methods:
            t0 = time.time()
            fitted, key = get_or_train_encoder(
                method=method, obs=encoder_obs, cfg=flat, device=device,
                encoder_seed=encoder_seed, dgp_seed=dgp_seed, cache=cache,
            )
            fitted_by_method[method] = (fitted, key)
            elapsed = time.time() - t0
            print(f"  [trial={trial_idx} z={z_dim} φ={phi_dim} {arch}]"
                  f" {method:14s} key={key[:10]}.. ({elapsed:.1f}s, ep={fitted.epochs_run})")

        if pretrain_only:
            # Cache is populated for this structural cell; skip eval.
            continue

        # ── h-models for h-predicted Y on observational data ─────────────────
        # Diagnostics (suff-i / suff-ii) are deferred until AFTER the CATE
        # inner loop completes, so a kill during diagnostics doesn't lose
        # CATE results. The rows below are emitted with diag_* = NaN; a
        # post-fill pass at the end of the cell updates them in place.
        cell_row_start = len(rows)
        exp_pools_by_ad: Dict[Tuple[float, float], Dict] = {}
        t_h = time.time()
        print(f"  [cell {outer_idx + 1}] training required h-models ...", flush=True)
        h_models: Dict[str, Optional[RandomForestRegressor]] = {}
        for method in spec.methods:
            if "h_y" not in _method_y_types(method, spec.y_types):
                h_models[method] = None
                continue
            t_m = time.time()
            if method in AXIS_INVARIANT_METHODS:
                h_key = (
                    method, dgp_seed, trial_seed, z_dim, float(z_s_fraction),
                    num_observational,
                )
                if h_key not in axis_invariant_h_cache:
                    axis_invariant_h_cache[h_key] = _train_h_model(
                        method, fitted_by_method[method][0], obs,
                        seed=trial_seed,
                    )
                h_models[method] = axis_invariant_h_cache[h_key]
            else:
                h_models[method] = _train_h_model(
                    method, fitted_by_method[method][0], obs, seed=trial_seed
                )
            print(f"    [cell {outer_idx + 1}] h-model {method:14s} ({time.time() - t_m:.1f}s)", flush=True)
        print(f"  [cell {outer_idx + 1}] h-models done ({time.time() - t_h:.1f}s)", flush=True)

        # ── Inner loop: alpha, delta, n, learner, y_type ─────────────────────
        for alpha, delta in product(spec.alphas, spec.deltas):
            # Reuse X, T, Z pools across α/δ values so features are comparable.
            rng_noise = np.random.default_rng(
                trial_seed * 1009 + int(round(alpha * 100)) * 31 + int(round(delta * 100))
            )

            # Compose exp_full directly (mirroring generate_experimental's body but
            # reusing our X_pool/Z_pool/T_pool). Z→S and S→Y nonlinearities are
            # applied to match dgp.generate_experimental.
            a = float(alpha); d = float(delta)
            tau_S = dgp_mod._surrogate_effect(Z_pool, params, flat, a)
            S_0_full = dgp_mod._surrogate_baseline(Z_pool, params, flat)
            S_1_full = S_0_full + tau_S
            S_full = (S_0_full + tau_S * T_pool
                      + rng_noise.normal(0, flat["sigma_S"], S_0_full.shape)).astype(np.float32)
            bypass_full = (Z_pool @ params.w_bypass).reshape(-1, 1)
            Y_full = (dgp_mod._outcome_from_s(S_full, params, flat)
                      + d * T_pool * bypass_full
                      + rng_noise.normal(0, flat["sigma_Y"], (N_max, 1))).astype(np.float32)
            Y_0_full = dgp_mod._outcome_from_s(S_0_full, params, flat)
            Y_1_full = (
                dgp_mod._outcome_from_s(S_1_full, params, flat)
                + d * bypass_full
            )
            tau_true_full = (Y_1_full - Y_0_full).flatten().astype(np.float32)

            # ── DGP summary row (one per dgp_seed × z_dim × z_s_fraction × α × δ) ──
            # Snapshot of the true-CATE distribution. Useful to diagnose "is the
            # CATE too small to detect", which inflates PEHE for every method.
            tau_quantiles = np.quantile(tau_true_full, [0.05, 0.25, 0.5, 0.75, 0.95])
            dgp_summary_rows.append(dict(
                slice_name=spec.name, dgp_seed=dgp_seed,
                trial_seed=trial_seed,
                num_observational=num_observational,
                z_dim=z_dim, z_s_fraction=float(z_s_fraction),
                alpha=float(alpha), delta=float(delta),
                tau_true_mean=float(np.mean(tau_true_full)),
                tau_true_std=float(np.std(tau_true_full)),
                tau_true_var=float(np.var(tau_true_full)),
                tau_true_p05=float(tau_quantiles[0]),
                tau_true_p25=float(tau_quantiles[1]),
                tau_true_p50=float(tau_quantiles[2]),
                tau_true_p75=float(tau_quantiles[3]),
                tau_true_p95=float(tau_quantiles[4]),
                tau_true_min=float(np.min(tau_true_full)),
                tau_true_max=float(np.max(tau_true_full)),
                snr_vs_sigmaY=float(np.var(tau_true_full) / (flat["sigma_Y"] ** 2)),
            ))
            if save_tau_arrays:
                np.save(
                    output_dir / f"tau_seed{dgp_seed}_z{z_dim}_a{alpha}_d{delta}.npy",
                    tau_true_full,
                )

            # Cache the experimental pool — needed later by suff-ii diagnostic
            # (which runs AFTER all CATE for this cell completes).
            exp_pools_by_ad[(float(alpha), float(delta))] = dict(
                X_pool=X_pool, T_pool=T_pool, S_full=S_full,
            )

            # ── Per sample size ──────────────────────────────────────────────
            for ss in spec.sample_sizes:
                # Threshold mode picks gbr/histgbr based on ss; pure modes are
                # constant. Recorded per-row in `cate_regressor` for clarity.
                cate_regressor = _resolve_regressor(ss)
                idx = idxs[ss]
                # Fold split for finetune slices: hold out a disjoint subset for
                # finetune so the encoder doesn't peek at CATE eval rows. Both
                # frozen and finetuned variants then evaluate on fold B, making
                # the "no-finetune vs finetune" comparison apples-to-apples.
                # When finetune_split_frac == 0 (default), fold A is empty and
                # both frozen + finetuned use full E (with leakage; original
                # behavior).
                if spec.finetune and spec.finetune_split_frac > 0:
                    n_a = max(1, int(ss * spec.finetune_split_frac))
                    idx_finetune = idx[:n_a]
                    idx_eval = idx[n_a:]
                else:
                    idx_finetune = idx          # finetune sees same rows as CATE (leaky)
                    idx_eval = idx
                ss_eval = len(idx_eval)         # actual rows the CATE estimator sees
                X_e = X_pool[idx_eval]; T_e = T_pool[idx_eval].flatten()
                S_e = S_full[idx_eval]; Y_e = Y_full[idx_eval].flatten()
                tau_true = tau_true_full[idx_eval]

                for method in spec.methods:
                    t_m = time.time()
                    fitted, key = fitted_by_method[method]
                    cate_feats, h_feats = _build_features(method, X_e, S_e, fitted)
                    # h-predicted Y (always trained on obs → no leakage from exp)
                    h_model = h_models[method]
                    Y_h = (
                        h_model.predict(h_feats).astype(np.float32)
                        if h_model is not None else None
                    )

                    learners = (
                        ["constant_ate"] if method == "constant_ate"
                        else spec.learners
                    )
                    y_types = _method_y_types(method, spec.y_types)
                    for learner in learners:
                        for y_type in y_types:
                            outcome = Y_e if y_type == "true_y" else Y_h
                            row_regressor = cate_regressor
                            try:
                                if method == "constant_ate":
                                    # Oracle constant-effect reference. Its
                                    # normalized PEHE is exactly 1 by definition.
                                    tau_hat = np.full(len(outcome), tau_true.mean())
                                elif method in AXIS_INVARIANT_METHODS:
                                    invariant_key = _axis_invariant_fit_key(
                                        method=method,
                                        dgp_seed=dgp_seed,
                                        trial_seed=trial_seed,
                                        z_dim=z_dim,
                                        z_s_fraction=z_s_fraction,
                                        alpha=alpha,
                                        delta=delta,
                                        num_observational=(
                                            num_observational
                                            if y_type == "h_y" else None
                                        ),
                                        n=ss_eval,
                                        learner=learner,
                                        y_type=y_type,
                                        regressor=cate_regressor,
                                    )
                                    if invariant_key not in axis_invariant_tau_cache:
                                        axis_invariant_tau_cache[invariant_key] = cate_mod.fit_cate(
                                            learner, cate_feats, T_e, outcome,
                                            seed=trial_seed,
                                            regressor=cate_regressor,
                                            n_iterations=cate_n_iterations,
                                        )
                                    tau_hat = axis_invariant_tau_cache[invariant_key]
                                else:
                                    tau_hat = cate_mod.fit_cate(
                                        learner, cate_feats, T_e, outcome,
                                        seed=trial_seed,
                                        regressor=cate_regressor,
                                        n_iterations=cate_n_iterations,
                                    )
                            except Exception as e:    # noqa: BLE001
                                # Some configurations (e.g. DML on raw-X with tiny n) are
                                # numerically pathological; record the row with NaN metrics
                                # rather than aborting the slice.
                                row = _empty_row(
                                    spec.name, method, phi_dim, arch, z_dim,
                                    z_s_fraction, flat["x_dim"], flat["s_dim"],
                                    alpha, delta, ss_eval, learner, y_type, "frozen",
                                    trial_seed, dgp_seed, key, fitted.epochs_run,
                                    cate_regressor=row_regressor,
                                    lambda_loss_s=flat["lambda_loss_s"],
                                    num_observational=num_observational,
                                )
                                row["error"] = type(e).__name__
                                rows.append(row)
                                continue
                            row = _empty_row(
                                spec.name, method, phi_dim, arch, z_dim,
                                z_s_fraction, flat["x_dim"], flat["s_dim"],
                                alpha, delta, ss_eval, learner, y_type, "frozen",
                                trial_seed, dgp_seed, key, fitted.epochs_run,
                                cate_regressor=row_regressor,
                                lambda_loss_s=flat["lambda_loss_s"],
                                num_observational=num_observational,
                            )
                            row = _row_metrics(row, tau_hat, tau_true)
                            # Diagnostics deferred to end-of-cell post-fill.
                            rows.append(row)

                    # ── Optional finetune variants ───────────────────────────────
                    # Gate: slice opted in AND this method is in finetune_methods AND
                    # the method has a torch module (neural). The previous hardcoded
                    # `method == "encoder_pred"` check is dropped — finetune_methods
                    # is the canonical source of truth.
                    #
                    # Two protocols:
                    # • finetune_xfit_folds == 1: single-fold finetune on
                    #   idx_finetune (== fold A or full E depending on split_frac).
                    #   CATE eval on (X_e, T_e, …) which is fold B (clean) or full
                    #   E (leaky), per the existing split logic above.
                    # • finetune_xfit_folds  > 1: K-fold cross-fitting. Every row
                    #   of idx is encoded by an encoder that did NOT see it during
                    #   finetune. CATE eval runs on the FULL idx (no split needed
                    #   for leakage avoidance — xfit handles it). Frozen rows
                    #   above already used idx_eval; for fair comparison run
                    #   finetune.yaml with finetune_split_frac=0 so frozen also
                    #   uses full E.
                    if (spec.finetune
                            and method in spec.finetune_methods
                            and fitted.torch_module is not None):
                        K = max(1, int(spec.finetune_xfit_folds))
                        ft_modes_iter = list(spec.finetune_modes)

                        freeze_bb = bool(flat.get("finetune_freeze_backbone", False))

                        def _do_finetune(ft_mode, ft_dict_local, fold_seed):
                            if ft_mode == "ftS":
                                return enc_mod.finetune_ftS(
                                    fitted.torch_module, ft_dict_local, flat, device,
                                    seed=fold_seed, use_y_loss=False,
                                    freeze_backbone=freeze_bb)
                            if ft_mode == "ftSY":
                                return enc_mod.finetune_ftS(
                                    fitted.torch_module, ft_dict_local, flat, device,
                                    seed=fold_seed, use_y_loss=True,
                                    freeze_backbone=freeze_bb)
                            if ft_mode == "ftY":
                                return enc_mod.finetune_ftY(
                                    fitted.torch_module, ft_dict_local, flat, device,
                                    seed=fold_seed,
                                    freeze_backbone=freeze_bb)
                            return None

                        if K > 1:
                            # ── K-fold cross-fitted finetune ────────────────────
                            # Eval on FULL idx; phi_xfit[i] comes from an encoder
                            # whose finetune set excluded fold(i). h-model also
                            # refit per fold on the matching encoder's φ(X_obs).
                            xfit_perm = np.random.default_rng(trial_seed * 71 + 13).permutation(len(idx))
                            fold_ids = xfit_perm % K   # roughly equal-size random folds

                            X_full = X_pool[idx]
                            T_full = T_pool[idx].flatten()
                            S_full_e = S_full[idx]
                            Y_full_e = Y_full[idx].flatten()
                            tau_true_full_e = tau_true_full[idx]
                            ss_xfit = len(idx)   # n_eval = full ss for xfit

                            for ft_mode in ft_modes_iter:
                                phi_xfit = np.zeros((ss_xfit, flat["phi_dim"]), dtype=np.float32)
                                Y_h_xfit = np.zeros(ss_xfit, dtype=np.float32)
                                any_fold_ok = False
                                for k in range(K):
                                    test_local = np.where(fold_ids == k)[0]
                                    train_local = np.where(fold_ids != k)[0]
                                    train_idx_full = idx[train_local]
                                    # T=1 filter on TRAIN side
                                    ft_T_train = T_pool[train_idx_full].flatten()
                                    if spec.finetune_t1_only:
                                        train_mask = ft_T_train.astype(bool)
                                        train_t1 = train_idx_full[train_mask]
                                    else:
                                        train_t1 = train_idx_full
                                    if len(train_t1) < 2:
                                        continue
                                    ft_dict_k = dict(
                                        X=X_pool[train_t1],
                                        T=T_pool[train_t1].flatten().reshape(-1, 1).astype(np.float32),
                                        S=S_full[train_t1],
                                        Y=Y_full[train_t1].flatten().reshape(-1, 1).astype(np.float32),
                                    )
                                    ft_enc_k = _do_finetune(ft_mode, ft_dict_k,
                                                            fold_seed=trial_seed * 100 + k)
                                    if ft_enc_k is None:
                                        continue
                                    # Encode the held-out fold
                                    test_X = X_pool[idx[test_local]]
                                    with torch.no_grad():
                                        phi_xfit[test_local] = ft_enc_k(
                                            torch.tensor(test_X)).numpy().astype(np.float32)
                                    # Per-fold h-model on (φ_k(X_obs), S_obs)
                                    with torch.no_grad():
                                        phi_obs_k = ft_enc_k(
                                            torch.tensor(obs["X"])).numpy().astype(np.float32)
                                    h_feats_obs_k = np.c_[phi_obs_k, obs["S"]]
                                    h_model_k = RandomForestRegressor(
                                        n_estimators=50, random_state=trial_seed, n_jobs=-1
                                    ).fit(h_feats_obs_k, obs["Y"].flatten())
                                    h_feats_test_k = np.c_[phi_xfit[test_local],
                                                           S_full_e[test_local]]
                                    Y_h_xfit[test_local] = h_model_k.predict(
                                        h_feats_test_k).astype(np.float32)
                                    any_fold_ok = True
                                if not any_fold_ok:
                                    continue

                                for learner in spec.learners:
                                    for y_type in spec.y_types:
                                        outcome = (Y_full_e if y_type == "true_y"
                                                   else Y_h_xfit)
                                        try:
                                            tau_hat = cate_mod.fit_cate(
                                                learner, phi_xfit, T_full, outcome,
                                                seed=trial_seed,
                                                regressor=cate_regressor,
                                                n_iterations=cate_n_iterations)
                                        except Exception:    # noqa: BLE001
                                            continue
                                        row = _empty_row(
                                            spec.name, method, phi_dim, arch, z_dim,
                                            z_s_fraction, flat["x_dim"], flat["s_dim"],
                                            alpha, delta, ss_xfit, learner, y_type, ft_mode,
                                            trial_seed, dgp_seed, key, fitted.epochs_run,
                                            cate_regressor=cate_regressor,
                                            lambda_loss_s=flat["lambda_loss_s"],
                                            num_observational=num_observational,
                                        )
                                        row = _row_metrics(row, tau_hat, tau_true_full_e)
                                        rows.append(row)
                            # End K-fold xfit branch
                        else:
                            # ── Single-fold finetune (existing protocol) ────────
                            ft_X = X_pool[idx_finetune]
                            ft_T = T_pool[idx_finetune].flatten()
                            ft_S = S_full[idx_finetune]
                            ft_Y = Y_full[idx_finetune].flatten()
                            if spec.finetune_t1_only:
                                mask = ft_T.astype(bool)
                                ft_X = ft_X[mask]; ft_T = ft_T[mask]
                                ft_S = ft_S[mask]; ft_Y = ft_Y[mask]
                            if len(ft_X) < 2:
                                print(f"    [cell {outer_idx + 1}] finetune skipped "
                                      f"({method}, ss_eval={ss_eval}): only {len(ft_X)} "
                                      f"rows after filters", flush=True)
                                ft_modes_iter = []

                            ft_dict = dict(X=ft_X,
                                           T=ft_T.reshape(-1, 1).astype(np.float32),
                                           S=ft_S,
                                           Y=ft_Y.reshape(-1, 1).astype(np.float32))
                            for ft_mode in ft_modes_iter:
                                ft_enc = _do_finetune(ft_mode, ft_dict, fold_seed=trial_seed)
                                if ft_enc is None:
                                    continue
                                with torch.no_grad():
                                    phi_ft = ft_enc(torch.tensor(X_e)).numpy().astype(np.float32)
                                # Refit h-model on (φ_ft(X_obs), S_obs).
                                phi_ft_obs = ft_enc(torch.tensor(obs["X"])).detach().numpy().astype(np.float32)
                                h_feats_ft_obs = np.c_[phi_ft_obs, obs["S"]]
                                h_model_ft = RandomForestRegressor(
                                    n_estimators=50, random_state=trial_seed, n_jobs=-1
                                ).fit(h_feats_ft_obs, obs["Y"].flatten())
                                h_feats_ft = np.c_[phi_ft, S_e].astype(np.float32)
                                Y_h_ft = h_model_ft.predict(h_feats_ft).astype(np.float32)
                                for learner in spec.learners:
                                    for y_type in spec.y_types:
                                        outcome = (Y_e if y_type == "true_y" else Y_h_ft)
                                        try:
                                            tau_hat = cate_mod.fit_cate(
                                                learner, phi_ft, T_e, outcome,
                                                seed=trial_seed,
                                                regressor=cate_regressor,
                                                n_iterations=cate_n_iterations)
                                        except Exception:    # noqa: BLE001
                                            continue
                                        row = _empty_row(
                                            spec.name, method, phi_dim, arch, z_dim,
                                            z_s_fraction, flat["x_dim"], flat["s_dim"],
                                            alpha, delta, ss_eval, learner, y_type, ft_mode,
                                            trial_seed, dgp_seed, key, fitted.epochs_run,
                                            cate_regressor=cate_regressor,
                                            lambda_loss_s=flat["lambda_loss_s"],
                                            num_observational=num_observational,
                                        )
                                        row = _row_metrics(row, tau_hat, tau_true)
                                        rows.append(row)

                    print(f"    [cell {outer_idx + 1}] α={alpha} δ={delta} n={ss} "
                          f"{method:14s} done ({time.time() - t_m:.1f}s, rows={len(rows)})",
                          flush=True)

                # Atomic checkpoint after each sample size (not each method) —
                # write to .tmp then os.replace so a kill mid-write can't leave
                # a corrupt parquet. Throttling per-ss instead of per-(ss, method)
                # cuts write IO by ~11× without meaningfully changing recovery.
                if rows:
                    out_path = output_dir / "results.parquet"
                    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
                    pd.DataFrame(rows).to_parquet(tmp_path, index=False)
                    os.replace(tmp_path, out_path)

        # ── Phase 2: diagnostics (deferred — CATE results are already saved) ─
        # Running diagnostics after CATE means a kill or timeout during the
        # diagnostic phase still leaves a complete results.parquet on disk.
        if spec.diagnostics:
            # Suff-i (H)
            diag_obs_per_method: Dict[str, Dict[str, float]] = {}
            t_diag = time.time()
            print(f"  [cell {outer_idx + 1}] diagnostics on H (post-CATE) ...", flush=True)
            for method in spec.methods:
                fitted, enc_key = fitted_by_method[method]
                if method in {"raw_x", "baseline_xs"}:
                    diag_obs_per_method[method] = {}
                    continue
                d_key = cache_mod.diag_cache_key(enc_key, "suff_i", trial_seed)
                cached_diag = cache_mod.load_diag(cache, d_key)
                if cached_diag is not None:
                    diag_obs_per_method[method] = cached_diag
                    print(f"    [cell {outer_idx + 1}] suff-i  {method:14s} (cached)", flush=True)
                    continue
                t_m = time.time()
                phi_obs = fitted.encode(obs["X"])
                diag_obs_per_method[method] = diag_mod.sufficiency_i_test(
                    phi_obs=phi_obs, S_obs=obs["S"], Y_obs=obs["Y"],
                    X_obs=obs["X"], seed=trial_seed,
                )
                cache_mod.save_diag(cache, d_key, diag_obs_per_method[method])
                print(f"    [cell {outer_idx + 1}] suff-i  {method:14s} ({time.time() - t_m:.1f}s)", flush=True)
            print(f"  [cell {outer_idx + 1}] diagnostics on H done ({time.time() - t_diag:.1f}s)", flush=True)

            # Suff-ii (E) per (α, δ), reusing the cached experimental pools.
            diag_exp_by_ad: Dict[Tuple[float, float], Dict[str, Dict[str, float]]] = {}
            for alpha, delta in product(spec.alphas, spec.deltas):
                pool = exp_pools_by_ad[(float(alpha), float(delta))]
                X_pool_d = pool["X_pool"]; T_pool_d = pool["T_pool"]; S_full_d = pool["S_full"]
                diag_exp_per_method: Dict[str, Dict[str, float]] = {}
                t_diag_e = time.time()
                print(f"  [cell {outer_idx + 1}] α={alpha} δ={delta}: diagnostics on E ...", flush=True)
                for method in spec.methods:
                    fitted, enc_key = fitted_by_method[method]
                    if method in {"raw_x", "baseline_xs"}:
                        diag_exp_per_method[method] = {}
                        continue
                    d_key = cache_mod.diag_cache_key(
                        enc_key, "suff_ii", trial_seed,
                        alpha=float(alpha), delta=float(delta),
                    )
                    cached_diag = cache_mod.load_diag(cache, d_key)
                    if cached_diag is not None:
                        diag_exp_per_method[method] = cached_diag
                        print(f"    [cell {outer_idx + 1}] suff-ii {method:14s} (cached)", flush=True)
                        continue
                    t_m = time.time()
                    phi_full = fitted.encode(X_pool_d)
                    diag_exp_per_method[method] = diag_mod.sufficiency_ii_residual_r2(
                        phi_exp=phi_full, S_exp=S_full_d, T_exp=T_pool_d,
                        X_exp=X_pool_d, seed=trial_seed,
                    )
                    cache_mod.save_diag(cache, d_key, diag_exp_per_method[method])
                    print(f"    [cell {outer_idx + 1}] suff-ii {method:14s} ({time.time() - t_m:.1f}s)", flush=True)
                diag_exp_by_ad[(float(alpha), float(delta))] = diag_exp_per_method
                print(f"  [cell {outer_idx + 1}] α={alpha} δ={delta}: diagnostics on E done ({time.time() - t_diag_e:.1f}s)", flush=True)

            # Post-fill the diag_* columns into the rows we appended in Phase 1.
            n_filled = 0
            for r in rows[cell_row_start:]:
                d_obs = diag_obs_per_method.get(r["method"], {})
                d_exp = diag_exp_by_ad.get(
                    (float(r["alpha"]), float(r["delta"])), {}
                ).get(r["method"], {})
                r["diag_r2_phi_s"]   = d_obs.get("r2_phi_s", _NA)
                r["diag_r2_phi_s_x"] = d_obs.get("r2_phi_s_x", _NA)
                r["diag_r2_gap"]     = d_obs.get("r2_gap", _NA)
                r["diag_r2_xs_resid_mean"] = d_exp.get("r2_xs_resid_mean", _NA)
                r["diag_r2_xs_resid_max"]  = d_exp.get("r2_xs_resid_max", _NA)
                n_filled += 1
            print(f"  [cell {outer_idx + 1}] diag-fill: {n_filled} rows updated", flush=True)

            # Atomic rewrite with the diagnostic columns now populated.
            if rows:
                out_path = output_dir / "results.parquet"
                tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
                pd.DataFrame(rows).to_parquet(tmp_path, index=False)
                os.replace(tmp_path, out_path)

        # Free this cell's cached pools before moving to the next cell.
        exp_pools_by_ad.clear()

        cell_elapsed = time.time() - cell_t0
        total_elapsed = time.time() - t_outer_start
        eta_min = (total_elapsed / (outer_idx + 1)) * (n_outer - outer_idx - 1) / 60.0
        print(f"[runner] cell {outer_idx + 1}/{n_outer} done in {cell_elapsed:.1f}s  "
              f"rows={len(rows)}  ETA≈{eta_min:.1f}min", flush=True)

    if pretrain_only:
        print(f"[runner] pretrain-only: cache populated, no parquet written.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    out_path = output_dir / "results.parquet"
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, out_path)
    print(f"[runner] wrote {len(df)} rows → {out_path}")

    if dgp_summary_rows:
        summary_df = pd.DataFrame(dgp_summary_rows).drop_duplicates()
        summary_path = output_dir / "dgp_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"[runner] wrote {len(summary_df)} dgp-summary rows → {summary_path}")

    return df
