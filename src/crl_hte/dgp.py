"""
Unified data-generating process for the CRL-HTE post-rebuttal experiments.

Latent structure
----------------
    X ~ N(0, I_x_dim)
    Z = X @ W_xz.T              (z_dim latent factors, supported on a subset of X)
    The z_dim latents are split into Z_s (first ns dims) and Z_y (last ny dims),
    where ns = round(z_dim * z_s_fraction).
        - Z_s: dims that drive S at baseline (encoder pretrained on H sees these).
        - Z_y: dims that φ never sees during pretraining.
                Used for both sufficiency-(ii) violation (alpha) and surrogacy
                violation (delta).

S generation (same in H and E)
------------------------------
    S_0 = beta_0 + Z @ beta_1.T         (loadings on Z_s only)
    tau_S = gamma_0
            + sqrt(1 - alpha^2) * (Z @ gamma_z.T)        # Z_s component
            +            alpha  * (Z @ gamma_y_norm.T)   # Z_y component (violation)
    S_1 = S_0 + tau_S
    S   = S_0 + tau_S * T + N(0, sigma_S)

    `gamma_y_norm` is rescaled under the actual covariance of Z so that
    Var((Z @ gamma_y_norm.T) @ w_sy) == Var((Z @ gamma_z.T) @ w_sy).
    The final scalar CATE variance is therefore constant across alpha; only the
    source of its heterogeneity changes.

Y generation
------------
    Y_0 = S_0 @ w_sy
    Y_1 = S_1 @ w_sy + delta * bypass(Z_y)
    Y   = S @ w_sy + delta * T * bypass(Z_y) + N(0, sigma_Y)

    bypass(Z_y) = Z @ w_bypass with w_bypass nonzero only on Z_y dims.
    `w_bypass` is rescaled under the actual covariance of Z so that
    Var(bypass) == Var((S_1-S_0) @ w_sy) at delta=1. Thus, the mediated and
    bypass paths contribute equal CATE variance when delta=1.

z_s_overlap mode
----------------
For sweeps that need NO Z_s/Z_y split (e.g. the original z_dim sweep where the
"main" result didn't have any Z_y component), pass `z_s_fraction=1.0` and
`alpha=0, delta=0`. Then Z_y is empty and the DGP reduces to: τ_S from all Z dims,
no bypass, no Z_y component. This matches step_mi_objectives.py up to ns=z_dim.

Reproducibility
---------------
A single rng (seeded from `dgp_seed`) generates all the static parameters
(W_xz, beta_0, beta_1, gamma_z, gamma_0, gamma_y_norm, w_sy, w_bypass, etc.).
Returned in a `DGPParams` dataclass. `generate_*` functions take a separate
sampling rng so the same parameter set can be paired with many sample-rng seeds
across trials.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


# ── Configurable nonlinearity for X→Z, Z→S, S→Y ──────────────────────────────
# Modes are intentionally chosen to NOT squash magnitudes to 0 (i.e. not tanh
# or sigmoid). They preserve sign and grow at least linearly in the tails so
# the encoders' job stays well-conditioned.
def _nonlin(arr: np.ndarray, mode: str) -> np.ndarray:
    if mode == "linear":
        return arr
    if mode == "leaky_relu":
        # slope 1 on positives, 0.5 on negatives — no info loss, clearly nonlinear
        return np.where(arr >= 0, arr, 0.5 * arr).astype(arr.dtype)
    if mode == "silu":
        # x * sigmoid(x); smooth, near-linear for large positive, slight tail
        return (arr / (1.0 + np.exp(-arr))).astype(arr.dtype)
    if mode == "cubic":
        # x + x³ — monotonic, no saturation, ~60% linear-R² on N(0,1)
        return (arr + arr ** 3).astype(arr.dtype)
    if mode == "square":
        # x² — even, zero linear-R² on any centered symmetric distribution.
        # Loses sign info on the projected coordinate, so use sparingly (e.g.
        # only at S→Y where Z/S still carry signed signal for the encoders).
        return (arr ** 2).astype(arr.dtype)
    raise ValueError(f"Unknown nonlinearity mode: {mode!r}. "
                     f"Expected one of: linear, leaky_relu, silu, cubic, square")


def _nonlin_modes(cfg: Dict) -> tuple:
    """Pull the three nonlinearity knobs out of the flat cfg dict."""
    return (
        str(cfg.get("x_to_z_nonlinearity", "linear")),
        str(cfg.get("z_to_s_nonlinearity", "linear")),
        str(cfg.get("s_to_y_nonlinearity", "linear")),
    )


def _latent_from_x(X: np.ndarray, params: "DGPParams", cfg: Dict) -> np.ndarray:
    """Canonical X→Z map shared by standalone generation and the slice runner."""
    nl_xz, _, _ = _nonlin_modes(cfg)
    if nl_xz == "projection_silu":
        scale = np.linalg.norm(params.W_xz, axis=1).clip(1e-6)
        projected = (X @ params.W_xz.T) / scale
        return _nonlin(projected, "silu") * scale
    if nl_xz == "smooth_interaction":
        scale = np.linalg.norm(params.W_xz, axis=1).clip(1e-6)
        first = (X @ params.W_xz.T) / scale
        second = (X @ np.roll(params.W_xz, 1, axis=1).T) / scale
        return (first + 0.5 * first * second) * scale / np.sqrt(1.25)
    return _nonlin(X, nl_xz) @ params.W_xz.T


def _surrogate_baseline(
    Z: np.ndarray, params: "DGPParams", cfg: Dict
) -> np.ndarray:
    """Canonical untreated surrogate mean S₀(Z)."""
    _, nl_zs, _ = _nonlin_modes(cfg)
    return params.beta_0[None, :] + _nonlin(Z, nl_zs) @ params.beta_1.T


def _surrogate_effect(
    Z: np.ndarray, params: "DGPParams", cfg: Dict, alpha: float
) -> np.ndarray:
    """Canonical treatment effect on S."""
    _, nl_zs, _ = _nonlin_modes(cfg)
    Z_nl = _nonlin(Z, nl_zs)
    a = float(alpha)
    tau_het = (
        np.sqrt(max(0.0, 1.0 - a * a)) * (Z_nl @ params.gamma_z.T)
        + a * (Z_nl @ params.gamma_y_norm.T)
    )
    return params.gamma_0[None, :] + tau_het


def _outcome_from_s(S: np.ndarray, params: "DGPParams", cfg: Dict) -> np.ndarray:
    """Canonical S→Y mean, excluding the direct bypass."""
    _, _, nl_sy = _nonlin_modes(cfg)
    return _nonlin(S, nl_sy) @ params.w_sy[:, None]


@dataclass
class DGPParams:
    """Static DGP parameters. One set per (dgp_seed, dgp config) combination."""

    # Latent structure
    W_xz: np.ndarray            # (z_dim, x_dim)
    z_indices_for_s: np.ndarray
    z_indices_for_y: np.ndarray
    rel_s: np.ndarray           # x_dim indices that Z_s loads on
    rel_y: np.ndarray           # x_dim indices that Z_y loads on (may be empty)

    # S parameters
    beta_0: np.ndarray
    beta_1: np.ndarray          # (s_dim, z_dim) — nonzero on Z_s
    gamma_0: np.ndarray
    gamma_z: np.ndarray         # (s_dim, z_dim) — nonzero on Z_s, baseline τ_S
    gamma_y_norm: np.ndarray    # (s_dim, z_dim) — nonzero on Z_y, alpha-violation τ_S
    w_sy: np.ndarray            # (s_dim,)

    # Y bypass (for delta surrogacy violation)
    w_bypass: np.ndarray        # (z_dim,) — nonzero on Z_y

    # Bookkeeping
    config: Dict = field(default_factory=dict)


def init_dgp_params(cfg: Dict, dgp_seed: int) -> DGPParams:
    """Sample all static DGP parameters from a single rng seeded by `dgp_seed`."""
    rng = np.random.default_rng(dgp_seed)
    x_dim, z_dim, s_dim = cfg["x_dim"], cfg["z_dim"], cfg["s_dim"]
    z_s_fraction = cfg.get("z_s_fraction", 0.5)

    ns = int(round(z_dim * z_s_fraction))
    ns = max(0, min(z_dim, ns))
    ny = z_dim - ns

    # ── X support: disjoint subsets for Z_s and Z_y, each of length z_dim ──
    # When ny=0 (z_s_fraction=1), Z_y is empty and only rel_s is allocated.
    n_to_pick = 2 * z_dim if ny > 0 else z_dim
    if n_to_pick > x_dim:
        raise ValueError(f"Need {n_to_pick} relevant X columns, x_dim={x_dim}")
    all_rel = rng.choice(x_dim, size=n_to_pick, replace=False)
    rel_s = all_rel[:z_dim]
    rel_y = all_rel[z_dim:] if ny > 0 else np.array([], dtype=int)

    W_xz = np.zeros((z_dim, x_dim))
    if ns > 0:
        # Each Z_s latent loads on the full rel_s subset (length z_dim).
        W_xz[:ns, rel_s] = rng.normal(0, 1.0, (ns, z_dim))
    if ny > 0:
        # Each Z_y latent loads on the full rel_y subset (length z_dim).
        W_xz[ns:, rel_y] = rng.normal(0, 1.0, (ny, z_dim))

    z_idx_s = np.arange(0, ns)
    z_idx_y = np.arange(ns, z_dim)

    # ── S parameters (loadings on Z_s only) ──
    # NOTE: draw order MUST match step_sufficiency2.initialize_data_generation_params:
    #   beta_1, gamma_z, gamma_y_raw, (normalize), beta_0, gamma_0, w_sy.
    # Reordering changes the numerical values for a given seed and breaks
    # reproducibility against the published rebuttal tables.
    beta_1 = np.zeros((s_dim, z_dim))
    if ns > 0:
        beta_1[:, z_idx_s] = rng.normal(0, 1.0, (s_dim, ns))

    # τ_S baseline component (Z_s only)
    gamma_z = np.zeros((s_dim, z_dim))
    if ns > 0:
        gamma_z[:, z_idx_s] = rng.normal(0, 0.3, (s_dim, ns))

    # τ_S sufficiency-violation component (Z_y only).  It is rescaled below
    # after w_sy is drawn so that the two components contribute equal variance
    # to the final scalar CATE, rather than merely to the vector-valued shift in S.
    gamma_y_raw = np.zeros((s_dim, z_dim))
    if ny > 0:
        gamma_y_raw[:, z_idx_y] = rng.normal(0, 0.3, (s_dim, ny))

    beta_0 = rng.normal(0, 0.5, size=(s_dim,))
    gamma_0 = rng.normal(0, 0.5, size=(s_dim,))
    w_sy = rng.normal(0, 1.0, size=(s_dim,))

    # In the canonical linear DGP, X ~ N(0, I) and Z = X W_xz^T, so
    # Cov(Z) = W_xz W_xz^T.  Match the variance of the *scalar outcome CATE*
    # contributed by Z_y to that contributed by Z_s.  With disjoint X support,
    # the two terms are independent; sqrt(1-alpha^2) and alpha therefore keep
    # Var(tau(X)) constant throughout the alpha sweep.  Leaving gamma_z fixed
    # preserves the alpha=0 DGP exactly.
    cov_z = W_xz @ W_xz.T
    cate_coef_z = gamma_z.T @ w_sy
    cate_coef_y_raw = gamma_y_raw.T @ w_sy
    var_cate_z = float(cate_coef_z @ cov_z @ cate_coef_z)
    var_cate_y_raw = float(cate_coef_y_raw @ cov_z @ cate_coef_y_raw)
    if var_cate_y_raw > 0 and var_cate_z > 0:
        scale = np.sqrt(var_cate_z / (var_cate_y_raw + 1e-12))
        gamma_y_norm = gamma_y_raw * scale
    else:
        gamma_y_norm = gamma_y_raw

    # ── Surrogacy bypass (Z_y only), variance-matched at delta=1 ──
    w_bp_raw = np.zeros(z_dim)
    if ny > 0:
        w_bp_raw[z_idx_y] = rng.normal(0, 1.0, (ny,))
    # At alpha=0, the heterogeneous mediated CATE has coefficient
    # gamma_z.T @ w_sy on Z. Match the bypass variance to it under the actual
    # latent covariance. Since the two paths use disjoint X support, their
    # covariance is zero in the canonical linear DGP.
    var_bp_raw = float(w_bp_raw @ cov_z @ w_bp_raw)
    if var_bp_raw > 0 and var_cate_z > 0:
        w_bypass = w_bp_raw * np.sqrt(var_cate_z / (var_bp_raw + 1e-12))
    else:
        w_bypass = w_bp_raw

    return DGPParams(
        W_xz=W_xz,
        z_indices_for_s=z_idx_s,
        z_indices_for_y=z_idx_y,
        rel_s=rel_s,
        rel_y=rel_y,
        beta_0=beta_0,
        beta_1=beta_1,
        gamma_0=gamma_0,
        gamma_z=gamma_z,
        gamma_y_norm=gamma_y_norm,
        w_sy=w_sy,
        w_bypass=w_bypass,
        config=dict(cfg),
    )


def generate_observational(
    n: int,
    params: DGPParams,
    cfg: Dict,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """
    Generate historical sample H. T=0 always (no treated units in H by setup).

    The bypass term vanishes (delta * T = 0). Sufficiency violation also has no
    effect on H because alpha enters τ_S which only matters when T=1. So H is
    invariant to (alpha, delta) — the encoder pretrained on H is reusable across
    all violation slices.
    """
    x_dim = cfg["x_dim"]
    sigma_S = cfg["sigma_S"]
    sigma_Y = cfg["sigma_Y"]
    X = rng.normal(0, 1, (n, x_dim)).astype(np.float32)
    Z = _latent_from_x(X, params, cfg)
    T = np.zeros((n, 1), dtype=np.float32)

    S_0 = _surrogate_baseline(Z, params, cfg)
    S = (S_0 + rng.normal(0, sigma_S, S_0.shape)).astype(np.float32)
    Y = (_outcome_from_s(S, params, cfg)
         + rng.normal(0, sigma_Y, (n, 1))).astype(np.float32)

    return dict(X=X, Z=Z.astype(np.float32), T=T, S=S, Y=Y)


def generate_experimental(
    n: int,
    params: DGPParams,
    cfg: Dict,
    rng: np.random.Generator,
    alpha: float = 0.0,
    delta: float = 0.0,
) -> Dict[str, np.ndarray]:
    """
    Generate experimental sample E with violation strengths alpha, delta.

    Returns a dict including potential outcomes (Y_0, Y_1, S_0, S_1) so
    `tau_true` can be computed for any metric.
    """
    x_dim = cfg["x_dim"]
    sigma_S = cfg["sigma_S"]
    sigma_Y = cfg["sigma_Y"]
    a = float(alpha)
    d = float(delta)

    X = rng.normal(0, 1, (n, x_dim)).astype(np.float32)
    Z = _latent_from_x(X, params, cfg)
    T = rng.binomial(1, 0.5, size=(n, 1)).astype(np.float32)

    tau_S = _surrogate_effect(Z, params, cfg, a)
    S_0 = _surrogate_baseline(Z, params, cfg)
    S_1 = S_0 + tau_S
    S = (S_0 + tau_S * T
         + rng.normal(0, sigma_S, S_0.shape)).astype(np.float32)

    bypass = (Z @ params.w_bypass).reshape(-1, 1)
    Y_0 = _outcome_from_s(S_0, params, cfg)
    Y_1 = _outcome_from_s(S_1, params, cfg) + d * bypass
    Y = (_outcome_from_s(S, params, cfg)
         + d * T * bypass
         + rng.normal(0, sigma_Y, (n, 1))).astype(np.float32)

    tau_true = (Y_1 - Y_0).flatten().astype(np.float32)

    return dict(
        X=X,
        Z=Z.astype(np.float32),
        T=T,
        S=S,
        Y=Y,
        S_0=S_0.astype(np.float32),
        S_1=S_1.astype(np.float32),
        Y_0=Y_0.astype(np.float32),
        Y_1=Y_1.astype(np.float32),
        tau_true=tau_true,
    )


def true_cate_components(params: DGPParams, Z: np.ndarray, alpha: float = 0.0,
                          delta: float = 0.0) -> Dict[str, np.ndarray]:
    """Decompose the true CATE into S-mediated and bypass components.

    Returns
    -------
    dict with keys:
        cate_S       : (n,) S-mediated CATE = (S_1 - S_0) @ w_sy
        cate_bypass  : (n,) bypass = delta * Z @ w_bypass
        cate_total   : sum of the above

    Useful for diagnostics and per-method decomposition reporting.
    """
    a = float(alpha); d = float(delta)
    tau_het = (np.sqrt(max(0.0, 1.0 - a * a)) * (Z @ params.gamma_z.T)
               + a * (Z @ params.gamma_y_norm.T))
    tau_S = params.gamma_0[None, :] + tau_het
    cate_S = tau_S @ params.w_sy
    cate_bypass = d * (Z @ params.w_bypass)
    return dict(
        cate_S=cate_S.astype(np.float32),
        cate_bypass=cate_bypass.astype(np.float32),
        cate_total=(cate_S + cate_bypass).astype(np.float32),
    )
