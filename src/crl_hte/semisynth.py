"""Semi-synthetic DGP from real UPMC perinatal data.

Real (X, S) loaded from a one-shot cache built by scripts/build_semisynth.py;
synthetic treatment T and τ_S(X) effect injected on top. The two `generate_*`
functions return the same dict schema as src/crl_hte/dgp.py, so runner.py is
agnostic to which DGP it's pulling from.

Cohorts (built once)
    H : MHP_OnboardingStatus is missing or Pending (not onboarded). T ≡ 0.
    E : MHP_OnboardingStatus == 1 (Done). T ~ Bern(0.5).

Generative mechanism
    S₀_H, S₀_E = PCA(common diagnosis trajectories)       # real, observed
    Legacy mode transports MHP-only S into H via p̂(S|X).

    g(X)   = γ₀ + X[rel_idx] · γ_x        (scalar; std==target_logit_std)
    τ_S(X) = g(X) · β_S / ‖β_S‖²          (anchored to β_S direction)
    S₁     = S₀ + τ_S(X)
    S      = S₀ + T·τ_S(X) + N(0, σ_S)

    logit p_t = X·μ_X_w + μ_X_b + λ_S · (S(t) · β_S)      # two separate fits
                                                          # so β_S survives X
    binary : Y_PPD(t) ~ Bernoulli(σ(logit p_t)), Y(t) = 1 − Y_PPD(t)
    risk   : Y_PPD(t) = σ(logit p_t) + N(0, σ_Y),  Y(t) = 1 − Y_PPD(t)

    Convention flip: the model internally generates Y_PPD ∈ {0,1} (=1 means
    patient has PPD, "lower is better"); we flip Y at output so Y=1 means
    "no PPD" and τ_true = p_0 − p_1 is positive when the treatment helps.
    This matches the "higher τ = better" convention used by the synthetic and
    synthetic pipeline so the same policy metric works across both datasets.

    By construction logit p_1 − logit p_0 = λ_S · g(X) (constructed sign:
    `tau_sign=neg` enforces g ≤ 0 ⇒ p_1 ≤ p_0 ⇒ τ_true = p_0 − p_1 ≥ 0).

`init_semisynth_params` mirrors `init_dgp_params`: a single `dgp_seed` rng
samples all static effect parameters; sample-time rngs are passed separately.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


_REF_SEED = 99999
_REF_N = 50_000


def split_mhp_cohorts(status, *, include_pending: bool):
    values = pd.to_numeric(pd.Series(status), errors="coerce")
    pending = values.eq(0.0).to_numpy()
    experimental = values.eq(1.0).to_numpy()
    no_mhp = values.isna().to_numpy()
    historical = no_mhp | pending if include_pending else no_mhp
    return historical, experimental, pending, no_mhp


def diagnosis_prevalence_audit(
    df: pd.DataFrame,
    status,
    *,
    prefixes,
    min_reference_rate: float,
):
    _, experimental, pending, no_mhp = split_mhp_cohorts(
        status, include_pending=True
    )
    columns = [
        c for c in df.columns
        if c.startswith(tuple(prefixes)) and not c.endswith("_missing")
    ]
    if not columns or not pending.any():
        return {"ratio": np.nan, "columns": tuple(columns)}

    values = df[columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    pending_rates = values.loc[pending].ne(0).mean(axis=0)
    reference_rates = values.loc[no_mhp | experimental].ne(0).mean(axis=0)
    eligible = reference_rates >= min_reference_rate
    ratios = pending_rates[eligible] / reference_rates[eligible]
    ratio = float(ratios.median()) if len(ratios) else np.nan
    return {
        "ratio": ratio,
        "columns": tuple(reference_rates[eligible].index),
    }


@dataclass
class SemiSynthCache:
    """Real-data artifacts loaded from the build-script cache.

    X_H, X_E carry the *full* leakage-filtered feature matrix (high-dim,
    typically several hundred dims). `sel_idx` indexes into X to pick out the
    subset on which μ_X and p̂(S|X) were fit. The encoder / CATE pipeline
    operates on the full X, while the DGP's known structural relations
    (logit μ_X, S transport) are computed on X[:, sel_idx].
    """
    X_H: np.ndarray            # (n_H, p_full) historical-cohort covariates (full)
    X_E: np.ndarray            # (n_E, p_full) experimental-cohort covariates (full)
    y_H: np.ndarray            # (n_H,)   real PPD label, diagnostics only
    y_E: np.ndarray            # (n_E,)   real PPD label, diagnostics only
    S_E: np.ndarray            # (n_E, s_dim) real surrogate representation
    S_H: Optional[np.ndarray]  # (n_H, s_dim) when surrogates are observed in H
    surrogate_source: str
    feat_names: np.ndarray     # (p_full,)
    sel_idx: np.ndarray        # (p_sel,)     indices into X used by μ_X and p̂(S|X)
    pSX_W: np.ndarray          # (s_dim, p_sel)  p̂(S|X[:, sel_idx]) ridge weights
    pSX_b: np.ndarray          # (s_dim,)
    pSX_sigma: np.ndarray      # (s_dim,)     residual std per S dim
    muX_w: np.ndarray          # (p_sel,)     μ_X(X[:, sel_idx])→Y logistic weights
    muX_b: float               # μ_X intercept
    betaS: np.ndarray          # (s_dim,)     S→Y logistic weights (no intercept)
    lambda_s: float            # λ_S — how much S contributes to logit Y

    @classmethod
    def load(cls, cache_dir: str | Path) -> "SemiSynthCache":
        d = np.load(Path(cache_dir) / "cache.npz", allow_pickle=False)
        return cls(
            X_H=d["X_H"], X_E=d["X_E"],
            y_H=d["y_H"], y_E=d["y_E"],
            S_E=d["S_E"],
            S_H=d["S_H"] if "S_H" in d.files else None,
            surrogate_source=(
                str(d["surrogate_source"].item())
                if "surrogate_source" in d.files else "mhp_extras"
            ),
            feat_names=d["feat_names"],
            sel_idx=d["sel_idx"].astype(np.int64),
            pSX_W=d["pSX_W"], pSX_b=d["pSX_b"], pSX_sigma=d["pSX_sigma"],
            muX_w=d["muX_w"], muX_b=float(d["muX_b"]),
            betaS=d["betaS"], lambda_s=float(d["lambda_s"]),
        )


@dataclass
class SemiSynthParams:
    """Static synthetic-effect parameters (one set per dgp_seed × cfg).

    g(X) is a *scalar* heterogeneity function (in logit units). τ_S is laid
    along β_S/‖β_S‖² so β_S · τ_S = g(X).

      g_form="linear"    →  g(X) = γ_0 + X[rel_idx] · γ_x
      g_form="pairwise"  →  g(X) = γ_0 + Σ_m γ_m · X_{a_m} · X_{b_m}
                            where (a_m, b_m) is a random pairing of rel_idx.
                            Non-monotonic in any single X coordinate (the
                            partial derivative depends on the partner X), so
                            pure linear-DR projections cannot represent it.
    """
    rel_idx: np.ndarray        # X-dim indices that drive g(X)
    gamma_0: float             # scalar offset in logit units
    gamma_x: np.ndarray        # (len(rel_idx),) — used when g_form == "linear"
    u_S: np.ndarray            # (s_dim,)  β_S / ‖β_S‖²   (precomputed)
    sigma_S: np.ndarray        # (s_dim,)  observation noise on S, per dim
    sigma_Y: float
    y_mode: str                # 'binary' | 'risk'
    g_form: str = "linear"     # 'linear' | 'pairwise'
    pair_a: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    pair_b: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    gamma_pair: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    config: Dict = field(default_factory=dict)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def init_semisynth_params(cache: SemiSynthCache, cfg: Dict, dgp_seed: int) -> SemiSynthParams:
    """Sample γ_0, γ_x and pin τ_S to the β_S direction.

    `rel_idx` is drawn from `cache.sel_idx` (the L1-selected, Y-informative X
    columns) — same semantics as the synthetic DGP, where τ heterogeneity
    lives on the same X dims that drive the latent structure.

    Variance-normalize γ_x on a reference X subset (drawn from H with a frozen
    seed) so std(g(X)) == `target_logit_std`. Logit shift on Y from T=0 to T=1
    is then exactly λ_S · g(X), independent of how β_S splits across S dims.
    """
    rng = np.random.default_rng(dgp_seed)
    sel_dim = len(cache.sel_idx)

    n_rel = max(1, int(round(float(cfg.get("rel_x_frac", 0.25)) * sel_dim)))
    # rel_idx are positions in the *full* X (not in sel_idx), drawn from sel_idx.
    rel_idx = np.sort(cache.sel_idx[rng.choice(sel_dim, size=n_rel, replace=False)])

    gamma_0 = float(rng.normal(0, 0.05))
    gamma_x_raw = rng.normal(0, 1.0, size=(n_rel,))

    sign = str(cfg.get("tau_sign", "both"))
    if sign == "pos":
        gamma_x_raw = np.abs(gamma_x_raw)
    elif sign == "neg":
        gamma_x_raw = -np.abs(gamma_x_raw)
    elif sign != "both":
        raise ValueError(f"tau_sign must be one of {{'pos','neg','both'}}, got {sign!r}")

    # Reference X subset for variance normalization (frozen seed).
    ref_rng = np.random.default_rng(_REF_SEED)
    pool = cache.X_H if cache.X_H.shape[0] >= 1000 else np.concatenate([cache.X_H, cache.X_E])
    n_ref = min(_REF_N, pool.shape[0])
    X_ref = pool[ref_rng.choice(pool.shape[0], size=n_ref, replace=False)]

    target = float(cfg.get("target_logit_std", 0.5))
    g_form = str(cfg.get("g_form", "linear"))

    # Default empty pair arrays (overwritten if g_form == "pairwise").
    pair_a = np.empty(0, dtype=np.int64)
    pair_b = np.empty(0, dtype=np.int64)
    gamma_pair = np.empty(0, dtype=np.float32)

    if g_form == "linear":
        g_ref = X_ref[:, rel_idx] @ gamma_x_raw
        gamma_x = gamma_x_raw * (target / (g_ref.std() + 1e-8))
    elif g_form == "pairwise":
        # Random pairing of rel_idx into ⌊n_rel/2⌋ disjoint pairs.
        perm = rng.permutation(rel_idx)
        n_pairs = len(perm) // 2
        if n_pairs == 0:
            raise ValueError(f"g_form='pairwise' needs ≥2 rel_idx columns; got {len(perm)}")
        pair_a = perm[: 2 * n_pairs : 2]
        pair_b = perm[1: 2 * n_pairs : 2]
        gamma_pair_raw = rng.normal(0, 1.0, size=(n_pairs,))
        if sign == "pos":
            gamma_pair_raw = np.abs(gamma_pair_raw)
        elif sign == "neg":
            gamma_pair_raw = -np.abs(gamma_pair_raw)
        # Variance-normalize on the same reference X subset.
        g_ref_pair = (X_ref[:, pair_a] * X_ref[:, pair_b]) @ gamma_pair_raw
        gamma_pair = (gamma_pair_raw * (target / (g_ref_pair.std() + 1e-8))).astype(np.float32)
        # gamma_x kept as a placeholder (unused under pairwise).
        gamma_x = np.zeros_like(gamma_x_raw)
    else:
        raise ValueError(f"g_form must be 'linear' or 'pairwise', got {g_form!r}")

    # Anchor: τ_S = g(X) · u_S, where u_S = β_S / ‖β_S‖².
    bs = cache.betaS
    u_S = bs / (float(np.dot(bs, bs)) + 1e-12)

    sigma_S = float(cfg.get("sigma_S_frac", 0.1)) * cache.S_E.std(axis=0)

    return SemiSynthParams(
        rel_idx=rel_idx,
        gamma_0=gamma_0,
        gamma_x=gamma_x.astype(np.float32),
        u_S=u_S.astype(np.float32),
        sigma_S=sigma_S.astype(np.float32),
        sigma_Y=float(cfg.get("sigma_Y", 0.05)),
        y_mode=str(cfg.get("y_mode", "binary")),
        g_form=g_form,
        pair_a=pair_a,
        pair_b=pair_b,
        gamma_pair=gamma_pair,
        config=dict(cfg),
    )


def _g(X: np.ndarray, params: SemiSynthParams) -> np.ndarray:
    """Scalar heterogeneity in logit units, shape (n,)."""
    if params.g_form == "pairwise":
        return params.gamma_0 + (X[:, params.pair_a] * X[:, params.pair_b]) @ params.gamma_pair
    # default: linear
    return params.gamma_0 + X[:, params.rel_idx] @ params.gamma_x


def _tau_S(X: np.ndarray, params: SemiSynthParams) -> np.ndarray:
    """τ_S(X) = g(X) · u_S, shape (n, s_dim). β_S · τ_S(X) ≡ g(X)."""
    return _g(X, params)[:, None] * params.u_S[None, :]


def _p_y(X: np.ndarray, S: np.ndarray, cache: SemiSynthCache) -> np.ndarray:
    """Two-piece logit: μ_X(X[:, sel_idx]) + λ_S · (S · β_S). Shape (n, 1).
    X may be the full-dim feature matrix; only the sel_idx columns participate
    in μ_X."""
    X_sel = X[:, cache.sel_idx]
    logit = X_sel @ cache.muX_w + cache.muX_b + cache.lambda_s * (S @ cache.betaS)
    return _sigmoid(logit).reshape(-1, 1)


def _draw_y(p: np.ndarray, params: SemiSynthParams, rng: np.random.Generator) -> np.ndarray:
    if params.y_mode == "binary":
        return rng.binomial(1, p).astype(np.float32)
    if params.y_mode == "risk":
        return (p + rng.normal(0, params.sigma_Y, p.shape)).astype(np.float32)
    raise ValueError(f"y_mode must be 'binary' or 'risk', got {params.y_mode!r}")


def _sample_idx(n_pool: int, n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(n_pool, size=n, replace=(n > n_pool))


def generate_observational(
    n: int,
    cache: SemiSynthCache,
    params: SemiSynthParams,
    cfg: Dict,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Sample n rows from the historical not-onboarded pool. T ≡ 0."""
    idx = _sample_idx(cache.X_H.shape[0], n, rng)
    X = cache.X_H[idx].astype(np.float32)
    if cache.S_H is None:
        X_sel = X[:, cache.sel_idx]
        S0_mean = X_sel @ cache.pSX_W.T + cache.pSX_b[None, :]
        S0 = (
            S0_mean + rng.normal(0, cache.pSX_sigma, S0_mean.shape)
        ).astype(np.float32)
    else:
        S0 = cache.S_H[idx].astype(np.float32)
    S = (S0 + rng.normal(0, params.sigma_S, S0.shape)).astype(np.float32)
    T = np.zeros((n, 1), dtype=np.float32)

    Y_ppd = _draw_y(_p_y(X, S, cache), params, rng)
    Y = (1.0 - Y_ppd).astype(np.float32)  # "no PPD" — higher = better
    return dict(X=X, T=T, S=S, Y=Y)


def generate_experimental(
    n: int,
    cache: SemiSynthCache,
    params: SemiSynthParams,
    cfg: Dict,
    rng: np.random.Generator,
    pool_indices: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Sample n rows from the MHP-completed pool. T ~ Bern(0.5).

    Returns potential outcomes (S_0, S_1, Y_0, Y_1) and oracle `tau_true`
    on the probability scale (p_1 − p_0).
    """
    available = (
        np.arange(cache.X_E.shape[0], dtype=np.int64)
        if pool_indices is None else np.asarray(pool_indices, dtype=np.int64)
    )
    if available.ndim != 1 or len(np.unique(available)) != len(available):
        raise ValueError("pool_indices must contain unique one-dimensional indices")
    if np.any(available < 0) or np.any(available >= cache.X_E.shape[0]):
        raise ValueError("pool_indices contains an out-of-range index")
    idx = available[_sample_idx(len(available), n, rng)]
    X = cache.X_E[idx].astype(np.float32)
    S0 = cache.S_E[idx].astype(np.float32)

    tau = _tau_S(X, params).astype(np.float32)
    surrogate_noise = rng.normal(0, params.sigma_S, S0.shape).astype(np.float32)
    S0 = (S0 + surrogate_noise).astype(np.float32)
    S1 = (S0 + tau).astype(np.float32)
    T = rng.binomial(1, 0.5, size=(n, 1)).astype(np.float32)
    S = np.where(T > 0.5, S1, S0).astype(np.float32)

    p0 = _p_y(X, S0, cache)
    p1 = _p_y(X, S1, cache)
    Y0_ppd = _draw_y(p0, params, rng)
    Y1_ppd = _draw_y(p1, params, rng)
    # Convention flip: Y=1 means "no PPD" (higher Y = better outcome), τ=p_0−p_1
    # so positive τ = treatment helps.
    Y0 = (1.0 - Y0_ppd).astype(np.float32)
    Y1 = (1.0 - Y1_ppd).astype(np.float32)
    Y = np.where(T > 0.5, Y1, Y0).astype(np.float32)
    tau_true = (p0 - p1).flatten().astype(np.float32)

    return dict(
        X=X, T=T, S=S, Y=Y, pool_idx=idx,
        S_0=S0, S_1=S1,
        Y_0=Y0, Y_1=Y1,
        tau_true=tau_true,
    )
