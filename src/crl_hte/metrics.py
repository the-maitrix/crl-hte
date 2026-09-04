"""
Evaluation metrics: estimation quality + policy quality.

Estimation:
    mse           : mean squared error of tau_hat vs tau_true
    pehe          : sqrt(mse)
    pehe_norm     : pehe / std(tau_true)  — comparable across DGPs of different scale

Policy / ranking:
    spearman_rho  : Spearman rank correlation tau_hat vs tau_true
    pearson_r     : Pearson correlation
    topk_value    : mean true CATE among top-k% by tau_hat
    topk_norm     : topk_value / oracle topk (in [0, 1] when oracle > 0)
    abs_regret    : V*(oracle policy) - V(treat-if-positive policy)
    auuc          : normalised area under uplift curve
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
from scipy.stats import spearmanr, pearsonr

# np.trapezoid is the NumPy ≥2.0 name; np.trapz exists in older versions.
_trapezoid = getattr(np, "trapezoid", np.trapz)


def mse(tau_hat: np.ndarray, tau_true: np.ndarray) -> float:
    return float(np.mean((tau_hat - tau_true) ** 2))


def pehe(tau_hat: np.ndarray, tau_true: np.ndarray) -> float:
    return float(np.sqrt(mse(tau_hat, tau_true)))


def pehe_norm(tau_hat: np.ndarray, tau_true: np.ndarray) -> float:
    sd = float(np.std(tau_true))
    if sd < 1e-12:
        return float("nan")
    return pehe(tau_hat, tau_true) / sd


def spearman_rho(tau_hat: np.ndarray, tau_true: np.ndarray) -> float:
    if np.std(tau_hat) < 1e-12 or np.std(tau_true) < 1e-12:
        return float("nan")
    return float(spearmanr(tau_hat, tau_true).correlation)


def pearson_r(tau_hat: np.ndarray, tau_true: np.ndarray) -> float:
    if np.std(tau_hat) < 1e-12 or np.std(tau_true) < 1e-12:
        return float("nan")
    return float(pearsonr(tau_hat, tau_true)[0])


def topk_value(tau_hat: np.ndarray, tau_true: np.ndarray, k_frac: float) -> float:
    """Mean true CATE among top k_frac by tau_hat ranking."""
    n = len(tau_hat)
    k = max(1, int(round(k_frac * n)))
    idx = np.argsort(-tau_hat)[:k]   # top-k by predicted
    return float(np.mean(tau_true[idx]))


def topk_norm(tau_hat: np.ndarray, tau_true: np.ndarray, k_frac: float) -> float:
    """topk_value(tau_hat) divided by oracle topk_value(tau_true)."""
    oracle = topk_value(tau_true, tau_true, k_frac)
    if abs(oracle) < 1e-12:
        return float("nan")
    return topk_value(tau_hat, tau_true, k_frac) / oracle


def policy_value(tau_hat: np.ndarray, score: np.ndarray, k_frac: float) -> float:
    """Mean of `score` among the top-k_frac% units ranked by `tau_hat`.

    `score` is the per-unit gold reward used to evaluate the policy: oracle
    `tau_true` for synth/semisynth (where it's available) or the AIPW
    pseudo-outcome ψ when the true effect is unavailable. Picks top-k by `tau_hat`,
    so higher = better is assumed throughout — matching the convention used
    by all three datasets after the semisynth sign flip.
    """
    n = len(score)
    k = max(1, int(round(k_frac * n)))
    return float(np.mean(score[np.argsort(-tau_hat)[:k]]))


def policy_value_norm(tau_hat: np.ndarray, score: np.ndarray, k_frac: float) -> float:
    """Normalized policy value: 0 = random policy, 1 = oracle policy.

        norm = (picked − random) / (oracle − random)

    `picked` = mean(score) among top-k by tau_hat. `random` = mean(score).
    `oracle` = mean(score) among top-k by score itself. Bounded above by 1
    for any policy that ranks better than random; can go negative for an
    adversarial ranker. Single shared headline policy metric for synth /
    semi-synthetic data.
    """
    rand = float(score.mean())
    picked = policy_value(tau_hat, score, k_frac)
    oracle = policy_value(score, score, k_frac)
    denom = oracle - rand
    if abs(denom) < 1e-12:
        return float("nan")
    return (picked - rand) / denom


def abs_regret(tau_hat: np.ndarray, tau_true: np.ndarray) -> float:
    """V*(oracle policy) - V(treat-if-positive). Lower is better.

    V(pi) = mean over units of pi(x) * tau_true(x), with pi in {0,1}.
    """
    pi_oracle = (tau_true > 0).astype(float)
    pi_hat = (tau_hat > 0).astype(float)
    v_star = float(np.mean(pi_oracle * tau_true))
    v_hat = float(np.mean(pi_hat * tau_true))
    return v_star - v_hat


def norm_regret(tau_hat: np.ndarray, tau_true: np.ndarray) -> float:
    """abs_regret / V*(oracle). Returns NaN if V* <= 0."""
    pi_oracle = (tau_true > 0).astype(float)
    v_star = float(np.mean(pi_oracle * tau_true))
    if v_star <= 1e-12:
        return float("nan")
    return abs_regret(tau_hat, tau_true) / v_star


def auuc(tau_hat: np.ndarray, tau_true: np.ndarray) -> float:
    """Normalised area under uplift curve (Qini-style, normalised by oracle)."""
    n = len(tau_hat)
    order = np.argsort(-tau_hat)
    cum_true = np.cumsum(tau_true[order])
    auc_pred = float(_trapezoid(cum_true, dx=1.0 / n))
    order_or = np.argsort(-tau_true)
    cum_or = np.cumsum(tau_true[order_or])
    auc_or = float(_trapezoid(cum_or, dx=1.0 / n))
    if abs(auc_or) < 1e-12:
        return float("nan")
    return auc_pred / auc_or


_DEFAULT_TOPK_FRACS = (0.1, 0.2, 0.5)


def compute_all(tau_hat: np.ndarray, tau_true: np.ndarray,
                topk_fracs: Iterable[float] = _DEFAULT_TOPK_FRACS) -> Dict[str, float]:
    out = {
        "mse": mse(tau_hat, tau_true),
        "pehe": pehe(tau_hat, tau_true),
        "pehe_norm": pehe_norm(tau_hat, tau_true),
        "spearman_rho": spearman_rho(tau_hat, tau_true),
        "pearson_r": pearson_r(tau_hat, tau_true),
        "abs_regret": abs_regret(tau_hat, tau_true),
        "norm_regret": norm_regret(tau_hat, tau_true),
        "auuc": auuc(tau_hat, tau_true),
    }
    for k in topk_fracs:
        kk = int(k * 100)
        out[f"topk_{kk}_value"]    = topk_value(tau_hat, tau_true, k)
        out[f"topk_{kk}_norm"]     = topk_norm(tau_hat, tau_true, k)
        out[f"policy_value_{kk}"]  = policy_value(tau_hat, tau_true, k)
        out[f"policy_norm_{kk}"]   = policy_value_norm(tau_hat, tau_true, k)
    return out
