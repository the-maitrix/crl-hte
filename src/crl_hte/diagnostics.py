"""
Practitioner diagnostics promised in the rebuttal:

    sufficiency_i_test
        Tests Y* ⊥ X | phi(X), S on historical data. Implementation: regress
        Y on (phi, S, X) with a held-out split; if the residuals after
        regressing on (phi, S) are independent of X, the conditional
        independence holds. We report:
            r2_phi_s   : R^2 of Y ~ (phi, S)
            r2_phi_s_x : R^2 of Y ~ (phi, S, X)
            r2_gap     : r2_phi_s_x - r2_phi_s   — close to 0 means Y is
                         already captured by (phi, S); X adds nothing.
        A small gap supports sufficiency(i). A large gap signals violation.

    sufficiency_ii_residual_r2
        Tests S ⊥ X | phi(X), T on experimental data. We compute:
            residual S = S - h_S(phi, T)
            R^2 from regressing residual on X
        Small R^2 ⇒ X has no extra info about S beyond (phi, T). Per ZHXq
        W2/Q2 and i2fn Q3, this is the diagnostic practitioners can run on
        their own data to detect sufficiency-(ii) violation.

Both diagnostics are method-agnostic: pass any phi function (sklearn or torch).
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


def _fit_rf(features: np.ndarray, target: np.ndarray, seed: int = 42, n_jobs: int = -1):
    rf = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=n_jobs)
    return rf.fit(features, target)


def sufficiency_i_test(
    phi_obs: np.ndarray,
    S_obs: np.ndarray,
    Y_obs: np.ndarray,
    X_obs: np.ndarray,
    seed: int = 42,
    test_size: float = 0.3,
) -> Dict[str, float]:
    """Run sufficiency-(i) diagnostic on historical data.

    Returns r2_phi_s, r2_phi_s_x, and the gap (the smaller, the better).
    """
    Y_obs = np.asarray(Y_obs).flatten()
    feat_phi_s = np.c_[phi_obs, S_obs]
    feat_full = np.c_[phi_obs, S_obs, X_obs]

    splits = train_test_split(
        feat_phi_s, feat_full, Y_obs,
        test_size=test_size, random_state=seed,
    )
    feat_ps_tr, feat_ps_te, feat_full_tr, feat_full_te, y_tr, y_te = splits

    m1 = _fit_rf(feat_ps_tr, y_tr, seed=seed)
    m2 = _fit_rf(feat_full_tr, y_tr, seed=seed)
    r2_ps = r2_score(y_te, m1.predict(feat_ps_te))
    r2_full = r2_score(y_te, m2.predict(feat_full_te))
    return {
        "r2_phi_s": float(r2_ps),
        "r2_phi_s_x": float(r2_full),
        "r2_gap": float(r2_full - r2_ps),
    }


def sufficiency_ii_residual_r2(
    phi_exp: np.ndarray,
    S_exp: np.ndarray,
    T_exp: np.ndarray,
    X_exp: np.ndarray,
    seed: int = 42,
    test_size: float = 0.3,
) -> Dict[str, float]:
    """Run sufficiency-(ii) diagnostic on experimental data.

    Per-output R^2 averaged across S dimensions.
    """
    T_exp = np.asarray(T_exp).reshape(-1, 1)
    feat_phi_t = np.c_[phi_exp, T_exp]

    # Train h_S(phi, T) on a train split
    feat_tr, feat_te, X_tr, X_te, S_tr, S_te = train_test_split(
        feat_phi_t, X_exp, S_exp, test_size=test_size, random_state=seed,
    )
    h_S = _fit_rf(feat_tr, S_tr, seed=seed)
    S_resid_te = S_te - h_S.predict(feat_te)

    # Regress residual on X (per output)
    if S_resid_te.ndim == 1:
        S_resid_te = S_resid_te[:, None]
    r2s = []
    for j in range(S_resid_te.shape[1]):
        m = _fit_rf(X_tr, S_tr[:, j] - h_S.predict(feat_tr)[:, j], seed=seed)
        pred = m.predict(X_te)
        r2s.append(r2_score(S_resid_te[:, j], pred))
    return {
        "r2_xs_resid_mean": float(np.mean(r2s)),
        "r2_xs_resid_max":  float(np.max(r2s)),
    }
