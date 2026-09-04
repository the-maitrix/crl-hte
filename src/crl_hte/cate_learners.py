"""
CATE estimators used in the post-rebuttal experiments.

We expose three meta-learners with a uniform interface:
    fit_cate(name, features, T, Y, seed, regressor, n_iterations) -> tau_hat

`features` is whatever low-dim representation the upstream method produced
(e.g. phi_exp, or [phi_exp, S_exp], or X_exp).

The base regressor is selectable:
    'histgbr'  → HistGradientBoostingRegressor (default; OpenMP-parallel,
                  ~3-5× faster than gbr on high-dim raw_x features)
    'gbr'      → GradientBoostingRegressor (sequential, what step_mi_objectives
                  used historically — kept for ablation / reproducibility)
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from causalml.inference.meta import BaseTLearner, BaseXLearner
from econml.dml import DML
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge


LEARNER_NAMES = ("t_learner", "x_learner", "dml_learner")
REGRESSOR_NAMES = ("histgbr", "gbr")


def _make_regressor(kind: str, seed: int, n_iterations: int):
    if kind == "histgbr":
        return HistGradientBoostingRegressor(max_iter=n_iterations, random_state=seed)
    if kind == "gbr":
        return GradientBoostingRegressor(n_estimators=n_iterations, random_state=seed)
    raise KeyError(f"Unknown CATE regressor: {kind!r}. "
                   f"Expected one of {REGRESSOR_NAMES}.")


def fit_cate(name: str, features: np.ndarray, treatment: np.ndarray,
             outcome: np.ndarray, seed: int = 42,
             regressor: str = "histgbr", n_iterations: int = 40) -> np.ndarray:
    """Fit a meta-learner and return tau_hat at every row of `features`."""
    def _reg():
        return _make_regressor(regressor, seed, n_iterations)

    if name == "t_learner":
        learner = BaseTLearner(learner=_reg())
        learner.fit(features, treatment, outcome)
        return np.asarray(learner.predict(features)).flatten()

    if name == "x_learner":
        learner = BaseXLearner(learner=_reg())
        learner.fit(features, treatment, outcome)
        return np.asarray(learner.predict(features)).flatten()

    if name == "dml_learner":
        dml = DML(
            model_y=_reg(),
            model_t=_reg(),
            model_final=Ridge(alpha=1.0),
            cv=3, random_state=seed,
        )
        dml.fit(outcome, treatment, X=features)
        return np.asarray(dml.effect(features)).flatten()

    raise KeyError(f"Unknown learner: {name}")


def fit_all_cate(features: np.ndarray, treatment: np.ndarray, outcome: np.ndarray,
                 seed: int = 42, regressor: str = "histgbr",
                 n_iterations: int = 40) -> Dict[str, np.ndarray]:
    """Convenience: fit all three learners and return dict of tau_hat arrays."""
    return {name: fit_cate(name, features, treatment, outcome, seed,
                           regressor=regressor, n_iterations=n_iterations)
            for name in LEARNER_NAMES}
