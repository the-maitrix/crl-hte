"""Run the perinatal semi-synthetic experiment.

The private source records are not distributed. Build ``cache.npz`` with
``scripts/build_semisynth.py`` or provide an equivalent cache containing the
arrays consumed by ``SemiSynthCache``. The main analysis uses ``--outcome-source predicted``;
``--outcome-source true`` reproduces the true-outcome appendix sensitivity.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from econml.dml import DML
from econml.dr import DRLearner
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from crl_hte import encoders, metrics
from crl_hte.semisynth import (
    SemiSynthCache,
    generate_experimental,
    generate_observational,
    init_semisynth_params,
)

METHODS = [
    "encoder_pred", "encoder_no_s", "encoder_no_y",
    "mi_mine", "mi_infonce", "mi_mine_cond", "mi_infonce_cond", "mi_vib",
    "autoencoder", "pls", "pca", "ica", "raw_x", "raw_x_s",
]


def encoder_config(x_dim: int, s_dim: int, phi_dim: int, epochs: int) -> dict:
    return {
        "x_dim": x_dim,
        "s_dim": s_dim,
        "y_dim": 1,
        "phi_dim": phi_dim,
        "arch": "small",
        "batch_size": 256,
        "lr": 1e-3,
        "epochs_obs": epochs,
        "early_stopping_patience": 10,
        "early_stopping_min_delta": 1e-4,
        "val_frac": 0.1,
        "select_best_epoch": True,
        "lambda_loss_s": 2.0,
        "infonce_temperature": 0.1,
        "proj_dim": 10,
    }


def dr_predict(features, treatment, outcome, test_features, seed):
    treatment = np.asarray(treatment).astype(int).ravel()
    outcome = np.asarray(outcome).ravel()
    pseudo = np.empty(len(outcome), dtype=np.float64)
    folds = KFold(n_splits=2, shuffle=True, random_state=seed)
    for train, valid in folds.split(features):
        models = []
        for arm in (0, 1):
            arm_train = train[treatment[train] == arm]
            model = HistGradientBoostingRegressor(max_iter=40, random_state=seed)
            model.fit(features[arm_train], outcome[arm_train])
            models.append(model)
        mu0 = models[0].predict(features[valid])
        mu1 = models[1].predict(features[valid])
        tv = treatment[valid]
        pseudo[valid] = (
            mu1 - mu0
            + tv / 0.5 * (outcome[valid] - mu1)
            - (1 - tv) / 0.5 * (outcome[valid] - mu0)
        )
    final = HistGradientBoostingRegressor(max_iter=40, random_state=seed)
    final.fit(features, pseudo)
    return final.predict(test_features), final.predict(features)


def econml_predict(features, treatment, outcome, test_features, seed):
    def nuisance():
        return HistGradientBoostingRegressor(max_iter=40, random_state=seed)

    model = DML(
        model_y=nuisance(),
        model_t=nuisance(),
        model_final=Ridge(alpha=1.0),
        cv=3,
        random_state=seed,
    )
    model.fit(np.asarray(outcome).ravel(), np.asarray(treatment).ravel(), X=features)
    return (
        np.asarray(model.effect(test_features)).ravel(),
        np.asarray(model.effect(features)).ravel(),
    )


def econml_dr_forest_predict(features, treatment, outcome, test_features, seed):
    model = DRLearner(
        model_propensity=DummyClassifier(strategy="prior"),
        model_regression=HistGradientBoostingRegressor(max_iter=60, random_state=seed),
        model_final=RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=10,
            max_features=1.0,
            n_jobs=-1,
            random_state=seed,
        ),
        cv=3,
        min_propensity=0.05,
        random_state=seed,
    )
    model.fit(np.asarray(outcome).ravel(), np.asarray(treatment).astype(int).ravel(), X=features)
    return (
        np.asarray(model.effect(test_features)).ravel(),
        np.asarray(model.effect(features)).ravel(),
    )


def scale_cate_features(train_features, test_features):
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    active = std > 1e-4
    if not active.any():
        raise ValueError("representation has no nonconstant experimental coordinates")
    train_scaled = (train_features[:, active] - mean[active]) / std[active]
    test_scaled = (test_features[:, active] - mean[active]) / std[active]
    return np.clip(train_scaled, -5.0, 5.0), np.clip(test_scaled, -5.0, 5.0)


def run(args) -> pd.DataFrame:
    with open(args.config) as handle:
        config = yaml.safe_load(handle)
    cache = SemiSynthCache.load(args.cache_dir)
    params = init_semisynth_params(cache, config["dgp"], dgp_seed=args.dgp_seed)
    x_mean = cache.X_H.mean(axis=0)
    x_std = cache.X_H.std(axis=0).clip(1e-6)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cate_functions = {
        "current_dr": dr_predict,
        "econml_dml": econml_predict,
        "econml_dr_forest": econml_dr_forest_predict,
    }
    cate_function = cate_functions[args.cate_learner]
    dataset = args.label or f"semisynthetic_{args.outcome_source}_phi{args.phi_dim}"
    ate_calibration = args.ate_calibration if args.outcome_source == "predicted" else "none"
    rows = []

    for seed in range(args.seed_start, args.seed_start + args.seeds):
        historical = generate_observational(
            args.historical_n,
            cache,
            params,
            config["dgp"],
            np.random.default_rng(1000 + seed),
        )
        x_historical = (historical["X"] - x_mean) / x_std
        observed = {"X": x_historical, "S": historical["S"], "Y": historical["Y"]}
        representation_config = encoder_config(
            x_historical.shape[1], historical["S"].shape[1], args.phi_dim, args.epochs
        )
        fitted = {
            method: encoders.fit_method(
                "raw_x" if method == "raw_x_s" else method,
                observed,
                representation_config,
                device,
                seed,
            )
            for method in args.methods
        }
        bridges = {}
        if args.outcome_source == "predicted":
            for method, representation in fitted.items():
                bridge_features = (
                    x_historical
                    if method == "raw_x"
                    else np.c_[representation.encode(x_historical), historical["S"]]
                )
                bridges[method] = RandomForestRegressor(
                    n_estimators=50, random_state=seed, n_jobs=-1
                ).fit(bridge_features, historical["Y"].ravel())

        split_rng = np.random.default_rng(5000 + seed)
        experimental_indices = split_rng.permutation(cache.X_E.shape[0])
        if args.test_n + max(args.sample_sizes) > len(experimental_indices):
            raise ValueError("experimental pool is too small for disjoint train and test samples")
        test_indices = experimental_indices[:args.test_n]
        train_indices = experimental_indices[args.test_n:]
        test = generate_experimental(
            args.test_n,
            cache,
            params,
            config["dgp"],
            np.random.default_rng(9000 + seed),
            pool_indices=test_indices,
        )
        x_test = (test["X"] - x_mean) / x_std

        for n in args.sample_sizes:
            experiment = generate_experimental(
                n,
                cache,
                params,
                config["dgp"],
                np.random.default_rng(seed * 10000 + n),
                pool_indices=train_indices,
            )
            if np.intersect1d(experiment["pool_idx"], test["pool_idx"]).size:
                raise RuntimeError("experimental training and test samples overlap")
            x_experiment = (experiment["X"] - x_mean) / x_std
            treatment = experiment["T"].ravel().astype(bool)
            observed_ate = float(
                experiment["Y"].ravel()[treatment].mean()
                - experiment["Y"].ravel()[~treatment].mean()
            )

            for method, representation in fitted.items():
                if args.outcome_source == "true" and method == "raw_x_s":
                    continue
                train_encoded = representation.encode(x_experiment)
                test_encoded = representation.encode(x_test)
                cate_outcome = experiment["Y"].ravel()
                if args.outcome_source == "predicted":
                    bridge_features = (
                        x_experiment
                        if method == "raw_x"
                        else np.c_[train_encoded, experiment["S"]]
                    )
                    cate_outcome = bridges[method].predict(bridge_features)
                train_features, test_features = train_encoded, test_encoded
                if not args.no_standardize_representations:
                    train_features, test_features = scale_cate_features(
                        train_features, test_features
                    )
                tau_hat, train_tau_hat = cate_function(
                    train_features,
                    experiment["T"],
                    cate_outcome,
                    test_features,
                    seed,
                )
                raw_tau_mean = float(tau_hat.mean())
                train_tau_mean = float(train_tau_hat.mean())
                if ate_calibration == "observed":
                    tau_hat += observed_ate - train_tau_mean
                result = metrics.compute_all(tau_hat, test["tau_true"], topk_fracs=(0.2,))
                row = {
                    "dataset": dataset,
                    "seed": seed,
                    "n": n,
                    "experimental_pool_n": cache.X_E.shape[0],
                    "experimental_test_n": len(test_indices),
                    "experimental_train_pool_n": len(train_indices),
                    "train_test_overlap_n": 0,
                    "phi_dim": args.phi_dim,
                    "method": method,
                    "cate_learner": args.cate_learner,
                    "outcome_source": args.outcome_source,
                    "ate_calibration": ate_calibration,
                    "observed_ate": observed_ate,
                    "raw_tau_hat_mean": raw_tau_mean,
                    "train_tau_hat_mean": train_tau_mean,
                    "standardized_representation": not args.no_standardize_representations,
                    "cate_feature_dim": train_features.shape[1],
                    "tau_true_mean": test["tau_true"].mean(),
                    "tau_true_std": test["tau_true"].std(),
                    "tau_hat_mean": tau_hat.mean(),
                    "tau_hat_std": tau_hat.std(),
                    "pehe": result["pehe"],
                    "pehe_norm": result["pehe_norm"],
                    "pearson": result["pearson_r"],
                    "spearman": result["spearman_rho"],
                    "policy_norm_20": result["policy_norm_20"],
                }
                if args.save_vectors:
                    row["tau_hat"] = tau_hat.astype(np.float32).tolist()
                    row["tau_true"] = test["tau_true"].astype(np.float32).tolist()
                rows.append(row)
    return pd.DataFrame(rows)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--config", default="configs/semisynth.yaml")
    parser.add_argument("--label")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dgp-seed", type=int, default=0)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=[100, 250, 500, 750, 1000])
    parser.add_argument("--historical-n", type=int, default=10000)
    parser.add_argument("--test-n", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--phi-dim", type=int, default=10)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument(
        "--cate-learner",
        choices=["current_dr", "econml_dml", "econml_dr_forest"],
        default="econml_dml",
    )
    parser.add_argument("--outcome-source", choices=["predicted", "true"], default="predicted")
    parser.add_argument("--ate-calibration", choices=["none", "observed"], default="none")
    parser.add_argument("--no-standardize-representations", action="store_true")
    parser.add_argument("--save-vectors", action="store_true")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = run(args)
    if output.suffix == ".parquet":
        frame.to_parquet(output, index=False)
    else:
        frame.to_csv(output, index=False)
    print(f"wrote {len(frame)} rows to {output}")


if __name__ == "__main__":
    main()
