from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from crl_hte import encoders
from crl_hte.semisynth import (
    SemiSynthCache,
    generate_experimental,
    generate_observational,
    init_semisynth_params,
)


DIAGNOSTIC_METHODS = (
    "encoder_pred",
    "mi_mine_cond",
    "mi_infonce_cond",
    "pls",
    "pca",
    "ica",
    "autoencoder",
)

METHOD_LABELS = {
    "encoder_pred": "Prediction encoder",
    "mi_mine_cond": "Conditional MINE",
    "mi_infonce_cond": "Conditional InfoNCE",
    "pls": "PLS",
    "pca": "PCA",
    "ica": "ICA",
    "autoencoder": "Autoencoder",
}


def encoder_config(x_dim, s_dim, phi_dim, epochs):
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


def classifier_predictions(features, outcome, seed):
    model = HistGradientBoostingClassifier(max_iter=60, random_state=seed)
    folds = StratifiedKFold(3, shuffle=True, random_state=seed)
    return cross_val_predict(model, features, outcome, cv=folds, method="predict_proba")[:, 1]


def regression_predictions(features, outcome, seed):
    model = RandomForestRegressor(
        n_estimators=80,
        min_samples_leaf=20,
        max_features=0.5,
        n_jobs=-1,
        random_state=seed,
    )
    folds = KFold(3, shuffle=True, random_state=seed)
    return cross_val_predict(model, features, outcome, cv=folds)


def draw_dag(axis):
    positions = {
        "X": (0.08, 0.66), "phi(X)": (0.34, 0.86), "T": (0.08, 0.18),
        "S": (0.55, 0.45), "Y*": (0.88, 0.45),
    }
    for label, (x, y) in positions.items():
        axis.text(x, y, label, ha="center", va="center", fontsize=11,
                  bbox=dict(boxstyle="round,pad=.35", fc="#eef3f8", ec="#345"))
    for start, end in (("X", "phi(X)"), ("X", "S"), ("X", "Y*"),
                       ("T", "S"), ("S", "Y*")):
        x1, y1 = positions[start]; x2, y2 = positions[end]
        axis.annotate("", xy=(x2, y2), xytext=(x1, y1),
                      arrowprops=dict(arrowstyle="->", color="#345", lw=1.5))
    axis.text(0.53, 0.12, "No direct T → Y* path", color="#a33", ha="center", fontsize=10)
    axis.set_title("Constructed causal structure")
    axis.set_xlim(0, 1); axis.set_ylim(0, 1); axis.axis("off")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="data/semisynth_cache")
    parser.add_argument("--config", default="configs/semisynth.yaml")
    parser.add_argument("--output-dir", default="results/corrected_semisynth_diagnostics")
    parser.add_argument("--historical-n", type=int, default=15000)
    parser.add_argument("--representation-train-n", type=int, default=10000)
    parser.add_argument("--experimental-n", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--phi-dim", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(args.config) as handle:
        config = yaml.safe_load(handle)
    cache = SemiSynthCache.load(args.cache_dir)
    params = init_semisynth_params(cache, config["dgp"], dgp_seed=0)
    historical = generate_observational(
        args.historical_n, cache, params, config["dgp"], np.random.default_rng(1000 + args.seed)
    )
    experimental = generate_experimental(
        args.experimental_n, cache, params, config["dgp"], np.random.default_rng(9000 + args.seed)
    )
    mean = cache.X_H.mean(axis=0)
    std = cache.X_H.std(axis=0).clip(1e-6)
    X_H = (historical["X"] - mean) / std
    X_E = (experimental["X"] - mean) / std
    S_H = historical["S"]
    S_E = experimental["S"]
    Y_H = historical["Y"].astype(int).ravel()
    Y_E = experimental["Y"].astype(int).ravel()
    T_E = experimental["T"].astype(int).ravel()

    if args.representation_train_n >= len(X_H):
        raise ValueError("representation-train-n must be smaller than historical-n")
    historical_train = np.arange(args.representation_train_n)
    historical_holdout = np.arange(args.representation_train_n, len(X_H))
    X_H_train, X_H_holdout = X_H[historical_train], X_H[historical_holdout]
    S_H_train, S_H_holdout = S_H[historical_train], S_H[historical_holdout]
    Y_H_train, Y_H_holdout = Y_H[historical_train], Y_H[historical_holdout]

    cfg = encoder_config(X_H.shape[1], S_H.shape[1], args.phi_dim, args.epochs)
    observed = {
        "X": X_H_train,
        "S": S_H_train,
        "Y": historical["Y"][historical_train],
    }
    fitted = {
        method: encoders.fit_method(method, observed, cfg, torch.device("cpu"), args.seed)
        for method in DIAGNOSTIC_METHODS
    }
    phi_H = {method: fit.encode(X_H_holdout) for method, fit in fitted.items()}
    phi_E = {method: fit.encode(X_E) for method, fit in fitted.items()}

    n = min(len(X_H), len(X_E))
    rng = np.random.default_rng(args.seed)
    hi = rng.choice(len(X_H), n, replace=False)
    ei = rng.choice(len(X_E), n, replace=False)
    domain_X = np.r_[X_H[hi], X_E[ei]]
    domain_y = np.r_[np.zeros(n), np.ones(n)]
    domain_model = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(max_iter=1000, C=0.1),
    )
    domain_folds = StratifiedKFold(5, shuffle=True, random_state=args.seed)
    domain_score = cross_val_predict(
        domain_model, domain_X, domain_y, cv=domain_folds, method="predict_proba"
    )[:, 1]
    domain_test_y = domain_y
    domain_auc = roc_auc_score(domain_test_y, domain_score)
    domain_overlap = {
        "historical_inside_05_95": float(
            ((domain_score[domain_test_y == 0] >= 0.05) & (domain_score[domain_test_y == 0] <= 0.95)).mean()
        ),
        "experimental_inside_05_95": float(
            ((domain_score[domain_test_y == 1] >= 0.05) & (domain_score[domain_test_y == 1] <= 0.95)).mean()
        ),
        "historical_quantiles": np.quantile(
            domain_score[domain_test_y == 0], [0.01, 0.05, 0.5, 0.95, 0.99]
        ).tolist(),
        "experimental_quantiles": np.quantile(
            domain_score[domain_test_y == 1], [0.01, 0.05, 0.5, 0.95, 0.99]
        ).tolist(),
    }

    selected = cache.sel_idx
    historical_permutation = rng.permutation(len(cache.X_H))
    reference = historical_permutation[:20000]
    historical_query = historical_permutation[20000:25000]
    experimental_query = rng.choice(len(cache.X_E), 5000, replace=False)
    selected_mean = cache.X_H[:, selected].mean(axis=0)
    selected_std = cache.X_H[:, selected].std(axis=0).clip(1e-6)
    historical_selected = (cache.X_H[:, selected] - selected_mean) / selected_std
    experimental_selected = (cache.X_E[:, selected] - selected_mean) / selected_std
    nearest = NearestNeighbors(n_neighbors=1).fit(historical_selected[reference])
    historical_distance = nearest.kneighbors(historical_selected[historical_query])[0][:, 0]
    experimental_distance = nearest.kneighbors(experimental_selected[experimental_query])[0][:, 0]
    domain_overlap.update({
        "median_experimental_to_historical_nn_ratio": float(
            np.median(experimental_distance) / np.median(historical_distance)
        ),
        "experimental_within_historical_nn_p95": float(
            (experimental_distance <= np.quantile(historical_distance, 0.95)).mean()
        ),
    })

    joint_support = {}
    historical_joint = np.c_[X_H[:, selected], S_H]
    for treatment in (0, 1):
        arm = T_E == treatment
        experimental_joint = np.c_[X_E[arm][:, selected], S_E[arm]]
        arm_n = min(len(historical_joint), len(experimental_joint))
        h_arm = rng.choice(len(historical_joint), arm_n, replace=False)
        e_arm = rng.choice(len(experimental_joint), arm_n, replace=False)
        joint_features = np.r_[historical_joint[h_arm], experimental_joint[e_arm]]
        joint_labels = np.r_[np.zeros(arm_n), np.ones(arm_n)]
        joint_train, joint_test = train_test_split(
            np.arange(len(joint_labels)), test_size=0.4,
            stratify=joint_labels, random_state=args.seed + treatment,
        )
        joint_model = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, C=0.1)
        )
        joint_model.fit(joint_features[joint_train], joint_labels[joint_train])
        joint_score = joint_model.predict_proba(joint_features[joint_test])[:, 1]
        joint_test_labels = joint_labels[joint_test]

        joint_scaler = StandardScaler().fit(historical_joint)
        historical_joint_z = joint_scaler.transform(historical_joint)
        experimental_joint_z = joint_scaler.transform(experimental_joint)
        query_n = min(3000, len(experimental_joint_z), len(historical_joint_z) - 10000)
        reference_n = min(10000, len(historical_joint_z) - query_n)
        historical_permutation = rng.permutation(len(historical_joint_z))
        reference_idx = historical_permutation[:reference_n]
        historical_query_idx = historical_permutation[reference_n:reference_n + query_n]
        experimental_query_idx = rng.choice(len(experimental_joint_z), query_n, replace=False)
        joint_nearest = NearestNeighbors(n_neighbors=1).fit(historical_joint_z[reference_idx])
        historical_joint_distance = joint_nearest.kneighbors(
            historical_joint_z[historical_query_idx]
        )[0][:, 0]
        experimental_joint_distance = joint_nearest.kneighbors(
            experimental_joint_z[experimental_query_idx]
        )[0][:, 0]
        experimental_joint_score = joint_score[joint_test_labels == 1]
        joint_support[f"treatment_{treatment}"] = {
            "domain_auc": roc_auc_score(joint_test_labels, joint_score),
            "experimental_inside_05_95": float(
                ((experimental_joint_score >= 0.05) & (experimental_joint_score <= 0.95)).mean()
            ),
            "median_experimental_to_historical_nn_ratio": float(
                np.median(experimental_joint_distance) / np.median(historical_joint_distance)
            ),
            "experimental_within_historical_nn_p95": float(
                (experimental_joint_distance <= np.quantile(historical_joint_distance, 0.95)).mean()
            ),
        }

    treatment_score = classifier_predictions(X_E, T_E, args.seed)
    treatment_auc = roc_auc_score(T_E, treatment_score)

    surrogacy_base = np.c_[X_E, S_E]
    surrogacy_plus_t = np.c_[surrogacy_base, T_E]
    base_score = classifier_predictions(surrogacy_base, Y_E, args.seed)
    treatment_added_score = classifier_predictions(surrogacy_plus_t, Y_E, args.seed)
    surrogacy_base_loss = log_loss(Y_E, base_score)
    surrogacy_t_loss = log_loss(Y_E, treatment_added_score)

    bridge = HistGradientBoostingClassifier(max_iter=80, random_state=args.seed)
    bridge.fit(
        np.c_[X_H_train, S_H_train], Y_H_train
    )
    bridge_h_score = bridge.predict_proba(
        np.c_[X_H_holdout, S_H_holdout]
    )[:, 1]
    controls = T_E == 0
    bridge_e_score = bridge.predict_proba(np.c_[X_E[controls], S_E[controls]])[:, 1]
    comparability = {
        "historical_auc": roc_auc_score(Y_H_holdout, bridge_h_score),
        "experimental_control_auc": roc_auc_score(Y_E[controls], bridge_e_score),
        "historical_logloss": log_loss(Y_H_holdout, bridge_h_score),
        "experimental_control_logloss": log_loss(Y_E[controls], bridge_e_score),
        "historical_calibration_bias": float(
            bridge_h_score.mean() - Y_H_holdout.mean()
        ),
        "experimental_control_calibration_bias": float(bridge_e_score.mean() - Y_E[controls].mean()),
    }

    sufficiency_i = {}
    sufficiency_ii_historical_proxy = {}
    sufficiency_ii_potential = {}
    for method in fitted:
        restricted_h = np.c_[phi_H[method], S_H_holdout]
        augmented_h = np.c_[restricted_h, X_H_holdout]
        restricted_score = classifier_predictions(restricted_h, Y_H_holdout, args.seed)
        augmented_score = classifier_predictions(augmented_h, Y_H_holdout, args.seed)
        sufficiency_i[method] = {
            "restricted_logloss": log_loss(Y_H_holdout, restricted_score),
            "augmented_logloss": log_loss(Y_H_holdout, augmented_score),
            "incremental_logloss": log_loss(Y_H_holdout, restricted_score) - log_loss(Y_H_holdout, augmented_score),
            "restricted_auc": roc_auc_score(Y_H_holdout, restricted_score),
            "augmented_auc": roc_auc_score(Y_H_holdout, augmented_score),
        }

        restricted_h_s = phi_H[method]
        augmented_h_s = np.c_[restricted_h_s, X_H_holdout]
        restricted_s = regression_predictions(restricted_h_s, S_H_holdout, args.seed)
        augmented_s = regression_predictions(augmented_h_s, S_H_holdout, args.seed)
        sufficiency_ii_historical_proxy[method] = {
            "restricted_r2": r2_score(S_H_holdout, restricted_s, multioutput="variance_weighted"),
            "augmented_r2": r2_score(S_H_holdout, augmented_s, multioutput="variance_weighted"),
            "incremental_r2": (
                r2_score(S_H_holdout, augmented_s, multioutput="variance_weighted")
                - r2_score(S_H_holdout, restricted_s, multioutput="variance_weighted")
            ),
        }

        sufficiency_ii_potential[method] = {}
        for treatment, potential_s in ((0, experimental["S_0"]), (1, experimental["S_1"])):
            restricted_e_s = regression_predictions(
                phi_E[method], potential_s, args.seed + treatment
            )
            augmented_e_s = regression_predictions(
                np.c_[phi_E[method], X_E], potential_s, args.seed + treatment
            )
            restricted_r2 = r2_score(
                potential_s, restricted_e_s, multioutput="variance_weighted"
            )
            augmented_r2 = r2_score(
                potential_s, augmented_e_s, multioutput="variance_weighted"
            )
            sufficiency_ii_potential[method][f"treatment_{treatment}"] = {
                "restricted_r2": restricted_r2,
                "augmented_r2": augmented_r2,
                "incremental_r2": augmented_r2 - restricted_r2,
            }

    metrics = {
        "phi_dim": args.phi_dim,
        "historical_representation_train_n": len(historical_train),
        "historical_diagnostic_holdout_n": len(historical_holdout),
        "exact_by_construction": {
            "treatment_randomization": True,
            "no_direct_treatment_to_target_path": True,
            "shared_historical_experimental_outcome_equation": True,
        },
        "treatment_fraction": float(T_E.mean()),
        "treatment_predictability_auc": treatment_auc,
        "domain_predictability_auc": domain_auc,
        "cross_sample_overlap": domain_overlap,
        "joint_support_by_treatment": joint_support,
        "surrogacy": {
            "without_treatment_logloss": surrogacy_base_loss,
            "with_treatment_logloss": surrogacy_t_loss,
            "treatment_incremental_logloss": surrogacy_base_loss - surrogacy_t_loss,
            "without_treatment_auc": roc_auc_score(Y_E, base_score),
            "with_treatment_auc": roc_auc_score(Y_E, treatment_added_score),
        },
        "comparability": comparability,
        "sufficiency_i": sufficiency_i,
        "sufficiency_ii_historical_proxy": sufficiency_ii_historical_proxy,
        "sufficiency_ii_potential": sufficiency_ii_potential,
    }

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))

    rows = []
    for method, values in sufficiency_i.items():
        rows.append({"assumption": "sufficiency_i", "method": method, **values})
    for method, values in sufficiency_ii_historical_proxy.items():
        rows.append({"assumption": "sufficiency_ii_historical_proxy", "method": method, **values})
    for method, by_treatment in sufficiency_ii_potential.items():
        for treatment, values in by_treatment.items():
            rows.append({
                "assumption": "sufficiency_ii_potential",
                "method": method,
                "treatment": treatment,
                **values,
            })
    pd.DataFrame(rows).to_csv(output / "sufficiency_metrics.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    draw_dag(axes[0, 0])

    axes[0, 1].hist(domain_score[domain_test_y == 0], bins=25, alpha=0.6, label="H")
    axes[0, 1].hist(domain_score[domain_test_y == 1], bins=25, alpha=0.6, label="E")
    axes[0, 1].set_title(
        f"Cross-sample X overlap: domain AUC={domain_auc:.3f}\n"
        f"E propensity overlap={domain_overlap['experimental_inside_05_95']:.1%}; "
        f"joint NN coverage T=0/1: "
        f"{joint_support['treatment_0']['experimental_within_historical_nn_p95']:.1%}/"
        f"{joint_support['treatment_1']['experimental_within_historical_nn_p95']:.1%}"
    )
    axes[0, 1].set_xlabel("Estimated P(E | X)"); axes[0, 1].legend()

    axes[0, 2].bar(["T from X", "Y from X,S", "Y from X,S,T"],
                   [treatment_auc, roc_auc_score(Y_E, base_score), roc_auc_score(Y_E, treatment_added_score)],
                   color=["#779", "#4a8", "#c77"])
    axes[0, 2].axhline(0.5, color="black", ls="--", lw=1)
    axes[0, 2].set_ylim(0.45, 1.0)
    axes[0, 2].set_title(f"Randomization and surrogacy\nΔ log loss from T={surrogacy_base_loss-surrogacy_t_loss:.4f}")

    axes[1, 0].bar(
        ["H holdout", "E controls"],
        [comparability["historical_logloss"], comparability["experimental_control_logloss"]],
        color=["#4a8", "#d98"],
    )
    axes[1, 0].set_title("Outcome model across cohorts\n(lower log loss is better)")
    axes[1, 0].set_ylabel("Log loss")

    methods = list(fitted)
    method_labels = [METHOD_LABELS[method] for method in methods]
    axes[1, 1].bar(
        method_labels, [sufficiency_i[m]["incremental_logloss"] for m in methods]
    )
    axes[1, 1].axhline(0, color="black", lw=1)
    axes[1, 1].set_title("Sufficiency (i): gain from adding X\nto (phi, S); smaller is better")
    axes[1, 1].set_ylabel("Restricted − augmented log loss")
    axes[1, 1].tick_params(axis="x", rotation=15)

    x = np.arange(len(methods)); width = 0.35
    axes[1, 2].bar(
        x - width / 2,
        [sufficiency_ii_potential[m]["treatment_0"]["incremental_r2"] for m in methods],
        width,
        label="S(0)",
    )
    axes[1, 2].bar(
        x + width / 2,
        [sufficiency_ii_potential[m]["treatment_1"]["incremental_r2"] for m in methods],
        width,
        label="S(1)",
    )
    axes[1, 2].set_xticks(x, method_labels, rotation=25, ha="right")
    axes[1, 2].set_title("Sufficiency (ii): gain from adding X to phi")
    axes[1, 2].set_ylabel("Increase in cross-validated R²"); axes[1, 2].legend()

    fig.tight_layout()
    fig.savefig(output / "assumption_diagnostics.png", dpi=180)
    fig.savefig(output / "assumption_diagnostics.pdf", bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(metrics, indent=2))
    print(output)


if __name__ == "__main__":
    main()
