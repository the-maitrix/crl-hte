"""Build the semi-synthetic cache from real UPMC perinatal CSVs.

Run once via sbatch (scripts/cluster/build_semisynth.sbatch). Reads three
CSVs from `data/all_patients/`, builds the H/E cohorts, picks X (common
features), constructs S = PCA(top-K MHP-extras), fits p̂(S|X) and the
baseline μ₀(X, S) → P(Y=1) logistic, and writes everything to one .npz.

Output: data/semisynth_cache/cache.npz — see SemiSynthCache.load().
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from crl_hte.semisynth import diagnosis_prevalence_audit, split_mhp_cohorts


def _load_cohorts(cfg: dict) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list, list]:
    """Read the comprehensive CSV and the EHR/MHP headers; return df, masks, headers."""
    root = Path(cfg["data"]["root"])
    print(f"[load] reading {cfg['data']['comprehensive_csv']} ...")
    df = pd.read_csv(root / cfg["data"]["comprehensive_csv"])
    ehr_cols = pd.read_csv(root / cfg["data"]["ehr_csv"], nrows=0).columns.tolist()
    mhp_cols = pd.read_csv(root / cfg["data"]["mhp_csv"], nrows=0).columns.tolist()

    status = df[cfg["cohort"]["mhp_status_col"]]
    include_pending = bool(cfg["cohort"].get("include_pending_in_historical", False))
    h_mask, e_mask, pending_mask, no_mhp_mask = split_mhp_cohorts(
        status, include_pending=include_pending
    )

    if include_pending and pending_mask.any():
        audit = diagnosis_prevalence_audit(
            df,
            status,
            prefixes=cfg["cohort"].get("diagnosis_feature_prefixes", ()),
            min_reference_rate=float(
                cfg["cohort"].get("diagnosis_min_reference_rate", 0.01)
            ),
        )
        ratio = audit["ratio"]
        minimum = float(
            cfg["cohort"].get("min_pending_diagnosis_prevalence_ratio", 0.5)
        )
        print(
            f"[load] pending diagnosis prevalence ratio={ratio:.3f} "
            f"across {len(audit['columns'])} features"
        )
        if not np.isfinite(ratio) or ratio < minimum:
            raise ValueError(
                "Diagnosis/problem-list features are implausibly suppressed for "
                "MHP-prescribed but not onboarded patients. Rebuild the processed "
                "CSVs from the source pregnancy and child exports."
            )

    print(
        f"[load] cohorts: H (not onboarded)={h_mask.sum()} "
        f"[no MHP={no_mhp_mask.sum()}, pending={pending_mask.sum()}]  "
        f"E (Done)={e_mask.sum()}"
    )
    return df, h_mask, e_mask, ehr_cols, mhp_cols


def _select_x_columns(df, h_mask, e_mask, ehr_cols, mhp_cols, cfg,
                      exclude_columns=()) -> list[str]:
    """Common-feature X = (ehr ∩ mhp) − ids − outcomes; missingness-filtered."""
    drop = set(cfg["cohort"]["id_cols"]
               + cfg["cohort"]["drop_outcome_cols"]
               + [cfg["cohort"]["mhp_status_col"]])
    excluded = set(exclude_columns)
    common = [
        c for c in ehr_cols
        if c in mhp_cols and c not in drop and c not in excluded and c in df.columns
    ]
    if cfg["features"]["drop_missing_indicators"]:
        common = [c for c in common if not c.endswith("_missing")]
    leak_pats = cfg["features"].get("drop_patterns", []) or []
    if leak_pats:
        before = len(common)
        common = [c for c in common if not any(p in c for p in leak_pats)]
        print(f"[X] leakage filter (patterns={leak_pats}): {before} → {len(common)}")

    y = df[cfg["cohort"]["outcome_col"]].fillna(0).astype(float).to_numpy()
    keep_mask = h_mask | e_mask
    tau_drop = float(cfg["features"]["tau_drop"])
    tau_inform = float(cfg["features"]["tau_inform"])

    kept, dropped = [], 0
    for c in common:
        col = df[c]
        miss = col[keep_mask].isna().mean()
        if miss <= tau_drop:
            kept.append(c)
            continue
        ind = col.isna().astype(float).to_numpy()
        if ind[keep_mask].std() < 1e-9:
            dropped += 1
            continue
        if abs(np.corrcoef(ind[keep_mask], y[keep_mask])[0, 1]) > tau_inform:
            kept.append(c)
        else:
            dropped += 1
    print(f"[X] kept {len(kept)} / {len(common)} common cols  (dropped {dropped} for missingness)")
    return kept


def _build_S(df, e_mask, mhp_cols, ehr_cols, drop_set, cfg) -> tuple[np.ndarray, list, dict]:
    """Pick top-K MHP-only cols by |corr(., Y)| on E, standardize, PCA → S.

    If `surrogate.include_common_patterns` is non-empty, also pull common-feature
    columns (mhp ∩ ehr) whose names match any of the substrings into the
    candidate pool. They are still treated as part of S for transport purposes.
    """
    extras = [c for c in mhp_cols
              if c not in ehr_cols and not c.endswith("_missing")
              and c in df.columns and c not in drop_set]
    print(f"[S] {len(extras)} candidate MHP-only columns")

    include_pats = cfg["surrogate"].get("include_common_patterns") or []
    if include_pats:
        # Re-included common features must still respect drop_patterns *minus*
        # the include patterns. So `Depression` is unblocked in S, but
        # `Postpartum` (a separate hard-block pattern) keeps `Depression_Postpartum`
        # — which is the actual PPD label — out of S.
        all_drop = cfg["features"].get("drop_patterns") or []
        active_drop = [p for p in all_drop if p not in include_pats]
        common_extras = [c for c in mhp_cols
                         if c in ehr_cols and not c.endswith("_missing")
                         and c in df.columns and c not in drop_set
                         and any(p in c for p in include_pats)
                         and not any(p in c for p in active_drop)]
        print(f"[S] include_common_patterns={include_pats} (still blocking {active_drop}) → "
              f"{len(common_extras)} additional common-feature columns")
        extras = extras + common_extras

    e_idx = np.where(e_mask)[0]
    y_E = df.loc[e_idx, cfg["cohort"]["outcome_col"]].fillna(0).astype(float).to_numpy()
    extras_E = df.loc[e_idx, extras].fillna(0).astype(np.float32).to_numpy()

    corrs = np.zeros(len(extras), dtype=float)
    for j in range(len(extras)):
        col = extras_E[:, j]
        if col.std() > 1e-9:
            corrs[j] = abs(np.corrcoef(col, y_E)[0, 1])

    K = int(cfg["surrogate"]["n_extras_for_pca"])
    top = np.argsort(corrs)[::-1][:K]
    sel_names = [extras[i] for i in top]
    print(f"[S] top-{K} extras by |corr(., Y)|:")
    for name, c in sorted(zip(sel_names, corrs[top]), key=lambda kv: -kv[1]):
        print(f"      {c:.4f}  {name}")

    sel = extras_E[:, top]
    if cfg["surrogate"]["standardize"]:
        scaler = StandardScaler().fit(sel)
        sel_z = scaler.transform(sel)
        ext_mean, ext_std = scaler.mean_, scaler.scale_
    else:
        sel_z = sel
        ext_mean, ext_std = np.zeros(K), np.ones(K)

    s_dim = int(cfg["surrogate"]["s_dim"])
    pca = PCA(n_components=s_dim, random_state=0).fit(sel_z)
    S_E = pca.transform(sel_z).astype(np.float32)
    print(f"[S] PCA explained-variance ratio: {np.round(pca.explained_variance_ratio_, 3)}  "
          f"(sum={pca.explained_variance_ratio_.sum():.3f})")

    pca_artifacts = dict(
        ext_mean=ext_mean.astype(np.float32),
        ext_std=ext_std.astype(np.float32),
        pca_mean=pca.mean_.astype(np.float32),
        pca_components=pca.components_.astype(np.float32),
    )
    return S_E, sel_names, pca_artifacts


def _build_common_diagnosis_S(
    df, h_mask, e_mask, ehr_cols, mhp_cols, cfg
) -> tuple[np.ndarray, np.ndarray, list, dict]:
    common = set(ehr_cols) & set(mhp_cols) & set(df.columns)
    prefixes = tuple(cfg["surrogate"]["diagnosis_feature_prefixes"])
    timepoints = tuple(cfg["surrogate"]["diagnosis_timepoints"])
    candidates = sorted(
        c for c in common
        if c.startswith(prefixes)
        and any(timepoint in c for timepoint in timepoints)
        and not c.endswith("_missing")
    )
    values = df[candidates].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    prevalence = values.loc[h_mask].ne(0).mean(axis=0)
    minimum = float(cfg["surrogate"]["min_prevalence"])
    maximum = float(cfg["surrogate"]["max_prevalence"])
    selected = prevalence[(prevalence >= minimum) & (prevalence <= maximum)].index.tolist()
    if len(selected) < int(cfg["surrogate"]["s_dim"]):
        raise ValueError(
            f"Only {len(selected)} common diagnosis outcomes passed prevalence filters"
        )

    historical = values.loc[h_mask, selected].to_numpy(dtype=np.float32)
    experimental = values.loc[e_mask, selected].to_numpy(dtype=np.float32)
    if cfg["surrogate"]["standardize"]:
        scaler = StandardScaler().fit(historical)
        historical_z = scaler.transform(historical)
        experimental_z = scaler.transform(experimental)
        ext_mean, ext_std = scaler.mean_, scaler.scale_
    else:
        historical_z, experimental_z = historical, experimental
        ext_mean, ext_std = np.zeros(len(selected)), np.ones(len(selected))

    pca = PCA(
        n_components=int(cfg["surrogate"]["s_dim"]), random_state=0
    ).fit(historical_z)
    S_H = pca.transform(historical_z).astype(np.float32)
    S_E = pca.transform(experimental_z).astype(np.float32)
    print(
        f"[S] common diagnosis outcomes: {len(candidates)} candidates, "
        f"{len(selected)} retained"
    )
    print(
        f"[S] historical PCA explained-variance ratio: "
        f"{np.round(pca.explained_variance_ratio_, 3)} "
        f"(sum={pca.explained_variance_ratio_.sum():.3f})"
    )
    artifacts = dict(
        ext_mean=np.asarray(ext_mean, dtype=np.float32),
        ext_std=np.asarray(ext_std, dtype=np.float32),
        pca_mean=pca.mean_.astype(np.float32),
        pca_components=pca.components_.astype(np.float32),
    )
    return S_H, S_E, selected, artifacts


def _fit_pSX(X_E, S_E, alpha) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Linear-Gaussian p̂(S|X) via ridge on E."""
    ridge = Ridge(alpha=alpha).fit(X_E, S_E)
    pred = ridge.predict(X_E)
    sigma = (S_E - pred).std(axis=0).astype(np.float32)
    r2 = 1.0 - ((S_E - pred) ** 2).sum() / ((S_E - S_E.mean(0)) ** 2).sum()
    print(f"[p(S|X)] ridge fit on E: R²={r2:.3f}, residual σ per dim={np.round(sigma, 3)}")
    return ridge.coef_.astype(np.float32), ridge.intercept_.astype(np.float32), sigma, r2


def _select_features_l1(X, y, names, K) -> tuple[np.ndarray, np.ndarray]:
    """Pick top-K X columns by |β| from a sparse L1-logistic fit.

    Sweeps a few C values until ≥ K coefficients are nonzero, then keeps the
    K largest in absolute value. Returns the selected indices and the L1
    coefficients on those (signed, for the sanity-check print).
    """
    best, best_C = None, None
    for C in (0.005, 0.01, 0.02, 0.05, 0.1, 0.5):
        lr = LogisticRegression(penalty="l1", solver="saga", C=C,
                                max_iter=2000, n_jobs=-1).fit(X, y)
        nnz = int((lr.coef_[0] != 0).sum())
        best, best_C = (lr, nnz), C
        if nnz >= K:
            break
    lr, nnz = best
    coefs = lr.coef_[0]
    order = np.argsort(np.abs(coefs))[::-1]
    sel = np.sort(order[:K])
    print(f"[select] L1 logistic at C={best_C} → {nnz} nonzero, keeping top {K} by |β|")
    return sel, coefs[sel]


def _print_signed_coefs(names, coefs, header):
    """Sort by signed coef (positive risk first), tabulate. Sanity-check signs."""
    pairs = sorted(zip(coefs, names), key=lambda kv: -kv[0])
    print(f"[{header}] signed coefficients (positive = risk-increasing):")
    for c, n in pairs:
        tag = "  (PROTECTIVE)" if c < 0 else ""
        print(f"      {c:+.4f}  {n}{tag}")


def _fit_muX(X, y, C, max_iter, cv) -> tuple[np.ndarray, float, float, np.ndarray]:
    """L2-logistic μ_X(X) → P(Y=1). Reports CV AUC + train AUC.
    Returns (β_x, b_x, cv_auc, β_x_signed_for_print)."""
    lr = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs").fit(X, y)
    train_auc = roc_auc_score(y, lr.predict_proba(X)[:, 1])
    cv_auc = float(cross_val_score(
        LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs"),
        X, y, cv=cv, scoring="roc_auc", n_jobs=-1).mean())
    print(f"[μ_X] L2-logistic(X→Y): train AUC={train_auc:.3f}, {cv}-fold CV AUC={cv_auc:.3f}")
    return lr.coef_[0].astype(np.float32), float(lr.intercept_[0]), cv_auc, lr.coef_[0]


def _fit_betaS(S, y, C, max_iter, cv) -> tuple[np.ndarray, float]:
    """L2-logistic(S → Y, no intercept). Reports CV AUC.
    No intercept so β_S captures purely the surrogate-Y direction."""
    lr = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs",
                            fit_intercept=False).fit(S, y)
    cv_auc = float(cross_val_score(
        LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs", fit_intercept=False),
        S, y, cv=cv, scoring="roc_auc", n_jobs=-1).mean())
    print(f"[β_S] L2-logistic(S→Y): {cv}-fold CV AUC={cv_auc:.3f}, "
          f"||β_S||={np.linalg.norm(lr.coef_[0]):.3f}, β_S={np.round(lr.coef_[0], 3)}")
    return lr.coef_[0].astype(np.float32), cv_auc


def main(cfg_path: str) -> None:
    cfg = yaml.safe_load(open(cfg_path))
    cache_dir = Path(cfg["data"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    df, h_mask, e_mask, ehr_cols, mhp_cols = _load_cohorts(cfg)

    surrogate_source = str(cfg["surrogate"].get("source", "mhp_extras"))
    if surrogate_source == "common_diagnoses":
        S_H, S_E, extras_names, pca_art = _build_common_diagnosis_S(
            df, h_mask, e_mask, ehr_cols, mhp_cols, cfg
        )
    else:
        S_H = S_E = extras_names = pca_art = None

    x_pre = _select_x_columns(
        df, h_mask, e_mask, ehr_cols, mhp_cols, cfg,
        exclude_columns=extras_names or (),
    )
    overlap = set(x_pre) & set(extras_names or ())
    if overlap:
        raise ValueError(f"Auxiliary-outcome source columns leaked into X: {sorted(overlap)}")
    X_pre = df[x_pre].fillna(0).astype(np.float32).to_numpy()
    y_full = df[cfg["cohort"]["outcome_col"]].fillna(0).astype(float).to_numpy()
    fit_mask = h_mask if surrogate_source == "common_diagnoses" else h_mask | e_mask
    y_fit = (y_full[fit_mask] > 0.5).astype(int)
    X_fit = X_pre[fit_mask]
    print(f"[X] pre-select: p={len(x_pre)}, "
          f"prevalence H={y_full[h_mask].mean():.3f} E={y_full[e_mask].mean():.3f}")

    K = int(cfg["features"]["n_features"])
    sel_idx, sel_l1 = _select_features_l1(X_fit, y_fit, x_pre, K)
    sel_names = [x_pre[i] for i in sel_idx]
    _print_signed_coefs(sel_names, sel_l1, "L1-selected historical features")

    X_H = X_pre[h_mask]
    X_E = X_pre[e_mask]
    y_H = y_full[h_mask].astype(np.float32)
    y_E = y_full[e_mask].astype(np.float32)
    print(f"[X] full X retained: shape (H,E)=({X_H.shape}, {X_E.shape})  "
          f"sel_idx covers {len(sel_idx)} dims")

    by = cfg["baseline_y"]
    muX_w, muX_b, _, _ = _fit_muX(
        X_H[:, sel_idx], (y_H > 0.5).astype(int),
        by["C_mu_x"], by["max_iter"], by["cv_folds"]
    )

    if surrogate_source == "common_diagnoses":
        pSX_W = np.empty((0, 0), dtype=np.float32)
        pSX_b = np.empty(0, dtype=np.float32)
        pSX_sigma = np.empty(0, dtype=np.float32)
        betaS, _ = _fit_betaS(
            S_H, (y_H > 0.5).astype(int),
            by["C_beta_s"], by["max_iter"], by["cv_folds"]
        )
    elif surrogate_source == "mhp_extras":
        drop_set = set(
            cfg["cohort"]["id_cols"] + cfg["cohort"]["drop_outcome_cols"]
            + [cfg["cohort"]["mhp_status_col"]]
        )
        S_E, extras_names, pca_art = _build_S(
            df, e_mask, mhp_cols, ehr_cols, drop_set, cfg
        )
        S_H = None
        pSX_W, pSX_b, pSX_sigma, _ = _fit_pSX(
            X_E[:, sel_idx], S_E, float(cfg["p_s_given_x"]["ridge_alpha"])
        )
        betaS, _ = _fit_betaS(
            S_E, (y_E > 0.5).astype(int),
            by["C_beta_s"], by["max_iter"], by["cv_folds"]
        )
    else:
        raise ValueError(f"Unknown surrogate source: {surrogate_source}")

    out = cache_dir / "cache.npz"
    payload = dict(
        X_H=X_H, X_E=X_E, y_H=y_H, y_E=y_E, S_E=S_E,
        feat_names=np.array(x_pre),
        sel_idx=sel_idx.astype(np.int32),
        sel_names=np.array(sel_names),
        extras_names=np.array(extras_names),
        pSX_W=pSX_W, pSX_b=pSX_b, pSX_sigma=pSX_sigma,
        muX_w=muX_w, muX_b=np.float32(muX_b),
        betaS=betaS, lambda_s=np.float32(by["lambda_s"]),
        surrogate_source=np.array(surrogate_source),
        **pca_art,
    )
    if S_H is not None:
        payload["S_H"] = S_H
    np.savez(out, **payload)
    print(f"[done] wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    main(p.parse_args().config)
