from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


METHODS = [
    "mi_infonce_cond", "mi_mine_cond", "encoder_pred",
    "pls", "pca", "ica", "autoencoder", "raw_x", "raw_x_s",
]
LABELS = {
    "mi_infonce_cond": "MI-InfoNCE",
    "mi_mine_cond": "MI-MINE",
    "encoder_pred": "Encoder",
    "pls": "PLS",
    "pca": "PCA",
    "ica": "ICA",
    "autoencoder": "Autoencoder",
    "raw_x": "Raw X: h(X)",
    "raw_x_s": "Raw X: h(X,S)",
}
COLORS = {
    # Shared methods use exactly the same palette as the main figure.
    "mi_infonce_cond": "#6baed6",
    "mi_mine_cond": "#08519c",
    "encoder_pred": "#3182bd",
    "pls": "#17becf",
    "pca": "#31a354",
    "ica": "#006d2c",
    "autoencoder": "#a1d99b",
    "raw_x": "#7f7f7f",
    "raw_x_s": "#252525",
}
LINESTYLES = {"raw_x": "--", "raw_x_s": "-"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    data = pd.read_parquet(args.input) if Path(args.input).suffix == ".parquet" else pd.read_csv(args.input)
    data = data[data.method.isin(METHODS)].copy()
    labels = dict(LABELS)
    if "outcome_source" in data and set(data.outcome_source.astype(str).str.lower()) == {"true"}:
        labels["raw_x"] = "Raw X"
    summary = (
        data.groupby(["method", "n"])[["pehe", "pehe_norm", "spearman", "policy_norm_20"]]
        .agg(["mean", "sem"])
        .reset_index()
    )
    summary.columns = [
        "method", "n",
        "pehe_mean", "pehe_sem",
        "pehe_norm_mean", "pehe_norm_sem",
        "spearman_mean", "spearman_sem",
        "policy_norm_20_mean", "policy_norm_20_sem",
    ]

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "fresh_grid_summary.csv", index=False)

    raw = data.loc[data.method == "raw_x", ["seed", "n", "pehe", "policy_norm_20"]].rename(
        columns={"pehe": "raw_pehe", "policy_norm_20": "raw_policy_norm_20"}
    )
    paired = data.loc[data.method != "raw_x"].merge(raw, on=["seed", "n"])
    paired["pehe_difference_vs_raw"] = paired.pehe - paired.raw_pehe
    paired["policy_difference_vs_raw"] = paired.policy_norm_20 - paired.raw_policy_norm_20
    paired_summary = paired.groupby(["method", "n"]).agg(
        pehe_difference_mean=("pehe_difference_vs_raw", "mean"),
        pehe_difference_sem=("pehe_difference_vs_raw", "sem"),
        pehe_win_fraction=("pehe_difference_vs_raw", lambda values: (values < 0).mean()),
        policy_difference_mean=("policy_difference_vs_raw", "mean"),
        policy_difference_sem=("policy_difference_vs_raw", "sem"),
        policy_win_fraction=("policy_difference_vs_raw", lambda values: (values > 0).mean()),
    ).reset_index()
    paired_summary.to_csv(output / "paired_comparisons_vs_raw.csv", index=False)

    methods_present = [method for method in METHODS if method in set(data.method)]
    phi_dim = int(data.phi_dim.iloc[0]) if "phi_dim" in data else None
    panels = [
        ("pehe_norm", "Normalized PEHE"),
        ("spearman", "Spearman correlation"),
        ("policy_norm_20", "Normalized policy value, top 20%"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, (metric, label) in zip(axes, panels):
        for method in methods_present:
            subset = summary[summary.method == method].sort_values("n")
            axis.errorbar(
                subset.n,
                subset[f"{metric}_mean"],
                yerr=subset[f"{metric}_sem"],
                marker="o",
                linewidth=1.8,
                capsize=3,
                label=labels[method],
                color=COLORS[method],
                linestyle=LINESTYLES.get(method, "-"),
            )
        axis.set_xscale("log")
        axis.set_xlabel("Experimental sample size")
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=8)
    suffix = f", phi dimension={phi_dim}" if phi_dim is not None else ""
    fig.suptitle(f"Perinatal semi-synthetic dataset — cross-fitted DML{suffix}")
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.savefig(output / "fresh_grid_results.png", dpi=180)
    plt.close(fig)
    print(summary.to_string(index=False))
    print(output)


if __name__ == "__main__":
    main()
