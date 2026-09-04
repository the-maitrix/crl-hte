from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


METHODS = [
    "mi_infonce_cond", "mi_mine_cond", "encoder_pred", "pls",
    "pca", "ica", "autoencoder", "raw_x",
]
LABELS = {
    "mi_infonce_cond": "MI-InfoNCE",
    "mi_mine_cond": "MI-MINE",
    "encoder_pred": "Encoder",
    "pls": "PLS",
    "pca": "PCA",
    "ica": "ICA",
    "autoencoder": "Autoencoder",
    "raw_x": "Raw X",
    "raw_x_s": "Raw X: h(X,S)",
}
LINESTYLES = {
    "raw_x": "--",
    "raw_x_s": "-",
}
COLORS = {
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
TEX_TO_METHOD = {
    "MI-InfoNCE": "mi_infonce_cond",
    "MI-MINE": "mi_mine_cond",
    "Encoder": "encoder_pred",
    "PLS": "pls",
    "PCA": "pca",
    "ICA": "ica",
    "Autoencoder": "autoencoder",
    "$X$-only": "raw_x",
}
SYNTHETIC_N = [100, 500, 1000, 2000, 5000]


def parse_synthetic_table(path: Path) -> pd.DataFrame:
    rows = []
    value_pattern = re.compile(r"(-?\d+\.\d+)±(\d+\.\d+)")
    for line in path.read_text().splitlines():
        if "&" not in line or "±" not in line:
            continue
        parts = [part.strip() for part in line.replace(r"\\", "").split("&")]
        method = TEX_TO_METHOD.get(parts[0])
        if method is None:
            continue
        for n, cell in zip(SYNTHETIC_N, parts[1:6]):
            match = value_pattern.search(cell)
            if match is None:
                raise ValueError(f"Cannot parse {cell!r} in {path}")
            rows.append({
                "method": method,
                "n": n,
                "mean": float(match.group(1)),
                "sem": float(match.group(2)),
            })
    frame = pd.DataFrame(rows)
    expected = len(METHODS) * len(SYNTHETIC_N)
    if len(frame) != expected:
        raise ValueError(f"Parsed {len(frame)} synthetic cells; expected {expected}")
    return frame


def summarize_frame(data: pd.DataFrame, metric: str, learner: str | None = None,
                    y_type: str | None = None) -> pd.DataFrame:
    data = data[data.method.isin(METHODS)]
    if learner is not None:
        data = data[data.learner == learner]
    if y_type is not None:
        data = data[data.y_type == y_type]
    summary = (
        data.groupby(["method", "n"])[metric]
        .agg(["mean", "sem"])
        .reset_index()
    )
    expected = len(METHODS) * data.n.nunique()
    if len(summary) != expected:
        raise ValueError(f"Summarized {len(summary)} cells; expected {expected}")
    return summary


def summarize_results(path: Path, metric: str, learner: str | None = None,
                      y_type: str | None = None) -> pd.DataFrame:
    data = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    return summarize_frame(data, metric, learner, y_type)


def apply_axis_style(axis):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.45, alpha=0.65)
    axis.tick_params(axis="both", labelsize=6, width=0.6, length=2.5)
    axis.set_axisbelow(True)


def plot_lines(axis, data, title, ylabel, ylim=None, ticks=None, constant_reference=None):
    for method in METHODS:
        subset = data[data.method == method].sort_values("n")
        x = subset.n.to_numpy()
        mean = subset["mean"].to_numpy()
        sem = subset["sem"].to_numpy()
        axis.plot(
            x, mean, marker="o", linestyle=LINESTYLES.get(method, "-"),
            color=COLORS[method], linewidth=1.1,
            markersize=2.2, markeredgewidth=0, zorder=3,
        )
        axis.fill_between(
            x, mean - sem, mean + sem,
            color=COLORS[method], alpha=0.14, linewidth=0, zorder=2,
        )
    if constant_reference is not None:
        axis.axhline(
            constant_reference, color="#222222", linestyle=(0, (3, 2)),
            linewidth=1.0, zorder=1,
        )
    axis.set_xscale("log")
    ticks = sorted(data.n.unique()) if ticks is None else ticks
    axis.set_xticks(ticks)
    axis.set_xticklabels([f"{n // 1000}k" if n >= 1000 else str(n) for n in ticks])
    axis.set_xlabel("experimental n", fontsize=6.5, labelpad=2)
    axis.set_ylabel(ylabel, fontsize=6.5, labelpad=2)
    axis.set_title(title, fontsize=7.3, pad=4, fontweight="semibold")
    if ylim is not None:
        axis.set_ylim(*ylim)
    apply_axis_style(axis)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-pehe-table", type=Path)
    parser.add_argument("--synthetic-policy-table", type=Path)
    parser.add_argument("--synthetic-results", type=Path)
    parser.add_argument("--synthetic-raw-comparison", type=Path)
    parser.add_argument("--semisynthetic-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.synthetic_raw_comparison is not None:
        if args.synthetic_results is None:
            parser.error("--synthetic-raw-comparison requires --synthetic-results")
        METHODS.append("raw_x_s")
        LABELS["raw_x"] = "Raw X: h(X)"
        synthetic_data = pd.read_csv(args.synthetic_results)
        raw_x_s = pd.read_csv(args.synthetic_raw_comparison)
        raw_x_s = raw_x_s[raw_x_s.method == "surrogate_index"].copy()
        raw_x_s["method"] = "raw_x_s"
        synthetic_data = pd.concat([synthetic_data, raw_x_s], ignore_index=True)
        synthetic_pehe = summarize_frame(
            synthetic_data, "pehe_norm", learner="dml_learner", y_type="h_y"
        )
        synthetic_policy = summarize_frame(
            synthetic_data, "policy_norm_20", learner="dml_learner", y_type="h_y"
        )
    elif args.synthetic_results is not None:
        synthetic_pehe = summarize_results(
            args.synthetic_results, "pehe_norm", learner="dml_learner", y_type="h_y"
        )
        synthetic_policy = summarize_results(
            args.synthetic_results, "policy_norm_20", learner="dml_learner", y_type="h_y"
        )
    else:
        if args.synthetic_pehe_table is None or args.synthetic_policy_table is None:
            parser.error("provide --synthetic-results or both synthetic table arguments")
        synthetic_pehe = parse_synthetic_table(args.synthetic_pehe_table)
        synthetic_policy = parse_synthetic_table(args.synthetic_policy_table)
    semisynthetic_pehe = summarize_results(args.semisynthetic_results, "pehe_norm")
    semisynthetic_policy = summarize_results(args.semisynthetic_results, "policy_norm_20")

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.35))
    plot_lines(
        axes[0], synthetic_pehe, "(a) Synthetic: PEHE", "Normalized PEHE",
        ticks=[50, 100, 250, 750],
    )
    plot_lines(
        axes[1], semisynthetic_pehe, "(b) Semi-synthetic: PEHE", "Normalized PEHE",
        ticks=[100, 250, 500, 1000],
    )
    plot_lines(
        axes[2], synthetic_policy, "(c) Synthetic: policy", "Normalized policy",
        (-0.1, 1.02), ticks=[50, 100, 250, 750],
    )
    plot_lines(
        axes[3], semisynthetic_policy, "(d) Semi-synthetic: policy", "Normalized policy",
        (-0.1, 1.02), ticks=[100, 250, 500, 1000],
    )

    handles = [
        Line2D([0], [0], color=COLORS[method], marker="o", linewidth=1.2,
               linestyle=LINESTYLES.get(method, "-"), markersize=3,
               label=LABELS[method])
        for method in METHODS
    ]
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.01),
        ncol=5 if len(METHODS) > 8 else 8, frameon=False, fontsize=6.1,
        handlelength=1.5, columnspacing=0.9, handletextpad=0.35,
    )
    top = 0.71 if len(METHODS) > 8 else 0.78
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.19, top=top, wspace=0.42)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
