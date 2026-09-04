from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


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
}
LEARNERS = {"x_learner": "xl", "dml_learner": "dml"}
Y_TYPES = {"h_y": "predictedy", "true_y": "truey"}
METRICS = {
    "pehe_norm": ("Normalized PEHE", "pehe_norm"),
    "policy_norm_20": ("Normalized policy value (top 20%)", "policy_norm_20"),
}
EXPECTED_N = [50, 100, 250, 500, 750]


def read_frame(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def validate(data: pd.DataFrame, expected_runs: int, axis: str | None = None,
             max_labeled_failures: int = 0):
    columns = {"method", "n", "learner", "y_type", "trial_seed", "phi_dim"}
    columns.update(METRICS)
    if axis:
        columns.add(axis)
    missing = columns - set(data.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if axis:
        expected_learners = [learner for learner in LEARNERS if learner in data.learner.unique()]
        if not expected_learners:
            raise ValueError("violation sweep contains no supported learner")
    else:
        expected_learners = list(LEARNERS)
    subset = data[
        data.method.isin(METHODS)
        & data.learner.isin(expected_learners)
        & data.y_type.isin(Y_TYPES)
    ]
    if axis != "phi_dim":
        subset = subset[subset.phi_dim == 10]
    if sorted(int(n) for n in subset.n.unique()) != EXPECTED_N:
        raise ValueError(f"expected n={EXPECTED_N}; found {sorted(subset.n.unique())}")
    if "dgp_seed" in subset and sorted(subset.dgp_seed.unique()) != [42]:
        raise ValueError(f"expected dgp_seed=[42]; found {sorted(subset.dgp_seed.unique())}")
    keys = ["method", "learner", "y_type", "n"] + ([axis] if axis else [])
    counts = subset.groupby(keys).trial_seed.nunique()
    axis_count = subset[axis].nunique() if axis else 1
    expected_cells = len(METHODS) * len(expected_learners) * len(Y_TYPES) * len(EXPECTED_N) * axis_count
    if len(counts) != expected_cells:
        raise ValueError(f"found {len(counts)} cells; expected {expected_cells}")
    bad = counts[counts != expected_runs]
    if not bad.empty:
        raise ValueError(f"cells do not have {expected_runs} runs:\n{bad.to_string()}")
    if subset.duplicated(keys + ["trial_seed"]).any():
        raise ValueError(f"duplicate rows for {keys + ['trial_seed']}")
    failed = subset[subset[list(METRICS)].isna().any(axis=1)]
    if len(failed) > max_labeled_failures:
        raise ValueError(
            f"{len(failed)} rows have missing paper metrics "
            f"(allowed: {max_labeled_failures})"
        )
    if not failed.empty:
        if "error" in failed and failed["error"].isna().any():
            raise ValueError("rows with missing metrics lack a recorded error label")
        complete = subset.dropna(subset=list(METRICS)).groupby(keys).trial_seed.nunique()
        if int(complete.min()) < expected_runs - 1:
            raise ValueError("a cell has more than one failed run")
        print(f"[validate] {len(failed)} labeled failed fits excluded from summaries")


def style_axis(axis):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.45, alpha=0.65)
    axis.tick_params(axis="both", labelsize=7, width=0.6, length=2.5)
    axis.set_axisbelow(True)


def summarize(data: pd.DataFrame, metric: str, keys: list[str]) -> pd.DataFrame:
    return data.groupby(keys)[metric].agg(["mean", "sem"]).reset_index()


def plot_main_panels(data: pd.DataFrame, output: Path):
    subset = data[
        data.method.isin(METHODS)
        & data.learner.isin(LEARNERS)
        & data.y_type.isin(Y_TYPES)
        & (data.phi_dim == 10)
    ]
    for metric, (ylabel, metric_slug) in METRICS.items():
        summary = summarize(subset, metric, ["method", "learner", "y_type", "n"])
        for learner, learner_slug in LEARNERS.items():
            for y_type, y_slug in Y_TYPES.items():
                cell = summary[(summary.learner == learner) & (summary.y_type == y_type)]
                fig, axis = plt.subplots(figsize=(4.15, 3.65))
                for method in METHODS:
                    rows = cell[cell.method == method].sort_values("n")
                    axis.plot(rows.n, rows["mean"], "-o", color=COLORS[method], lw=1.3, ms=3,
                              label=LABELS[method])
                    axis.fill_between(rows.n, rows["mean"] - rows["sem"], rows["mean"] + rows["sem"],
                                      color=COLORS[method], alpha=0.16, linewidth=0)
                axis.set_xscale("log")
                axis.set_xticks(EXPECTED_N, [str(n) for n in EXPECTED_N])
                axis.set_xlabel("experimental sample size $n$")
                axis.set_ylabel(ylabel)
                axis.legend(ncol=2, frameon=False, fontsize=7, loc="best")
                style_axis(axis)
                fig.tight_layout()
                fig.savefig(output / f"{metric_slug}_phi10_{learner_slug}_{y_slug}.pdf",
                            bbox_inches="tight")
                plt.close(fig)


AXIS_XLABELS = {
    "alpha": r"$\alpha$",
    "delta": r"$\delta$",
    "phi_dim": r"$m=\dim(\phi)$",
}


def plot_violation_rows(data: pd.DataFrame, axis_name: str, output: Path,
                        shown_n: list[int], learner: str = "x_learner"):
    subset = data[
        data.method.isin(METHODS)
        & (data.learner == learner)
        & data.y_type.isin(Y_TYPES)
    ]
    if axis_name != "phi_dim":
        subset = subset[subset.phi_dim == 10]
    axis_values = sorted(subset[axis_name].unique())
    for metric, (ylabel, metric_slug) in METRICS.items():
        summary = summarize(subset, metric, ["method", "y_type", "n", axis_name])
        for y_type, y_slug in Y_TYPES.items():
            fig, axes = plt.subplots(1, len(shown_n), figsize=(7.2, 2.35), sharey=True)
            for panel, n in zip(axes, shown_n):
                cell = summary[(summary.y_type == y_type) & (summary.n == n)]
                for method in METHODS:
                    rows = cell[cell.method == method].sort_values(axis_name)
                    panel.plot(rows[axis_name], rows["mean"], "-o", color=COLORS[method],
                               lw=1.1, ms=2.5, label=LABELS[method])
                    panel.fill_between(
                        rows[axis_name], rows["mean"] - rows["sem"], rows["mean"] + rows["sem"],
                        color=COLORS[method], alpha=0.14, linewidth=0,
                    )
                if axis_name == "phi_dim":
                    panel.set_xscale("log")
                    panel.set_xticks(axis_values, [str(int(v)) for v in axis_values])
                    panel.minorticks_off()
                else:
                    panel.set_xticks(axis_values)
                panel.set_xlabel(AXIS_XLABELS[axis_name])
                panel.set_title(f"$n={n}$", fontsize=8)
                style_axis(panel)
            axes[0].set_ylabel(ylabel)
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.03),
                       ncol=8, frameon=False, fontsize=6.2, handlelength=1.4)
            fig.subplots_adjust(left=0.075, right=0.995, bottom=0.2, top=0.78, wspace=0.18)
            slug = f"row_{metric_slug}_vs_{axis_name}"
            if axis_name != "phi_dim":
                slug += "_phi10"
            learner_slug = LEARNERS[learner]
            fig.savefig(output / f"{slug}_{learner_slug}_{y_slug}.pdf", bbox_inches="tight")
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", required=True, type=Path)
    parser.add_argument("--alpha", required=True, type=Path)
    parser.add_argument("--delta", required=True, type=Path)
    parser.add_argument("--m-sweep", type=Path)
    parser.add_argument("--m-sweep-max-failures", type=int, default=0)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--expected-runs", type=int, default=10)
    parser.add_argument("--violation-sample-sizes", type=int, nargs="+", default=[100, 500, 750])
    args = parser.parse_args()
    main_data = read_frame(args.main)
    alpha_data = read_frame(args.alpha)
    delta_data = read_frame(args.delta)
    validate(main_data, args.expected_runs)
    validate(alpha_data, args.expected_runs, "alpha")
    validate(delta_data, args.expected_runs, "delta")
    main_output = args.output_root / "main"
    alpha_output = args.output_root / "alpha-sweep"
    delta_output = args.output_root / "delta-sweep"
    for directory in (main_output, alpha_output, delta_output):
        directory.mkdir(parents=True, exist_ok=True)
    plot_main_panels(main_data, main_output)
    for learner in LEARNERS:
        if learner in alpha_data.learner.unique():
            plot_violation_rows(alpha_data, "alpha", alpha_output,
                                args.violation_sample_sizes, learner)
        if learner in delta_data.learner.unique():
            plot_violation_rows(delta_data, "delta", delta_output,
                                args.violation_sample_sizes, learner)
    if args.m_sweep is not None:
        m_data = read_frame(args.m_sweep)
        validate(m_data, args.expected_runs, "phi_dim",
                 max_labeled_failures=args.m_sweep_max_failures)
        m_output = args.output_root / "m-sweep"
        m_output.mkdir(parents=True, exist_ok=True)
        for learner in LEARNERS:
            if learner in m_data.learner.unique():
                plot_violation_rows(m_data, "phi_dim", m_output,
                                    args.violation_sample_sizes, learner)
    print(f"wrote paper panels to {args.output_root}")


if __name__ == "__main__":
    main()
