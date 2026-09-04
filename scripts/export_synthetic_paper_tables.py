from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METHODS = [
    "mi_infonce_cond", "mi_mine_cond", "encoder_pred", "pls",
    "pca", "ica", "autoencoder", "raw_x",
]
RAW_X_S_METHOD = "raw_x_s"
LABELS = {
    "mi_infonce_cond": "MI-InfoNCE",
    "mi_mine_cond": "MI-MINE",
    "encoder_pred": "Encoder",
    "pls": "PLS",
    "pca": "PCA",
    "ica": "ICA",
    "autoencoder": "Autoencoder",
    "raw_x": r"Raw $X$, $h(X)$",
    RAW_X_S_METHOD: r"Raw $X$, $h(X,S)$",
}
LEARNERS = {
    "x_learner": ("X-learner", "xl"),
    "dml_learner": ("DML", "dml"),
}
Y_TYPES = [("h_y", r"$\widehat Y$ target"), ("true_y", r"$Y$ target")]
METRICS = {
    "pehe_norm": ("Normalized PEHE", "pehe_norm", "lower"),
    "policy_norm_20": (r"Normalized top-20\% policy value", "policy_norm_20", "higher"),
}
EXPECTED_N = [50, 100, 250, 500, 750]


def read_frame(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def validate(data: pd.DataFrame, expected_runs: int) -> list[int]:
    required = {
        "method", "n", "learner", "y_type", "trial_seed", "phi_dim",
        "pehe_norm", "policy_norm_20",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    data = data[
        data.method.isin(METHODS)
        & data.learner.isin(LEARNERS)
        & data.y_type.isin(dict(Y_TYPES))
        & (data.phi_dim == 10)
    ]
    sample_sizes = sorted(int(n) for n in data.n.unique())
    if sample_sizes != EXPECTED_N:
        raise ValueError(f"expected n={EXPECTED_N}; found {sample_sizes}")
    expected_cells = len(METHODS) * len(LEARNERS) * len(Y_TYPES) * len(EXPECTED_N)
    counts = data.groupby(["method", "learner", "y_type", "n"]).trial_seed.nunique()
    if len(counts) != expected_cells:
        raise ValueError(f"found {len(counts)} cells; expected {expected_cells}")
    bad = counts[counts != expected_runs]
    if not bad.empty:
        raise ValueError(f"cells do not have {expected_runs} runs:\n{bad.to_string()}")
    if "dgp_seed" in data and sorted(data.dgp_seed.unique()) != [42]:
        raise ValueError(f"expected dgp_seed=[42]; found {sorted(data.dgp_seed.unique())}")
    if data.duplicated(["method", "learner", "y_type", "n", "trial_seed"]).any():
        raise ValueError("duplicate method/learner/y_type/n/trial_seed rows")
    if data[["pehe_norm", "policy_norm_20"]].isna().any().any():
        raise ValueError("paper metrics contain missing values")
    return sorted(int(n) for n in data.n.unique())


def add_raw_x_s(data: pd.DataFrame, raw_comparison: pd.DataFrame, expected_runs: int) -> pd.DataFrame:
    required = {
        "method", "n", "learner", "y_type", "trial_seed", "phi_dim",
        "pehe_norm", "policy_norm_20",
    }
    missing = required - set(raw_comparison.columns)
    if missing:
        raise ValueError(f"raw comparison is missing columns: {sorted(missing)}")
    raw_x_s = raw_comparison[
        (raw_comparison.method == "surrogate_index")
        & raw_comparison.learner.isin(LEARNERS)
        & (raw_comparison.y_type == "h_y")
        & (raw_comparison.phi_dim == 10)
    ].copy()
    sample_sizes = sorted(int(n) for n in raw_x_s.n.unique())
    if sample_sizes != EXPECTED_N:
        raise ValueError(f"raw comparison expected n={EXPECTED_N}; found {sample_sizes}")
    counts = raw_x_s.groupby(["learner", "n"]).trial_seed.nunique()
    expected_cells = len(LEARNERS) * len(EXPECTED_N)
    if len(counts) != expected_cells or (counts != expected_runs).any():
        raise ValueError(
            f"raw comparison must have {expected_runs} runs in each learner/n cell:\n"
            f"{counts.to_string()}"
        )
    if raw_x_s.duplicated(["learner", "n", "trial_seed"]).any():
        raise ValueError("raw comparison has duplicate learner/n/trial_seed rows")
    if raw_x_s[["pehe_norm", "policy_norm_20"]].isna().any().any():
        raise ValueError("raw-comparison paper metrics contain missing values")
    raw_x_s["method"] = RAW_X_S_METHOD
    return pd.concat([data, raw_x_s], ignore_index=True, sort=False)


def render(data: pd.DataFrame, learner: str, metric: str, output: Path, expected_runs: int):
    learner_name, learner_slug = LEARNERS[learner]
    metric_name, metric_slug, direction = METRICS[metric]
    display_methods = METHODS + ([RAW_X_S_METHOD] if (data.method == RAW_X_S_METHOD).any() else [])
    subset = data[
        data.method.isin(display_methods)
        & (data.learner == learner)
        & data.y_type.isin(dict(Y_TYPES))
        & (data.phi_dim == 10)
    ]
    sample_sizes = sorted(int(n) for n in subset.n.unique())
    summary = subset.groupby(["method", "y_type", "n"])[metric].agg(["mean", "sem"])
    rows = []
    for method in display_methods:
        cells = []
        for y_type, _ in Y_TYPES:
            for n in sample_sizes:
                if (method, y_type, n) not in summary.index:
                    cells.append(r"---")
                    continue
                mean = summary.loc[(method, y_type, n), "mean"]
                sem = summary.loc[(method, y_type, n), "sem"]
                cell = f"{mean:.3f}$\\pm${sem:.3f}"
                at_n = summary.xs((y_type, n), level=("y_type", "n"))["mean"]
                best = at_n.min() if direction == "lower" else at_n.max()
                if abs(mean - best) < 1e-12:
                    cell = rf"\textbf{{{cell}}}"
                cells.append(cell)
        rows.append(f"{LABELS[method]} & " + " & ".join(cells) + r" \\")
    columns = "c" * len(sample_sizes)
    label = f"tab:synth_{metric_slug}_{learner_slug}"
    text = "\n".join([
        r"\begin{table}[t]",
        r"\centering",
        r"\resizebox{\linewidth}{!}{%",
        rf"\begin{{tabular}}{{l|{columns}|{columns}}}",
        r"\toprule",
        " & " + " & ".join(
            rf"\multicolumn{{{len(sample_sizes)}}}{{c}}{{{title}}}"
            for _, title in Y_TYPES
        ) + r" \\",
        "Method & " + " & ".join(
            f"$n={n}$" for _ in Y_TYPES for n in sample_sizes
        ) + r" \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}}",
        rf"\caption{{{metric_name} on the synthetic data, {learner_name} ($\dim(\phi)=10$). Entries are means $\pm$ standard errors over {expected_runs} paired data resamples; {direction} is better. For the $Y$ target, the two raw-$X$ variants coincide, so we report a single value under $h(X)$.}}",
        rf"\label{{{label}}}",
        r"\end{table}",
        "",
    ])
    output.write_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--raw-comparison", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-runs", type=int, default=10)
    args = parser.parse_args()
    data = read_frame(args.input)
    sample_sizes = validate(data, args.expected_runs)
    if args.raw_comparison is not None:
        data = add_raw_x_s(data, read_frame(args.raw_comparison), args.expected_runs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for learner, (_, learner_slug) in LEARNERS.items():
        for metric, (_, metric_slug, _) in METRICS.items():
            render(
                data,
                learner,
                metric,
                args.output_dir / f"papertable_{metric_slug}_phi10_{learner_slug}.tex",
                args.expected_runs,
            )
    print(f"wrote 4 tables for n={sample_sizes} to {args.output_dir}")


if __name__ == "__main__":
    main()
