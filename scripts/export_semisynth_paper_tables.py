from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METHODS = [
    "mi_infonce_cond", "mi_mine_cond", "encoder_pred",
    "pls", "pca", "ica", "autoencoder", "raw_x", "raw_x_s",
]
LABELS = {
    "encoder_pred": "Encoder",
    "mi_mine_cond": "MI-MINE",
    "mi_infonce_cond": "MI-InfoNCE",
    "pls": "PLS",
    "autoencoder": "Autoencoder",
    "pca": "PCA",
    "ica": "ICA",
    "raw_x": r"Raw $X$, $h(X)$",
    "raw_x_s": r"Raw $X$, $h(X,S)$",
}


def summarize(data):
    return (
        data[data.method.isin(METHODS)]
        .groupby(["method", "n"])[["pehe_norm", "policy_norm_20"]]
        .agg(["mean", "sem"])
    )


def value(summary, method, n, metric, best):
    mean = summary.loc[(method, n), (metric, "mean")]
    sem = summary.loc[(method, n), (metric, "sem")]
    formatted = f"{mean:.3f}$\\pm${sem:.3f}"
    if abs(mean - best) < 1e-12:
        return rf"\textbf{{{formatted}}}"
    return formatted


def metric_table(summary, sample_sizes, metric, output, phi_dim, n_seeds,
                 label_suffix="", caption_note=""):
    lower = metric == "pehe_norm"
    best = {
        n: (summary.xs(n, level="n")[(metric, "mean")].min()
            if lower else summary.xs(n, level="n")[(metric, "mean")].max())
        for n in sample_sizes
    }
    rows = []
    available = set(summary.index.get_level_values("method"))
    for method in (method for method in METHODS if method in available):
        cells = " & ".join(value(summary, method, n, metric, best[n]) for n in sample_sizes)
        method_label = "Raw $X$" if caption_note and method == "raw_x" else LABELS[method]
        rows.append(f"{method_label} & {cells} \\\\")
    metric_name = "Normalized PEHE" if lower else "Normalized top-20\\% policy value"
    direction = "lower" if lower else "higher"
    label = "pehe" if lower else "policy"
    text = "\n".join([
        r"\begin{table}[t]",
        r"\centering",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{l" + "c" * len(sample_sizes) + "}",
        r"\toprule",
        "Method & " + " & ".join(f"$n={n}$" for n in sample_sizes) + r" \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}}",
        rf"\caption{{{metric_name} on the perinatal semi-synthetic dataset at $\dim(\phi)={phi_dim}${caption_note}. Entries are means $\pm$ standard errors over {n_seeds} runs; {direction} is better.}}",
        rf"\label{{tab:corrected_semisynth_phi{phi_dim}_{label}{label_suffix}}}",
        r"\end{table}",
        "",
    ])
    output.write_text(text)


def sensitivity_table(phi5, phi10, output, n_seeds):
    sizes = sorted(set(phi5.index.get_level_values("n")) & set(phi10.index.get_level_values("n")))
    rows = []
    for n in sizes:
        cells = []
        for summary, metric in ((phi5, "pehe_norm"), (phi10, "pehe_norm"),
                                (phi5, "policy_norm_20"), (phi10, "policy_norm_20")):
            mean = summary.loc[("encoder_pred", n), (metric, "mean")]
            sem = summary.loc[("encoder_pred", n), (metric, "sem")]
            cells.append(f"{mean:.3f}$\\pm${sem:.3f}")
        rows.append(f"{n} & " + " & ".join(cells) + r" \\")
    text = "\n".join([
        r"\begin{table}[t]",
        r"\centering",
        r"\begin{tabular}{rcccc}",
        r"\toprule",
        r"& \multicolumn{2}{c}{Normalized PEHE} & \multicolumn{2}{c}{Policy@20\%} \\",
        r"$n$ & $\dim(\phi)=5$ & $\dim(\phi)=10$ & $\dim(\phi)=5$ & $\dim(\phi)=10$ \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{Representation-dimension sensitivity for the prediction encoder on the perinatal semi-synthetic dataset. Entries are means $\pm$ standard errors over {n_seeds} runs.}}",
        r"\label{tab:corrected_semisynth_phi_sensitivity}",
        r"\end{table}",
        "",
    ])
    output.write_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phi5", required=True)
    parser.add_argument("--phi10", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-suffix", default="")
    parser.add_argument("--caption-note", default="")
    args = parser.parse_args()

    path5, path10 = Path(args.phi5), Path(args.phi10)
    data5 = pd.read_parquet(path5) if path5.suffix == ".parquet" else pd.read_csv(path5)
    data10 = pd.read_parquet(path10) if path10.suffix == ".parquet" else pd.read_csv(path10)
    summary5 = summarize(data5)
    summary10 = summarize(data10)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sizes = sorted(data10.n.unique())
    n5 = data5.seed.nunique()
    n10 = data10.seed.nunique()
    if n5 != n10:
        raise ValueError(f"Seed counts differ: phi=5 has {n5}, phi=10 has {n10}")
    metric_table(
        summary10, sizes, "pehe_norm", output / "full_baselines_phi10_pehe.tex",
        10, n10, args.label_suffix, args.caption_note,
    )
    metric_table(
        summary10, sizes, "policy_norm_20", output / "full_baselines_phi10_policy.tex",
        10, n10, args.label_suffix, args.caption_note,
    )
    sensitivity_table(summary5, summary10, output / "phi_sensitivity.tex", n10)
    print(output)


if __name__ == "__main__":
    main()
