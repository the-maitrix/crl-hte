from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


EXPECTED_N = [100, 250, 500, 750, 1000]
EXPECTED_SEEDS = list(range(10))
PAPER_METHODS = {
    "mi_infonce_cond", "mi_mine_cond", "encoder_pred", "pls",
    "pca", "ica", "autoencoder", "raw_x", "raw_x_s",
}


def collect(run_dir: Path, phi_dim: int, outcome_source: str) -> pd.DataFrame:
    files = sorted(run_dir.glob(f"phi{phi_dim}_{outcome_source}_seed*.csv"))
    if len(files) != 10:
        raise ValueError(f"expected 10 files for phi={phi_dim}, outcome={outcome_source}; found {len(files)}")
    data = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    if sorted(data.seed.unique()) != EXPECTED_SEEDS:
        raise ValueError(f"unexpected seeds for phi={phi_dim}, outcome={outcome_source}")
    if sorted(data.n.unique()) != EXPECTED_N:
        raise ValueError(f"unexpected sample sizes for phi={phi_dim}, outcome={outcome_source}")
    if set(data["cate_learner"]) != {"econml_dml"}:
        raise ValueError(f"paper results must use econml_dml for phi={phi_dim}, outcome={outcome_source}")
    if set(data["ate_calibration"]) != {"none"}:
        raise ValueError(f"paper results must be uncalibrated for phi={phi_dim}, outcome={outcome_source}")
    expected_methods = PAPER_METHODS - ({"raw_x_s"} if outcome_source == "true" else set())
    data = data[data.method.isin(expected_methods)].copy()
    if set(data.method) != expected_methods:
        missing = sorted(expected_methods - set(data.method))
        raise ValueError(f"missing paper methods for phi={phi_dim}, outcome={outcome_source}: {missing}")
    expected_rows = len(EXPECTED_SEEDS) * len(EXPECTED_N) * len(expected_methods)
    if len(data) != expected_rows:
        raise ValueError(f"expected {expected_rows} paper rows; found {len(data)}")
    if data[["pehe_norm", "policy_norm_20"]].isna().any().any():
        raise ValueError("paper metrics contain missing values")
    if data.duplicated(["seed", "n", "method"]).any():
        raise ValueError("duplicate seed/n/method rows")
    return data


def run(*args: str):
    subprocess.run([sys.executable, *args], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--paper-output", default="results/paper", type=Path)
    args = parser.parse_args()
    args.paper_output.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for phi_dim in (5, 10):
        for outcome_source in ("predicted", "true"):
            data = collect(args.run_dir, phi_dim, outcome_source)
            suffix = f"true_y_phi{phi_dim}" if outcome_source == "true" else f"phi{phi_dim}"
            output = args.paper_output / f"semisynthetic_{suffix}.csv"
            data.to_csv(output, index=False)
            outputs[(phi_dim, outcome_source)] = output

    generated = args.paper_output / "semisynthetic"
    run(
        "scripts/export_semisynth_paper_tables.py",
        "--phi5", str(outputs[(5, "predicted")]),
        "--phi10", str(outputs[(10, "predicted")]),
        "--output-dir", str(generated / "tables"),
    )
    run(
        "scripts/export_semisynth_paper_tables.py",
        "--phi5", str(outputs[(5, "true")]),
        "--phi10", str(outputs[(10, "true")]),
        "--output-dir", str(generated / "tables_true_y"),
        "--label-suffix", "_true_y",
        "--caption-note", " using true experimental $Y$",
    )
    for phi_dim, outcome_source, directory in (
        (10, "predicted", "phi10"),
        (5, "predicted", "phi5"),
        (10, "true", "true_y_phi10"),
    ):
        run(
            "scripts/plot_corrected_semisynth_fresh.py",
            "--input", str(outputs[(phi_dim, outcome_source)]),
            "--output-dir", str(generated / directory),
        )
    print(args.paper_output)


if __name__ == "__main__":
    main()
