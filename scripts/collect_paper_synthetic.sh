#!/usr/bin/env bash
set -euo pipefail

RUN_TS="${1:?usage: collect_paper_synthetic.sh <run_timestamp>}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

for slice in \
  paper_main \
  raw_x_comparison \
  alpha_sweep_dml_varmatched \
  delta_sweep_dml_varmatched \
  m_sweep; do
  "$PYTHON" scripts/concat_results.py --run "${slice}_${RUN_TS}" --strict-schema
done

"$PYTHON" scripts/export_public_results.py \
  --input "results/synthetic/raw/paper_main_${RUN_TS}/master.parquet" \
  --output results/paper/synthetic_main.csv
"$PYTHON" scripts/export_public_results.py \
  --input "results/synthetic/raw/raw_x_comparison_${RUN_TS}/master.parquet" \
  --output results/paper/synthetic_raw_x_comparison.csv
"$PYTHON" scripts/export_public_results.py \
  --input "results/synthetic/raw/alpha_sweep_dml_varmatched_${RUN_TS}/master.parquet" \
  --output results/paper/synthetic_alpha_dml_varmatched.csv
"$PYTHON" scripts/export_public_results.py \
  --input "results/synthetic/raw/delta_sweep_dml_varmatched_${RUN_TS}/master.parquet" \
  --output results/paper/synthetic_delta_dml_varmatched.csv
"$PYTHON" scripts/export_public_results.py \
  --input "results/synthetic/raw/m_sweep_${RUN_TS}/master.parquet" \
  --output results/paper/synthetic_m_sweep.csv

"$PYTHON" scripts/export_synthetic_paper_tables.py \
  --input results/paper/synthetic_main.csv \
  --raw-comparison results/paper/synthetic_raw_x_comparison.csv \
  --output-dir results/paper/synthetic/main
"$PYTHON" scripts/export_synthetic_paper_panels.py \
  --main results/paper/synthetic_main.csv \
  --alpha results/paper/synthetic_alpha_dml_varmatched.csv \
  --delta results/paper/synthetic_delta_dml_varmatched.csv \
  --m-sweep results/paper/synthetic_m_sweep.csv \
  --m-sweep-max-failures 4 \
  --output-root results/paper/synthetic

if [[ -f results/paper/semisynthetic_phi10.csv ]]; then
  "$PYTHON" scripts/plot_harmonized_main_figure.py \
    --synthetic-results results/paper/synthetic_main.csv \
    --semisynthetic-results results/paper/semisynthetic_phi10.csv \
    --output results/paper/main_figure.pdf
fi

printf 'main=%s\nraw_x=%s\nalpha=%s\ndelta=%s\nm=%s\n' \
  "paper_main_${RUN_TS}" \
  "raw_x_comparison_${RUN_TS}" \
  "alpha_sweep_dml_varmatched_${RUN_TS}" \
  "delta_sweep_dml_varmatched_${RUN_TS}" \
  "m_sweep_${RUN_TS}" \
  > results/paper/synthetic/RUNS.txt
