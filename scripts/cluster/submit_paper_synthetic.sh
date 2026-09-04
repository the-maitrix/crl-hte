#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_TS="${RUN_TS:-$(date +"%Y%m%d_%H%M%S")}"
export RUN_TS

for slice in \
  paper_main \
  raw_x_comparison \
  alpha_sweep_dml_varmatched \
  delta_sweep_dml_varmatched \
  m_sweep; do
  bash "$HERE/submit_slice.sh" "$slice" 10
done

echo "Collect after all five evaluation arrays finish:"
echo "PYTHON=.venv/bin/python bash scripts/collect_paper_synthetic.sh $RUN_TS"
