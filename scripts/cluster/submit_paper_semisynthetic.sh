#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ID="${RUN_ID:-paper_$(date +"%Y%m%d_%H%M%S")}"
CACHE_DIR="${CACHE_DIR:-data/semisynth_cache}"

for phi in 5 10; do
  for outcome in predicted true; do
    PHI_DIM="$phi" OUTCOME_SOURCE="$outcome" RUN_ID="$RUN_ID" CACHE_DIR="$CACHE_DIR" \
      sbatch --array=0-9 "$HERE/run_paper_semisynthetic.sbatch"
  done
done

echo "$RUN_ID"
