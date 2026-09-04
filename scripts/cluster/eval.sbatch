#!/bin/bash
# Phase 2 — CPU array job. One trial per array task. Loads encoders from cache.
#
# Usage (via submit_slice.sh):
#   SLICE=paper_main sbatch --array=0-9 scripts/cluster/eval.sbatch
#
# Each array task writes its own results/synthetic/raw/<slice>_<timestamp>/results.parquet.
# Merge with `python scripts/concat_results.py` after all tasks finish.
#
#SBATCH --job-name=crl-eval
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

SLICE="${SLICE:?set SLICE env var (e.g. paper_main)}"
BASE_SEED="${BASE_SEED:-1000}"
DEVICE="${DEVICE:-cpu}"
# DIAGNOSTICS env var is now optional. If unset, the YAML's sweep.diagnostics
# (default false in configs/default.yaml) is honored. Set DIAGNOSTICS=true|false
# explicitly to override the YAML for one submission.
RUN_TS="${RUN_TS:-$(date +"%Y%m%d_%H%M%S")}"
export RUN_TS                        # picked up by run_slice.py for parent dir
REPO="${REPO:-${SLURM_SUBMIT_DIR:?submit from the repository root or set REPO}}"
VENV_PY="${VENV_PY:-python}"

cd "$REPO"

export PYTHONNOUSERSITE=1   # don't fall back to ~/.local (older numpy lives there)
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1   # flush stdout per line so the SLURM .out updates live
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

SEED=$((BASE_SEED + SLURM_ARRAY_TASK_ID))
echo "host=$(hostname)  slice=$SLICE  task=$SLURM_ARRAY_TASK_ID  seed=$SEED  device=$DEVICE  start=$(date)"
[[ "$DEVICE" == "cuda" ]] && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

EXTRA_OVERRIDES=()
if [ -n "${DIAGNOSTICS:-}" ]; then
    EXTRA_OVERRIDES+=("sweep.diagnostics=${DIAGNOSTICS}")
fi

"$VENV_PY" scripts/run_slice.py "configs/slices/${SLICE}.yaml" \
    "slice.n_trials=1" \
    "slice.base_seed=${SEED}" \
    "${EXTRA_OVERRIDES[@]}" \
    --device "$DEVICE" \
    --no-progress

echo "end=$(date)"
sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,TotalCPU,MaxRSS,ReqMem --noheader || true
