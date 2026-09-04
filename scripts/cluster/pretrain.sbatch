#!/bin/bash
# Phase 1 — populate the encoder cache for one slice. No CATE eval.
#
# Defaults to CPU. submit_slice.sh overrides to GPU for heavy slices via:
#   sbatch --partition=general --gres=gpu:1 --time=04:00:00 ... pretrain.sbatch
#
# Usage (via submit_slice.sh):
#   SLICE=paper_main sbatch scripts/cluster/pretrain.sbatch
#
#SBATCH --job-name=crl-pretrain
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%A.out
#SBATCH --error=logs/%x-%A.err

set -euo pipefail

SLICE="${SLICE:?set SLICE env var (e.g. paper_main)}"
DEVICE="${DEVICE:-cpu}"
RUN_TS="${RUN_TS:-$(date +"%Y%m%d_%H%M%S")}"
export RUN_TS
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

echo "host=$(hostname)  slice=$SLICE  device=$DEVICE  start=$(date)"
[[ "$DEVICE" == "cuda" ]] && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"$VENV_PY" scripts/run_slice.py "configs/slices/${SLICE}.yaml" \
    "slice.n_trials=1" \
    --device "$DEVICE" \
    --no-progress \
    --pretrain-only

echo "end=$(date)"
sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,TotalCPU,MaxRSS,ReqMem --noheader || true
