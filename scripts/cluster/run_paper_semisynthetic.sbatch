#!/bin/bash
#SBATCH --job-name=semisynth-paper
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

PHI_DIM="${PHI_DIM:?set PHI_DIM}"
OUTCOME_SOURCE="${OUTCOME_SOURCE:?set OUTCOME_SOURCE}"
RUN_ID="${RUN_ID:?set RUN_ID}"
CACHE_DIR="${CACHE_DIR:-data/semisynth_cache}"
REPO="${REPO:-${SLURM_SUBMIT_DIR:?submit from the repository root or set REPO}}"
PYTHON="${PYTHON:-python}"
SEED="${SLURM_ARRAY_TASK_ID}"
OUTPUT="$REPO/results/semisynthetic/$RUN_ID/phi${PHI_DIM}_${OUTCOME_SOURCE}_seed${SEED}.csv"

cd "$REPO"
mkdir -p "$(dirname "$OUTPUT")" logs
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

"$PYTHON" scripts/run_semisynthetic.py \
  --cache-dir "$CACHE_DIR" \
  --phi-dim "$PHI_DIM" \
  --outcome-source "$OUTCOME_SOURCE" \
  --seed-start "$SEED" \
  --seeds 1 \
  --cate-learner econml_dml \
  --ate-calibration none \
  --methods mi_infonce_cond mi_mine_cond encoder_pred pls pca ica autoencoder raw_x raw_x_s \
  --output "$OUTPUT"
