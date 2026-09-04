#!/usr/bin/env bash
# Submit pretrain (Phase 1) + eval array (Phase 2) for one slice.
#
# Usage:
#   scripts/cluster/submit_slice.sh <slice> [n_trials]
#
# Per-slice resource policy: start small; heavier slices override measured needs.
set -euo pipefail

SLICE="${1:?usage: submit_slice.sh <slice> [n_trials]}"
N="${2:-}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PRETRAIN_OVERRIDE=(--partition=preempt --qos=preempt_cpu_qos)
EVAL_OVERRIDE=(--partition=preempt --qos=preempt_cpu_qos)
PRETRAIN_DEVICE="cpu"
EVAL_DEVICE="cpu"

case "$SLICE" in
  paper_main)
    N="${N:-10}"
    PRETRAIN_OVERRIDE=(--partition=general --qos=normal --gres=gpu:1 --mem=4G --time=02:00:00)
    PRETRAIN_DEVICE="cuda"
    EVAL_OVERRIDE+=(--cpus-per-task=4 --mem=4G --time=06:00:00)
    ;;
  no_s_ablation)
    PRETRAIN_OVERRIDE=(--partition=general --qos=normal --gres=gpu:1 --mem=4G --time=03:00:00)
    PRETRAIN_DEVICE="cuda"
    EVAL_OVERRIDE+=(--cpus-per-task=2 --mem=4G --time=04:00:00)
    ;;
  nh_sweep)
    PRETRAIN_OVERRIDE=(--partition=general --qos=normal --gres=gpu:1 --mem=4G --time=04:00:00)
    PRETRAIN_DEVICE="cuda"
    EVAL_OVERRIDE+=(--cpus-per-task=4 --mem=4G --time=06:00:00)
    ;;
  lt_o)
    PRETRAIN_OVERRIDE=(--partition=general --qos=normal --gres=gpu:1 --mem=4G --time=03:00:00)
    PRETRAIN_DEVICE="cuda"
    # Five-fold nuisances include the 10,000-row historical sample.
    EVAL_OVERRIDE+=(--cpus-per-task=4 --mem=6G --time=12:00:00)
    ;;
  nonlinear_xz)
    # Forty nonlinear encoders across 5 DGP seeds × 2 bottlenecks.
    PRETRAIN_OVERRIDE=(--partition=general --qos=normal --gres=gpu:1 --mem=4G --time=04:00:00)
    PRETRAIN_DEVICE="cuda"
    EVAL_OVERRIDE+=(--cpus-per-task=4 --mem=4G --time=06:00:00)
    ;;
  main_synthetic|paper_repro|nonlinear_zs|nonlinear_sy)
    # 11 methods × 5 sample sizes × 3 learners × 2 y_types per cell, plus
    # raw_x CATE fits on 1000-feature data + diagnostics on the full E pool.
    # The 1cpu/2G/2h defaults TIMEOUT here; give it real resources.
    EVAL_OVERRIDE+=(--cpus-per-task=4 --mem=4G --time=06:00:00)
    ;;
  raw_x_comparison)
    N="${N:-10}"
    EVAL_OVERRIDE+=(--cpus-per-task=4 --mem=4G --time=02:00:00)
    ;;
  alpha_sweep_dml_varmatched|delta_sweep_dml_varmatched)
    N="${N:-10}"
    PRETRAIN_OVERRIDE=(--partition=general --qos=normal --gres=gpu:1 --mem=4G --time=02:00:00)
    PRETRAIN_DEVICE="cuda"
    EVAL_OVERRIDE+=(--cpus-per-task=4 --mem=4G --time=06:00:00)
    ;;
  m_sweep)
    N="${N:-10}"
    PRETRAIN_OVERRIDE=(--partition=general --qos=normal --gres=gpu:1 --mem=4G --time=03:00:00)
    PRETRAIN_DEVICE="cuda"
    EVAL_OVERRIDE+=(--cpus-per-task=4 --mem=4G --time=08:00:00)
    ;;
  lambda_sweep)
    PRETRAIN_OVERRIDE=(--partition=general --qos=normal --gres=gpu:1 --mem=4G --time=03:00:00)
    PRETRAIN_DEVICE="cuda"
    EVAL_OVERRIDE+=(--cpus-per-task=4 --mem=4G --time=06:00:00)
    ;;
  arch_sweep)
    EVAL_OVERRIDE+=(--cpus-per-task=4 --mem=4G --time=06:00:00)
    ;;
  zdim_sweep)
    # 192 base encoders incl. z=40 → train on GPU.
    PRETRAIN_OVERRIDE=(--partition=general --qos=normal --gres=gpu:1 --mem=4G --time=04:00:00)
    PRETRAIN_DEVICE="cuda"
    # 16 struct cells × full CATE grid is the long pole; phi=40 h-models
    # are bigger so bump memory.
    EVAL_OVERRIDE+=(--mem=4G --time=12:00:00)
    ;;
  finetune)
    PRETRAIN_OVERRIDE=(--partition=general --qos=normal --gres=gpu:1 --mem=4G --time=03:00:00)
    PRETRAIN_DEVICE="cuda"
    # Focused one-encoder comparison: 3 alpha cells x 5 sample sizes. The
    # encoder is readout-only adapted, so CPU is sufficient and avoids five
    # needless GPU allocations.
    EVAL_OVERRIDE+=(--cpus-per-task=2 --mem=4G --time=04:00:00)
    ;;
  finetune_undertrained)
    # Smaller than `finetune` (1 mode × 1 cell × 5 xfit folds × 3α × 3n × 5
    # trials = 225 finetune events; readout-only update, ~5s each on CPU).
    # ~30-60 min total — CPU is fine. Sized tight per cluster eff stats.
    EVAL_OVERRIDE+=(--cpus-per-task=2 --mem=4G --time=04:00:00)
    ;;
  *)
    echo "warn: unknown slice '$SLICE', using defaults" >&2
    ;;
esac

N="${N:-5}"

# Timestamp computed once and shared across pretrain + every eval array task,
# so all per-task output dirs land under one parent: <slice>_<RUN_TS>/.
RUN_TS="${RUN_TS:-$(date +"%Y%m%d_%H%M%S")}"
echo "[$SLICE] run timestamp = $RUN_TS"
mkdir -p "$HERE/../../logs"

PRETRAIN_ID=$(SLICE="$SLICE" DEVICE="$PRETRAIN_DEVICE" RUN_TS="$RUN_TS" \
    sbatch --parsable \
    "${PRETRAIN_OVERRIDE[@]}" "$HERE/pretrain.sbatch")
echo "[$SLICE] pretrain → job $PRETRAIN_ID  (device=$PRETRAIN_DEVICE)"

EVAL_ID=$(SLICE="$SLICE" DEVICE="$EVAL_DEVICE" RUN_TS="$RUN_TS" \
    sbatch --parsable \
    --dependency=afterok:"$PRETRAIN_ID" \
    --array=0-$((N - 1)) \
    "${EVAL_OVERRIDE[@]}" "$HERE/eval.sbatch")
echo "[$SLICE] eval     → job $EVAL_ID  (array 0-$((N-1)), device=$EVAL_DEVICE)"
echo "[$SLICE] outputs will be under results/synthetic/raw/${SLICE}_${RUN_TS}/"
