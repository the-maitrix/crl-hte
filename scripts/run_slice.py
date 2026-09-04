"""
Run a single slice end-to-end.

Usage
-----
    python scripts/run_slice.py configs/slices/alpha_sweep.yaml
    python scripts/run_slice.py configs/slices/main_synthetic.yaml \\
        slice.n_trials=2 sweep.sample_sizes=[100,500] runtime.device=cuda

Outputs
-------
    results/synthetic/raw/<slice_name>_<timestamp>/
        config.yaml      — frozen, fully resolved config used for the run
        results.parquet  — long-format result table
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

# Add src/ to path when invoked as a script
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from omegaconf import OmegaConf

from crl_hte.config import load_config, to_yaml_str
from crl_hte.runner import run_slice
from crl_hte.encoders import get_device


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run a slice from a YAML config.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "slice_config",
        help="Path to a slice YAML (e.g. configs/slices/alpha_sweep.yaml).",
    )
    p.add_argument(
        "overrides", nargs="*",
        help="OmegaConf dot-list overrides (e.g. encoder.phi_dim=10).",
    )
    p.add_argument("--default-config", default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument("--output-root", default=None,
                   help="Override runtime.output_root from the config.")
    p.add_argument("--cache-dir", default=None,
                   help="Override runtime.cache_dir from the config.")
    p.add_argument("--device", default=None,
                   help="cuda | mps | cpu (overrides runtime.device).")
    p.add_argument("--no-progress", action="store_true",
                   help="Disable tqdm progress bar (useful for SLURM logs).")
    p.add_argument("--print-config", action="store_true",
                   help="Print resolved config and exit without running.")
    p.add_argument("--save-tau-arrays", action="store_true",
                   help="Dump full true-CATE arrays as .npy alongside results "
                        "(one per dgp_seed × z_dim × α × δ).")
    p.add_argument("--pretrain-only", action="store_true",
                   help="Train and cache encoders only; skip CATE eval. Use "
                        "to populate the encoder cache from a GPU job before "
                        "running CATE eval on CPU.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    overrides = list(args.overrides)
    if args.device:
        overrides.append(f"runtime.device={args.device}")
    if args.output_root:
        overrides.append(f"runtime.output_root={args.output_root}")
    if args.cache_dir:
        overrides.append(f"runtime.cache_dir={args.cache_dir}")
    if args.save_tau_arrays:
        overrides.append("slice.save_tau_arrays=true")

    cfg = load_config(
        default_yaml=args.default_config,
        slice_yaml=args.slice_config,
        overrides=overrides,
    )

    if args.print_config:
        print(to_yaml_str(cfg))
        return 0

    # Parent run dir shared across all array tasks. Prefer RUN_TS env var
    # (set by submit_slice.sh once per submission) so every task lands under
    # the same parent. Fall back to datetime.now() for ad-hoc runs.
    import os
    run_ts = os.environ.get("RUN_TS") or datetime.now().strftime("%Y%m%d_%H%M%S")
    slice_name = cfg.slice.name
    parent_dir = Path(cfg.runtime.output_root) / f"{slice_name}_{run_ts}"

    # Per-task subdir disambiguates the SLURM array. Pretrain (no array id)
    # gets a flat 'pretrain/' so it doesn't collide with task-id subdirs.
    array_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if args.pretrain_only:
        subdir = "pretrain"
    elif array_id is not None:
        subdir = f"task{array_id}_seed{cfg.slice.base_seed}"
    else:
        subdir = f"seed{cfg.slice.base_seed}"
    output_dir = parent_dir / subdir
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    cache_dir = Path(cfg.runtime.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = ROOT / cache_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = output_dir / "config.yaml"
    cfg_path.write_text(to_yaml_str(cfg))
    print(f"[run_slice] resolved config → {cfg_path}")

    device = get_device(args.device or cfg.runtime.get("device"))
    t0 = time.time()
    df = run_slice(cfg, output_dir=output_dir, cache_dir=cache_dir,
                   device=device, progress=not args.no_progress,
                   pretrain_only=args.pretrain_only)
    elapsed = time.time() - t0
    print(f"[run_slice] DONE in {elapsed/60:.1f} min  rows={len(df)}  → {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
