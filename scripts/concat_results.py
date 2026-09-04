"""
Merge slice parquets into a single master parquet.

Layout: each run lives under results/synthetic/raw/<slice>_<RUN_TS>/ with per-task
subdirs (task<N>_seed<S>/results.parquet). Each parent dir is one experiment;
this script walks the per-task subdirs and concatenates their parquets, by
default into <parent>/master.parquet.

Usage
-----
    # Merge a specific run, write master.parquet inside that run's parent dir
    python scripts/concat_results.py --run main_synthetic_20260430_120232

    # Merge multiple runs by glob pattern (legacy / cross-run aggregation)
    python scripts/concat_results.py --pattern 'main_synthetic_*' \\
        --output results/main_synthetic_master.parquet

    # Filter by mtime (e.g. last 6 hours)
    python scripts/concat_results.py --since-hours 6
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# When `scripts/run_slice.py` runs under SLURM, output dirs are named like
# `<slice>_<YYYYMMDD_HHMMSS>_task<N>_seed<S>/`. Use this regex to extract task/seed.
DIR_RE = re.compile(
    r"^(?P<slice>.+?)_(?P<ts>\d{8}_\d{6})(?:_task(?P<task>\d+))?(?:_seed(?P<seed>\d+))?$"
)
TASK_DIR_RE = re.compile(r"^task(?P<task>\d+)_seed(?P<seed>\d+)$")


def _parse_args():
    p = argparse.ArgumentParser(description="Merge slice parquets into a master parquet.")
    p.add_argument(
        "--input-root",
        default=str(ROOT / "results" / "synthetic" / "raw"),
    )
    p.add_argument("--run", default=None,
                   help="Single run dir name under input-root "
                        "(e.g. 'main_synthetic_20260430_120232'). If set, "
                        "writes master.parquet INTO that run dir by default.")
    p.add_argument("--pattern", default=None,
                   help="Glob pattern for run dirs (e.g. 'main_synthetic_*'). "
                        "Mutually exclusive with --run.")
    p.add_argument("--output", default=None,
                   help="Where to write the master parquet. Default: "
                        "<run_dir>/master.parquet for --run, else "
                        "results/master.parquet for --pattern.")
    p.add_argument("--since-hours", type=float, default=None,
                   help="Only include parquets modified within the last N hours.")
    p.add_argument("--strict-schema", action="store_true",
                   help="Fail if any parquet has a different column set.")
    return p.parse_args()


def _collect_parquets(run_dirs):
    """Each run dir holds task<N>_seed<S>/results.parquet subdirs (new layout).
    Falls back to <run_dir>/results.parquet (legacy flat layout)."""
    files = []
    for d in run_dirs:
        nested = sorted(d.glob("*/results.parquet"))
        if nested:
            files.extend(nested)
        elif (d / "results.parquet").is_file():
            files.append(d / "results.parquet")
    return files


def main() -> int:
    args = _parse_args()
    input_root = Path(args.input_root)

    if args.run and args.pattern:
        print("[concat] --run and --pattern are mutually exclusive", file=sys.stderr)
        return 2

    if args.run:
        run_dirs = [input_root / args.run]
        if not run_dirs[0].is_dir():
            print(f"[concat] no such run dir: {run_dirs[0]}")
            return 1
        default_output = run_dirs[0] / "master.parquet"
    else:
        pattern = args.pattern or "*"
        run_dirs = sorted(d for d in input_root.glob(pattern) if d.is_dir())
        default_output = ROOT / "results" / "synthetic" / "master.parquet"

    files = _collect_parquets(run_dirs)

    if args.since_hours is not None:
        cutoff = time.time() - args.since_hours * 3600.0
        files = [f for f in files if f.stat().st_mtime >= cutoff]

    if not files:
        print(f"[concat] no parquet files found under {input_root} "
              + (f"run={args.run!r}" if args.run else f"pattern={args.pattern!r}")
              + (f" since-hours={args.since_hours}" if args.since_hours else ""))
        return 1

    dfs = []
    columns_seen: set = set()
    for f in files:
        df = pd.read_parquet(f)
        df["source_file"] = str(f.relative_to(input_root))
        # Pull task/seed out of the nested task directory and timestamp from its parent run.
        task_match = TASK_DIR_RE.match(f.parent.name)
        run_match = DIR_RE.match(f.parent.parent.name if task_match else f.parent.name)
        df["array_task_id"] = int(task_match.group("task")) if task_match else None
        df["run_seed"] = int(task_match.group("seed")) if task_match else None
        df["run_timestamp"] = run_match.group("ts") if run_match else None
        if columns_seen and set(df.columns) != columns_seen:
            diff_added = set(df.columns) - columns_seen
            diff_dropped = columns_seen - set(df.columns)
            msg = (f"[concat] schema mismatch in {f.relative_to(input_root)}: "
                   f"+{sorted(diff_added)} -{sorted(diff_dropped)}")
            if args.strict_schema:
                print(msg, file=sys.stderr)
                return 2
            print(msg)
        columns_seen |= set(df.columns)
        dfs.append(df)
        print(f"[concat] {f.relative_to(input_root)}: {len(df)} rows")

    master = pd.concat(dfs, ignore_index=True)

    out = Path(args.output) if args.output else default_output
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    master.to_parquet(tmp, index=False)
    import os as _os
    _os.replace(tmp, out)
    print(f"[concat] wrote {len(master)} rows from {len(files)} parquets → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
