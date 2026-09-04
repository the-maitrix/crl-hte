"""Export aggregate experiment rows without individual-level prediction vectors."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--outcome-source", choices=["predicted", "true"])
    args = parser.parse_args()
    source, output = Path(args.input), Path(args.output)
    frame = pd.read_parquet(source) if source.suffix == ".parquet" else pd.read_csv(source)
    frame = frame.drop(columns=["tau_hat", "tau_true"], errors="ignore")
    if args.outcome_source:
        frame["outcome_source"] = (
            "observed_y" if args.outcome_source == "true" else args.outcome_source
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"wrote {len(frame)} aggregate rows to {output}")


if __name__ == "__main__":
    main()
