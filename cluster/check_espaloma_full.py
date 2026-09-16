"""Did the full-pool espaloma arms produce comparable held-out predictions?

Reads content: rows per arm, the encoder column matching the arm asked for,
finite RMSE in a physical range, and that both arms held out the SAME scans.
The last is what a file count cannot see, and it is the check that caught two
arms answering different questions earlier in this project.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="/work/ben/torsion_phase/espaloma_full")
    p.add_argument("--min-rows", type=int, default=1000)
    args = p.parse_args()

    problems, tables = [], {}
    for arm in ("frozen", "finetuned_reference"):
        path = pathlib.Path(args.root) / arm / "per_molecule.csv"
        if not path.exists():
            problems.append(f"{arm}: {path} missing")
            continue
        with open(path) as fh:
            rows = list(csv.DictReader(fh))
        if len(rows) < args.min_rows:
            problems.append(f"{arm}: {len(rows):,} rows, under {args.min_rows:,}")
        if not rows:
            continue
        declared = {r.get("encoder", "") for r in rows}
        if declared != {arm}:
            problems.append(f"{arm}: table says encoder={declared}")
        values = np.array([float(r["rmse"]) for r in rows if r.get("rmse")])
        if not values.size or not np.isfinite(values).all():
            problems.append(f"{arm}: rmse empty or not finite")
        elif not (0.0 < np.median(values) < 20.0):
            problems.append(f"{arm}: median rmse {np.median(values):.3f} is not "
                            f"a torsion profile error")
        tables[arm] = {(r["molecule_id"], r.get("driven_bond", "")) for r in rows}
        print(f"{arm:<22}{len(rows):>7} rows, median rmse "
              f"{np.median(values):.4f} kcal/mol")

    if len(tables) == 2:
        a, b = tables["frozen"], tables["finetuned_reference"]
        shared = a & b
        print(f"held out by both: {len(shared):,}; frozen only {len(a - b):,}; "
              f"fine-tuned only {len(b - a):,}")
        if len(shared) < 0.95 * max(len(a), len(b)):
            problems.append(f"only {len(shared):,} scans shared, under 95 percent "
                            f"of the larger arm: the splits disagree")

    for problem in problems:
        print(f"  PROBLEM {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
