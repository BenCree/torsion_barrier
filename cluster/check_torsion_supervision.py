"""Did both supervision arms produce comparable held-out predictions?

Reads the tables' own content: that each arm wrote rows, that the supervision
column says what the task was asked for, that the RMSE column is finite and in a
physical range, and THAT THE TWO ARMS HELD OUT THE SAME SCANS. The last one is
the check that matters: an arm answering an easier question looks better, and a
file count cannot see it.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import numpy as np


def load(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/work/ben/torsion_phase/supervision")
    args = parser.parse_args()

    problems, tables = [], {}
    for mode in ("scan", "bond"):
        path = pathlib.Path(args.root) / mode / "per_molecule.csv"
        if not path.exists():
            problems.append(f"{mode}: {path} is missing")
            continue
        rows = load(path)
        if not rows:
            problems.append(f"{mode}: no rows")
            continue
        declared = {r.get("supervision", "") for r in rows}
        if declared != {mode}:
            problems.append(f"{mode}: the table says supervision={declared}")
        values = np.asarray([float(r["rmse"]) for r in rows if r.get("rmse")])
        if not values.size or not np.isfinite(values).all():
            problems.append(f"{mode}: rmse is empty or not finite")
        elif not (0.0 < np.median(values) < 20.0):
            problems.append(f"{mode}: median rmse {np.median(values):.3f} "
                            f"kcal/mol is not a torsion profile error")
        tables[mode] = {(r["molecule_id"], r.get("driven_bond", "")) for r in rows}
        print(f"{mode:<6}{len(rows):>7} rows, median rmse "
              f"{np.median(values):.4f} kcal/mol, "
              f"{len(tables[mode]):,} distinct scans")

    if len(tables) == 2:
        shared = tables["scan"] & tables["bond"]
        only_scan = len(tables["scan"] - tables["bond"])
        only_bond = len(tables["bond"] - tables["scan"])
        print(f"held out by both: {len(shared):,}; "
              f"scan only {only_scan:,}; bond only {only_bond:,}")
        if not shared:
            problems.append("the two arms share no held-out scan; nothing to pair")
        elif len(shared) < 0.8 * max(len(tables["scan"]), len(tables["bond"])):
            problems.append(f"only {len(shared):,} scans are shared, under 80 "
                            f"percent of the larger arm: the splits disagree")

    for problem in problems:
        print(f"  PROBLEM {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
