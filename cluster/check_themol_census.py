"""Did the census produce usable rows, or well-formed emptiness?

Reads the outputs' own content: how many scans each shard decomposed, that the
fitted quantities are in range, and THAT THE NAMED CONTROLS BEHAVE. The last is
what separates a working census from a broken Fourier fit, and a file count
cannot see it: ethane must come back with no asymmetry, and a shard where it
does not is not a shard with fewer rows, it is a shard whose numbers are wrong.
"""
from __future__ import annotations

import argparse
import csv
import glob
import pathlib
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/work/ben/torsion_phase/census")
    parser.add_argument("--min-rows", type=int, default=20000,
                        help="a shard holds about 71,000 usable scans")
    args = parser.parse_args()

    files = sorted(glob.glob(str(pathlib.Path(args.root) / "census_*.csv")))
    if not files:
        print(f"no census output under {args.root}")
        return 1

    problems, total = [], 0
    odd_all, ceiling_all = [], []
    for path in files:
        with open(path) as fh:
            rows = list(csv.DictReader(fh))
        name = pathlib.PurePath(path).name
        if len(rows) < args.min_rows:
            problems.append(f"{name}: {len(rows):,} rows, under {args.min_rows:,}")
        if not rows:
            continue
        total += len(rows)
        odd = np.array([float(r["odd_fraction"]) for r in rows])
        ceiling = np.array([float(r["rms_full"]) for r in rows])
        js = np.array([float(r["js_distance"]) for r in rows])
        odd_all.append(np.median(odd))
        ceiling_all.append(np.median(ceiling))
        if not ((odd >= 0) & (odd <= 1)).all():
            problems.append(f"{name}: odd_fraction outside [0, 1]")
        if not ((js >= 0) & (js <= 1.001)).all():
            problems.append(f"{name}: js_distance outside [0, 1]")
        if np.median(ceiling) > 1.0:
            problems.append(f"{name}: six-harmonic residual median "
                            f"{np.median(ceiling):.3f} kcal/mol is too large for "
                            f"a torsion profile; check the energy units")

    named = sorted(glob.glob(str(pathlib.Path(args.root) / "named_*.csv")))
    ethane_seen, ethane_bad = 0, 0
    for path in named:
        with open(path) as fh:
            for row in csv.DictReader(fh):
                if row["name"] != "ethane" or not int(row["present"]):
                    continue
                ethane_seen += 1
                if float(row["odd_fraction"]) > 0.01:
                    ethane_bad += 1
    if ethane_seen and ethane_bad:
        problems.append(f"ethane came back asymmetric in {ethane_bad} of "
                        f"{ethane_seen} shards that contain it; the fit is wrong")

    print(f"{len(files)} shards, {total:,} scans decomposed")
    if odd_all:
        print(f"  median odd fraction across shards {np.median(odd_all):.3f}")
        print(f"  median six-harmonic residual      {np.median(ceiling_all):.4f} kcal/mol")
    print(f"  ethane present in {ethane_seen} shards, asymmetric in {ethane_bad}")
    for problem in problems[:10]:
        print(f"  PROBLEM {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
