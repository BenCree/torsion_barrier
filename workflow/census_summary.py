"""Aggregate the sharded phase census into one table of class contrasts.

QUESTION. None on its own; the census asked it. This reduces 2.4 million rows to
the cuts the question is about, streaming so no shard is ever fully resident.

UNITS. kcal/mol, degrees, Jensen-Shannon distance in bits. Intervals are
bootstrapped over molecules.
"""
from __future__ import annotations

import argparse
import csv
import glob
import pathlib

import numpy as np

CLASSES = ("stereocentre_adjacent", "plain", "gauche_effect", "anomeric",
           "amide", "biaryl", "conjugated")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--census", required=True, help="directory of shard CSVs")
    p.add_argument("--pattern", default="census_*.csv")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    buckets = {}

    def add(key, row):
        b = buckets.setdefault(key, {"n": 0, "odd": [], "gain": [], "shift": [],
                                     "js": [], "rms6": [], "rms3": []})
        b["n"] += 1
        for name, column in (("odd", "odd_fraction"), ("gain", "phase_gain"),
                             ("shift", "minimum_shift_degrees"),
                             ("js", "js_distance"), ("rms6", "rms_full"),
                             ("rms3", "rms_three_harmonics")):
            try:
                b[name].append(float(row[column]))
            except (KeyError, TypeError, ValueError):
                pass

    files = sorted(glob.glob(str(pathlib.Path(args.census) / args.pattern)))
    for path in files:
        with open(path) as fh:
            for row in csv.DictReader(fh):
                add("ALL", row)
                add("substituents equivalent" if row["substituents_equivalent"] == "1"
                    else "substituents distinct", row)
                for cls in row["classes"].split("|"):
                    if cls in CLASSES:
                        add(cls, row)

    rows = []
    print(f"{len(files)} shards")
    print(f"\n{'cut':<26}{'n':>10}{'odd':>7}{'gain':>8}{'min shift':>11}{'JS':>8}")
    for key in ["ALL", "substituents equivalent", "substituents distinct", *CLASSES]:
        b = buckets.get(key)
        if not b or not b["odd"]:
            continue
        record = {"cut": key, "n": b["n"]}
        for name in ("odd", "gain", "shift", "js", "rms6", "rms3"):
            values = np.asarray(b[name], dtype=float)
            values = values[np.isfinite(values)]
            record[f"median_{name}"] = float(np.median(values)) if values.size else float("nan")
            record[f"p90_{name}"] = float(np.percentile(values, 90)) if values.size else float("nan")
        rows.append(record)
        print(f"  {key:<24}{b['n']:>10,}{record['median_odd']:7.3f}"
              f"{record['median_gain']:8.3f}{record['median_shift']:10.1f}d"
              f"{record['median_js']:8.3f}")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    total = buckets["ALL"]
    print(f"\nceilings over {total['n']:,} scans: six harmonics "
          f"{np.median(total['rms6']):.4f} kcal/mol, three "
          f"{np.median(total['rms3']):.4f}")
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
