"""Scan supervision against bond supervision, paired on the same held-out scans.

QUESTION. Each torsiondrive currently embeds its own reference geometry, so a
molecule scanned at bond A and at bond B yields two different representations of
the same bond and nothing says they should agree. Does supervising every scanned
bond of a molecule off ONE shared embedding beat supervising one bond per
geometry? What answers it either way: the paired per-scan difference in profile
RMSE on identical held-out rows, with an interval resampled over molecules, and
the sign test across scans beside it.

ON WHAT. Two `per_molecule` tables written by the same script, the same cohort,
the same split, the same seed, differing only in --supervision. Joined on
(molecule_id, driven_bond), which is the scan's identity; a molecule scanned at
two rotors is two rows and must stay two rows.

HOW. Inner join, so only scans both arms held out are compared. The count in and
the count out are both reported: an arm that quietly held out different rows
would otherwise look better by having answered an easier question.

CONTROLS. Both arms carry MMFF94 and the flat profile in their own tables, and
those columns are joined through so that an arm beating the other still has to
beat them. The sign test is computed here and reported beside the verdicts,
because `relate` expresses correlations and not paired differences.

UNITS. kcal/mol. One row is one (molecule, driven bond) scan.

COST. A join, seconds, CPU.
"""
from __future__ import annotations

import argparse
import csv
import pathlib


def read(path, arm):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    for row in rows:
        key = (row["molecule_id"], row.get("driven_bond", ""))
        out[key] = row
    if len(out) != len(rows):
        print(f"  {arm}: {len(rows)} rows collapsed to {len(out)} identities; "
              f"a (molecule, driven bond) pair appears twice")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", required=True, help="per_molecule of --supervision scan")
    parser.add_argument("--bond", required=True, help="per_molecule of --supervision bond")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import numpy as np
    from scipy import stats

    left, right = read(args.scan, "scan"), read(args.bond, "bond")
    shared = sorted(set(left) & set(right))
    print(f"{len(left)} scan rows, {len(right)} bond rows, {len(shared)} shared")
    if not shared:
        print("no scan was held out by both arms; nothing to compare")
        return 1

    rows = []
    for key in shared:
        a, b = left[key], right[key]
        def number(row, name):
            try:
                return float(row[name])
            except (KeyError, TypeError, ValueError):
                return float("nan")
        rows.append({
            "molecule_id": key[0],
            "driven_bond": key[1],
            "smiles": a.get("smiles", ""),
            "n_atoms": a.get("n_atoms", ""),
            "n_rotors": a.get("n_rotors", ""),
            "qm_barrier_kcal_mol": number(a, "qm_barrier_kcal_mol"),
            "rmse_scan": number(a, "rmse"),
            "rmse_bond": number(b, "rmse"),
            "rmse_mmff94": number(a, "rmse_mmff94"),
            "rmse_flat": number(a, "rmse_flat"),
            "rho_scan": number(a, "rho"),
            "rho_bond": number(b, "rho"),
            "improvement": number(a, "rmse") - number(b, "rmse"),
        })

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    scan = np.asarray([r["rmse_scan"] for r in rows])
    bond = np.asarray([r["rmse_bond"] for r in rows])
    keep = np.isfinite(scan) & np.isfinite(bond)
    scan, bond = scan[keep], bond[keep]
    better = int((bond < scan).sum())
    ties = int((bond == scan).sum())
    # A sign test across scans, reported beside the verdicts because `relate`
    # computes correlations and a paired difference is not one.
    p = float(stats.binomtest(better, better + int((bond > scan).sum()), 0.5).pvalue)

    # Resampled over MOLECULES, not scans: a molecule scanned at three rotors
    # gives three correlated rows and resampling them independently would report
    # an interval narrower than the data supports.
    ids = np.asarray([r["molecule_id"] for r in rows])[keep]
    unique = np.unique(ids)
    rng = np.random.default_rng(0)
    medians = []
    for _ in range(2000):
        draw = rng.choice(unique, size=unique.size, replace=True)
        pick = np.concatenate([np.flatnonzero(ids == m) for m in draw])
        medians.append(float(np.median(scan[pick] - bond[pick])))
    low, high = float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))

    print(f"median RMSE   scan {np.median(scan):.4f}   bond {np.median(bond):.4f}")
    print(f"median paired improvement {np.median(scan - bond):+.4f} kcal/mol "
          f"[{low:+.4f}, {high:+.4f}] over {unique.size} molecules")
    print(f"sign test: bond supervision better on {better} of {scan.size} scans "
          f"({ties} ties), p = {p:.3g}")
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
