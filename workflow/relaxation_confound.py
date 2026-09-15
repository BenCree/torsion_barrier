"""Is the odd part the rotor's chemistry, or the rest of the molecule relaxing?

QUESTION. A torsiondrive re-optimises every other degree of freedom at each
constrained angle. So a profile's asymmetric component could be a property of
the bond, or it could be the rest of the molecule settling differently on the
way round. Methanediol is the case that raised it: achiral, and 0.582 odd.

WHAT WOULD ANSWER IT EITHER WAY. Per scan, the odd fraction against how much the
non-rotating part of the molecule actually moves across its own frames. A strong
positive correlation says the census has been measuring relaxation; none says
the asymmetry belongs to the rotor.

HOW. The molecule is split at the driven bond and the LARGER fragment kept. That
fragment does not rotate about the bond, so superposing it on its own first
frame and taking the residual heavy-atom RMSD removes the rotation by
construction and leaves internal relaxation. The spread of that RMSD across the
24 frames is the number.

CONTROLS. Rotors whose substituents are symmetry-equivalent have essentially no
odd content; if relaxation were the cause they would still show the same
relaxation spread, so the two groups' spreads are reported side by side. The
named molecules are carried through, methanediol above all.

UNITS. Angstrom for relaxation, dimensionless for the odd fraction.

COST. Seconds, CPU.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib

import numpy as np


def fragment_sides(bonds, n_atoms, j, k):
    """The two atom sets the driven bond separates, by flood fill."""
    graph = {i: set() for i in range(n_atoms)}
    for a, b in bonds:
        a, b = int(a), int(b)
        if {a, b} == {j, k}:
            continue
        graph[a].add(b)
        graph[b].add(a)
    seen, stack = {j}, [j]
    while stack:
        node = stack.pop()
        for nxt in graph[node]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen, set(range(n_atoms)) - seen


def internal_rmsd(frames, indices):
    """Residual heavy-atom RMSD of one fragment against its own first frame.

    Superposed by Kabsch each time, so rigid rotation and translation are gone
    and what remains is the fragment deforming.
    """
    if len(indices) < 3:
        return float("nan")
    reference = frames[0][indices]
    reference = reference - reference.mean(0)
    out = []
    for frame in frames[1:]:
        x = frame[indices]
        x = x - x.mean(0)
        u, _, vt = np.linalg.svd(x.T @ reference)
        d = np.sign(np.linalg.det(u @ vt))
        rotation = u @ np.diag([1.0, 1.0, d]) @ vt
        out.append(float(np.sqrt(((x @ rotation - reference) ** 2).sum(1).mean())))
    return float(np.max(out)) if out else float("nan")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--torsiondrives", required=True)
    p.add_argument("--census", required=True,
                   help="the phase census over the same cache")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    from scipy.stats import spearmanr

    with open(args.census) as fh:
        census = {r["molecule_id"]: r for r in csv.DictReader(fh)}

    cache = np.load(args.torsiondrives, allow_pickle=False)
    meta = json.loads(str(cache["meta"]))
    if args.limit:
        meta = meta[:args.limit]

    rows = []
    for entry in meta:
        record = census.get(entry["molecule_id"])
        if record is None:
            continue
        key = entry["key"]
        driven = entry.get("driven_dihedral")
        if not driven or len(driven) != 4:
            continue
        j, k = int(driven[1]), int(driven[2])
        symbols = entry["symbols"]
        heavy = {i for i, s in enumerate(symbols) if s != "H"}
        left, right = fragment_sides(cache[f"bonds::{key}"], len(symbols), j, k)
        # The larger side, so the measurement has atoms to work with.
        side = sorted((left if len(left & heavy) >= len(right & heavy) else right)
                      & heavy)
        frames = np.asarray(cache[f"xyz::{key}"], dtype=float)
        rows.append({
            "molecule_id": entry["molecule_id"],
            "n_heavy_fragment": len(side),
            "relaxation_angstrom": internal_rmsd(frames, side),
            "odd_fraction": float(record["odd_fraction"]),
            "js_distance": float(record["js_distance"]),
            "substituents_equivalent": int(record["substituents_equivalent"]),
            "classes": record["classes"],
        })

    rows = [r for r in rows if np.isfinite(r["relaxation_angstrom"])]
    if not rows:
        print("nothing measured")
        return 1
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    relax = np.array([r["relaxation_angstrom"] for r in rows])
    odd = np.array([r["odd_fraction"] for r in rows])
    rho = spearmanr(relax, odd)
    print(f"{len(rows):,} scans with a fragment of 3 or more heavy atoms")
    print(f"  fragment relaxation: median {np.median(relax):.3f} A, "
          f"90th {np.percentile(relax, 90):.3f}")
    print(f"  rho(relaxation, odd fraction) = {rho.statistic:+.3f}, "
          f"p = {rho.pvalue:.3g}")

    print(f"\n{'group':<26}{'n':>7}{'relaxation':>12}{'odd':>8}")
    for label, mask in (
        ("substituents equivalent", np.array([r["substituents_equivalent"] == 1
                                              for r in rows])),
        ("substituents distinct", np.array([r["substituents_equivalent"] == 0
                                            for r in rows])),
        ("odd fraction below 0.01", odd < 0.01),
        ("odd fraction above 0.20", odd > 0.20),
    ):
        if not mask.any():
            continue
        print(f"  {label:<24}{int(mask.sum()):>7}"
              f"{np.median(relax[mask]):12.3f}{np.median(odd[mask]):8.3f}")
    # THE DECISIVE CUT. If relaxation were the cause, then AT MATCHED
    # RELAXATION a rotor with symmetry-equivalent substituents and one without
    # would carry the same odd content. Stratifying removes relaxation as an
    # explanation rather than arguing about a correlation of 0.2.
    print(f"\nAt matched relaxation, odd fraction by whether the rotor's")
    print(f"substituents are symmetry-equivalent:")
    print(f"\n{'relaxation band (A)':<24}{'equivalent':>18}{'distinct':>18}")
    equivalent = np.array([r["substituents_equivalent"] == 1 for r in rows])
    edges = np.percentile(relax, [0, 20, 40, 60, 80, 100])
    for low, high in zip(edges[:-1], edges[1:]):
        band = (relax >= low) & (relax <= high)
        a, b = band & equivalent, band & ~equivalent
        if a.sum() < 20 or b.sum() < 20:
            continue
        print(f"  {low:.3f} to {high:.3f}{'':<8}"
              f"{np.median(odd[a]):10.3f} (n={int(a.sum()):>4})"
              f"{np.median(odd[b]):10.3f} (n={int(b.sum()):>5})")

    # And the other direction: among rotors that barely relax at all, is there
    # still asymmetry? If yes it cannot have come from relaxation.
    still = relax < np.percentile(relax, 25)
    print(f"\nAmong the quarter that relax least (under "
          f"{np.percentile(relax, 25):.3f} A): "
          f"median odd {np.median(odd[still]):.3f}, "
          f"{int((odd[still] > 0.2).sum()):,} of {int(still.sum()):,} "
          f"still above 0.20")
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
