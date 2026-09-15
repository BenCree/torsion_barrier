"""Did the THEMol conversion produce usable scans, or well-formed emptiness?

Reads the npz's own content: how many scans it holds, that every one has its
four arrays, that the geometries and energies agree in length, and that the
barriers are physical. A file count would pass on three empty archives, which is
how ERRORS #285 happened.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np


def check(path, minimum):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    problems = []
    if len(meta) < minimum:
        problems.append(f"{len(meta):,} scans, fewer than the {minimum:,} asked for")
    step = max(1, len(meta) // 200)
    barriers = []
    for entry in meta[::step]:
        key = entry["key"]
        for field in ("xyz", "energy", "bonds", "bond_orders"):
            if f"{field}::{key}" not in z:
                problems.append(f"scan {key} has no {field}")
        xyz = z[f"xyz::{key}"]
        energy = z[f"energy::{key}"]
        if len(xyz) != len(energy) or len(xyz) != len(entry["grid"]):
            problems.append(f"scan {key}: {len(xyz)} frames, {len(energy)} "
                            f"energies, {len(entry['grid'])} grid points")
        if xyz.shape[1] != entry["n_atoms"]:
            problems.append(f"scan {key}: {xyz.shape[1]} atoms against "
                            f"{entry['n_atoms']} declared")
        if not np.isfinite(energy).all() or abs(energy.min()) > 1e-6:
            problems.append(f"scan {key}: energies are not finite and "
                            f"minimum-referenced")
        barriers.append(float(energy.max()))
    barriers = np.asarray(barriers)
    if barriers.size and (np.median(barriers) > 60 or np.median(barriers) <= 0):
        problems.append(f"median barrier {np.median(barriers):.2f} kcal/mol is "
                        f"not a torsion barrier; check the energy units")
    return len(meta), barriers, problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/work/ben/torsion_phase/data/themol")
    parser.add_argument("--sizes", default="15000,150000,1500000")
    args = parser.parse_args()

    failed = 0
    for size in (int(s) for s in args.sizes.split(",")):
        path = pathlib.Path(args.root) / f"themol_{size}.npz"
        if not path.exists():
            print(f"MISSING  {path}")
            failed += 1
            continue
        try:
            n, barriers, problems = check(path, int(size * 0.9))
        except Exception as error:                       # noqa: BLE001
            print(f"UNREADABLE  {path}: {type(error).__name__}: {error}")
            failed += 1
            continue
        verdict = "OK      " if not problems else "PROBLEM "
        print(f"{verdict}{path.name}: {n:,} scans, median barrier "
              f"{np.median(barriers):.2f} kcal/mol")
        for problem in problems[:5]:
            print(f"           {problem}")
        failed += bool(problems)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
