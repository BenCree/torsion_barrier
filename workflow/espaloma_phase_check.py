"""Does espaloma's TRAINED model reproduce the odd part of a torsion profile?

QUESTION. The phase census showed that a free phase buys 0.021 kcal/mol on the
746 rotors whose substituents are symmetry-equivalent, and that the odd content
lives instead on rotors beside a stereocentre, 3,570 scans at a median odd
fraction of 0.333. Algebra says espaloma's functional form CAN reach those: a
fixed phase per torsion, summed over torsions whose substituent offsets differ,
spans an arbitrary bond phase. That is a statement about the family, not about
the trained network. Nothing in espaloma's loss singles out the odd component,
so whether it lands there is an empirical question and this is it.

WHAT WOULD ANSWER IT EITHER WAY. Per scan, the odd component of espaloma's own
predicted torsion profile against the odd component of the QM profile: the ratio
of their magnitudes, and the cosine similarity of the two six-vectors b_n. A
ratio near 1 with a similarity near 1 says espaloma already has the phase and
the question is closed. A ratio near 0 says the form permits it and the trained
model does not produce it, and the phase is unclaimed.

ON WHAT. The merged pool, on the geometries of each scan's own frames. One
independent unit is a molecule.

HOW. espaloma 0.3.2 emits K_n for every proper torsion, so its torsion energy at
a frame is sum_t sum_n K_n,t cos(n phi_t). That is evaluated on the QM geometries
and decomposed with the same least squares the census used. Two variants,
because they answer different questions: the torsions about the DRIVEN bond
alone, which is what our head predicts, and every proper torsion in the
molecule, which is what espaloma actually contributes to an MM energy.

WHAT THIS IS NOT. Not an accuracy comparison. espaloma's torsion term is one
piece of an MM energy that also has bonds, angles and 1-4 nonbonded, so its
absolute offset from the QM profile is meaningless here and is not reported.
Only the SHAPE of the odd component is, which is well defined for both.

CONTROLS. The same named molecules the census used: ethane and biphenyl must
come back with no odd content from either side, and 2-butanol and
2-methoxytetrahydropyran with odd content from QM.

UNITS. kcal/mol for the b_n amplitudes, dimensionless for ratios.

COST. One espaloma forward per molecule, about 5.5 ms, plus arithmetic.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib

import numpy as np

PERIODICITIES = 6

# espaloma/units.py:21 sets ENERGY_UNIT = HARTREE_PER_PARTICLE, so the field
# named `k` on n4 is in HARTREE. Reading it as kcal/mol makes every predicted
# barrier 627 times too small, which on a first pass looked like espaloma
# predicting no torsional energy at all rather than like a unit error.
HARTREE_KCAL = 627.5094740631


def dihedrals(frame, quads):
    """(n_torsions,) dihedrals in radians, the RDKit and torsiondrive convention."""
    p0, p1, p2, p3 = (frame[quads[:, i]] for i in range(4))
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1, axis=1, keepdims=True)
    v = b0 - (b0 * b1).sum(1, keepdims=True) * b1
    w = b2 - (b2 * b1).sum(1, keepdims=True) * b1
    return np.arctan2((np.cross(b1, v) * w).sum(1), (v * w).sum(1))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--torsiondrives", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--quarantine", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--espaloma-version", default="0.3.2")
    args = p.parse_args()

    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    import espaloma as esp
    import torch
    from espaloma_encoder import graph_of
    from phase_census import harmonics

    model = esp.get_model(args.espaloma_version).eval()
    cache = np.load(args.torsiondrives, allow_pickle=False)
    meta = json.loads(str(cache["meta"]))
    if args.limit:
        meta = meta[:args.limit]

    rows, quarantine = [], []
    for entry in meta:
        key = entry["key"]
        try:
            driven = entry.get("driven_dihedral")
            if not driven or len(driven) != 4:
                raise ValueError("no driven dihedral")
            j, k = int(driven[1]), int(driven[2])
            graph = graph_of(entry["symbols"], cache[f"bonds::{key}"],
                             cache[f"bond_orders::{key}"])
            with torch.no_grad():
                out = model(graph)
            quads = out.nodes["n4"].data["idxs"].cpu().numpy().astype(int)
            coefficients = out.nodes["n4"].data["k"].cpu().numpy() * HARTREE_KCAL
            if not len(quads):
                raise ValueError("espaloma found no proper torsion")
            # deploy.py:241: "assuming both (a,b,c,d) and (d,c,b,a) are listed
            # for every torsion, only pick one of the orderings". A reversed
            # torsion has the SAME dihedral, so summing all of them is exactly
            # twice the energy. espaloma's own OpenMM export takes idx0 < idx3.
            keep = quads[:, 0] < quads[:, 3]
            if keep.any():
                quads, coefficients = quads[keep], coefficients[keep]
            about_driven = np.array([{int(q[1]), int(q[2])} == {j, k} for q in quads])

            xyz = np.asarray(cache[f"xyz::{key}"], dtype=float)
            n = np.arange(1, PERIODICITIES + 1)
            energies = {"all": [], "driven": []}
            for frame in xyz:
                phi = dihedrals(frame, quads)
                terms = (coefficients * np.cos(phi[:, None] * n[None, :])).sum(1)
                energies["all"].append(float(terms.sum()))
                energies["driven"].append(float(terms[about_driven].sum()))

            grid = entry["grid"]
            qm = harmonics(grid, cache[f"energy::{key}"])
            row = {
                "molecule_id": entry["molecule_id"],
                "smiles": entry.get("smiles", ""),
                "driven_bond": f"{j}-{k}",
                "n_torsions_about_bond": int(about_driven.sum()),
                "qm_odd_magnitude": float(np.linalg.norm(qm["b"])),
                "qm_even_magnitude": float(np.linalg.norm(qm["a"])),
            }
            for variant in ("driven", "all"):
                fit = harmonics(grid, energies[variant])
                qb, eb = qm["b"], fit["b"]
                denominator = np.linalg.norm(qb) * np.linalg.norm(eb)
                row[f"esp_{variant}_odd_magnitude"] = float(np.linalg.norm(eb))
                row[f"esp_{variant}_odd_ratio"] = float(
                    np.linalg.norm(eb) / np.linalg.norm(qb)) if np.linalg.norm(qb) > 1e-9 else float("nan")
                row[f"esp_{variant}_odd_similarity"] = float(
                    (qb * eb).sum() / denominator) if denominator > 1e-12 else float("nan")
            rows.append(row)
        except Exception as error:                       # noqa: BLE001
            quarantine.append((entry.get("molecule_id", "?"),
                               f"{type(error).__name__}: {error}"))

    if not rows:
        print("nothing scored")
        return 1
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pathlib.Path(args.quarantine).open("w") as fh:
        fh.write("molecule_id,reason\n")
        for name, reason in quarantine:
            fh.write(f'{name},"{reason}"\n')

    qm_odd = np.array([r["qm_odd_magnitude"] for r in rows])
    # Only scans with real odd content can say anything about reproducing it.
    strong = qm_odd > np.percentile(qm_odd, 75)
    print(f"{len(rows):,} scans, {len(quarantine):,} quarantined")
    print(f"QM odd magnitude: median {np.median(qm_odd):.4f} kcal/mol, "
          f"upper quartile above {np.percentile(qm_odd, 75):.4f}")
    for variant in ("driven", "all"):
        ratio = np.array([r[f"esp_{variant}_odd_ratio"] for r in rows], dtype=float)
        similarity = np.array([r[f"esp_{variant}_odd_similarity"] for r in rows], dtype=float)
        for label, mask in (("all scans", np.ones(len(rows), bool)),
                            ("top quartile by QM odd content", strong)):
            r, s = ratio[mask], similarity[mask]
            r, s = r[np.isfinite(r)], s[np.isfinite(s)]
            if not r.size:
                continue
            print(f"  espaloma[{variant}], {label:<32}"
                  f"odd ratio median {np.median(r):6.3f}   "
                  f"direction similarity median {np.median(s):+6.3f}  n={r.size}")
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
