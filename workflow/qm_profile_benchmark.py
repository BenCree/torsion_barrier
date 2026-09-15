"""Every model's full 24-point profile against QM, on one set of scans.

QUESTION. Which torsion model predicts a complete QM profile best, per bond, at
every angle? Not the pose angle, not the barrier, the whole curve.

ON WHAT. The merged pool's scans. `--multi-rotor-only` restricts to the 3,669
scans from the 1,164 molecules that separate OpenFF datasets scanned at two or
more rotors, which is the same holdable set job 79351 uses, so these numbers and
that job's are on identical rows.

HOW. Every model becomes the same object, (periodicity, phase, force constant)
per proper torsion, and the profile of the driven bond is the sum over the
torsions about it evaluated at the scan's own 24 geometries. Sage 2.3.0 from its
OpenMM PeriodicTorsionForce; espaloma 0.3.2 from its n4 coefficients, which are
HARTREE and listed twice per torsion; MMFF94 by difference, since RDKit gives a
total rather than a torsion term, so its column is the whole force field and is
labelled that way.

CONTROLS. The flat profile on the same rows.

UNITS. kcal/mol, each profile relative to its own minimum. Degrees for how far
the predicted minimum sits from the QM one. Jensen-Shannon distance at 500 K.

COST. One force field system and one espaloma forward per scan. CPU.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib

import numpy as np

HARTREE_KCAL = 627.5094740631
BETA = 1.0 / (0.0019872041 * 500.0)


def metrics(predicted, truth, grid):
    predicted = np.asarray(predicted, float)
    predicted = predicted - predicted.min()
    truth = np.asarray(truth, float)
    truth = truth - truth.min()
    grid = np.asarray(grid, float)
    shift = float(grid[int(np.argmin(predicted))] - grid[int(np.argmin(truth))])

    def w(curve):
        x = np.exp(-BETA * np.clip(curve, 0, 700))
        return x / x.sum()

    p, q = w(predicted), w(truth)
    m = 0.5 * (p + q)
    kl = lambda x, y: float((x[(x > 0) & (y > 0)]
                             * np.log2(x[(x > 0) & (y > 0)] / y[(x > 0) & (y > 0)])).sum())
    return {"rmse": float(np.sqrt(np.mean((predicted - truth) ** 2))),
            "minimum_shift_degrees": abs((shift + 180.0) % 360.0 - 180.0),
            "js_distance": float(np.sqrt(max(0.5 * kl(p, m) + 0.5 * kl(q, m), 0)))}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--torsiondrives", required=True)
    p.add_argument("--forcefield", default="openff-2.3.0.offxml")
    p.add_argument("--espaloma-version", default="0.3.2")
    p.add_argument("--multi-rotor-only", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--quarantine", required=True)
    args = p.parse_args()

    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    from espaloma_encoder import rdkit_from_cache
    from fit_torsion_head import mmff94_profile
    from openff.toolkit import ForceField, Molecule
    import openmm
    from openmm import unit as u
    import espaloma as esp
    import torch

    field = ForceField(args.forcefield)
    model = esp.get_model(args.espaloma_version).eval()

    cache = np.load(args.torsiondrives, allow_pickle=False)
    meta = json.loads(str(cache["meta"]))
    if args.multi_rotor_only:
        by_molecule = {}
        for e in meta:
            by_molecule.setdefault(e.get("inchi_key") or e["molecule_id"], []).append(e)
        meta = [e for group in by_molecule.values() if
                len({tuple(sorted(x["driven_dihedral"][1:3])) for x in group}) > 1
                for e in group]
        print(f"multi-rotor only: {len(meta):,} scans")
    if args.limit:
        meta = meta[:args.limit]

    rows, quarantine = [], []
    for entry in meta:
        key = entry["key"]
        try:
            driven = entry["driven_dihedral"]
            j, k = int(driven[1]), int(driven[2])
            rdmol = rdkit_from_cache(entry["symbols"], cache[f"bonds::{key}"],
                                     cache[f"bond_orders::{key}"])
            molecule = Molecule.from_rdkit(rdmol, allow_undefined_stereo=True)
            if molecule.n_atoms != rdmol.GetNumAtoms():
                raise ValueError(f"OpenFF {molecule.n_atoms} vs RDKit "
                                 f"{rdmol.GetNumAtoms()} atoms")
            xyz = np.asarray(cache[f"xyz::{key}"], float)
            qm = np.asarray(cache[f"energy::{key}"], float)
            grid = entry["grid"]

            terms = {}
            system = field.create_openmm_system(molecule.to_topology())
            sage = []
            for force in system.getForces():
                if isinstance(force, openmm.PeriodicTorsionForce):
                    for i in range(force.getNumTorsions()):
                        a, b, c, d, n, phase, amp = force.getTorsionParameters(i)
                        sage.append(((a, b, c, d), int(n),
                                     phase.value_in_unit(u.radian),
                                     amp.value_in_unit(u.kilocalorie_per_mole)))
            terms["sage"] = sage

            graph = esp.Graph(molecule).heterograph
            with torch.no_grad():
                out = model(graph)
            quads = out.nodes["n4"].data["idxs"].cpu().numpy().astype(int)
            coefficients = out.nodes["n4"].data["k"].cpu().numpy() * HARTREE_KCAL
            keep = quads[:, 0] < quads[:, 3]
            if keep.any():
                quads, coefficients = quads[keep], coefficients[keep]
            terms["espaloma"] = [
                (tuple(int(x) for x in q), n + 1, 0.0, float(row[n]))
                for q, row in zip(quads, coefficients)
                for n in range(coefficients.shape[1]) if abs(row[n]) > 1e-12]

            from openff.nagl.utils._tensors import calculate_dihedrals
            for name, term_list in terms.items():
                about = [t for t in term_list if {t[0][1], t[0][2]} == {j, k}]
                if not about:
                    continue
                profile = np.zeros(len(xyz))
                for quad, n, phase, amp in about:
                    q = [int(x) for x in quad]
                    delta = np.array([calculate_dihedrals(
                        f[None, q[0]], f[None, q[1]], f[None, q[2]], f[None, q[3]])[0]
                        for f in xyz])
                    profile += amp * np.cos(n * delta - phase)
                rows.append(dict(molecule_id=entry["molecule_id"], model=name,
                                 driven_bond=f"{j}-{k}",
                                 n_heavy=sum(1 for s in entry["symbols"] if s != "H"),
                                 qm_barrier=float(np.ptp(qm)),
                                 **metrics(profile, qm, grid)))

            mmff = mmff94_profile(entry["symbols"], cache[f"bonds::{key}"],
                                  cache[f"bond_orders::{key}"], xyz)
            if mmff is not None:
                rows.append(dict(molecule_id=entry["molecule_id"],
                                 model="mmff94_total", driven_bond=f"{j}-{k}",
                                 n_heavy=sum(1 for s in entry["symbols"] if s != "H"),
                                 qm_barrier=float(np.ptp(qm)),
                                 **metrics(mmff, qm, grid)))
            rows.append(dict(molecule_id=entry["molecule_id"], model="flat",
                             driven_bond=f"{j}-{k}",
                             n_heavy=sum(1 for s in entry["symbols"] if s != "H"),
                             qm_barrier=float(np.ptp(qm)),
                             **metrics(np.zeros(len(qm)), qm, grid)))
        except Exception as error:                       # noqa: BLE001
            quarantine.append((entry.get("molecule_id", "?"),
                               f"{type(error).__name__}: {error}"))

    if not rows:
        print(f"nothing scored; first failure: {quarantine[:1]}")
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

    print(f"{len({r['molecule_id'] for r in rows}):,} molecules, "
          f"{len(rows):,} rows, {len(quarantine):,} failed")
    print(f"\n{'model':<16}{'RMSE':>8}{'min shift':>12}{'JS':>8}{'n':>8}")
    for name in sorted({r["model"] for r in rows}):
        subset = [r for r in rows if r["model"] == name]
        print(f"{name:<16}"
              f"{np.median([r['rmse'] for r in subset]):8.3f}"
              f"{np.median([r['minimum_shift_degrees'] for r in subset]):11.1f}d"
              f"{np.median([r['js_distance'] for r in subset]):8.3f}"
              f"{len(subset):8d}")
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
