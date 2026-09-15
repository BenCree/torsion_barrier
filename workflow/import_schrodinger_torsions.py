"""Convert a Schrodinger MLFF_test_data torsion set into this project's cache.

QUESTION. None on its own. This is a format conversion, separated from
everything that computes, so that TorsionTest2000 and TorsionNet500 arrive as the
same npz every rule here already reads and no scorer needs a second input path.

WHAT IT TAKES. A directory of the set's json files and an output npz. Nothing
about TorsionTest2000 is written into this file: TorsionNet500, tautobase and
anything else in that repository are a different --directory, not a second
script.

THE REFERENCE IS NOT OURS. TorsionTest2000 is wB97X-D3BJ/def2-TZVPD and
TorsionNet500 carries DLPNO-CCSD(T)/CBS beside it, while every molecule this
model was trained on is B3LYP-D3BJ/DZVP. Those are different references and this
project has already measured two of them sitting 0.767 kcal/mol apart on the same
497 molecules. The level of theory is written into the npz so that no number
computed from it can be quoted without it.

HOW. Energies are converted from Hartree to kcal/mol and expressed relative to
each scan's own minimum, because the offset between molecules is not a quantity a
torsion term is asked to predict. Connectivity comes from the COORDINATES via
`rdDetermineBonds` at the net charge the file states, not from the SMILES string,
so the bond indices are in the same order as the geometry with no atom mapping
step to get wrong. A scan whose bonds cannot be perceived is quarantined with its
reason and the rest continue.

UNITS. kcal/mol relative to the scan minimum; Angstrom; degrees.

COST. File reading and RDKit perception, seconds per thousand scans. No GPU.
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys

import numpy as np

HARTREE_KCAL = 627.5094740631


def energy_key(record):
    keys = [k for k in record if k.startswith("E[")]
    if not keys:
        raise ValueError("no energy key")
    return keys


def perceive(symbols, xyz, charge):
    """Bonds and bond orders from one geometry, in that geometry's atom order."""
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds
    from rdkit.Geometry import Point3D

    rw = Chem.RWMol()
    conformer = Chem.Conformer(len(symbols))
    for index, symbol in enumerate(symbols):
        rw.AddAtom(Chem.Atom(symbol))
        conformer.SetAtomPosition(index, Point3D(*map(float, xyz[index])))
    molecule = rw.GetMol()
    molecule.AddConformer(conformer)
    rdDetermineBonds.DetermineBonds(molecule, charge=int(charge))
    bonds = np.asarray(
        [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in molecule.GetBonds()],
        dtype=np.int32,
    )
    orders = np.asarray(
        [b.GetBondTypeAsDouble() for b in molecule.GetBonds()], dtype=np.float64
    )
    return bonds, orders, Chem.MolToSmiles(Chem.RemoveHs(molecule))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True,
                        help="a directory of that set's json files")
    parser.add_argument("--pattern", default="*.json")
    parser.add_argument("--reference", required=True,
                        help="which energy key to take as the label, e.g. "
                             "'E[wB97X-D3BJ/def2-TZVPD](Ha)'")
    parser.add_argument("--dataset", required=True,
                        help="the name written into the npz, e.g. TorsionTest2000")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--quarantine", required=True)
    args = parser.parse_args()

    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    paths = sorted(glob.glob(str(pathlib.Path(args.directory) / args.pattern)))
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        print(f"no files matched {args.pattern!r} in {args.directory}")
        return 1

    arrays, meta, quarantine = {}, [], []
    for path in paths:
        name = pathlib.PurePath(path).stem
        try:
            scan = json.load(open(path))
            if len(scan) < 4:
                raise ValueError(f"{len(scan)} points is too few for a scan")
            if args.reference not in scan[0]:
                raise ValueError(
                    f"no {args.reference}; file has {energy_key(scan[0])}")
            atoms = scan[0]["torsion_atoms"]
            order = np.argsort([float(r["torsion_angle"]) for r in scan])
            scan = [scan[int(i)] for i in order]

            symbols = list(scan[0]["elements"])
            xyz = np.asarray([r["coordinates"] for r in scan], dtype=np.float64)
            if any(list(r["elements"]) != symbols for r in scan):
                raise ValueError("the elements change along the scan")
            energies = np.asarray(
                [float(r[args.reference]) for r in scan], dtype=np.float64
            ) * HARTREE_KCAL
            energies = energies - energies.min()

            # Perception uses the LOWEST-energy frame. A torsiondrive passes
            # through eclipsed geometries where a distance criterion can invent
            # a bond that is not there.
            bonds, orders, smiles = perceive(
                symbols, xyz[int(np.argmin(energies))], scan[0].get("charge", 0)
            )
            if not len(bonds):
                raise ValueError("no bonds perceived")

            key = str(len(meta))
            arrays[f"xyz::{key}"] = xyz
            arrays[f"energy::{key}"] = energies
            arrays[f"bonds::{key}"] = bonds
            arrays[f"bond_orders::{key}"] = orders
            meta.append({
                "key": key,
                "molecule_id": name,
                "symbols": symbols,
                "n_atoms": len(symbols),
                "grid": [float(r["torsion_angle"]) for r in scan],
                "driven_dihedral": [int(a) for a in atoms],
                "driven_bond": sorted(int(a) for a in atoms[1:3]),
                "smiles": scan[0].get("smiles", smiles),
                "perceived_smiles": smiles,
                "inchi_key": name,
                "charge": int(scan[0].get("charge", 0)),
                "barrier_kcal_mol": float(energies.max()),
            })
        except Exception as error:                       # noqa: BLE001
            quarantine.append((name, f"{type(error).__name__}: {error}"))

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, meta=json.dumps(meta), dataset=args.dataset,
                        specification=args.reference, **arrays)
    q = pathlib.Path(args.quarantine)
    q.parent.mkdir(parents=True, exist_ok=True)
    with q.open("w") as fh:
        fh.write("molecule_id,reason\n")
        for name, reason in quarantine:
            fh.write(f'{name},"{reason}"\n')

    if not meta:
        print(f"every one of {len(paths)} scans was quarantined")
        return 1
    heavy = np.asarray([sum(1 for s in m["symbols"] if s != "H") for m in meta])
    charged = sum(1 for m in meta if m["charge"] != 0)
    elements = sorted({s for m in meta for s in m["symbols"]})
    points = np.asarray([len(m["grid"]) for m in meta])
    print(f"{len(meta):,} scans of {len(paths):,} converted, "
          f"{len(quarantine):,} quarantined")
    print(f"  reference    {args.reference}")
    print(f"  charged      {charged:,} ({100*charged/len(meta):.0f} percent)")
    print(f"  elements     {' '.join(elements)}")
    print(f"  heavy atoms  median {np.median(heavy):.0f}, "
          f"{heavy.min()} to {heavy.max()}")
    print(f"  scan points  median {np.median(points):.0f}, "
          f"{points.min()} to {points.max()}")
    print(f"  barrier      median {np.median([m['barrier_kcal_mol'] for m in meta]):.2f} kcal/mol")
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
