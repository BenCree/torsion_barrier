"""Convert THEMol TorsionScan shards into this project's cache format.

QUESTION. None on its own. A format conversion, kept apart from everything that
computes, so that a change to the model does not cost another pass over 345 GB.

WHAT IT TAKES. Which shards, how many scans, and where to write. The three
cohort sizes G4 asks about are three command lines, not three scripts.

THE SET. ByteDance-Seed/THEMol, CC-BY-NC-4.0, TorsionScan subset: 50 shards of
about 83,951 scans each, so roughly 4.2 million, at B3LYP-D3BJ/DZVP. That is the
SAME level of theory as the OpenFF torsiondrives this model trained on, which is
why this set and not another: a change of cohort size is then the only thing
changing, where TorsionTest2000 would change the reference as well.

WHAT IS IN A SHARD. One HDF5 group per scan, named by a hash, holding
`atomic_numbers`, `mapped_isomeric_smiles`, `torsion_atom_indices` and one
`constraint k` subgroup per scan point with `coords`, `energy` and `forces`.
Energies are kcal/mol already, not Hartree; barriers come out at a few kcal/mol
and a Hartree reading would make them thousands. `torsion_atom_indices` is
0-based into the coordinate array, NOT the 1-based atom map numbers of the
SMILES beside it, which name the same atoms one apart.

THE BOND ORDERS COME FROM THE MAPPED SMILES, whose atom map number i is array
index i-1, so no perception and no matching. Building every bond single, which
this project has done once before, parametrises a different molecule.

Scan lengths VARY, 6 to 25 points, with 24 the mode. A scan shorter than
--min-points is quarantined rather than padded, because a torsion term fitted to
six points of a profile is fitted to a different quantity.

UNITS. kcal/mol relative to each scan's own minimum; Angstrom; degrees.

COST. About 40 seconds and 7 GB of reading per shard. No GPU.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

BOND_TYPES = {1.0: "SINGLE", 1.5: "AROMATIC", 2.0: "DOUBLE", 3.0: "TRIPLE"}


def connectivity(mapped_smiles):
    """Bonds, orders and symbols in the ORDER THE COORDINATES USE.

    The map number on each atom is its position in the coordinate array, one
    based. Reading the SMILES without applying that map gives RDKit's own
    canonical order and silently scrambles every geometry.
    """
    from rdkit import Chem

    # removeHs MUST be off. THEMol writes every hydrogen explicitly, as [H:21],
    # and RDKit's default merges them into implicit counts, so a 33-atom molecule
    # comes back with 17 atoms and the coordinates line up with nothing. The
    # atom-count check below is what caught it: 83,942 of 83,951 scans in the
    # first shard failed, and a reader that had trusted the SMILES would have
    # scrambled every geometry instead.
    parameters = Chem.SmilesParserParams()
    parameters.removeHs = False
    parameters.sanitize = True
    molecule = Chem.MolFromSmiles(mapped_smiles, parameters)
    if molecule is None:
        raise ValueError("RDKit could not parse the mapped SMILES")
    order = {}
    for atom in molecule.GetAtoms():
        number = atom.GetAtomMapNum()
        if number == 0:
            raise ValueError("an atom carries no map number")
        order[atom.GetIdx()] = number - 1
    if sorted(order.values()) != list(range(molecule.GetNumAtoms())):
        raise ValueError("the atom map numbers are not a permutation")

    bonds, orders = [], []
    for bond in molecule.GetBonds():
        i, j = order[bond.GetBeginAtomIdx()], order[bond.GetEndAtomIdx()]
        value = bond.GetBondTypeAsDouble()
        if round(value * 2) / 2 not in BOND_TYPES:
            raise ValueError(f"bond order {value} is not one of {sorted(BOND_TYPES)}")
        bonds.append([i, j])
        orders.append(value)
    symbols = [None] * molecule.GetNumAtoms()
    for atom in molecule.GetAtoms():
        symbols[order[atom.GetIdx()]] = atom.GetSymbol()
    return (np.asarray(bonds, dtype=np.int32),
            np.asarray(orders, dtype=np.float64), symbols)


def dihedral(frame, atoms):
    """One dihedral in degrees, the RDKit and torsiondrive sign convention."""
    b0 = frame[atoms[0]] - frame[atoms[1]]
    b1 = frame[atoms[2]] - frame[atoms[1]]
    b2 = frame[atoms[3]] - frame[atoms[2]]
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w),
                                       np.dot(v, w))))


def scans(path, min_points, quarantine):
    """Yield one converted scan at a time, so a shard is never fully resident.

    How many to take is the caller's business: this stops when the file does.
    """
    import h5py

    with h5py.File(path, "r") as handle:
        for name in handle:
            group = handle[name]
            try:
                points = sorted(
                    (k for k in group if k.startswith("constraint ")),
                    key=lambda k: int(k.split()[1]),
                )
                if len(points) < min_points:
                    raise ValueError(f"{len(points)} points is fewer than "
                                     f"{min_points}")
                smiles = group["mapped_isomeric_smiles"][()]
                smiles = smiles.decode() if isinstance(smiles, bytes) else str(smiles)
                bonds, orders, symbols = connectivity(smiles)
                atoms = [int(a) for a in np.asarray(group["torsion_atom_indices"][()]).ravel()]
                if len(atoms) != 4:
                    raise ValueError(f"torsion_atom_indices has {len(atoms)} entries")

                numbers = np.asarray(group["atomic_numbers"][()]).ravel()
                if len(numbers) != len(symbols):
                    raise ValueError(f"{len(numbers)} atoms against "
                                     f"{len(symbols)} in the SMILES")
                xyz = np.stack([np.asarray(group[k]["coords"][()], dtype=np.float64)
                                for k in points])
                energies = np.asarray([float(group[k]["energy"][()]) for k in points])
                grid = [dihedral(frame, atoms) for frame in xyz]
                order = np.argsort(grid)
                yield {
                    "molecule_id": name,
                    "symbols": symbols,
                    "n_atoms": len(symbols),
                    "grid": [float(grid[int(i)]) for i in order],
                    "driven_dihedral": atoms,
                    "driven_bond": sorted(atoms[1:3]),
                    "smiles": smiles,
                    "inchi_key": name,
                    "barrier_kcal_mol": float(energies.max() - energies.min()),
                }, xyz[order], (energies - energies.min())[order], bonds, orders
            except Exception as error:                   # noqa: BLE001
                quarantine.append((name, f"{type(error).__name__}: {error}"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", nargs="+", required=True,
                        help="paths to torsion_*.h5")
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after this many scans; 0 for every one")
    parser.add_argument("--min-points", type=int, default=20)
    parser.add_argument("--out", required=True)
    parser.add_argument("--quarantine", required=True)
    args = parser.parse_args()

    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    arrays, meta, quarantine = {}, [], []
    budget = args.limit or None
    for shard in args.shards:
        if budget is not None and budget <= 0:
            break
        taken = 0
        for entry, xyz, energies, bonds, orders in scans(
                shard, args.min_points, quarantine):
            key = str(len(meta))
            arrays[f"xyz::{key}"] = xyz.astype(np.float32)
            arrays[f"energy::{key}"] = energies
            arrays[f"bonds::{key}"] = bonds
            arrays[f"bond_orders::{key}"] = orders
            meta.append(dict(entry, key=key, source=pathlib.PurePath(shard).name))
            taken += 1
            if budget is not None:
                budget -= 1
                if budget <= 0:
                    break
        print(f"  {pathlib.PurePath(shard).name}: {taken:,} scans "
              f"({len(quarantine):,} quarantined so far)", flush=True)

    if not meta:
        print("no scan converted")
        return 1
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, meta=json.dumps(meta), dataset="THEMol TorsionScan",
             specification="b3lyp-d3bj/dzvp", **arrays)
    q = pathlib.Path(args.quarantine)
    q.parent.mkdir(parents=True, exist_ok=True)
    with q.open("w") as fh:
        fh.write("molecule_id,reason\n")
        for name, reason in quarantine:
            fh.write(f'{name},"{reason}"\n')

    heavy = np.asarray([sum(1 for s in m["symbols"] if s != "H") for m in meta])
    points = np.asarray([len(m["grid"]) for m in meta])
    barriers = np.asarray([m["barrier_kcal_mol"] for m in meta])
    elements = sorted({s for m in meta for s in m["symbols"]})
    print(f"{len(meta):,} scans converted, {len(quarantine):,} quarantined")
    print(f"  elements     {' '.join(elements)}")
    print(f"  heavy atoms  median {np.median(heavy):.0f}, "
          f"{heavy.min()} to {heavy.max()}")
    print(f"  scan points  median {np.median(points):.0f}, "
          f"{points.min()} to {points.max()}")
    print(f"  barrier      median {np.median(barriers):.2f} kcal/mol, "
          f"90th percentile {np.percentile(barriers, 90):.2f}")
    print(f"  -> {out} ({out.stat().st_size/1e9:.2f} GB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
