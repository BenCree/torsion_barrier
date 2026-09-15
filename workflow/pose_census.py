"""How often does the fixed-phase blind spot occur in ligands people run FEP on?

QUESTION. The phase census measured, on 15,289 QM scans, that removing the
asymmetric part of a torsion profile moves the minimum by 24 degrees and the
Boltzmann populations by a Jensen-Shannon distance of 0.292 on rotors beside a
stereocentre, against Sage's published 0.30 and presto's 0.12. That is a
statement about a QM dataset. This asks the practical version: what fraction of
the rotatable bonds in real docked ligands are of that kind?

WHY IT NEEDS NO MODEL. The classification is a property of the molecule, not of
anything trained: whether the bond's substituents are symmetry-equivalent
decides whether a fixed-phase per-torsion form can reach a bond phase at all,
and whether a stereocentre sits beside the rotor decides whether the profile can
be asymmetric in the first place. So this number stands whether or not any head
here ever works, and it is the one that says whether the blind spot is a
curiosity or something a docking or RBFE campaign meets every day.

ON WHAT. The docked poses of the OpenFE Industry Benchmark 2024 as staged for
`rbfe_ceiling`: 146 systems, each with a top pose and its alternative docking
modes. One independent unit is a ligand; one row is one rotatable bond in one
pose.

HOW. RDKit perceives the molecule from the SDF, which already carries bond
orders and charges, so none of the valence repair the QM caches needed applies
here. Rotatable bonds are the acyclic single bonds between two heavy atoms that
each carry another heavy atom, which is the set a torsion term parameterises.
The same `classify` and `substituents_equivalent` the QM census used are
imported rather than reimplemented, so the two cannot drift.

CONTROLS. The QM census on the same classes is the comparison: if the class
proportions in docked ligands look nothing like the proportions in the QM pool,
then the QM pool was not representative of the application and that is itself
the finding.

UNITS. Counts and fractions of rotatable bonds. Degrees for the torsion angle
the pose sits at.

COST. Seconds, CPU. No GPU, no model, no network.
"""
from __future__ import annotations

import argparse
import csv
import glob
import pathlib

import numpy as np


def rotatable_bonds(mol):
    """Acyclic single bonds between two substituted heavy atoms.

    Not RDKit's RotatableBondSmarts, which excludes amides and a few other
    cases by convention: an amide is exactly the kind of bond a torsion term
    parameterises and the QM census measured, so excluding it here would make
    the two sets disagree for a reason that has nothing to do with the question.
    """
    out = []
    for bond in mol.GetBonds():
        if bond.IsInRing() or bond.GetBondTypeAsDouble() != 1.0:
            continue
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        if a.GetAtomicNum() == 1 or b.GetAtomicNum() == 1:
            continue
        heavy = lambda atom, other: sum(
            1 for n in atom.GetNeighbors()
            if n.GetIdx() != other.GetIdx() and n.GetAtomicNum() > 1)
        if heavy(a, b) and heavy(b, a):
            out.append((a.GetIdx(), b.GetIdx()))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--poses", required=True,
                   help="a directory of per-system directories of SDF files")
    p.add_argument("--pattern", default="*/*.sdf")
    p.add_argument("--modes", action="store_true",
                   help="include the *_modes.sdf alternative docking modes")
    p.add_argument("--out", required=True)
    p.add_argument("--quarantine", required=True)
    args = p.parse_args()

    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdMolTransforms
    RDLogger.DisableLog("rdApp.*")
    from phase_census import classify, substituents_equivalent

    paths = sorted(glob.glob(str(pathlib.Path(args.poses) / args.pattern)))
    if not args.modes:
        paths = [p for p in paths if not p.endswith("_modes.sdf")]
    if not paths:
        print(f"no SDF matched {args.pattern!r} under {args.poses}")
        return 1

    rows, quarantine = [], []
    for path in paths:
        system = pathlib.PurePath(path).parent.name
        ligand = pathlib.PurePath(path).stem
        try:
            for pose_index, mol in enumerate(
                    Chem.SDMolSupplier(path, removeHs=False, sanitize=True)):
                if mol is None:
                    raise ValueError("RDKit could not read a pose")
                conformer = mol.GetConformer() if mol.GetNumConformers() else None
                for j, k in rotatable_bonds(mol):
                    classes = classify(mol, j, k)
                    angle = ""
                    if conformer is not None:
                        # The torsion the pose actually sits at, through the
                        # heaviest substituent on each side so the reported
                        # angle is reproducible rather than whichever neighbour
                        # RDKit happens to list first.
                        def pick(centre, other):
                            neighbours = [n.GetIdx() for n in
                                          mol.GetAtomWithIdx(centre).GetNeighbors()
                                          if n.GetIdx() != other
                                          and n.GetAtomicNum() > 1]
                            return max(neighbours,
                                       key=lambda i: mol.GetAtomWithIdx(i).GetAtomicNum(),
                                       default=None)
                        i, l = pick(j, k), pick(k, j)
                        if i is not None and l is not None:
                            angle = float(rdMolTransforms.GetDihedralDeg(
                                conformer, i, j, k, l))
                    rows.append({
                        "system": system, "ligand": ligand,
                        "pose": pose_index,
                        "is_mode": int(path.endswith("_modes.sdf")),
                        "bond": f"{j}-{k}",
                        "n_heavy": mol.GetNumHeavyAtoms(),
                        "classes": "|".join(classes),
                        "substituents_equivalent": int(
                            substituents_equivalent(mol, j, k)),
                        "torsion_degrees": angle,
                    })
        except Exception as error:                       # noqa: BLE001
            quarantine.append((f"{system}/{ligand}",
                               f"{type(error).__name__}: {error}"))

    if not rows:
        print("no rotatable bond found")
        return 1
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pathlib.Path(args.quarantine).open("w") as fh:
        fh.write("ligand,reason\n")
        for name, reason in quarantine:
            fh.write(f'{name},"{reason}"\n')

    ligands = len({(r["system"], r["ligand"]) for r in rows})
    systems = len({r["system"] for r in rows})
    per_ligand = {}
    for r in rows:
        per_ligand.setdefault((r["system"], r["ligand"]), []).append(r)
    counts = np.array([len(v) for v in per_ligand.values()])

    print(f"{systems} systems, {ligands:,} ligands, {len(rows):,} rotatable bonds, "
          f"{len(quarantine)} unreadable")
    print(f"  rotatable bonds per ligand: median {np.median(counts):.0f}, "
          f"{counts.min()} to {counts.max()}")
    print(f"\n{'class':<26}{'bonds':>8}{'share':>8}   ligands with at least one")
    total = len(rows)
    for label, test in (
        ("stereocentre adjacent",
         lambda r: "stereocentre_adjacent" in r["classes"].split("|")),
        ("plain", lambda r: "plain" in r["classes"].split("|")),
        ("anomeric", lambda r: "anomeric" in r["classes"].split("|")),
        ("amide", lambda r: "amide" in r["classes"].split("|")),
        ("biaryl", lambda r: "biaryl" in r["classes"].split("|")),
        ("conjugated", lambda r: "conjugated" in r["classes"].split("|")),
        ("substituents equivalent", lambda r: r["substituents_equivalent"] == 1),
    ):
        hit = [r for r in rows if test(r)]
        with_any = len({(r["system"], r["ligand"]) for r in hit})
        print(f"  {label:<24}{len(hit):>8}{len(hit)/total:>8.1%}"
              f"   {with_any:,} of {ligands:,} ({with_any/ligands:.0%})")
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
