"""Torsional strain of docked poses under each model, against pose accuracy.

QUESTION. A bound ligand is not at its torsional minimum, so absolute strain is
not a truth signal. The relative one is: a near-native pose should be assigned
less torsional strain than a wrong pose. Does it, and by which model?

ON WHAT. OpenFE Industry Benchmark 2024 as staged for rbfe_ceiling: 146 systems,
2,146 reference poses, 10 docking modes each. Accuracy is symmetry-corrected
heavy-atom RMSD of each mode to that ligand's reference pose. One independent
unit is a ligand.

HOW. Every model is reduced to the same object, a list of (periodicity, phase,
force constant) per proper torsion, so one code path scores them all. Sage 2.3.0
from its OpenMM PeriodicTorsionForce; espaloma 0.3.2 from its n4 coefficients,
which are in HARTREE and listed twice per torsion; and this project's own
readouts, which emit (a_n, b_n) per torsion and convert to an amplitude
sqrt(a^2+b^2) with phase atan2(b, a), the unphased arm being the same with b=0.

THOSE LAST TWO ARE EXTRAPOLATING. They were trained on 870 OpenFF Gen3
molecules; these are 2,146 industry benchmark ligands, larger and of different
chemistry. The row carries the model name so that is never lost. Strain of a bond is its
profile at the pose angle minus the minimum of that profile over a dense
rotation of the bond; strain of a pose is the sum over rotatable bonds.

CONTROLS. The reference pose itself is scored, and heavy-atom count travels with
every row, because strain grows with rotor count and a correlation with RMSD
could be a correlation with size.

UNITS. kcal/mol for strain, Angstrom for RMSD, degrees for torsion angles.

COST. One force field system and one espaloma forward per ligand. CPU.
"""
from __future__ import annotations

import argparse
import csv
import glob
import pathlib

import numpy as np

HARTREE_KCAL = 627.5094740631
GRID = np.linspace(-np.pi, np.pi, 361, endpoint=False)


def rotatable_bonds(mol):
    out = []
    for bond in mol.GetBonds():
        if bond.IsInRing() or bond.GetBondTypeAsDouble() != 1.0:
            continue
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        if a.GetAtomicNum() == 1 or b.GetAtomicNum() == 1:
            continue
        heavy = lambda x, y: sum(1 for n in x.GetNeighbors()
                                 if n.GetIdx() != y.GetIdx() and n.GetAtomicNum() > 1)
        if heavy(a, b) and heavy(b, a):
            out.append((a.GetIdx(), b.GetIdx()))
    return out


def sage_terms(molecule, field):
    """[(i,j,k,l), n, phase_rad, k_kcal] from Sage's own OpenMM system."""
    from openmm import openmm as mm_module  # noqa: F401
    import openmm
    from openmm import unit as u

    system = field.create_openmm_system(molecule.to_topology())
    terms = []
    for force in system.getForces():
        if not isinstance(force, openmm.PeriodicTorsionForce):
            continue
        for index in range(force.getNumTorsions()):
            i, j, k, l, n, phase, amplitude = force.getTorsionParameters(index)
            terms.append(((i, j, k, l), int(n),
                          phase.value_in_unit(u.radian),
                          amplitude.value_in_unit(u.kilocalorie_per_mole)))
    return terms


def espaloma_terms(molecule, model, torch):
    """The same object from espaloma. Hartree, and every torsion listed twice."""
    import espaloma as esp

    graph = esp.Graph(molecule).heterograph
    with torch.no_grad():
        out = model(graph)
    quads = out.nodes["n4"].data["idxs"].cpu().numpy().astype(int)
    k = out.nodes["n4"].data["k"].cpu().numpy() * HARTREE_KCAL
    keep = quads[:, 0] < quads[:, 3]
    if keep.any():
        quads, k = quads[keep], k[keep]
    terms = []
    for quad, row in zip(quads, k):
        for n, amplitude in enumerate(row, start=1):
            if abs(amplitude) > 1e-12:
                terms.append((tuple(int(x) for x in quad), n, 0.0,
                              float(amplitude)))
    return terms


def head_terms(molecule, checkpoint, torch, encoder_parts, phased):
    """(quad, n, phase, k) from a trained per-torsion readout in this project."""
    import numpy as np
    from espaloma_encoder import build_encoder, embed_graph
    import espaloma as esp
    from fit_torsion_head import enumerate_torsions

    state, args_used = checkpoint["state"], checkpoint["args"]
    encoder, width = build_encoder(args_used["espaloma_version"])
    encoder = encoder.eval()
    graph = esp.Graph(molecule).heterograph
    rdmol = molecule.to_rdkit()
    groups = enumerate_torsions(
        np.asarray([[b.GetBeginAtomIdx(), b.GetEndAtomIdx()]
                    for b in rdmol.GetBonds()], dtype=int),
        rdmol.GetNumAtoms())
    quads = [q for bond in sorted(groups) for q in groups[bond]]
    if not quads:
        return []
    from phased_torsion import build_readout

    head = build_readout(width, args_used["periodicities"], phased,
                         args_used["hidden"], torch).double()
    head.load_state_dict(state)
    head.eval()
    with torch.no_grad():
        h = embed_graph(encoder, graph, 1, False)[0].double()
        indices = torch.tensor(quads, dtype=torch.long)
        forward = h[indices].reshape(len(indices), -1)
        reverse = h[indices.flip(-1)].reshape(len(indices), -1)
        k = head.out(head.pool(forward) + head.pool(reverse)).cpu().numpy()
    periodicities = args_used["periodicities"]
    terms = []
    for quad, row in zip(quads, k):
        for n in range(1, periodicities + 1):
            a = float(row[n - 1])
            b = float(row[periodicities + n - 1]) if phased else 0.0
            amplitude = float(np.hypot(a, b))
            if amplitude > 1e-12:
                terms.append((tuple(int(x) for x in quad), n,
                              float(np.arctan2(b, a)), amplitude))
    return terms


def strain(mol, conformer, terms, bonds):
    """Per-bond strain: the profile at the pose angle minus its own minimum."""
    from rdkit.Chem import rdMolTransforms

    by_bond = {}
    for quad, n, phase, amplitude in terms:
        key = tuple(sorted(quad[1:3]))
        by_bond.setdefault(key, []).append((quad, n, phase, amplitude))

    rows = []
    for j, k in bonds:
        group = by_bond.get(tuple(sorted((j, k))), [])
        if not group:
            continue
        total = np.zeros_like(GRID)
        at_pose = 0.0
        n_atoms = mol.GetNumAtoms()
        for quad, n, phase, amplitude in group:
            if max(quad) >= n_atoms:
                continue
            try:
                delta = np.radians(rdMolTransforms.GetDihedralDeg(
                    conformer, *[int(x) for x in quad]))
            except Exception:                            # noqa: BLE001
                continue
            total += amplitude * np.cos(n * (delta + GRID) - phase)
            at_pose += amplitude * np.cos(n * delta - phase)
        rows.append({"bond": f"{j}-{k}", "strain": float(at_pose - total.min())})
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--poses", required=True)
    p.add_argument("--forcefield", default="openff-2.3.0.offxml")
    p.add_argument("--espaloma-version", default="0.3.2")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--quarantine", required=True)
    args = p.parse_args()

    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
    from openff.toolkit import ForceField, Molecule
    import espaloma as esp
    import torch

    field = ForceField(args.forcefield)
    model = esp.get_model(args.espaloma_version).eval()

    references = sorted(q for q in glob.glob(str(pathlib.Path(args.poses) / "*/*.sdf"))
                        if not q.endswith("_modes.sdf"))
    if args.limit:
        references = references[:args.limit]

    rows, quarantine = [], []
    for path in references:
        system = pathlib.PurePath(path).parent.name
        ligand = pathlib.PurePath(path).stem
        try:
            # The SDFs carry implicit hydrogens: 28 atoms for 25 heavy. OpenFF
            # adds them, giving 46, and every torsion index then points past the
            # conformer. Added here with coordinates so both sides agree.
            def read(p):
                return [Chem.AddHs(x, addCoords=True)
                        for x in Chem.SDMolSupplier(p, removeHs=False)
                        if x is not None]

            reference = read(path)[0]
            modes_path = path.replace(".sdf", "_modes.sdf")
            modes = read(modes_path) if pathlib.Path(modes_path).exists() else []
            molecule = Molecule.from_rdkit(reference, allow_undefined_stereo=True)
            # The force fields index into the OpenFF molecule; the conformers
            # come from RDKit. If those two disagree on how many atoms there
            # are, every torsion index is suspect and the whole ligand goes,
            # rather than a dihedral being read off the wrong atoms silently.
            if molecule.n_atoms != reference.GetNumAtoms():
                raise ValueError(f"OpenFF has {molecule.n_atoms} atoms, RDKit "
                                 f"{reference.GetNumAtoms()}")
            bonds = rotatable_bonds(reference)
            if not bonds:
                continue
            terms = {"sage": sage_terms(molecule, field),
                     "espaloma": espaloma_terms(molecule, model, torch)}
            heavy = reference.GetNumHeavyAtoms()
            reference_noH = Chem.RemoveHs(Chem.Mol(reference))

            for label, mol in [("reference", reference)] + [
                    (f"mode{i}", m) for i, m in enumerate(modes)]:
                if not mol.GetNumConformers():
                    continue
                if mol.GetNumAtoms() != reference.GetNumAtoms():
                    quarantine.append((f"{system}/{ligand}/{label}",
                                       f"{mol.GetNumAtoms()} atoms against the "
                                       f"reference's {reference.GetNumAtoms()}"))
                    continue
                rmsd = 0.0
                if label != "reference":
                    try:
                        rmsd = float(AllChem.GetBestRMS(
                            Chem.RemoveHs(Chem.Mol(mol)), reference_noH))
                    except Exception:                    # noqa: BLE001
                        rmsd = float("nan")
                conformer = mol.GetConformer()
                for name, term_list in terms.items():
                    per_bond = strain(mol, conformer, term_list, bonds)
                    rows.append({
                        "system": system, "ligand": ligand, "pose": label,
                        "model": name, "n_heavy": heavy, "n_bonds": len(per_bond),
                        "rmsd_to_reference": rmsd,
                        "strain_kcal_mol": float(sum(r["strain"] for r in per_bond)),
                        "max_bond_strain": float(max((r["strain"] for r in per_bond),
                                                     default=0.0)),
                    })
        except Exception as error:                       # noqa: BLE001
            quarantine.append((f"{system}/{ligand}",
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
        fh.write("ligand,reason\n")
        for name, reason in quarantine:
            fh.write(f'{name},"{reason}"\n')

    ligands = len({(r["system"], r["ligand"]) for r in rows})
    print(f"{ligands:,} ligands, {len(rows):,} pose-model rows, "
          f"{len(quarantine)} failed")

    # THE BENCHMARK, one number per model: within a ligand, does a mode's
    # torsional strain track how wrong that mode is? Spearman over that
    # ligand's 10 gnina modes, median over ligands, with a sign test across
    # ligands because 10 modes per ligand resolves nothing on its own.
    from scipy.stats import spearmanr, binomtest

    print(f"\n{'model':<12}{'rho(strain, RMSD)':>20}{'positive':>12}{'p':>12}"
          f"{'ligands':>9}")
    for name in sorted({r["model"] for r in rows}):
        per_ligand = {}
        for r in rows:
            if r["model"] != name or r["pose"] == "reference":
                continue
            per_ligand.setdefault((r["system"], r["ligand"]), []).append(r)
        values = []
        for group in per_ligand.values():
            if len(group) < 4:
                continue
            strain_values = [g["strain_kcal_mol"] for g in group]
            rmsd_values = [g["rmsd_to_reference"] for g in group]
            if len(set(strain_values)) < 2 or len(set(rmsd_values)) < 2:
                continue
            rho = spearmanr(strain_values, rmsd_values).statistic
            if np.isfinite(rho):
                values.append(float(rho))
        if not values:
            continue
        positive = sum(1 for v in values if v > 0)
        p = binomtest(positive, len(values), 0.5).pvalue
        print(f"{name:<12}{np.median(values):20.3f}{positive:>8}/{len(values):<4}"
              f"{p:12.3g}{len(values):9d}")
    print(f"\n{'model':<10}{'pose':<12}{'strain median':>15}{'90th':>9}{'n':>8}")
    for name in ("sage", "espaloma"):
        for pose in ("reference", "modes"):
            subset = [r for r in rows if r["model"] == name
                      and ((r["pose"] == "reference") == (pose == "reference"))]
            if not subset:
                continue
            s = np.array([r["strain_kcal_mol"] for r in subset])
            print(f"{name:<10}{pose:<12}{np.median(s):15.3f}"
                  f"{np.percentile(s, 90):9.3f}{len(s):8d}")
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
