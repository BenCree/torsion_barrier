"""Subtract everything a force field already computes, leaving the torsion residual.

QUESTION. None on its own. This changes the TARGET the head is fitted to, and
that is the single identifiable defect in the model as built.

WHY. A torsion profile is mostly 1-4 electrostatics and van der Waals, both of
which vary strongly with the dihedral and both of which are analytically
computable given charges. Fitting the TOTAL profile makes the head learn
Coulomb's law from 604 molecules instead of being handed it. BespokeFit, presto
and espaloma all fit the residual instead, and `model.py`'s own docstring says a
transferable parameter has to be fitted against E_QM - E_MM,no-torsion, calling
it "a different target rather than a later step". It was blocked on charges and
is not any more.

THE CHARGE MODEL IS NOT A CHOICE. Sage 2.3.0's NAGLCharges handler names
openff-gnn-am1bcc-1.0.0.pt with hash 7981e7f5..., and every other parameter in
Sage 2.3.0 was refit WITH those charges. Computing the baseline with any other
charge model would make the residual absorb the difference between two charge
models rather than the physics, so the handler is left exactly as the force
field ships it.

WHAT IS REMOVED. Proper torsions only. Bonds, angles, impropers, vdW and
electrostatics stay, because those are what the residual is being cleared of.
The 1-4 scaling Sage declares is kept as declared.

OUTPUT. An npz in this project's own cache format with `energy::key` replaced by
the residual, so every rule downstream reads it unchanged and the arms differ in
the target and nothing else.

UNITS. kcal/mol, each profile relative to its own minimum on both sides before
subtracting, so a constant offset between QM and MM cannot enter.

COST. An Interchange per molecule, seconds each, CPU. This is the expensive part
and it is done once.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--torsiondrives", required=True)
    p.add_argument("--forcefield", default="openff-2.3.0.offxml")
    p.add_argument("--out", required=True)
    p.add_argument("--quarantine", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--keep-torsions", action="store_true",
                   help="diagnostic: leave proper torsions in, so the residual "
                        "is QM minus the FULL force field and Sage's own error "
                        "is what comes out")
    args = p.parse_args()

    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    from espaloma_encoder import rdkit_from_cache
    from openff.toolkit import ForceField, Molecule
    import openmm
    from openmm import unit as openmm_unit

    field = ForceField(args.forcefield)
    if not args.keep_torsions:
        field.deregister_parameter_handler("ProperTorsions")
    charge_handler = [h for h in field.registered_parameter_handlers
                      if "Charge" in h]
    print(f"{args.forcefield}: handlers {field.registered_parameter_handlers}")
    print(f"  charges from {charge_handler}")

    cache = np.load(args.torsiondrives, allow_pickle=False)
    meta = json.loads(str(cache["meta"]))
    if args.limit:
        meta = meta[:args.limit]

    arrays, kept, quarantine = {}, [], []
    for entry in meta:
        key = entry["key"]
        try:
            molecule = Molecule.from_rdkit(
                rdkit_from_cache(entry["symbols"], cache[f"bonds::{key}"],
                                 cache[f"bond_orders::{key}"]),
                allow_undefined_stereo=True,
            )
            system = field.create_openmm_system(molecule.to_topology())
            integrator = openmm.VerletIntegrator(1.0 * openmm_unit.femtosecond)
            context = openmm.Context(
                system, integrator,
                openmm.Platform.getPlatformByName("Reference"))

            xyz = np.asarray(cache[f"xyz::{key}"], dtype=float)
            mm = []
            for frame in xyz:
                context.setPositions(frame * openmm_unit.angstrom)
                mm.append(context.getState(getEnergy=True).getPotentialEnergy()
                          .value_in_unit(openmm_unit.kilocalorie_per_mole))
            del context, integrator
            mm = np.asarray(mm, dtype=float)
            if not np.isfinite(mm).all():
                raise ValueError("the force field returned a non-finite energy")

            qm = np.asarray(cache[f"energy::{key}"], dtype=float)
            # Both relative to their own minimum before subtracting, so the
            # constant offset between a QM total energy and an MM one, which is
            # arbitrary and enormous, cannot enter the residual.
            residual = (qm - qm.min()) - (mm - mm.min())
            residual = residual - residual.min()

            new = str(len(kept))
            arrays[f"xyz::{new}"] = cache[f"xyz::{key}"]
            arrays[f"energy::{new}"] = residual
            arrays[f"bonds::{new}"] = cache[f"bonds::{key}"]
            arrays[f"bond_orders::{new}"] = cache[f"bond_orders::{key}"]
            kept.append(dict(entry, key=new,
                             mm_range_kcal_mol=float(np.ptp(mm)),
                             qm_range_kcal_mol=float(np.ptp(qm)),
                             residual_range_kcal_mol=float(np.ptp(residual))))
        except Exception as error:                       # noqa: BLE001
            quarantine.append((entry.get("molecule_id", "?"),
                               f"{type(error).__name__}: {error}"))

    if not kept:
        print(f"every one of {len(meta)} molecules failed; first: {quarantine[:1]}")
        return 1
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, meta=json.dumps(kept),
        dataset=f"{str(cache['dataset'])} minus {args.forcefield}"
                f"{'' if args.keep_torsions else ' without proper torsions'}",
        specification=str(cache["specification"]), **arrays)
    with pathlib.Path(args.quarantine).open("w") as fh:
        fh.write("molecule_id,reason\n")
        for name, reason in quarantine:
            fh.write(f'{name},"{reason}"\n')

    qm = np.array([m["qm_range_kcal_mol"] for m in kept])
    mmr = np.array([m["mm_range_kcal_mol"] for m in kept])
    res = np.array([m["residual_range_kcal_mol"] for m in kept])
    print(f"\n{len(kept):,} of {len(meta):,} molecules, {len(quarantine):,} quarantined")
    print(f"  QM profile range        median {np.median(qm):6.3f} kcal/mol")
    print(f"  MM baseline range       median {np.median(mmr):6.3f}")
    print(f"  RESIDUAL range          median {np.median(res):6.3f}"
          f"   ({np.median(res)/np.median(qm):.0%} of the QM range)")
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
