"""Pretrained potentials on a torsiondrive set, with no training whatever.

WHY THIS IS THE CALIBRATION EVERY OTHER NUMBER NEEDS. Our "frozen encoder" arm
freezes the representation but still FITS a head to QM torsion data. A pretrained
potential used as published fits nothing: it computes the total energy of each
scan conformer and differences it from that scan's minimum. That is the protocol
behind MACE-OFF23's published 0.25 kcal/mol mean barrier error on TorsionNet500,
so it is the only thing directly comparable to it, and it is the floor the fitted
head has to clear to be worth having at all.

ON WHAT. Any torsiondrive cache. Run on TorsionNet500, which is held out from
training and can therefore be scored by an untrained method without leaking:
there is nothing to leak into.

THE METRIC IS REPORTED BOTH WAYS. `barrier_error` is |predicted range - QM range|
over a scan, which is what the published comparisons use. `rmse` is the whole
profile, which is stricter and is what this project is trying to minimise. A
method can get the barrier right with the wrong shape.

UNITS. kcal/mol, every profile relative to its own minimum. One row per molecule
per potential. Cost is measured per profile on a named device, not estimated.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

EV_PER_KCAL_MOL = 23.060547830618307


def dpa3_calculator(branch, device, checkpoint, workdir):
    """DPA-3.1-3M as a POTENTIAL, not a descriptor.

    The multi-task checkpoint holds one shared descriptor and 31 fitting nets,
    one per training domain, so it cannot return an energy until a branch is
    chosen and frozen out. `SPICE2` is the organic-molecule branch.

    THE DISTINCTION THAT MATTERS FOR READING THE RESULT. The DPA-3 paper's
    TorsionNet-500 numbers are DPA3-L12 and L24 trained ON SPICE-MACE-OFF. This
    is DPA-3.1-3M, the multi-task OpenLAM checkpoint, whose pretraining is 26 of
    31 branches materials. They share an architecture and nothing else, so this
    measures the checkpoint we actually have, which is the thing worth knowing.
    """
    import subprocess

    frozen = pathlib.Path(workdir) / f"dpa3_{branch}.pth"
    if not frozen.exists():
        frozen.parent.mkdir(parents=True, exist_ok=True)
        command = [
            pathlib.Path(sys.executable).with_name("dp").as_posix(),
            "--pt", "freeze", "-c", str(checkpoint),
            "-o", str(frozen), "--head", branch,
        ]
        print(f"  freezing branch {branch!r} -> {frozen}", flush=True)
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0 or not frozen.exists():
            # --head and --model-branch have both been the flag name across
            # versions; try the other before giving up.
            command[-2] = "--model-branch"
            result = subprocess.run(command, capture_output=True, text=True)
        if not frozen.exists():
            raise SystemExit(
                f"could not freeze branch {branch!r}:\n{result.stdout[-800:]}\n{result.stderr[-800:]}"
            )
    from deepmd.calculator import DP

    return DP(model=str(frozen))


def mace_calculator(name, device, dtype="float64"):
    from mace.calculators import mace_off

    if name.startswith("mace_off23_"):
        return mace_off(model=name.split("_")[-1], device=device, default_dtype=dtype)
    if name == "mace_omol":
        from mace.calculators import MACECalculator

        return MACECalculator(
            model_paths=str(pathlib.Path.home() / ".cache/mace/MACE-omol-0-extra-large-1024.model"),
            device=device, default_dtype=dtype,
        )
    raise SystemExit(f"unknown potential {name!r}")


def _write(args, molecule_rows, profile_rows, skipped):
    import pandas as pd

    for path, frame in ((args.per_molecule, pd.DataFrame(molecule_rows)),
                        (args.profiles, pd.DataFrame(profile_rows))):
        target = pathlib.Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(target, index=False)
    quarantine = pathlib.Path(args.quarantine)
    quarantine.parent.mkdir(parents=True, exist_ok=True)
    with quarantine.open("w") as handle:
        handle.write("potential,molecule_id,reason\n")
        for potential, molecule_id, reason in skipped:
            handle.write(f'"{potential}","{molecule_id}","{str(reason)[:200]}"\n')


def _summarise(molecule_rows, skipped=()):
    import pandas as pd

    table = pd.DataFrame(molecule_rows)
    if table.empty:
        print("no profiles computed")
        return
    print(f"\n{len(table)} rows, {len(skipped)} quarantined")
    print(f"  {'potential':22s} {'n':>5}  {'barrier MAE':>12}  {'profile RMSE':>13}  {'rho':>7}")
    for name, group in table.groupby("potential"):
        print(f"  {name:22s} {len(group):5d}  "
              f"{group.barrier_error.mean():6.3f} mean  {group.rmse.median():7.3f} med  "
              f"{group.rho.median():+.3f}")
    flat = table.groupby("molecule_id").rmse_flat.first()
    print(f"  {'flat profile':22s} {len(flat):5d}  {'':12s}  {flat.median():7.3f} med")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torsiondrives", required=True)
    parser.add_argument("--potentials", default="mace_off23_medium")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", default="/home/ben/models/DPA-3.1-3M/DPA-3.1-3M.pt")
    parser.add_argument("--workdir", default="data/frozen")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--per-molecule", required=True)
    parser.add_argument("--profiles", required=True)
    parser.add_argument("--quarantine", required=True)
    args = parser.parse_args()

    from ase import Atoms
    from scipy.stats import spearmanr

    cache = np.load(args.torsiondrives, allow_pickle=False)
    entries = json.loads(str(cache["meta"]))
    if args.limit:
        entries = entries[: args.limit]
    print(f"{len(entries)} molecules from {pathlib.Path(args.torsiondrives).name}")

    molecule_rows, profile_rows, skipped = [], [], []
    for name in [p.strip() for p in args.potentials.split(",") if p.strip()]:
        if name.startswith("dpa3:"):
            calculator = dpa3_calculator(
                name.split(":", 1)[1], args.device, args.checkpoint, args.workdir
            )
        else:
            calculator = mace_calculator(name, args.device)
        started = time.time()
        done = 0
        for entry in entries:
            key = entry["key"]
            try:
                xyz = np.asarray(cache[f"xyz::{key}"], dtype=float)
                qm = np.asarray(cache[f"energy::{key}"], dtype=float)
                energies = []
                for frame in xyz:
                    atoms = Atoms(symbols=entry["symbols"], positions=frame)
                    atoms.calc = calculator
                    energies.append(atoms.get_potential_energy())
                predicted = np.asarray(energies) * EV_PER_KCAL_MOL
                predicted -= predicted.min()
            except Exception as error:  # noqa: BLE001
                skipped.append((name, entry["molecule_id"], f"{type(error).__name__}: {error}"))
                continue
            if not np.isfinite(predicted).all():
                skipped.append((name, entry["molecule_id"], "non-finite energy"))
                continue

            for point, angle in enumerate(entry["grid"]):
                profile_rows.append({
                    "molecule_id": entry["molecule_id"], "potential": name,
                    "grid_degrees": angle,
                    "qm_kcal_mol": qm[point], "predicted_kcal_mol": predicted[point],
                })
            molecule_rows.append({
                "molecule_id": entry["molecule_id"], "potential": name,
                "smiles": entry.get("smiles", ""),
                "n_atoms": entry["n_atoms"],
                "n_heavy": sum(1 for s in entry["symbols"] if s != "H"),
                "qm_barrier_kcal_mol": float(qm.max()),
                "predicted_barrier_kcal_mol": float(predicted.max()),
                "barrier_error": float(abs(predicted.max() - qm.max())),
                "rmse": float(np.sqrt(np.mean((predicted - qm) ** 2))),
                "rmse_flat": float(np.sqrt(np.mean(qm ** 2))),
                "rho": float(spearmanr(predicted, qm).statistic) if np.ptp(predicted) > 0 else np.nan,
            })
            done += 1
            if done % 50 == 0:
                rate = done / (time.time() - started)
                print(f"  [{name}] {done}/{len(entries)}  {rate:.2f} profiles/s", flush=True)
        seconds = time.time() - started
        for row in molecule_rows:
            if row["potential"] == name:
                row["seconds_per_profile"] = seconds / max(done, 1)
        print(f"  [{name}] {done} profiles in {seconds:.0f} s "
              f"({seconds/max(done,1):.3f} s each on {args.device})", flush=True)

        # Written after EVERY potential, not once at the end. The large model
        # takes 40 minutes and an earlier version made the small and medium
        # results invisible until all three finished, so a run that was two
        # thirds useful looked like a run with no output at all.
        _write(args, molecule_rows, profile_rows, skipped)
        _summarise(molecule_rows)

    _write(args, molecule_rows, profile_rows, skipped)
    _summarise(molecule_rows, skipped)


if __name__ == "__main__":
    main()
