"""Can the model produce its profile from a SMILES string alone?

QUESTION. Every accuracy number in RESULT_stage3.md was computed with the true
dihedrals of all 24 QM-optimised scan geometries in hand, because `run_arm`'s
reference arm embeds one geometry and then reads the phase off each real frame.
Every cost number was computed the other way, one encoder pass and the phase
rotated in closed form, which is the only thing deployment can do. Enamine REAL
and ZINC22 distribute SMILES, not coordinates, so a precomputed torsional table
starts from a conformer somebody generated, not from a torsiondrive. This asks
what that costs. What would answer it either way: the RMSE between the profile
predicted from a generated conformer and the QM profile, against 0.767 kcal/mol,
which is how far two DFT references for the same molecules sit apart, and
against the 1.254 kcal/mol the privileged path reaches on the same molecules.

ON WHAT. The held-out molecules of the merged pool, addressed by SMILES and
connectivity. One independent unit is one molecule.

HOW. Four geometry sources through ONE code path. They are all `reference`-arm
records and differ only in `reference_xyz` and `deltas`:

  privileged   the QM minimum embedded, with the true dihedrals of all 24
               optimised scan frames. This reproduces the published arm and
               needs 24 QM geometries per molecule.
  qm_rotated   the QM minimum embedded, with the phase rotated in closed form,
               delta_ij(phi) = delta_ij(0) + phi. One QM geometry.
  etkdg        an ETKDGv3 conformer generated from the SMILES and minimised
               under MMFF94s, rotated the same way. No QM at all.
  noise        the QM minimum displaced by Gaussian noise whose scale is the
               measured median ETKDG-to-QM heavy-atom RMSD. The random control:
               the same displacement with none of the chemistry.

Alignment is absolute, not relative. A generated conformer sits at its own
driven dihedral phi_0, so the rotation applied at grid point p is
radians(grid_p) - phi_0 and index p is the molecule turned to the same absolute
angle the QM scan reports. Comparing without this compares two profiles that
have been slid past each other.

The 2D input is the archive's own connectivity, not its SMILES string: elements,
bonds and bond orders, no coordinates, which is exactly what a SMILES supplies
and what Enamine distributes. Parsing the pool's SMILES instead loses a quarter
of the molecules to unwritten formal charges, and they are the charged species
this model is best at.

CONTROLS. Standard, the privileged arm, which is the number already reported.
Random, the noise source above.

UNITS. kcal/mol. One number is the RMSE across 24 grid points between two
profiles of one molecule's driven bond, each taken relative to its own minimum.
Reported as the median over molecules with a bootstrap interval resampled over
molecules.

COST. One frozen-arm training run on the same split, then n_molecules x (3 + K)
encoder passes at 18.4 ms each. Minutes of GPU.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

BOND_TYPES = {1.0: "SINGLE", 1.5: "AROMATIC", 2.0: "DOUBLE", 3.0: "TRIPLE"}


DEFAULT_VALENCE = {"B": 3, "C": 4, "N": 3, "O": 2, "P": 3, "S": 2,
                   "F": 1, "Cl": 1, "Br": 1, "I": 1}


def molecule_from_cache(symbols, bonds, orders):
    """An RDKit molecule in the ARCHIVE's atom order, with the archive's orders.

    This is the 2D input a SMILES would supply: elements, bonds and bond orders,
    no coordinates. Going through the SMILES string instead is a detour that
    fails on exactly the molecules that matter. 26 percent of the merged pool's
    SMILES will not parse ("Explicit valence for atom # 9 N, 4, is greater than
    permitted") because a quaternary nitrogen is written without its charge, and
    those are the charged species the model is best at.

    Hydrogens are all explicit in a torsiondrive geometry, so implicit valence is
    switched off. Formal charges are then assigned from valence, since an atom
    bonded past its default valence is charged and RDKit will not guess it.
    """
    from rdkit import Chem

    rw = Chem.RWMol()
    for symbol in symbols:
        atom = Chem.Atom(symbol)
        atom.SetNoImplicit(True)
        rw.AddAtom(atom)
    for (i, j), order in zip(bonds, orders):
        name = BOND_TYPES.get(round(float(order) * 2) / 2)
        if name is None:
            raise ValueError(f"bond order {order} is not one of {sorted(BOND_TYPES)}")
        rw.AddBond(int(i), int(j), getattr(Chem.BondType, name))

    for atom in rw.GetAtoms():
        default = DEFAULT_VALENCE.get(atom.GetSymbol())
        if default is None:
            continue
        valence = sum(b.GetBondTypeAsDouble() for b in atom.GetBonds())
        excess = int(round(valence)) - default
        if excess > 0:
            # Nitrogen and oxygen go positive when over-coordinated, boron
            # negative. Sulfur and phosphorus expand instead, so they are left.
            if atom.GetSymbol() in ("N", "O"):
                atom.SetFormalCharge(excess)
            elif atom.GetSymbol() == "B":
                atom.SetFormalCharge(-excess)

    molecule = rw.GetMol()
    Chem.SanitizeMol(molecule)
    return molecule


def etkdg_conformers(reference, n_conformers, seed):
    """Conformers from connectivity alone, in the reference's own atom order.

    Returns (coordinates, seconds, n_embedded). The timing is the number the
    Fugaku estimate needs, so it covers embedding and minimisation and nothing
    else: no file reading, no molecule construction.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    target = Chem.Mol(reference)
    target.RemoveAllConformers()

    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = int(seed)
    parameters.pruneRmsThresh = -1.0     # every conformer is kept; see build_records
    parameters.useRandomCoords = False

    started = time.perf_counter()
    ids = AllChem.EmbedMultipleConfs(target, numConfs=int(n_conformers),
                                     params=parameters)
    if len(ids):
        AllChem.MMFFOptimizeMoleculeConfs(target, maxIters=500)
    seconds = time.perf_counter() - started
    if not len(ids):
        raise ValueError("ETKDGv3 embedded no conformer")

    coordinates = np.stack(
        [target.GetConformer(int(i)).GetPositions() for i in ids]
    )
    return coordinates, seconds, len(ids)


def heavy_rmsd(a, b, symbols):
    """Heavy-atom RMSD after optimal superposition, in Angstrom."""
    keep = np.asarray([s != "H" for s in symbols])
    x, y = np.asarray(a)[keep], np.asarray(b)[keep]
    x = x - x.mean(0)
    y = y - y.mean(0)
    u, _, vt = np.linalg.svd(x.T @ y)
    d = np.sign(np.linalg.det(u @ vt))
    rotation = u @ np.diag([1.0, 1.0, d]) @ vt
    return float(np.sqrt(((x @ rotation - y) ** 2).sum(1).mean()))


def rotated_deltas(delta0, grid, phi0):
    """Dihedrals of every torsion about the driven bond, at each grid angle.

    Turning the driven bond by phi adds phi to EVERY torsion whose central bond
    it is, exactly, for a rigid rotation. So the phase at grid angle g is the
    phase of this geometry plus the difference between g and where this geometry
    already sits. It is the same closed form `rotate_phase` applies to s_n, done
    one step earlier so that both paths reach the head through one function.
    """
    increments = np.radians(np.asarray(grid, dtype=float)) - phi0
    return np.asarray(delta0, dtype=float)[None, :] + increments[:, None]


def wrap(angle):
    """To (-pi, pi]."""
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def rmse(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - a.min() - (b - b.min())) ** 2)))


def boot_ci(values, n=2000, seed=0):
    values = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if values.size < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, values.size, size=(n, values.size))
    medians = np.median(values[draws], axis=1)
    return float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))


def build_records(prepared, args, quarantine):
    """One record per (molecule, geometry source), ready for `run_arm`.

    Every record keeps the molecule's own key in `molecule_id` and carries the
    source in `source`, so the predictions dictionary `run_arm` returns can be
    taken apart again afterwards.
    """
    import torch
    from openff.nagl.utils._tensors import calculate_dihedrals
    from deepmd.pt.utils.env import DEVICE

    records, timings, rmsds = [], [], []
    for molecule in prepared:
        indices = molecule["indices"].cpu().numpy()
        grid = molecule["grid"]
        qm_reference = molecule["reference_xyz"][0]

        def deltas_of(frame):
            return calculate_dihedrals(
                frame[indices[:, 0]], frame[indices[:, 1]],
                frame[indices[:, 2]], frame[indices[:, 3]],
            )

        def driven_angle(frame):
            a = molecule["driven_dihedral"]
            return float(calculate_dihedrals(
                frame[None, a[0]], frame[None, a[1]],
                frame[None, a[2]], frame[None, a[3]],
            )[0])

        def add(source, frame, deltas, extra=None):
            records.append(dict(
                molecule,
                key=f"{molecule['key']}|{source}",
                molecule_id=molecule["molecule_id"],
                source=source,
                reference_xyz=np.asarray(frame, dtype=float)[None],
                deltas=torch.tensor(np.asarray(deltas, dtype=float),
                                    dtype=torch.float64, device=DEVICE),
                extra=extra or {},
            ))

        # 1. the published path: one embedding, the true dihedrals of 24 frames.
        add("privileged", qm_reference, molecule["deltas"].cpu().numpy())

        # 2. the same geometry, phase rotated in closed form.
        phi0 = driven_angle(qm_reference)
        add("qm_rotated", qm_reference,
            rotated_deltas(deltas_of(qm_reference), grid, phi0))

        # 3. generated conformers, nothing but the SMILES.
        try:
            conformers, seconds, n_made = etkdg_conformers(
                molecule_from_cache(molecule["symbols"], molecule["bonds"],
                                    molecule["bond_orders"]),
                args.conformers, args.seed,
            )
        except Exception as error:                        # noqa: BLE001
            quarantine.append((molecule["molecule_id"],
                               f"conformers: {type(error).__name__}: {error}"))
            conformers, seconds, n_made = None, float("nan"), 0
        if n_made:
            timings.append(seconds / n_made)
            for c, frame in enumerate(conformers):
                d = heavy_rmsd(frame, qm_reference, molecule["symbols"])
                rmsds.append(d)
                add(f"etkdg{c}", frame,
                    rotated_deltas(deltas_of(frame), grid, driven_angle(frame)),
                    {"heavy_rmsd": d, "etkdg_seconds": seconds / n_made})

        molecule["_n_conformers"] = n_made
    return records, timings, rmsds


def add_noise_control(records, prepared, sigma, seed):
    """The random control, added after sigma is known from the real conformers."""
    import torch
    from openff.nagl.utils._tensors import calculate_dihedrals
    from deepmd.pt.utils.env import DEVICE

    rng = np.random.default_rng(seed)
    by_id = {m["molecule_id"]: m for m in prepared}
    made = []
    for molecule in by_id.values():
        if not molecule.get("_n_conformers"):
            continue
        indices = molecule["indices"].cpu().numpy()
        frame = molecule["reference_xyz"][0] + rng.normal(
            0.0, sigma, size=molecule["reference_xyz"][0].shape
        )
        d0 = calculate_dihedrals(frame[indices[:, 0]], frame[indices[:, 1]],
                                 frame[indices[:, 2]], frame[indices[:, 3]])
        a = molecule["driven_dihedral"]
        phi0 = float(calculate_dihedrals(frame[None, a[0]], frame[None, a[1]],
                                         frame[None, a[2]], frame[None, a[3]])[0])
        made.append(dict(
            molecule,
            key=f"{molecule['key']}|noise",
            molecule_id=molecule["molecule_id"],
            source="noise",
            reference_xyz=frame[None],
            deltas=torch.tensor(rotated_deltas(d0, molecule["grid"], phi0),
                                dtype=torch.float64, device=DEVICE),
            extra={"heavy_rmsd": heavy_rmsd(frame, molecule["reference_xyz"][0],
                                            molecule["symbols"])},
        ))
    records.extend(made)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torsiondrives", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--head", default="SPICE2")
    parser.add_argument("--conformers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0,
                        help="0 for every molecule in the held-out split")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--out", required=True)
    parser.add_argument("--profiles", required=True)
    parser.add_argument("--quarantine", required=True)
    parser.add_argument("--figure", required=True)
    args = parser.parse_args()

    import torch
    from finetune_torsion import (prepare, build_head, split_molecules, run_arm,
                                  build_descriptor)
    from deepmd.pt.utils.env import DEVICE

    cache = np.load(args.torsiondrives, allow_pickle=False)
    entries = json.loads(str(cache["meta"]))
    prepared, skipped = prepare(cache, entries, rotors="driven")
    quarantine = [(m, r) for m, r in skipped]

    # prepare() drops the connectivity it used; the conformer builder needs it.
    for molecule in prepared:
        molecule["bonds"] = cache[f"bonds::{molecule['key']}"]
        molecule["bond_orders"] = cache[f"bond_orders::{molecule['key']}"]
        entry = next(e for e in entries if e["key"] == molecule["key"])
        molecule["driven_dihedral"] = list(entry["driven_dihedral"])

    rng = np.random.default_rng(args.seed)

    class _Split:
        val_fraction, test_fraction, split_mode = 0.15, 0.25, "molecule"
        sampler, folds, fold_index, seed = "scaffold", 1, -1, args.seed

    # The same cohort, sampler, fractions and seed the published arms used, so
    # the privileged number below is comparable to the 1.289 the frozen arm
    # reached in RESULT_stage3.md. It is ONE fold of that five-fold run, so the
    # held-out set is a quarter of the size and the interval is wider.
    train_idx, val_idx, test_idx, how = split_molecules(prepared, _Split, rng)
    train = [prepared[i] for i in train_idx]
    validation = [prepared[i] for i in val_idx]
    held_out = [prepared[i] for i in test_idx]
    if args.limit:
        held_out = held_out[:args.limit]
    print(f"{len(train)} train, {len(validation)} validation, "
          f"{len(held_out)} held out, split by {how}", flush=True)

    records, timings, rmsds = build_records(held_out, args, quarantine)
    sigma = float(np.median(rmsds)) / np.sqrt(3.0) if rmsds else 0.0
    records = add_noise_control(records, held_out, sigma, args.seed)
    print(f"{len(records)} records over {len(held_out)} molecules; "
          f"ETKDG {np.median(timings)*1e3:.1f} ms per conformer; "
          f"heavy-atom RMSD to QM median {np.median(rmsds):.3f} A; "
          f"noise sigma {sigma:.3f} A per coordinate", flush=True)

    class _Args:
        pass
    a = _Args()
    for name, value in dict(
        checkpoint=args.checkpoint, config=args.config, head=args.head,
        epochs=args.epochs, head_lr=0.003, encoder_lr=3e-5, batch_size=0,
        clip=5.0, torsion_hidden="128,128", coefficient_hidden="64",
        energy_hidden="128", periodicities=6, nonlinear_head=False,
        weight_decay=0.01, encoder_anchor=1.0, ema=0.999, patience=60,
        rotors="driven", seed=args.seed, figure_every=0, live_figure=None,
    ).items():
        setattr(a, name, value)

    descriptor, type_map = build_descriptor(args.config, args.checkpoint, args.head)
    initial = build_head(descriptor.get_dim_out(), a).to(DEVICE).state_dict()
    del descriptor

    history = pathlib.Path(args.out).with_name(
        pathlib.Path(args.out).stem + "_history.csv")
    history.write_text("arm,fold,epoch,train_mse,val_rmse,best_rmse,"
                       "best_epoch,seconds\n")
    predictions, seconds, epoch, val = run_arm(
        "frozen", False, "reference", records, train, validation, a, type_map,
        initial, history_path=history,
    )
    print(f"trained in {seconds/60:.1f} min, best epoch {epoch}, "
          f"validation RMSE {val:.4f}", flush=True)

    by_source = {}
    for record in records:
        by_source.setdefault(record["molecule_id"], {})[record["source"]] = \
            predictions[record["key"]]

    rows, profile_rows = [], []
    for molecule in held_out:
        got = by_source.get(molecule["molecule_id"], {})
        if "privileged" not in got:
            continue
        qm = molecule["qm"]
        conformer_keys = sorted(k for k in got if k.startswith("etkdg"))
        for source, prediction in got.items():
            rows.append({
                "molecule_id": molecule["molecule_id"],
                "smiles": molecule["smiles"],
                "n_atoms": molecule["n_atoms"],
                "heavy_atoms": sum(1 for s in molecule["symbols"] if s != "H"),
                "source": "etkdg" if source.startswith("etkdg") else source,
                "replicate": source[5:] if source.startswith("etkdg") else "",
                "rmse_vs_qm": rmse(prediction, qm),
                "rmse_vs_privileged": rmse(prediction, got["privileged"]),
                "barrier_predicted": float(np.ptp(prediction)),
                "barrier_qm": float(np.ptp(qm)),
                "heavy_rmsd": float(next(
                    (r["extra"].get("heavy_rmsd", float("nan"))
                     for r in records
                     if r["molecule_id"] == molecule["molecule_id"]
                     and r["source"] == source), float("nan"))),
            })
        if conformer_keys:
            stack = np.stack([got[k] for k in conformer_keys])
            stack = stack - stack.min(axis=1, keepdims=True)
            rows.append({
                "molecule_id": molecule["molecule_id"],
                "smiles": molecule["smiles"],
                "n_atoms": molecule["n_atoms"],
                "heavy_atoms": sum(1 for s in molecule["symbols"] if s != "H"),
                "source": "etkdg_spread", "replicate": "",
                "rmse_vs_qm": float(np.sqrt(np.mean(stack.var(axis=0)))),
                "rmse_vs_privileged": float("nan"),
                "barrier_predicted": float(np.ptp(stack.mean(0))),
                "barrier_qm": float(np.ptp(qm)),
                "heavy_rmsd": float("nan"),
            })
        for p, angle in enumerate(molecule["grid"]):
            row = {"molecule_id": molecule["molecule_id"], "angle": float(angle),
                   "qm": float(qm[p] - qm.min())}
            for source, prediction in got.items():
                name = "etkdg0" if source == "etkdg0" else source
                if name in ("privileged", "qm_rotated", "etkdg0", "noise"):
                    row[name] = float(prediction[p] - prediction.min())
            profile_rows.append(row)

    import csv
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pathlib.Path(args.profiles).open("w", newline="") as fh:
        names = ["molecule_id", "angle", "qm", "privileged", "qm_rotated",
                 "etkdg0", "noise"]
        writer = csv.DictWriter(fh, fieldnames=names, extrasaction="ignore",
                                restval="")
        writer.writeheader()
        writer.writerows(profile_rows)
    with pathlib.Path(args.quarantine).open("w", newline="") as fh:
        fh.write("molecule_id,reason\n")
        for molecule_id, reason in quarantine:
            fh.write(f'{molecule_id},"{reason}"\n')

    print(f"\n{'source':<16}{'n':>6}{'median RMSE to QM':>20}"
          f"{'95% CI over molecules':>26}")
    summary = {}
    for source in ("privileged", "qm_rotated", "etkdg", "noise", "etkdg_spread"):
        values = [r["rmse_vs_qm"] for r in rows if r["source"] == source]
        if not values:
            continue
        low, high = boot_ci(values, seed=args.seed)
        summary[source] = (float(np.median(values)), low, high, len(values))
        print(f"{source:<16}{len(values):>6}{np.median(values):>20.3f}"
              f"{f'[{low:.3f}, {high:.3f}]':>26}")
    print(f"\nETKDG conformer generation: {np.median(timings)*1e3:.1f} ms per "
          f"conformer, median over {len(timings)} molecules")

    figure(args.figure, rows, profile_rows, summary, held_out, timings)
    print(f"  -> {args.out}\n  -> {args.figure}")
    return 0


def figure(path, rows, profile_rows, summary, held_out, timings):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE, ORANGE, GREEN, VERM, GREY = ("#0072B2", "#E69F00", "#009E73",
                                       "#D55E00", "#666666")
    matplotlib.rcParams.update({
        "figure.dpi": 200, "savefig.dpi": 400, "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 7, "axes.spines.top": False, "axes.spines.right": False,
    })
    colour = {"privileged": BLUE, "qm_rotated": GREEN, "etkdg": ORANGE,
              "noise": GREY}
    label = {"privileged": "privileged: 24 QM geometries",
             "qm_rotated": "one QM geometry, phase rotated",
             "etkdg": "ETKDG conformer from the SMILES",
             "noise": "random control: displaced QM geometry"}

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))

    ax = axes[0]
    for source in ("privileged", "qm_rotated", "etkdg", "noise"):
        values = np.asarray([r["rmse_vs_qm"] for r in rows
                             if r["source"] == source], dtype=float)
        if not values.size:
            continue
        order = np.sort(values)
        ax.plot(order, np.linspace(0, 1, order.size), lw=1.2,
                color=colour[source], label=label[source])
    ax.axvline(0.767, color="black", ls=":", lw=0.8)
    ax.text(0.767, 0.03, " label uncertainty", fontsize=5.6, rotation=90,
            va="bottom", ha="left")
    ax.set_xscale("log")
    ax.set_xlabel("profile RMSE to QM (kcal/mol)")
    ax.set_ylabel("fraction of molecules")
    ax.set_title("a  what the geometry costs", loc="left", fontsize=8,
                 fontweight="bold")
    ax.legend(frameon=False, fontsize=5.4, loc="upper left")

    ax = axes[1]
    x = np.asarray([r["heavy_rmsd"] for r in rows if r["source"] == "etkdg"],
                   dtype=float)
    y = np.asarray([r["rmse_vs_privileged"] for r in rows
                    if r["source"] == "etkdg"], dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    ax.plot(x[keep], y[keep], "o", ms=1.6, alpha=0.35, color=ORANGE, mew=0)
    ax.set_xlabel("heavy-atom RMSD, conformer to QM minimum (A)")
    ax.set_ylabel("RMSE to the privileged profile\n(kcal/mol)")
    ax.set_title("b  does a worse conformer predict worse?", loc="left",
                 fontsize=8, fontweight="bold")

    ax = axes[2]
    ids = sorted({r["molecule_id"] for r in profile_rows})
    pick = ids[len(ids) // 3] if ids else None
    sub = sorted((r for r in profile_rows if r["molecule_id"] == pick),
                 key=lambda r: r["angle"])
    if sub:
        angles = [r["angle"] for r in sub]
        ax.plot(angles, [r["qm"] for r in sub], "k-", lw=1.4, label="QM")
        for source in ("privileged", "qm_rotated", "etkdg0"):
            values = [r.get(source) for r in sub]
            if all(v is not None and v != "" for v in values):
                ax.plot(angles, [float(v) for v in values], lw=1.0,
                        color=colour["etkdg" if source == "etkdg0" else source],
                        label=label["etkdg" if source == "etkdg0" else source]
                        .split(":")[0])
        ax.set_xlabel("driven dihedral (degrees)")
        ax.set_ylabel("relative energy (kcal/mol)")
        ax.set_title("c  one molecule, three inputs", loc="left", fontsize=8,
                     fontweight="bold")
        ax.legend(frameon=False, fontsize=5.4)

    n_mol = len({r["molecule_id"] for r in rows})
    fig.text(0.5, -0.09,
             f"Held-out OpenFF Gen3 molecules, n = {n_mol}, resampled over "
             f"molecules. ETKDGv3 + MMFF94s at "
             f"{np.median(timings)*1e3:.0f} ms per conformer.",
             ha="center", fontsize=6.2, color=GREY)
    fig.tight_layout()
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)


if __name__ == "__main__":
    sys.exit(main())
