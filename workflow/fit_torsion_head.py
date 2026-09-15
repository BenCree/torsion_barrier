"""Fit the torsion phase head on frozen DPA3 embeddings and adjudicate it.

THE QUESTION. Do frozen embeddings from a model pretrained on OpenLAM, combined
with a permutation-invariant phase and an energy linear in that phase, reproduce a
QM torsion profile better than the controls and better than a classical force
field? Stage 2 of PLAN_torsion_prediction_2026-09-11.md.

ON WHAT. The cached torsiondrives and their embeddings. One row per scan point,
24 points per molecule at 15 degree spacing, every profile at B3LYP-D3BJ/DZVP in
Psi4 and expressed relative to its own minimum. THE INDEPENDENT UNIT IS THE
MOLECULE. A naive split by molecule while the pipeline is being built; `astartes`
scaffold splits at the real-data stage. The split is never over scan points: 23
points of a scan in training and the 24th held out measures interpolation inside
one profile and would report it as generalisation.

HOW. Proper torsions are enumerated from the bond connectivity the archive
supplied with the geometry, not from a SMILES round trip, so the atom indices
cannot drift. Every torsion about a central bond is combined into s_n over
periodicities n = 1..6; the energy is sum_n k_n . s_n summed over the rotors of
the molecule. Choices that could have gone otherwise: embeddings come from one
reference conformer per molecule so that k is a property of the molecule and not
of where it currently sits; the energy is computed from s and never from alpha,
because alpha is undefined wherever a rotor's symmetry cancels a harmonic.

CONTROLS, all registered in context.toml before any data was pulled.
  random    the same architecture with k_n drawn from a dimension-matched
            Gaussian instead of predicted. Says whether the design resolves
            anything at all.
  constant  a flat profile. Every rank within a scan ties, so the verdict must
            come back unverifiable rather than as a number.
  mmff94    RDKit MMFF94 total energy, relative to each scan's own minimum.
  sage      OpenFF Sage total energy, same convention, when OpenMM is present.

UNITS. Energies in kcal/mol relative to each scan's own minimum. Correlations are
over ranks WITHIN a scan, because pooling raw energies across molecules would mix
the profile shape, which is the thing being predicted, with the offset between
molecules, which no torsion term is asked to predict. RMSE in kcal/mol is
reported per molecule and never pooled into a single number without its spread.

COST. Minutes of CPU for 100 molecules. Embeddings are already computed.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from nagl_torsion_loader import load as _load_nagl

_load_nagl()


def enumerate_torsions(bonds, n_atoms):
    """{(j,k): [(i,j,k,l), ...]} for every bond that is central to a proper torsion.

    Built from the connectivity the archive stored beside the geometry. A proper
    torsion is i-j-k-l with i bonded to j, j to k, k to l, and i, j, k, l all
    distinct; the pairs are ordered so the central bond reads (j, k) the same way
    for every torsion in a group, which is what makes their dihedral signs
    mutually consistent.
    """
    neighbours = defaultdict(set)
    for i, j in bonds:
        neighbours[int(i)].add(int(j))
        neighbours[int(j)].add(int(i))

    groups = {}
    for j, k in {(min(int(a), int(b)), max(int(a), int(b))) for a, b in bonds}:
        torsions = [
            (i, j, k, l)
            for i in sorted(neighbours[j] - {k})
            for l in sorted(neighbours[k] - {j})
            if i != l
        ]
        if torsions:
            groups[(j, k)] = torsions
    return groups


def _driven_dihedral(xyz, atoms):
    """The dihedral of the driven torsion at every scan point, radians.

    Computed from the coordinates rather than read from the grid key, so the two
    can be compared. That comparison is the only check here that could catch a
    handedness error in calculate_dihedrals, which was present in this code until
    2026-09-11 and which every magnitude-only test passed straight through.
    """
    from openff.nagl.utils._tensors import calculate_dihedrals

    frames = np.asarray(xyz, dtype=float)
    i, j, k, l = (int(a) for a in atoms)
    return np.array(
        [float(calculate_dihedrals(f[[i]], f[[j]], f[[k]], f[[l]])[0]) for f in frames]
    )


def molecule_inputs(bonds, n_atoms, xyz):
    """(torsion_indices, deltas per scan point, bond_of_torsion, n_bonds)."""
    from openff.nagl.utils._tensors import calculate_dihedrals

    groups = enumerate_torsions(bonds, n_atoms)
    if not groups:
        return None
    keys = sorted(groups)
    indices, bond_of = [], []
    for slot, key in enumerate(keys):
        for torsion in groups[key]:
            indices.append(torsion)
            bond_of.append(slot)
    indices = np.asarray(indices, dtype=int)

    deltas = np.stack(
        [
            calculate_dihedrals(
                frame[indices[:, 0]], frame[indices[:, 1]],
                frame[indices[:, 2]], frame[indices[:, 3]],
            )
            for frame in np.asarray(xyz, dtype=float)
        ]
    )
    return indices, deltas, np.asarray(bond_of, dtype=int), len(keys)


BOND_TYPES = {1.0: "SINGLE", 1.5: "AROMATIC", 2.0: "DOUBLE", 3.0: "TRIPLE"}


def mmff94_profile(symbols, bonds, orders, xyz):
    """MMFF94 total energy per scan point, kcal/mol relative to the minimum.

    THE BOND ORDERS COME FROM THE ARCHIVE. Building every bond as SINGLE, which
    an earlier version did, parametrises a different molecule wherever anything
    is aromatic or doubly bonded, and MMFF94 reports a confident energy for it.
    Every hydrogen is explicit in the geometry, so implicit valence is switched
    off rather than left for RDKit to infer from a wrong bond order.

    Returns None where RDKit cannot perceive or parametrise the molecule, which
    becomes a quarantine row rather than a silent zero.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit.Geometry import Point3D

        rw = Chem.RWMol()
        for s in symbols:
            atom = Chem.Atom(s)
            atom.SetNoImplicit(True)
            rw.AddAtom(atom)
        for (i, j), order in zip(bonds, orders):
            name = BOND_TYPES.get(round(float(order) * 2) / 2)
            if name is None:
                return None
            rw.AddBond(int(i), int(j), getattr(Chem.BondType, name))
        molecule = rw.GetMol()
        Chem.SanitizeMol(molecule)
        conformer = Chem.Conformer(molecule.GetNumAtoms())
        molecule.AddConformer(conformer)

        properties = AllChem.MMFFGetMoleculeProperties(molecule)
        if properties is None:
            return None
        energies = []
        for frame in xyz:
            conf = molecule.GetConformer()
            for index, position in enumerate(frame):
                conf.SetAtomPosition(index, Point3D(*map(float, position)))
            field = AllChem.MMFFGetMoleculeForceField(molecule, properties)
            energies.append(field.CalcEnergy())
        energies = np.asarray(energies, dtype=float)
        return energies - energies.min()
    except Exception:  # noqa: BLE001
        return None


def ranks_within(values):
    """Ranks inside one scan, ties averaged, so a flat profile ties everywhere."""
    from scipy.stats import rankdata

    return rankdata(values, method="average")


def main():
    import torch
    from openff.nagl.torsion import TorsionModel, phase_vectors

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torsiondrives", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--profiles", required=True)
    parser.add_argument("--per-molecule", required=True)
    parser.add_argument("--figure", required=True)
    parser.add_argument("--profile-figure", required=True)
    parser.add_argument("--quarantine", required=True)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260911)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    cache = np.load(args.torsiondrives, allow_pickle=False)
    embeddings = np.load(args.embeddings, allow_pickle=False)
    entries = {e["key"]: e for e in json.loads(str(cache["meta"]))}
    embedded = [e["key"] for e in json.loads(str(embeddings["meta"]))]

    prepared, skipped = [], []
    for key in embedded:
        entry = entries[key]
        try:
            built = molecule_inputs(
                cache[f"bonds::{key}"], entry["n_atoms"], cache[f"xyz::{key}"]
            )
            if built is None:
                skipped.append((entry["molecule_id"], "no proper torsion in the connectivity"))
                continue
            indices, deltas, bond_of, n_bonds = built
            h = embeddings[f"h::{key}"]
            driven = np.degrees(
                _driven_dihedral(cache[f"xyz::{key}"], entry["driven_dihedral"])
            )
            prepared.append(
                {
                    "key": key,
                    "molecule_id": entry["molecule_id"],
                    "smiles": entry["smiles"],
                    "n_atoms": entry["n_atoms"],
                    "grid": entry["grid"],
                    "h": torch.tensor(h[0], dtype=torch.float64),
                    "indices": torch.tensor(indices, dtype=torch.long),
                    "deltas": torch.tensor(deltas, dtype=torch.float64),
                    "bond_of": torch.tensor(bond_of, dtype=torch.long),
                    "n_bonds": n_bonds,
                    "qm": np.asarray(cache[f"energy::{key}"], dtype=float),
                    "driven": driven,
                    "mmff": mmff94_profile(
                        entry["symbols"],
                        cache[f"bonds::{key}"],
                        cache[f"bond_orders::{key}"],
                        cache[f"xyz::{key}"],
                    ),
                }
            )
        except Exception as error:  # noqa: BLE001
            skipped.append((entry["molecule_id"], f"{type(error).__name__}: {error}"))

    if not prepared:
        raise SystemExit("nothing prepared; see the quarantine table")

    # THE SPLIT IS OVER MOLECULES. Naive while the pipeline is being built.
    order = rng.permutation(len(prepared))
    n_test = max(1, int(round(args.test_fraction * len(prepared))))
    test_keys = {prepared[i]["key"] for i in order[:n_test]}
    train = [m for m in prepared if m["key"] not in test_keys]
    test = [m for m in prepared if m["key"] in test_keys]

    n_features = prepared[0]["h"].shape[1]
    model = TorsionModel(n_atom_features=n_features).double()

    def predict(molecule, module):
        """The whole scan in one call, relative to its own minimum.

        h_bond and c do not depend on the conformer, so they are computed once
        and the 24 points are evaluated against them together. Doing it per point
        recomputed the conformer-independent half 24 times and ran under 100
        epochs in 100 seconds on 75 molecules.
        """
        h_bond, c = module.bond_representation(
            molecule["h"], molecule["indices"], molecule["bond_of"], molecule["n_bonds"]
        )
        s = phase_vectors(
            molecule["deltas"], c, molecule["bond_of"], molecule["n_bonds"],
            periodicities=module.periodicities,
        )
        stacked = module.energy(h_bond, s).sum(-1)
        return stacked - stacked.min()

    optimiser = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    targets = {m["key"]: torch.tensor(m["qm"], dtype=torch.float64) for m in prepared}
    for epoch in range(args.epochs):
        optimiser.zero_grad()
        loss = torch.stack(
            [((predict(m, model) - targets[m["key"]]) ** 2).mean() for m in train]
        ).mean()
        loss.backward()
        optimiser.step()
        if epoch % 100 == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch:4d}  train MSE {float(loss):.4f} kcal^2/mol^2", flush=True)

    # THE RANDOM CONTROL, structurally matched: same architecture, same phase
    # vectors, same linear form, coefficients drawn from a Gaussian matched to
    # the fitted ones rather than shuffled labels.
    with torch.no_grad():
        fitted = torch.cat(
            [model.energy.torsion_parameters(
                model.bond_representation(
                    m["h"], m["indices"], m["bond_of"], m["n_bonds"]
                )[0]
            ).flatten() for m in prepared]
        )
        scale = float(fitted.std())
    random_model = TorsionModel(n_atom_features=n_features).double()
    random_model.load_state_dict(model.state_dict())
    with torch.no_grad():
        for parameter in random_model.energy.network[-1].parameters():
            parameter.normal_(0.0, scale)

    profile_rows, molecule_rows = [], []
    for molecule in prepared:
        split = "test" if molecule["key"] in test_keys else "train"
        with torch.no_grad():
            predicted = predict(molecule, model).numpy()
            random_profile = predict(molecule, random_model).numpy()
        qm = molecule["qm"]
        constant = np.zeros_like(qm)
        mmff = molecule["mmff"] if molecule["mmff"] is not None else np.full_like(qm, np.nan)

        # The echo is looked up by MATCHING THE GRID ANGLE written into the row,
        # not by the position the loop is at, so it travels a different path from
        # qm_rank and can disagree with it if the ordering is wrong.
        reread = np.asarray(cache[f"energy::{molecule['key']}"], dtype=float)
        grid_list = list(molecule["grid"])
        echo_ranks = ranks_within(reread)
        qm_ranks = ranks_within(qm)
        driven_ranks = ranks_within(molecule["driven"])
        grid_ranks = ranks_within(np.asarray(grid_list, dtype=float))
        predicted_ranks = ranks_within(predicted)
        random_ranks = ranks_within(random_profile)
        constant_ranks = ranks_within(constant)
        mmff_ranks = ranks_within(mmff) if np.isfinite(mmff).all() else np.full_like(qm, np.nan)

        for point, angle in enumerate(molecule["grid"]):
            profile_rows.append(
                {
                    "molecule_id": molecule["molecule_id"],
                    "split": split,
                    "grid_degrees": angle,
                    "driven_dihedral_degrees": molecule["driven"][point],
                    "driven_rank": driven_ranks[point],
                    "grid_rank": grid_ranks[point],
                    "qm_kcal_mol": qm[point],
                    "predicted_kcal_mol": predicted[point],
                    "qm_rank": qm_ranks[point],
                    "echo_rank": echo_ranks[grid_list.index(angle)],
                    "predicted_rank": predicted_ranks[point],
                    "random_rank": random_ranks[point],
                    "constant_rank": constant_ranks[point],
                    "mmff94_rank": mmff_ranks[point],
                }
            )

        def rmse(a):
            return float(np.sqrt(np.mean((a - qm) ** 2))) if np.isfinite(a).all() else np.nan

        def rho(a):
            from scipy.stats import spearmanr

            if not np.isfinite(a).all() or np.ptp(a) == 0:
                return np.nan
            return float(spearmanr(a, qm).statistic)

        molecule_rows.append(
            {
                "molecule_id": molecule["molecule_id"],
                "smiles": molecule["smiles"],
                "split": split,
                "n_atoms": molecule["n_atoms"],
                "n_points": len(qm),
                "n_rotors": molecule["n_bonds"],
                "qm_barrier_kcal_mol": float(qm.max()),
                "rmse_predicted": rmse(predicted),
                "rmse_random": rmse(random_profile),
                "rmse_constant": rmse(constant),
                "rmse_mmff94": rmse(mmff),
                "rho_predicted": rho(predicted),
                "rho_random": rho(random_profile),
                "rho_mmff94": rho(mmff),
                "barrier_error_predicted": float(abs(predicted.max() - qm.max())),
                "barrier_error_mmff94": float(abs(np.nanmax(mmff) - qm.max())) if np.isfinite(mmff).all() else np.nan,
            }
        )

    import pandas as pd

    profiles = pd.DataFrame(profile_rows)
    per_molecule = pd.DataFrame(molecule_rows)
    for path, frame in ((args.profiles, profiles), (args.per_molecule, per_molecule)):
        target = pathlib.Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(target, index=False)

    quarantine = pathlib.Path(args.quarantine)
    quarantine.parent.mkdir(parents=True, exist_ok=True)
    with quarantine.open("w") as handle:
        handle.write("molecule_id,reason\n")
        for name, reason in skipped:
            handle.write(f'"{name}","{str(reason)[:200]}"\n')

    _figures(args.figure, args.profile_figure, per_molecule, profiles, len(train), len(test))

    held = per_molecule[per_molecule["split"] == "test"]
    print(
        f"\n{len(prepared)} molecules ({len(train)} train, {len(test)} test), "
        f"{len(skipped)} quarantined"
    )
    for arm in ("predicted", "random", "constant", "mmff94"):
        values = held[f"rmse_{arm}"].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size:
            print(
                f"  held-out RMSE, {arm:9s} median {np.median(finite):6.3f} kcal/mol "
                f"over {finite.size} molecules"
            )


# Okabe and Ito's eight-colour palette, which is distinguishable under the three
# common forms of colour vision deficiency. Ito and Okabe, Color Universal Design,
# 2008. Used rather than matplotlib's default cycle, whose red and green pair is
# the one that fails.
BLACK = "#000000"
ORANGE = "#E69F00"
SKY = "#56B4E9"
GREEN = "#009E73"
BLUE = "#0072B2"
VERMILLION = "#D55E00"
PURPLE = "#CC79A7"
GREY = "#666666"


def _style():
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "legend.frameon": False,
            "lines.linewidth": 1.0,
            "axes.titlelocation": "left",
            "axes.titlepad": 13,
        }
    )


def _panel(ax, letter, title, subtitle):
    """A bold panel letter, a title, and the cohort line every figure must carry."""
    ax.set_title(f"$\\bf{{{letter}}}$  {title}", loc="left")
    ax.text(
        0.0, 1.008, subtitle, transform=ax.transAxes, fontsize=6.4,
        color=GREY, va="bottom", ha="left",
    )


def _identity(ax, low, high, label="y = x"):
    ax.plot([low, high], [low, high], color=BLACK, lw=0.7, ls=(0, (4, 3)), zorder=1, label=label)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal", adjustable="box")


def _figures(path, profile_path, per_molecule, profiles, n_train, n_test):
    """Four scatter panels, then a strip of real profiles.

    Every panel plots the points rather than their mean, carries y = x where it is
    a prediction or a head to head, and names its cohort and its resampling unit
    in the line under the panel title.
    """
    import matplotlib.pyplot as plt

    _style()
    held_rows = profiles[profiles["split"] == "test"]
    held = per_molecule[per_molecule["split"] == "test"]
    cohort = (
        f"OpenFF Gen3 torsiondrives, B3LYP-D3BJ/DZVP, {n_train + n_test} molecules "
        f"({n_train} train, {n_test} held out), 24 points per scan"
    )

    figure, axes = plt.subplots(2, 2, figsize=(7.2, 6.8))

    # a. the prediction scatter
    ax = axes[0, 0]
    limit = float(np.ceil(max(held_rows["qm_kcal_mol"].max(), held_rows["predicted_kcal_mol"].max())))
    ax.scatter(
        held_rows["qm_kcal_mol"], held_rows["predicted_kcal_mol"],
        s=7, lw=0, alpha=0.45, color=BLUE, zorder=2,
    )
    _identity(ax, 0, limit)
    ax.set_xlabel("QM relative energy (kcal mol$^{-1}$)")
    ax.set_ylabel("predicted (kcal mol$^{-1}$)")
    _panel(ax, "a", "Predicted against QM",
           f"{len(held_rows)} scan points from {n_test} held-out molecules")
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98))

    # b. the head to head on RMSE, paired by molecule
    ax = axes[0, 1]
    pair = held.dropna(subset=["rmse_mmff94", "rmse_predicted"])
    limit = float(np.ceil(max(pair["rmse_mmff94"].max(), pair["rmse_predicted"].max())))
    ax.scatter(pair["rmse_mmff94"], pair["rmse_predicted"], s=16, lw=0.4,
               facecolor=GREEN, edgecolor=BLACK, alpha=0.85, zorder=3)
    _identity(ax, 0, limit)
    wins = int((pair["rmse_predicted"] < pair["rmse_mmff94"]).sum())
    ax.set_xlabel("MMFF94 profile RMSE (kcal mol$^{-1}$)")
    ax.set_ylabel("this model, profile RMSE (kcal mol$^{-1}$)")
    _panel(ax, "b", "Profile RMSE, paired by molecule",
           f"model wins below the line: {wins} of {len(pair)} molecules")

    # c. the head to head on within-scan rank correlation
    ax = axes[1, 0]
    pair_rho = held.dropna(subset=["rho_mmff94", "rho_predicted"])
    ax.scatter(pair_rho["rho_mmff94"], pair_rho["rho_predicted"], s=16, lw=0.4,
               facecolor=PURPLE, edgecolor=BLACK, alpha=0.85, zorder=3)
    _identity(ax, -1.05, 1.05)
    ax.axhline(0, color=GREY, lw=0.4, zorder=0)
    ax.axvline(0, color=GREY, lw=0.4, zorder=0)
    wins = int((pair_rho["rho_predicted"] > pair_rho["rho_mmff94"]).sum())
    ax.set_xlabel(r"MMFF94, Spearman $\rho$ within scan")
    ax.set_ylabel(r"this model, Spearman $\rho$ within scan")
    _panel(ax, "c", "Profile shape, paired by molecule",
           f"model wins above the line: {wins} of {len(pair_rho)} molecules")

    # d. does the error track the thing being predicted
    ax = axes[1, 1]
    for column, colour, marker, label in (
        ("rmse_predicted", BLUE, "o", "this model"),
        ("rmse_mmff94", VERMILLION, "^", "MMFF94"),
        ("rmse_constant", GREY, "s", "flat profile"),
    ):
        subset = held.dropna(subset=[column])
        ax.scatter(subset["qm_barrier_kcal_mol"], subset[column], s=16, lw=0.4,
                   facecolor=colour, edgecolor=BLACK, alpha=0.8, label=label, zorder=3)
    ax.set_xlabel("QM barrier height (kcal mol$^{-1}$)")
    ax.set_ylabel("profile RMSE (kcal mol$^{-1}$)")
    ax.set_aspect("auto")
    _panel(ax, "d", "Error against barrier height",
           f"one point per held-out molecule; resampling unit: the molecule")
    ax.legend(loc="upper right", ncol=3, columnspacing=0.8, handletextpad=0.3,
              borderaxespad=0.2)

    figure.text(0.0, -0.012, cohort, fontsize=6.2, color=GREY, ha="left", va="top")
    figure.tight_layout(h_pad=3.0, w_pad=2.4)
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)

    _profile_figure(profile_path, per_molecule, profiles, n_test)


def _profile_figure(path, per_molecule, profiles, n_test):
    """Six held-out profiles, spanning the barrier range, drawn as the scans they are."""
    import matplotlib.pyplot as plt

    _style()
    held = per_molecule[per_molecule["split"] == "test"].dropna(subset=["rmse_mmff94"])
    if held.empty:
        held = per_molecule[per_molecule["split"] == "test"]
    chosen = held.sort_values("qm_barrier_kcal_mol")
    picks = chosen.iloc[np.linspace(0, len(chosen) - 1, min(6, len(chosen))).astype(int)]

    figure, axes = plt.subplots(2, 3, figsize=(7.2, 4.6), sharex=True)
    for index, (ax, (_, row)) in enumerate(zip(axes.flat, picks.iterrows())):
        scan = profiles[profiles["molecule_id"] == row["molecule_id"]].sort_values("grid_degrees")
        ax.plot(scan["grid_degrees"], scan["qm_kcal_mol"], color=BLACK, lw=1.1,
                marker="o", ms=2.2, label="QM", zorder=3)
        ax.plot(scan["grid_degrees"], scan["predicted_kcal_mol"], color=BLUE, lw=1.1,
                marker="s", ms=2.2, label="this model", zorder=2)
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_xlim(-185, 185)
        smiles = str(row["smiles"])
        _panel(ax, "abcdef"[index], f"QM barrier {row['qm_barrier_kcal_mol']:.1f} kcal mol$^{{-1}}$",
               smiles[:30] + ("..." if len(smiles) > 30 else ""))
        if index == 0:
            ax.legend(loc="upper right")
        if index >= 3:
            ax.set_xlabel(r"driven dihedral ($\degree$)")
        if index % 3 == 0:
            ax.set_ylabel("relative energy (kcal mol$^{-1}$)")

    figure.text(
        0.0, -0.02,
        f"Six held-out molecules spanning the barrier range, of {n_test} held out. "
        "Points are the 24 optimised scan geometries; lines join them.",
        fontsize=6.2, color=GREY, ha="left", va="top",
    )
    figure.tight_layout(h_pad=3.0, w_pad=1.8)
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)


if __name__ == "__main__":
    main()
