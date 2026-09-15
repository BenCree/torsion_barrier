"""Frozen against fine-tuned DPA3, both arms in one run on one split.

THE QUESTION. Stage 2 showed that a frozen OpenLAM representation with a
permutation-invariant phase head beats predicting a flat profile (19 of 23 held-out
molecules, p = 0.0026) and does not beat MMFF94 (15 of 23, p = 0.210). Does
fine-tuning the encoder on the QM torsion data close that gap? A paired
comparison over held-out molecules whose interval excludes zero answers it either
way; one that spans zero says 25 molecules could not tell.

ON WHAT. The cached torsiondrives, 24 points per scan at B3LYP-D3BJ/DZVP in
Psi4, energies relative to each scan's own minimum. THE INDEPENDENT UNIT IS THE
MOLECULE and the split is over molecules, never over scan points.

WHICH ROTORS CARRY THE ENERGY, and why it decides whether the model is a
function of the bond at all.

A torsiondrive turns ONE bond. The QM profile is one number per scan point, so a
model that sums a Fourier series over every central bond in the molecule is fitted
against a target that constrains only the SUM. The per-bond decomposition is then
unidentifiable: energy can sit on bonds that are not moving and nothing in the
loss notices. Measured on the cached OpenFF Gen3 molecules, the enumeration finds
a median of 8 central bonds carrying a proper torsion while exactly 1 is driven,
so there are eight places to put an energy and one constraint.

`--rotors driven`, the default, keeps ONLY the bond the scan turned: its torsions
are the only ones enumerated, the only dihedrals computed and the only ones the
head sees. The target then supervises that bond directly and E(ij, phi) becomes a
function of ij rather than a term in an unidentified sum. It is also cheaper,
since the other seven bonds were being computed and then added to a sum the data
could not attribute. `--rotors all` keeps the old behaviour as the arm that
measures what the restriction costs.

This is also what makes the model composable at inference: for a new molecule,
predict E(ij, phi) for each rotatable bond and add them up. Learning the sum first
and hoping the parts are right is the wrong order.

THREE ARMS, and the third is the one that matters.

  frozen               the stage 2 baseline. Encoder fixed, embeddings from one
                       reference conformer per molecule.
  finetuned_reference  encoder updated, but still shown only the reference
                       conformer.
  finetuned_per_angle  encoder updated AND shown every scan geometry.

The middle arm exists because it is the one that sounds right and cannot work.
Shown only a reference conformer, the encoder never sees a rotated geometry, so
no gradient reaches it that carries any information about how the energy varies
with the angle; it can only learn which molecules have large barriers. All of the
angle dependence sits in the phase, which the encoder does not touch. Measuring
that rather than asserting it is the point of keeping the arm.

WHAT IS HELD IDENTICAL. All three share the split, the seed, the head
initialisation, the epoch budget, the head learning rate and the stopping rule.
All three run in one process: a before and after built from two runs can differ
by the split without anyone noticing.

HOW OVERFITTING IS HELD OFF, because 3M pretrained parameters against 60 training
molecules is the regime where fine-tuning destroys a representation rather than
adapting it. Four things, each standard practice for fine-tuning on a small set:

  a validation split      carved out of TRAIN, never out of test.
                          The epoch with the best validation RMSE is the one kept,
                          and training stops after `--patience` epochs without
                          improvement. Without this the comparison would measure
                          which arm overfits fastest.
  AdamW with weight decay on the head.
  an anchor to the        an L2 penalty on how far the encoder has moved from its
  pretrained encoder      pretrained weights, at `--encoder-anchor`. This is the
                          "L2 towards the initial model" that the fine-tuning
                          literature recommends over plain weight decay, which
                          would instead pull a pretrained representation toward
                          zero.
  a low encoder rate      two orders of magnitude below the head's, on a
                          REDUCE-ON-PLATEAU schedule. AIMNet2 uses plateau and
                          converges in 400 to 500 epochs; cosine was the earlier
                          choice here and it is the wrong pairing with early
                          stopping, because cosine anneals against a fixed epoch
                          budget that the stopping rule then does not use.
  an EMA of the weights   MACE-OFF23 keeps an exponential moving average of the
                          weights and evaluates with it. It costs one extra copy
                          of the parameters and reliably smooths the noisy
                          validation curve that a small training set produces;
                          ours oscillated between 1.85 and 2.30 kcal/mol from one
                          epoch to the next, which makes "the best epoch" partly
                          a draw from that noise. `--ema 0` disables it.

THE STOPPING RULE IS PART OF THE COMPARISON. An arm that needs more epochs is not
penalised for it, because every arm gets the same budget and the same patience
and each keeps its own best epoch.

MINI-BATCHES, not full batch. An earlier version accumulated gradients over the
whole training set and stepped once per epoch. At 60 molecules that is workable;
at 888 it is 250 updates for 250 epochs, and at the 34,571 usable QCArchive
records it would be ONE parameter update every 2.4 hours, which is not training.
Molecules are shuffled each epoch and the optimiser steps every `--batch-size`
of them, so an epoch at 888 molecules with batch 32 gives 27 updates rather than
1. Gradients are still accumulated one molecule at a time INSIDE a batch, because
molecules have different atom counts and holding a whole batch's graphs with a 3M
parameter encoder in them is what reached 15.3 GiB on an 18.4 GiB card.

WHY PER-ANGLE IS AFFORDABLE. All 24 scan geometries go through the encoder in one
batched call. Measured 2026-09-11 on a 16-atom molecule: 41.6 ms for the whole
scan against 407.9 ms one at a time, agreeing to 5.4e-7. So the per-angle arm
costs 2.4 times the reference arm rather than 24 times.

CONTROLS. The random and flat-profile controls are carried through both arms
unchanged, so a fine-tuned arm that beats the frozen one still has to beat them.

UNITS. kcal/mol relative to each scan's own minimum. One row of the per-molecule
table is one molecule in one arm.

COST. Measured and written into the output. Expect minutes per arm on one card.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from nagl_torsion_loader import load as _load_nagl

_load_nagl()

from fit_torsion_head import enumerate_torsions, mmff94_profile, ranks_within


def DEVICE():
    """The device, without asking deepmd for it.

    `deepmd.pt.utils.env.DEVICE` was imported in five places, which made every
    arm of this script depend on DPA3 being installed. espaloma needs DGL and
    cannot share an environment with deepmd, so the lookup had to stop being
    encoder-specific before a second encoder could exist at all. deepmd's own
    DEVICE is still preferred when it is importable, so the DPA3 arms are
    unchanged down to which card they land on.
    """
    import torch

    try:
        from deepmd.pt.utils.env import DEVICE as _D
        return _D
    except Exception:                                    # noqa: BLE001
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_descriptor(*args, **kwargs):
    """Imported lazily: dpa3_embeddings pulls in deepmd at module scope."""
    from dpa3_embeddings import build_descriptor as _b

    return _b(*args, **kwargs)


def embed_torch(descriptor, type_map, symbols, xyz, grad: bool):
    """(n_frames, n_atoms, n_features) embeddings, keeping the graph when grad is True."""
    import torch
    from deepmd.pt.utils.env import GLOBAL_PT_FLOAT_PRECISION
    DEVICE_ = DEVICE()
    from deepmd.pt.utils.nlist import extend_input_and_build_neighbor_list

    frames = np.atleast_3d(np.asarray(xyz, dtype=float))
    if frames.shape[-1] != 3:
        frames = np.asarray(xyz, dtype=float)[None]
    n_frames = frames.shape[0]
    atype = torch.tensor(
        [[type_map.index(s) for s in symbols]] * n_frames, dtype=torch.long, device=DEVICE_
    )
    coord = torch.tensor(frames, dtype=GLOBAL_PT_FLOAT_PRECISION, device=DEVICE_)
    extended, extended_types, mapping, nlist = extend_input_and_build_neighbor_list(
        coord, atype, descriptor.get_rcut(), descriptor.get_sel(),
        mixed_types=descriptor.mixed_types(), box=None,
    )
    context = torch.enable_grad() if grad else torch.no_grad()
    with context:
        return descriptor(extended, extended_types, nlist, mapping)[0]


def scaffold_groups(prepared):
    """A Bemis-Murcko scaffold per molecule, with acyclic ones kept distinct.

    Acyclic molecules have no Murcko scaffold. Lumping them into one group would
    force them all into a single fold, which on OpenFF Gen3 is 12 percent of the
    set. Each gets its own group instead, so they spread across folds while no
    molecule's scaffold ever spans a fold boundary.
    """
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    groups = []
    for index, molecule in enumerate(prepared):
        scaffold = ""
        try:
            if Chem.MolFromSmiles(molecule["smiles"]):
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(molecule["smiles"])
        except Exception:  # noqa: BLE001
            scaffold = ""
        groups.append(scaffold if scaffold else f"__acyclic_{index}")
    return groups


def cross_bond_folds(prepared, n_folds, seed):
    """Folds over BONDS, restricted to molecules scanned at two or more rotors.

    A torsiondrive labels one bond per molecule, so the profiles this model emits
    for a molecule's OTHER bonds have never been checked against anything. The
    only place they can be checked is the 1,167 molecules that separate OpenFF
    datasets happened to scan at different rotors.

    Every held-out scan here belongs to a molecule whose other scans are in
    training, so the scaffold, the elements and one of the molecule's own rotors
    are all seen. Only that particular bond is not. Scans of molecules with a
    single rotor go into training and are never held out, because they cannot
    answer this question.
    """
    by_molecule = {}
    for index, molecule in enumerate(prepared):
        identity = molecule.get("inchi_key") or molecule["molecule_id"]
        by_molecule.setdefault(identity, []).append(index)

    eligible = []
    for identity, members in by_molecule.items():
        bonds = {tuple(prepared[i]["driven_bond"]) for i in members}
        if len(bonds) > 1:
            eligible.extend(members)
    rng = np.random.default_rng(seed)
    eligible = [eligible[i] for i in rng.permutation(len(eligible))]
    folds = [sorted(eligible[i::n_folds]) for i in range(n_folds)]
    print(f"  cross-bond: {len(eligible)} scans from "
          f"{len({(prepared[i].get('inchi_key') or prepared[i]['molecule_id']) for i in eligible})} "
          f"multi-rotor molecules are holdable; "
          f"{len(prepared) - len(eligible)} single-rotor scans stay in training")
    return folds, None


def scaffold_folds(prepared, n_folds, seed):
    """K folds over MOLECULES, grouped so no scaffold spans a fold boundary.

    Every molecule lands in exactly one fold, so across the K runs each is held
    out exactly once and the paired comparison has n = every molecule regardless
    of K. Folds are filled greedily largest-group-first into whichever fold is
    currently smallest, which keeps them close to equal in size where a random
    assignment of unequal groups would not.
    """
    groups = scaffold_groups(prepared)
    members = {}
    for index, group in enumerate(groups):
        members.setdefault(group, []).append(index)

    rng = np.random.default_rng(seed)
    order = sorted(members, key=lambda g: (-len(members[g]), g))
    # Shuffle within equal sizes so the assignment is not alphabetical.
    order = [order[i] for i in rng.permutation(len(order))]
    order.sort(key=lambda g: -len(members[g]))

    folds = [[] for _ in range(n_folds)]
    for group in order:
        target = min(range(n_folds), key=lambda f: len(folds[f]))
        folds[target].extend(members[group])
    return folds, groups


def identity_split(prepared, args):
    """Train, validation and test by a HASH OF THE MOLECULE, not its position.

    Every other splitter here permutes a list, so two runs over lists of
    DIFFERENT LENGTHS split differently at one seed. That is exactly the
    scan-against-bond pair: 15,659 scan records and 13,424 grouped ones, seed
    20260911, held out 3,915 and 3,356 molecules with nothing making them the
    same molecules. A paired comparison over rows the two arms did not share is
    not a paired comparison, and the check would only have caught it after
    thirty GPU-hours. Measured on SLURM job 79364, cancelled at nine minutes.

    Hashing the identity makes the partition a property of the molecule, so
    grouping its scans, adding a molecule, or dropping one leaves every other
    molecule where it was. The split is still over molecules and never over scan
    points.

    NOT the default: it is a random split by construction and carries none of
    astartes' scaffold extrapolation, so it is the weaker test wherever the
    scaffold sampler works. It exists for comparisons that must share rows.
    """
    import hashlib

    train, validation, test = [], [], []
    for index, molecule in enumerate(prepared):
        identity = molecule.get("inchi_key") or molecule["molecule_id"]
        digest = hashlib.blake2b(f"{args.seed}:{identity}".encode(),
                                 digest_size=8).digest()
        u = int.from_bytes(digest, "big") / float(1 << 64)
        if u < args.test_fraction:
            test.append(index)
        elif u < args.test_fraction + args.val_fraction:
            validation.append(index)
        else:
            train.append(index)
    return train, validation, test, "hashed molecule identity"


def split_molecules(prepared, args, rng):
    """Three-way split OVER MOLECULES, by Bemis-Murcko scaffold where possible.

    A random split of torsiondrives flatters a model: the same scaffold appears
    on both sides and the held-out molecules are near neighbours of training
    ones. `astartes` (Burns et al., JOSS 8(91):5996, 2023) implements the
    extrapolative splitters as a library rather than as a script every project
    rewrites, and its scaffold sampler puts whole Bemis-Murcko scaffolds on one
    side or the other.

    THE SPLIT IS NEVER OVER SCAN POINTS. 23 points of a scan in training and the
    24th held out measures interpolation inside one profile and would report it
    as generalisation.

    Falls back to a random split over molecules if astartes is unavailable or the
    sampler raises, and says which was used rather than pretending. A quiet
    fallback to random is the exact failure this function exists to prevent.
    """
    n = len(prepared)
    smiles = [m["smiles"] for m in prepared]
    if args.sampler != "random":
        try:
            from astartes.molecules import train_val_test_split_molecules

            result = train_val_test_split_molecules(
                molecules=np.array(smiles),
                train_size=1.0 - args.test_fraction - args.val_fraction,
                val_size=args.val_fraction,
                test_size=args.test_fraction,
                sampler=args.sampler,
                random_state=args.seed,
                return_indices=True,
            )
            train_idx, val_idx, test_idx = result[-3], result[-2], result[-1]
            if min(len(train_idx), len(val_idx), len(test_idx)) > 0:
                return (
                    list(map(int, train_idx)), list(map(int, val_idx)),
                    list(map(int, test_idx)), f"astartes {args.sampler}",
                )
            reason = "a split came back empty"
        except Exception as error:  # noqa: BLE001
            reason = f"{type(error).__name__}: {error}"
        print(f"  astartes {args.sampler} unavailable ({reason}); "
              f"falling back to a random split over molecules", flush=True)

    order = rng.permutation(n)
    n_test = max(1, int(round(args.test_fraction * n)))
    n_val = max(1, int(round(args.val_fraction * n)))
    return (
        list(map(int, order[n_test + n_val:])), list(map(int, order[n_test:n_test + n_val])),
        list(map(int, order[:n_test])), "random over molecules",
    )


def prepare(cache, entries, rotors="driven", encoder="dpa3"):
    """Everything that does not depend on the encoder, computed once for both arms.

    A graph encoder additionally builds each molecule's featurised graph here. It
    belongs here and not in the training loop because the graph, like the
    embedding it produces, does not depend on the conformer: building it per
    frame would be 24 times the work for the same bytes.
    """
    import torch
    from openff.nagl.utils._tensors import calculate_dihedrals
    DEVICE_ = DEVICE()

    prepared, skipped = [], []
    for entry in entries:
        key = entry["key"]
        try:
            bonds = cache[f"bonds::{key}"]
            xyz = np.asarray(cache[f"xyz::{key}"], dtype=float)
            groups = enumerate_torsions(bonds, entry["n_atoms"])
            if not groups:
                skipped.append((entry["molecule_id"], "no proper torsion"))
                continue
            keys = sorted(groups)
            driven = entry["driven_dihedral"]
            driven_bond = (min(driven[1], driven[2]), max(driven[1], driven[2]))
            if driven_bond not in keys:
                skipped.append((entry["molecule_id"],
                                f"driven bond {driven_bond} is central to no proper torsion"))
                continue
            # Only the driven bond is kept when that is what carries the energy.
            # Enumerating the rest and then summing them into a target that
            # cannot attribute them is both unidentifiable and wasted work.
            if rotors == "driven":
                keys = [driven_bond]
            driven_slot = keys.index(driven_bond)
            indices, bond_of = [], []
            for slot, bond in enumerate(keys):
                for torsion in groups[bond]:
                    indices.append(torsion)
                    bond_of.append(slot)
            indices = np.asarray(indices, dtype=int)
            deltas = np.stack(
                [
                    calculate_dihedrals(
                        f[indices[:, 0]], f[indices[:, 1]], f[indices[:, 2]], f[indices[:, 3]]
                    )
                    for f in xyz
                ]
            )
            energies = np.asarray(cache[f"energy::{key}"], dtype=float)
            # BEFORE the append, not after. Attaching it afterwards left a
            # molecule in `prepared` with no graph whenever the build raised,
            # and recorded it as skipped at the same time, so training reached
            # it and died on KeyError: 'graph' after the split had been made.
            graph = None
            if encoder in GRAPH_ENCODERS:
                graph_of = GRAPH_ENCODERS[encoder]()[1]
                graph = graph_of(entry["symbols"], bonds,
                                 cache[f"bond_orders::{key}"])
            prepared.append(
                {
                    "key": key,
                    "molecule_id": entry["molecule_id"],
                    "smiles": entry["smiles"],
                    "inchi_key": entry.get("inchi_key", ""),
                    "driven_bond": entry.get(
                        "driven_bond",
                        sorted(entry["driven_dihedral"][1:3]) if entry.get("driven_dihedral") else [],
                    ),
                    "symbols": entry["symbols"],
                    "n_atoms": entry["n_atoms"],
                    "grid": entry["grid"],
                    "reference_xyz": xyz[int(np.argmin(energies))][None],
                    "scan_xyz": xyz,
                    "indices": torch.tensor(indices, dtype=torch.long, device=DEVICE_),
                    "deltas": torch.tensor(deltas, dtype=torch.float64, device=DEVICE_),
                    "bond_of": torch.tensor(bond_of, dtype=torch.long, device=DEVICE_),
                    "n_bonds": len(keys),
                    "driven_slot": driven_slot,
                    "n_bonds_available": len(groups),
                    "qm": energies,
                    "qm_t": torch.tensor(energies, dtype=torch.float64, device=DEVICE_),
                    "mmff": mmff94_profile(
                        entry["symbols"], bonds, cache[f"bond_orders::{key}"], xyz
                    ),
                    "graph": graph,
                }
            )
        except Exception as error:  # noqa: BLE001
            skipped.append((entry["molecule_id"], f"{type(error).__name__}: {error}"))
    return prepared, skipped


def _espaloma_parts():
    from espaloma_encoder import build_encoder, graph_of, embed_graph

    return build_encoder, graph_of, embed_graph


def _nagl_parts():
    from nagl_encoder import build_encoder, graph_of, embed_graph

    return build_encoder, graph_of, embed_graph


# Encoders whose embedding is a function of the molecular GRAPH and not of the
# conformer. Everything in here pays one forward pass per molecule where DPA3
# pays one per conformer, and gives a k_n that cannot depend on which geometry
# happened to be in hand.
GRAPH_ENCODERS = {"espaloma": _espaloma_parts, "nagl": _nagl_parts}


def group_scans(prepared, quarantine):
    """One record per MOLECULE, carrying every bond that molecule was scanned at.

    WHY THIS EXISTS. `h_bond` and `s` are both scatter_add by bond, so the
    energy of bond b is built only from the torsions whose central bond it is:
    enumerating a molecule's other rotors leaves the driven column bit-identical
    and changes no gradient. The loss is therefore ALREADY per bond and not per
    molecule. What is not per bond is the geometry. Each scan embeds its own
    reference conformer, so one molecule scanned at bond A and at bond B yields
    two different h, hence two different representations OF THE SAME BOND, and
    nothing in the loss says they should agree. k_n is supposed to be a property
    of the chemistry (model.py, `bond_representation`); this makes the training
    say so, by giving every scanned bond of a molecule one shared embedding.

    WHAT IS NOT GROUPED. Two QCArchive entries carrying one InChIKey need not
    agree on atom order. Grouping those would evaluate bond A's dihedrals on
    bond B's atoms and nothing downstream would raise, so a group is formed only
    where the symbol lists AND the enumerated torsion sets match exactly. The
    rest stay as single-scan records and are counted in the report.

    Returns records shaped like `prepare`'s, with one extra key `supervised`: a
    list of {key, slot, deltas, qm, qm_t, grid, driven_bond, mmff, scan_xyz}.
    """
    import torch

    groups = {}
    for scan in prepared:
        identity = scan.get("inchi_key") or scan["molecule_id"]
        groups.setdefault(identity, []).append(scan)

    def supervised_of(scan):
        return {
            "key": scan["key"], "slot": scan["driven_slot"],
            "deltas": scan["deltas"], "qm": scan["qm"], "qm_t": scan["qm_t"],
            "grid": scan["grid"], "driven_bond": scan["driven_bond"],
            "mmff": scan["mmff"], "scan_xyz": scan["scan_xyz"],
            "molecule_id": scan["molecule_id"],
        }

    def compatible(a, b):
        return (list(a["symbols"]) == list(b["symbols"])
                and a["indices"].shape == b["indices"].shape
                and bool(torch.equal(a["indices"], b["indices"]))
                and bool(torch.equal(a["bond_of"], b["bond_of"])))

    records, ungrouped = [], 0
    for identity, members in groups.items():
        first, rest = members[0], []
        for other in members[1:]:
            if compatible(first, other):
                rest.append(other)
            else:
                ungrouped += 1
                quarantine.append((other["molecule_id"],
                                   f"shares InChIKey {identity} but not its atom "
                                   f"order or torsion set; kept as its own record"))
                records.append(dict(other, supervised=[supervised_of(other)]))
        family = [first, *rest]
        # The shared geometry is the lowest-energy frame any of its scans found.
        # Any single scan's minimum is a minimum along one rotor only.
        best = min(family, key=lambda m: float(np.min(m["qm"])))
        records.append(dict(
            first,
            reference_xyz=best["reference_xyz"],
            supervised=[supervised_of(m) for m in family],
        ))

    sizes = np.asarray([len(r["supervised"]) for r in records])
    multi = int((sizes > 1).sum())
    print(f"  per-bond supervision: {len(prepared)} scans -> {len(records)} "
          f"molecule records, {multi} of them carrying {int(sizes[sizes>1].sum())} "
          f"bonds ({sizes.max()} at most); {ungrouped} scans could not be "
          f"grouped with their InChIKey twin")
    return records


def supervised_folds(records, n_folds, seed):
    """Folds over SUPERVISED BONDS, so a molecule can be split across them.

    The cross-bond question is whether a bond the model never saw is predicted
    as well as one it did. Under per-bond supervision a molecule is one record,
    so holding out whole records would hold out every bond of it at once and ask
    a different question. This holds out individual supervised entries of
    multi-bond records instead; single-bond records always train.
    """
    eligible = [(i, j) for i, r in enumerate(records)
                if len(r["supervised"]) > 1
                for j in range(len(r["supervised"]))]
    rng = np.random.default_rng(seed)
    eligible = [eligible[i] for i in rng.permutation(len(eligible))]
    folds = [sorted(eligible[i::n_folds]) for i in range(n_folds)]
    print(f"  cross-bond, per-bond supervision: {len(eligible)} holdable bonds "
          f"across {len({i for i, _ in eligible})} multi-rotor molecules")
    return folds


def build_head(n_features, args):
    """The torsion head, with every width and periodicity taken from arguments.

    Defaults reproduce the library's, so an unsearched run is unchanged.
    """
    from openff.nagl.torsion import TorsionModel

    widths = lambda s: tuple(int(x) for x in str(s).split(",") if x.strip())
    model = TorsionModel(
        n_atom_features=n_features,
        periodicities=tuple(range(1, args.periodicities + 1)),
        torsion_hidden=widths(args.torsion_hidden),
        nonlinear=args.nonlinear_head,
    ).double()
    # The coefficient and energy networks are built inside TorsionModel with
    # fixed widths; rebuild them here so the search can reach them too.
    from openff.nagl.torsion import TorsionCoefficients, TorsionEnergyHead

    model.coefficients = TorsionCoefficients(
        n_features, hidden=widths(args.coefficient_hidden)
    ).double()
    model.energy = TorsionEnergyHead(
        model.pooling.n_outputs,
        periodicities=tuple(range(1, args.periodicities + 1)),
        hidden=widths(args.energy_hidden),
        nonlinear=args.nonlinear_head,
    ).double()
    return model


def _reduce(per_bond, molecule, rotors):
    """(n_points, n_bonds) -> (n_points,). The driven bond alone, or every bond.

    `driven` is the default because it is the only reduction the data identifies:
    a torsiondrive turns one bond, so summing over all of them fits a target that
    constrains only the sum and leaves the per-bond function free.
    """
    if rotors == "driven":
        return per_bond[..., molecule["driven_slot"]]
    return per_bond.sum(-1)


ARMS = (
    ("frozen", False, "reference"),
    ("finetuned_reference", True, "reference"),
    ("finetuned_per_angle", True, "per_angle"),
    # THE CONTROL THAT SEPARATES THE TWO CHANGES. `finetuned_per_angle` updates
    # the encoder AND shows it every conformer, so its advantage over `frozen`
    # cannot be attributed to either on its own. This arm holds the weights fixed
    # and still shows every conformer, so the difference between it and
    # `finetuned_per_angle` is fine-tuning alone, and the difference between it
    # and `frozen` is the extra geometry alone. Added 2026-09-11 after the first
    # three arms returned a 2.7 percent advantage that was uninterpretable.
    ("frozen_per_angle", False, "per_angle"),
)


def run_arm(name, finetune, geometry, held_out, train, validation, args, type_map,
            initial_head_state, history_path):
    """Train one arm and return its predictions, its cost and its chosen epoch."""
    import torch
    from openff.nagl.torsion import TorsionModel, phase_vectors
    DEVICE_ = DEVICE()

    # NOT `name`: that is the arm label this function was called with, and
    # shadowing it wrote "espaloma" into every row of the training-curve CSV
    # where "frozen" or "finetuned_reference" belonged.
    encoder_name = getattr(args, "encoder", "dpa3")
    espaloma = encoder_name in GRAPH_ENCODERS
    if espaloma:
        build_encoder, _graph_of, embed_graph = GRAPH_ENCODERS[encoder_name]()

        descriptor, n_features = build_encoder(args.espaloma_version)
        descriptor = descriptor.to(DEVICE_)
        # The DGL graphs are built on CPU by `esp.Graph` and their node features
        # go through CUDA weights, so they have to be moved too. Once, here,
        # rather than inside the forward: a graph move is cheap but 604
        # molecules by 400 epochs is not, and the graph does not change.
        for molecule in list(train) + list(validation) + list(held_out):
            if molecule.get("graph") is not None and molecule["graph"].device != DEVICE_:
                molecule["graph"] = molecule["graph"].to(DEVICE_)
    else:
        descriptor, _ = build_descriptor(args.config, args.checkpoint, args.head)
        n_features = descriptor.get_dim_out()
    descriptor.train(finetune)
    for parameter in descriptor.parameters():
        parameter.requires_grad_(finetune)

    def embed(molecule, frames_key, grad):
        """(n_frames, n_atoms, n_features), whichever encoder is behind it.

        The espaloma branch ignores the coordinates, which is the whole point:
        its embedding is a function of the molecular graph, so every frame of a
        scan shares one set of coefficients by construction rather than by
        a training signal that has to discover it.
        """
        if espaloma:
            n = 1 if frames_key == "reference_xyz" else len(molecule["deltas"])
            return embed_graph(descriptor, molecule["graph"], n, grad)
        return embed_torch(descriptor, type_map, molecule["symbols"],
                           molecule[frames_key], grad)

    head = build_head(n_features, args).to(DEVICE_)
    head.load_state_dict(initial_head_state)

    # The pretrained weights are kept so the encoder can be anchored to them.
    # Detached clones, so they are a fixed reference and not a second set of
    # parameters that drifts along with the first.
    anchor = (
        {n: p.detach().clone() for n, p in descriptor.named_parameters()}
        if finetune and args.encoder_anchor > 0
        else None
    )

    groups = [
        {"params": head.parameters(), "lr": args.head_lr, "weight_decay": args.weight_decay}
    ]
    if finetune:
        # No weight decay on the encoder: decay pulls toward zero, which is not
        # where a pretrained representation should be pulled. The anchor term
        # below pulls toward the pretrained weights instead.
        groups.append(
            {"params": descriptor.parameters(), "lr": args.encoder_lr, "weight_decay": 0.0}
        )
    optimiser = torch.optim.AdamW(groups)
    # Plateau rather than cosine: cosine anneals against a fixed epoch budget,
    # and early stopping means the budget is not the thing that ends training.
    schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=max(args.patience // 4, 5),
        min_lr=1e-7,
    )

    def profile(molecule, grad):
        """The whole scan, relative to its own minimum.

        In the per-angle arm every scan point has its own embeddings, so the 24
        points are laid out as one flat graph with each frame's atom indices
        offset by n_atoms. That is ordinary graph batching, and it means the
        batched, already tested library path is used unchanged rather than a
        second implementation written here.
        """
        n_points = len(molecule["deltas"])
        if geometry == "reference":
            h = embed(molecule, "reference_xyz", grad)[0].double()
            h_bond, c = head.bond_representation(
                h, molecule["indices"], molecule["bond_of"], molecule["n_bonds"]
            )
            s = phase_vectors(
                molecule["deltas"], c, molecule["bond_of"], molecule["n_bonds"],
                periodicities=head.periodicities,
            )
            energies = _reduce(head.energy(h_bond, s), molecule, args.rotors)
        else:
            h = embed(molecule, "scan_xyz", grad).double()
            n_atoms = h.shape[1]
            frame = torch.arange(n_points, device=h.device)
            indices = (
                molecule["indices"][None] + frame[:, None, None] * n_atoms
            ).reshape(-1, 4)
            bond_of = (
                molecule["bond_of"][None] + frame[:, None] * molecule["n_bonds"]
            ).reshape(-1)
            h_bond, c = head.bond_representation(
                h.reshape(-1, h.shape[-1]), indices, bond_of,
                n_points * molecule["n_bonds"],
            )
            s = phase_vectors(
                molecule["deltas"].reshape(-1), c, bond_of,
                n_points * molecule["n_bonds"], periodicities=head.periodicities,
            )
            energies = _reduce(
                head.energy(h_bond, s).reshape(n_points, molecule["n_bonds"]),
                molecule, args.rotors,
            )
        return energies - energies.min()

    def profile_record(record, grad):
        """Every scanned bond of one molecule, from ONE embedding of ONE geometry.

        This is the whole of what per-bond supervision changes. The encoder runs
        once and `bond_representation` runs once, because neither depends on the
        conformer; each scanned bond then takes its own column of the energy
        table with its own phase. A molecule scanned at three rotors costs one
        encoder pass and contributes three terms to the loss, where the
        scan-level path costs three passes and lets each bond be represented
        from a geometry chosen to suit it.
        """
        h = embed(record, "reference_xyz", grad)[0].double()
        h_bond, c = head.bond_representation(
            h, record["indices"], record["bond_of"], record["n_bonds"]
        )
        out = []
        for entry in record["supervised"]:
            s = phase_vectors(
                entry["deltas"], c, record["bond_of"], record["n_bonds"],
                periodicities=head.periodicities,
            )
            energies = head.energy(h_bond, s)[..., entry["slot"]]
            out.append(energies - energies.min())
        return out

    per_bond = getattr(args, "supervision", "scan") == "bond"

    # Gradients are accumulated ONE MOLECULE AT A TIME rather than stacking every
    # loss and calling backward once. The arithmetic is identical, since the
    # gradient of a mean is the mean of the gradients, but the graph of each
    # molecule is freed as soon as it has contributed. Holding all 75 at once,
    # with a 3M-parameter encoder and 24 frames each in the graph, reached 15.3
    # GiB on an 18.4 GiB card and stopped.
    # Exponential moving average of the weights, evaluated instead of the raw
    # ones. MACE-OFF23 does this; on a small training set the epoch-to-epoch
    # validation noise is large enough that picking "the best epoch" from the raw
    # weights is partly picking a lucky draw.
    ema = None
    if args.ema:
        ema = torch.optim.swa_utils.AveragedModel(
            torch.nn.ModuleList([head, descriptor]),
            avg_fn=lambda averaged, current, n: args.ema * averaged
            + (1.0 - args.ema) * current,
        )

    def _swap_in_ema():
        """Copy the EMA weights into the live modules, returning the originals."""
        if ema is None:
            return None
        live = torch.nn.ModuleList([head, descriptor])
        saved = {k: v.detach().clone() for k, v in live.state_dict().items()}
        live.load_state_dict(
            {k: v for k, v in ema.module.state_dict().items()}, strict=False
        )
        return saved

    def _swap_out_ema(saved):
        if saved is not None:
            torch.nn.ModuleList([head, descriptor]).load_state_dict(saved, strict=False)

    def validation_rmse():
        saved = _swap_in_ema()
        head.eval()
        descriptor.eval()
        with torch.no_grad():
            if per_bond:
                # The mean is over BONDS, not molecules, so a molecule scanned at
                # three rotors counts three times here exactly as it does in the
                # training loss. Weighting it once would make the validation
                # metric disagree with the thing being minimised.
                values = [
                    float(torch.sqrt(((predicted - entry["qm_t"]) ** 2).mean()))
                    for m in validation
                    for predicted, entry in zip(profile_record(m, False),
                                                m["supervised"])
                ]
            else:
                values = [
                    float(torch.sqrt(((profile(m, False) - m["qm_t"]) ** 2).mean()))
                    for m in validation
                ]
        _swap_out_ema(saved)
        head.train()
        descriptor.train(finetune)
        return float(np.mean(values))

    started = time.time()
    parameters = [p for g in groups for p in g["params"]]
    best = {"rmse": float("inf"), "epoch": -1, "head": None, "encoder": None}
    since_improved = 0
    # Batch size scaled with the training set rather than fixed. AIMNet2 uses an
    # effective batch of 2048 on 20M conformers, which is about 0.01 percent of
    # the set; NAGL uses 1000 molecules. A batch that is a fixed 32 is 3.6
    # percent of 888 molecules and 0.09 percent of 34,571, which are different
    # optimisation problems. Clamped to the 32 to 512 range the literature sits in.
    batch_size = args.batch_size or int(np.clip(round(0.04 * len(train)), 8, 512))
    order_rng = np.random.default_rng(args.seed)

    for epoch in range(args.epochs):
        # Shuffled every epoch, so the batches are not the same partition each
        # time. Without this, mini-batching just fits a fixed set of subsets.
        order = order_rng.permutation(len(train))
        total, n_batches = 0.0, 0
        for start in range(0, len(order), batch_size):
            batch = [train[i] for i in order[start:start + batch_size]]
            optimiser.zero_grad(set_to_none=True)
            # THE DIVISOR IS THE NUMBER OF BONDS IN THE BATCH, not molecules.
            # Dividing by molecules would weight a molecule scanned at one rotor
            # the same as one scanned at four, which is the "one molecule, one
            # number" the per-bond mode exists to remove.
            n_terms = (sum(len(m["supervised"]) for m in batch) if per_bond
                       else len(batch))
            for molecule in batch:
                if per_bond:
                    predicted = profile_record(molecule, finetune)
                    loss = sum(
                        ((p - entry["qm_t"]) ** 2).mean()
                        for p, entry in zip(predicted, molecule["supervised"])
                    ) / n_terms
                else:
                    loss = ((profile(molecule, finetune) - molecule["qm_t"]) ** 2).mean() / n_terms
                loss.backward()
                total += float(loss.detach())
            if anchor is not None:
                # Scaled by the batch fraction so the anchor's weight per EPOCH
                # does not depend on how many batches an epoch happens to have.
                penalty = args.encoder_anchor * (len(batch) / len(train)) * sum(
                    ((p - anchor[n]) ** 2).sum() for n, p in descriptor.named_parameters()
                )
                penalty.backward()
            torch.nn.utils.clip_grad_norm_(parameters, args.clip)
            optimiser.step()
            if ema is not None:
                ema.update_parameters(torch.nn.ModuleList([head, descriptor]))
            n_batches += 1
        total /= max(n_batches, 1)
        current = validation_rmse()
        schedule.step(current)
        if current < best["rmse"] - 1e-6:
            best = {
                "rmse": current, "epoch": epoch,
                "head": {k: v.detach().clone() for k, v in head.state_dict().items()},
                "encoder": (
                    {k: v.detach().clone() for k, v in descriptor.state_dict().items()}
                    if finetune else None
                ),
            }
            since_improved = 0
        else:
            since_improved += 1

        # Written every epoch and flushed, so the curves can be drawn while the
        # run is still going rather than only if it finishes.
        with open(history_path, "a") as handle:
            arm, _, fold = name.partition("|")
            handle.write(f"{arm},{fold or 'fold0'},{epoch},{total:.6f},{current:.6f},"
                         f"{best['rmse']:.6f},{best['epoch']},{time.time() - started:.1f}\n")

        # The curves are redrawn in-process, so matplotlib is imported once for
        # the whole run rather than per figure. Measured cost is about 0.35 s a
        # redraw against a cheapest-arm epoch of 1.7 s, so at the default of
        # every 25 epochs the overhead is under 1 percent even for the fastest
        # arm and is invisible for the per-angle one at 112 s an epoch.
        if args.figure_every and args.live_figure and (
            epoch % args.figure_every == 0 or epoch == args.epochs - 1
        ):
            try:
                import pandas as pd

                from compare_arms import curves as _curves

                _curves(pd.read_csv(history_path), args.live_figure)
            except Exception as error:  # noqa: BLE001
                # A figure must never end a run. Say so and carry on.
                print(f"  [{name}] live figure skipped: {type(error).__name__}: {error}",
                      flush=True)

        if epoch % 25 == 0 or epoch == args.epochs - 1:
            print(f"  [{name}] epoch {epoch:4d}  train MSE {total:.4f}  "
                  f"val RMSE {current:.4f}  best {best['rmse']:.4f} @ {best['epoch']}",
                  flush=True)
        if since_improved >= args.patience:
            print(f"  [{name}] stopped at epoch {epoch}, "
                  f"{args.patience} epochs without improvement", flush=True)
            break
    seconds = time.time() - started

    # THE KEPT MODEL IS THE BEST VALIDATION EPOCH, not the last one.
    head.load_state_dict(best["head"])
    if best["encoder"] is not None:
        descriptor.load_state_dict(best["encoder"])
    head.eval()
    descriptor.eval()
    predictions = {}
    with torch.no_grad():
        for molecule in held_out:
            if per_bond:
                # Keyed by the SCAN, not the molecule, so every table downstream
                # keeps one row per (molecule, driven bond) and nothing has to
                # learn about the grouping.
                for predicted, entry in zip(profile_record(molecule, False),
                                            molecule["supervised"]):
                    predictions[entry["key"]] = predicted.cpu().numpy()
            else:
                predictions[molecule["key"]] = profile(molecule, False).cpu().numpy()
    return predictions, seconds, best["epoch"], best["rmse"]


def fold_sets(prepared, test_idx, args, per_bond, fold_index):
    """(train, validation, held_out) for one fold, in either supervision mode.

    Under per-bond supervision with a cross-bond split the unit held out is a
    BOND, not a molecule, so a multi-rotor molecule appears on both sides with
    disjoint `supervised` lists: the bonds it keeps are trained on, the bond it
    loses is predicted. Holding the whole molecule out instead would remove
    every rotor of it at once and ask a different question, which is the one the
    scaffold split already asks.
    """
    entry_level = per_bond and args.split_mode == "cross_bond" and args.folds > 1

    if entry_level:
        held = {}
        for record_index, entry_index in test_idx:
            held.setdefault(record_index, set()).add(entry_index)
        held_out = [
            dict(prepared[i], supervised=[prepared[i]["supervised"][j]
                                          for j in sorted(js)])
            for i, js in sorted(held.items())
        ]
        pool = []
        for i, record in enumerate(prepared):
            keep = [e for j, e in enumerate(record["supervised"])
                    if j not in held.get(i, set())]
            if keep:
                pool.append(dict(record, supervised=keep))
        order = np.random.default_rng(args.seed + fold_index).permutation(len(pool))
        n_val = max(1, int(round(args.val_fraction * len(pool))))
        validation = [pool[int(i)] for i in order[:n_val]]
        train = [pool[int(i)] for i in order[n_val:]]
        return train, validation, held_out

    held_out = [prepared[i] for i in test_idx]
    remaining = [i for i in range(len(prepared)) if i not in set(test_idx)]
    order = np.random.default_rng(args.seed + fold_index).permutation(remaining)
    n_val = max(1, int(round(args.val_fraction * len(order))))
    return ([prepared[int(i)] for i in order[n_val:]],
            [prepared[int(i)] for i in order[:n_val]], held_out)


def main():
    import torch
    from openff.nagl.torsion import TorsionModel
    DEVICE_ = DEVICE()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torsiondrives", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--config", default="")
    parser.add_argument("--head", default="SPICE2")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--head-lr", type=float, default=0.003)
    parser.add_argument("--encoder-lr", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=0,
                        help="molecules per optimiser step; 0 scales it with the training set")
    parser.add_argument("--clip", type=float, default=5.0)
    # Architecture, previously fixed in the library defaults. Exposed so a
    # hyperparameter search can reach them: the accuracy result should not be
    # blamed on an architecture nobody tuned.
    parser.add_argument("--torsion-hidden", default="128,128",
                        help="Janossy pooling widths, comma separated")
    parser.add_argument("--coefficient-hidden", default="64",
                        help="widths of the c_ij network")
    parser.add_argument("--energy-hidden", default="128",
                        help="widths of the network predicting k_n from h_bond")
    parser.add_argument("--periodicities", type=int, default=6,
                        help="n = 1..this. 6 is espaloma's choice")
    parser.add_argument("--nonlinear-head", action="store_true",
                        help="NN([h_r : s]) instead of sum_n k_n . s_n; gives up the "
                             "Fourier reading in exchange for a more flexible function")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--encoder-anchor", type=float, default=1.0,
                        help="L2 penalty toward the pretrained encoder weights")
    parser.add_argument("--ema", type=float, default=0.999,
                        help="decay for the weight EMA used at evaluation; 0 disables")
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument(
        "--encoder", choices=("dpa3", "espaloma", "nagl"), default="dpa3",
        help="which representation the head sits on. 'dpa3' is a pretrained "
             "atomistic potential's descriptor and depends on the conformer; "
             "'espaloma' is a graph network and does not, so k_n becomes a "
             "function of the chemical graph by construction.")
    parser.add_argument(
        "--espaloma-version", default="0.3.2",
        help="the graph encoder's release. '0.3.2' for espaloma; for nagl, a "
             "released model name such as openff-gnn-am1bcc-1.0.0.pt")
    parser.add_argument(
        "--split-by-identity", action="store_true",
        help="split on a hash of each molecule's InChIKey rather than on its "
             "position in the list, so two runs over lists of different lengths "
             "hold out the SAME molecules. Required for any paired comparison "
             "whose arms prepare different numbers of records.")
    parser.add_argument(
        "--supervision", choices=("scan", "bond"), default="scan",
        help="'scan' trains one record per torsiondrive, each embedding its own "
             "reference geometry. 'bond' groups a molecule's scans into one "
             "record, embeds ONE geometry, and supervises every bond that "
             "molecule was scanned at off that single representation, with the "
             "batch loss divided by bonds rather than molecules.")
    parser.add_argument("--rotors", choices=("driven", "all"), default="driven",
                        help="which central bonds carry the energy")
    parser.add_argument("--fold-index", type=int, default=-1,
                        help="run only this fold; -1 runs them all. For a SLURM array, "
                             "where one task is one (fold, arm) so 15 combinations fill 15 cards.")
    parser.add_argument("--arms", default="",
                        help="comma-separated arm names; empty runs all of them")
    parser.add_argument("--split-mode", choices=("molecule", "cross_bond"), default="molecule",
                        help="molecule: hold out whole molecules, which tests transfer to unseen "
                             "chemistry. cross_bond: hold out ONE BOND of a molecule whose OTHER "
                             "bonds are in training, which tests whether k_n is a function of the "
                             "bond or whether the model emits one profile per molecule.")
    parser.add_argument("--folds", type=int, default=1,
                        help="K-fold over molecules, grouped by scaffold; 1 means a single split")
    parser.add_argument("--sampler", default="scaffold",
                        choices=("scaffold", "kmeans", "sphere_exclusion", "random"),
                        help="astartes sampler. scaffold is the extrapolative default")
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--profiles", required=True)
    parser.add_argument("--per-molecule", required=True)
    parser.add_argument("--quarantine", required=True)
    parser.add_argument("--history", required=True,
                        help="per-epoch train and validation curves, written as it goes")
    parser.add_argument("--figure-every", type=int, default=25,
                        help="redraw the live curves every N epochs; 0 disables")
    parser.add_argument("--live-figure", default=None,
                        help="path for the live training curves")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    cache = np.load(args.torsiondrives, allow_pickle=False)
    entries = json.loads(str(cache["meta"]))
    per_bond = args.supervision == "bond"
    if per_bond and args.rotors != "all":
        # Every scanned bond of one molecule has to be a distinct column of one
        # energy table, and `rotors=driven` enumerates only the driven bond, so
        # two scans of one molecule would carry incompatible torsion sets and
        # could never be grouped. Forced rather than refused, and said out loud.
        print(f"  --supervision bond needs every rotor enumerated; "
              f"--rotors {args.rotors!r} overridden to 'all'")
        args.rotors = "all"
    prepared, skipped = prepare(cache, entries, rotors=args.rotors,
                                encoder=args.encoder)
    if not prepared:
        raise SystemExit("nothing prepared")
    if per_bond:
        prepared = group_scans(prepared, skipped)

    if per_bond and args.folds > 1 and args.split_mode == "cross_bond":
        folds = supervised_folds(prepared, args.folds, args.seed)
        split_method = f"{args.folds}-fold, cross-bond over bonds"
        print(f"split ({split_method}): fold sizes {[len(f) for f in folds]}")
    elif args.folds > 1 and args.split_mode == "cross_bond":
        folds, _ = cross_bond_folds(prepared, args.folds, args.seed)
        split_method = f"{args.folds}-fold, cross-bond"
        print(f"split ({split_method}): fold sizes {[len(f) for f in folds]}")
    elif args.folds > 1:
        folds, _ = scaffold_folds(prepared, args.folds, args.seed)
        split_method = f"{args.folds}-fold, scaffold-grouped"
        print(f"split ({split_method}): fold sizes {[len(f) for f in folds]}, "
              f"{len(prepared)} molecules")
    else:
        if args.split_by_identity:
            train_idx, val_idx, test_idx, how = identity_split(prepared, args)
        else:
            train_idx, val_idx, test_idx, how = split_molecules(prepared, args, rng)
        folds = [test_idx]
        split_method = how
        print(f"split ({how}): {len(train_idx)} train, {len(val_idx)} validation, "
              f"{len(test_idx)} test molecules, of {len(prepared)}")

    if args.encoder in GRAPH_ENCODERS:
        build_encoder = GRAPH_ENCODERS[args.encoder]()[0]

        _probe, n_features = build_encoder(args.espaloma_version)
        type_map = None
        del _probe
    else:
        reference_descriptor, type_map = build_descriptor(
            args.config, args.checkpoint, args.head)
        n_features = reference_descriptor.get_dim_out()
        del reference_descriptor
    print(f"encoder {args.encoder}: {n_features} features per atom")

    history_path = pathlib.Path(args.history)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(
        "encoder,fold,epoch,train_mse,val_rmse,best_val_rmse,best_epoch,seconds\n"
    )

    selected_arms = [
        a for a in ARMS
        if not args.arms or a[0] in {s.strip() for s in args.arms.split(",")}
    ]
    if not selected_arms:
        raise SystemExit(f"--arms {args.arms!r} matched none of {[a[0] for a in ARMS]}")
    selected_folds = (
        list(enumerate(folds)) if args.fold_index < 0
        else [(args.fold_index, folds[args.fold_index])]
    )
    print(f"running {len(selected_arms)} arm(s) x {len(selected_folds)} fold(s): "
          f"{[a[0] for a in selected_arms]}, folds {[i for i, _ in selected_folds]}")

    results, timings, chosen, fold_of = {}, {}, {}, {}
    for fold_index, test_idx in selected_folds:
        if args.folds > 1:
            # Validation comes out of THIS fold's training portion, never out of
            # its test portion, so the epoch choice never touches the held-out set.
            train, validation, held_out = fold_sets(
                prepared, test_idx, args, per_bond, fold_index)
        else:
            train = [prepared[i] for i in train_idx]
            validation = [prepared[i] for i in val_idx]
            held_out = [prepared[i] for i in test_idx]
        for molecule in held_out:
            for key in ([e["key"] for e in molecule["supervised"]] if per_bond
                        else [molecule["key"]]):
                fold_of[key] = fold_index

        torch.manual_seed(args.seed)
        initial_head_state = build_head(n_features, args).to(DEVICE_).state_dict()
        for name, finetune, geometry in selected_arms:
            tag = f"{name}|fold{fold_index}"
            torch.manual_seed(args.seed)
            predictions, seconds, epoch, val = run_arm(
                tag, finetune, geometry, held_out, train,
                validation, args, type_map, initial_head_state, history_path,
            )
            results.setdefault(name, {}).update(predictions)
            timings[name] = timings.get(name, 0.0) + seconds
            chosen[name] = (epoch, val)
            print(f"  [{tag}] {seconds:.1f} s, kept epoch {epoch} at "
                  f"validation RMSE {val:.4f} kcal/mol", flush=True)

    import pandas as pd
    from scipy.stats import spearmanr

    profile_rows, molecule_rows = [], []
    ran_keys = set()
    for arm_results in results.values():
        ran_keys.update(arm_results)

    # ONE REPORTED ROW IS ONE (MOLECULE, DRIVEN BOND) SCAN in both modes. Under
    # per-bond supervision `prepared` holds molecule records, so they are taken
    # apart again here rather than letting a multi-rotor molecule report the
    # first of its bonds and drop the rest.
    if per_bond:
        reportable = [
            dict(record, key=entry["key"], molecule_id=entry["molecule_id"],
                 qm=entry["qm"], grid=entry["grid"], mmff=entry["mmff"],
                 driven_bond=entry["driven_bond"])
            for record in prepared for entry in record["supervised"]
        ]
    else:
        reportable = prepared

    for molecule in reportable:
        if molecule["key"] not in ran_keys:
            continue          # not held out by any fold this invocation ran
        split = "test"        # in k-fold every molecule is held out exactly once
        fold = fold_of.get(molecule["key"], 0)
        qm = molecule["qm"]
        mmff = molecule["mmff"] if molecule["mmff"] is not None else np.full_like(qm, np.nan)
        for arm, _, _ in selected_arms:
            if molecule["key"] not in results.get(arm, {}):
                continue
            predicted = results[arm][molecule["key"]]
            for point, angle in enumerate(molecule["grid"]):
                profile_rows.append(
                    {
                        "molecule_id": molecule["molecule_id"], "encoder": arm,
                        "split": split, "fold": fold, "grid_degrees": angle,
                        "qm_kcal_mol": qm[point], "predicted_kcal_mol": predicted[point],
                        "qm_rank": ranks_within(qm)[point],
                        "predicted_rank": ranks_within(predicted)[point],
                    }
                )
            molecule_rows.append(
                {
                    "molecule_id": molecule["molecule_id"], "encoder": arm, "split": split,
                    "fold": fold,
                    "inchi_key": molecule.get("inchi_key", ""),
                    "driven_bond": "-".join(str(b) for b in molecule.get("driven_bond", [])),
                    "smiles": molecule["smiles"], "n_atoms": molecule["n_atoms"],
                    "n_rotors": molecule["n_bonds"],
                    "qm_barrier_kcal_mol": float(qm.max()),
                    "predicted_barrier_kcal_mol": float(predicted.max()),
                    "rmse": float(np.sqrt(np.mean((predicted - qm) ** 2))),
                    "rmse_mmff94": float(np.sqrt(np.mean((mmff - qm) ** 2)))
                    if np.isfinite(mmff).all() else np.nan,
                    "rmse_flat": float(np.sqrt(np.mean(qm**2))),
                    "rho": float(spearmanr(predicted, qm).statistic)
                    if np.ptp(predicted) > 0 else np.nan,
                    "rho_mmff94": float(spearmanr(mmff, qm).statistic)
                    if np.isfinite(mmff).all() and np.ptp(mmff) > 0 else np.nan,
                    "train_seconds": timings[arm],
                    "split_method": split_method,
                    "supervision": args.supervision,
                    "rotors": args.rotors,
                    "n_rotors_used": molecule["n_bonds"],
                    "n_rotors_available": molecule["n_bonds_available"],
                    "kept_epoch": chosen[arm][0],
                    "validation_rmse": chosen[arm][1],
                }
            )

    for path, frame in (
        (args.profiles, pd.DataFrame(profile_rows)),
        (args.per_molecule, pd.DataFrame(molecule_rows)),
    ):
        target = pathlib.Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(target, index=False)

    quarantine = pathlib.Path(args.quarantine)
    quarantine.parent.mkdir(parents=True, exist_ok=True)
    with quarantine.open("w") as handle:
        handle.write("molecule_id,reason\n")
        for name, reason in skipped:
            handle.write(f'"{name}","{str(reason)[:200]}"\n')

    frame = pd.DataFrame(molecule_rows)
    print(f"\n{len(prepared)} molecules, {split_method}, {len(skipped)} quarantined. "
          f"Every molecule is held out exactly once per arm.")
    for arm, _, _ in selected_arms:
        subset = frame[frame.encoder == arm]
        if subset.empty:
            continue
        print(f"  {arm:20s} held out  RMSE mean {subset.rmse.mean():6.3f}  "
              f"median {subset.rmse.median():6.3f} kcal/mol  "
              f"rho median {subset.rho.median():+.3f}  n={len(subset)}  "
              f"({subset.train_seconds.iloc[0]/60:.1f} min total)")
    mm = frame[frame.encoder == selected_arms[0][0]]
    if "rmse_mmff94" in mm and mm.rmse_mmff94.notna().any():
        print(f"  {'MMFF94':20s} held out  RMSE mean {mm.rmse_mmff94.mean():6.3f}  "
              f"median {mm.rmse_mmff94.median():6.3f} kcal/mol  n={mm.rmse_mmff94.notna().sum()}")
    print(f"  {'flat profile':20s} held out  RMSE mean {mm.rmse_flat.mean():6.3f}  "
          f"median {mm.rmse_flat.median():6.3f} kcal/mol  n={len(mm)}")


if __name__ == "__main__":
    main()
