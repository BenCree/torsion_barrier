"""Tests for per-bond supervision, written against known answers.

Each one is two-sided where it can be: a case that must group and a case that
must not, a fold that must hold a bond out and a molecule that must never be
eligible. The last test is the one that decides whether the mode is worth
running at all, and it is a claim about the architecture rather than about this
file: enumerating a molecule's other rotors must leave the driven bond's energy
unchanged, because both `h_bond` and `s` accumulate by scatter_add over the
torsions of one bond. If that ever stops holding, `--rotors driven` and
`--supervision bond` stop being comparable and every arm in RESULT_stage3.md
would need rerunning.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from finetune_torsion import fold_sets, group_scans, supervised_folds  # noqa: E402


def scan(inchi_key, key, slot, symbols=("C", "C", "C", "C"),
         indices=((0, 1, 2, 3),), bond_of=(0,), n_bonds=1, minimum=0.0):
    """A `prepare`-shaped record with only what the grouping reads."""
    points = 4
    return {
        "key": key,
        "molecule_id": f"{inchi_key}-{key}",
        "inchi_key": inchi_key,
        "smiles": "CCCC",
        "symbols": list(symbols),
        "n_atoms": len(symbols),
        "grid": [0.0, 90.0, 180.0, 270.0],
        "driven_bond": [1, 2],
        "driven_slot": slot,
        "indices": torch.tensor(indices, dtype=torch.long),
        "bond_of": torch.tensor(bond_of, dtype=torch.long),
        "deltas": torch.zeros((points, len(indices)), dtype=torch.float64),
        "n_bonds": n_bonds,
        "n_bonds_available": n_bonds,
        "qm": np.linspace(minimum, minimum + 3.0, points),
        "qm_t": torch.linspace(minimum, minimum + 3.0, points, dtype=torch.float64),
        "mmff": None,
        "reference_xyz": np.full((1, len(symbols), 3), float(minimum)),
        "scan_xyz": np.zeros((points, len(symbols), 3)),
    }


class Args:
    val_fraction = 0.25
    split_mode = "cross_bond"
    folds = 2
    seed = 0


def test_compatible_scans_of_one_molecule_become_one_record():
    quarantine = []
    records = group_scans([scan("AAA", "0", 0, n_bonds=2),
                           scan("AAA", "1", 1, n_bonds=2)], quarantine)
    assert len(records) == 1
    assert [e["key"] for e in records[0]["supervised"]] == ["0", "1"]
    assert [e["slot"] for e in records[0]["supervised"]] == [0, 1]
    assert quarantine == []


def test_scans_that_disagree_on_atom_order_are_not_grouped():
    """The failure this guards is silent: bond A's dihedrals on bond B's atoms."""
    quarantine = []
    records = group_scans(
        [scan("AAA", "0", 0, n_bonds=2),
         scan("AAA", "1", 1, n_bonds=2, indices=((3, 2, 1, 0),))],
        quarantine,
    )
    assert len(records) == 2
    assert all(len(r["supervised"]) == 1 for r in records)
    assert len(quarantine) == 1 and "atom" in quarantine[0][1]


def test_scans_with_different_elements_are_not_grouped():
    quarantine = []
    records = group_scans(
        [scan("AAA", "0", 0, n_bonds=2),
         scan("AAA", "1", 1, n_bonds=2, symbols=("C", "C", "C", "N"))],
        quarantine,
    )
    assert len(records) == 2
    assert len(quarantine) == 1


def test_different_molecules_are_never_grouped():
    quarantine = []
    records = group_scans([scan("AAA", "0", 0), scan("BBB", "1", 0)], quarantine)
    assert len(records) == 2
    assert quarantine == []


def test_shared_geometry_is_the_lowest_energy_frame_of_any_scan():
    """Any one scan's minimum is a minimum along one rotor only."""
    quarantine = []
    records = group_scans([scan("AAA", "0", 0, n_bonds=2, minimum=5.0),
                           scan("AAA", "1", 1, n_bonds=2, minimum=-2.0)],
                          quarantine)
    assert records[0]["reference_xyz"][0, 0, 0] == pytest.approx(-2.0)


def test_folds_hold_out_bonds_only_from_multi_rotor_molecules():
    quarantine = []
    records = group_scans(
        [scan("AAA", "0", 0, n_bonds=2), scan("AAA", "1", 1, n_bonds=2),
         scan("BBB", "2", 0)],
        quarantine,
    )
    folds = supervised_folds(records, 2, seed=0)
    held = [pair for fold in folds for pair in fold]
    assert len(held) == 2, "both bonds of the two-rotor molecule are holdable"
    assert sorted(held) == [(0, 0), (0, 1)]
    single = [i for i, r in enumerate(records) if len(r["supervised"]) == 1]
    assert all(i not in {a for a, _ in held} for i in single)


def test_every_holdable_bond_is_held_out_exactly_once():
    quarantine = []
    scans = [scan("AAA", str(j), j, n_bonds=4) for j in range(4)]
    records = group_scans(scans, quarantine)
    folds = supervised_folds(records, 2, seed=1)
    held = [pair for fold in folds for pair in fold]
    assert sorted(held) == [(0, j) for j in range(4)]
    assert len(held) == len(set(held))


def test_a_molecule_appears_on_both_sides_with_disjoint_bonds():
    quarantine = []
    records = group_scans([scan("AAA", str(j), j, n_bonds=4) for j in range(4)],
                          quarantine)
    folds = supervised_folds(records, 2, seed=1)
    train, validation, held_out = fold_sets(records, folds[0], Args, True, 0)

    held_keys = {e["key"] for r in held_out for e in r["supervised"]}
    trained_keys = {e["key"] for r in train + validation for e in r["supervised"]}
    assert held_keys, "the fold held nothing out"
    assert not (held_keys & trained_keys), "a held-out bond is also trained on"
    assert held_keys | trained_keys == {"0", "1", "2", "3"}


def test_scan_mode_folds_hold_out_whole_records():
    quarantine = []
    records = group_scans([scan("AAA", "0", 0), scan("BBB", "1", 0)], quarantine)
    train, validation, held_out = fold_sets(records, [0], Args, False, 0)
    assert [r["key"] for r in held_out] == ["0"]
    assert {r["key"] for r in train + validation} == {"1"}


def test_enumerating_other_bonds_leaves_the_driven_column_unchanged():
    """The claim the whole mode rests on, checked against the real model.

    The energy of bond b is built by scatter_add over the torsions whose central
    bond is b, in both `bond_representation` and `phase_vectors`. So predicting
    one bond and predicting four must agree to machine precision on the bond they
    share. If this fails, `--rotors driven` and `--rotors all` are different
    models and not two reductions of one.
    """
    from openff.nagl.torsion import TorsionModel, phase_vectors

    torch.manual_seed(0)
    n_atoms, n_features = 10, 8
    model = TorsionModel(n_atom_features=n_features).double().eval()
    h = torch.randn(n_atoms, n_features, dtype=torch.float64)

    # Four torsions: two about bond (1,2), two about bond (5,6).
    all_indices = torch.tensor([[0, 1, 2, 3], [4, 1, 2, 7],
                                [3, 5, 6, 8], [9, 5, 6, 0]], dtype=torch.long)
    all_bond_of = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    deltas = torch.randn(6, 4, dtype=torch.float64)

    def energy(indices, bond_of, n_bonds, d):
        h_bond, c = model.bond_representation(h, indices, bond_of, n_bonds)
        s = phase_vectors(d, c, bond_of, n_bonds,
                          periodicities=model.periodicities)
        return model.energy(h_bond, s)

    with torch.no_grad():
        both = energy(all_indices, all_bond_of, 2, deltas)[..., 0]
        alone = energy(all_indices[:2], torch.zeros(2, dtype=torch.long), 1,
                       deltas[:, :2])[..., 0]
    difference = float((both - alone).abs().max())
    assert difference < 1e-14, (
        f"the driven column changed by {difference:.3g} when other rotors were "
        f"enumerated; scatter_add reordering is 1e-17, this is not that"
    )


def test_the_other_column_is_not_the_same_number():
    """The negative half: the two bonds must not be accidentally identical."""
    from openff.nagl.torsion import TorsionModel, phase_vectors

    torch.manual_seed(0)
    model = TorsionModel(n_atom_features=8).double().eval()
    h = torch.randn(10, 8, dtype=torch.float64)
    indices = torch.tensor([[0, 1, 2, 3], [4, 1, 2, 7],
                            [3, 5, 6, 8], [9, 5, 6, 0]], dtype=torch.long)
    bond_of = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    deltas = torch.randn(6, 4, dtype=torch.float64)
    with torch.no_grad():
        h_bond, c = model.bond_representation(h, indices, bond_of, 2)
        s = phase_vectors(deltas, c, bond_of, 2, periodicities=model.periodicities)
        energies = model.energy(h_bond, s)
    assert not torch.allclose(energies[..., 0], energies[..., 1])


# ---------------------------------------------------------------------------
# The split has to survive grouping. Added 2026-09-13 after SLURM job 79364 was
# cancelled nine minutes in: the two arms prepared 15,659 and 13,424 records,
# permuted lists of different lengths at one seed, and held out 3,915 and 3,356
# molecules with nothing making them the same molecules.


class IdentityArgs:
    seed = 20260911
    val_fraction = 0.15
    test_fraction = 0.25


def test_grouping_does_not_move_a_molecule_between_splits():
    """The defect that cancelled job 79364, as a test."""
    from finetune_torsion import identity_split

    scans = [scan(f"KEY{j:03d}", str(j), 0, n_bonds=2) for j in range(200)]
    scans += [scan("KEY000", "200", 1, n_bonds=2),
              scan("KEY001", "201", 1, n_bonds=2)]
    ungrouped = scans
    grouped = group_scans(scans, [])
    assert len(grouped) < len(ungrouped), "the fixture must actually group"

    def partition(records):
        train, validation, test = identity_split(records, IdentityArgs)[:3]
        return {
            "train": {records[i].get("inchi_key") for i in train},
            "validation": {records[i].get("inchi_key") for i in validation},
            "test": {records[i].get("inchi_key") for i in test},
        }

    assert partition(ungrouped) == partition(grouped)


def test_identity_split_is_a_partition():
    from finetune_torsion import identity_split

    records = [scan(f"KEY{j:04d}", str(j), 0) for j in range(500)]
    train, validation, test, how = identity_split(records, IdentityArgs)
    assert how == "hashed molecule identity"
    assert sorted(train + validation + test) == list(range(500))
    assert not (set(train) & set(validation) & set(test))


def test_identity_split_lands_near_the_requested_fractions():
    from finetune_torsion import identity_split

    records = [scan(f"KEY{j:05d}", str(j), 0) for j in range(4000)]
    train, validation, test, _ = identity_split(records, IdentityArgs)
    assert 0.22 < len(test) / 4000 < 0.28
    assert 0.12 < len(validation) / 4000 < 0.18


def test_identity_split_moves_when_the_seed_moves():
    """The negative half: a split that ignored the seed would also be stable."""
    from finetune_torsion import identity_split

    records = [scan(f"KEY{j:04d}", str(j), 0) for j in range(400)]

    class Other(IdentityArgs):
        seed = 7

    assert identity_split(records, IdentityArgs)[2] != identity_split(records, Other)[2]


def test_adding_a_molecule_leaves_every_other_one_where_it_was():
    from finetune_torsion import identity_split

    records = [scan(f"KEY{j:04d}", str(j), 0) for j in range(300)]
    before = {records[i]["inchi_key"] for i in identity_split(records, IdentityArgs)[2]}
    records.append(scan("NEWKEY", "300", 0))
    after = {records[i]["inchi_key"] for i in identity_split(records, IdentityArgs)[2]}
    assert before <= after and len(after - before) <= 1
