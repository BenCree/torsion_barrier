"""Tests for the stage 2 scripts, beside the scripts and run by the same command.

Each of these exists because of a specific way the pipeline can be wrong while
returning confident numbers, not to cover lines.
"""

import numpy as np
import pytest

from fit_torsion_head import BOND_TYPES, enumerate_torsions, mmff94_profile, ranks_within


def _toluene():
    """Built by RDKit so the reference MMFF94 answer is RDKit's own."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    molecule = Chem.AddHs(Chem.MolFromSmiles("Cc1ccccc1"))
    AllChem.EmbedMultipleConfs(molecule, numConfs=4, randomSeed=7)
    symbols = [atom.GetSymbol() for atom in molecule.GetAtoms()]
    bonds = np.array([[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in molecule.GetBonds()])
    orders = np.array([b.GetBondTypeAsDouble() for b in molecule.GetBonds()])
    xyz = np.stack([c.GetPositions() for c in molecule.GetConformers()])
    return molecule, symbols, bonds, orders, xyz


def test_mmff94_matches_rdkits_own_answer():
    """The baseline must be MMFF94, not something computed in its name."""
    from rdkit.Chem import AllChem

    molecule, symbols, bonds, orders, xyz = _toluene()
    ours = mmff94_profile(symbols, bonds, orders, xyz)

    properties = AllChem.MMFFGetMoleculeProperties(molecule)
    reference = np.array(
        [
            AllChem.MMFFGetMoleculeForceField(molecule, properties, confId=c.GetId()).CalcEnergy()
            for c in molecule.GetConformers()
        ]
    )
    reference -= reference.min()
    assert ours == pytest.approx(reference, abs=1e-9)


def test_discarding_bond_orders_changes_the_answer():
    """THE DEFECT THIS GUARDS, stated as a measurement rather than a warning.

    An earlier version stored only the atom pairs and built every bond as SINGLE.
    On toluene that does not merely shift the energies: it moves the minimum to a
    different conformer, so the ranks the claim is adjudicated on would have been
    wrong while the column was still called mmff94.
    """
    _, symbols, bonds, orders, xyz = _toluene()
    correct = mmff94_profile(symbols, bonds, orders, xyz)
    all_single = mmff94_profile(symbols, bonds, np.ones_like(orders), xyz)
    assert all_single is not None
    assert int(np.argmin(correct)) != int(np.argmin(all_single))


def test_unknown_bond_order_is_refused_rather_than_guessed():
    _, symbols, bonds, orders, xyz = _toluene()
    broken = orders.copy()
    broken[0] = 2.7
    assert mmff94_profile(symbols, bonds, broken, xyz) is None
    assert set(BOND_TYPES) == {1.0, 1.5, 2.0, 3.0}


def test_torsion_enumeration_on_a_known_graph():
    """Ethane: one central bond, nine proper torsions, and no others."""
    bonds = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (1, 6), (1, 7)]
    groups = enumerate_torsions(bonds, 8)
    assert set(groups) == {(0, 1)}, "only the C-C bond is central to a proper torsion"
    assert len(groups[(0, 1)]) == 9
    assert all(tuple(sorted(t[1:3])) == (0, 1) for t in groups[(0, 1)])
    assert all(len(set(t)) == 4 for t in groups[(0, 1)])


def test_torsion_enumeration_excludes_three_membered_rings():
    """In a cyclopropane the i and l of a torsion can be the same atom."""
    bonds = [(0, 1), (1, 2), (2, 0)]
    for key, torsions in enumerate_torsions(bonds, 3).items():
        assert all(t[0] != t[3] for t in torsions), f"{key} produced i == l"


def test_a_flat_profile_ties_everywhere():
    """The constant control must tie, so that its verdict is unverifiable.

    A tie scored as a wrong answer is one of the four defects that reached
    committed results on 2026-08-22, and this is what would catch it recurring.
    """
    ranks = ranks_within(np.zeros(24))
    assert len(set(ranks.tolist())) == 1
    assert ranks[0] == pytest.approx(12.5)


def test_ranks_are_within_one_scan_and_ascending():
    assert ranks_within(np.array([3.0, 1.0, 2.0])).tolist() == [3.0, 1.0, 2.0]
    assert ranks_within(np.array([0.0, 5.0, 5.0])).tolist() == [1.0, 2.5, 2.5]
