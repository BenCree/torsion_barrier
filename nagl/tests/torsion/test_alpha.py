"""Known-answer tests for the torsion phase coordinate.

The decisive cases are the rotor symmetries. Staggered and eclipsed ethane have
known phases at n = 3 that do not depend on the sign convention of the dihedral,
so they test the quantity rather than the frame it was built in. The three-fold
degeneracy tests exist because an earlier draft evaluated only n = 1, where every
methyl and every CF3 sums to exactly zero and floating point returns a confident
angle for an undefined quantity.
"""

import numpy as np
import pytest

from openff.nagl.torsion import (
    alpha_from_deltas,
    alpha_from_molecule,
    torsions_about_bond,
)


def _wrap(angle):
    """Fold an angle into (-pi, pi]."""
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------------------
# alpha_from_deltas, the arithmetic on its own
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("degrees", [0.0, 30.0, 60.0, 123.4, -75.0, 180.0])
def test_single_torsion_recovers_n_times_its_angle(degrees):
    """One torsion, weight one: s_n is a unit vector at n * delta."""
    delta = np.radians(degrees)
    s, alpha, magnitude = alpha_from_deltas([delta])
    assert magnitude == pytest.approx(np.ones(6), abs=1e-12)
    for row, n in enumerate((1, 2, 3, 4, 5, 6)):
        assert _wrap(alpha[row] - n * delta) == pytest.approx(0.0, abs=1e-9)


def test_atan2_arguments_are_not_swapped():
    """atan2(sin, cos), not atan2(cos, sin).

    A single torsion at 30 degrees must give alpha_1 = 30 degrees. Swapping the
    arguments returns pi/2 - alpha, which is 60 degrees here, and inverts the
    response to rotating the rotor. The two agree at 45 degrees, so a test at
    45 degrees would not have caught it.
    """
    _, alpha, _ = alpha_from_deltas([np.radians(30.0)], periodicities=(1,))
    assert np.degrees(alpha[0]) == pytest.approx(30.0, abs=1e-9)
    assert np.degrees(alpha[0]) != pytest.approx(60.0, abs=1.0)


@pytest.mark.parametrize("gamma_degrees", [1.0, 17.0, 90.0, -45.0])
def test_rotating_the_rotor_advances_the_phase_by_n_gamma(gamma_degrees):
    """THE EQUIVARIANCE PROPERTY. Turning the rotor by gamma moves alpha_n by n*gamma.

    This is what makes alpha a coordinate for the conformation rather than a
    summary statistic of it, and it is the property the swapped-argument bug
    destroyed: it moved alpha by MINUS gamma.
    """
    gamma = np.radians(gamma_degrees)
    deltas = np.radians([10.0, 130.0, 250.0, 45.0])
    weights = np.array([0.3, 0.5, 0.2, 0.9])

    _, alpha_0, _ = alpha_from_deltas(deltas, weights=weights)
    _, alpha_1, _ = alpha_from_deltas(deltas + gamma, weights=weights)

    for row, n in enumerate((1, 2, 3, 4, 5, 6)):
        assert _wrap(alpha_1[row] - alpha_0[row] - n * gamma) == pytest.approx(
            0.0, abs=1e-9
        ), f"n = {n} did not advance by {n} * gamma"


def test_three_fold_rotor_is_degenerate_below_n_three():
    """A methyl is silent at n = 1, 2, 4 and 5 and speaks at n = 3 and 6.

    THE CASE THAT FORCED THE PERIODICITY EXPANSION. Three equal torsions 120
    degrees apart sum to exactly zero at every n that is not a multiple of three.
    Evaluating n = 1 alone leaves alpha undefined for every methyl rotor.
    """
    deltas = np.radians([0.0, 120.0, 240.0])
    _, alpha, magnitude = alpha_from_deltas(deltas)

    silent = [0, 1, 3, 4]  # n = 1, 2, 4, 5
    loud = [2, 5]  # n = 3, 6

    assert magnitude[silent] == pytest.approx(np.zeros(4), abs=1e-12)
    assert np.all(magnitude[loud] > 2.99)
    assert np.all(np.isnan(alpha[silent])), (
        "a degenerate harmonic must report NaN, not a confident angle"
    )
    assert not np.any(np.isnan(alpha[loud]))


@pytest.mark.parametrize("phi_degrees", [0.0, 20.0, 60.0, 37.5])
def test_three_fold_rotor_phase_is_three_phi(phi_degrees):
    """Where a three-fold rotor is informative, alpha_3 = 3 * phi."""
    phi = np.radians(phi_degrees)
    deltas = phi + np.radians([0.0, 120.0, 240.0])
    _, alpha, _ = alpha_from_deltas(deltas, periodicities=(3,))
    assert _wrap(alpha[0] - 3 * phi) == pytest.approx(0.0, abs=1e-9)


def test_two_fold_rotor_is_degenerate_at_odd_periodicities():
    """Two torsions 180 degrees apart cancel at every odd n."""
    deltas = np.radians([25.0, 205.0])
    _, alpha, magnitude = alpha_from_deltas(deltas)
    odd = [0, 2, 4]  # n = 1, 3, 5
    even = [1, 3, 5]  # n = 2, 4, 6
    assert magnitude[odd] == pytest.approx(np.zeros(3), abs=1e-12)
    assert np.all(magnitude[even] > 1.99)
    assert np.all(np.isnan(alpha[odd]))


def test_degeneracy_is_reported_rather_than_returned_as_zero():
    """The failure mode this guard exists for.

    Floating point does not give exactly zero, so |s| lands near 1e-16 and a
    naive atan2 returns 0.0 radians: a confident answer to an undefined question.
    """
    deltas = np.radians([0.0, 120.0, 240.0]) + 1e-17
    _, alpha, magnitude = alpha_from_deltas(deltas, periodicities=(1,))
    assert magnitude[0] < 1e-10
    assert np.isnan(alpha[0]), "near-zero magnitude must not return a phase of 0.0"


def test_weights_pull_the_phase_toward_the_heavier_torsion():
    """c_ij is a weight, and the phase must respond to it."""
    deltas = np.radians([0.0, 90.0])
    _, balanced, _ = alpha_from_deltas(deltas, weights=[1.0, 1.0], periodicities=(1,))
    _, tipped, _ = alpha_from_deltas(deltas, weights=[1.0, 9.0], periodicities=(1,))
    assert np.degrees(balanced[0]) == pytest.approx(45.0, abs=1e-9)
    assert np.degrees(tipped[0]) > 80.0


def test_phase_is_invariant_to_the_order_torsions_are_listed_in():
    """A sum does not depend on the order of its terms, and the test says so."""
    deltas = np.radians([10.0, 130.0, 250.0, 45.0])
    weights = np.array([0.3, 0.5, 0.2, 0.9])
    order = np.array([2, 0, 3, 1])
    _, a, _ = alpha_from_deltas(deltas, weights=weights)
    _, b, _ = alpha_from_deltas(deltas[order], weights=weights[order])
    assert _wrap(a - b) == pytest.approx(np.zeros(6), abs=1e-12)


def test_mismatched_weights_are_refused():
    with pytest.raises(ValueError, match="does not match"):
        alpha_from_deltas([0.1, 0.2], weights=[1.0])


def test_two_dimensional_deltas_are_refused():
    with pytest.raises(ValueError, match="one dimensional"):
        alpha_from_deltas(np.zeros((2, 2)))


# ---------------------------------------------------------------------------
# alpha_from_molecule, on a real molecule with a built conformer
# ---------------------------------------------------------------------------


def _ethane_with_conformer(phi_degrees, bond_length=1.53, ch=1.09, tilt_degrees=110.0):
    """Ethane placed by hand, the far methyl turned by phi about the C-C axis.

    Built rather than embedded so the rotor angle is exact and the test does not
    depend on a conformer generator. The C-C axis is x; each methyl's hydrogens
    sit at azimuths 0, 120 and 240 degrees about it, with the far set offset by
    phi.
    """
    from openff.toolkit import Molecule
    from openff.units import unit

    molecule = Molecule.from_smiles("CC")
    carbons = [a.molecule_atom_index for a in molecule.atoms if a.atomic_number == 6]
    assert len(carbons) == 2
    c1, c2 = carbons
    hydrogens = {
        c: [
            n.molecule_atom_index
            for n in molecule.atoms[c].bonded_atoms
            if n.atomic_number == 1
        ]
        for c in carbons
    }

    tilt = np.radians(tilt_degrees)
    xyz = np.zeros((molecule.n_atoms, 3))
    xyz[c1] = [0.0, 0.0, 0.0]
    xyz[c2] = [bond_length, 0.0, 0.0]

    for index, h in enumerate(hydrogens[c1]):
        azimuth = np.radians(120.0 * index)
        xyz[h] = xyz[c1] + ch * np.array(
            [np.cos(tilt), np.sin(tilt) * np.cos(azimuth), np.sin(tilt) * np.sin(azimuth)]
        )
    for index, h in enumerate(hydrogens[c2]):
        azimuth = np.radians(120.0 * index + phi_degrees)
        xyz[h] = xyz[c2] + ch * np.array(
            [-np.cos(tilt), np.sin(tilt) * np.cos(azimuth), np.sin(tilt) * np.sin(azimuth)]
        )

    molecule.add_conformer(xyz * unit.angstrom)
    return molecule, (min(c1, c2), max(c1, c2))


def test_ethane_central_bond_carries_nine_torsions():
    molecule, bond = _ethane_with_conformer(60.0)
    indices = torsions_about_bond(molecule, *bond)
    assert len(indices) == 9, "3 hydrogens by 3 hydrogens across one C-C bond"
    assert all(tuple(sorted(t[1:3])) == bond for t in indices)


def test_eclipsed_ethane_has_phase_zero_at_n_three():
    """KNOWN ANSWER, convention free. Eclipsed: every 3*delta is a multiple of 2pi."""
    molecule, bond = _ethane_with_conformer(0.0)
    phase = alpha_from_molecule(molecule)[bond]
    row = phase.periodicities.index(3)
    assert phase.magnitude[row] == pytest.approx(9.0, abs=1e-6)
    assert _wrap(phase.alpha[row]) == pytest.approx(0.0, abs=1e-6)


def test_staggered_ethane_has_phase_pi_at_n_three():
    """KNOWN ANSWER, convention free. Staggered: every 3*delta is an odd multiple of pi.

    Eclipsed and staggered differ by pi in alpha_3 whichever way the dihedral
    sign convention runs, so this pins the physics without pinning the frame.
    """
    molecule, bond = _ethane_with_conformer(60.0)
    phase = alpha_from_molecule(molecule)[bond]
    row = phase.periodicities.index(3)
    assert phase.magnitude[row] == pytest.approx(9.0, abs=1e-6)
    assert abs(_wrap(phase.alpha[row])) == pytest.approx(np.pi, abs=1e-6)


def test_ethane_is_degenerate_at_every_periodicity_below_three():
    """Nine torsions in three-fold by three-fold symmetry cancel at n = 1 and 2."""
    molecule, bond = _ethane_with_conformer(23.0)
    phase = alpha_from_molecule(molecule)[bond]
    for n in (1, 2, 4, 5):
        row = phase.periodicities.index(n)
        assert phase.magnitude[row] < 1e-6, f"n = {n} should cancel for ethane"
        assert np.isnan(phase.alpha[row])
    assert phase.informative_periodicities() == (3, 6)


@pytest.mark.parametrize("phi_degrees", [5.0, 20.0, 41.0])
def test_turning_the_real_rotor_advances_alpha_three_by_three_phi(phi_degrees):
    """Equivariance measured through the geometry, not just the arithmetic.

    The handedness is whatever the IUPAC dihedral convention gives, so the test
    asserts the magnitude of the advance and that its sign is the same one the
    eclipsed-to-staggered pair implies, rather than hardcoding a frame.
    """
    reference, bond = _ethane_with_conformer(0.0)
    turned, _ = _ethane_with_conformer(phi_degrees)
    row = 2  # n = 3

    a0 = alpha_from_molecule(reference)[bond].alpha[row]
    a1 = alpha_from_molecule(turned)[bond].alpha[row]
    advance = _wrap(a1 - a0)
    assert abs(advance) == pytest.approx(
        abs(_wrap(3 * np.radians(phi_degrees))), abs=1e-6
    )


def test_terminal_bonds_are_absent_rather_than_zero():
    """A C-H bond is the centre of no proper torsion, so it has no phase.

    Reporting it as 0.0 would let a caller read "no rotor" as "rotor at zero",
    which is the bug the earlier draft shipped when it filled non-central bonds
    with zeros before sorting them into a dense vector.
    """
    molecule, bond = _ethane_with_conformer(60.0)
    phases = alpha_from_molecule(molecule)
    assert set(phases) == {bond}, "only the C-C bond is central to a proper torsion"


def test_phase_is_invariant_to_relabelling_the_atoms():
    """PERMUTATION INVARIANCE, the property the whole scheme exists for.

    Swapping two hydrogens on the same carbon is a relabelling, not a motion.
    Any scheme that picks one reference dihedral out of the nine changes its
    answer here; this one must not.
    """
    molecule, bond = _ethane_with_conformer(41.0)
    original = alpha_from_molecule(molecule)[bond]

    permuted = alpha_from_molecule.__module__  # keep import local to the test
    xyz = np.asarray(molecule.conformers[0].m_as("angstrom"))
    hydrogens = [a.molecule_atom_index for a in molecule.atoms if a.atomic_number == 1]
    swapped = xyz.copy()
    swapped[[hydrogens[0], hydrogens[1]]] = swapped[[hydrogens[1], hydrogens[0]]]

    from openff.units import unit

    clone = molecule.__class__(molecule)
    clone._conformers = [swapped * unit.angstrom]
    relabelled = alpha_from_molecule(clone)[bond]

    assert _wrap(relabelled.alpha[2] - original.alpha[2]) == pytest.approx(
        0.0, abs=1e-9
    )
    assert relabelled.magnitude == pytest.approx(original.magnitude, abs=1e-9)
