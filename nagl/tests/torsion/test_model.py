"""Known-answer and symmetry tests for the torsion energy head.

Three things are being tested and they fail in different ways. The symmetries
(reversal, relabelling, rotor periodicity) are enforced by construction, so a
test that finds them broken has found a coding error, not a training problem.
The Fourier reading is an identity and must hold to machine precision. The
recovery test is the known answer for the whole pipeline: a model fitted to a
torsion profile generated from known coefficients must return those coefficients,
and if it cannot do that on synthetic data there is no reason to trust it on QM.
"""

import numpy as np
import pytest
import torch

from openff.nagl.torsion import (
    JanossyTorsionPooling,
    rotate_phase,
    TorsionCoefficients,
    TorsionEnergyHead,
    TorsionModel,
    phase_vectors,
)

PERIODICITIES = (1, 2, 3, 4, 5, 6)


@pytest.fixture
def rotor():
    """A three-fold by three-fold rotor: nine torsions about one central bond.

    Atoms 0, 1, 2 are the near substituents; 3 is the near centre, 4 the far
    centre; 5, 6, 7 are the far substituents.
    """
    torsion_indices = torch.tensor(
        [[i, 3, 4, l] for i in (0, 1, 2) for l in (5, 6, 7)], dtype=torch.long
    )
    bond_of_torsion = torch.zeros(len(torsion_indices), dtype=torch.long)
    return torsion_indices, bond_of_torsion


def _rotor_deltas(phi, near=(0.0, 120.0, 240.0), far=(0.0, 120.0, 240.0)):
    """The nine dihedrals of a rotor turned by phi degrees."""
    return torch.tensor(
        [np.radians(f + phi - n) for n in near for f in far], dtype=torch.float64
    )


# ---------------------------------------------------------------------------
# symmetries enforced by construction
# ---------------------------------------------------------------------------


def test_pooling_is_invariant_to_reading_the_torsion_backwards():
    """i-j-k-l and l-k-j-i are one torsion, so h_r must not distinguish them."""
    torch.manual_seed(0)
    pooling = JanossyTorsionPooling(n_features=8, hidden=(16, 16)).double()
    h = torch.randn(6, 8, dtype=torch.float64)
    forward = torch.tensor([[0, 1, 2, 3], [4, 1, 2, 5]])
    reverse = forward.flip(-1)
    assert torch.allclose(pooling(h, forward), pooling(h, reverse), atol=1e-12)


def test_coefficients_are_invariant_to_swapping_the_terminal_atoms():
    """c_ijkl is built from h_i + h_l, so i <-> l cannot change it."""
    torch.manual_seed(0)
    coefficients = TorsionCoefficients(n_features=8).double()
    h = torch.randn(6, 8, dtype=torch.float64)
    forward = torch.tensor([[0, 1, 2, 3], [4, 1, 2, 5]])
    assert torch.allclose(
        coefficients(h, forward), coefficients(h, forward.flip(-1)), atol=1e-12
    )


def test_energy_is_unchanged_when_a_three_fold_rotor_turns_by_120_degrees(rotor):
    """THE PHYSICAL SYMMETRY TEST.

    Turning a methyl by 120 degrees permutes its torsions and returns the same
    molecule, so the energy must be identical, not merely close. Any scheme that
    picks one reference dihedral out of the nine satisfies this only by accident.
    """
    torsion_indices, bond_of_torsion = rotor
    torch.manual_seed(0)
    model = TorsionModel(n_atom_features=8, torsion_hidden=(16, 16)).double()

    # The three substituents on each end must be chemically identical for the
    # 120 degree turn to be a symmetry at all. With distinct embeddings the
    # rotor is a substituted one, the turn is a real motion, and the energy is
    # supposed to change. That distinction is the test.
    h = torch.randn(8, 8, dtype=torch.float64)
    h[[1, 2]] = h[0].clone()
    h[[6, 7]] = h[5].clone()

    energies = [
        float(
            model(h, torsion_indices, _rotor_deltas(phi), bond_of_torsion, 1)[0].sum()
        )
        for phi in (0.0, 120.0, 240.0)
    ]
    assert energies[1] == pytest.approx(energies[0], abs=1e-10)
    assert energies[2] == pytest.approx(energies[0], abs=1e-10)


def test_energy_is_periodic_in_the_rotor_angle(rotor):
    """A full turn returns the molecule to itself."""
    torsion_indices, bond_of_torsion = rotor
    torch.manual_seed(0)
    model = TorsionModel(n_atom_features=8, torsion_hidden=(16, 16)).double()
    h = torch.randn(8, 8, dtype=torch.float64)
    at = lambda phi: float(
        model(h, torsion_indices, _rotor_deltas(phi), bond_of_torsion, 1)[0].sum()
    )
    assert at(360.0) == pytest.approx(at(0.0), abs=1e-10)
    assert at(47.0 + 360.0) == pytest.approx(at(47.0), abs=1e-10)


def test_energy_is_invariant_to_relabelling_the_substituents(rotor):
    """PERMUTATION INVARIANCE end to end.

    Renaming the atoms is not a motion. Every embedding is distinct here, so the
    invariance is a property of the architecture rather than an accident of
    equal inputs: the atom names change, each embedding travels with its atom,
    and every dihedral stays attached to the torsion that owns it.
    """
    torsion_indices, bond_of_torsion = rotor
    torch.manual_seed(0)
    model = TorsionModel(n_atom_features=8, torsion_hidden=(16, 16)).double()
    h = torch.randn(8, 8, dtype=torch.float64)
    deltas = _rotor_deltas(23.0)

    reference = model(h, torsion_indices, deltas, bond_of_torsion, 1)[0].sum()

    relabel = {0: 2, 1: 0, 2: 1, 3: 3, 4: 4, 5: 6, 6: 7, 7: 5}
    permuted_indices = torch.tensor(
        [[relabel[int(a)] for a in row] for row in torsion_indices], dtype=torch.long
    )
    permuted_h = torch.empty_like(h)
    for old_index, new_index in relabel.items():
        permuted_h[new_index] = h[old_index]

    permuted = model(permuted_h, permuted_indices, deltas, bond_of_torsion, 1)[0].sum()
    assert float(permuted) == pytest.approx(float(reference), abs=1e-10)


def test_energy_is_invariant_to_the_order_the_torsions_are_listed_in(rotor):
    """The phase is a sum over torsions, so listing them differently cannot move it.

    Separate from relabelling because it tests the scatter that accumulates
    torsions into their central bond rather than the naming of the atoms.
    """
    torsion_indices, bond_of_torsion = rotor
    torch.manual_seed(0)
    model = TorsionModel(n_atom_features=8, torsion_hidden=(16, 16)).double()
    h = torch.randn(8, 8, dtype=torch.float64)
    deltas = _rotor_deltas(23.0)

    order = torch.randperm(len(torsion_indices), generator=torch.manual_seed(7))
    reference = model(h, torsion_indices, deltas, bond_of_torsion, 1)[0].sum()
    shuffled = model(
        h, torsion_indices[order], deltas[order], bond_of_torsion, 1
    )[0].sum()
    assert float(shuffled) == pytest.approx(float(reference), abs=1e-10)


# ---------------------------------------------------------------------------
# the Fourier reading, which is an identity
# ---------------------------------------------------------------------------


def test_energy_equals_the_fourier_series_of_its_own_parameters():
    """E = sum_n k_n . s_n, with k_n the numbers `torsion_parameters` returns.

    If this identity does not hold then the parameters written into a force
    field are not the ones the model used, which is the failure that would make
    the whole scheme quietly wrong while every loss curve looked healthy.
    """
    torch.manual_seed(0)
    head = TorsionEnergyHead(n_bond_features=12, periodicities=PERIODICITIES).double()
    h_bond = torch.randn(5, 12, dtype=torch.float64)
    s = torch.randn(5, 6, 2, dtype=torch.float64)

    energy = head(h_bond, s)
    k = head.torsion_parameters(h_bond)
    assert torch.allclose(energy, (k * s).sum(dim=(1, 2)), atol=1e-12)


def test_nonlinear_head_refuses_to_report_parameters():
    head = TorsionEnergyHead(n_bond_features=12, nonlinear=True)
    with pytest.raises(NotImplementedError, match="no Fourier coefficients"):
        head.torsion_parameters(torch.randn(2, 12))


def test_energy_is_linear_in_the_phase_vector():
    """Doubling s doubles E. This is what makes the coefficients parameters."""
    torch.manual_seed(0)
    head = TorsionEnergyHead(n_bond_features=12).double()
    h_bond = torch.randn(3, 12, dtype=torch.float64)
    s = torch.randn(3, 6, 2, dtype=torch.float64)
    assert torch.allclose(head(h_bond, 2.0 * s), 2.0 * head(h_bond, s), atol=1e-12)


# ---------------------------------------------------------------------------
# gradients, which pose refinement needs
# ---------------------------------------------------------------------------


def test_forces_flow_back_to_the_dihedral_angles(rotor):
    """Pose refinement minimises the energy, so dE/ddelta must exist and be finite.

    Computed from s rather than alpha for exactly this reason: alpha has a branch
    cut and is NaN wherever a rotor's symmetry cancels its harmonic, and a NaN
    gradient at a methyl would be silent until an optimisation stopped moving.
    """
    torsion_indices, bond_of_torsion = rotor
    torch.manual_seed(0)
    model = TorsionModel(n_atom_features=8, torsion_hidden=(16, 16)).double()
    h = torch.randn(8, 8, dtype=torch.float64)
    h[[1, 2]] = h[0].clone()  # a real methyl, so n = 1 and n = 2 really do cancel
    h[[6, 7]] = h[5].clone()
    deltas = _rotor_deltas(0.0).requires_grad_(True)

    energy, _, alpha, magnitude, _ = model(
        h, torsion_indices, deltas, bond_of_torsion, 1
    )
    energy.sum().backward()

    assert deltas.grad is not None
    assert torch.isfinite(deltas.grad).all(), "a NaN force would stall a minimisation"
    # The rotor is degenerate at n = 1 and 2, so alpha there is NaN by design and
    # the gradient survives it anyway. That is the point of the s parametrisation.
    assert torch.isnan(alpha[0, :2]).all()
    assert (magnitude[0, :2] < 1e-8).all()


# ---------------------------------------------------------------------------
# the known answer for the whole pipeline
# ---------------------------------------------------------------------------


def test_model_recovers_known_fourier_coefficients_from_a_torsion_scan():
    """THE PIPELINE KNOWN-ANSWER TEST.

    A profile is generated from coefficients chosen in advance; the model is
    fitted to it; the coefficients it reports must be the ones used to make the
    data. A model that cannot do this on synthetic data has no claim on QM data,
    and a test written the other way round, fitting first and reading the answer
    off afterwards, would test nothing.
    """
    torsion_indices = torch.tensor(
        [[i, 3, 4, l] for i in (0, 1, 2) for l in (5, 6, 7)], dtype=torch.long
    )
    bond_of_torsion = torch.zeros(9, dtype=torch.long)
    angles = torch.linspace(0, 360, 73, dtype=torch.float64)[:-1]
    scan = torch.stack([_rotor_deltas(float(phi)) for phi in angles])

    torch.manual_seed(0)
    model = TorsionModel(n_atom_features=8, torsion_hidden=(32, 32)).double()
    h = torch.randn(8, 8, dtype=torch.float64)

    # Truth: a fixed set of coefficients acting on the model's own phase vectors,
    # so the target is inside the hypothesis class and the test measures fitting
    # rather than expressiveness. The coefficients are frozen before fitting.
    with torch.no_grad():
        c = model.coefficients(h, torsion_indices)
        s_all = torch.stack(
            [phase_vectors(d, c, bond_of_torsion, 1, PERIODICITIES)[0] for d in scan]
        )
    truth = torch.zeros(6, 2, dtype=torch.float64)
    truth[2, 0] = 1.5   # a three-fold barrier of 1.5
    truth[2, 1] = -0.4  # with a phase offset
    truth[5, 0] = 0.2   # and a small six-fold term
    target = (s_all * truth).sum(dim=(1, 2))

    # Detached because only the energy head is being fitted here; h_bond is a
    # fixed input to that fit and does not depend on the conformer by design.
    with torch.no_grad():
        h_bond = model.forward(h, torsion_indices, scan[0], bond_of_torsion, 1)[4]
    h_bond_scan = h_bond.expand(len(scan), -1)

    optimiser = torch.optim.Adam(model.energy.parameters(), lr=0.02)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=4000)
    for _ in range(4000):
        optimiser.zero_grad()
        loss = torch.mean((model.energy(h_bond_scan, s_all) - target) ** 2)
        loss.backward()
        optimiser.step()
        schedule.step()

    rmse = float(torch.sqrt(loss.detach()))
    with torch.no_grad():
        recovered = model.energy.torsion_parameters(h_bond)[0]

    assert rmse < 1e-4, f"could not fit its own hypothesis class, RMSE {rmse}"
    assert torch.allclose(recovered, truth, atol=1e-3), (
        f"recovered coefficients\n{recovered}\ndo not match the truth\n{truth}"
    )


# ---------------------------------------------------------------------------
# batching, which must not change a single number
# ---------------------------------------------------------------------------


def test_batched_phase_vectors_agree_with_one_at_a_time(rotor):
    """A whole scan in one call must equal the points computed separately.

    The batched path exists for speed, and a speed change that alters a result is
    a defect rather than an optimisation, so the test is equality to machine
    precision rather than closeness.
    """
    torsion_indices, bond_of_torsion = rotor
    torch.manual_seed(0)
    c = torch.randn(len(torsion_indices), dtype=torch.float64)
    scan = torch.stack([_rotor_deltas(float(phi)) for phi in torch.linspace(0, 345, 24)])

    batched = phase_vectors(scan, c, bond_of_torsion, 1, PERIODICITIES)
    one_at_a_time = torch.stack(
        [phase_vectors(d, c, bond_of_torsion, 1, PERIODICITIES) for d in scan]
    )
    assert batched.shape == (24, 1, 6, 2)
    assert torch.allclose(batched, one_at_a_time, atol=1e-14)


def test_batched_model_energy_agrees_with_one_at_a_time(rotor):
    """The same, end to end through the model."""
    torsion_indices, bond_of_torsion = rotor
    torch.manual_seed(0)
    model = TorsionModel(n_atom_features=8, torsion_hidden=(16, 16)).double()
    h = torch.randn(8, 8, dtype=torch.float64)
    scan = torch.stack([_rotor_deltas(float(phi)) for phi in torch.linspace(0, 345, 24)])

    with torch.no_grad():
        batched = model(h, torsion_indices, scan, bond_of_torsion, 1)[0]
        separate = torch.stack(
            [model(h, torsion_indices, d, bond_of_torsion, 1)[0] for d in scan]
        )
    assert batched.shape == (24, 1)
    assert torch.allclose(batched, separate, atol=1e-12)


def test_bond_representation_does_not_depend_on_the_conformer(rotor):
    """h_bond is a property of the molecule, and the whole parameter reading rests on it.

    If the bond representation moved with the geometry then k_n would change every
    time the molecule turned, and the Fourier coefficients would describe a
    different function at every scan point.
    """
    torsion_indices, bond_of_torsion = rotor
    torch.manual_seed(0)
    model = TorsionModel(n_atom_features=8, torsion_hidden=(16, 16)).double()
    h = torch.randn(8, 8, dtype=torch.float64)
    with torch.no_grad():
        a = model.bond_representation(h, torsion_indices, bond_of_torsion, 1)[0]
        b = model(h, torsion_indices, _rotor_deltas(137.0), bond_of_torsion, 1)[4]
    assert torch.allclose(a, b, atol=1e-14)


# ---------------------------------------------------------------------------
# predicting an arbitrary (bond, angle) from one geometry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("phi_degrees", [0.0, 17.0, 90.0, -123.0, 180.0])
def test_rotating_the_phase_analytically_equals_rotating_the_geometry(rotor, phi_degrees):
    """THE IDENTITY THE WHOLE PREDICTION RESTS ON.

    s_n(phi) = R(n phi) s_n(0) is exact only because turning a rotor adds phi to
    every torsion about it at once. If it were approximate, predicting a profile
    from one conformer would be an approximation too, and the error would grow
    with the angle rather than announcing itself.
    """
    torsion_indices, bond_of_torsion = rotor
    torch.manual_seed(0)
    c = torch.randn(len(torsion_indices), dtype=torch.float64)
    phi = torch.tensor([np.radians(phi_degrees)], dtype=torch.float64)

    s0 = phase_vectors(_rotor_deltas(0.0), c, bond_of_torsion, 1, PERIODICITIES)
    analytic = rotate_phase(s0, phi, PERIODICITIES)[0]
    recomputed = phase_vectors(
        _rotor_deltas(phi_degrees), c, bond_of_torsion, 1, PERIODICITIES
    )
    assert torch.allclose(analytic, recomputed, atol=1e-12)


def test_profile_from_one_geometry_equals_the_scan(rotor):
    """A whole profile predicted from a single conformer must equal the scan.

    Computed both ways: once by turning the rotor in the geometry and running the
    model at each of 24 angles, once by running the model at one geometry and
    rotating the phase analytically.
    """
    torsion_indices, bond_of_torsion = rotor
    torch.manual_seed(0)
    model = TorsionModel(n_atom_features=8, torsion_hidden=(16, 16)).double()
    h = torch.randn(8, 8, dtype=torch.float64)
    angles = torch.linspace(0, 345, 24, dtype=torch.float64)

    with torch.no_grad():
        from_one = model.profile(
            h, torsion_indices, _rotor_deltas(0.0), bond_of_torsion, 1,
            torch.deg2rad(angles),
        )
        from_scan = torch.stack([
            model(h, torsion_indices, _rotor_deltas(float(a)), bond_of_torsion, 1)[0]
            for a in angles
        ])
    assert from_one.shape == (24, 1)
    assert torch.allclose(from_one, from_scan, atol=1e-11)


def test_profile_answers_each_bond_separately(rotor):
    """(bond, angle) -> energy, with the bonds independent.

    Two rotors about different central bonds, turned by the same angle, must give
    different energies unless their chemistry is identical. A model that returned
    one number per angle regardless of the bond would pass every other test here.
    """
    torch.manual_seed(0)
    model = TorsionModel(n_atom_features=8, torsion_hidden=(16, 16)).double()
    h = torch.randn(10, 8, dtype=torch.float64)
    # two central bonds, 3-4 and 4-9, sharing atom 4
    indices = torch.tensor(
        [[i, 3, 4, l] for i in (0, 1, 2) for l in (5, 6, 7)]
        + [[i, 4, 9, l] for i in (3, 5) for l in (8,)],
        dtype=torch.long,
    )
    bond_of = torch.tensor([0] * 9 + [1] * 2, dtype=torch.long)
    deltas = torch.rand(len(indices), dtype=torch.float64) * 2 * np.pi
    phi = torch.deg2rad(torch.linspace(0, 350, 36, dtype=torch.float64))

    with torch.no_grad():
        energies = model.profile(h, indices, deltas, bond_of, 2, phi)
    assert energies.shape == (36, 2)
    assert not torch.allclose(energies[:, 0], energies[:, 1], atol=1e-6), (
        "two chemically different rotors returned the same profile"
    )


def test_profile_is_periodic_and_differentiable_in_phi(rotor):
    """Pose refinement turns rotors, so dE/dphi must exist and a full turn must close."""
    torsion_indices, bond_of_torsion = rotor
    torch.manual_seed(0)
    model = TorsionModel(n_atom_features=8, torsion_hidden=(16, 16)).double()
    h = torch.randn(8, 8, dtype=torch.float64)
    phi = torch.tensor([0.3, 0.3 + 2 * np.pi], dtype=torch.float64, requires_grad=True)

    energies = model.profile(
        h, torsion_indices, _rotor_deltas(0.0), bond_of_torsion, 1, phi
    )
    assert float(energies[0]) == pytest.approx(float(energies[1]), abs=1e-9)
    energies.sum().backward()
    assert phi.grad is not None and torch.isfinite(phi.grad).all()
