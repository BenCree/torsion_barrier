"""A torsion energy head that is linear in the phase, so its weights are parameters.

THE ARCHITECTURE, and why it is shaped this way.

    h_i                     atom embeddings from any encoder (DPA3, NAGL, ...)
    h_r_ijkl = NN([h_i:h_j:h_k:h_l]) + NN([h_l:h_k:h_j:h_i])     Janossy, symmetric
    c_ijkl   = NN(h_i + h_l)                                     torsion weight
    s_n      = sum_ijkl c_ijkl * [cos(n d_ijkl), sin(n d_ijkl)]  the phase, per bond
    E        = sum_n  k_n . s_n,      k = NN(h_r)                the energy

The last line is the choice that matters. s_n depends only on where the molecule
currently is; k_n depends only on what it is. The energy is one inner product
between the two, and because it is linear in s_n it expands to a six-term Fourier
series in the rotor angle with both the barrier height |k_n| and the phase
atan2(k_n_sin, k_n_cos) fitted. `TorsionEnergyHead(nonlinear=True)` gives up that
reading for a more flexible function and exists to measure what the linearity
costs.

WHAT k_n IS NOT. It is not a force field torsion parameter and must not be written
into one. Braun et al., Best Practices for Foundations in Molecular Simulations,
LiveCoMS 1(1):5957, 2019, section 3.5: proper torsions are where nonbonded
exclusions end, and the terminal atoms keep scaled nonbonded interactions, 1/1.2
on electrostatics and 1/2 on Lennard-Jones in AMBER. A QM rotation profile
therefore contains the explicit torsion term AND the changing 1-4 nonbonded terms.
Fitted against the total profile, which is what this head is for, k_n absorbs the
1-4 contribution; a force field that also computes 1-4 interactions would count it
twice, and nothing in the training loss would show it. A transferable parameter
has to be fitted against E_QM - E_MM,no-torsion instead, which is a different
target rather than a later step. `torsion_parameters` exists for inspection and
for the Fourier identity test, not as an export path.

Espaloma (Wang et al., Chem. Sci. 13(41):12016, 2022) reaches the same place from
the other side: it fits K_n for n = 1..6 at fixed phase phi_0 = 0, using a signed
K_n to stand in for phi_0 = pi. Fitting both components of k_n instead is fitting
the barrier height and the phase together, and doing it through s_n keeps the
result independent of how the atoms happen to be numbered.

The encoder is not imported here. The head takes atom embeddings and geometry and
returns an energy, so a pretrained DPA3 and a from-scratch graph network are
interchangeable behind it and can be compared on one axis.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

from openff.nagl.torsion.alpha import DEFAULT_PERIODICITIES


def _mlp(sizes: Sequence[int], activation=nn.SiLU) -> nn.Sequential:
    layers = []
    for index in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[index], sizes[index + 1]))
        if index < len(sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class JanossyTorsionPooling(nn.Module):
    """h_r for each torsion, symmetric under reversing it.

    Reversing i-j-k-l to l-k-j-i is the same torsion read the other way, so the
    representation must not change. Summing the network over both orderings
    enforces that by construction rather than by training against it, which is
    the Janossy pooling argument (Murphy et al., ICLR 2019) and the same device
    espaloma uses for bonds and angles.
    """

    def __init__(self, n_features: int, hidden: Sequence[int] = (128, 128)):
        super().__init__()
        self.n_features = n_features
        self.network = _mlp([4 * n_features, *hidden])
        self.n_outputs = hidden[-1]

    def forward(self, h: torch.Tensor, torsion_indices: torch.Tensor) -> torch.Tensor:
        """h: (n_atoms, n_features). torsion_indices: (n_torsions, 4). -> (n_torsions, n_outputs)."""
        forward = h[torsion_indices].reshape(len(torsion_indices), -1)
        reverse = h[torsion_indices.flip(-1)].reshape(len(torsion_indices), -1)
        return self.network(forward) + self.network(reverse)


class TorsionCoefficients(nn.Module):
    """c_ijkl, the weight each torsion carries into its bond's phase.

    Built from the two terminal atoms, which are the ones the rotation moves,
    and summed rather than concatenated so that i <-> l is invariant by
    construction for the same reason as above.
    """

    def __init__(self, n_features: int, hidden: Sequence[int] = (64,)):
        super().__init__()
        self.network = _mlp([n_features, *hidden, 1])

    def forward(self, h: torch.Tensor, torsion_indices: torch.Tensor) -> torch.Tensor:
        """-> (n_torsions,)."""
        terminal = h[torsion_indices[:, 0]] + h[torsion_indices[:, 3]]
        return self.network(terminal).squeeze(-1)


def phase_vectors(
    deltas: torch.Tensor,
    coefficients: torch.Tensor,
    bond_of_torsion: torch.Tensor,
    n_bonds: int,
    periodicities: Sequence[int] = DEFAULT_PERIODICITIES,
) -> torch.Tensor:
    """s_n for every bond, differentiably.

    Batched over any leading dimensions of `deltas`, so a whole torsion scan is
    one call. This matters more than it looks: c_ijkl and the bond representation
    do not depend on the conformer, so evaluating 24 scan points one at a time
    recomputes the conformer-independent half 24 times. Measured 2026-09-11, the
    per-point loop ran under 100 epochs in 100 seconds on 75 molecules.

    Parameters
    ----------
    deltas
        (..., n_torsions) dihedral angles in radians. Leading dimensions are
        treated as a batch, typically the scan points of one molecule.
    coefficients
        (n_torsions,) the c_ijkl. Broadcast across the batch, because the weight
        of a torsion is a property of the chemistry and not of the conformer.
    bond_of_torsion
        (n_torsions,) which central bond each torsion belongs to.
    n_bonds
        How many central bonds to accumulate into.

    Returns
    -------
    (..., n_bonds, n_periodicities, 2)
    """
    n = torch.as_tensor(periodicities, dtype=deltas.dtype, device=deltas.device)
    angles = deltas.unsqueeze(-1) * n  # (..., n_torsions, n_periodicities)
    weighted = coefficients.reshape(
        *(1,) * (deltas.dim() - 1), -1, 1, 1
    ) * torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)

    batch_shape = deltas.shape[:-1]
    s = torch.zeros(
        (*batch_shape, n_bonds, len(periodicities), 2),
        dtype=deltas.dtype,
        device=deltas.device,
    )
    index = bond_of_torsion.reshape(
        *(1,) * len(batch_shape), -1, 1, 1
    ).expand_as(weighted)
    return s.scatter_add(-3, index, weighted)


def alpha_from_phase_vectors(
    s: torch.Tensor, magnitude_tolerance: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(alpha, magnitude) from s, with NaN where the harmonic is degenerate.

    Reported for reading and for diagnosis. The energy is computed from s
    directly and never from alpha, because s is smooth everywhere while alpha
    has a branch cut and is undefined wherever a rotor's symmetry cancels it.
    """
    magnitude = torch.linalg.norm(s, dim=-1)
    alpha = torch.atan2(s[..., 1], s[..., 0])
    return alpha.masked_fill(magnitude <= magnitude_tolerance, float("nan")), magnitude


def rotate_phase(
    s: torch.Tensor,
    phi: torch.Tensor,
    periodicities: Sequence[int] = DEFAULT_PERIODICITIES,
) -> torch.Tensor:
    """s_n at a rotor turned by phi, in closed form from s_n at phi = 0.

    THE IDENTITY THIS RESTS ON. Turning a rotor by phi adds phi to EVERY torsion
    about that bond at once, so

        s_n(phi) = sum_ij c_ij [cos(n(d_ij + phi)), sin(n(d_ij + phi))]
                 = R(n phi) s_n(0)

    where R is the 2x2 rotation. The coefficients c_ij do not appear, because
    they are properties of the chemistry and not of where the rotor sits.

    What this buys is the whole point of the parametrisation: the energy at any
    angle follows from ONE geometry. A profile does not need 24 geometries, a
    scan does not need to be run to be predicted, and the torsion potential of a
    docked pose is available analytically in every rotor at once, with gradients,
    which is what a pose refinement needs.

    Parameters
    ----------
    s
        (..., n_periodicities, 2) the phase at phi = 0, from `phase_vectors`.
    phi
        (n_angles,) rotations in radians, or a scalar tensor.

    Returns
    -------
    (n_angles, ..., n_periodicities, 2)
    """
    n = torch.as_tensor(periodicities, dtype=s.dtype, device=s.device)
    angles = phi.reshape(-1, *([1] * (s.dim() - 1))) * n.reshape(
        *([1] * (s.dim() - 1)), -1
    )
    cos, sin = torch.cos(angles), torch.sin(angles)
    x, y = s[..., 0], s[..., 1]
    return torch.stack([cos * x - sin * y, sin * x + cos * y], dim=-1)


class TorsionEnergyHead(nn.Module):
    """E for each rotor, from its representation and its phase.

    `nonlinear=False`, the default, gives E = sum_n k_n . s_n with k predicted
    from h_r, which is a Fourier series whose coefficients are transferable
    torsion parameters. `nonlinear=True` replaces it with NN([h_r : s]) and is
    the arm that measures what the linear form gives up.
    """

    def __init__(
        self,
        n_bond_features: int,
        periodicities: Sequence[int] = DEFAULT_PERIODICITIES,
        hidden: Sequence[int] = (128,),
        nonlinear: bool = False,
    ):
        super().__init__()
        self.periodicities = tuple(periodicities)
        self.nonlinear = nonlinear
        n_coefficients = 2 * len(self.periodicities)
        if nonlinear:
            self.network = _mlp([n_bond_features + n_coefficients, *hidden, 1])
        else:
            self.network = _mlp([n_bond_features, *hidden, n_coefficients])

    def forward(self, h_bond: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """h_bond: (n_bonds, F). s: (..., n_bonds, n_periodicities, 2). -> (..., n_bonds).

        h_bond carries no batch dimension on purpose: the bond representation is a
        property of the molecule, so it broadcasts across every conformer of that
        molecule rather than being recomputed for each.
        """
        flat = s.reshape(*s.shape[:-2], -1)
        if self.nonlinear:
            expanded = h_bond.expand(*flat.shape[:-1], h_bond.shape[-1])
            return self.network(torch.cat([expanded, flat], dim=-1)).squeeze(-1)
        return (self.network(h_bond) * flat).sum(-1)

    def torsion_parameters(self, h_bond: torch.Tensor) -> torch.Tensor:
        """The k_n, shaped (n_bonds, n_periodicities, 2). For inspection, not export.

        Column 0 multiplies cos(n * delta) and column 1 sin(n * delta), so the
        barrier height is |k_n| and the phase atan2(k_n1, k_n0). Fitted against a
        total QM profile these absorb the 1-4 nonbonded contribution and are NOT
        SMIRNOFF torsion parameters; see the module docstring. Raises for the
        nonlinear arm, which has no such reading.
        """
        if self.nonlinear:
            raise NotImplementedError(
                "the nonlinear head has no Fourier coefficients to read off; "
                "it was built to measure what the linear form costs"
            )
        return self.network(h_bond).reshape(len(h_bond), len(self.periodicities), 2)


class TorsionModel(nn.Module):
    """Encoder-agnostic assembly of the four pieces.

    Takes atom embeddings from whatever produced them. With DPA-3.1-3M that is a
    pretrained 3M-parameter atomistic model being fine-tuned; with NAGL it is a
    graph network trained from scratch. The head does not know which, which is
    what lets the two be compared on one axis.
    """

    def __init__(
        self,
        n_atom_features: int,
        periodicities: Sequence[int] = DEFAULT_PERIODICITIES,
        torsion_hidden: Sequence[int] = (128, 128),
        nonlinear: bool = False,
    ):
        super().__init__()
        self.periodicities = tuple(periodicities)
        self.pooling = JanossyTorsionPooling(n_atom_features, torsion_hidden)
        self.coefficients = TorsionCoefficients(n_atom_features)
        self.energy = TorsionEnergyHead(
            self.pooling.n_outputs, periodicities, nonlinear=nonlinear
        )

    def bond_representation(
        self, h: torch.Tensor, torsion_indices: torch.Tensor,
        bond_of_torsion: torch.Tensor, n_bonds: int,
    ):
        """(h_bond, c) for one molecule. Neither depends on the conformer.

        Split out so a scan can compute them once. The bond representation must
        not depend on the geometry or the coefficients k_n would change every time
        the molecule moved; pooling over the torsions of a bond with no geometry
        in it is what keeps that true.
        """
        h_torsion = self.pooling(h, torsion_indices)
        c = self.coefficients(h, torsion_indices)
        h_bond = torch.zeros(
            (n_bonds, h_torsion.shape[1]), dtype=h_torsion.dtype, device=h_torsion.device
        )
        h_bond = h_bond.scatter_add(
            0, bond_of_torsion.view(-1, 1).expand_as(h_torsion), h_torsion
        )
        return h_bond, c

    def forward(
        self,
        h: torch.Tensor,
        torsion_indices: torch.Tensor,
        deltas: torch.Tensor,
        bond_of_torsion: torch.Tensor,
        n_bonds: int,
    ):
        """-> (energy per bond, s, alpha, magnitude, h_bond).

        `deltas` may carry leading batch dimensions, in which case every returned
        array except h_bond carries them too.
        """
        h_bond, c = self.bond_representation(
            h, torsion_indices, bond_of_torsion, n_bonds
        )
        s = phase_vectors(
            deltas, c, bond_of_torsion, n_bonds, periodicities=self.periodicities
        )
        alpha, magnitude = alpha_from_phase_vectors(s)
        return self.energy(h_bond, s), s, alpha, magnitude, h_bond

    def molecule_energy(self, *args, **kwargs) -> torch.Tensor:
        """The total torsion energy, summed over rotors."""
        return self.forward(*args, **kwargs)[0].sum()

    def profile(
        self,
        h: torch.Tensor,
        torsion_indices: torch.Tensor,
        deltas: torch.Tensor,
        bond_of_torsion: torch.Tensor,
        n_bonds: int,
        phi: torch.Tensor,
    ) -> torch.Tensor:
        """E for every bond at every angle, from ONE geometry. -> (n_angles, n_bonds).

        This is the function the model is for: given a molecule, a bond and an
        angle, return the energy. `deltas` are the dihedrals in whatever single
        conformer is to hand, and `phi` is how far each rotor is then turned from
        it, so `phi = 0` returns the energy of the conformer as given.

        Each bond is answered independently and at every angle at once, which is
        what makes the output a torsion potential rather than a single number.
        """
        h_bond, c = self.bond_representation(
            h, torsion_indices, bond_of_torsion, n_bonds
        )
        s0 = phase_vectors(
            deltas, c, bond_of_torsion, n_bonds, periodicities=self.periodicities
        )
        s = rotate_phase(s0, phi, self.periodicities)
        return self.energy(h_bond, s)
