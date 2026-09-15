"""A permutation-invariant phase coordinate for the conformation of a rotor.

WHAT THIS IS FOR. A substituted C-C bond carries up to nine proper torsions
i-j-k-l sharing the central bond j-k. Reporting "the" dihedral of that bond means
picking one of the nine, and the answer then depends on how the atoms happen to
be labelled. TorsionNet (Rai et al., JCIM 62(4):785, 2022) predicts a profile as
a function of one chosen dihedral and inherits that choice. Espaloma (Wang et
al., Chem. Sci. 13(41):12016, 2022) avoids the problem by refusing to fit phases
at all, fixing phi_0 = 0 and using signed barrier heights across periodicities
n = 1..6.

This module takes the third route: combine all torsions about a bond into a
single phase that does not depend on the labelling, so it can be handed to a
network as "where on the rotor this conformer sits".

    s_n      = sum_ij  c_ij * [cos(n * delta_ij), sin(n * delta_ij)]
    alpha_n  = atan2(s_n[1], s_n[0])

WHY THE PERIODICITY IS NOT OPTIONAL. Computing s at n = 1 alone, which earlier
drafts of this work did, makes alpha undefined for any three-fold rotor. Three
torsions 120 degrees apart with equal weights sum to exactly zero, so every
methyl and every CF3 is degenerate at n = 1 and again at n = 2, and only at n = 3
does the sum become maximal with alpha = 3 * phi. Espaloma's n = 1..6 expansion
is what makes a phase well defined at all; keeping the phase while dropping the
periodicities is the one combination that cannot work.

DEGENERACY IS REPORTED, NOT HIDDEN. When |s_n| is near zero the phase carries no
information, and floating point will not give exactly zero, so a naive atan2
returns a confident angle for an undefined quantity. `alpha` is set to NaN below
`magnitude_tolerance` and `magnitude` is always returned so the caller can see
which harmonics are informative for a given rotor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

DEFAULT_PERIODICITIES: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)
DEFAULT_MAGNITUDE_TOLERANCE: float = 1e-8


@dataclass(frozen=True)
class TorsionPhase:
    """The phase coordinate of one rotor, at every periodicity requested.

    Attributes
    ----------
    central_bond
        The (j, k) atom indices of the rotor, sorted, so the key does not depend
        on which direction the bond was read in.
    periodicities
        The n values, in the order the arrays are indexed.
    s
        Shape (n_periodicities, 2). The weighted circular sum at each n. This is
        the fixed-width quantity a network consumes.
    alpha
        Shape (n_periodicities,). The phase at each n, in radians, or NaN where
        the corresponding magnitude is below tolerance.
    magnitude
        Shape (n_periodicities,). |s_n|. Small means that harmonic carries no
        information for this rotor, which is a fact about its symmetry.
    torsion_indices
        The (i, j, k, l) tuples that were combined, in the order used.
    deltas
        The dihedral angle of each of those torsions, in radians.
    """

    central_bond: Tuple[int, int]
    periodicities: Tuple[int, ...]
    s: np.ndarray
    alpha: np.ndarray
    magnitude: np.ndarray
    torsion_indices: Tuple[Tuple[int, int, int, int], ...]
    deltas: np.ndarray

    def informative_periodicities(self, tolerance: float = 1e-6) -> Tuple[int, ...]:
        """The n whose magnitude clears `tolerance`, so whose phase is usable."""
        return tuple(
            int(n) for n, m in zip(self.periodicities, self.magnitude) if m > tolerance
        )


def alpha_from_deltas(
    deltas: Sequence[float],
    weights: Optional[Sequence[float]] = None,
    periodicities: Iterable[int] = DEFAULT_PERIODICITIES,
    magnitude_tolerance: float = DEFAULT_MAGNITUDE_TOLERANCE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine the dihedrals of one rotor into a phase at each periodicity.

    Parameters
    ----------
    deltas
        The dihedral angles of every torsion sharing the central bond, radians.
    weights
        The c_ij. Uniform if not given. In a trained model these come from the
        network; uniform is the right default for testing the geometry alone.
    periodicities
        The n values to evaluate.
    magnitude_tolerance
        Below this |s_n|, the phase is undefined and returned as NaN.

    Returns
    -------
    s, alpha, magnitude
    """
    deltas = np.asarray(deltas, dtype=float)
    if deltas.ndim != 1:
        raise ValueError(f"deltas must be one dimensional, got shape {deltas.shape}")
    if weights is None:
        weights = np.ones_like(deltas)
    weights = np.asarray(weights, dtype=float)
    if weights.shape != deltas.shape:
        raise ValueError(
            f"weights shape {weights.shape} does not match deltas {deltas.shape}"
        )

    periodicities = tuple(int(n) for n in periodicities)
    s = np.empty((len(periodicities), 2), dtype=float)
    for row, n in enumerate(periodicities):
        angles = n * deltas
        s[row, 0] = float(np.sum(weights * np.cos(angles)))
        s[row, 1] = float(np.sum(weights * np.sin(angles)))

    magnitude = np.linalg.norm(s, axis=1)
    # atan2(sin, cos). Passing them the other way round reflects the phase to
    # pi/2 - alpha and inverts its response to rotating the rotor.
    alpha = np.arctan2(s[:, 1], s[:, 0])
    alpha[magnitude <= magnitude_tolerance] = np.nan
    return s, alpha, magnitude


def torsions_about_bond(molecule, j: int, k: int):
    """Every proper torsion i-j-k-l of `molecule` whose central bond is (j, k).

    The returned tuples are oriented so the central pair reads (j, k) in that
    order, which is what makes the dihedral signs of the group mutually
    consistent. Ordering within the group is by (i, l) so the result does not
    depend on the order `molecule.propers` happens to yield.
    """
    found = []
    for proper in molecule.propers:
        indices = tuple(atom.molecule_atom_index for atom in proper)
        if (indices[1], indices[2]) == (j, k):
            found.append(indices)
        elif (indices[1], indices[2]) == (k, j):
            found.append(indices[::-1])
    return tuple(sorted(found, key=lambda t: (t[0], t[3])))


def alpha_from_molecule(
    molecule,
    conformer_index: int = 0,
    weights: Optional[Dict[Tuple[int, int], Sequence[float]]] = None,
    periodicities: Iterable[int] = DEFAULT_PERIODICITIES,
    magnitude_tolerance: float = DEFAULT_MAGNITUDE_TOLERANCE,
) -> Dict[Tuple[int, int], TorsionPhase]:
    """The phase coordinate of every rotor in a conformer.

    Parameters
    ----------
    molecule
        An `openff.toolkit.Molecule` carrying at least one conformer.
    conformer_index
        Which conformer to read.
    weights
        Optional {(j, k): c_ij} keyed by sorted central bond. Uniform if absent.

    Returns
    -------
    {(j, k): TorsionPhase} keyed by sorted central bond index pair. Bonds with no
    proper torsion, such as a terminal bond, are absent rather than present with
    a zero phase, so a caller cannot mistake "no rotor" for "phase of zero".
    """
    from openff.nagl.utils._tensors import calculate_dihedrals

    if not molecule.conformers:
        raise ValueError("molecule has no conformers, so it has no torsion phase")
    xyz = np.asarray(
        molecule.conformers[conformer_index].m_as("angstrom"), dtype=float
    )

    central_bonds = []
    for bond in molecule.bonds:
        j, k = bond.atom1_index, bond.atom2_index
        key = (min(j, k), max(j, k))
        if key not in central_bonds:
            central_bonds.append(key)

    phases: Dict[Tuple[int, int], TorsionPhase] = {}
    for key in central_bonds:
        j, k = key
        indices = torsions_about_bond(molecule, j, k)
        if not indices:
            continue
        idx = np.asarray(indices, dtype=int)
        deltas = calculate_dihedrals(
            xyz[idx[:, 0]], xyz[idx[:, 1]], xyz[idx[:, 2]], xyz[idx[:, 3]]
        )
        deltas = np.asarray(deltas, dtype=float)
        w = None if weights is None else weights.get(key)
        s, alpha, magnitude = alpha_from_deltas(
            deltas,
            weights=w,
            periodicities=periodicities,
            magnitude_tolerance=magnitude_tolerance,
        )
        phases[key] = TorsionPhase(
            central_bond=key,
            periodicities=tuple(int(n) for n in periodicities),
            s=s,
            alpha=alpha,
            magnitude=magnitude,
            torsion_indices=indices,
            deltas=deltas,
        )
    return phases
