"""espaloma's stage-1 graph network as an encoder for the torsion phase head.

QUESTION. None on its own. This is an encoder adapter, so that the head can be
measured on two representations rather than one.

WHY THIS ENCODER. Three properties, and the third is the one that matters:

  it is fast          espaloma parametrises a system in under 100 microseconds
                      on a GPU (Wang et al., Chem. Sci. 13(41):12016, 2022, S5),
                      against 18.4 ms for one DPA-3.1-3M forward pass here.
  it is purpose-built espaloma 0.4.0 reaches 1.18 kcal/mol on TorsionNet500
                      (Clark et al., ChemRxiv 2026, Table 1) with a torsion
                      readout whose phase is FIXED to {0, pi}: "we do not fit
                      phases and periodicities of torsions as they are
                      discrete". Our head fits the phase continuously, so the
                      two differ in exactly one thing.
  it has no geometry  h is a function of the molecular graph alone. No
                      coordinates enter it.

The third kills a problem measured here on 2026-09-13: feeding the DPA3 head an
ETKDG conformer instead of the QM minimum cost 0.326 kcal/mol, 1.324 to 1.650,
with intervals that do not overlap. With a graph encoder there is no conformer
in the k_n path at all, so that degradation cannot happen, and Enamine REAL and
ZINC22 become usable from the SMILES with no conformer generation step.

WHAT IT TAKES. Symbols, bonds and bond orders, which is what a SMILES supplies.
Coordinates are accepted and ignored, so this is interchangeable with
`embed_torch` and the arms can share one training loop.

THE EMBEDDING IS 512 FEATURES, not DPA3's 128, so the Janossy network's first
layer is four times wider. That is a difference between the arms and is reported
rather than hidden.

COST. One graph build per molecule, cached, then one GNN forward.
"""
from __future__ import annotations

import numpy as np

BOND_TYPES = {1.0: "SINGLE", 1.5: "AROMATIC", 2.0: "DOUBLE", 3.0: "TRIPLE"}
DEFAULT_VALENCE = {"B": 3, "C": 4, "N": 3, "O": 2, "P": 3, "S": 2,
                   "F": 1, "Cl": 1, "Br": 1, "I": 1}


def rdkit_from_cache(symbols, bonds, orders):
    """An RDKit molecule in the ARCHIVE's atom order, charges assigned by valence.

    Identical in intent to `conformer_sensitivity.molecule_from_cache`, and
    duplicated here on purpose: that module imports the DPA3 stack, and this one
    has to run in an environment that cannot have deepmd in it. Going through
    the pool's SMILES string instead loses about a quarter of the molecules to
    quaternary nitrogens written without their charge.
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
        excess = int(round(sum(b.GetBondTypeAsDouble()
                               for b in atom.GetBonds()))) - default
        if excess > 0:
            if atom.GetSymbol() in ("N", "O"):
                atom.SetFormalCharge(excess)
            elif atom.GetSymbol() == "B":
                atom.SetFormalCharge(-excess)
        elif excess < 0 and atom.GetSymbol() in ("N", "O"):
            # UNDER-coordinated nitrogen and oxygen are negative, and leaving
            # them neutral makes RDKit call them radicals. The OpenFF Toolkit
            # then refuses the molecule outright, which cost 100 of 870
            # molecules on the first espaloma run, almost all of them azides
            # written as N=[N+]=[N] with the terminal charge missing. Restricted
            # to N and O: a three-valent carbon is as likely to be a real
            # radical as a carbanion, and guessing there would invent charges.
            atom.SetFormalCharge(excess)
    molecule = rw.GetMol()
    Chem.SanitizeMol(molecule)
    return molecule


def build_encoder(version: str = "0.3.2", stages: int = 1):
    """(stage-1 module, n_features). The released model, split at the readout.

    `esp.get_model` returns a live Module, so the GNN is a submodule that can be
    frozen or fine-tuned like any other. Only the first `stages` children are
    kept: the rest are espaloma's own parameter readouts, which are exactly what
    this experiment replaces.
    """
    import espaloma as esp
    import torch

    model = esp.get_model(version)
    stage1 = torch.nn.Sequential(*list(model.children())[:stages])
    width = None
    for parameter in reversed(list(stage1.parameters())):
        if parameter.dim() == 2:
            width = parameter.shape[0]
            break
    return stage1, width


def graph_of(symbols, bonds, orders):
    """The DGL heterograph espaloma reads. Conformer-independent, so build once."""
    import espaloma as esp
    from openff.toolkit import Molecule

    molecule = Molecule.from_rdkit(rdkit_from_cache(symbols, bonds, orders),
                                   allow_undefined_stereo=True)
    return esp.Graph(molecule).heterograph


def embed_graph(encoder, graph, n_frames: int, grad: bool):
    """(n_frames, n_atoms, n_features), the SAME embedding repeated per frame.

    Repeated rather than recomputed, and that is the point rather than a saving:
    the representation does not depend on which conformer is in hand, so every
    frame of a scan sees one set of coefficients by construction. The shape
    matches `embed_torch` so the training loop does not branch.
    """
    import torch

    context = torch.enable_grad() if grad else torch.no_grad()
    with context:
        out = encoder(graph)
        h = out.nodes["n1"].data["h"]
    return h.unsqueeze(0).expand(max(int(n_frames), 1), *h.shape)
