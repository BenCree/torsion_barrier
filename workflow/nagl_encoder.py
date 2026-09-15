"""openff-nagl's charge-model graph network as an encoder for the torsion head.

WHY THIS ARM EXISTS, and it is expected to lose. NAGL's GNN was trained to
predict AM1-BCC partial charges, which are an electrostatic property. Once the
target becomes E_QM - E_MM,no-torsion, the electrostatics have been SUBTRACTED
OUT and the residual is the non-electrostatic part: hyperconjugation, sterics,
and the stereochemical asymmetry the phase census found. That is precisely what
NAGL's embedding was trained away from, while espaloma's was trained on MM
valence parameters including torsions. So the prediction registered before
running it is that NAGL loses to espaloma on the residual target.

It is worth running anyway because that is a falsifiable prediction about what
the residual contains. If NAGL wins, the reasoning above is wrong and it is
better to know before any of it is written up.

PRACTICALLY IT IS THE SAFER DEPENDENCY. openff-nagl 0.5.5 imports and loads a
model with no DGL installed, checked in the openfe2 environment; espaloma
requires DGL, which is effectively unmaintained and pins the torch and CUDA
versions. For work meant to be installable by somebody else that matters.

WHAT IT TAKES. Symbols, bonds and bond orders, which is what a SMILES supplies.
Coordinates are accepted and ignored: like espaloma's, this embedding is a
function of the molecular graph alone, so k_n cannot depend on the conformer.

COST. One graph build per molecule, cached, then one GNN forward.
"""
from __future__ import annotations

from espaloma_encoder import rdkit_from_cache  # noqa: F401  (re-exported)


_MODEL = {}


def _model(version: str = "openff-gnn-am1bcc-1.0.0.pt"):
    """The released model, loaded once.

    Kept whole rather than split apart, because the featurisation a NAGL model
    expects is stored in its own config and `_convert_to_nagl_molecule` applies
    it. Hand-building the feature list instead is how a silently wrong input
    reaches a network that will happily return numbers for it.
    """
    if version not in _MODEL:
        from openff.nagl import GNNModel
        from openff.nagl_models import get_models_by_type

        path = next(p for p in get_models_by_type("am1bcc") if p.name == version)
        _MODEL[version] = GNNModel.load(path, eval_mode=True)
    return _MODEL[version]


def build_encoder(version: str = "openff-gnn-am1bcc-1.0.0.pt", stages: int = 1):
    """(convolution module, n_features).

    Only the convolution is kept: the readout after it emits charges, which is
    exactly the part this experiment replaces.
    """
    model = _model(version)
    convolution = model.convolution_module
    width = None
    for parameter in reversed(list(convolution.parameters())):
        if parameter.dim() == 2:
            width = parameter.shape[0]
            break
    return convolution, width


def graph_of(symbols, bonds, orders, version: str = "openff-gnn-am1bcc-1.0.0.pt"):
    """The featurised molecule NAGL reads. Conformer-independent, so build once.

    NAGL dispatches to DGLMolecule where DGL is installed and to its own
    GraphMolecule where it is not, so this works in an environment with neither
    espaloma nor DGL in it, which is the practical reason this arm exists.
    """
    from openff.toolkit import Molecule

    molecule = Molecule.from_rdkit(rdkit_from_cache(symbols, bonds, orders),
                                   allow_undefined_stereo=True)
    return _model(version)._convert_to_nagl_molecule(molecule)


def embed_graph(encoder, graph, n_frames: int, grad: bool):
    """(n_frames, n_atoms, n_features), the SAME embedding repeated per frame."""
    import torch

    context = torch.enable_grad() if grad else torch.no_grad()
    with context:
        encoder(graph)
        h = graph.graph.ndata["h"]
    return h.unsqueeze(0).expand(max(int(n_frames), 1), *h.shape)
