"""Load the torsion modules from the openff-nagl checkout without importing the package.

WHY THIS EXISTS. `openff/nagl/__init__.py` imports pytorch_lightning, and the
`dpa3` environment holds DeePMD-kit and PyTorch and has no reason to carry a
training framework it will not use. The three modules actually needed,
`utils/_tensors.py`, `torsion/alpha.py` and `torsion/model.py`, depend on nothing
beyond numpy and torch except one constant that `model.py` takes from `alpha.py`.

So the parent packages are registered as empty modules first and the three files
are executed directly. The alternative, installing the whole nagl stack into the
encoder's environment, would pull in a second copy of PyTorch Lightning and
openff-toolkit to make one import of a six-element tuple work.

The files themselves are unchanged and are the same ones the 70 library tests run
against, which is the point: this loads the tested code rather than a copy of it.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import types


def _default_root():
    """Where the torsion modules live, checked in order.

    On paros they sit in the openff-nagl checkout. On a cluster node that
    checkout does not exist, so the modules are staged beside the scripts and
    found either by NAGL_TORSION_ROOT or as a `nagl/` directory next to this
    file. Searching rather than hardcoding is what stops a staged run from
    failing on the node with a path that only exists on the workstation.
    """
    candidates = [
        os.environ.get("NAGL_TORSION_ROOT"),
        pathlib.Path(__file__).parent / "nagl",
        "/home/ben/code/openff-nagl",
    ]
    for candidate in candidates:
        if candidate and pathlib.Path(candidate).joinpath(
            "openff/nagl/torsion/model.py"
        ).is_file():
            return pathlib.Path(candidate)
    return pathlib.Path("/home/ben/code/openff-nagl")


NAGL_ROOT = _default_root()


def load(root: pathlib.Path = NAGL_ROOT):
    """Return (model_module, tensors_module). Idempotent."""
    if "openff.nagl.torsion.model" in sys.modules:
        return sys.modules["openff.nagl.torsion.model"], sys.modules["openff.nagl.utils._tensors"]

    if not root.joinpath("openff/nagl/torsion/model.py").is_file():
        raise SystemExit(
            f"torsion modules not found under {root}. Set NAGL_TORSION_ROOT or "
            f"stage them as a 'nagl' directory beside the scripts."
        )

    # IF A REAL openff-nagl IS INSTALLED, EXTEND IT, NEVER REPLACE IT.
    # Fabricating the namespace unconditionally shadowed the installed package
    # three separate ways in one session: openff-toolkit could not read
    # `openff.nagl.__version__` when it built its toolkit registry, the NAGL
    # encoder could not import `GNNModel`, and Sage 2.3.0's NAGLCharges handler
    # could not assign a charge, each failing somewhere far from here. The
    # loader exists because the CHECKOUT's `__init__` imports pytorch_lightning;
    # an installed openff-nagl 0.5.5 does not, so where one is present it is
    # imported and its search path is widened to reach the torsion modules.
    real = None
    try:
        import openff.nagl as real  # noqa: F401
        if not (getattr(real, "__file__", None) or hasattr(real, "__version__")):
            real = None
    except Exception:                                    # noqa: BLE001
        real = None
    if real is not None:
        extra = str(root / "openff/nagl")
        paths = list(getattr(real, "__path__", []))
        if extra not in paths:
            try:
                real.__path__.append(extra)
            except AttributeError:
                real.__path__ = [*paths, extra]

    for name in ("openff.nagl", "openff.nagl.torsion", "openff.nagl.utils"):
        if name == "openff.nagl" and real is not None:
            continue
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(root / name.replace(".", "/"))]
            if name == "openff.nagl":
                # openff-toolkit builds its GLOBAL_TOOLKIT_REGISTRY at import,
                # and its NAGL wrapper decides NAGL "is available" from whether
                # `openff.nagl` imports, then reads `openff.nagl.__version__`.
                # This synthetic package satisfied the first and not the second,
                # so importing openff.toolkit after this loader raised
                # "cannot import name '__version__' from 'openff.nagl' (unknown
                # location)" and took the whole espaloma arm down. The value
                # says what it is rather than impersonating a release: the
                # toolkit only stores it, and nothing here can assign charges.
                module.__version__ = "0+torsion-modules-loaded-by-path"
            sys.modules[name] = module

    def _exec(name, relative):
        path = root / relative
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    _exec("openff.nagl.torsion.alpha", "openff/nagl/torsion/alpha.py")
    tensors = _exec("openff.nagl.utils._tensors", "openff/nagl/utils/_tensors.py")
    model = _exec("openff.nagl.torsion.model", "openff/nagl/torsion/model.py")

    # The package's own __init__ re-exports these; do the same so callers can
    # write `from openff.nagl.torsion import TorsionModel` exactly as they would
    # against the installed package.
    torsion = sys.modules["openff.nagl.torsion"]
    for attribute in ("TorsionModel", "TorsionEnergyHead", "JanossyTorsionPooling",
                      "TorsionCoefficients", "phase_vectors", "alpha_from_phase_vectors"):
        setattr(torsion, attribute, getattr(model, attribute))
    alpha = sys.modules["openff.nagl.torsion.alpha"]
    for attribute in ("alpha_from_deltas", "alpha_from_molecule", "torsions_about_bond",
                      "TorsionPhase", "DEFAULT_PERIODICITIES"):
        setattr(torsion, attribute, getattr(alpha, attribute))
    return model, tensors
