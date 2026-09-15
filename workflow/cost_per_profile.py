"""What the phase parametrisation buys, measured rather than asserted.

The accuracy result is settled and negative: the head is 2.3x worse than the
pretrained potential it is built on. What it does that the potential cannot is
answer (bond, angle) -> energy in closed form. This measures that difference in
the three forms it takes.

  a whole profile from ONE geometry. E(phi) = sum_n k_n . R(n phi) s_n(0), so 24
    points cost one encoder pass and 24 cheap rotations. The potential needs 24
    geometries and 24 forward passes.

  an ARBITRARY angle with no geometry. To ask the potential for 90.5 degrees you
    must first build that conformer. The head evaluates a rotation matrix.

  ANALYTIC dE/dphi. Torsion-space refinement needs the gradient; the potential
    gives forces in Cartesian space and the chain rule to phi has to be assembled.

Reported per profile on a named device, timed after a warm-up, so it is
comparable to the seconds_per_profile already in the zero-shot tables.

THE ENCODER PASS IS INCLUDED. An earlier version of this fed the head random
embeddings and timed the head alone, which understated the cost by a factor of
about 27 and would have been quoted as a speed-up. The head cannot run without
embeddings, so one DPA3 forward pass on the reference conformer is part of the
cost of the first angle and of no subsequent angle, which is the whole point.
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


def main():
    import torch
    from openff.nagl.torsion import TorsionModel

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torsiondrives", required=True)
    parser.add_argument("--n-molecules", type=int, default=50)
    parser.add_argument("--angles", type=int, default=24)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", default="/home/ben/models/DPA-3.1-3M/DPA-3.1-3M.pt")
    parser.add_argument("--config", default="/home/ben/models/DPA-3.1-3M/input_pretrain.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from fit_torsion_head import enumerate_torsions
    from openff.nagl.utils._tensors import calculate_dihedrals

    cache = np.load(args.torsiondrives, allow_pickle=False)
    entries = json.loads(str(cache["meta"]))[: args.n_molecules]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    prepared = []
    for entry in entries:
        bonds = cache[f"bonds::{entry['key']}"]
        groups = enumerate_torsions(bonds, entry["n_atoms"])
        driven = entry["driven_dihedral"]
        key = (min(driven[1], driven[2]), max(driven[1], driven[2]))
        if key not in groups:
            continue
        indices = np.asarray(groups[key], dtype=int)
        xyz = np.asarray(cache[f"xyz::{entry['key']}"], dtype=float)[0]
        deltas = calculate_dihedrals(
            xyz[indices[:, 0]], xyz[indices[:, 1]], xyz[indices[:, 2]], xyz[indices[:, 3]]
        )
        prepared.append({
            "symbols": entry["symbols"],
            "reference_xyz": xyz,
            "h": None,
            "indices": torch.tensor(indices, dtype=torch.long, device=device),
            "deltas": torch.tensor(np.asarray(deltas), dtype=torch.float64, device=device),
            "bond_of": torch.zeros(len(indices), dtype=torch.long, device=device),
        })
    # the real encoder, not random numbers
    from dpa3_embeddings import build_descriptor
    from finetune_torsion import embed_torch

    descriptor, type_map = build_descriptor(args.config, args.checkpoint, "SPICE2")
    descriptor.eval()
    model = TorsionModel(n_atom_features=descriptor.get_dim_out()).double().to(device).eval()
    phi = torch.deg2rad(torch.linspace(0, 360, args.angles + 1, dtype=torch.float64,
                                       device=device)[:-1])

    def run(n_angles, with_encoder=True):
        angles = phi[:n_angles]
        with torch.no_grad():
            for m in prepared:
                h = (embed_torch(descriptor, type_map, m["symbols"], m["reference_xyz"], False)[0].double()
                     if with_encoder else m["cached_h"])
                model.profile(h, m["indices"], m["deltas"], m["bond_of"], 1, angles)
        if device.type == "cuda":
            torch.cuda.synchronize()

    with torch.no_grad():
        for m in prepared:
            m["cached_h"] = embed_torch(
                descriptor, type_map, m["symbols"], m["reference_xyz"], False
            )[0].double()
    run(args.angles); run(args.angles, with_encoder=False)      # warm up both paths
    rows = []
    for label, with_encoder in (("encoder + head", True), ("head only", False)):
        for n_angles in (1, 24, 360, 3600):
            started = time.time()
            repeats = 3
            for _ in range(repeats):
                run(n_angles, with_encoder)
            seconds = (time.time() - started) / repeats / len(prepared)
            rows.append({"path": label, "n_angles": n_angles, "seconds_per_molecule": seconds})
            print(f"  {label:14s} {n_angles:5d} angles: {seconds*1000:8.3f} ms per molecule "
                  f"({seconds/n_angles*1e6:8.2f} us per angle)")

    # gradient in the rotor angle, which is what a torsion-space refinement needs
    m = prepared[0]
    angles = phi.clone().requires_grad_(True)
    started = time.time()
    for _ in range(10):
        e = model.profile(m["cached_h"], m["indices"], m["deltas"], m["bond_of"], 1, angles).sum()
        e.backward()
        angles.grad = None
    grad_seconds = (time.time() - started) / 10
    print(f"  head, analytic dE/dphi at 24 angles: {grad_seconds*1000:.3f} ms")

    import pandas as pd
    frame = pd.DataFrame(rows)
    frame["device"] = str(device)
    frame["grad_seconds_24"] = grad_seconds
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(f"\n  -> {args.out}")


if __name__ == "__main__":
    main()
