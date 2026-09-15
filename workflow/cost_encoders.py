"""What one molecule's whole torsional description costs, by stack.

QUESTION. For every rotatable bond of one molecule, at 24 angles, and again
when 24 different conformers must be scored: what does each stack pay? The
answer is not one number because the stacks scale differently in two separate
places, so both are timed apart.

  parameter pass   produces k_n or K_n for every bond. Paid ONCE per molecule by
                   a graph encoder, because its representation does not read the
                   geometry, and once per CONFORMER by an atomistic descriptor,
                   because its representation does. That difference is the whole
                   comparison and it does not show up at one conformer.
  query            evaluates the profile at N angles from those coefficients.
                   Pure arithmetic on a Fourier series, no framework, so it is
                   the same for every stack and is timed to show it is
                   negligible rather than assumed to be.

THE STACKS.
  dpa3              DPA-3.1-3M descriptor, 128 features, then this head.
  espaloma          espaloma 0.3.2 stage 1, 512 features, then this head.
  espaloma_native   espaloma 0.3.2 in full, its own Janossy readout emitting
                    K_n at a phase fixed to {0, pi}. The thing we would have to
                    beat, timed in its own environment.

They cannot share a process: espaloma needs DGL and DPA3 needs deepmd. So this
is one invocation per stack under its own interpreter, and the tables are merged
afterwards rather than compared across a boundary that was never measured.

CONTROLS. `data/cost_per_profile_cuda.csv` for the dpa3 stack, which this must
reproduce; and 1, 24, 360 and 3,600 angles on the same molecules, because flat
in angles is a prediction of the closed-form rotation and not a fact.

UNITS. Seconds per molecule. The device is written into every row.

COST. Minutes.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import time

import numpy as np


def timed(call, repeats, warmup=3, sync=None):
    for _ in range(warmup):
        call()
    if sync:
        sync()
    started = time.perf_counter()
    for _ in range(repeats):
        call()
    if sync:
        sync()
    return (time.perf_counter() - started) / repeats


def query_seconds(n_bonds, n_angles, n_periodicities=6, repeats=200):
    """Evaluating the Fourier series, in numpy, with no framework at all.

    This is what a precomputed table costs to read back, and it is the same
    arithmetic whichever stack produced the coefficients.
    """
    k = np.random.default_rng(0).normal(size=(n_bonds, n_periodicities, 2))
    phi = np.linspace(-np.pi, np.pi, n_angles, endpoint=False)
    n = np.arange(1, n_periodicities + 1)

    def once():
        angles = phi[:, None] * n[None, :]
        basis = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
        return np.einsum("bnc,anc->ab", k, basis)

    return timed(once, repeats, warmup=2)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stack", required=True,
                   choices=("dpa3", "espaloma", "espaloma_native"))
    p.add_argument("--torsiondrives", required=True)
    p.add_argument("--n-molecules", type=int, default=50)
    p.add_argument("--device", default="cuda")
    p.add_argument("--angles", default="1,24,360,3600")
    p.add_argument("--conformers", default="1,24")
    p.add_argument("--checkpoint", default="/home/ben/models/DPA-3.1-3M/DPA-3.1-3M.pt")
    p.add_argument("--config", default="/home/ben/models/DPA-3.1-3M/input_pretrain.json")
    p.add_argument("--espaloma-version", default="0.3.2")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    import torch

    device = torch.device(args.device)
    sync = torch.cuda.synchronize if device.type == "cuda" else None

    cache = np.load(args.torsiondrives, allow_pickle=False)
    entries = json.loads(str(cache["meta"]))[:args.n_molecules]

    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from fit_torsion_head import enumerate_torsions

    rows, skipped = [], []
    if args.stack == "dpa3":
        from finetune_torsion import embed_torch
        from dpa3_embeddings import build_descriptor
        from openff.nagl.torsion import TorsionModel, phase_vectors

        descriptor, type_map = build_descriptor(args.config, args.checkpoint, "SPICE2")
        head = TorsionModel(n_atom_features=descriptor.get_dim_out()).double().to(device).eval()
    elif args.stack in ("espaloma", "espaloma_native"):
        from espaloma_encoder import build_encoder, graph_of, embed_graph
        from openff.nagl.torsion import TorsionModel, phase_vectors
        import espaloma as esp

        if args.stack == "espaloma":
            encoder, width = build_encoder(args.espaloma_version)
            encoder = encoder.to(device).eval()
            head = TorsionModel(n_atom_features=width).double().to(device).eval()
        else:
            native = esp.get_model(args.espaloma_version).to(device).eval()

    for entry in entries:
        key = entry["key"]
        try:
            bonds = cache[f"bonds::{key}"]
            groups = enumerate_torsions(bonds, entry["n_atoms"])
            if not groups:
                raise ValueError("no proper torsion")
            n_bonds = len(groups)
            keys = sorted(groups)
            indices, bond_of = [], []
            for slot, bond in enumerate(keys):
                for torsion in groups[bond]:
                    indices.append(torsion)
                    bond_of.append(slot)
            xyz = np.asarray(cache[f"xyz::{key}"], dtype=float)

            if args.stack == "dpa3":
                idx = torch.tensor(indices, dtype=torch.long, device=device)
                bof = torch.tensor(bond_of, dtype=torch.long, device=device)

                def parameters():
                    h = embed_torch(descriptor, type_map, entry["symbols"],
                                    xyz[:1], False)[0].double()
                    return head.bond_representation(h, idx, bof, n_bonds)
            else:
                graph = graph_of(entry["symbols"], bonds,
                                 cache[f"bond_orders::{key}"]).to(device)
                if args.stack == "espaloma_native":
                    def parameters():
                        return native(graph)
                else:
                    idx = torch.tensor(indices, dtype=torch.long, device=device)
                    bof = torch.tensor(bond_of, dtype=torch.long, device=device)

                    def parameters():
                        h = embed_graph(encoder, graph, 1, False)[0].double()
                        return head.bond_representation(h, idx, bof, n_bonds)

            with torch.no_grad():
                per_pass = timed(parameters, repeats=20, sync=sync)

            for n_conformers in (int(c) for c in args.conformers.split(",")):
                # A graph encoder reads no geometry, so its parameters are the
                # same for every conformer and the pass is paid once. An
                # atomistic descriptor reads the geometry, so it is paid again.
                passes = n_conformers if args.stack == "dpa3" else 1
                for n_angles in (int(a) for a in args.angles.split(",")):
                    rows.append({
                        "stack": args.stack, "device": str(device),
                        "molecule_id": entry["molecule_id"],
                        "n_atoms": entry["n_atoms"], "n_bonds": n_bonds,
                        "n_conformers": n_conformers, "n_angles": n_angles,
                        "parameter_passes": passes,
                        "parameter_seconds": per_pass * passes,
                        "query_seconds": query_seconds(n_bonds, n_angles),
                        "total_seconds": per_pass * passes
                        + query_seconds(n_bonds, n_angles),
                    })
        except Exception as error:                       # noqa: BLE001
            skipped.append((entry["molecule_id"], f"{type(error).__name__}: {error}"))

    if not rows:
        print(f"every one of {len(entries)} molecules failed; first: {skipped[:1]}")
        return 1
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"{args.stack} on {device}: {len({r['molecule_id'] for r in rows})} "
          f"molecules, {len(skipped)} skipped")
    for n_conformers in (int(c) for c in args.conformers.split(",")):
        for n_angles in (24, 360):
            sub = [r for r in rows if r["n_conformers"] == n_conformers
                   and r["n_angles"] == n_angles]
            if sub:
                print(f"  {n_conformers:>3} conformer(s), {n_angles:>4} angles, "
                      f"all bonds: {np.median([r['total_seconds'] for r in sub])*1e3:8.3f} ms"
                      f"  (parameters {np.median([r['parameter_seconds'] for r in sub])*1e3:7.3f}, "
                      f"query {np.median([r['query_seconds'] for r in sub])*1e3:6.3f})")
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
