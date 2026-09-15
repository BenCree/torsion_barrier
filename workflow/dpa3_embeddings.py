"""Per-atom DPA3 embeddings for a set of torsiondrive records.

WHAT THIS IS AND WHY IT IS A SEPARATE SCRIPT. DeePMD-kit caps at Python 3.12 and
the torsion head lives in an environment at 3.13, so the two cannot share a
process. Embeddings are therefore computed once and cached, which is also what
"frozen encoder" means at stage 2 of PLAN_torsion_prediction_2026-09-11.md.

Reads the cache written by pull_torsiondrives.py rather than QCArchive, so a
change here costs GPU time and not a second pass over the archive.

THE CHOICE THAT COULD HAVE GONE OTHERWISE, stated because it changes what the
model is rather than how well it does.

DPA3 embeddings depend on the geometry they are computed from. Measured
2026-09-11 on hand-built ethane, the three hydrogens of one methyl differ by
0.0025 in embedding units while a carbon and a hydrogen differ by 1.005, so the
geometry dependence is real and small.

    --geometry reference   (default) every scan point of a molecule gets the
                           embeddings of ONE conformer. h_r is then a property of
                           the molecule, the whole conformational dependence sits
                           in the phase s_n, and the fitted k_n are transferable
                           torsion parameters that can be written into a force
                           field.

    --geometry per_point   each scan point gets its own embeddings. More
                           expressive, and no longer a parameter predictor: k_n
                           would change every time the molecule moved, so nothing
                           can be read off and written out.

The default is `reference` because readable parameters are the point. `per_point`
exists to measure what that costs, which is an arm of the experiment rather than
a setting.

Usage
-----
    python dpa3_embeddings.py \
        --torsiondrives data/torsiondrives.npz \
        --checkpoint /home/ben/models/DPA-3.1-3M/DPA-3.1-3M.pt \
        --config /home/ben/models/DPA-3.1-3M/input_pretrain.json \
        --head SPICE2 --out data/embeddings.npz
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch

BOHR = 0.529177210903


def build_descriptor(config_path: str, checkpoint_path: str, head: str):
    """The pretrained DPA3 descriptor, with the weights of one multi-task head.

    The descriptor is shared across all 20 heads of DPA-3.1-3M, so the head only
    selects which copy of identical weights is read. `SPICE2` is the default
    because it is the organic-molecule head and its name says so in the record.
    """
    from deepmd.pt.model.descriptor.dpa3 import DescrptDPA3
    from deepmd.pt.utils.env import DEVICE

    config = json.loads(pathlib.Path(config_path).read_text())
    shared = config["model"]["shared_dict"]
    type_map = shared["type_map_all"]
    settings = {k: v for k, v in shared["dpa3_descriptor"].items() if k != "type"}
    # The pretrain config names the activation the way the trainer wrote it; the
    # constructor expects the registered name. Same function, two spellings.
    settings["activation_function"] = settings["activation_function"].replace(
        "custom_silu", "silut"
    )

    descriptor = DescrptDPA3(ntypes=len(type_map), type_map=type_map, **settings)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    prefix = f"model.{head}.atomic_model.descriptor."
    state = {
        k[len(prefix):]: v for k, v in checkpoint["model"].items() if k.startswith(prefix)
    }
    if not state:
        heads = sorted({k.split(".")[1] for k in checkpoint["model"] if k.count(".") > 1})
        raise SystemExit(f"no head named {head!r}. available: {heads}")
    missing, unexpected = descriptor.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(
            f"checkpoint does not match the configured descriptor: "
            f"{len(missing)} missing, {len(unexpected)} unexpected"
        )
    return descriptor.to(DEVICE).eval(), type_map


def embed(descriptor, type_map, symbols, xyz_angstrom):
    """(n_atoms, 128) embeddings for one geometry."""
    from deepmd.pt.utils.env import DEVICE, GLOBAL_PT_FLOAT_PRECISION
    from deepmd.pt.utils.nlist import extend_input_and_build_neighbor_list

    atype = torch.tensor(
        [[type_map.index(s) for s in symbols]], dtype=torch.long, device=DEVICE
    )
    coord = torch.tensor(
        np.asarray(xyz_angstrom)[None], dtype=GLOBAL_PT_FLOAT_PRECISION, device=DEVICE
    )
    extended, extended_types, mapping, nlist = extend_input_and_build_neighbor_list(
        coord,
        atype,
        descriptor.get_rcut(),
        descriptor.get_sel(),
        mixed_types=descriptor.mixed_types(),
        box=None,
    )
    with torch.no_grad():
        return descriptor(extended, extended_types, nlist, mapping)[0][0].cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torsiondrives", required=True, help="cache from pull_torsiondrives.py")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--head", default="SPICE2")
    parser.add_argument("--geometry", choices=("reference", "per_point"), default="reference")
    parser.add_argument("--out", required=True)
    parser.add_argument("--quarantine", required=True, help="CSV of entries skipped, with the reason")
    args = parser.parse_args()

    cache = np.load(args.torsiondrives, allow_pickle=False)
    entries = json.loads(str(cache["meta"]))
    descriptor, type_map = build_descriptor(args.config, args.checkpoint, args.head)

    arrays, meta, skipped = {}, [], []
    for entry in entries:
        key = entry["key"]
        name = entry["molecule_id"]
        try:
            symbols = entry["symbols"]
            unknown = sorted(set(symbols) - set(type_map))
            if unknown:
                skipped.append((name, f"elements outside the type map: {unknown}"))
                continue

            xyz = cache[f"xyz::{key}"]
            if args.geometry == "reference":
                energies = cache[f"energy::{key}"]
                h = embed(descriptor, type_map, symbols, xyz[int(np.argmin(energies))])[None]
            else:
                h = np.stack([embed(descriptor, type_map, symbols, g) for g in xyz])

            arrays[f"h::{key}"] = h.astype(np.float32)
            meta.append({"key": key, "molecule_id": name, "n_atoms": len(symbols)})
        except Exception as error:  # noqa: BLE001
            skipped.append((name, f"{type(error).__name__}: {error}"))

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        meta=json.dumps(meta),
        geometry=args.geometry,
        head=args.head,
        dataset=str(cache["dataset"]),
        **arrays,
    )

    quarantine = pathlib.Path(args.quarantine)
    quarantine.parent.mkdir(parents=True, exist_ok=True)
    with quarantine.open("w") as handle:
        handle.write("molecule_id,reason\n")
        for name, reason in skipped:
            handle.write(f'"{name}","{str(reason)[:200]}"\n')

    print(
        f"{len(meta)} molecules embedded, {len(skipped)} quarantined, "
        f"geometry={args.geometry}, head={args.head}"
    )


if __name__ == "__main__":
    main()
