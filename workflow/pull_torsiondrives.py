"""Cache a QCArchive torsiondrive dataset: geometries, energies, topology.

WHY THIS IS A SEPARATE RULE. Two consumers need the same records, the DPA3
encoder and the fit, and each entry carries 24 constrained optimisations. So the
fetch happens once and writes a file.

EVERYTHING IS FETCHED IN BULK, and getting this wrong cost 20x. qcportal loads a
dataset lazily and every uncached access is a round trip. Measured 2026-09-11:

    one record at a time, reading .final_molecule      5.65 s per molecule
    fetch_records in chunks, still reading .final_molecule   5.3 s per molecule
    fetch_records, then ONE get_molecules by id        0.27 s per molecule

The middle row is the trap. `fetch_records(include=["minimum_optimizations"])`
does batch the optimisation records, and on its own it looks like a 20x speedup,
but only until something reads a geometry: `.final_molecule` is a SEPARATE lazy
fetch, one per grid point, and at 24 points per scan that is where all the time
goes. `.final_molecule_id` is already present and costs nothing, so the ids are
collected across a whole chunk and pulled with a single `client.get_molecules`.
432 geometries came back in 1.0 s that way.

For the 20,997 usable molecules that is 1.6 hours rather than 31.

WHAT IS STORED, and what is deliberately not. Per entry: every optimised geometry
of the scan in Angstrom, its energy in kcal/mol relative to the minimum of that
scan, the element symbols, the bond connectivity the archive supplies WITH ITS BOND
ORDERS, the atom indices of the driven torsion, and the canonical SMILES.

The bond orders are not optional. An earlier version stored only the atom pairs,
and the MMFF94 baseline built every bond as SINGLE, so it parametrised a
different molecule wherever anything was aromatic or doubly bonded and reported
the result in a column called mmff94. Nothing is derived here:
dihedrals, torsion enumeration and ranks belong to the rules that use them, so a
change to any of those does not invalidate a day of fetching.

ENERGIES ARE RELATIVE TO EACH SCAN'S OWN MINIMUM. A torsion profile is a shape,
and the absolute electronic energy of a molecule is set by everything in it. The
offset between molecules is not a quantity any torsion term is asked to predict,
so it is removed here rather than left for a later step to forget.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import time

import numpy as np

BOHR = 0.529177210903
HARTREE_PER_KCAL_MOL = 627.509474


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="QCArchive torsiondrive dataset name")
    parser.add_argument("--method", default="b3lyp-d3bj", help="canonical QC method to select")
    parser.add_argument("--basis", default="dzvp")
    parser.add_argument("--specification", default=None,
                        help="override; normally the spec is FOUND by method and basis")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--max-barrier", type=float, default=100.0,
                        help="kcal/mol; a scan above this is quarantined, not dropped silently")
    parser.add_argument("--chunk", type=int, default=200,
                        help="records fetched per server request; 0 fetches one at a time")
    parser.add_argument("--limit", type=int, default=0, help="0 means every entry")
    parser.add_argument("--offset", type=int, default=0,
                        help="skip this many entries first, so one dataset can be sharded "
                             "across array tasks. The entry order is the archive's own and "
                             "is stable, so shards do not overlap or leave gaps.")
    parser.add_argument("--address", default="https://api.qcarchive.molssi.org:443")
    parser.add_argument("--out", required=True)
    parser.add_argument("--quarantine", required=True)
    args = parser.parse_args()

    from qcportal import PortalClient

    client = PortalClient(args.address)
    dataset = client.get_dataset("torsiondrive", args.dataset)

    # THE SPECIFICATION IS FOUND, NOT ASSUMED. 4 of the 70 usable torsiondrive
    # datasets do not name their DFT specification "default": the Sage sets call
    # theirs "default-256-257-..." after the record ids they were built from, and
    # one calls it "B3LYP-d3bj/dzvp". Hardcoding "default" would have taken the
    # wrong specification or none. Method names also come in two spellings,
    # "b3lyp-d3(bj)" and "b3lyp-d3bj", which are canonicalised and joined rather
    # than treated as different levels of theory.
    def canon(method, basis):
        return f"{(method or '').lower().replace('d3(bj)', 'd3bj')}/{(basis or '').lower()}"

    wanted = canon(args.method, args.basis)
    available = {}
    for name in dataset.specification_names:
        q = dataset.specifications[name].specification.optimization_specification.qc_specification
        available[name] = canon(q.method, q.basis)
    if args.specification:
        specification = args.specification
    else:
        matches = [n for n, level in available.items() if level == wanted]
        if not matches:
            raise SystemExit(
                f"{args.dataset!r} has no {wanted} specification. available: {available}"
            )
        specification = matches[0]
    print(f"dataset {args.dataset!r}: using specification {specification!r} = {wanted}",
          flush=True)

    names = list(dataset.entry_names)
    total_available = len(names)
    names = names[args.offset:]
    if args.limit:
        names = names[: args.limit]
    if args.offset or args.limit:
        print(f"  shard: entries [{args.offset}, {args.offset + len(names)}) "
              f"of {total_available}", flush=True)

    arrays, meta, skipped = {}, [], []
    started = time.time()

    geometry_cache = {}

    def prefetch(chunk_names):
        """Two requests for a whole chunk: the records, then all their geometries.

        Fails soft: if either bulk call raises, the per-entry loop still works
        through the lazy path, just 20 times slower, and the reason is printed
        rather than ending the run.
        """
        if not args.chunk:
            return
        geometry_cache.clear()
        try:
            dataset.fetch_entries(entry_names=chunk_names)
            dataset.fetch_records(
                entry_names=chunk_names, specification_names=[specification],
                include=["minimum_optimizations"],
            )
        except Exception as error:  # noqa: BLE001
            print(f"  bulk record fetch failed ({type(error).__name__}: {error}); "
                  f"falling back to the lazy path", flush=True)
            return
        try:
            ids = []
            for name in chunk_names:
                record = dataset.get_record(name, specification)
                if record is None:
                    continue
                for key in record.minimum_optimizations:
                    # An optimisation that never finished has no final molecule and
                    # its id is None. One None in the list makes the whole bulk call
                    # fail pydantic validation and drops the chunk to the lazy path,
                    # which is 20 times slower; it happened on the first chunk of the
                    # first run. Those points are dropped here and the entry is
                    # quarantined below when its grid comes up short.
                    molecule_id = record.minimum_optimizations[key].final_molecule_id
                    if molecule_id is not None:
                        ids.append(int(molecule_id))
            for start in range(0, len(ids), 1000):
                batch = ids[start:start + 1000]
                for molecule_id, molecule in zip(batch, client.get_molecules(batch)):
                    geometry_cache[molecule_id] = molecule
        except Exception as error:  # noqa: BLE001
            print(f"  bulk geometry fetch failed ({type(error).__name__}: {error}); "
                  f"falling back to the lazy path", flush=True)

    def final_molecule(optimisation):
        """The optimised geometry, from the chunk cache where possible."""
        cached = geometry_cache.get(optimisation.final_molecule_id)
        return cached if cached is not None else optimisation.final_molecule

    chunk = args.chunk or 1
    for position, name in enumerate(names):
        if position % chunk == 0:
            prefetch(names[position:position + chunk])
        # A fetch of tens of thousands of records that prints only at the end is
        # not watchable, and this one runs for hours.
        if args.progress_every and position and position % args.progress_every == 0:
            rate = position / max(time.time() - started, 1e-9)
            remaining = (len(names) - position) / max(rate, 1e-9)
            print(f"  {position}/{len(names)} fetched, {len(skipped)} quarantined, "
                  f"{rate:.2f}/s, about {remaining/60:.0f} min left", flush=True)
        # Fail soft per entry, loud in the quarantine table. One unreachable
        # record must not end a pass over eight thousand.
        try:
            entry = dataset.get_entry(name)
            record = dataset.get_record(name, specification)
            if str(record.status).rsplit(".", 1)[-1].lower() != "complete":
                skipped.append((name, f"status {record.status}"))
                continue

            optimisations = record.minimum_optimizations
            keys = sorted(optimisations, key=lambda k: k[0])
            if len(keys) < 3:
                skipped.append((name, f"only {len(keys)} grid points"))
                continue

            molecules = [final_molecule(optimisations[k]) for k in keys]
            symbols = [str(s) for s in molecules[0].symbols]
            xyz = np.stack(
                [np.asarray(m.geometry, dtype=float).reshape(-1, 3) * BOHR for m in molecules]
            )
            if any(len(m.symbols) != len(symbols) for m in molecules):
                skipped.append((name, "atom count changes across the scan"))
                continue

            energies = np.array(
                [float(record.final_energies[k]) for k in keys]
            ) * HARTREE_PER_KCAL_MOL
            energies -= energies.min()

            # OUTLIER POLICY, stated rather than implicit. MACE-OFF23 removed 808
            # configurations whose force error exceeded 2 eV/A; the analogous
            # failure here is a scan whose barrier is physically impossible,
            # which means the optimisation fell into a different molecule rather
            # than turning a bond. A large finite barrier is a result to look at;
            # 1e7 is a bug. The threshold is an argument and every rejection is a
            # quarantine row with its number, never a silent drop.
            if not np.isfinite(energies).all():
                skipped.append((name, "non-finite energy in the scan"))
                continue
            if energies.max() > args.max_barrier:
                skipped.append(
                    (name, f"barrier {energies.max():.1f} kcal/mol exceeds "
                           f"--max-barrier {args.max_barrier}")
                )
                continue

            connectivity = molecules[0].connectivity
            if not connectivity:
                skipped.append((name, "no connectivity in the archive record"))
                continue

            key = f"{len(meta)}"
            arrays[f"xyz::{key}"] = xyz.astype(np.float32)
            arrays[f"energy::{key}"] = energies.astype(np.float64)
            arrays[f"bonds::{key}"] = np.array(
                [[int(i), int(j)] for i, j, _ in connectivity], dtype=np.int32
            )
            arrays[f"bond_orders::{key}"] = np.array(
                [float(order) for _, _, order in connectivity], dtype=np.float32
            )
            meta.append(
                {
                    "key": key,
                    "molecule_id": name,
                    "symbols": symbols,
                    "n_atoms": len(symbols),
                    "grid": [int(k[0]) for k in keys],
                    "driven_dihedral": [
                        int(a) for a in entry.additional_keywords["dihedrals"][0]
                    ],
                    "smiles": entry.attributes.get("canonical_smiles", ""),
                    "inchi_key": entry.attributes.get("inchi_key", ""),
                    "barrier_kcal_mol": float(energies.max()),
                }
            )
        except Exception as error:  # noqa: BLE001
            skipped.append((name, f"{type(error).__name__}: {error}"))

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        meta=json.dumps(meta),
        dataset=args.dataset,
        specification=specification,
        **arrays,
    )

    quarantine = pathlib.Path(args.quarantine)
    quarantine.parent.mkdir(parents=True, exist_ok=True)
    with quarantine.open("w") as handle:
        handle.write("molecule_id,reason\n")
        for name, reason in skipped:
            handle.write(f'"{name}","{str(reason)[:200]}"\n')

    barriers = np.array([m["barrier_kcal_mol"] for m in meta]) if meta else np.array([0.0])
    print(
        f"{len(meta)} entries cached, {len(skipped)} quarantined. "
        f"barrier kcal/mol: median {np.median(barriers):.2f}, "
        f"range {barriers.min():.2f} to {barriers.max():.2f}"
    )


if __name__ == "__main__":
    main()
