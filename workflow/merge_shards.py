"""Combine fetched shards into one cache the training reads.

The fetch is sharded so that 8,737 molecules do not become one serial task; the
training wants one file. This joins them, rekeyed so nothing collides.

WHAT IS CHECKED, because a training set that is quietly short is the kind of
thing nothing downstream notices. Every shard named in the plan must be present
or the merge stops and says which are missing.

THE DEDUPLICATION KEY IS (InChIKey, DRIVEN BOND), NOT InChIKey. An earlier
version used the InChIKey alone and so treated two scans of DIFFERENT bonds in
the same molecule as duplicates of each other. They are not duplicates, they are
different measurements: 1,167 molecules in this pool were scanned at two or more
distinct rotors by different OpenFF datasets, and collapsing them discarded 3,677
scans, about 28 percent of the usable data. They are also the only cross-bond
validation available anywhere, since a torsiondrive labels one bond per molecule.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", required=True, help="directory of shard npz files")
    parser.add_argument("--pattern", default="*.npz")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--shards",
        default=None,
        help="the plan file; if given, every listed shard matching --pattern must exist",
    )
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    import numpy as np

    pool = pathlib.Path(args.pool)
    files = sorted(pool.glob(args.pattern))

    if args.shards:
        planned = []
        for line in pathlib.Path(args.shards).read_text().strip().splitlines():
            slug, offset, limit = line.split("\t")
            planned.append(pool / f"{slug}_{offset}_{int(limit)}.npz")
        wanted = [f for f in planned if f in set(files) or not f.exists()]
        wanted = [f for f in planned if pathlib.PurePath(f.name).match(args.pattern)]
        missing = [f.name for f in wanted if not f.exists()]
        if missing and not args.allow_missing:
            print(f"{len(missing)} of {len(wanted)} planned shards missing; "
                  f"first few: {missing[:5]}")
            sys.exit(1)
        files = [f for f in wanted if f.exists()]

    if not files:
        print(f"no shards matched {args.pattern!r} in {pool}")
        sys.exit(1)

    arrays, meta, seen, duplicates = {}, [], set(), 0
    for path in files:
        z = np.load(path, allow_pickle=False)
        for entry in json.loads(str(z["meta"])):
            driven = tuple(entry.get("driven_dihedral", ()))
            bond = tuple(sorted(driven[1:3])) if len(driven) == 4 else driven
            identity = (entry.get("inchi_key") or entry["molecule_id"], bond)
            if identity in seen:
                duplicates += 1
                continue
            seen.add(identity)
            new_key = str(len(meta))
            for field in ("xyz", "energy", "bonds", "bond_orders"):
                source = f"{field}::{entry['key']}"
                if source in z:
                    arrays[f"{field}::{new_key}"] = z[source]
            meta.append(dict(entry, key=new_key, source=path.name,
                             driven_bond=list(bond)))

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        meta=json.dumps(meta),
        dataset="merged pool",
        specification="b3lyp-d3bj/dzvp",
        **arrays,
    )

    heavy = np.array(
        [sum(1 for s in m["symbols"] if s != "H") for m in meta]
    ) if meta else np.array([0])
    distinct_molecules = len({m.get("inchi_key") or m["molecule_id"] for m in meta})
    print(f"{len(files)} shards -> {len(meta):,} unique (molecule, driven bond) scans "
          f"across {distinct_molecules:,} molecules "
          f"({duplicates:,} true duplicates dropped)")
    print(f"  heavy atoms: median {np.median(heavy):.0f}, "
          f"range {int(heavy.min())} to {int(heavy.max())}")
    print(f"  scan points: {sum(len(m['grid']) for m in meta):,}")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
