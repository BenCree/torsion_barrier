"""Acceptance check for the sharded QCArchive fetch.

READS THE FILES' OWN CONTENT, never a count. A shard can exit 0 having written a
well-formed npz holding zero molecules, or holding geometries that are all zeros
because a lazy fetch silently returned nothing. Four assertions per shard, and
every shard that fails is named with its number.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", default="/work/ben/torsion_phase/data/pool")
    parser.add_argument("--shards", default="/work/ben/torsion_phase/data/shards.tsv")
    args = parser.parse_args()

    import numpy as np

    expected = [
        line.split("\t") for line in
        pathlib.Path(args.shards).read_text().strip().splitlines()
    ]
    pool = pathlib.Path(args.pool)
    problems, molecules, points = [], 0, 0
    for slug, offset, limit in expected:
        path = pool / f"{slug}_{offset}_{int(limit)}.npz"
        if not path.exists():
            problems.append(f"{path.name}: missing")
            continue
        try:
            z = np.load(path, allow_pickle=False)
            meta = json.loads(str(z["meta"]))
        except Exception as error:  # noqa: BLE001
            problems.append(f"{path.name}: unreadable, {type(error).__name__}")
            continue
        if not meta:
            problems.append(f"{path.name}: zero molecules")
            continue
        first = meta[0]["key"]
        xyz = z[f"xyz::{first}"]
        if not np.isfinite(xyz).all() or np.abs(xyz).max() == 0:
            problems.append(f"{path.name}: geometries all zero or non-finite")
            continue
        energies = z[f"energy::{first}"]
        if not np.isfinite(energies).all():
            problems.append(f"{path.name}: non-finite energies")
            continue
        molecules += len(meta)
        points += sum(len(m["grid"]) for m in meta)

    print(f"{len(expected) - len(problems)} of {len(expected)} shards good: "
          f"{molecules:,} molecules, {points:,} optimised geometries")
    for problem in problems[:20]:
        print(f"  FAIL {problem}")
    if len(problems) > 20:
        print(f"  ... and {len(problems) - 20} more")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
