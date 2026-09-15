"""Draw a test set from the pool MATCHED to a reference set's composition.

WHY MATCHED AND NOT RANDOM. TorsionNet-500 is the standard benchmark, so the
field has had years to tune against it and SPICE2, which is in DPA-3.1-3M's own
pretraining, may overlap it. A fair check is whether a pretrained potential does
as well on molecules nobody has optimised against.

A plain random draw would not answer that. The pool's median is 13 heavy atoms
and 42% of it is XtalPi fragments at 6, against TorsionNet-500's ~17, so a random
draw is systematically smaller and any difference would be molecule size rather
than familiarity. This matches the reference's heavy-atom histogram bin by bin
and restricts to its element set, so size and chemistry are held and only
familiarity differs.

Molecules present in the reference are excluded by InChIKey, so the draw is
disjoint from it.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np


def load(path):
    z = np.load(path, allow_pickle=False)
    return z, json.loads(str(z["meta"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    pool_z, pool = load(args.pool)
    _, reference = load(args.reference)

    def heavy(entry):
        return sum(1 for s in entry["symbols"] if s != "H")

    ref_elements = {s for e in reference for s in e["symbols"]}
    ref_keys = {e.get("inchi_key") for e in reference if e.get("inchi_key")}
    ref_heavy = np.array([heavy(e) for e in reference])
    print(f"reference: {len(reference)} molecules, elements {sorted(ref_elements)}, "
          f"heavy atoms median {np.median(ref_heavy):.0f} range {ref_heavy.min()}-{ref_heavy.max()}")

    eligible = [
        e for e in pool
        if set(e["symbols"]) <= ref_elements
        and e.get("inchi_key") not in ref_keys
    ]
    print(f"pool: {len(pool)} molecules -> {len(eligible)} eligible "
          f"(element subset, not in the reference)")

    # match the reference's heavy-atom histogram bin by bin
    by_size = {}
    for entry in eligible:
        by_size.setdefault(heavy(entry), []).append(entry)
    counts = {}
    for h in ref_heavy:
        counts[int(h)] = counts.get(int(h), 0) + 1

    rng = np.random.default_rng(args.seed)
    chosen, shortfall = [], 0
    for size, wanted in sorted(counts.items()):
        wanted = int(round(wanted * args.n / len(reference)))
        available = by_size.get(size, [])
        take = min(wanted, len(available))
        shortfall += wanted - take
        if take:
            picked = rng.choice(len(available), take, replace=False)
            chosen.extend(available[i] for i in picked)
    # top up from the nearest sizes if some bins were empty
    if shortfall > 0:
        rest = [e for e in eligible if e not in chosen]
        if rest:
            extra = rng.choice(len(rest), min(shortfall, len(rest)), replace=False)
            chosen.extend(rest[i] for i in extra)

    arrays, meta = {}, []
    for entry in chosen:
        new_key = str(len(meta))
        for field in ("xyz", "energy", "bonds", "bond_orders"):
            source = f"{field}::{entry['key']}"
            if source in pool_z:
                arrays[f"{field}::{new_key}"] = pool_z[source]
        meta.append(dict(entry, key=new_key))

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, meta=json.dumps(meta), dataset="size-matched pool draw",
                        specification="b3lyp-d3bj/dzvp", **arrays)
    got = np.array([heavy(e) for e in meta])
    print(f"\ndrew {len(meta)} molecules")
    print(f"  heavy atoms: median {np.median(got):.0f} (reference {np.median(ref_heavy):.0f}), "
          f"mean {got.mean():.1f} (reference {ref_heavy.mean():.1f})")
    print(f"  barriers: median {np.median([m['barrier_kcal_mol'] for m in meta]):.2f} kcal/mol "
          f"(reference {np.median([m['barrier_kcal_mol'] for m in reference]):.2f})")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
