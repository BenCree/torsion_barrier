"""Decide which QCArchive torsiondrive datasets to fetch, and how to shard them.

WHY THIS IS ITS OWN STEP. The selection is a scientific decision, not a detail of
the download, and it has already been wrong twice: `record_count` from the
dataset listing counts records across every specification, so a set with 12 specs
reports 12 times its molecules, and a dataset whose DFT specification is not
named "default" would have been fetched at the wrong level of theory or not at
all. Both are checked here, once, and written to a file the array reads.

WHAT IS SELECTED
  in    every torsiondrive dataset carrying a b3lyp-d3bj/dzvp specification,
        canonicalising "b3lyp-d3(bj)" and "b3lyp-d3bj" as the same method
  out   the four "OpenFF SMIRNOFF Sage" sets, which are the torsiondrives Sage's
        own torsion parameters were fitted to. Evaluating against Sage on data it
        was fitted to would put the standard in-sample and this model out
  out   TorsionNet500, the held-out benchmark, fetched separately and not opened
        until the final stage

Writes `sources.tsv` (slug, dataset name, specification, molecules) and
`shards.tsv` (slug, offset, limit), one line per array task.
"""

from __future__ import annotations

import argparse
import pathlib
import re


def slugify(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:48]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", default="b3lyp-d3bj")
    parser.add_argument("--basis", default="dzvp")
    parser.add_argument("--shard", type=int, default=300, help="molecules per array task")
    parser.add_argument("--address", default="https://api.qcarchive.molssi.org:443")
    parser.add_argument("--exclude", default="smirnoff sage,torsionnet",
                        help="comma-separated substrings; a dataset whose name contains one is out")
    parser.add_argument("--sources", required=True)
    parser.add_argument("--shards", required=True)
    args = parser.parse_args()

    from qcportal import PortalClient

    def canon(method, basis):
        return f"{(method or '').lower().replace('d3(bj)', 'd3bj')}/{(basis or '').lower()}"

    wanted = canon(args.method, args.basis)
    drop = [s.strip().lower() for s in args.exclude.split(",") if s.strip()]
    client = PortalClient(args.address)

    rows, skipped = [], []
    listing = [
        d for d in client.list_datasets()
        if d["dataset_type"] == "torsiondrive" and d["record_count"] > 0
    ]
    for entry in listing:
        name = entry["dataset_name"]
        if any(s in name.lower() for s in drop):
            skipped.append((name, "excluded by name"))
            continue
        try:
            dataset = client.get_dataset("torsiondrive", name)
            specs = {}
            for spec_name in dataset.specification_names:
                q = dataset.specifications[spec_name].specification.optimization_specification.qc_specification
                specs[spec_name] = canon(q.method, q.basis)
            matching = [s for s, level in specs.items() if level == wanted]
            if not matching:
                skipped.append((name, f"no {wanted}; has {sorted(set(specs.values()))[:2]}"))
                continue
            n = len(dataset.entry_names)     # MOLECULES, not records
        except Exception as error:  # noqa: BLE001
            skipped.append((name, f"{type(error).__name__}: {error}"))
            continue
        rows.append((slugify(name), name, matching[0], n))

    rows.sort(key=lambda r: -r[3])
    sources = pathlib.Path(args.sources)
    sources.parent.mkdir(parents=True, exist_ok=True)
    with sources.open("w") as handle:
        for slug, name, spec, n in rows:
            handle.write(f"{slug}\t{name}\t{spec}\t{n}\n")

    shards = []
    for slug, _, _, n in rows:
        for offset in range(0, n, args.shard):
            shards.append((slug, offset, min(args.shard, n - offset)))
    with pathlib.Path(args.shards).open("w") as handle:
        for slug, offset, limit in shards:
            handle.write(f"{slug}\t{offset}\t{limit}\n")

    total = sum(r[3] for r in rows)
    print(f"{len(rows)} datasets selected, {total:,} molecules, {len(shards)} shards "
          f"of up to {args.shard}")
    print(f"  at 5.3 s per molecule: {total*5.3/3600:.1f} GPU-free hours serially, "
          f"{total*5.3/3600/8:.1f} h at 8 concurrent tasks")
    print(f"  {len(skipped)} datasets skipped:")
    for name, reason in skipped[:6]:
        print(f"    {name[:52]:52s} {reason[:60]}")


if __name__ == "__main__":
    main()
