"""Stage shards of a HuggingFace dataset repository onto local disk.

QUESTION. None on its own. This is a transfer, separated from everything that
computes, so that a change to the model does not cost another 345 GB.

WHAT IT TAKES. The repository, the subfolder inside it, which shards, and where
to put them. Nothing about THEMol is written into this file: the second dataset
that needs staging is meant to be a different command line, not a second script.

HOW. `huggingface_hub.hf_hub_download` one file at a time with `local_dir` set,
which writes the real bytes into the output directory and resumes a partial file
rather than restarting it. NOT the default cache layout: that stores a blob and
puts a RELATIVE symlink beside it, and a hard link to a symlink carries the
relative target to a place it does not resolve, so the first version of this
script left a shard that looked present, failed `exists()`, and took the whole
run down with it on shard 0 of 50.

Every shard is checked against the size the API reports before it counts as
done; a short file is deleted and retried once, then quarantined with its reason
while the remaining shards continue. Nothing raises out of the per-shard loop: a
partial set that is on disk is worth more than a complete one that is not.

COST. Network bound. 6.9 GB per shard of THEMol/TorsionScan, 50 shards.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
import urllib.request


def _ssl_context():
    """The compute nodes' Python has no CA bundle of its own; curl does.

    Without this the listing raises CERTIFICATE_VERIFY_FAILED on thira while
    huggingface_hub, which carries certifi through requests, downloads fine.
    """
    import ssl
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:                                    # noqa: BLE001
        return ssl.create_default_context()


def remote_listing(repo: str, subfolder: str, repo_type: str) -> dict[str, int]:
    url = (f"https://huggingface.co/api/{repo_type}s/{repo}/tree/main/"
           f"{subfolder}?recursive=true&expand=true")
    with urllib.request.urlopen(url, context=_ssl_context()) as fh:
        tree = json.load(fh)
    return {x["path"]: x.get("size", 0) for x in tree if x["type"] == "file"}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True, help="e.g. ByteDance-Seed/THEMol")
    p.add_argument("--repo-type", default="dataset", choices=["dataset", "model"])
    p.add_argument("--subfolder", default="", help="e.g. TorsionScan")
    p.add_argument("--out", required=True)
    p.add_argument("--shards", default="all",
                   help="'all', or a comma list / a:b range of indices into the "
                        "sorted remote listing")
    p.add_argument("--quarantine", default=None)
    args = p.parse_args()

    from huggingface_hub import hf_hub_download

    listing = remote_listing(args.repo, args.subfolder, args.repo_type)
    if not listing:
        print(f"no files under {args.subfolder!r} in {args.repo}")
        return 1
    names = sorted(listing)

    if args.shards == "all":
        chosen = names
    elif ":" in args.shards:
        a, b = args.shards.split(":")
        chosen = names[int(a or 0):int(b or len(names))]
    else:
        chosen = [names[int(i)] for i in args.shards.split(",")]

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / ".hf_cache"

    quarantine, done, skipped, moved_bytes = [], 0, 0, 0
    t_all = time.time()
    for k, name in enumerate(chosen, 1):
        want = listing[name]
        dest = out / name
        if dest.exists() and not dest.is_symlink() and dest.stat().st_size == want:
            skipped += 1
            continue
        reason = ""
        for attempt in (1, 2):
            t0 = time.time()
            try:
                got = pathlib.Path(hf_hub_download(
                    repo_id=args.repo, repo_type=args.repo_type, filename=name,
                    local_dir=str(out),
                ))
                size = got.stat().st_size
                if size != want:
                    got.unlink()
                    reason = f"size {size} != {want}"
                    continue
            except Exception as exc:                      # noqa: BLE001
                reason = f"{type(exc).__name__}: {exc}"
                continue
            done += 1
            moved_bytes += size
            dt = time.time() - t0
            print(f"[{k}/{len(chosen)}] {name} {size/1e9:.2f} GB "
                  f"in {dt/60:.1f} min ({size/1e6/max(dt,1e-9):.0f} MB/s)",
                  flush=True)
            reason = ""
            break
        if reason:
            quarantine.append({"shard": name, "reason": reason})
            print(f"[{k}/{len(chosen)}] {name} QUARANTINED {reason}", flush=True)

    if args.quarantine:
        q = pathlib.Path(args.quarantine)
        q.parent.mkdir(parents=True, exist_ok=True)
        with q.open("w") as fh:
            fh.write("shard,reason\n")
            for row in quarantine:
                fh.write(f"{row['shard']},\"{row['reason']}\"\n")

    elapsed = (time.time() - t_all) / 3600
    print(f"{done} fetched, {skipped} already present, {len(quarantine)} "
          f"quarantined, of {len(chosen)} requested; "
          f"{moved_bytes/1e9:.1f} GB in {elapsed:.2f} h -> {out}")
    return 1 if quarantine and done == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
