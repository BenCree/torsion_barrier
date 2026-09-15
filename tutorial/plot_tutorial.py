"""One figure: how much of a torsion profile is the asymmetric kind, by benchmark.

The census writes a `source` column naming the file each scan came from, so the
two benchmarks can be drawn apart without a second pass over the data.
"""
from __future__ import annotations

import argparse
import csv
import pathlib

import numpy as np

BLUE, ORANGE, GREY = "#0072B2", "#E69F00", "#666666"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--census", required=True)
    p.add_argument("--figure", required=True)
    args = p.parse_args()

    with open(args.census) as fh:
        rows = list(csv.DictReader(fh))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({
        "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))

    sources = sorted({r["source"] for r in rows})
    colours = dict(zip(sources, (BLUE, ORANGE)))

    ax = axes[0]
    for source in sources:
        values = np.sort([float(r["odd_fraction"]) for r in rows
                          if r["source"] == source])
        if not values.size:
            continue
        ax.plot(values, np.linspace(0, 1, values.size), lw=1.4,
                color=colours[source],
                label=f"{source} (n = {values.size:,}, median "
                      f"{np.median(values):.3f})")
    ax.set_xlabel("fraction of the profile that is asymmetric")
    ax.set_ylabel("fraction of scans")
    ax.set_title("a  how much asymmetry is there", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=6)

    ax = axes[1]
    for source in sources:
        values = np.sort([float(r["js_distance"]) for r in rows
                          if r["source"] == source])
        if not values.size:
            continue
        ax.plot(values, np.linspace(0, 1, values.size), lw=1.4,
                color=colours[source], label=source)
    # presto's Table 1, for scale: these are whole-force-field errors on
    # TorsionNet500, so a comparable number here means ignoring the asymmetry
    # costs as much as the gap between two force fields.
    for value, label in ((0.12, "presto"), (0.20, "espaloma"), (0.30, "Sage")):
        ax.axvline(value, color=GREY, ls=":", lw=0.8)
        ax.text(value, 0.02, f" {label}", fontsize=5.6, rotation=90,
                va="bottom", color=GREY)
    ax.set_xlabel("Jensen-Shannon distance from dropping the asymmetric part")
    ax.set_ylabel("fraction of scans")
    ax.set_title("b  what dropping it costs", loc="left", fontweight="bold")

    fig.text(0.5, -0.06,
             "Dotted lines are whole-force-field errors from Clark et al., "
             "ChemRxiv 2026, Table 1, on TorsionNet500.",
             ha="center", fontsize=6, color=GREY)
    fig.tight_layout()
    pathlib.Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure)
    print(f"  -> {args.figure}")


if __name__ == "__main__":
    main()
