"""What a generated conformer costs, drawn from the table rather than the run.

QUESTION. The run's own figure showed one held-out molecule whose ETKDG profile
was nearly flat against a 5.5 kcal/mol QM double barrier, while its privileged
and rotated profiles tracked. Is that one molecule or is it the mode? A
systematic flattening matters more than the RMSE does: a docking refinement that
underpredicts every barrier will not move a pose, and an RMSE of 1.65 is reached
just as well by a model that is right on average and by one that predicts
nothing.

ON WHAT. `data/conformer_sensitivity.csv`, one row per (molecule, geometry
source), 217 held-out molecules.

HOW. The slope of predicted barrier against QM barrier, through the origin, per
source. Slope 1 is calibrated, slope below 1 is flattening. Beside it the
cumulative RMSE and the conformer-to-conformer spread.

CONTROLS. The privileged source is the standard: whatever the slope is there is
what the model does when given everything. The noise source is the random
control and is drawn on the same axes.

UNITS. kcal/mol. One point is one molecule under one source.

COST. Seconds, CPU.
"""
from __future__ import annotations

import argparse
import csv
import pathlib

import numpy as np

BLUE, ORANGE, GREEN, VERM, GREY = ("#0072B2", "#E69F00", "#009E73", "#D55E00",
                                   "#666666")
COLOUR = {"privileged": BLUE, "qm_rotated": GREEN, "etkdg": ORANGE,
          "noise": GREY}
LABEL = {"privileged": "privileged: 24 QM geometries",
         "qm_rotated": "one QM geometry, phase rotated",
         "etkdg": "ETKDG conformer, no QM at all",
         "noise": "random control: displaced geometry"}


def boot_slope(x, y, n=2000, seed=0):
    """Least-squares slope through the origin, resampled over molecules."""
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, x.size, size=(n, x.size))
    slopes = (x[draws] * y[draws]).sum(1) / np.maximum((x[draws] ** 2).sum(1), 1e-12)
    point = float((x * y).sum() / max((x ** 2).sum(), 1e-12))
    return point, float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", required=True)
    parser.add_argument("--figure", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    with open(args.table) as fh:
        rows = list(csv.DictReader(fh))

    def column(source, name):
        out = []
        for row in rows:
            if row["source"] != source:
                continue
            try:
                out.append(float(row[name]))
            except (KeyError, ValueError):
                continue
        return np.asarray(out, dtype=float)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({
        "figure.dpi": 200, "savefig.dpi": 400, "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 7, "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6))

    summary = []
    ax = axes[0]
    for source in ("privileged", "qm_rotated", "etkdg", "noise"):
        values = np.sort(column(source, "rmse_vs_qm"))
        if not values.size:
            continue
        ax.plot(values, np.linspace(0, 1, values.size), lw=1.2,
                color=COLOUR[source], label=LABEL[source])
    ax.axvline(0.767, color="black", ls=":", lw=0.8)
    ax.text(0.80, 0.52, "label\nuncertainty", fontsize=5.4, color="black")
    ax.axvline(0.724, color=VERM, ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlim(0.2, 30)
    ax.set_xlabel("profile RMSE to QM (kcal/mol)")
    ax.set_ylabel("fraction of molecules")
    ax.set_title("a  what the geometry costs", loc="left", fontsize=8,
                 fontweight="bold", pad=8)
    ax.legend(frameon=False, fontsize=5.2, loc="center right")

    ax = axes[1]
    top = 0.0
    for source in ("privileged", "etkdg"):
        x = column(source, "barrier_qm")
        y = column(source, "barrier_predicted")
        keep = np.isfinite(x) & np.isfinite(y)
        x, y = x[keep], y[keep]
        if not x.size:
            continue
        slope, low, high = boot_slope(x, y)
        summary.append({"source": source, "n": int(x.size), "slope": slope,
                        "slope_low": low, "slope_high": high,
                        "median_rmse": float(np.median(column(source, "rmse_vs_qm")))})
        top = max(top, float(x.max()), float(y.max()))
        ax.plot(x, y, "o", ms=1.7, alpha=0.35, color=COLOUR[source], mew=0,
                label=f"{LABEL[source].split(':')[0]}, slope {slope:.2f} "
                      f"[{low:.2f}, {high:.2f}]")
    limit = min(top * 1.05, 25.0)
    ax.plot([0, limit], [0, limit], "k-", lw=0.8, label="y = x")
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_xlabel("QM barrier (kcal/mol)")
    ax.set_ylabel("predicted barrier (kcal/mol)")
    ax.set_title("b  is the barrier flattened?", loc="left", fontsize=8,
                 fontweight="bold", pad=8)
    ax.legend(frameon=False, fontsize=5.2, loc="upper left")

    ax = axes[2]
    spread = column("etkdg_spread", "rmse_vs_qm")
    gap_etkdg = column("etkdg", "rmse_vs_privileged")
    gap_etkdg = gap_etkdg[np.isfinite(gap_etkdg)]
    parts = [p for p in (spread, gap_etkdg) if p.size]
    ax.boxplot(parts, widths=0.5, showfliers=False,
               medianprops={"color": VERM, "lw": 1.2})
    ax.set_xticks(range(1, len(parts) + 1))
    ax.set_xticklabels(["between\nconformers", "conformer vs\nprivileged"],
                       fontsize=6)
    ax.set_ylabel("profile RMSE (kcal/mol)")
    ax.set_title("c  drawn conformer, or the method?", loc="left", fontsize=8,
                 fontweight="bold", pad=8)
    if spread.size and gap_etkdg.size:
        ax.text(0.5, 0.02,
                f"medians {np.median(spread):.2f} and {np.median(gap_etkdg):.2f}",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=5.8,
                color=GREY)

    n = len({row["molecule_id"] for row in rows})
    fig.text(0.5, -0.10,
             f"Held-out OpenFF Gen3 molecules, n = {n}, one fold, resampled over "
             f"molecules. Dashed line in a is MACE-OFF23 small on the same set.",
             ha="center", fontsize=6.2, color=GREY)
    fig.tight_layout()
    pathlib.Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure)

    with open(args.summary, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    for row in summary:
        print(f"{row['source']:<12} n={row['n']:<5} barrier slope "
              f"{row['slope']:.3f} [{row['slope_low']:.3f}, {row['slope_high']:.3f}], "
              f"median RMSE {row['median_rmse']:.3f}")
    print(f"  -> {args.figure}\n  -> {args.summary}")


if __name__ == "__main__":
    main()
