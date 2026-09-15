"""Accuracy against inference cost: the trade-off the torsion work actually found.

The head is 2.3x less accurate than the pretrained potential it is built on and
about two orders of magnitude cheaper per profile, and its cost does not grow
with how finely the profile is sampled. Neither number means much alone. Drawn
together they are the result.

ONE POINT IS ONE MODEL. x is the wall time to predict every scan point of the
whole cohort, measured not estimated, on one RTX 4000 Ada. y is the median
per-molecule profile RMSE against B3LYP-D3BJ/DZVP on the same molecules. Down and
left is better, so the lower-left frontier is the set of models not dominated by
another.

WHAT THE x AXIS DOES NOT INCLUDE, stated because it flatters us. The pretrained
potentials were never trained on this data; our head was, at roughly two GPU-hours
per fold. This is inference cost. And the potentials are timed through an ASE
calculator one conformer at a time, which is overhead-bound, so a batched
implementation would move them left. The structural claim, that our cost is flat
in the number of angles while theirs is linear, survives both caveats.
"""

from __future__ import annotations

import argparse
import glob
import pathlib

import numpy as np

# Okabe and Ito, distinguishable under the common colour vision deficiencies
BLUE, ORANGE, GREEN, VERM, PURPLE, SKY, GREY = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#666666")


def nature_style(use_tex=True):
    import matplotlib
    matplotlib.use("Agg")
    rc = {
        "figure.dpi": 300, "savefig.dpi": 600, "savefig.bbox": "tight",
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.minor.size": 1.4, "ytick.minor.size": 1.4,
        "legend.frameon": False, "lines.linewidth": 0.9,
        "axes.titlelocation": "left", "axes.titlepad": 6,
        "axes.labelpad": 2.5,
    }
    if use_tex:
        rc.update({
            "text.usetex": True,
            # sfmath is not in this TeX installation, so maths is set with the
            # default family rather than forced sans. Nature sets figure text in
            # Helvetica; the body text here follows and the few maths symbols do
            # not, which is a smaller sin than the figure failing to build.
            "text.latex.preamble": r"\usepackage{helvet}\renewcommand{\familydefault}{\sfdefault}",
        })
    matplotlib.rcParams.update(rc)
    return matplotlib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--zeroshot-glob", required=True)
    parser.add_argument("--cost", required=True)
    parser.add_argument("--figure", required=True)
    parser.add_argument("--table", required=True)
    parser.add_argument("--no-tex", action="store_true")
    args = parser.parse_args()

    import pandas as pd
    try:
        matplotlib = nature_style(not args.no_tex)
        import matplotlib.pyplot as plt
        plt.figure(); plt.close()          # force a LaTeX round trip early
    except Exception as error:             # noqa: BLE001
        print(f"  LaTeX unavailable ({type(error).__name__}); falling back to mathtext")
        matplotlib = nature_style(False)
        import matplotlib.pyplot as plt

    rows = []
    for path in sorted(glob.glob(args.zeroshot_glob)):
        frame = pd.read_csv(path)
        if "seconds_per_profile" not in frame:
            continue
        for name, group in frame.groupby("potential"):
            rows.append({
                "model": name, "rmse": group.rmse.median(),
                "seconds_total": group.seconds_per_profile.iloc[0] * len(group),
                "n": len(group), "trained_here": False,
            })
    ours = pd.concat([pd.read_csv(f) for f in glob.glob(f"{args.runs}/*/per_molecule.csv")])
    cost = pd.read_csv(args.cost)
    per_profile = float(cost[(cost.path == "encoder + head") & (cost.n_angles == 24)]
                        .seconds_per_molecule.iloc[0])
    for arm, group in ours.groupby("encoder"):
        rows.append({"model": f"ours: {arm}", "rmse": group.rmse.median(),
                     "seconds_total": per_profile * len(group), "n": len(group),
                     "trained_here": True})
    n_molecules = int(ours.molecule_id.nunique())
    rows.append({"model": "flat profile", "rmse": float(ours.rmse_flat.median()),
                 "seconds_total": 1e-4 * n_molecules, "n": n_molecules, "trained_here": False})
    table = pd.DataFrame(rows).sort_values("seconds_total")
    table.to_csv(args.table, index=False)

    LOOK = {
        "ours: finetuned_per_angle": (BLUE, "o", r"this work, fine-tuned"),
        "ours: frozen":              (SKY, "o", r"this work, frozen"),
        "ours: frozen_per_angle":    (SKY, "o", None),
        "ours: finetuned_reference": (SKY, "o", None),
        "dpa3:SPICE2":               (VERM, "s", r"DPA-3.1-3M"),
        "mace_off23_small":          (GREEN, "^", r"MACE-OFF23 S"),
        "mace_off23_medium":         (ORANGE, "^", r"MACE-OFF23 M"),
        "mace_off23_large":          (PURPLE, "^", r"MACE-OFF23 L"),
        "flat profile":              (GREY, "x", r"flat profile"),
    }

    figure, ax = plt.subplots(figsize=(3.5, 3.0))
    for _, row in table.iterrows():
        colour, marker, label = LOOK.get(row.model, (GREY, "o", None))
        ax.scatter(row.seconds_total, row.rmse, s=34 if label else 16, marker=marker,
                   facecolor=colour, edgecolor="black", linewidth=0.45,
                   alpha=1.0 if label else 0.55, zorder=4, label=label)
    # the frontier of models not dominated by another
    front = table.sort_values("seconds_total")
    best, xs, ys = np.inf, [], []
    for _, row in front.iterrows():
        if row.rmse < best:
            best = row.rmse; xs.append(row.seconds_total); ys.append(row.rmse)
    ax.plot(xs, ys, color="black", lw=0.6, ls=(0, (3, 2)), zorder=2, alpha=0.5)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"inference time for the full cohort (s)")
    ax.set_ylabel(r"profile RMSE (kcal\,mol$^{-1}$)")
    ax.set_title(r"\textbf{a}" if not args.no_tex else "a", loc="left")
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.5)
    ax.legend(loc="lower left", handletextpad=0.35, borderpad=0.2, labelspacing=0.35)
    ax.annotate("", xy=(0.06, 0.06), xytext=(0.30, 0.30), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", lw=0.6, color=GREY))
    ax.text(0.32, 0.30, "better", transform=ax.transAxes, fontsize=6, color=GREY)

    pathlib.Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.figure)
    print(f"  {len(table)} models, cohort of {n_molecules} molecules x 24 points")
    for _, r in table.iterrows():
        print(f"    {r.model:28s} {r.seconds_total:9.1f} s   RMSE {r.rmse:.3f}")
    print(f"  -> {args.figure}")


if __name__ == "__main__":
    main()
