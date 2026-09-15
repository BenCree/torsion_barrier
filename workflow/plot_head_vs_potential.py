"""Our fitted head against the zero-shot potential it is built on, drawn.

Summary statistics said the head is 2.2x worse in median RMSE. This asks the
question they cannot: is the head's profile qualitatively right and merely
mis-scaled, or is it the wrong shape? Those have different fixes. A shape that
is right with the wrong amplitude is a calibration problem; a shape that is
wrong is an architecture problem.

Six molecules spanning the barrier range, plus the three the head wins on by the
largest margin, so both directions are shown rather than only the losses.
"""

from __future__ import annotations

import argparse
import glob
import pathlib

import numpy as np

BLACK, GREY = "#000000", "#666666"
BLUE, ORANGE, GREEN = "#0072B2", "#E69F00", "#009E73"


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours-runs", required=True)
    parser.add_argument("--zeroshot-profiles", required=True)
    parser.add_argument("--zeroshot-per-molecule", required=True)
    parser.add_argument("--figure", required=True)
    args = parser.parse_args()

    matplotlib.rcParams.update({
        "figure.dpi": 200, "savefig.dpi": 400, "savefig.bbox": "tight",
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "xtick.direction": "in", "ytick.direction": "in",
        "legend.frameon": False, "lines.linewidth": 1.0,
        "axes.titlelocation": "left", "axes.titlepad": 13,
    })

    ours_p = pd.concat([pd.read_csv(f) for f in glob.glob(f"{args.ours_runs}/*/profiles.csv")])
    ours_m = pd.concat([pd.read_csv(f) for f in glob.glob(f"{args.ours_runs}/*/per_molecule.csv")])
    zs_p = pd.read_csv(args.zeroshot_profiles)
    zs_m = pd.read_csv(args.zeroshot_per_molecule).set_index("molecule_id")

    m = ours_m.set_index("molecule_id")
    shared = m.index.intersection(zs_m.index)
    delta = (m.loc[shared, "rmse"] - zs_m.loc[shared, "rmse"]).sort_values()
    picks = list(delta.index[:3]) + list(
        m.loc[shared].sort_values("qm_barrier_kcal_mol").index[::max(len(shared)//3, 1)][:3]
    )
    picks = list(dict.fromkeys(picks))[:6]

    figure, axes = plt.subplots(2, 3, figsize=(7.4, 4.8), sharex=True)
    for index, (ax, mol) in enumerate(zip(axes.flat, picks)):
        o = ours_p[ours_p.molecule_id == mol].sort_values("grid_degrees")
        z = zs_p[zs_p.molecule_id == mol].sort_values("grid_degrees")
        ax.plot(o.grid_degrees, o.qm_kcal_mol, color=BLACK, lw=1.3, marker="o", ms=2.4,
                label="QM", zorder=5)
        ax.plot(o.grid_degrees, o.predicted_kcal_mol, color=BLUE, lw=1.0, marker="s", ms=1.8,
                label="our head (frozen)")
        if len(z):
            ax.plot(z.grid_degrees, z.predicted_kcal_mol, color=ORANGE, lw=1.0, marker="^",
                    ms=1.8, label="DPA3 zero-shot")
        ax.set_xticks([-180, -90, 0, 90, 180]); ax.set_xlim(-185, 185)
        d = float(delta.get(mol, np.nan))
        verdict = "head better" if d < 0 else "head worse"
        ax.set_title(f"$\\bf{{{'abcdef'[index]}}}$  {verdict} by {abs(d):.2f}", loc="left")
        smiles = str(m.loc[mol, "smiles"]) if "smiles" in m.columns else ""
        ax.text(0.0, 1.008, smiles[:34], transform=ax.transAxes, fontsize=6.2,
                color=GREY, va="bottom")
        if index == 0:
            ax.legend(loc="upper right", fontsize=5.8)
        if index >= 3:
            ax.set_xlabel(r"driven dihedral ($\degree$)")
        if index % 3 == 0:
            ax.set_ylabel("relative energy (kcal mol$^{-1}$)")

    figure.text(0.0, -0.03,
                "OpenFF Gen3, held-out molecules from the completed folds. Left column is where the "
                "head wins by the largest margin; the rest span the barrier range. "
                "Our head sees ONE geometry and emits an analytic profile; DPA3 sees all 24.",
                fontsize=6.4, color=GREY, ha="left", va="top")
    figure.tight_layout(h_pad=3.0, w_pad=1.8)
    pathlib.Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.figure)
    print(f"drew {len(picks)} molecules -> {args.figure}")


if __name__ == "__main__":
    main()
