"""The four panels that carry the stage 3 verdict.

Every claim in RESULT_stage3.md is one of these, so a reader can check the
conclusion against the data rather than against the prose.

  a  the paired comparison. One point is one held-out molecule, our head against
     the potential it is built on, on identical rows, with y = x.
  b  the distributions, because the median hid that a third of molecules are
     already under 1 kcal/mol and that our tenth percentile is its median.
  c  predicted barrier against QM barrier. Slope below 1 with a positive
     intercept is regression toward the mean, which is under-fitting and is why
     no global rescale helps.
  d  data scaling. Two points and the extrapolation they imply, drawn on log-log
     so the exponent is the slope.
"""

from __future__ import annotations

import argparse
import glob
import pathlib

import numpy as np

BLACK, GREY = "#000000", "#666666"
BLUE, ORANGE, GREEN, VERM = "#0072B2", "#E69F00", "#009E73", "#D55E00"


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from scipy.stats import binomtest, linregress

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--zeroshot", required=True)
    parser.add_argument("--figure", required=True)
    args = parser.parse_args()

    matplotlib.rcParams.update({
        "figure.dpi": 200, "savefig.dpi": 400, "savefig.bbox": "tight",
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "xtick.direction": "in", "ytick.direction": "in", "legend.frameon": False,
        "lines.linewidth": 1.0, "axes.titlelocation": "left", "axes.titlepad": 13,
    })

    def panel(ax, letter, title, subtitle):
        ax.set_title(f"$\\bf{{{letter}}}$  {title}", loc="left")
        ax.text(0.0, 1.008, subtitle, transform=ax.transAxes, fontsize=6.3,
                color=GREY, va="bottom")

    ours = pd.concat([pd.read_csv(f) for f in glob.glob(f"{args.runs}/*/per_molecule.csv")])
    zs = pd.read_csv(args.zeroshot).set_index("molecule_id")
    figure, axes = plt.subplots(2, 2, figsize=(7.4, 6.6))

    # a. paired scatter, best arm available
    arm = "finetuned_per_angle" if "finetuned_per_angle" in set(ours.encoder) else "frozen"
    o = ours[ours.encoder == arm].set_index("molecule_id")
    shared = o.index.intersection(zs.index)
    x, y = zs.loc[shared, "rmse"], o.loc[shared, "rmse"]
    ax = axes[0, 0]
    lim = float(np.percentile(np.concatenate([x, y]), 98))
    ax.scatter(x, y, s=11, lw=0.3, facecolor=BLUE, edgecolor=BLACK, alpha=0.55, zorder=3)
    ax.plot([0, lim], [0, lim], color=BLACK, lw=0.8, ls=(0, (4, 3)), zorder=1, label="y = x")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal", adjustable="box")
    wins = int((y < x).sum())
    p = binomtest(wins, len(shared), 0.5).pvalue
    ax.set_xlabel("DPA3 zero-shot, profile RMSE (kcal mol$^{-1}$)")
    ax.set_ylabel(f"our head, profile RMSE (kcal mol$^{{-1}}$)")
    panel(ax, "a", "Paired on identical held-out molecules",
          f"{arm}, n={len(shared)}. below the line we win: {wins} ({100*wins/len(shared):.0f}%), p={p:.1e}")
    ax.legend(loc="lower right")

    # b. distributions
    ax = axes[0, 1]
    bins = np.linspace(0, 4, 45)
    for d, colour, label in ((zs.loc[shared, "rmse"], ORANGE, "DPA3 zero-shot"),
                            (o.loc[shared, "rmse"], BLUE, "our head")):
        ax.hist(np.clip(d, 0, 4), bins=bins, histtype="step", lw=1.3, color=colour,
                label=f"{label}, median {d.median():.2f}")
    ax.axvline(1.0, color=GREY, lw=0.7, ls=":")
    ax.text(1.05, ax.get_ylim()[1]*0.92, "1 kcal mol$^{-1}$", fontsize=6, color=GREY)
    ax.set_xlabel("profile RMSE (kcal mol$^{-1}$)"); ax.set_ylabel("molecules")
    panel(ax, "b", "The whole distribution, not the median",
          f"under 1: ours {100*(o.loc[shared,'rmse']<1).mean():.0f}%, "
          f"potential {100*(zs.loc[shared,'rmse']<1).mean():.0f}%")
    ax.legend(loc="upper right")

    # c. barrier calibration
    ax = axes[1, 0]
    for d, colour, label in ((ours[ours.encoder == arm], BLUE, "our head"),
                            (zs.reset_index(), ORANGE, "DPA3 zero-shot")):
        d = d.dropna(subset=["predicted_barrier_kcal_mol", "qm_barrier_kcal_mol"])
        s = linregress(d.qm_barrier_kcal_mol, d.predicted_barrier_kcal_mol)
        ax.scatter(d.qm_barrier_kcal_mol, d.predicted_barrier_kcal_mol, s=7, lw=0,
                   alpha=0.35, color=colour)
        xs = np.linspace(0, d.qm_barrier_kcal_mol.max(), 50)
        ax.plot(xs, s.intercept + s.slope*xs, color=colour, lw=1.4,
                label=f"{label}: slope {s.slope:.3f}")
    top = float(ours.qm_barrier_kcal_mol.max())
    ax.plot([0, top], [0, top], color=BLACK, lw=0.8, ls=(0, (4, 3)), label="y = x")
    ax.set_xlabel("QM barrier (kcal mol$^{-1}$)"); ax.set_ylabel("predicted barrier")
    panel(ax, "c", "Regression toward the mean, not a scale error",
          "slope below 1 with a positive intercept; a global rescale changes RMSE by 0.001")
    ax.legend(loc="upper left")

    # d. data scaling
    ax = axes[1, 1]
    n = np.array([60.0, 604.0]); v = np.array([1.5711, 1.4103])
    a = -np.log(v[1]/v[0])/np.log(n[1]/n[0])
    xs = np.logspace(np.log10(50), np.log10(2e5), 60)
    ax.plot(xs, v[0]*(xs/n[0])**(-a), color=BLUE, lw=1.3,
            label=f"fitted power law, a = {a:.3f}")
    ax.scatter(n, v, s=42, facecolor=BLUE, edgecolor=BLACK, lw=0.7, zorder=5,
               label="measured")
    ax.axhline(0.521, color=ORANGE, lw=1.2, ls="--", label="DPA3 zero-shot, 0.521")
    ax.axvline(13167, color=GREY, lw=0.9, ls=":")
    ax.text(13167*1.1, 1.8, "entire usable\nQCArchive pool", fontsize=6, color=GREY)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("training molecules"); ax.set_ylabel("best validation RMSE (kcal mol$^{-1}$)")
    panel(ax, "d", "Data will not close the gap",
          f"10x the data bought 10.2%; molecular ML typically has a = 0.2 to 0.5")
    ax.legend(loc="lower left")

    figure.text(0.0, -0.02,
                "OpenFF Gen3, 870 molecules at B3LYP-D3BJ/DZVP, 5 scaffold-grouped folds, "
                "every molecule held out exactly once per arm. Resampling unit: the molecule.",
                fontsize=6.4, color=GREY, ha="left", va="top")
    figure.tight_layout(h_pad=3.0, w_pad=2.4)
    pathlib.Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.figure)
    print(f"  arm shown: {arm}, n={len(shared)}, we win {wins} ({100*wins/len(shared):.0f}%), p={p:.2e}")
    print(f"  scaling exponent a = {a:.3f}")


if __name__ == "__main__":
    main()
