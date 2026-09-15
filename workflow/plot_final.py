"""The 888 result in one figure: four arms, two pretrained potentials, 870 molecules.

Every number in RESULT_stage3.md is here so the conclusion can be checked against
the data. The horizontal band on panels a and b is the LABEL UNCERTAINTY: two DFT
references for the same molecules differ by 0.767 kcal/mol on barrier height while
the labels reproduce themselves to 0.0008, so a difference smaller than that band
is smaller than not knowing which reference was meant. It is drawn because four of
today's comparisons fell inside it.
"""

from __future__ import annotations

import argparse
import glob
import pathlib

import numpy as np

BLACK, GREY = "#000000", "#666666"
BLUE, ORANGE, GREEN, VERM, PURPLE = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"
ARMS = [("frozen", "frozen\n1 conformer", GREY),
        ("finetuned_reference", "fine-tuned\n1 conformer", ORANGE),
        ("frozen_per_angle", "frozen\n24 conformers", GREEN),
        ("finetuned_per_angle", "fine-tuned\n24 conformers", BLUE)]
REF_UNCERTAINTY = 0.767


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from scipy.stats import binomtest

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--dpa3", required=True)
    parser.add_argument("--mace", required=True)
    parser.add_argument("--figure", required=True)
    args = parser.parse_args()

    matplotlib.rcParams.update({
        "figure.dpi": 200, "savefig.dpi": 400, "savefig.bbox": "tight",
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.3,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "xtick.direction": "in", "ytick.direction": "in", "legend.frameon": False,
        "lines.linewidth": 1.0, "axes.titlelocation": "left", "axes.titlepad": 14,
    })

    def panel(ax, letter, title, subtitle):
        ax.set_title(f"$\\bf{{{letter}}}$  {title}", loc="left")
        ax.text(0.0, 1.012, subtitle, transform=ax.transAxes, fontsize=6.2,
                color=GREY, va="bottom")

    o = pd.concat([pd.read_csv(f) for f in glob.glob(f"{args.runs}/*/per_molecule.csv")])
    o["barrier_error"] = (o.predicted_barrier_kcal_mol - o.qm_barrier_kcal_mol).abs()
    d = pd.read_csv(args.dpa3).set_index("molecule_id")
    mc = pd.read_csv(args.mace)
    mc = mc[mc.potential == "mace_off23_large"].set_index("molecule_id")

    figure, axes = plt.subplots(2, 2, figsize=(7.6, 6.8))

    # a. every arm and both potentials, one point per molecule
    ax = axes[0, 0]
    labels, positions = [], []
    for i, (arm, label, colour) in enumerate(ARMS):
        g = o[o.encoder == arm]
        jitter = np.random.default_rng(0).normal(0, 0.055, len(g))
        ax.scatter(np.full(len(g), i) + jitter, np.clip(g.rmse, 0, 5), s=3, lw=0,
                   alpha=0.18, color=colour)
        ax.scatter([i], [g.rmse.median()], s=58, marker="D", facecolor=colour,
                   edgecolor=BLACK, lw=0.8, zorder=6)
        labels.append(label); positions.append(i)
    for j, (name, frame, colour) in enumerate((("DPA3\nzero-shot", d, VERM),
                                               ("MACE-OFF23 L\nzero-shot", mc, PURPLE))):
        i = len(ARMS) + j
        jitter = np.random.default_rng(1).normal(0, 0.055, len(frame))
        ax.scatter(np.full(len(frame), i) + jitter, np.clip(frame.rmse, 0, 5), s=3, lw=0,
                   alpha=0.18, color=colour)
        ax.scatter([i], [frame.rmse.median()], s=58, marker="D", facecolor=colour,
                   edgecolor=BLACK, lw=0.8, zorder=6)
        labels.append(name); positions.append(i)
    ax.axhline(3.122, color=BLACK, lw=0.8, ls=":", label="flat profile, 3.12")
    ax.set_xticks(positions); ax.set_xticklabels(labels, fontsize=6.0)
    ax.set_ylabel("profile RMSE (kcal mol$^{-1}$)"); ax.set_ylim(0, 5)
    panel(ax, "a", "Every arm, one point per held-out molecule",
          "diamond is the median. 870 molecules, 5 scaffold folds")
    ax.legend(loc="upper right")

    # b. the arms alone, against the label uncertainty
    ax = axes[0, 1]
    meds = [o[o.encoder == a].rmse.median() for a, _, _ in ARMS]
    base = meds[0]
    ax.axhspan(base - REF_UNCERTAINTY/2, base + REF_UNCERTAINTY/2, color=GREY, alpha=0.16,
               label=f"within the label uncertainty ({REF_UNCERTAINTY:.2f})")
    for i, ((arm, label, colour), med) in enumerate(zip(ARMS, meds)):
        ax.bar(i, med, color=colour, edgecolor=BLACK, lw=0.6, width=0.62)
        ax.text(i, med + 0.03, f"{med:.3f}", ha="center", fontsize=6.4)
    ax.axhline(d.rmse.median(), color=VERM, lw=1.3, ls="--",
               label=f"DPA3 zero-shot, {d.rmse.median():.3f}")
    ax.axhline(mc.rmse.median(), color=PURPLE, lw=1.3, ls="--",
               label=f"MACE-OFF23 L, {mc.rmse.median():.3f}")
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels([l for _, l, _ in ARMS], fontsize=6.0)
    ax.set_ylabel("median profile RMSE (kcal mol$^{-1}$)")
    panel(ax, "b", "All four arms are the same model",
          "the spread across arms is 0.12; the label uncertainty is 0.77")
    ax.legend(loc="lower left")

    # c. paired against the best potential
    ax = axes[1, 0]
    best = o[o.encoder == "finetuned_per_angle"].set_index("molecule_id")
    sh = best.index.intersection(mc.index)
    x, y = mc.loc[sh, "rmse"], best.loc[sh, "rmse"]
    lim = float(np.percentile(np.concatenate([x, y]), 98))
    ax.scatter(x, y, s=10, lw=0.25, facecolor=BLUE, edgecolor=BLACK, alpha=0.5, zorder=3)
    ax.plot([0, lim], [0, lim], color=BLACK, lw=0.8, ls=(0, (4, 3)), label="y = x")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal", adjustable="box")
    w = int((y < x).sum()); p = binomtest(w, len(sh), 0.5).pvalue
    ax.set_xlabel("MACE-OFF23 large, zero-shot (kcal mol$^{-1}$)")
    ax.set_ylabel("our best arm (kcal mol$^{-1}$)")
    panel(ax, "c", "Against the best pretrained potential",
          f"we win {w} of {len(sh)} ({100*w/len(sh):.0f}%), p = {p:.1e}")
    ax.legend(loc="lower right")

    # d. NABH, the tail metric
    ax = axes[1, 1]
    names = [l.replace("\n", " ") for _, l, _ in ARMS] + ["DPA3 0-shot", "MACE-OFF23 L"]
    vals = [100*(o[o.encoder == a].barrier_error > 1).mean() for a, _, _ in ARMS]
    vals += [100*(d.barrier_error > 1).mean(), 100*(mc.barrier_error > 1).mean()]
    colours = [c for _, _, c in ARMS] + [VERM, PURPLE]
    ax.barh(range(len(vals)), vals, color=colours, edgecolor=BLACK, lw=0.6)
    for i, v in enumerate(vals):
        ax.text(v + 1, i, f"{v:.0f}%", va="center", fontsize=6.4)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=6.0)
    ax.invert_yaxis(); ax.set_xlabel("barriers wrong by more than 1 kcal mol$^{-1}$ (%)")
    ax.set_xlim(0, 72)
    panel(ax, "d", "The tail: how often the rotamer would be called wrong",
          "1 kcal/mol is where a conformational preference flips")

    figure.text(0.0, -0.02,
                "OpenFF Gen3, 870 molecules at B3LYP-D3BJ/DZVP, 24 scan points each, 5 "
                "scaffold-grouped folds, every molecule held out exactly once per arm. "
                "Resampling unit: the molecule. Pretrained potentials are untrained on this data.",
                fontsize=6.3, color=GREY, ha="left", va="top")
    figure.tight_layout(h_pad=3.2, w_pad=2.4)
    pathlib.Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.figure)
    print(f"  -> {args.figure}")


if __name__ == "__main__":
    main()
