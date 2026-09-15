"""Before and after, drawn from whatever the run has written so far.

Reads the per-epoch history as it is being written, so the curves can be looked
at during a run rather than only after one. Three figures:

  training curves     train MSE and validation RMSE per epoch, per arm, with the
                      kept epoch marked. This is the figure that says whether an
                      arm is learning or memorising.
  before and after    total RMSE per arm, and the paired per-molecule change on
                      held-out molecules. Paired, because the arms share a split.
  profiles            the QM scan with every arm drawn on it, for held-out
                      molecules across the barrier range. Before and after on one
                      axis, against the truth.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np

BLACK, ORANGE, SKY = "#000000", "#E69F00", "#56B4E9"
GREEN, BLUE, VERMILLION, PURPLE, GREY = "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#666666"

ARM_STYLE = {
    "frozen": (GREY, "frozen encoder"),
    "finetuned_reference": (ORANGE, "fine-tuned, reference conformer"),
    "finetuned_per_angle": (BLUE, "fine-tuned, every scan geometry"),
}


def _style():
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "figure.dpi": 200, "savefig.dpi": 400, "savefig.bbox": "tight",
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "legend.frameon": False, "lines.linewidth": 1.0,
        "axes.titlelocation": "left", "axes.titlepad": 13,
    })


def _panel(ax, letter, title, subtitle):
    ax.set_title(f"$\\bf{{{letter}}}$  {title}", loc="left")
    ax.text(0.0, 1.008, subtitle, transform=ax.transAxes, fontsize=6.4,
            color=GREY, va="bottom", ha="left")


def curves(history, path):
    import matplotlib.pyplot as plt

    _style()
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for arm, (colour, label) in ARM_STYLE.items():
        subset = history[history["encoder"] == arm]
        if subset.empty:
            continue
        axes[0].plot(subset["epoch"], subset["train_mse"], color=colour, label=label)
        axes[1].plot(subset["epoch"], subset["val_rmse"], color=colour, label=label)
        best = subset.loc[subset["val_rmse"].idxmin()]
        axes[1].scatter([best["epoch"]], [best["val_rmse"]], s=26, color=colour,
                        edgecolor=BLACK, lw=0.5, zorder=5)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("training MSE (kcal$^2$ mol$^{-2}$)")
    _panel(axes[0], "a", "Training loss", "60 training molecules")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("validation RMSE (kcal mol$^{-1}$)")
    _panel(axes[1], "b", "Validation, which chooses the epoch",
           "15 molecules held out of TRAIN; filled point is the epoch kept")
    axes[1].legend(loc="upper right")
    figure.tight_layout(w_pad=2.4)
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)


def before_after(per_molecule, path):
    import matplotlib.pyplot as plt
    from scipy.stats import binomtest

    _style()
    figure, axes = plt.subplots(1, 3, figsize=(9.6, 3.4))
    arms = [a for a in ARM_STYLE if a in set(per_molecule["encoder"])]

    # a. total RMSE per arm and split, points not bars
    ax = axes[0]
    positions, labels = [], []
    for index, arm in enumerate(arms):
        colour = ARM_STYLE[arm][0]
        for offset, (split, marker) in enumerate((("train", "o"), ("test", "^"))):
            subset = per_molecule[
                (per_molecule["encoder"] == arm) & (per_molecule["split"] == split)
            ]
            x = index + (offset - 0.5) * 0.34
            ax.scatter(np.full(len(subset), x) + np.random.default_rng(0).normal(0, 0.03, len(subset)),
                       subset["rmse"], s=7, lw=0, alpha=0.35, color=colour)
            ax.scatter([x], [subset["rmse"].mean()], s=42, marker=marker,
                       facecolor=colour, edgecolor=BLACK, lw=0.7, zorder=5)
        positions.append(index)
        labels.append(ARM_STYLE[arm][1].replace(", ", "\n"))
    reference = per_molecule[per_molecule["split"] == "test"]
    if "rmse_mmff94" in reference and reference["rmse_mmff94"].notna().any():
        ax.axhline(reference["rmse_mmff94"].dropna().mean(), color=VERMILLION,
                   lw=0.9, ls="--", label="MMFF94, mean")
    ax.axhline(reference["rmse_flat"].mean(), color=GREY, lw=0.9, ls=":",
               label="flat profile, mean")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=6.2)
    ax.set_ylabel("profile RMSE (kcal mol$^{-1}$)")
    _panel(ax, "a", "Total RMSE",
           "one small point per molecule; large = mean. circle train, triangle test")
    ax.legend(loc="upper right")

    # b and c. paired change on held-out molecules
    base = per_molecule[(per_molecule["encoder"] == "frozen") & (per_molecule["split"] == "test")]
    base = base.set_index("molecule_id")
    for index, arm in enumerate([a for a in arms if a != "frozen"][:2]):
        ax = axes[1 + index]
        after = per_molecule[
            (per_molecule["encoder"] == arm) & (per_molecule["split"] == "test")
        ].set_index("molecule_id")
        shared = base.index.intersection(after.index)
        x, y = base.loc[shared, "rmse"], after.loc[shared, "rmse"]
        limit = float(np.ceil(max(x.max(), y.max())))
        ax.scatter(x, y, s=18, lw=0.4, facecolor=ARM_STYLE[arm][0],
                   edgecolor=BLACK, alpha=0.85, zorder=3)
        ax.plot([0, limit], [0, limit], color=BLACK, lw=0.7, ls=(0, (4, 3)), zorder=1)
        ax.set_xlim(0, limit); ax.set_ylim(0, limit); ax.set_aspect("equal", adjustable="box")
        wins = int((y < x).sum())
        p = binomtest(wins, len(shared), 0.5).pvalue if len(shared) else float("nan")
        ax.set_xlabel("frozen encoder, RMSE (kcal mol$^{-1}$)")
        ax.set_ylabel(f"{ARM_STYLE[arm][1]}, RMSE")
        _panel(ax, "bc"[index], ARM_STYLE[arm][1].capitalize(),
               f"better below the line: {wins} of {len(shared)} held out, p = {p:.3f}")

    figure.text(0.0, -0.02,
                "OpenFF Gen3 torsiondrives, B3LYP-D3BJ/DZVP. Arms share one split, "
                "one head initialisation and one seed; only the encoder differs.",
                fontsize=6.4, color=GREY, ha="left", va="top")
    figure.tight_layout(w_pad=2.6)
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)


def profiles_figure(profiles, per_molecule, path):
    import matplotlib.pyplot as plt

    _style()
    held = per_molecule[
        (per_molecule["split"] == "test") & (per_molecule["encoder"] == "frozen")
    ].sort_values("qm_barrier_kcal_mol")
    if held.empty:
        return
    picks = held.iloc[np.linspace(0, len(held) - 1, min(6, len(held))).astype(int)]

    figure, axes = plt.subplots(2, 3, figsize=(7.2, 4.6), sharex=True)
    for index, (ax, (_, row)) in enumerate(zip(axes.flat, picks.iterrows())):
        scan = profiles[profiles["molecule_id"] == row["molecule_id"]]
        qm = scan[scan["encoder"] == "frozen"].sort_values("grid_degrees")
        ax.plot(qm["grid_degrees"], qm["qm_kcal_mol"], color=BLACK, lw=1.3,
                marker="o", ms=2.4, label="QM", zorder=5)
        for arm, (colour, label) in ARM_STYLE.items():
            arm_scan = scan[scan["encoder"] == arm].sort_values("grid_degrees")
            if arm_scan.empty:
                continue
            ax.plot(arm_scan["grid_degrees"], arm_scan["predicted_kcal_mol"],
                    color=colour, lw=1.0, marker="s", ms=1.8, alpha=0.9, label=label)
        ax.set_xticks([-180, -90, 0, 90, 180]); ax.set_xlim(-185, 185)
        smiles = str(row["smiles"])
        _panel(ax, "abcdef"[index],
               f"QM barrier {row['qm_barrier_kcal_mol']:.1f} kcal mol$^{{-1}}$",
               smiles[:30] + ("..." if len(smiles) > 30 else ""))
        if index == 0:
            ax.legend(loc="upper right", fontsize=5.6)
        if index >= 3:
            ax.set_xlabel(r"driven dihedral ($\degree$)")
        if index % 3 == 0:
            ax.set_ylabel("relative energy (kcal mol$^{-1}$)")
    figure.text(0.0, -0.02,
                "Six held-out molecules spanning the barrier range. Points are the 24 "
                "optimised scan geometries.", fontsize=6.4, color=GREY, ha="left", va="top")
    figure.tight_layout(h_pad=3.0, w_pad=1.8)
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)


def main():
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", required=True)
    parser.add_argument("--per-molecule")
    parser.add_argument("--profiles")
    parser.add_argument("--curves", required=True)
    parser.add_argument("--before-after")
    parser.add_argument("--profile-figure")
    parser.add_argument("--summary")
    args = parser.parse_args()

    history = pd.read_csv(args.history)
    curves(history, args.curves)
    print(f"curves: {len(history)} epochs logged across "
          f"{history['encoder'].nunique()} arm(s) -> {args.curves}")

    if args.per_molecule and pathlib.Path(args.per_molecule).exists():
        per_molecule = pd.read_csv(args.per_molecule)
        if args.before_after:
            before_after(per_molecule, args.before_after)
        if args.profiles and args.profile_figure and pathlib.Path(args.profiles).exists():
            profiles_figure(pd.read_csv(args.profiles), per_molecule, args.profile_figure)

        lines = ["encoder,split,n,rmse_mean,rmse_median,rho_median,kept_epoch"]
        print()
        for arm in [a for a in ARM_STYLE if a in set(per_molecule["encoder"])]:
            for split in ("train", "validation", "test"):
                subset = per_molecule[
                    (per_molecule["encoder"] == arm) & (per_molecule["split"] == split)
                ]
                if subset.empty:
                    continue
                epoch = int(subset["kept_epoch"].iloc[0]) if "kept_epoch" in subset else -1
                lines.append(f"{arm},{split},{len(subset)},{subset['rmse'].mean():.4f},"
                             f"{subset['rmse'].median():.4f},{subset['rho'].median():.4f},{epoch}")
                print(f"  {arm:22s} {split:10s} n={len(subset):3d}  "
                      f"RMSE mean {subset['rmse'].mean():6.3f} median {subset['rmse'].median():6.3f}"
                      f"  rho median {subset['rho'].median():+.3f}  kept epoch {epoch}")
        if args.summary:
            pathlib.Path(args.summary).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
