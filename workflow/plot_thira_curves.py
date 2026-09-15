"""Training curves for every fold and arm of the cluster campaign.

One panel per arm, one line per fold, so a fold that failed or diverged is
visible rather than averaged away. The flat-profile control is drawn as a
horizontal reference, because an arm whose validation sits on it has learned
nothing whatever its training loss says.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np

BLACK, GREY = "#000000", "#666666"
ARM_STYLE = {
    "frozen": ("#666666", "frozen encoder"),
    "finetuned_reference": ("#E69F00", "fine-tuned, reference conformer"),
    "finetuned_per_angle": ("#0072B2", "fine-tuned, every scan geometry"),
}
FOLD_ALPHA = [1.0, 0.8, 0.62, 0.46, 0.34]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--figure", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    matplotlib.rcParams.update({
        "figure.dpi": 200, "savefig.dpi": 400, "savefig.bbox": "tight",
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "xtick.direction": "in", "ytick.direction": "in",
        "legend.frameon": False, "lines.linewidth": 1.0,
        "axes.titlelocation": "left", "axes.titlepad": 13,
    })

    frames, rows = [], []
    for path in sorted(pathlib.Path(args.runs).glob("*/loss.csv")):
        table = pd.read_csv(path)
        if table.empty:
            continue
        table["run"] = path.parent.name
        frames.append(table)
        best = table.loc[table.val_rmse.idxmin()]
        rows.append({
            "run": path.parent.name,
            "encoder": table.encoder.iloc[0],
            "fold": table.fold.iloc[0],
            "epochs": len(table),
            "best_val_rmse": float(best.val_rmse),
            "best_epoch": int(best.epoch),
            "seconds": float(table.seconds.max()),
        })
    if not frames:
        raise SystemExit(f"no loss.csv under {args.runs}")
    history = pd.concat(frames, ignore_index=True)
    summary = pd.DataFrame(rows).sort_values(["encoder", "fold"])
    pathlib.Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary, index=False)

    arms = [a for a in ARM_STYLE if a in set(history.encoder)]
    figure, axes = plt.subplots(1, len(arms), figsize=(3.3 * len(arms), 3.1), sharey=True)
    axes = np.atleast_1d(axes)
    for index, arm in enumerate(arms):
        ax = axes[index]
        colour, label = ARM_STYLE[arm]
        subset = history[history.encoder == arm]
        for order, (fold, group) in enumerate(subset.groupby("fold")):
            group = group.sort_values("epoch")
            ax.plot(group.epoch, group.val_rmse, color=colour,
                    alpha=FOLD_ALPHA[order % len(FOLD_ALPHA)], label=str(fold))
            best = group.loc[group.val_rmse.idxmin()]
            ax.scatter([best.epoch], [best.val_rmse], s=18, color=colour,
                       edgecolor=BLACK, lw=0.4, zorder=5)
        ax.set_xlabel("epoch")
        if index == 0:
            ax.set_ylabel("validation RMSE (kcal mol$^{-1}$)")
        ax.set_title(f"$\\bf{{{'abcdef'[index]}}}$  {label}", loc="left")
        ax.text(0.0, 1.008, f"{subset.fold.nunique()} folds, "
                f"best {subset.val_rmse.min():.3f} kcal mol$^{{-1}}$",
                transform=ax.transAxes, fontsize=6.4, color=GREY, va="bottom")
        ax.legend(title="fold", loc="upper right", ncol=2, title_fontsize=6)

    figure.text(0.0, -0.03,
                "OpenFF Gen3, 870 molecules at B3LYP-D3BJ/DZVP, 5 scaffold-grouped folds. "
                "Validation molecules come from each fold's own training portion; "
                "filled point is the epoch kept. Resampling unit: the molecule.",
                fontsize=6.4, color=GREY, ha="left", va="top")
    figure.tight_layout(w_pad=2.0)
    pathlib.Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.figure)

    print(f"{len(summary)} runs, {len(history)} epochs total")
    for _, row in summary.iterrows():
        print(f"  {row.run:28s} {row.epochs:4d} ep  best {row.best_val_rmse:.4f} "
              f"@ {row.best_epoch:3d}  ({row.seconds/60:.1f} min)")


if __name__ == "__main__":
    main()
