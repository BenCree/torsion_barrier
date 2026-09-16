"""The learning curves of the full-pool arms, and where each one stopped.

WHAT A PER-MOLECULE TABLE CANNOT SAY. It reports the error at the kept epoch.
It does not say whether the validation curve was flat or still falling there,
whether the kept epoch was a plateau or a lucky spike, or how far training and
validation had separated. Those decide whether a headline number is a converged
result or an accident of when early stopping fired.

THE COMPARISON THIS EXISTS FOR. Fine-tuning the encoder helped at 855 molecules
and hurt at 15,468. If that reversal is overfitting it appears here as the
validation curve rising while the training loss keeps falling. If instead the
fine-tuned run simply had not converged, the validation curve is still falling
at the last epoch, and the honest statement is that the arm was stopped early
rather than beaten.

THE TEST LINE MUST BE THE SAME STATISTIC AS THE CURVE, AND IT NEARLY WAS NOT.
`val_rmse` is the MEAN over scans of each scan's own RMSE. The headline test
number everywhere else in this project is the MEDIAN over molecules, 1.166
against a mean of 1.728, because the distribution has a long tail. Drawing the
median as a reference line under a curve of means invites the reader to see a
train-test gap that is a difference of statistic and nothing else. The mean is
drawn, and the median is drawn beside it and labelled.

BOTH CURVES ON ONE AXIS, IN THE SAME UNITS. `train_mse` is a mean squared error
and `val_rmse` a root mean squared error, so plotting them together without
taking the square root of the first compares a square with a root and makes the
training loss look far smaller than it is. The square root is taken here.

Arms are named on the command line as `name=path`, so a third arm is an argument
rather than an edit.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np


def named(pairs):
    out = {}
    for item in pairs:
        name, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"expected name=path, got {item!r}")
        out[name] = path
    return out


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loss", nargs="+", required=True, help="name=path")
    parser.add_argument("--per-molecule", nargs="+", required=True, help="name=path")
    parser.add_argument("--figure", required=True)
    parser.add_argument("--table", required=True)
    args = parser.parse_args()

    matplotlib.rcParams.update({
        "figure.dpi": 200, "savefig.dpi": 400, "savefig.bbox": "tight",
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "xtick.direction": "in", "ytick.direction": "in", "legend.frameon": False,
        "axes.titlelocation": "left", "axes.titlepad": 12,
    })
    GREY = "#666666"
    COLOURS = {"frozen": "#0072B2", "finetuned": "#D55E00"}

    losses = {k: pd.read_csv(v) for k, v in named(args.loss).items()}
    finals = {k: pd.read_csv(v) for k, v in named(args.per_molecule).items()}

    figure, axes = plt.subplots(1, 3, figsize=(10.4, 3.3))
    summary = []

    for index, (name, loss) in enumerate(losses.items()):
        colour = COLOURS.get(name, f"C{index}")
        loss = loss.sort_values("epoch")
        train = np.sqrt(loss.train_mse.to_numpy(float))
        val = loss.val_rmse.to_numpy(float)
        best_epoch = int(loss.best_epoch.iloc[-1])
        best_val = float(loss.best_val_rmse.iloc[-1])
        test_mean = finals[name].rmse.mean() if name in finals else float("nan")
        test_median = finals[name].rmse.median() if name in finals else float("nan")
        hours = (finals[name].train_seconds.iloc[0] / 3600.0
                 if name in finals else float("nan"))

        axes[0].plot(loss.epoch, train, lw=1.2, color=colour, label=f"{name}, training")
        axes[0].plot(loss.epoch, val, lw=1.2, ls=(0, (4, 2)), color=colour,
                     label=f"{name}, validation")
        axes[0].plot([best_epoch], [best_val], "o", ms=5, color=colour,
                     markeredgecolor="white", markeredgewidth=0.8, zorder=3)

        axes[1].plot(loss.epoch, val, lw=1.2, color=colour, label=name)
        axes[1].plot([best_epoch], [best_val], "o", ms=5, color=colour,
                     markeredgecolor="white", markeredgewidth=0.8, zorder=3)
        axes[1].axhline(test_mean, lw=0.9, ls=(0, (1, 2)), color=colour)
        axes[1].axhline(test_median, lw=0.9, ls=(0, (5, 2, 1, 2)), color=colour,
                        alpha=0.55)

        axes[2].plot(loss.seconds / 3600.0, val, lw=1.2, color=colour, label=name)

        # Is the validation curve still falling where training stopped? Compare
        # the last tenth of epochs with the tenth before it, on the validation
        # curve only: a run stopped mid-descent is not a run that was beaten.
        tail = max(len(val) // 10, 3)
        drift = float(np.median(val[-tail:]) - np.median(val[-2 * tail:-tail]))
        summary.append({
            "arm": name, "epochs_run": int(loss.epoch.max()),
            "kept_epoch": best_epoch, "best_validation_rmse": best_val,
            "final_validation_rmse": float(val[-1]),
            "test_mean_rmse": float(test_mean),
            "test_median_rmse": float(test_median), "train_hours": float(hours),
            "epochs_after_best": int(loss.epoch.max()) - best_epoch,
            "validation_drift_last_tenth": drift,
            "still_falling": bool(drift < -0.005),
        })

    axes[0].set_title("$\\bf{a}$  loss, both curves", loc="left")
    axes[0].text(0.0, -0.30, "", transform=axes[0].transAxes)
    axes[0].text(0.0, 1.008, "solid: training. dashed: validation. marker: the "
                 "kept epoch.", transform=axes[0].transAxes, fontsize=6.2,
                 color=GREY, va="bottom")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("RMSE (kcal mol$^{-1}$)")
    axes[0].set_yscale("log")
    axes[0].legend(loc="upper right")

    axes[1].set_title("$\\bf{b}$  validation only", loc="left")
    axes[1].text(0.0, 1.008, "dotted: the MEAN test RMSE, the same statistic the "
                 "curve plots. dot-dashed: the median.",
                 transform=axes[1].transAxes, fontsize=6.2, color=GREY, va="bottom")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("validation RMSE (kcal mol$^{-1}$)")
    axes[1].legend(loc="upper right")

    axes[2].set_title("$\\bf{c}$  what the wall clock bought", loc="left")
    axes[2].text(0.0, 1.008, "the same validation curve against hours, not epochs",
                 transform=axes[2].transAxes, fontsize=6.2, color=GREY, va="bottom")
    axes[2].set_xlabel("training time (hours)")
    axes[2].set_ylabel("validation RMSE (kcal mol$^{-1}$)")
    axes[2].legend(loc="upper right")

    frame = pd.DataFrame(summary)
    figure.text(0.0, -0.04, "Full pool, 15,468 molecules split by hashed molecular "
                "identity, one espaloma stage-1 encoder. Early stopping on the "
                "validation RMSE, which is the mean over scans of each scan's own "
                "RMSE; the kept epoch is the marker. The training curve is the "
                "square root of the training MSE, so both axes carry kcal/mol.",
                fontsize=6.4, color=GREY, ha="left", va="top")
    figure.tight_layout(w_pad=2.2)
    pathlib.Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.figure)
    plt.close(figure)

    pathlib.Path(args.table).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.table, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(frame.to_string(index=False))
    for row in summary:
        if row["still_falling"]:
            print(f"\n  {row['arm']}: the validation curve was STILL FALLING at "
                  f"the last epoch ({row['validation_drift_last_tenth']:+.4f} "
                  f"over the last tenth). It was stopped, not beaten.")
    print(f"\n  wrote {args.figure}, {args.table}")


if __name__ == "__main__":
    main()
