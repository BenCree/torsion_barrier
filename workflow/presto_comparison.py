"""Our head on TorsionNet500, in the column presto, espaloma and Grappa report in.

WHAT THIS DECIDES. presto's Table 1 gives whole force field errors on
TorsionNet500 as a Jensen-Shannon distance between Boltzmann populations:
Sage 0.30, espaloma 0.20, presto 0.12, AceFF 2.0 0.09. Every number this project
has produced against that table has been on a DIFFERENT cohort, which makes the
comparison worth little. This evaluates the same trained model on TorsionNet500
itself, so the row is comparable in cohort.

WHAT IT STILL DOES NOT DECIDE, AND THE FIGURE SAYS SO. presto relaxes with the
force field at every scan angle and this does not: our head reads the QM
geometry and predicts its energy. Matching the cohort removes one of the two
caveats and not both. A reader who takes the reference lines as a like-for-like
ranking has been misled, so they are drawn as reference lines and labelled as a
different protocol rather than plotted as competing points.

LEAKAGE IS EXCLUDED UPSTREAM, AND COUNTED HERE. 50 of TorsionNet500's 497
molecules share an InChIKey with the training pool and are dropped by the run
itself, leaving 447. The count is printed rather than assumed, because a
benchmark that quietly kept a tenth of itself in training is the failure this
whole comparison would otherwise invite.

BOTH COHORTS, ONE MODEL. The held-out pool travels beside TorsionNet500 in
every table here. The two are scored by one model in one run, so a difference
between them is the cohorts differing and not two training runs differing.

THE CONTROLS TRAVEL TOO. MMFF94 and the flat profile are scored on the same
molecules with the same statistic. The flat profile's JS is the number a method
that predicts nothing achieves, and without it a JS of 0.28 has no scale.

MEDIANS. MMFF94's mean RMSE on these cohorts is in the hundreds of kcal/mol
because RDKit's perception fails on phosphates and charged species; those
molecules are counted and excluded from the MMFF94 row alone, never from ours.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np

KCAL_PER_K = 0.0019872041
PRESTO_TABLE1 = {"Sage": 0.30, "espaloma": 0.20, "presto": 0.12, "AceFF 2.0": 0.09}


def boltzmann(curve, kelvin):
    beta = 1.0 / (KCAL_PER_K * kelvin)
    w = np.exp(-beta * np.clip(curve - np.min(curve), 0.0, 700.0))
    total = w.sum()
    return w / total if total > 0 else np.full_like(w, 1.0 / w.size)


def jensen_shannon(p, q):
    m = 0.5 * (p + q)

    def kl(a, b):
        keep = a > 0
        return float(np.sum(a[keep] * np.log2(a[keep] / b[keep])))

    return float(np.sqrt(max(0.5 * kl(p, m) + 0.5 * kl(q, m), 0.0)))


def boot_median(values, rng, n=2000):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if values.size < 3:
        return float("nan"), float("nan"), float("nan"), int(values.size)
    draws = np.median(values[rng.integers(0, values.size, size=(n, values.size))], axis=1)
    return (float(np.median(values)), float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)), int(values.size))


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
    parser.add_argument("--arms", nargs="+", required=True,
                        help="name=directory, each holding per_molecule.csv "
                             "and profiles.csv")
    parser.add_argument("--table", required=True)
    parser.add_argument("--figure", required=True)
    parser.add_argument("--kelvin", type=float, default=500.0)
    parser.add_argument("--mmff-ceiling", type=float, default=100.0)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260916)
    args = parser.parse_args()

    matplotlib.rcParams.update({
        "figure.dpi": 200, "savefig.dpi": 400, "savefig.bbox": "tight",
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "xtick.direction": "in", "ytick.direction": "in", "legend.frameon": False,
        "axes.titlelocation": "left", "axes.titlepad": 13,
    })
    GREY = "#666666"
    rng = np.random.default_rng(args.seed)
    COHORT = {"external": "TorsionNet500", "pool": "our held-out pool"}

    rows, held = [], {}
    for arm, directory in named(args.arms).items():
        root = pathlib.Path(directory)
        molecules = pd.read_csv(root / "per_molecule.csv")
        molecules = molecules[~molecules.molecule_id.duplicated(keep=False)]
        molecules = molecules.set_index("molecule_id")
        profiles = pd.read_csv(root / "profiles.csv")

        js = {}
        for molecule_id, block in profiles.groupby("molecule_id"):
            block = block.sort_values("grid_degrees")
            qm = block.qm_kcal_mol.to_numpy(float)
            predicted = block.predicted_kcal_mol.to_numpy(float)
            if qm.size >= 4 and np.isfinite(qm).all() and np.isfinite(predicted).all():
                js[molecule_id] = jensen_shannon(boltzmann(qm, args.kelvin),
                                                 boltzmann(predicted, args.kelvin))
        molecules["js"] = pd.Series(js)
        held[arm] = molecules

        for cohort, block in molecules.groupby("cohort"):
            blown = int((block.rmse_mmff94 > args.mmff_ceiling).sum())
            mmff = block.rmse_mmff94.where(block.rmse_mmff94 <= args.mmff_ceiling)
            entries = (
                ("our head, JS", block.js, ""),
                ("our head, RMSE", block.rmse, " kcal/mol"),
                ("MMFF94, RMSE", mmff, " kcal/mol"),
                ("flat profile, RMSE", block.rmse_flat, " kcal/mol"),
            )
            # THE BARRIER SLOPE, because the scatter in panel c is a fan that
            # sits below the line and a median RMSE does not say so. A slope
            # below 1 is systematic undershoot: the head places the minima
            # correctly and compresses the amplitude, which is what least
            # squares does to a skewed target and is why the RMSE and the
            # population distance disagree about anomeric rotors.
            finite = block[["qm_barrier_kcal_mol",
                            "predicted_barrier_kcal_mol"]].dropna()
            slope = (float(np.polyfit(finite.qm_barrier_kcal_mol,
                                      finite.predicted_barrier_kcal_mol, 1)[0])
                     if len(finite) > 2 else float("nan"))
            print(f"\n  {arm}, {COHORT.get(cohort, cohort)} ({len(block)} molecules"
                  + (f", {blown} MMFF94 perception failures excluded from its row"
                     if blown else "") + "):")
            print(f"    barrier slope          {slope:7.3f}  "
                  f"(1.0 would be no systematic under or overshoot)")
            rows.append({"arm": arm, "cohort": cohort, "metric": "barrier slope",
                         "median": slope, "lo": float("nan"), "hi": float("nan"),
                         "n": len(finite)})
            for label, values, unit in entries:
                median, lo, hi, n = boot_median(values, rng, args.bootstrap)
                print(f"    {label:22s} {median:7.3f} [{lo:.3f}, {hi:.3f}]{unit}"
                      f"  n = {n}")
                rows.append({"arm": arm, "cohort": cohort, "metric": label,
                             "median": median, "lo": lo, "hi": hi, "n": n})

    frame = pd.DataFrame(rows)
    pathlib.Path(args.table).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.table, index=False)

    print(f"\n  presto's Table 1, TorsionNet500, WHOLE FORCE FIELD, relaxed at "
          f"every angle:")
    for name, value in PRESTO_TABLE1.items():
        print(f"    {name:12s} {value:.2f}")
    for arm in held:
        external = frame[(frame.arm == arm) & (frame.cohort == "external")
                         & (frame.metric == "our head, JS")]
        if not external.empty:
            row = external.iloc[0]
            inside = PRESTO_TABLE1["AceFF 2.0"] <= row["median"] <= PRESTO_TABLE1["Sage"]
            print(f"    {arm + ', same cohort':24s} {row['median']:.2f} "
                  f"[{row['lo']:.2f}, {row['hi']:.2f}]   "
                  f"{'inside' if inside else 'outside'} that range, "
                  f"NOT the same protocol")

    # ---- the figure ---------------------------------------------------------
    arms = list(held)
    figure, axes = plt.subplots(1, 2 + len(arms), figsize=(3.4 * (2 + len(arms)), 3.4))

    ax = axes[0]
    order, labels, colours = [], [], []
    for arm in arms:
        for cohort, colour in (("external", "#D55E00"), ("pool", "#0072B2")):
            block = held[arm][held[arm].cohort == cohort]
            if block.empty:
                continue
            order.append(block.js.dropna().to_numpy())
            labels.append(f"{COHORT.get(cohort, cohort)}\n{arm} (n = {len(block)})")
            colours.append(colour)
    parts = ax.violinplot(order, showextrema=False, widths=0.8)
    for body, colour in zip(parts["bodies"], colours):
        body.set_facecolor(colour); body.set_alpha(0.45); body.set_linewidth(0)
    for index, (values, colour) in enumerate(zip(order, colours), start=1):
        ax.plot([index], [np.median(values)], "o", ms=5, color=colour,
                markeredgecolor="white", markeredgewidth=0.8, zorder=4)
    for name, value in PRESTO_TABLE1.items():
        ax.axhline(value, lw=0.8, ls=(0, (4, 3)), color=GREY)
        ax.text(len(order) + 0.55, value, f" {name} {value:.2f}", fontsize=5.8,
                color=GREY, va="center")
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(labels, fontsize=5.8)
    ax.set_ylabel(f"Jensen-Shannon distance at {args.kelvin:.0f} K")
    ax.set_ylim(0, 1)
    ax.set_title("$\\bf{a}$  population overlap", loc="left")
    ax.text(0.0, 1.008, "dashed: presto's Table 1, a DIFFERENT protocol "
            "(relaxed at every angle)", transform=ax.transAxes, fontsize=6.2,
            color=GREY, va="bottom")

    ax = axes[1]
    for arm in arms:
        for cohort, colour in (("external", "#D55E00"), ("pool", "#0072B2")):
            block = held[arm][held[arm].cohort == cohort]
            if block.empty:
                continue
            values = np.sort(block.rmse.dropna().to_numpy())
            ax.plot(values, np.linspace(0, 1, values.size), lw=1.3, color=colour,
                    ls="-" if arm == arms[0] else (0, (4, 2)),
                    label=f"{COHORT.get(cohort, cohort)}, {arm}")
    flat = held[arms[0]].rmse_flat.dropna()
    ax.axvline(float(flat.median()), lw=0.9, ls=":", color=GREY)
    ax.text(float(flat.median()), 0.06, " flat profile", fontsize=5.8, color=GREY)
    ax.set_xlim(0, 6)
    ax.set_xlabel("profile RMSE (kcal mol$^{-1}$)")
    ax.set_ylabel("fraction of molecules at or below")
    ax.set_title("$\\bf{b}$  where the error sits", loc="left")
    ax.text(0.0, 1.008, "the whole distribution, not a median",
            transform=ax.transAxes, fontsize=6.2, color=GREY, va="bottom")
    ax.legend(loc="lower right")

    for offset, arm in enumerate(arms):
        ax = axes[2 + offset]
        block = held[arm][held[arm].cohort == "external"]
        ax.scatter(block.qm_barrier_kcal_mol, block.predicted_barrier_kcal_mol,
                   s=11, lw=0.3, facecolor="#D55E00", edgecolor="black", alpha=0.75)
        limit = float(np.nanpercentile(
            np.concatenate([block.qm_barrier_kcal_mol.to_numpy(float),
                            block.predicted_barrier_kcal_mol.to_numpy(float)]), 99))
        ax.plot([0, limit], [0, limit], lw=0.8, ls=(0, (4, 3)), color="black")
        ax.set_xlim(0, limit); ax.set_ylim(0, limit)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("QM barrier (kcal mol$^{-1}$)")
        ax.set_ylabel(f"predicted barrier, {arm}")
        ax.set_title(f"$\\bf{{{'cde'[offset]}}}$  TorsionNet500 barriers", loc="left")
        ax.text(0.0, 1.008, f"{len(block)} molecules, y = x drawn. Points below "
                f"the line are undershoots.", transform=ax.transAxes,
                fontsize=6.2, color=GREY, va="bottom")

    figure.text(0.0, -0.03, f"One model per arm, scoring both cohorts in one run. "
                f"TorsionNet500 is 497 scans; the 50 sharing an InChIKey with the "
                f"training pool are excluded by the run, leaving 447. Our head "
                f"reads the QM geometry at each angle and does not relax, so the "
                f"reference lines are a different protocol on the same cohort.",
                fontsize=6.4, color=GREY, ha="left", va="top")
    figure.tight_layout(w_pad=2.2)
    figure.savefig(args.figure)
    plt.close(figure)
    print(f"\n  wrote {args.figure}, {args.table}")


if __name__ == "__main__":
    main()
