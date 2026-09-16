"""Which rotor classes the trained head cannot do, and how the profiles compare to presto.

THE QUESTION. A median RMSE over a held-out pool is one number over a mixture.
If the error is concentrated in nameable rotor classes, the honest statement is
an applicability domain: works except on these. If it is spread evenly, there is
no domain to state and the tail is not nameable. Both are results.

WHAT IS COMPARED, AND AGAINST WHAT.

  Per class, the median profile RMSE and the median Jensen-Shannon distance
  between the QM and predicted Boltzmann populations at 500 K, each with a
  bootstrap interval resampled over MOLECULES, and the difference between that
  class and every molecule not in it.

  THREE CONTROLS. `rmse_flat`, the RMS of the QM profile itself, which is what
  predicting a flat curve scores and so is the null any method must beat.
  MMFF94 as the standard. And PERMUTED CLASS LABELS: the class sizes are kept
  and the labels shuffled, which says how large a class difference this cohort
  produces from nothing. Without it a class of 40 molecules sitting 0.3 above
  the pool cannot be told from the width of its own interval.

WHY JS AT 500 K AND NOT RMSE ALONE. presto's paper argues that RMSE "is
sensitive to barrier height errors, which do not affect equilibrium
distributions", and reports JS. Its Table 1, on TorsionNet500, gives whole
force field errors of Sage 0.30, espaloma 0.20, presto 0.12, AceFF 2.0 0.09.
Those are a DIFFERENT cohort and a different protocol: presto relaxes with the
force field at every angle and this does not. The number here is comparable in
kind, not like for like, and the print-out says so rather than leaving a reader
to assume otherwise.

IRREGULAR SCANS ARE A PROPERTY OF THE DATA, NOT OF THE MODEL, AND THEY ARE NOT
SPREAD EVENLY OVER THE CLASSES. A normal torsiondrive here is 24 points over
345 degrees. 187 of 3,885 held-out scans are not: some cover 90 degrees and
climb to 50 kcal/mol across them, which is a scan that hit a clash rather than
a torsion profile, and no model should be scored on reproducing it. They are
4.8 percent of the pool but 24.1 percent of amide rotors and 17.3 percent of
biaryls, so leaving them in does not merely add noise, it moves some classes
and not others. With them in, amide reads unresolved at +0.032 [-0.207, +0.310];
with them out it is +0.401 [+0.204, +0.706], a failure. The class analysis
therefore runs on the regular scans and the full-pool figure is printed beside
it, rather than one being chosen and the other left unsaid.

MEDIANS, NEVER MEANS. MMFF94's mean RMSE over this pool is 903.3 kcal/mol
against a median of 1.866, because RDKit's perception fails on phosphates and
charged sugars and returns astronomic energies. Those molecules are QUARANTINED
into a named table with the value that disqualified them, not dropped in
silence, and every MMFF94 statistic is a median.

ONE TAG IS ABOUT THE MOLECULE AND NOT THE ROTOR, AND IS LABELLED THAT WAY.
Both the UMAP and the PCA of this cohort show a satellite off the main cloud
whose population distances are the worst on the map. It is the capped peptides:
Ace-Xaa-Yaa-Nme rotamers from the OpenFF protein sets, a median of 50 heavy
atoms against the pool's 25. They are tagged here so the group has a name and a
number instead of a position on a projection, and the tag is by backbone SMARTS
rather than by the `ace-...-rotamer-1` naming, which is a convention and not
evidence. It is a MOLECULE tag: every other class in this table is a property
of the driven bond.

THE CLASSIFIER IS NOT A SECOND ONE. `phase_census.classify` already assigns
anomeric, gauche_effect, amide, biaryl, conjugated and stereocentre_adjacent
from the driven bond's own neighbourhood, and the classes are NOT exclusive: an
amide is usually also conjugated. Each is therefore tested against everything
lacking that tag, never against the other classes.

ATOM INDICES COME FROM THE CACHE, NOT THE SMILES. `driven_bond` in the per
molecule table indexes the cache's atom ordering, so the molecule is rebuilt
with `rdkit_from_cache` from the same symbols and bonds the scan used. Parsing
`molecule_id` instead would silently misindex: it is a lowercased mapped SMILES,
so its `cl` is carbon followed by nothing and its `nh` is not `nH`.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

KCAL_PER_K = 0.0019872041
PRESTO_TABLE1 = {"Sage": 0.30, "espaloma": 0.20, "presto": 0.12, "AceFF 2.0": 0.09}


def boltzmann(curve, kelvin):
    """Populations of one torsion profile, the quantity presto compares."""
    beta = 1.0 / (KCAL_PER_K * kelvin)
    w = np.exp(-beta * np.clip(curve - np.min(curve), 0.0, 700.0))
    total = w.sum()
    return w / total if total > 0 else np.full_like(w, 1.0 / w.size)


def jensen_shannon(p, q):
    """Jensen-Shannon DISTANCE, the square root of the divergence, in [0, 1]."""
    m = 0.5 * (p + q)

    def kl(a, b):
        keep = a > 0
        return float(np.sum(a[keep] * np.log2(a[keep] / b[keep])))

    return float(np.sqrt(max(0.5 * kl(p, m) + 0.5 * kl(q, m), 0.0)))


def boot_median(values, rng, n=2000):
    """Median and a percentile interval, resampled over the values themselves."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 3:
        return float("nan"), float("nan"), float("nan"), values.size
    draws = rng.integers(0, values.size, size=(n, values.size))
    medians = np.median(values[draws], axis=1)
    return (float(np.median(values)), float(np.percentile(medians, 2.5)),
            float(np.percentile(medians, 97.5)), int(values.size))


def boot_difference(inside, outside, rng, n=2000):
    """median(inside) - median(outside), both resampled, over molecules."""
    inside = np.asarray(inside, float)[np.isfinite(inside)]
    outside = np.asarray(outside, float)[np.isfinite(outside)]
    if inside.size < 3 or outside.size < 3:
        return float("nan"), float("nan"), float("nan")
    a = np.median(inside[rng.integers(0, inside.size, size=(n, inside.size))], axis=1)
    b = np.median(outside[rng.integers(0, outside.size, size=(n, outside.size))], axis=1)
    d = a - b
    return (float(np.median(inside) - np.median(outside)),
            float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-molecule", required=True)
    parser.add_argument("--profiles", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--table", required=True)
    parser.add_argument("--quarantine", required=True)
    parser.add_argument("--class-figure", required=True)
    parser.add_argument("--pca-figure", required=True)
    parser.add_argument("--gallery-figure", required=True)
    parser.add_argument("--kelvin", type=float, default=500.0)
    parser.add_argument("--mmff-ceiling", type=float, default=100.0,
                        help="RMSE above which an MMFF94 number is a perception "
                             "failure rather than a bad prediction")
    parser.add_argument("--panels", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260916)
    args = parser.parse_args()

    matplotlib.rcParams.update({
        "figure.dpi": 200, "savefig.dpi": 400, "savefig.bbox": "tight",
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "xtick.direction": "in", "ytick.direction": "in", "legend.frameon": False,
        "axes.titlelocation": "left", "axes.titlepad": 12,
    })
    GREY = "#666666"
    rng = np.random.default_rng(args.seed)

    molecules = pd.read_csv(args.per_molecule)
    molecules = molecules[molecules.split == "test"]

    # A FEW MOLECULES WERE SCANNED AT TWO BONDS, AND THE PROFILE TABLE CANNOT
    # SAY WHICH CURVE IS WHICH. It carries `molecule_id` and no driven bond, so
    # for those the two scans are indistinguishable and any JS computed from
    # them would be attributed to the wrong rotor. Four peptides, eight rows.
    # They are quarantined with that reason rather than silently collapsed,
    # which would have given both bonds the same population distance.
    repeated = molecules.molecule_id.duplicated(keep=False)
    ambiguous = molecules[repeated].copy()
    molecules = molecules[~repeated].set_index("molecule_id")

    profiles = pd.read_csv(args.profiles)
    profiles = profiles[profiles.split == "test"]
    profiles = profiles[~profiles.molecule_id.isin(ambiguous.molecule_id)]
    print(f"{len(molecules)} held-out molecules, {len(profiles)} scan points")

    # WHICH SCANS ARE A TORSION PROFILE AND WHICH ARE SOMETHING ELSE. The
    # modal geometry of this cohort is the standard, so a cohort scanned at a
    # different stride still works: nothing here hard-codes 24 or 345.
    shape = profiles.groupby("molecule_id").grid_degrees.agg(
        points="count", low="min", high="max")
    shape["span"] = shape.high - shape.low
    modal_points = int(shape.points.mode().iloc[0])
    modal_span = float(shape.span.mode().iloc[0])
    molecules = molecules.join(shape)
    regular = ((molecules.points == modal_points)
               & (np.isclose(molecules.span, modal_span)))
    molecules["regular_grid"] = regular.fillna(False)
    print(f"  a regular scan here is {modal_points} points over "
          f"{modal_span:.0f} degrees: {int(regular.sum())} of {len(molecules)} "
          f"are, {int((~regular).sum())} are not")
    if len(ambiguous):
        print(f"  {len(ambiguous)} rows ({ambiguous.molecule_id.nunique()} molecules) "
              f"quarantined: scanned at two bonds, profiles keyed on molecule_id alone")

    # ---- Jensen-Shannon per molecule, and the QM and predicted curves -------
    curves, js = {}, {}
    for molecule_id, block in profiles.groupby("molecule_id"):
        block = block.sort_values("grid_degrees")
        qm = block.qm_kcal_mol.to_numpy(float)
        pred = block.predicted_kcal_mol.to_numpy(float)
        if qm.size < 4 or not np.isfinite(qm).all() or not np.isfinite(pred).all():
            continue
        curves[molecule_id] = (block.grid_degrees.to_numpy(float), qm, pred)
        js[molecule_id] = jensen_shannon(boltzmann(qm, args.kelvin),
                                         boltzmann(pred, args.kelvin))
    molecules["js"] = pd.Series(js)

    # ---- the rotor class of each driven bond, from the cache's own atoms ----
    from espaloma_encoder import rdkit_from_cache
    from phase_census import classify

    z = np.load(args.cache, allow_pickle=False)
    meta = {m["molecule_id"]: m for m in json.loads(str(z["meta"]))}
    labels, failed_class = {}, 0
    for molecule_id, row in molecules.iterrows():
        entry = meta.get(molecule_id)
        if entry is None or not row.driven_bond or not isinstance(row.driven_bond, str):
            continue
        try:
            j, k = (int(v) for v in row.driven_bond.split("-"))
            key = entry["key"]
            mol = rdkit_from_cache(entry["symbols"], z[f"bonds::{key}"],
                                   z[f"bond_orders::{key}"])
            if mol is None or max(j, k) >= mol.GetNumAtoms():
                failed_class += 1
                continue
            labels[molecule_id] = classify(mol, j, k)
        except Exception:
            failed_class += 1
    print(f"  classified {len(labels)}; {failed_class} could not be rebuilt from the cache")

    # The capped peptides, by backbone rather than by name. Two or more
    # -N-C(H)-C(=O)-N- residues is a peptide; one is an amide and already tagged.
    from rdkit import Chem as _Chem
    backbone = _Chem.MolFromSmarts("[NX3][CX4;H1,H2][CX3](=O)[NX3]")
    for molecule_id, row in molecules.iterrows():
        mol = (_Chem.MolFromSmiles(row.smiles)
               if isinstance(row.smiles, str) else None)
        if mol is None:
            continue
        if len(mol.GetSubstructMatches(backbone)) >= 2:
            labels.setdefault(molecule_id, []).append("capped_peptide")

    classes = sorted({c for v in labels.values() for c in v})
    for name in classes:
        molecules[f"is_{name}"] = pd.Series(
            {m: (name in v) for m, v in labels.items()})

    # ---- MMFF94, quarantined rather than dropped ---------------------------
    blown = molecules[molecules.rmse_mmff94 > args.mmff_ceiling]
    quarantine = pathlib.Path(args.quarantine)
    quarantine.parent.mkdir(parents=True, exist_ok=True)
    with quarantine.open("w") as handle:
        handle.write("molecule_id,rmse_mmff94,reason\n")
        for molecule_id, row in blown.iterrows():
            handle.write(f'"{molecule_id}",{row.rmse_mmff94:.6g},'
                         f'"MMFF94 above {args.mmff_ceiling} kcal/mol: RDKit '
                         f'perception failure, not a prediction"\n')
        for molecule_id, row in molecules[~molecules.regular_grid].iterrows():
            handle.write(f'"{molecule_id}",{row.rmse_mmff94:.6g},'
                         f'"irregular scan: {int(row.points)} points over '
                         f'{row.span:.0f} degrees, not {modal_points} over '
                         f'{modal_span:.0f}; excluded from the class analysis"\n')
        for _, row in ambiguous.iterrows():
            handle.write(f'"{row.molecule_id}",{row.rmse_mmff94:.6g},'
                         f'"scanned at two bonds ({row.driven_bond} among them); '
                         f'profiles.csv is keyed on molecule_id alone so the '
                         f'curve cannot be attributed to a rotor"\n')
    mmff = molecules.rmse_mmff94.where(molecules.rmse_mmff94 <= args.mmff_ceiling)
    print(f"  MMFF94: {len(blown)} of {len(molecules)} quarantined above "
          f"{args.mmff_ceiling} kcal/mol -> {quarantine}")

    # ---- the pool, and the two controls ------------------------------------
    print(f"\n  pool of {len(molecules)}, median [95% CI] over molecules:")
    rows = []
    for name, values in (("our head, RMSE", molecules.rmse),
                         ("MMFF94, RMSE", mmff),
                         ("flat profile, RMSE", molecules.rmse_flat),
                         ("our head, JS", molecules.js)):
        m, lo, hi, n = boot_median(values, rng, args.bootstrap)
        unit = "" if "JS" in name else " kcal/mol"
        print(f"    {name:22s} {m:6.3f} [{lo:.3f}, {hi:.3f}]{unit}  n = {n}")
        rows.append({"group": "ALL", "metric": name, "median": m,
                     "lo": lo, "hi": hi, "n": n})

    js_median = molecules.js.median()
    print(f"\n  against presto's Table 1 (TorsionNet500, whole force field, "
          f"a DIFFERENT cohort and protocol):")
    for name, value in PRESTO_TABLE1.items():
        print(f"    {name:12s} {value:.2f}")
    print(f"    {'ours, here':12s} {js_median:.2f}   "
          f"{'inside' if 0.09 <= js_median <= 0.30 else 'outside'} that range")

    # ---- per class ----------------------------------------------------------
    clean = molecules[molecules.regular_grid]
    print(f"\n  by rotor class on the {len(clean)} regular scans, each against "
          f"every molecule WITHOUT that tag. `all` is the same difference with "
          f"the {int((~molecules.regular_grid).sum())} irregular scans left in:")
    class_rows = []
    for name in classes:
        flag = clean[f"is_{name}"].fillna(False)
        inside, outside = clean[flag], clean[~flag]
        whole = molecules[f"is_{name}"].fillna(False)
        d_all, *_ = boot_difference(molecules.rmse[whole],
                                    molecules.rmse[~whole], rng, args.bootstrap)
        if len(inside) < 3:
            continue
        r_med, r_lo, r_hi, n = boot_median(inside.rmse, rng, args.bootstrap)
        j_med, *_ = boot_median(inside.js, rng, args.bootstrap)
        d, d_lo, d_hi = boot_difference(inside.rmse, outside.rmse, rng, args.bootstrap)
        jd, jd_lo, jd_hi = boot_difference(inside.js, outside.js, rng, args.bootstrap)
        verdict = ("WORSE" if d_lo > 0 else "better" if d_hi < 0 else "unresolved")
        class_rows.append({"group": name, "n": n, "rmse": r_med, "rmse_lo": r_lo,
                           "rmse_hi": r_hi, "js": j_med, "delta_rmse": d,
                           "delta_lo": d_lo, "delta_hi": d_hi, "delta_js": jd,
                           "delta_js_lo": jd_lo, "delta_js_hi": jd_hi,
                           "verdict": verdict, "delta_rmse_all_scans": d_all,
                           "irregular_fraction": float(
                               (~molecules.regular_grid[whole]).mean())})
        print(f"    {name:22s} n = {n:5d}  RMSE {r_med:5.3f} [{r_lo:.3f}, {r_hi:.3f}]"
              f"  JS {j_med:5.3f}  vs rest {d:+6.3f} [{d_lo:+.3f}, {d_hi:+.3f}]"
              f"  {verdict:11s} all {d_all:+6.3f}"
              f"  irregular {100 * (~molecules.regular_grid[whole]).mean():4.1f}%")

    # ---- the random control: permuted class labels -------------------------
    sizes = [r["n"] for r in class_rows]
    rmse_all = clean.rmse.to_numpy(float)
    rmse_all = rmse_all[np.isfinite(rmse_all)]
    null = []
    for _ in range(400):
        order = rng.permutation(rmse_all.size)
        for size in sizes:
            pick = order[:size]
            rest = order[size:]
            null.append(abs(np.median(rmse_all[pick]) - np.median(rmse_all[rest])))
    envelope = float(np.percentile(null, 95))
    print(f"\n  random control, class labels permuted at the same class sizes:")
    print(f"    95th percentile of |median difference| = {envelope:.3f} kcal/mol")
    real = [r for r in class_rows if abs(r["delta_rmse"]) > envelope]
    print(f"    {len(real)} of {len(class_rows)} classes exceed it: "
          f"{', '.join(r['group'] for r in real) or 'none'}")

    frame = pd.DataFrame(class_rows)
    frame["null_envelope"] = envelope
    pathlib.Path(args.table).parent.mkdir(parents=True, exist_ok=True)
    pd.concat([pd.DataFrame(rows), frame]).to_csv(args.table, index=False)

    # ---- a. the classes, drawn ---------------------------------------------
    order = frame.sort_values("delta_rmse")
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 0.42 * len(order) + 2.1))
    for ax, (column, lo, hi, label) in zip(axes, (
            ("delta_rmse", "delta_lo", "delta_hi",
             "RMSE difference from the rest (kcal mol$^{-1}$)"),
            ("delta_js", "delta_js_lo", "delta_js_hi",
             "JS difference from the rest"))):
        y = np.arange(len(order))
        # EACH PANEL IS COLOURED BY ITS OWN INTERVAL. Colouring the JS panel by
        # the RMSE verdict would paint anomeric as a failure there, when its
        # population distance is +0.000 [-0.03, +0.02]: its barrier HEIGHTS are
        # wrong and its equilibrium populations are not. That is presto's whole
        # argument for reporting JS, and a shared colour would have hidden it.
        colours = ["#D55E00" if lo_v > 0 else "#0072B2" if hi_v < 0 else GREY
                   for lo_v, hi_v in zip(order[lo], order[hi])]
        ax.errorbar(order[column], y,
                    xerr=[order[column] - order[lo], order[hi] - order[column]],
                    fmt="o", ms=4, lw=0, elinewidth=1.1, ecolor=colours)
        for index, (value, colour) in enumerate(zip(order[column], colours)):
            ax.plot([value], [index], "o", ms=4, color=colour)
        ax.axvline(0, color="black", lw=0.7, ls=(0, (4, 3)))
        if column == "delta_rmse":
            ax.axvspan(-envelope, envelope, color=GREY, alpha=0.12, lw=0)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{r.group}  (n = {r.n})" for r in order.itertuples()])
        ax.set_xlabel(label)
    axes[0].set_title("$\\bf{a}$  profile RMSE, class minus the rest", loc="left")
    axes[0].text(0.0, 1.004, "grey band: the 95th percentile of permuted class "
                 "labels at the same sizes", transform=axes[0].transAxes,
                 fontsize=6.2, color=GREY, va="bottom")
    axes[1].set_title("$\\bf{b}$  population overlap, class minus the rest", loc="left")
    axes[1].text(0.0, 1.004, f"Jensen-Shannon distance at {args.kelvin:.0f} K",
                 transform=axes[1].transAxes, fontsize=6.2, color=GREY, va="bottom")
    figure.text(0.0, -0.02, f"{len(molecules)} held-out OpenFF Gen3 molecules, one "
                f"driven bond each. Classes are NOT exclusive, so each is compared "
                f"with every molecule lacking that tag. Bars are 95% bootstrap "
                f"intervals resampled over molecules.",
                fontsize=6.4, color=GREY, ha="left", va="top")
    figure.tight_layout(w_pad=2.4)
    figure.savefig(args.class_figure)
    plt.close(figure)

    # ---- b. PCA -------------------------------------------------------------
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    from sklearn.decomposition import PCA

    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp, keep = [], []
    for molecule_id, row in molecules.iterrows():
        mol = Chem.MolFromSmiles(row.smiles) if isinstance(row.smiles, str) else None
        if mol is None:
            continue
        fp.append(np.asarray(gen.GetFingerprint(mol), dtype=np.float32))
        keep.append(molecule_id)
    fp = np.asarray(fp)
    sub = molecules.loc[keep].copy()
    pca = PCA(n_components=2, random_state=args.seed).fit(fp)
    coords = pca.transform(fp)
    sub["p1"], sub["p2"] = coords[:, 0], coords[:, 1]
    explained = pca.explained_variance_ratio_

    figure, axes = plt.subplots(1, 3, figsize=(10.4, 3.4))
    top = float(np.percentile(sub.rmse, 95))
    s = axes[0].scatter(sub.p1, sub.p2, c=np.clip(sub.rmse, 0, top), s=8, lw=0,
                        cmap="viridis", vmin=0, vmax=top)
    axes[0].set_title("$\\bf{a}$  profile RMSE", loc="left")
    axes[0].text(0.0, 1.008, f"median {sub.rmse.median():.3f} kcal mol$^{{-1}}$, "
                 f"clipped at the 95th percentile", transform=axes[0].transAxes,
                 fontsize=6.2, color=GREY, va="bottom")
    plt.colorbar(s, ax=axes[0], label="kcal mol$^{-1}$")

    jtop = float(np.percentile(sub.js.dropna(), 95)) if sub.js.notna().any() else 1.0
    s = axes[1].scatter(sub.p1, sub.p2, c=np.clip(sub.js, 0, jtop), s=8, lw=0,
                        cmap="magma", vmin=0, vmax=jtop)
    axes[1].set_title("$\\bf{b}$  population overlap", loc="left")
    axes[1].text(0.0, 1.008, f"JS at {args.kelvin:.0f} K, median "
                 f"{sub.js.median():.3f}", transform=axes[1].transAxes,
                 fontsize=6.2, color=GREY, va="bottom")
    plt.colorbar(s, ax=axes[1], label="JS distance")

    palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
               "#56B4E9", "#7F7F7F"]
    axes[2].scatter(sub.p1, sub.p2, s=6, lw=0, color="#DDDDDD")
    for colour, name in zip(palette, classes):
        flag = sub.get(f"is_{name}")
        if flag is None:
            continue
        flag = flag.fillna(False)
        if flag.sum() < 3:
            continue
        axes[2].scatter(sub.p1[flag].mean(), sub.p2[flag].mean(), s=70, lw=0.8,
                        marker="o", color=colour, edgecolor="white",
                        label=f"{name} ({int(flag.sum())})", zorder=3)
    axes[2].set_title("$\\bf{c}$  class centroids", loc="left")
    axes[2].text(0.0, 1.008, "one marker is a class mean, grey is every molecule",
                 transform=axes[2].transAxes, fontsize=6.2, color=GREY, va="bottom")
    axes[2].legend(loc="best", fontsize=5.8)
    for ax in axes:
        ax.set_xlabel(f"PC1 ({100 * explained[0]:.1f}% of variance)")
        ax.set_ylabel(f"PC2 ({100 * explained[1]:.1f}%)")
        ax.set_xticks([]); ax.set_yticks([])
    figure.text(0.0, -0.04, f"PCA of ECFP4 (radius 2, 2048 bit) over {len(sub)} "
                f"held-out molecules. The two components carry "
                f"{100 * explained.sum():.1f}% of the variance between them, so "
                f"distance on this plane is a weak summary of similarity.",
                fontsize=6.4, color=GREY, ha="left", va="top")
    figure.tight_layout(w_pad=2.0)
    figure.savefig(args.pca_figure)
    plt.close(figure)

    # ---- c. the QM drives, for the molecules the head improves most --------
    improved = molecules.assign(gain=mmff - molecules.rmse).dropna(subset=["gain"])
    improved = improved[improved.index.isin(curves) & improved.regular_grid]
    picks = list(improved.nlargest(args.panels, "gain").index)
    picks += list(improved.nsmallest(args.panels // 2, "gain").index)

    columns = 4
    rowcount = int(np.ceil(len(picks) / columns))
    figure, axes = plt.subplots(rowcount, columns,
                                figsize=(2.65 * columns, 2.25 * rowcount))
    for ax, molecule_id in zip(np.atleast_1d(axes).flat, picks):
        grid, qm, pred = curves[molecule_id]
        row = molecules.loc[molecule_id]
        ax.plot(grid, qm, "-o", ms=2.6, lw=1.4, color="black", label="QM (B3LYP-D3BJ)")
        ax.plot(grid, pred, "-o", ms=2.6, lw=1.4, color="#0072B2", label="our head")
        ax.axhline(0, color=GREY, lw=0.6, ls=(0, (4, 3)))
        tags = [c for c in classes if bool(row.get(f"is_{c}", False))]
        gain = improved.gain.get(molecule_id, float("nan"))
        ax.set_title(f"{'; '.join(tags) or 'plain'}", loc="left", fontsize=7)
        ax.text(0.0, 1.005, f"RMSE {row.rmse:.2f}, MMFF94 {row.rmse_mmff94:.2f}, "
                f"JS {row.js:.3f}, gain {gain:+.2f}", transform=ax.transAxes,
                fontsize=5.8, color=GREY, va="bottom")
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_xlabel("driven dihedral (degrees)")
        ax.set_ylabel("energy (kcal mol$^{-1}$)")
    for ax in np.atleast_1d(axes).flat[len(picks):]:
        ax.set_visible(False)
    handles, labels_ = np.atleast_1d(axes).flat[0].get_legend_handles_labels()
    figure.legend(handles, labels_, loc="lower center", ncol=2,
                  bbox_to_anchor=(0.5, -0.035))
    figure.text(0.0, -0.045, f"The {args.panels} molecules our head improves most "
                f"over MMFF94 by profile RMSE, then the {args.panels // 2} it "
                f"improves least. Gain is MMFF94 RMSE minus ours, kcal mol$^{{-1}}$. "
                f"Regular scans only ({modal_points} points over "
                f"{modal_span:.0f} degrees); molecules whose MMFF94 exceeded "
                f"{args.mmff_ceiling:.0f} kcal/mol are quarantined and not shown.",
                fontsize=6.4, color=GREY, ha="left", va="top")
    figure.tight_layout(h_pad=2.8, w_pad=2.0)
    figure.savefig(args.gallery_figure)
    plt.close(figure)

    print(f"\n  wrote {args.class_figure}, {args.pca_figure}, "
          f"{args.gallery_figure}, {args.table}")


if __name__ == "__main__":
    main()
