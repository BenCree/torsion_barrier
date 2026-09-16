"""Does any region of this cohort's chemical space carry the error, found rather than named.

THE QUESTION THIS ASKS THAT THE CLASS TABLE DOES NOT. `failure_groups.py` tests
rotor classes chosen in advance from chemistry: anomeric, amide, biaryl and the
rest. That can only find what it was told to look for. This clusters the cohort
by structure and asks the same question of every cluster, so a failure region
with no name gets one, and a class table that has already found everything is
confirmed by finding nothing new.

BUTINA, NOT k-MEANS, AND NOT A PROJECTION. Butina at a Tanimoto threshold is
the grouping used as the resampling unit elsewhere in this project, it needs no
cluster count chosen in advance, and it clusters in the full 2048-bit space.
Clustering a 2-D projection instead would cluster the projection's artifacts:
PCA's first two components carry 9.2 percent of the variance here, so nearby on
that plane and similar are not the same statement.

THREE PROJECTIONS, ONE CLUSTERING. PCA, t-SNE and UMAP are drawn from the same
fingerprints and coloured by the same Butina labels. They disagree about layout
and that is the point: a cluster that holds together in all three is a real
neighbourhood, and a region that looks tight in one projection alone is that
projection's doing. None of them is used to decide anything; the statistics are
computed in fingerprint space.

WHY EVERY CLUSTER IS NAMED. A result reading "cluster 7 is worse by 0.9" is not
a finding, because nobody can act on an index. Each tested cluster reports its
Murcko scaffold and the rotor-class composition of its members, so a hit is
either a chemistry or is visibly the peptides again.

BUTINA ALONE LEAVES MOST OF THIS COHORT UNTESTED, SO IT IS NOT USED ALONE.
At Tanimoto distance 0.4 these 3,698 molecules fall into 2,941 clusters, and
only two hold thirty or more members: 2.3 percent of the pool. That is a result
about the cohort, not a failure of the method, and it explains the flat UMAP
directly, since a set that is four fifths singletons has no neighbourhoods for
error to localise in. But it also means Butina tests almost nothing. A k-means
partition of the same fingerprints is therefore run beside it: it covers every
molecule by construction, so the question "does any region carry the error" is
asked of the whole pool rather than of its two tight corners. Both are reported
with the same statistic and the same control. k-means on binary fingerprints
minimises Euclidean distance, which is monotone in Hamming distance on binary
vectors, so its clusters are compact in bit-difference and not in Tanimoto; that
is a different notion of near from Butina's and the disagreement is informative
rather than a defect.

THE PERMUTED CONTROL IS WHAT MAKES A SMALL CLUSTER READABLE. Testing every
cluster of 30 or more against the rest is many comparisons on a pool with a long
RMSE tail, and the tail alone will hand some 30-molecule cluster a large median
difference. The labels are therefore shuffled at the same cluster sizes and the
95th percentile of the largest |difference| per shuffle is reported, which is
the level a cluster has to clear to be worth naming.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

KCAL_PER_K = 0.0019872041


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


def boot_difference(inside, outside, rng, n=2000):
    inside = np.asarray(inside, float)[np.isfinite(inside)]
    outside = np.asarray(outside, float)[np.isfinite(outside)]
    if inside.size < 3 or outside.size < 3:
        return float("nan"), float("nan"), float("nan")
    a = np.median(inside[rng.integers(0, inside.size, size=(n, inside.size))], axis=1)
    b = np.median(outside[rng.integers(0, outside.size, size=(n, outside.size))], axis=1)
    d = a - b
    return (float(np.median(inside) - np.median(outside)),
            float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))


def butina(fingerprints, threshold):
    """Butina clusters at a Tanimoto DISTANCE threshold, using RDKit's own."""
    from rdkit import DataStructs
    from rdkit.ML.Cluster import Butina

    distances = []
    for index in range(1, len(fingerprints)):
        similarity = DataStructs.BulkTanimotoSimilarity(
            fingerprints[index], fingerprints[:index])
        distances.extend(1.0 - s for s in similarity)
    return Butina.ClusterData(distances, len(fingerprints), threshold,
                              isDistData=True)


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
    parser.add_argument("--figure", required=True)
    parser.add_argument("--cluster-figure", required=True)
    parser.add_argument("--threshold", type=float, default=0.4,
                        help="Butina Tanimoto DISTANCE cutoff")
    parser.add_argument("--sensitivity", default="0.3,0.4,0.5",
                        help="thresholds whose cluster counts are reported, so "
                             "the one chosen is visibly not the only one tried")
    parser.add_argument("--min-cluster", type=int, default=30)
    parser.add_argument("--kmeans-k", type=int, default=40,
                        help="clusters in the partition that covers every "
                             "molecule, run beside Butina")
    parser.add_argument("--kelvin", type=float, default=500.0)
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

    molecules = pd.read_csv(args.per_molecule)
    molecules = molecules[molecules.split == "test"]
    molecules = molecules[~molecules.molecule_id.duplicated(keep=False)]
    molecules = molecules.set_index("molecule_id")

    profiles = pd.read_csv(args.profiles)
    profiles = profiles[profiles.split == "test"]

    shape = profiles.groupby("molecule_id").grid_degrees.agg(
        points="count", low="min", high="max")
    shape["span"] = shape.high - shape.low
    modal_points = int(shape.points.mode().iloc[0])
    modal_span = float(shape.span.mode().iloc[0])
    molecules = molecules.join(shape)
    molecules = molecules[(molecules.points == modal_points)
                          & (np.isclose(molecules.span, modal_span))]
    print(f"{len(molecules)} held-out molecules on regular scans "
          f"({modal_points} points over {modal_span:.0f} degrees)")

    js = {}
    for molecule_id, block in profiles[
            profiles.molecule_id.isin(molecules.index)].groupby("molecule_id"):
        block = block.sort_values("grid_degrees")
        qm = block.qm_kcal_mol.to_numpy(float)
        pred = block.predicted_kcal_mol.to_numpy(float)
        if np.isfinite(qm).all() and np.isfinite(pred).all():
            js[molecule_id] = jensen_shannon(boltzmann(qm, args.kelvin),
                                             boltzmann(pred, args.kelvin))
    molecules["js"] = pd.Series(js)

    # ---- fingerprints, clusters, scaffolds ---------------------------------
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    bits, arrays, keep, scaffolds = [], [], [], {}
    for molecule_id, row in molecules.iterrows():
        mol = Chem.MolFromSmiles(row.smiles) if isinstance(row.smiles, str) else None
        if mol is None:
            continue
        bits.append(generator.GetFingerprint(mol))
        arrays.append(np.asarray(bits[-1], dtype=np.float32))
        keep.append(molecule_id)
        try:
            scaffolds[molecule_id] = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        except Exception:
            scaffolds[molecule_id] = ""
    arrays = np.asarray(arrays)
    sub = molecules.loc[keep].copy()
    sub["scaffold"] = pd.Series(scaffolds)

    for value in (float(v) for v in args.sensitivity.split(",")):
        groups = butina(bits, value)
        big = [g for g in groups if len(g) >= args.min_cluster]
        covered = sum(len(g) for g in big)
        mark = "  <- used" if abs(value - args.threshold) < 1e-9 else ""
        print(f"  Tanimoto distance {value:.2f}: {len(groups):5d} clusters, "
              f"{len(big):3d} with {args.min_cluster}+ members covering "
              f"{100 * covered / len(sub):.1f}% of molecules{mark}")

    groups = butina(bits, args.threshold)
    label = np.full(len(sub), -1)
    for index, group in enumerate(groups):
        for member in group:
            label[member] = index
    sub["cluster"] = label
    tested = [i for i, g in enumerate(groups) if len(g) >= args.min_cluster]

    # ---- the rotor classes, carried so a cluster can be recognised ---------
    from espaloma_encoder import rdkit_from_cache
    from phase_census import classify

    cache = np.load(args.cache, allow_pickle=False)
    meta = {m["molecule_id"]: m for m in json.loads(str(cache["meta"]))}
    tags = {}
    for molecule_id, row in sub.iterrows():
        entry = meta.get(molecule_id)
        if entry is None or not isinstance(row.driven_bond, str):
            continue
        try:
            j, k = (int(v) for v in row.driven_bond.split("-"))
            key = entry["key"]
            mol = rdkit_from_cache(entry["symbols"], cache[f"bonds::{key}"],
                                   cache[f"bond_orders::{key}"])
            if mol is not None and max(j, k) < mol.GetNumAtoms():
                tags[molecule_id] = classify(mol, j, k)
        except Exception:
            continue

    def assess(labels, method):
        """One partition, every cluster of `--min-cluster` or more against the rest."""
        rows, sizes = [], []
        for index in sorted(set(labels)):
            flag = labels == index
            if flag.sum() < args.min_cluster:
                continue
            inside, outside = sub[flag], sub[~flag]
            sizes.append(int(flag.sum()))
            d, lo, hi = boot_difference(inside.rmse, outside.rmse, rng, args.bootstrap)
            jd, jlo, jhi = boot_difference(inside.js, outside.js, rng, args.bootstrap)
            composition = {}
            for molecule_id in inside.index:
                for tag in tags.get(molecule_id, []):
                    composition[tag] = composition.get(tag, 0) + 1
            top = sorted(composition.items(), key=lambda kv: -kv[1])[:2]
            scaffold = inside.scaffold.mode()
            rows.append({
                "method": method, "cluster": int(index), "n": int(flag.sum()),
                "rmse": float(inside.rmse.median()), "js": float(inside.js.median()),
                "delta_rmse": d, "delta_lo": lo, "delta_hi": hi,
                "delta_js": jd, "delta_js_lo": jlo, "delta_js_hi": jhi,
                "scaffold": (scaffold.iloc[0] if len(scaffold) else "") or "(acyclic)",
                "composition": "; ".join(
                    f"{name} {100 * count / flag.sum():.0f}%" for name, count in top),
            })

        # The control shuffles the labels at the same cluster sizes and keeps the
        # LARGEST difference from each shuffle, because that is the statistic
        # being read off the table: the most extreme of many comparisons.
        values = sub.rmse.to_numpy(float)
        values = values[np.isfinite(values)]
        null = []
        for _ in range(400):
            order = rng.permutation(values.size)
            begin, worst = 0, 0.0
            for size in sizes:
                pick = order[begin:begin + size]
                rest = np.concatenate([order[:begin], order[begin + size:]])
                worst = max(worst,
                            abs(np.median(values[pick]) - np.median(values[rest])))
                begin += size
            null.append(worst)
        envelope = float(np.percentile(null, 95)) if null else float("nan")

        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        frame = frame.sort_values("delta_rmse", ascending=False)
        frame["null_envelope"] = envelope
        frame["clears_envelope"] = frame.delta_rmse.abs() > envelope
        frame["verdict"] = np.where(
            (frame.delta_lo > 0) & frame.clears_envelope, "WORSE",
            np.where((frame.delta_hi < 0) & frame.clears_envelope,
                     "better", "unresolved"))
        covered = frame.n.sum()
        print(f"\n  {method}: {len(frame)} clusters of {args.min_cluster}+ "
              f"members, covering {100 * covered / len(sub):.1f}% of molecules")
        for row in frame.itertuples():
            print(f"    cluster {row.cluster:4d}  n = {row.n:4d}  "
                  f"RMSE {row.rmse:5.3f}  vs rest {row.delta_rmse:+6.3f} "
                  f"[{row.delta_lo:+.3f}, {row.delta_hi:+.3f}]  JS {row.js:5.3f}"
                  f"  {row.verdict:10s}  {row.composition}")
            if row.verdict != "unresolved":
                print(f"                  scaffold {row.scaffold[:70]}")
        print(f"    permuted labels at the same sizes, 95th percentile of the "
              f"LARGEST |difference| per shuffle = {envelope:.3f} kcal/mol")
        named = frame[frame.verdict != "unresolved"]
        print(f"    {len(named)} of {len(frame)} clear it with an interval "
              f"excluding zero")
        return frame

    from sklearn.cluster import MiniBatchKMeans

    kmeans = MiniBatchKMeans(n_clusters=args.kmeans_k, random_state=args.seed,
                             n_init=10).fit_predict(arrays)
    sub["kmeans"] = kmeans

    butina_frame = assess(sub.cluster.to_numpy(), f"Butina d={args.threshold:.2f}")
    kmeans_frame = assess(kmeans, f"k-means k={args.kmeans_k}")
    frame = pd.concat([butina_frame, kmeans_frame], ignore_index=True)
    envelope = float(butina_frame.null_envelope.iloc[0]) if len(butina_frame) else 0.0

    pathlib.Path(args.table).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.table, index=False)

    # ---- the three projections ---------------------------------------------
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import umap

    layouts = {}
    layouts["PCA"] = PCA(n_components=2, random_state=args.seed).fit_transform(arrays)
    layouts["t-SNE"] = TSNE(n_components=2, metric="jaccard", init="random",
                            perplexity=30, random_state=args.seed).fit_transform(arrays)
    layouts["UMAP"] = umap.UMAP(n_neighbors=20, min_dist=0.25, metric="jaccard",
                                random_state=args.seed).fit_transform(arrays)

    drawn = kmeans_frame if len(kmeans_frame) >= len(butina_frame) else butina_frame
    drawn_labels = (sub.kmeans.to_numpy() if drawn is kmeans_frame
                    else sub.cluster.to_numpy())
    worst = list(drawn.nlargest(4, "delta_rmse").cluster)
    palette = ["#D55E00", "#E69F00", "#CC79A7", "#882255"]

    figure, axes = plt.subplots(2, 3, figsize=(10.6, 6.6))
    top = float(np.percentile(sub.rmse, 95))
    for column, (name, coords) in enumerate(layouts.items()):
        ax = axes[0, column]
        s = ax.scatter(coords[:, 0], coords[:, 1], c=np.clip(sub.rmse, 0, top),
                       s=7, lw=0, cmap="viridis", vmin=0, vmax=top)
        ax.set_title(f"$\\bf{{{'abc'[column]}}}$  {name}, profile RMSE", loc="left")
        if column == 2:
            plt.colorbar(s, ax=ax, label="kcal mol$^{-1}$")

        ax = axes[1, column]
        ax.scatter(coords[:, 0], coords[:, 1], s=6, lw=0, color="#DDDDDD")
        for colour, index in zip(palette, worst):
            flag = drawn_labels == index
            row = drawn[drawn.cluster == index].iloc[0]
            ax.scatter(coords[flag, 0], coords[flag, 1], s=11, lw=0, color=colour,
                       label=f"cluster {index} (n = {row.n}, {row.delta_rmse:+.2f})")
        ax.set_title(f"$\\bf{{{'def'[column]}}}$  {name}, the four worst "
                     f"{drawn.method.iloc[0]} clusters", loc="left")
        if column == 0:
            ax.legend(loc="best", fontsize=5.6)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].text(0.0, 1.008, f"median {sub.rmse.median():.3f} kcal mol$^{{-1}}$, "
                    f"clipped at the 95th percentile", transform=axes[0, 0].transAxes,
                    fontsize=6.2, color=GREY, va="bottom")
    axes[1, 0].text(0.0, 1.008, "clustered in the full 2048-bit space, not in "
                    "any projection", transform=axes[1, 0].transAxes,
                    fontsize=6.2, color=GREY, va="bottom")
    figure.text(0.0, -0.02, f"{len(sub)} held-out molecules, ECFP4 (radius 2, 2048 "
                f"bit). Top row is the same colour scale in all three; bottom row "
                f"is the same {drawn.method.iloc[0]} partition in all three, "
                f"computed in the full 2048-bit space. The bracketed number is "
                f"the cluster's median RMSE minus every molecule outside it, "
                f"kcal mol$^{{-1}}$. One point is one molecule.",
                fontsize=6.4, color=GREY, ha="left", va="top")
    figure.tight_layout(h_pad=2.6, w_pad=1.8)
    figure.savefig(args.figure)
    plt.close(figure)

    # ---- the clusters themselves, as a forest -------------------------------
    order = frame.sort_values("delta_rmse")
    order = order.assign(label=[f"{r.method} {r.cluster} (n = {r.n}) {r.composition}"
                                for r in order.itertuples()])
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 0.32 * len(order) + 2.2))
    for ax, (column, lo, hi, xlabel) in zip(axes, (
            ("delta_rmse", "delta_lo", "delta_hi",
             "RMSE difference from every molecule outside the cluster (kcal mol$^{-1}$)"),
            ("delta_js", "delta_js_lo", "delta_js_hi",
             "JS difference from every molecule outside the cluster"))):
        y = np.arange(len(order))
        colours = ["#D55E00" if l > 0 else "#0072B2" if h < 0 else GREY
                   for l, h in zip(order[lo], order[hi])]
        ax.errorbar(order[column], y,
                    xerr=[order[column] - order[lo], order[hi] - order[column]],
                    fmt="none", elinewidth=1.1, ecolor=colours)
        for index, (value, colour) in enumerate(zip(order[column], colours)):
            ax.plot([value], [index], "o", ms=4, color=colour)
        ax.axvline(0, color="black", lw=0.7, ls=(0, (4, 3)))
        if column == "delta_rmse":
            ax.axvspan(-envelope, envelope, color=GREY, alpha=0.12, lw=0)
        ax.set_yticks(y)
        ax.set_yticklabels(order.label, fontsize=5.6)
        ax.set_xlabel(xlabel)
    axes[0].set_title("$\\bf{a}$  profile RMSE, by cluster", loc="left")
    axes[0].text(0.0, 1.004, "grey band: the largest difference permuted cluster "
                 "labels of the same sizes produce, 95th percentile",
                 transform=axes[0].transAxes, fontsize=6.2, color=GREY, va="bottom")
    axes[1].set_title("$\\bf{b}$  population overlap, by cluster", loc="left")
    axes[1].text(0.0, 1.004, f"Jensen-Shannon distance at {args.kelvin:.0f} K",
                 transform=axes[1].transAxes, fontsize=6.2, color=GREY, va="bottom")
    figure.text(0.0, -0.02, f"{len(sub)} held-out molecules. Butina at Tanimoto "
                f"distance {args.threshold:.2f} gives {len(groups)} clusters, "
                f"almost all singletons, so a k-means partition covering every "
                f"molecule is tested beside it; clusters of "
                f"{args.min_cluster} or more are shown. "
                f"Labels carry each cluster's two commonest rotor classes. Bars "
                f"are 95% bootstrap intervals resampled over molecules.",
                fontsize=6.4, color=GREY, ha="left", va="top")
    figure.tight_layout(w_pad=2.2)
    figure.savefig(args.cluster_figure)
    plt.close(figure)

    print(f"\n  wrote {args.figure}, {args.cluster_figure}, {args.table}")


if __name__ == "__main__":
    main()
