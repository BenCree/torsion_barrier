"""Where in chemical space each model fails, and on what.

THREE QUESTIONS, THREE FIGURES.

  a UMAP of the cohort's ECFP4 space, coloured by each model's profile RMSE and
    by the difference between them. If the failures are localised, the map shows
    a region rather than a scatter, and a region is something you can name.

  b the difference (our head minus the zero-shot potential) against molecular
    descriptors, to say WHICH property predicts who wins rather than only that
    someone does.

  c the twenty molecules with the largest difference in each direction, drawn,
    with the driven torsion highlighted. A structure is the only form in which
    "charged and exotic" stops being a guess.

The colour scale is shared between the two RMSE panels so they can be compared
by eye, and it is clipped at a percentile so a handful of catastrophic failures
do not flatten everything else to one colour.
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib

import numpy as np


def fingerprints(smiles):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    rows, keep = [], []
    for index, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s) if s else None
        if mol is None:
            continue
        rows.append(np.asarray(gen.GetFingerprint(mol), dtype=np.uint8))
        keep.append(index)
    return np.asarray(rows), np.asarray(keep)


def descriptors(smiles):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, rdMolDescriptors

    RDLogger.DisableLog("rdApp.*")
    out = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s) if s else None
        if mol is None:
            out.append({})
            continue
        out.append({
            "heavy_atoms": mol.GetNumHeavyAtoms(),
            "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
            "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "formal_charge": Chem.GetFormalCharge(mol),
            "tpsa": Descriptors.TPSA(mol),
            "logp": Descriptors.MolLogP(mol),
            "n_hetero": rdMolDescriptors.CalcNumHeteroatoms(mol),
        })
    return out


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from scipy.stats import spearmanr

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours-runs", required=True)
    parser.add_argument("--zeroshot", required=True)
    parser.add_argument("--cache", required=True, help="npz carrying the driven torsion atoms")
    parser.add_argument("--umap-figure", required=True)
    parser.add_argument("--trend-figure", required=True)
    parser.add_argument("--grid-figure", required=True)
    parser.add_argument("--table", required=True)
    parser.add_argument("--seed", type=int, default=20260911)
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

    ours = pd.concat([pd.read_csv(f) for f in glob.glob(f"{args.ours_runs}/*/per_molecule.csv")])
    zs = pd.read_csv(args.zeroshot).set_index("molecule_id")
    ours = ours.set_index("molecule_id")
    shared = ours.index.intersection(zs.index)
    df = pd.DataFrame({
        "smiles": ours.loc[shared, "smiles"],
        "ours": ours.loc[shared, "rmse"],
        "zeroshot": zs.loc[shared, "rmse"],
        "barrier": ours.loc[shared, "qm_barrier_kcal_mol"],
        "rho_ours": ours.loc[shared, "rho"],
        "rho_zs": zs.loc[shared, "rho"],
    })
    df["delta"] = df.ours - df.zeroshot
    print(f"{len(df)} molecules held out by both")

    desc = pd.DataFrame(descriptors(df.smiles.tolist()), index=df.index)
    df = df.join(desc)
    df.to_csv(args.table)

    # ---- a. UMAP -----------------------------------------------------------
    fp, keep = fingerprints(df.smiles.tolist())
    sub = df.iloc[keep].copy()
    import umap

    embedding = umap.UMAP(
        n_neighbors=20, min_dist=0.25, metric="jaccard", random_state=args.seed
    ).fit_transform(fp)
    sub["u1"], sub["u2"] = embedding[:, 0], embedding[:, 1]

    figure, axes = plt.subplots(1, 3, figsize=(10.2, 3.3))
    top = float(np.percentile(np.concatenate([sub.ours, sub.zeroshot]), 95))
    for index, (column, title) in enumerate(
        (("ours", "our head, profile RMSE"), ("zeroshot", "DPA3 zero-shot, profile RMSE"))
    ):
        ax = axes[index]
        s = ax.scatter(sub.u1, sub.u2, c=np.clip(sub[column], 0, top), s=9, lw=0,
                       cmap="viridis", vmin=0, vmax=top)
        ax.set_title(f"$\\bf{{{'abc'[index]}}}$  {title}", loc="left")
        ax.text(0.0, 1.008, f"median {sub[column].median():.3f} kcal mol$^{{-1}}$, "
                f"scale clipped at the 95th percentile", transform=ax.transAxes,
                fontsize=6.2, color=GREY, va="bottom")
        plt.colorbar(s, ax=ax, label="kcal mol$^{-1}$")
        ax.set_xticks([]); ax.set_yticks([])
    ax = axes[2]
    lim = float(np.percentile(np.abs(sub.delta), 95))
    s = ax.scatter(sub.u1, sub.u2, c=np.clip(sub.delta, -lim, lim), s=9, lw=0,
                   cmap="coolwarm", vmin=-lim, vmax=lim)
    ax.set_title("$\\bf{c}$  difference, ours minus zero-shot", loc="left")
    ax.text(0.0, 1.008, "blue: our head better. red: the potential better.",
            transform=ax.transAxes, fontsize=6.2, color=GREY, va="bottom")
    plt.colorbar(s, ax=ax, label="kcal mol$^{-1}$")
    ax.set_xticks([]); ax.set_yticks([])
    figure.text(0.0, -0.04, f"UMAP of ECFP4 (radius 2, 2048 bit), Jaccard metric, "
                f"{len(sub)} held-out OpenFF Gen3 molecules. One point is one molecule.",
                fontsize=6.4, color=GREY, ha="left", va="top")
    figure.tight_layout(w_pad=2.0)
    figure.savefig(args.umap_figure)
    plt.close(figure)

    # ---- b. what predicts the difference -----------------------------------
    props = ["heavy_atoms", "aromatic_rings", "rotatable_bonds", "formal_charge",
             "tpsa", "logp", "n_hetero", "barrier"]
    figure, axes = plt.subplots(2, 4, figsize=(10.2, 4.8))
    print("\n  what predicts (ours - zero-shot):")
    for index, (ax, prop) in enumerate(zip(axes.flat, props)):
        d = df.dropna(subset=[prop, "delta"])
        r, p = spearmanr(d[prop], d.delta)
        ax.scatter(d[prop], d.delta, s=7, lw=0, alpha=0.45, color="#0072B2")
        ax.axhline(0, color="black", lw=0.7, ls=(0, (4, 3)))
        ax.set_xlabel(prop.replace("_", " "))
        if index % 4 == 0:
            ax.set_ylabel("ours - zero-shot (kcal mol$^{-1}$)")
        ax.set_title(f"$\\bf{{{'abcdefgh'[index]}}}$", loc="left")
        ax.text(0.0, 1.008, f"rho = {r:+.3f}, p = {p:.1e}", transform=ax.transAxes,
                fontsize=6.2, color=GREY, va="bottom")
        print(f"    {prop:18s} rho {r:+.3f}  p {p:.2e}")
    figure.text(0.0, -0.03, "Positive means the zero-shot potential is better on that molecule. "
                f"{len(df)} held-out molecules.", fontsize=6.4, color=GREY, ha="left", va="top")
    figure.tight_layout(h_pad=2.6, w_pad=1.8)
    figure.savefig(args.trend_figure)
    plt.close(figure)

    # ---- c. the extremes, drawn --------------------------------------------
    from rdkit import Chem
    from rdkit.Chem import Draw

    z = np.load(args.cache, allow_pickle=False)
    driven = {m["molecule_id"]: m["driven_dihedral"] for m in json.loads(str(z["meta"]))}

    picks = pd.concat([df.nsmallest(10, "delta"), df.nlargest(10, "delta")])
    mols, legends, highlights = [], [], []
    for molecule_id, row in picks.iterrows():
        mol = Chem.MolFromSmiles(row.smiles) if row.smiles else None
        if mol is None:
            continue
        mols.append(mol)
        atoms = [a for a in driven.get(molecule_id, []) if a < mol.GetNumAtoms()]
        highlights.append(atoms)
        who = "OURS" if row.delta < 0 else "zero-shot"
        legends.append(f"{who} better by {abs(row.delta):.2f}\n"
                       f"ours {row.ours:.2f}  zs {row.zeroshot:.2f}  barrier {row.barrier:.1f}")
    image = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(300, 240),
                                 legends=legends, highlightAtomLists=highlights,
                                 useSVG=False, returnPNG=False)
    pathlib.Path(args.grid_figure).parent.mkdir(parents=True, exist_ok=True)
    # MolsToGridImage returns a PIL image here and raw PNG bytes elsewhere,
    # depending on the RDKit build and whether it thinks it is in a notebook.
    if hasattr(image, "save"):
        image.save(args.grid_figure)
    else:
        with open(args.grid_figure, "wb") as handle:
            handle.write(image.data if hasattr(image, "data") else image)
    print(f"\n  wrote {args.umap_figure}, {args.trend_figure}, {args.grid_figure}")


if __name__ == "__main__":
    main()
