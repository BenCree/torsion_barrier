# Tutorial: what a torsion term cannot draw

Twenty minutes, no GPU, two public benchmarks, three rules.

A molecular mechanics torsion term is `k·cos(nφ)`. Cosine is symmetric, so it
can only produce profiles where turning a bond +30° and −30° cost the same. This
tutorial measures how much of a real QM torsion profile is *not* like that.

## Set up

```bash
mamba env create -f environment.yml
mamba activate torsion-phase-tutorial
```

Five packages: numpy, scipy, matplotlib, rdkit, snakemake. No torch.

## Run it

```bash
snakemake -n            # always look at the job list first
snakemake -c4
```

That clones Schrodinger's public MLFF validation data (598 MB), converts
TorsionNet500 and TorsionTest2000 into this project's npz, and decomposes every
profile in both.

## What you should see

Each scan's 24 QM energies are fitted twice: once allowing sine terms, once not.
The sine terms are the asymmetric part, and they are exactly what a fitted phase
buys and a fixed-phase force field cannot have.

The controls come first, because a census without them cannot tell a real signal
from a broken fit. These are the named molecules that are actually present in
the two benchmarks, with the value the tutorial prints:

| molecule | asymmetry | why |
|---|---|---|
| butane | **0.0000** | symmetric rotor |
| biphenyl | **0.0000** | symmetric twofold rotor |
| styrene | **0.0000** | planar, twofold |
| N-methylacetamide | **0.0000** | the amide is even |
| 2-butanol | 0.1239 | a stereocentre beside the rotor breaks the mirror |
| 2-methoxytetrahydropyran | 0.8131 | anomeric **and** chiral at the anomeric carbon |

**If butane or biphenyl comes back with any asymmetry, the fit is wrong and
nothing else in the output means anything.** That is what
`results/tutorial_named.csv` is for. Ethane would be the cleanest control of all
and is not in either benchmark, which is itself worth noticing about them.

Then the headline. The tutorial prints one line per benchmark:

```
torsionnet500      265 scans   odd 0.0000   gain 0.000   JS 0.002
torsiontest2000   1701 scans   odd 0.1049   gain 0.386   JS 0.157
```

**TorsionNet500 has no asymmetry at all.** It is small neutral fragments with
symmetric rotors, and the phase buys nothing on it: 0.000 kcal/mol, a
Jensen-Shannon distance of 0.002. TorsionTest2000, which spans eleven elements
with some ionic species, gives 0.386 kcal/mol and 0.157.

For scale, presto's Table 1 puts the *entire* error of Sage 2.3.0 at a
Jensen-Shannon distance of 0.30 on TorsionNet500 and presto's own at 0.12. So on
TorsionTest2000 the asymmetric part alone costs more than the gap between two
force fields, and on TorsionNet500 it costs nothing.

That is the point of running both. **The benchmark the field reports on cannot
detect the effect.** Scaled up, 2.4 million THEMol scans give 0.196 overall and
0.312 on rotors beside a stereocentre.

## What gets thrown away, and why

622 of 2,588 scans are quarantined, almost all with
`KekulizeException: Can't kekulize mol`. The importer builds the molecule from
the archive's own bond orders, and RDKit cannot kekulize an aromatic nitrogen
ring without knowing which nitrogen carries the hydrogen. Tetrazoles and
imidazoles are the usual casualties, and TorsionNet500 loses more than half its
scans this way, so its 265 is a subset rather than the full 497.

`results/tutorial_quarantine.csv` lists every one with its reason. A run that
silently kept them would be worse than one that drops them loudly.

## Then look at the numbers yourself

```python
import csv, numpy as np
rows = list(csv.DictReader(open("results/tutorial_census.csv")))
for source in sorted({r["source"] for r in rows}):
    odd = [float(r["odd_fraction"]) for r in rows if r["source"] == source]
    js  = [float(r["js_distance"])  for r in rows if r["source"] == source]
    print(f"{source:<18}{len(odd):>6} scans   odd {np.median(odd):.4f}   "
          f"JS {np.median(js):.3f}")
```

`results/tutorial_census.csv` has one row per scan with its chemistry class,
whether its substituents are symmetry-equivalent, the six-harmonic residual, how
far the minimum moves when the asymmetric part is removed, and the
Jensen-Shannon distance between the two Boltzmann populations at 500 K.

## What to try next

- `--format themol` on `phase_census.py` reads THEMol's HDF5 shards directly,
  4.2 million scans at the same level of theory. See `../DATA.md`.
- `cut -d, -f7 results/tutorial_census.csv | sort | uniq -c` for the class
  breakdown. Biaryls, amides and conjugated systems come back even; rotors
  beside a stereocentre do not.
- `../workflow/pose_census.py` applies the same classifier to docked ligands,
  which is how the 22.8 percent figure in the top-level README was obtained.

## What this tutorial does not cover

The model arms: training a torsion head on a pretrained encoder needs torch,
deepmd or DGL, and a GPU. Those live in `../workflow` and the headline results
from them are in the top-level README, including the ones that failed.
