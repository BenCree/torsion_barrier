# Methods: what is computed, and with what

Written 2026-09-11. Versions are the ones the runs actually used, read from the
environments rather than from a requirements file.

## 1. The quantity being predicted

For a central bond `ij` of a molecule and a rotation `phi` about it,

    E(ij, phi)  =  sum_{n=1..6}  k_n(ij) . s_n(ij, phi)          kcal/mol

relative to the minimum of that bond's own profile. Three pieces.

**The phase.** A substituted C-C bond carries up to nine proper torsions
`a-i-j-b` sharing it. Reporting one of them means choosing which, and the answer
then depends on how the atoms happen to be numbered. All of them are combined:

    s_n(ij)  =  sum_{ab}  c_ab [ cos(n * delta_ab), sin(n * delta_ab) ]

`delta_ab` is the dihedral of torsion `a-i-j-b`, `c_ab` is a learned weight, and
`n` runs 1 to 6. The periodicity expansion is not decoration: at n = 1 three
torsions 120 degrees apart sum to exactly zero, so every methyl and every CF3 is
degenerate there and floating point returns a confident angle for an undefined
quantity. A three-fold rotor gives |s_n| = [0, 0, 3, 0, 0, 3] over n = 1..6,
which is its symmetry read off the coordinate. Degenerate harmonics report NaN
with their magnitude beside them; the energy is computed from `s` and never from
the phase angle, which has a branch cut.

**Turning the rotor.** Turning a bond by `phi` adds `phi` to every torsion about
it at once, so

    s_n(ij, phi) = R(n phi) s_n(ij, 0)

with R the 2x2 rotation. Exact, not approximate: verified to 1e-12 against
recomputing the phase from rotated coordinates. A whole 24-point profile
therefore follows from ONE geometry, analytically and with gradients, which is
what a pose refinement needs.

**The coefficients.** `k_n(ij) = NN(h_ij)` where `h_ij` is a Janossy-pooled
representation of the bond:

    h_ab  = NN([h_a:h_i:h_j:h_b]) + NN([h_b:h_j:h_i:h_a])       symmetric under reversal
    c_ab  = NN(h_a + h_b)                                        symmetric under a <-> b
    h_ij  = sum over the torsions of that bond

Both symmetrisations are enforced by construction rather than trained for, which
is Janossy pooling (Murphy et al., ICLR 2019) and the device espaloma uses for
bonds and angles.

Because the energy is LINEAR in `s_n`, it expands to a six-term Fourier series in
the rotor angle with both barrier height `|k_n|` and phase `atan2(k_n1, k_n0)`
fitted. Espaloma fits the height alone at fixed phase 0.

**These are not force field parameters.** Fitted against a total QM profile,
`k_n` absorbs the 1-4 nonbonded contribution that a proper torsion's terminal
atoms retain (AMBER scales those to 1/1.2 and 1/2 rather than excluding them;
Braun et al., LiveCoMS 1(1):5957, 2019, section 3.5). A force field that also
computes 1-4 interactions would count it twice, invisibly. A transferable
parameter has to be fitted against `E_QM - E_MM,no-torsion`, which is a different
target and not what this is.

## 2. The encoder

Per-atom embeddings `h_a`, 128-dimensional, from **DPA-3.1-3M**
(`deepmodelingcommunity/DPA-3.1-3M`, CC-BY-4.0), the shared descriptor of its 31
multi-task branches, read through the `SPICE2` branch. Loaded with 326 tensors
and nothing missing or unexpected. Its pretraining is OpenLAM-v1 and is
predominantly materials: 26 of the 31 branches are metals, oxides, perovskites,
catalysis surfaces and electrolytes, and only 5 are organic chemistry
(`Domains_Drug`, `SPICE2`, `Domains_Transition1x`, `Organic_Reactions`,
`solvated_protein_fragments`). Those 5 are also the only leakage risk against a
drug-like benchmark, and the canonical-SMILES check has not yet been run.

Embeddings are taken from ONE reference conformer per molecule, the lowest-energy
point of its scan, so `h_ij` is a property of the molecule and `k_n` does not
change when it moves. `--geometry per_angle` gives each scan point its own
embeddings and is an arm, not a setting.

## 3. Geometry

Dihedrals by the standard IUPAC construction, one plane normal crossed with the
normalised central bond. The handedness is pinned to an external source: one
optimised QCArchive geometry and the value RDKit's `GetDihedralDeg` computes for
it, kept as an offline fixture. Measured against torsiondrive's own grid keys on
3 molecules by 24 points, agreement to 0.010 degrees, and on the 100-molecule
cohort to 0.0000 degrees wrapped.

Only the DRIVEN bond is enumerated. A torsiondrive turns one bond, so the profile
constrains only that bond's energy; summing over all central bonds fits a target
that cannot attribute it. On these molecules that is a median of 48 torsions
across 8 bonds reduced to 6 torsions across 1, an 8.8-fold reduction through the
pooling and coefficient networks.

## 4. Data

QCArchive torsiondrive records, all at **B3LYP-D3BJ/DZVP in Psi4**, the OpenFF
default specification, so train and test are one level of theory. Per record:
every optimised scan geometry in Angstrom, its energy in kcal/mol relative to
that scan's own minimum, element symbols, the archive's bond connectivity WITH
bond orders, the driven torsion's atom indices, and canonical SMILES. Fetched
once and cached; nothing is derived at fetch time.

Splits are three-way over MOLECULES by Bemis-Murcko scaffold, `astartes`
scaffold sampler, verified to leave zero scaffolds shared between train and test.
Validation molecules come out of TRAIN and never out of test. Never over scan
points.

## 5. Optimisation

AdamW. Head at 3e-3 with weight decay 1e-2; encoder at 3e-5 with NO weight decay
and an L2 anchor to its pretrained weights instead, because decay pulls toward
zero and a pretrained representation does not belong there. Cosine annealed.
Gradients accumulated one molecule at a time, which is identical arithmetic to
one backward over the batch and frees each graph as it goes; holding all of them
reached 15.3 GiB on an 18.4 GiB card and stopped. Gradient norm clipped at 5.
Early stopping on validation RMSE with patience 60, and **the kept model is the
best validation epoch, not the last**.

## 6. Software

| | version |
|---|---|
| DeePMD-kit | 3.2.0, CUDA 12.9 build |
| PyTorch | 2.12.0 |
| DPA-3.1-3M | HuggingFace `deepmodelingcommunity/DPA-3.1-3M`, 47 MB checkpoint |
| qcportal | 0.59 |
| RDKit | 2026.03.6 (MMFF94 baseline, scaffolds) |
| astartes | 1.3.3 (scaffold splits) |
| openff-toolkit | 0.16.8 |
| NumPy / SciPy / pandas | 2.5.3 / 1.18.0 / 3.0.5 |
| matplotlib | 3.11.1, Agg |
| Snakemake | 9.x, python 3.14 |
| claimcheck | `/home/ben/code/claimcheck` |
| torsion modules | `openff-nagl` fork `BenCree/openff-nagl`, branch `add-bond-example`, commit 009014b |
| GPU | NVIDIA RTX 4000 Ada, 18.75 GiB, driver 550.127.05 |

Two environments because DeePMD-kit caps at Python 3.12: `dpa3` (3.12.14) runs
the encoder and the training, `nagl` (3.13.2) runs QCArchive and the analysis.
The torsion modules are loaded from the fork by path rather than installed, so
the encoder's environment does not carry PyTorch Lightning to satisfy one import.
Both environments are pinned in `envs/dpa3.explicit.txt` and
`envs/nagl.explicit.txt`.

## 7. Outputs

| file | what |
|---|---|
| `results/loss.csv` | **the loss curves**. One row per epoch per arm: `encoder, epoch, train_mse, val_rmse, best_val_rmse, best_epoch, seconds`. Written and flushed every epoch, so it can be read during a run. |
| `results/training_curves.png` | redrawn from it every 25 epochs, in-process, measured at about 0.35 s a redraw |
| `data/per_molecule_arms.csv` | one row per molecule per arm: RMSE, barrier error, Spearman, split, kept epoch, rotor counts |
| `data/profiles_arms.csv` | one row per scan point per arm |
| `data/*_quarantine.csv` | every skipped record with its reason |
| `results/claims.json` | the registered claims recomputed, with intervals and verdicts |

Nothing computes outside Snakemake. Every number above is produced by a rule
whose inputs include the scripts and the three library files, so editing the
architecture invalidates the result rather than silently reusing it.
