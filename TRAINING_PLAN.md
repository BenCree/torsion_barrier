# Training plan: (bond, angle) to energy

Written 2026-09-11, before the campaign. Supersedes the staging in
PLAN_torsion_prediction_2026-09-11.md for everything downstream of stage 2.

## What is being learned

    E(ij, phi) = sum_n  k_n(ij) . R(n phi) s_n(ij, 0)

`k_n` comes from the chemistry through the pooled atom embeddings and does not
depend on where the rotor sits. `s_n(ij, 0)` comes from the dihedrals of one
conformer. `R(n phi)` turns the rotor analytically, so the whole profile follows
from a single geometry, exactly, and with gradients. Verified to 1e-12 against
recomputing the phase from rotated coordinates.

That is the deliverable: a function of a bond and an angle, not a single number
per molecule. At inference a molecule's torsion potential is the sum over its
rotatable bonds, each predicted independently.

## What the data can supervise, and what it cannot

A torsiondrive turns ONE bond. Its profile is one number per scan point. So the
only reduction the data identifies is the driven bond's own energy. Summing a
Fourier series over every central bond, which the first implementation did, fits
a target that constrains only the sum: on OpenFF Gen3 that is a median of 48
torsions across 8 bonds predicted against 24 points, with 7 of the 8 bonds not
moving. Restricting to the driven bond removes 8.8 times the torsions from the
head and makes the per-bond function identifiable.

The approximation that remains, stated because it does not go away: the QM
profile is not purely the driven bond's torsion energy, since everything else
relaxes as it turns. Attributing it to that bond is what force field torsion
fitting does. OpenFF Gen3 was constructed to make that valid, one effective
rotating bond and minimal non-bonded contamination. TorsionNet500 is real
drug-like fragments and will not be as clean. The gap between them measures how
much, and is a result rather than a failure.

## Data

All at B3LYP-D3BJ/DZVP in Psi4, so train and test are one level of theory.

| role | sources | records |
|---|---|---|
| train and validate | gen3, xtalpi, fragments1, fragments2, silicontx, protein_frags, group1 | 24,649 |
| **held out** | torsionnet500 | 500 |
| **excluded** | the four "OpenFF SMIRNOFF Sage" sets | 4,674 |

The Sage sets are Sage's own fitting data. Evaluating there would put the
standard in-sample and this model out.

Acquisition is `pull_all.sh`, one Snakemake rule per source, about 35 hours of
QCArchive fetching, done once and cached. TorsionNet500 is pulled first and is
never trained on, never validated on, and not looked at until stage 4.

## Splits

`astartes` (Burns et al., JOSS 8(91):5996, 2023), **scaffold** sampler. Verified
on the first 100: zero Bemis-Murcko scaffolds shared between train and test,
where a random split would put near neighbours on both sides.

* Outer: 5-fold, grouped by scaffold. Every molecule is held out exactly once, so
  the paired test has n = every molecule regardless of k.
* Inner: 15 percent of each fold's TRAIN, for early stopping. Never from test.
* Never over scan points. 23 points of a scan in train and the 24th held out
  measures interpolation inside one profile.

Five folds rather than ten: k does not change the held-out count, only the
training fraction, and on 14 GPUs the spare capacity buys arms instead.

## Arms

Seven, because on 14 free GPUs 7 arms by 5 folds is 35 jobs at the same 7.8 hour
makespan as 3 arms by 3 folds. Each answers a different question; the pairwise
comparisons that will be reported are named here, before the run.

| arm | what it asks |
|---|---|
| A1 frozen, driven rotor | what a pretrained OpenLAM representation already carries |
| A2 fine-tuned, reference conformer, driven | does tuning the encoder help when it never sees a rotated geometry |
| A3 fine-tuned, per-angle, driven | **the main arm** |
| A4 frozen, all bonds | what restricting to the driven rotor bought |
| A5 fine-tuned per-angle, nonlinear head | what the Fourier linearity costs |
| C1 random coefficients | whether the design resolves anything |
| C2 flat profile | whether a profile was predicted at all |

Declared comparisons: A3 against A1, A3 against A2, A1 against A4, A3 against A5,
every arm against C1 and C2, and every arm against MMFF94. Twelve, fixed now so
they cannot be chosen after the fact.

External baselines at stage 4 on TorsionNet500: MMFF94, OpenFF Sage, g-xTB
(Froitzheim et al.), ANI-2x, MACE-OFF23 M and L. MACE-OFF23's published 0.25
kcal/mol mean barrier error is on this benchmark and is the number to beat; our
own protocol is labelled as ours until the published one is reproduced.

## Optimisation

Held identical across arms: split, seed, head initialisation, epoch budget, head
learning rate, stopping rule. Only the named difference varies.

* AdamW, head learning rate 3e-3, weight decay 1e-2.
* Encoder learning rate 3e-5, cosine annealed, **no weight decay**: decay pulls
  toward zero, which is not where a pretrained representation belongs.
* An L2 anchor to the pretrained encoder weights instead, which is the "L2 toward
  the initial model" the fine-tuning literature prefers for small sets.
* Gradients accumulated one molecule at a time. Holding all of them reached
  15.3 GiB on an 18.4 GiB card and stopped.
* Early stopping on validation RMSE, patience 60, and **the kept model is the
  best validation epoch, not the last**. Without this the comparison measures
  which arm overfits fastest: on 60 molecules the frozen arm's training MSE fell
  two orders of magnitude while validation turned upward at epoch 138.

## What is reported

Per CLAUDE.md: a p value, an interval with its resampling unit named, and R
squared. The resampling unit is the molecule, never the scan point.

1. **Profile RMSE against QM**, kcal/mol. The standard.
2. **Barrier error**, kcal/mol, the quantity the published comparisons use.
3. **Within-scan Spearman**, which separates the minimum being in the right place
   from the scale being right.
4. **Cost**, GPU-seconds per profile, measured on a named card.

Paired by molecule with a sign test, because the arms share a split. At n = 888
the design resolves a 53.5 percent win rate; at the n = 23 of stage 2 it needed
72 percent, which is why the MMFF94 comparison came back unresolved rather than
null.

## Stages

| stage | cohort | arms | cost | the bar |
|---|---|---|---|---|
| 3 | gen3, 888 | 7 x 5 folds | 7.8 h on 14 GPUs | A3 beats C2 with an interval excluding it, and the driven-rotor restriction beats all-bonds |
| 4 | + the full 24,649 | the arms that survived stage 3 | pull 35 h, train 2 to 3 days | profile RMSE below MMFF94 with an interval excluding it |
| 5 | TorsionNet500, first look | best arm only | hours | our numbers for the full external panel on one set of rows |
| 6 | PoseBusters | best arm | 20 to 40 GPU-h | fraction of poses passing the internal-geometry checks, against the same panel |

Stage 4 does not start until stage 3 has a verdict, and stage 5 opens
TorsionNet500 exactly once.

## What would stop this

* A3 does not beat A1. Fine-tuning the encoder buys nothing and the frozen
  representation is the whole story.
* The driven-rotor restriction does not beat all-bonds. Then the identifiability
  argument was wrong and something else explains the spurious structure.
* Nothing beats MMFF94 at n = 888. A 1996 force field is then the honest answer
  and the method does not ship.
* Cost per profile exceeds g-xTB for no accuracy gain.
