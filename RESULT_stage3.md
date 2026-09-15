# Stage 3: the torsion head does not beat the potential it is built on

Run 2026-09-11, recorded 2026-09-12. 870 OpenFF Gen3 molecules at
B3LYP-D3BJ/DZVP, 5 scaffold-grouped folds, every molecule held out exactly once
per arm. The prediction registered in TRAINING_PLAN.md was that a
permutation-invariant torsion phase on a fine-tuned DPA3 encoder would predict QM
torsion profiles competitively. **It does not, and the margin is large.**

## The result

Profile RMSE against QM, held out, paired by molecule.

| | median | <1.0 kcal/mol | <0.5 |
|---|---|---|---|
| DPA-3.1-3M zero-shot, no training at all | **0.521** | 84% | 47% |
| our head, per-angle fine-tuned | 1.175 | 42% | 10% |
| our head, frozen encoder | 1.289 | 33% | 9% |
| our head, reference-conformer fine-tuned | 1.467 | 28% | 7% |
| flat profile (control) | 3.122 | | |

Paired on the two completed frozen folds, the pretrained potential wins on 83 to
91 percent of held-out molecules, p = 4.6e-19 and 1.5e-30. Our tenth percentile
is the potential's median.

The head does clear its controls decisively: 166 of 174 and 161 of 174 against a
flat profile, and 229 of 304 against MMFF94 whose median is 2.10 to 2.23. Beating
a 1996 force field is not the relevant bar once a pretrained potential is sitting
at 0.52 on the same rows.

## Why, as far as the measurements go

**Under-fitting, not miscalibration.** Predicted barrier against QM barrier has
slope 0.695 with intercept +1.185: the head compresses everything toward an
average profile. A single global rescale fitted on one fold and applied to
another changes the RMSE by 0.001, so it is not a scale error. DPA3 sits at slope
1.053.

**The failures are diffuse.** On a UMAP of ECFP4 space the head's errors are
scattered across the whole map while the potential's are localised to a few
points. There is no chemotype to fix.

**Data will not close it.** 10x the training data (60 to 604 molecules) bought
10.2 percent, a scaling exponent of 0.047 where molecular ML typically sits at
0.2 to 0.5. Extrapolated, the entire 13,167-molecule usable pool projects to
1.22 against the potential's 0.52; reaching 0.52 by data alone needs about 1e12
molecules. Two points and a changed training setup, so an estimate rather than a
law, but even a four-times steeper exponent needs seven times the whole pool.

**Fine-tuning the encoder buys almost nothing.** Updating it while showing it one
conformer: 0.5 percent, plateaued. Updating it while showing it all 24: 2.7
percent, and that arm confounds fine-tuning with the extra geometry. The control
that separates them, `frozen_per_angle`, was launched as job 79322.

## What is worth keeping

The negative result is the smallest part of what this produced.

**DPA-3.1-3M beats MACE-OFF23 on TorsionNet500** and the ranking reverses on a
matched draw. See RESULT_benchmarks.md.

**Published TorsionNet-500 numbers are not comparable across papers.** Two DFT
references for the same 497 molecules disagree by 0.767 kcal/mol mean on barrier
heights, 25 percent of them by more than 1 kcal/mol, while the labels themselves
reproduce to 0.0008. Any number quoted on that benchmark is a statement about
which reference was used.

**The code.** A corrected dihedral with its handedness pinned to an external
fixture, a permutation-invariant phase with the periodicity expansion that makes
it defined for three-fold rotors, an energy linear in that phase with an exact
closed-form rotation verified to 1e-12, 78 tests. A 20x faster QCArchive fetch.
All of it independent of whether the head is accurate.

## What would have to change to revisit this

Not more data and not more training. The head sees one conformer's dihedrals and
128 numbers per atom and must emit 12 coefficients explaining 24 points; the
potential gets a fresh geometry at every point. Either that restriction is
relaxed, which costs the analytic-profile property that was the point, or the
architecture gains capacity in a way that does not just memorise. Neither is a
small change and neither is indicated by anything measured here.
