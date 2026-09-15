# Stage 2: a frozen DPA3 encoder beats predicting nothing, and does not beat MMFF94

Run 2026-09-11. 100 OpenFF Gen3 torsiondrives at B3LYP-D3BJ/DZVP in Psi4, 24
points per scan, split by molecule 75 train and 25 held out, DPA-3.1-3M
embeddings frozen at one reference conformer per molecule.

## The result

**The model clears its own control and does not clear the force field.**

| comparison, held out, paired by molecule | model ahead | p |
|---|---|---|
| against a flat profile, RMSE | 19 of 23 | **0.0026** |
| against MMFF94, profile shape (rho) | 15 of 23 | 0.210 |
| against MMFF94, profile RMSE | 13 of 23 | 0.678 |

Pooled within-scan rank correlation against QM, held-out rows only, bootstrap
over molecules:

| arm | rho | 95% CI |
|---|---|---|
| this model | +0.603 | [+0.468, +0.719] |
| MMFF94 | +0.435 | [+0.234, +0.604] |
| random | +0.005 | [-0.116, +0.105] |

The model's interval and MMFF94's overlap. **Both MMFF94 comparisons are
UNRESOLVED, not null**: 23 molecules cannot separate them, and the median paired
differences, +0.061 in rho and -0.134 kcal/mol in RMSE, both favour the model
without reaching significance.

The stage 2 bar in PLAN_torsion_prediction_2026-09-11.md was "profile RMSE below
the constant control, with an interval excluding it". **Met.**

## H1 as registered is in-sample and must not be quoted as generalisation

`H1_the_model_tracks_the_qm_profile` returned `agrees` at rho = 0.876
[0.824, 0.919] and that number is contaminated. The claim named the column and
not the split, so it was recomputed over all 100 molecules, 75 of which the model
was fitted on, while MMFF94 has no training and was out of fold throughout. The
same statistic on held-out rows alone is **+0.603**, which would have refuted the
claimed 0.85.

This is a defect in how the claim was written, not in the data. The fix for
stage 5 is a held-out-only table declared as its own dataset, with the claim
registered against it before the run. The verdict file keeps its record as it
stands; no claim has been rewritten after seeing a result.

## K2 is refuted, and the refutation is about the estimator

`K2_the_geometry_carries_the_dihedral_the_archive_recorded` claimed 1.0 and
returned 0.870 [0.849, 0.894]. The data agree perfectly: the dihedral computed
from the stored geometry matches the value torsiondrive constrained it to at
**0.0000 degrees for all 2400 points**, wrapped. The shortfall is entirely the
+-180 wrap, where 54 of 100 molecules compute their 180 degree point as -180.0,
the same angle with the opposite rank, pinning those molecules at exactly 0.7600.

A rank correlation does not respect a circular quantity. The claim stands refuted
as registered. The circular-safe form is a claim on cos and sin rather than on
ranks, and it is registered that way for stage 4.

## What the profile figure shows, which the summary statistics do not

`results/torsion_profiles.png`, six held-out molecules across the barrier range.
Two systematic errors, both visible and both actionable.

**The amplitude overshoots.** On `C1CC=C(C1)O`, QM barrier 4.9 kcal/mol and the
model predicts about 9.3, with the shape essentially correct. The same on
`C(C(Cl)(Cl)Cl)S(=O)(=O)N`, 5.6 against about 8. This is the dominant error and
it is a scale problem rather than a shape problem, which is why the rank
correlations look better than the RMSE does.

**Spurious high-frequency structure.** On `c1c(cc(cc1O)F)C2CC2` the QM profile is
a flat-topped two-fold well and the model adds a three-fold oscillation on top of
it. Two candidate causes, and they are separable: 12 free coefficients per rotor
on 75 molecules with no regularisation, and the energy summing over EVERY central
bond in the molecule rather than the driven one, so the non-driven dihedrals,
which relax during a torsiondrive, contribute their own series.

## What this says about the next stage

The encoder is frozen and the head is small, so the honest reading is that this
measures what a frozen OpenLAM representation carries, not what the architecture
can do. Three things to change before stage 5, in the order they cost:

1. Restrict the energy to the driven rotor, or report both, and see whether the
   high-frequency structure goes with it. Free.
2. Regularise the coefficients, or shrink toward a small number of periodicities.
   Free.
3. Fine-tune the encoder rather than freezing it. The GPU spend.

None of these should run before a held-out-only claim table exists, because the
stage 2 result above is the second time in this project that an in-sample number
looked like a generalisation number.
