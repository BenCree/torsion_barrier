# What was taken from papers training similar models, and what was not

Read 2026-09-11. Five works that train something comparable. Each row of the
table is a decision here that changed, or was deliberately kept, because of them.

## The protocols

| | TorsionNet | MACE-OFF23 | AIMNet2 | NAGL | espaloma |
|---|---|---|---|---|---|
| what it predicts | torsion profiles | energies and forces | energies, charges, forces, multipoles | partial charges | MM force field parameters |
| training set | 50,000 fragments, **1.2M DFT energies** | ~950,000 configurations | multi-million conformers | Enamine sets | QM small molecules + peptides |
| optimiser | not reported | Adam + amsgrad | **AdamW** | Adam | not retrieved |
| learning rate | not reported | 0.01 then 0.00025, two phase | **reduce on plateau** | 0.001 | not retrieved |
| batch | not reported | not reported | **effective 2048** over 8 GPUs | **1000 molecules** | not retrieved |
| epochs | not reported | until energy error stops falling | **400 to 500** | not reported | not retrieved |
| weight averaging | not reported | **EMA on weights** | not reported | not reported | not retrieved |
| split | not reported | 95/5 **at the molecule level** | not reported | not reported | not retrieved |
| outliers | not reported | **808 configurations removed, force error > 2 eV/A** | not reported | not reported | not reported |
| hardware | cloud DFT plus training | 1x A100, **6 to 14 days** | **8x V100, DDP** | not reported | not retrieved |
| headline | 1.3 kcal/mol RMSD | 0.25 kcal/mol barrier error on TorsionNet500 | | | |

Sources: Rai et al., JCIM 62(4):785, 2022. Kovacs et al., arXiv:2312.15211 and
JACS 2024. AIMNet2, Chem. Sci. 16(23):10228, 2025. openff-nagl's own example
configs. Wang et al., Chem. Sci. 13(41):12016, 2022 and arXiv:2307.07085.
Performance practice from arXiv:2504.16068, Digital Discovery 5(4):1558, 2026.

## Adopted

**Mini-batches, scaled with the training set.** The loop was full-batch: one
parameter update per pass. At 60 molecules that works; at the 34,571 usable
QCArchive records it is one update every 2.4 hours, which is not training. AIMNet2
runs an effective batch of 2048 and NAGL 1000 molecules, both a small percentage
of their sets, so the batch here is 4 percent of the training molecules clamped to
[8, 512]. That gives 25 to 41 updates an epoch at every scale rather than 1.

**Reduce on plateau, not cosine.** AIMNet2 uses plateau and converges in 400 to
500 epochs. Cosine anneals against a fixed epoch budget, and with early stopping
the budget is not what ends training, so the two were mismatched.

**An EMA of the weights, evaluated instead of the raw ones.** MACE-OFF23 does
this. On a small training set it matters more than usual: the validation curve
here swung between 1.85 and 2.30 kcal/mol from one epoch to the next, so choosing
"the best epoch" from raw weights is partly choosing a lucky draw.

**An explicit outlier policy.** MACE-OFF23 removed 808 configurations whose force
error exceeded 2 eV/A. The analogous failure here is a scan whose barrier is
physically impossible, which means the optimisation fell into a different molecule
rather than turning a bond. `--max-barrier` quarantines those with their number
in the table; a large finite barrier is a result to look at and 1e7 is a bug.

**Molecule-level splits**, which MACE-OFF23 also uses. Ours are stricter, by
Bemis-Murcko scaffold rather than at random.

## Not adopted, and why

**A two-phase learning rate.** MACE-OFF23 switches its energy and force weights
between phases because it fits both. There are no forces in this loss, so there
is nothing for a second phase to reweight.

**A multi-target loss.** AIMNet2 weights energy, charges, forces, dipole and
quadrupole together. The target here is one profile per bond and adding terms
without data to constrain them is how the all-bonds reduction went wrong.

**`torch.compile` and custom kernels**, which the NequIP performance paper is
largely about. Worth trying on the DPA3 forward, which is the expensive part, but
it is an optimisation and it comes after the science question is answered.

## The finding that changes the data plan, not the optimiser

MACE-OFF23 deliberately augmented SPICE with QMugs molecules of **50 to 90 atoms**,
stated purpose "to facilitate learning of intramolecular non-bonded interactions".

OpenFF Gen3, which this is currently trained on, is **median 9 heavy atoms** and
was constructed specifically to MINIMISE non-bonded contamination. So a Gen3-only
model cannot learn the intramolecular non-bonded part of a torsion profile, and
TorsionNet500 is drug-like fragments where that part is real. That is a predicted
failure, written before the test set is opened. The fix is to include the larger
fragment sets, XtalPi at 8,737 records and OpenFF benchmark ligand fragments at
8,052, rather than training on Gen3 alone.

## The scale to keep in mind

TorsionNet used 50,000 fragments and 1.2 million DFT energies to reach 1.3
kcal/mol RMSD. This model currently sees 60 molecules. Any comparison before
TorsionNet500 is opened is between different sets on different metrics and is not
a comparison.
