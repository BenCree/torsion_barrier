# Findings

Everything measured, positive and negative, with the cohort and the number.
Dates are when the measurement ran, 2026-09-13 to 2026-09-15.

The project set out to build a torsion-profile predictor on a pretrained
atomistic encoder. **It does not beat anything.** The measurement instrument
built to justify it found something the model did not: a structural limitation
shared by every learned torsion force field, which is invisible on the benchmark
they are all validated against.

---

## 1. What a cosine torsion term cannot draw

A molecular mechanics torsion term is `k·cos(nφ)`. Cosine is symmetric, so such
a term can only produce profiles where turning a bond +30° and −30° cost the
same. Fitting each QM profile twice, once allowing sine terms and once not,
measures how much of a real profile is the asymmetric kind. That difference is a
**ceiling** on what any phase-fitting architecture can gain over any fixed-phase
one, so it bounds espaloma, Grappa, Sage and this project's own head at once.

### 1.1 The asymmetry is stereochemistry, not the electronic effects one would guess

THEMol TorsionScan, **2,465,308 scans**, B3LYP-D3(BJ)/DZVP. `gain` is kcal/mol,
`JS` is the Jensen-Shannon distance between the Boltzmann populations at 500 K
with and without the asymmetric part.

| cut | n | odd | gain | min shift | JS |
|---|---|---|---|---|---|
| all | 2,465,308 | 0.096 | 0.436 | 9.5° | 0.196 |
| substituents symmetry-equivalent | 38,710 | 0.011 | 0.061 | 6.0° | 0.035 |
| **stereocentre adjacent** | 945,751 | **0.266** | **0.796** | 17.5° | **0.312** |
| gauche effect | 221,994 | 0.194 | 0.655 | 15.5° | 0.307 |
| plain | 543,288 | 0.149 | 0.409 | 33.5° | 0.210 |
| anomeric | 560,456 | 0.030 | 0.362 | 4.0° | 0.134 |
| amide | 340,991 | 0.031 | 0.375 | 4.5° | 0.141 |
| biaryl | 97,986 | 0.002 | 0.055 | 4.5° | 0.148 |
| conjugated | 988,100 | 0.018 | 0.216 | 3.5° | 0.094 |

Biaryls, amides, conjugated systems and anomeric centres come back **even**.
1,2-difluoroethane is exactly 0.0000 despite the gauche effect being the
textbook electronic case: those effects change which minimum is deepest, not
where it sits. What is odd is rotors without local mirror symmetry, of which a
neighbouring stereocentre is the cleanest detector rather than the cause.

`workflow/phase_census.py`, `cluster/themol_census.sbatch`

### 1.2 RMSE hides it; the population overlap does not

Removing the asymmetric part costs 0.436 kcal/mol of RMSE, which looks
negligible, and a Jensen-Shannon distance of 0.196, which is not. presto's own
paper makes the same argument for reporting JS: the RMSE "is sensitive to
barrier height errors, which do not affect equilibrium distributions". Its
Table 1 gives whole-force-field errors on TorsionNet500 of **Sage 0.30,
espaloma 0.20, presto 0.12, AceFF 2.0 0.09**.

So dropping the phase costs about as much as the gap between the best and worst
force field in that table, and on stereocentre-adjacent rotors slightly more.
As a population statement, JS 0.3 is getting a 50/50 equilibrium wrong by 20/80.

### 1.3 Controls

| molecule | asymmetry | expected |
|---|---|---|
| ethane | 0.0000, 99.96% of power at n=3 | symmetric |
| butane | 0.0000 | symmetric |
| biphenyl | 0.0000 | symmetric |
| styrene | 0.0000 | symmetric |
| N-methylacetamide | 0.0000 | symmetric |
| 1,2-difluoroethane | 0.0000 | symmetric |
| 2-butanol | 0.1239 | asymmetric, stereocentre |
| methanediol | 0.5823 | asymmetric |
| 2-methoxytetrahydropyran | 0.8131 | asymmetric |

A fit that gives ethane a phase is broken and nothing else in its output means
anything. `workflow/phase_census.py` writes these to `phase_named.csv` on every
run.

### 1.4 It is not relaxation

A torsiondrive re-optimises everything else at each angle, so the asymmetry
could be the rest of the molecule settling rather than the rotor. It is not.
Splitting each molecule at the driven bond, superposing the larger fragment on
its own first frame and taking the residual heavy-atom RMSD isolates relaxation
from rotation. 13,041 scans:

| relaxation band (Å) | equivalent substituents | distinct |
|---|---|---|
| 0.000 – 0.020 | 0.000 (n=186) | 0.007 (n=2,423) |
| 0.020 – 0.047 | 0.001 (n=142) | 0.027 (n=2,467) |
| 0.047 – 0.148 | 0.018 (n=102) | 0.072 (n=2,507) |
| 0.148 – 0.563 | 0.011 (n=110) | 0.114 (n=2,499) |
| 0.563 – 3.433 | 0.006 (n=99) | **0.181** (n=2,510) |

**At matched relaxation the two columns diverge rather than converge**, to a
factor of 30 in the top band. And among the quarter of scans that barely move at
all, under 0.025 Å, **968 of 3,260 are still above 0.20 odd**. Relaxation
modulates the effect (ρ = +0.202, p = 3.3e-120, about 4% of the variance) and
does not produce it: a rotor with no local asymmetry to express does not acquire
one by relaxing. `workflow/relaxation_confound.py`

---

## 2. The benchmark the field reports on cannot detect it

| set | scans | asymmetry | JS cost |
|---|---|---|---|
| **TorsionNet500** | 265 | **0.0000** | **0.002** |
| TorsionTest2000 | 1,701 | 0.1049 | 0.157 |
| drug-like OpenFF pool | 15,289 | 0.061 | 0.118 |
| THEMol | 2,465,308 | 0.096 | 0.196 |

TorsionNet500 is 500 small neutral fragments with symmetric rotors. Its median
odd **magnitude** is 0.0003 kcal/mol against the drug-like pool's 0.5267.
espaloma, Grappa, presto and TorsionNet all validate on it.

Reproducible on a laptop in five seconds: `tutorial/`.

---

## 3. It is common in the ligands people run FEP on

OpenFE Industry Benchmark 2024, **2,146 ligands, 11,091 rotatable bonds**,
median 5 per ligand:

| class | bonds | share | ligands with ≥1 |
|---|---|---|---|
| **stereocentre adjacent** | 2,525 | **22.8%** | **714 (33%)** |
| conjugated | 4,357 | 39.3% | 1,662 (77%) |
| plain | 3,105 | 28.0% | 1,326 (62%) |
| anomeric | 1,974 | 17.8% | 1,173 (55%) |
| biaryl | 1,288 | 11.6% | 1,170 (55%) |
| substituents equivalent | 1,540 | 13.9% | 884 (41%) |
| amide | 1,006 | 9.1% | 705 (33%) |

The affected class occurs at 22.8% here and 23.3% in the QM pool, so the
measurement cohort is representative of the application on the axis that
matters. Amide (2.9% vs 9.1%) and symmetry-equivalent (4.9% vs 13.9%) differ
more. `workflow/pose_census.py`

---

## 4. espaloma's trained model partly reproduces the asymmetry, and not at the driven bond

12,978 scans. Ratio is espaloma's odd magnitude over QM's; similarity is the
cosine between the two six-vectors, where 0 is random.

| | odd ratio | direction similarity |
|---|---|---|
| all torsions, all scans | 1.011 | +0.267 |
| all torsions, top quartile by QM odd | 0.452 | +0.524 |
| **driven bond only, top quartile** | **0.220** | **+0.059** |

Its functional form can reach a bond-level phase by summing fixed-phase terms
over torsions with different substituent offsets, and it partly does, but the
part it reproduces comes from the *other* torsions relaxing. For the bond being
turned it is indistinguishable from random. `workflow/espaloma_phase_check.py`

---

## 4b. More data helps this head, and fine-tuning stops helping

The per-bond head on espaloma's embedding, driven bond, scaffold-hashed split
with both arms holding out identical rows by construction.

| cohort | frozen | encoder fine-tuned |
|---|---|---|
| 855 molecules, Gen3 | 1.595 | **1.368** |
| **15,468 molecules, full pool** | **1.166** | 1.207 |

Held out: 213 and **3,893** scans, zero rows in one arm and not the other. On the
full pool, rho +0.844 against MMFF94's median 1.866 and a flat profile's 3.135.
Training 163.5 minutes frozen, 261.7 fine-tuned.

**Two things reverse.** Eighteen times more data took the frozen arm from 1.595
to 1.166, a 27 percent reduction. This project measured a data-scaling exponent
of **0.047** from 60 to 604 molecules on the DPA-3.1-3M encoder, which predicts
essentially no gain; **that exponent does not transfer** to a purpose-trained
graph encoder at this scale, and any argument resting on it is about DPA3 only.

And fine-tuning the encoder **helped at 855 molecules and hurts at 15,468**,
1.166 to 1.207, while costing 60 percent more wall clock. With 529 training
molecules an unfrozen encoder buys capacity; with 11,600 the frozen
representation is already sufficient and moving it overfits.

For scale, espaloma's own published figure is 1.18 and presto's 0.62, on a
different cohort with per-angle relaxation, so those are not a like-for-like row.

`cluster/torsion_espaloma_full.sbatch`, job 79460

---

## 5. Model comparison on full 24-point profiles

296 multi-rotor molecules of the OpenFF pool, driven bond, against QM.

| model | RMSE | min shift | JS | n |
|---|---|---|---|---|
| MMFF94 (whole force field) | 0.615 | 0.0° | 0.103 | 268 |

MMFF94's **mean** is not quotable anywhere in this project: on the full pool it
reads 903.3 kcal/mol against a median of 1.866, because RDKit's perception fails
on a handful of charged or hypervalent species and returns an astronomic energy
rather than an error. Medians only.
| espaloma 0.3.2 | 1.083 | 22.5° | 0.152 | 298 |
| Sage 2.3.0 | 1.124 | 45.0° | 0.168 | 298 |
| flat profile | 2.650 | 45.0° | 0.456 | 300 |

MMFF94's column is the total force field, every other row is the torsion term
alone, which is why it wins. `workflow/qm_profile_benchmark.py`

### Ceilings

| | kcal/mol |
|---|---|
| best possible six-harmonic fit | 0.1736 |
| best possible three harmonics (Grappa's form) | 0.4543 |
| two DFT references on the same 497 molecules | 0.767 |

The last bounds every RMSE anyone reports against either reference.

---

## 6. Cost, including the axis nobody reports

50 molecules, every rotatable bond, on one RTX-class card.

| stack | 1 conformer, 24 angles | 24 conformers | 360 angles |
|---|---|---|---|
| DPA-3.1-3M + this head | 18.772 ms | **450.303 ms** | 18.811 |
| espaloma stage 1 + this head | 1.582 ms | **1.582 ms** | 1.622 |
| espaloma's own readout | 5.536 ms | **5.536 ms** | 5.583 |

Flat in angles for all three: 24 to 360 costs 0.04 ms, the numpy query. Flat in
**conformers** only for graph encoders, because an atomistic descriptor reads
the geometry and must be rerun. Published elsewhere for scale: espaloma
parametrises a system in under 100 µs on a GPU; presto takes about 900 s per
molecule; AceFF-2 runs at 36.7 ns/day on an RTX 4090 in MLIP/MM.
`workflow/cost_encoders.py`

---

## 7. Negative results

Stated with the same weight as the rest.

**A fitted phase does not help this head.** Per-torsion readout on espaloma
embeddings, identical in every respect except 6 or 12 outputs, 213 held out:

| arm | RMSE | min shift | JS |
|---|---|---|---|
| unphased, frozen | 1.220 | 60.0° | 0.309 |
| phased, frozen | 1.128 | 75.0° | 0.308 |
| unphased, fine-tuned | 1.288 | 60.0° | 0.309 |
| phased, fine-tuned | 1.204 | 45.0° | 0.325 |

Paired: 112 of 213 on RMSE frozen, 111 of 213 fine-tuned. A coin flip, and
fine-tuning the encoder does not recover it. The census says the asymmetry is
there and worth having; this head does not capture it.
`workflow/phased_torsion.py`

**Per-bond supervision is null.** Giving every scanned bond of a molecule one
shared embedding rather than letting each scan embed its own geometry, 3,941
scans held out by both arms with zero rows in one and not the other: median
paired improvement **−0.0133 kcal/mol [−0.0252, +0.0020]**, sign test 1,917 of
3,941, p = 0.091. `workflow/compare_supervision.py`

**Subtracting the MM baseline made it worse**, 1.562 against 1.368.
`E_QM − E_MM,no-torsion` with Sage 2.3.0 and its own NAGL charges. The baseline
correlates 0.87 to 0.997 with the QM profile, so subtracting it removes most of
the signal and leaves the force field's own errors, which are not a smooth
function of chemistry. `workflow/mm_baseline.py`

**Torsional strain does not rank docked poses.** 2,146 ligands, 47,182 rows,
gnina at `--num_modes 10`: Spearman between a mode's strain and its RMSD to the
crystal pose is **0.055** for both Sage and espaloma. Significant at that n and
negligible in size. `workflow/pose_strain.py`

**Pretrained potentials are better on expanded-element and ionic chemistry, not
worse.** TorsionTest2000, 2,088 scans, 11 elements, 114 ionic: MACE-OFF23 large
0.172, DPA3:SPICE2 0.304, MACE-OFF23 small 0.387 kcal/mol, against 0.438, 0.386
and 0.534 on TorsionNet500. Part of that is a change of reference, since
TorsionTest2000 is wB97X-D3BJ/def2-TZVPD, much closer to MACE-OFF23's own
training level. Either way the niche is not there.

**A pretrained atomistic encoder is the wrong choice for this.** DPA-3.1-3M
costs 12× espaloma's stage 1 per molecule, 450 ms against 1.6 ms across 24
conformers, and scores no better on the same split. On the full pool the
espaloma head reaches 1.166 where the DPA3 arms reached 1.25 on a cohort 18
times smaller, so the gap is not closed by giving DPA3 the same data either.

**Flat cost in the number of angles is not novel.** It follows from predicting
the coefficients of a closed-form function of the angle rather than energies at
geometries, which every parameter predictor has done since GAFF. The 26× and
385× figures this project reported earlier were computed against potentials,
the only class where it pays.

**Not measured: whether the untrained bonds are right.** The cross-bond test,
holding out one rotor of a molecule whose other rotors are in training, **timed
out**. Five tasks × 20 h wall clock, cut at epoch 141 to 162 with validation
1.51 to 1.61 and still improving, and `finetune_torsion.py` writes its tables
only at the end, so 100 GPU-hours produced loss curves and no predictions. A
further five tasks had already died in the first three minutes: the
`finetuned_per_angle` arm OOMs at 23.5 GiB on this cohort. The fix is a periodic
checkpoint, not a longer clock.

---

## 8. What a conformer costs, and what the closed form does not

217 held-out molecules, four geometry sources through one code path.

| the head was given | median RMSE to QM | 95% CI |
|---|---|---|
| 24 QM scan geometries (privileged) | 1.324 | [1.235, 1.453] |
| one QM geometry, phase rotated in closed form | 1.341 | [1.283, 1.446] |
| an ETKDG conformer, no QM at all | 1.650 | [1.598, 1.686] |
| the QM minimum displaced by noise (control) | 55.2 | [28.9, 90.9] |
| between eight conformers of one molecule | 0.445 | [0.368, 0.548] |

Dropping 23 of the 24 QM geometries costs **0.017 kcal/mol**, so the closed-form
rotation is free and cost and accuracy can be quoted on the same input. Starting
from a generated conformer costs **0.326**, intervals not overlapping, and most
of that is which conformer was drawn rather than bias. Barrier slope 0.862
[0.799, 0.925] privileged against 0.877 [0.852, 0.904] from ETKDG: no extra
flattening, and both exclude 1, so the head under-predicts barriers by about 13%
whatever geometry it gets. `workflow/conformer_sensitivity.py`

---

## 9. Traps that cost time, written down so they cost nobody else any

**espaloma's `n4.data["k"]` is in Hartree**, not kcal/mol
(`espaloma/units.py:21`), and **every proper torsion is listed twice**,
`(a,b,c,d)` and `(d,c,b,a)`, which have identical dihedrals
(`espaloma/graphs/deploy.py:241`). Reading it as kcal/mol without
de-duplicating makes barriers 1,255× too small. The first run of §4 read
espaloma as producing 0.001 to 0.039 kcal/mol against QM's 3.3 to 9.0 and it
looked like a finding.

**THEMol's mapped SMILES write every hydrogen explicitly.** RDKit's default
merges them into implicit counts, so a 33-atom molecule comes back with 17 and
the coordinates line up with nothing. 83,942 of 83,951 scans in the first shard
failed an atom-count check that existed only because the check was written.

**THEMol's `torsion_atom_indices` is 0-based** into the coordinate array, not
the 1-based atom map numbers of the SMILES beside it.

**TorsionNet500's DLPNO key is `E[DLPNO-CCSD(T)](Ha)`**, not the `/CBS` its own
README names.

**Industry benchmark SDFs carry implicit hydrogens.** `Molecule.from_rdkit`
adds them, so a 28-atom RDKit molecule becomes 46 in OpenFF and every torsion
index points past the conformer.

**A quarter of the OpenFF pool's SMILES will not parse**, because a quaternary
nitrogen is written without its charge. Building from the archive's own
connectivity and assigning formal charges by valence recovers them; azides need
the *under*-coordinated case too, which was 100 of 870 molecules on one run.

---

## Caveats

The JS comparison to presto's Table 1 is **across cohorts and protocols**:
theirs relaxes with the force field at each angle and ours does not, and their
400 scans are not ours. Within our own cohort and one protocol, Sage's total
error is JS 0.168 and removing the phase costs 0.118, which is the like-for-like
version and is the weaker of the two statements.

All of it is three days of work by one person and has been reviewed by nobody.

## Reproducing

`tutorial/` is five conda packages and no GPU and reproduces §2 in five seconds.
`DATA.md` says where every cohort comes from. Every number above has a script
named beside it and a Snakemake rule behind that.
