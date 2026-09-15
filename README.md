# torsion_barrier

Measuring what a molecular mechanics torsion term structurally cannot represent,
and how often it matters.

A force field torsion term is `k·cos(nφ)`. Cosine is symmetric, so **a standard
torsion term can only draw symmetric profiles**: turn a bond +30° and −30° and
it must give the same energy. Real profiles are not always symmetric. This
repository measures how much of a real torsion profile is the asymmetric kind,
which chemistry it belongs to, and what ignoring it costs.

**[FINDINGS.md](FINDINGS.md) is the complete record**: every measurement,
positive and negative, with its cohort, its number and the script behind it.
This README is the short version.

Everything here is Snakemake rules with claims registered in `context.toml`
before the data existed, adjudicated by
[claimcheck](https://github.com/BenCree/claimcheck).

## The measurements

### Asymmetry is stereochemistry, not the electronic effects everyone assumes

2,465,308 THEMol scans at B3LYP-D3(BJ)/DZVP, decomposed by least squares into
even and odd harmonics. `gain` is the kcal/mol a fitted phase buys; `JS` is the
Jensen-Shannon distance between the Boltzmann populations at 500 K with and
without the odd part, which is the statistic presto reports because RMSE "is
sensitive to barrier height errors, which do not affect equilibrium
distributions".

| cut | n | odd | gain | min shift | JS |
|---|---|---|---|---|---|
| all | 2,465,308 | 0.096 | 0.436 | 9.5° | 0.196 |
| substituents symmetry-equivalent | 38,710 | 0.011 | 0.061 | 6.0° | 0.035 |
| **stereocentre adjacent** | 945,751 | **0.266** | **0.796** | 17.5° | **0.312** |
| gauche effect | 221,994 | 0.194 | 0.655 | 15.5° | 0.307 |
| anomeric | 560,456 | 0.030 | 0.362 | 4.0° | 0.134 |
| amide | 340,991 | 0.031 | 0.375 | 4.5° | 0.141 |
| biaryl | 97,986 | 0.002 | 0.055 | 4.5° | 0.148 |
| conjugated | 988,100 | 0.018 | 0.216 | 3.5° | 0.094 |

Biaryls, amides, conjugated systems and anomeric centres are **even**. The
asymmetry sits on rotors beside a stereocentre. For scale, presto's Table 1
gives Sage a JS of 0.30, espaloma 0.20, presto 0.12 and AceFF 2.0 0.09, so the
odd part alone is worth about as much as the whole Sage-to-presto improvement.

Controls, from a smaller cohort where named molecules are present: ethane
returns 99.96% of its power at n = 3 with **zero** odd part; butane, biphenyl,
styrene, N-methylacetamide and 1,2-difluoroethane are even to four decimals;
2-butanol, which has a stereocentre beside the rotor, is 0.124 odd; and
2-methoxytetrahydropyran is 0.813.

### The standard benchmark cannot see it

Median odd magnitude is **0.0003 kcal/mol on TorsionNet500** against **0.5267**
on a drug-like OpenFF pool. TorsionNet500 is small neutral fragments with
symmetric rotors. Every number the field reports on it is blind to this.

### It is common in ligands people run FEP on

2,146 ligands of the OpenFE Industry Benchmark 2024, 11,091 rotatable bonds:
**22.8% are stereocentre-adjacent**, present in **33% of ligands**. The QM pool
has 23.3%, so the cohort is representative on the axis that matters.

### Ceilings

Best possible six-harmonic fit **0.1736 kcal/mol**; three harmonics, which is
Grappa's functional form, **0.4543**. Two DFT references for the same 497
molecules disagree by **0.767 kcal/mol**, which bounds every RMSE reported
against either.

## Negative results, stated plainly

- **A fitted phase on a frozen espaloma embedding changes nothing**: 112 of 213
  held-out molecules on RMSE, 102 of 213 on JS. A coin flip.
- **Flat cost in the number of angles is not novel.** Every parameter predictor
  since GAFF has it, because the angle enters a closed form whose coefficients
  are predicted once. Measured: espaloma plus this head 1.582 ms at both 24 and
  3,600 angles; DPA3 18.772 ms at one conformer and 450.303 ms at 24, because
  its embedding reads the geometry and a graph encoder's does not.
- **A pretrained atomistic encoder is the wrong choice here.** DPA-3.1-3M costs
  12× espaloma's stage 1 per molecule and scores no better on the same split.
- **Subtracting the MM baseline made it worse**, 1.562 against 1.368.
- **Torsional strain does not rank docked poses.** 2,146 ligands, 47,182 rows:
  Spearman between a gnina mode's strain and its RMSD to the crystal pose is
  **0.055** for both Sage and espaloma. Significant at that n, and negligible.
- **Pretrained potentials are better on expanded-element and ionic chemistry,
  not worse.** TorsionTest2000, 2,088 scans: MACE-OFF23 large 0.172, DPA3 0.304,
  MACE-OFF23 small 0.387 kcal/mol.

## Layout

```
workflow/     the Snakemake rules, the experiments, context.toml
cluster/      SLURM array jobs and their content-reading acceptance checks
nagl/         the torsion head and phase coordinate, for openff-nagl
results/      figures
DATA.md       where every cohort comes from, and the units that bite
```

## Running it

```bash
snakemake -n <target>        # always first
snakemake -c1 --resources gpu=1 <target>
```

`workflow/config.yaml` and the `cluster/*.sbatch` files carry 25 absolute paths
to interpreters, model checkpoints and staged data on the machine this ran on.
They are left as they are rather than templated, because a path that was really
used is a record and a placeholder is not. Change them for your own machine;
the list is `grep -rhoE "/(home|work)/[A-Za-z0-9_./-]+" workflow cluster`.

Nothing computes outside a rule. `workflow/context.toml` declares the datasets,
their units and the claims; `claimcheck` recomputes each claim and returns
`agrees`, `refuted` or `unverifiable`, where a claim inside a wide interval is
unverifiable rather than confirmed.

## References

Wang et al., *End-to-end differentiable construction of molecular mechanics
force fields*, Chem. Sci. 13(41):12016, 2022 (espaloma) ·
Seute et al., *Grappa*, Chem. Sci. 16:2907, 2025 ·
Clark et al., *Fast training of bespoke SMIRNOFF-format force fields using
machine learning potentials*, ChemRxiv 2026 (presto) ·
Rai et al., *TorsionNet*, JCIM 62(4):785, 2022 ·
Farr et al., *AceFF*, arXiv:2601.00581 ·
THEMol, arXiv:2605.14973
