#!/bin/bash
#SBATCH --job-name=themolcen
#SBATCH --partition=all
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=6:00:00
#SBATCH --output=/work/ben/torsion_phase/logs/census_%A_%a.out
#SBATCH --exclude=g0,g1
#
# THE PHASE CENSUS ON THEMol. One array task is one shard.
#
# QUESTION. Turn one bond and plot energy against angle: how much of that plot
# is the ASYMMETRIC kind, the kind a force field's k cos(n phi) cannot draw
# because cosine is symmetric? And whose chemistry is it?
#
# This is least squares on QM energies. No model, no training, no GPU. The
# number it produces is a CEILING on what any phase-fitting torsion form can
# gain over any fixed-phase one, so a small answer closes the question for every
# such model at once and a large one says which chemistry is out of reach.
#
# ON WHAT. ByteDance-Seed/THEMol TorsionScan, CC-BY-NC-4.0: 4,192,791 scans over
# 2,436,985 molecules, 93,994,576 constrained-optimisation geometries at
# B3LYP-D3(BJ)/DZVP, twelve elements, up to 50 heavy atoms, in-ring and non-ring.
# The SAME level of theory as this project's own pool, so the cohort grows 275x
# without a change of reference, which matters because a change of reference
# reversed the MACE/DPA3 ranking between TorsionNet500 and TorsionTest2000.
#
# WHY IT IS WORTH 4 MILLION. The claim it tests, that torsional asymmetry is
# stereochemistry and not the electronic effects everyone assumes, rests on class
# contrasts that today have 443 amide scans and 796 gauche-effect ones. Here they
# are of order a hundred thousand each. And the criticism it supports, that
# TorsionNet500 contains none of this effect, currently reads 497 scans against
# 15,289.
#
# CONTROLS. Named molecules with textbook answers travel in every task and are
# reported whether found or not: ethane must come back with no asymmetry at all,
# biphenyl and N-methylacetamide even, a rotor beside a stereocentre odd. A
# census without those cannot tell a real signal from a broken fit.
#
# UNITS. kcal/mol for residuals, degrees for how far the minimum moves when the
# asymmetric part is removed, and Jensen-Shannon distance in bits for how wrong
# the conformer populations become, which is comparable to Table 1 of Clark et
# al., ChemRxiv 2026, where Sage is 0.30 and presto 0.12.
#
# COST. CPU only, no --gres, so nothing on a GPU is displaced. About 71,000
# usable scans per shard after the twenty-point filter, a few milliseconds each.
#
# Nothing is set -e: a shard that will not read must not end the census.

set -uo pipefail
ROOT=/work/ben/torsion_phase
SHARD=/work/ben/themol/TorsionScan/torsion_${SLURM_ARRAY_TASK_ID}.h5
OUT=$ROOT/census
mkdir -p "$OUT" "$ROOT/logs"

if [ ! -f "$SHARD" ]; then
  echo "shard $SHARD is missing; nothing to do"
  exit 0
fi

export NAGL_TORSION_ROOT=$ROOT/code_perbond/nagl
cd "$ROOT"
echo "task $SLURM_ARRAY_TASK_ID -> $(basename $SHARD) on $(hostname)"
/home/ben/miniforge3/envs/dpa3/bin/python bin/phase_census.py \
  --torsiondrives "$SHARD" \
  --format themol \
  --min-points 20 \
  --out "$OUT/census_${SLURM_ARRAY_TASK_ID}.csv" \
  --named "$OUT/named_${SLURM_ARRAY_TASK_ID}.csv" \
  --quarantine "$OUT/quarantine_${SLURM_ARRAY_TASK_ID}.csv"
echo "exit $?"
