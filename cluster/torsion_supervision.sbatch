#!/bin/bash
#SBATCH --job-name=torssup
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --output=/work/ben/torsion_phase/logs/sup_%A_%a.out
#SBATCH --exclude=g0,g1
#
# SCAN SUPERVISION AGAINST BOND SUPERVISION. One array task is one mode.
#
# QUESTION. Each torsiondrive currently embeds its OWN reference geometry, so a
# molecule scanned at bond A and at bond B produces two different embeddings and
# therefore two different representations of the SAME bond, with nothing in the
# loss saying they should agree. k_n is documented as a property of the
# chemistry. Does one embedding per molecule, serving every bond that molecule
# was scanned at, predict held-out profiles better than one embedding per scan?
#
# The loss is per bond in BOTH modes, and that is not what changes. h_bond and s
# are each scatter_add by central bond, so the driven column is bit-identical
# whether one rotor is enumerated or four: measured at 1.4e-17, a tenth of one
# float64 eps, in test_per_bond_supervision.py. What changes is the geometry the
# representation is built from, and the divisor, which becomes the number of
# bonds in the batch rather than the number of molecules.
#
# ON WHAT. The merged pool, 15,659 (molecule, driven bond) scans at
# B3LYP-D3BJ/DZVP, of which 3,669 belong to the 1,164 molecules that separate
# OpenFF datasets happened to scan at two or more distinct rotors. One
# independent unit is a MOLECULE; one reported row is one scan.
#
# THE EPOCH BUDGET IS DELIBERATELY SHORT. 120 epochs at about 450 s each is 15 h,
# inside the 20 h limit; the comparable G5 run peaked at epoch 99. Both arms are
# therefore equally under-trained, which a paired comparison tolerates and an
# absolute number does not. Do not quote either arm's RMSE against the 1.254 of
# RESULT_stage3.md: quote the paired difference.
#
# THE SPLIT IS HASHED ON EACH MOLECULE'S InChIKey, not on its position, because
# the two arms prepare different numbers of records: 15,659 scans against 13,424
# grouped ones. A permuted list at one seed splits those differently, and the
# first submission of this job did exactly that, holding out 3,915 molecules in
# one arm and 3,356 in the other with nothing making them the same molecules. It
# was cancelled at nine minutes. The cost of that split is that it is random
# rather than scaffold-extrapolative, and astartes falls back to random on this
# cohort anyway: about a quarter of the pool's SMILES will not parse because a
# quaternary nitrogen is written without its charge.
#
# CONTROLS. MMFF94 and the flat profile travel in both tables and are joined
# through by compare_supervision.py, so an arm beating the other still has to
# beat them. The registered claims are S1 (a refutation says the mode is not a
# no-op) and K4 (a known answer that says the join is intact), both written into
# context.toml on 2026-09-13 before either file existed.
#
# UNITS. kcal/mol relative to each scan's own minimum.
#
# COST. Two GPU tasks, up to 20 h each, on the two idle RTX 4000 Ada nodes so
# nothing already running is displaced. The frozen arm only: the per-angle arm
# OOMs at 23.5 GiB on this cohort, which is what killed five of ten tasks of
# job 79351.

set -uo pipefail
ROOT=/work/ben/torsion_phase
PY=$HOME/miniforge3/envs/dpa3/bin/python
MODES=(scan bond)
MODE=${MODES[$SLURM_ARRAY_TASK_ID]}
ROTORS=$( [ "$MODE" = bond ] && echo all || echo driven )
OUT=$ROOT/supervision/$MODE
mkdir -p "$OUT"
echo "task $SLURM_ARRAY_TASK_ID -> supervision $MODE, rotors $ROTORS on $(hostname)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NAGL_TORSION_ROOT=$ROOT/code_perbond/nagl
cd "$ROOT/code_perbond"
$PY finetune_torsion.py \
  --torsiondrives "$ROOT/data/pool_scans.npz" \
  --checkpoint "$ROOT/models/DPA-3.1-3M/DPA-3.1-3M.pt" \
  --config "$ROOT/models/DPA-3.1-3M/input_pretrain.json" \
  --head SPICE2 \
  --supervision "$MODE" --rotors "$ROTORS" --arms frozen \
  --split-by-identity \
  --epochs 120 --patience 40 \
  --head-lr 0.003 --encoder-lr 3e-5 --ema 0.999 --seed 20260911 \
  --profiles "$OUT/profiles.csv" \
  --per-molecule "$OUT/per_molecule.csv" \
  --quarantine "$OUT/quarantine.csv" \
  --history "$OUT/loss.csv"
echo "exit $?"
