#!/bin/bash
#SBATCH --job-name=torsfit
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/work/ben/torsion_phase/logs/%A_%a.out
#SBATCH --exclude=k0,k1,k2
#
# One array task is one (fold, arm), so 5 folds x 3 arms fill 15 cards at once.
# k0-k2 are excluded because Ben's fepab array holds them.
#
# QUESTION. Does fine-tuning a DPA3 encoder on QM torsion data, with a
# permutation-invariant phase and an energy linear in it, beat a frozen encoder,
# and does the encoder have to see the rotated geometries for it to? The paired
# per-molecule comparison over all 888 held-out molecules answers both.
#
# ON WHAT. OpenFF Gen3, 888 torsiondrives at B3LYP-D3BJ/DZVP in Psi4, 24 points
# per scan. 5 scaffold-grouped folds over MOLECULES, so every molecule is held
# out exactly once per arm and the paired test has n = 888.
#
# CONTROLS. MMFF94 and the flat profile, computed for every molecule and carried
# in the same table.
#
# UNITS. kcal/mol relative to each scan's own minimum. One row of the
# per-molecule table is one molecule in one arm.
#
# COST. Measured locally on an RTX 4000 Ada and scaled: frozen 0.6 h, fine-tuned
# reference 1.3 h, per-angle 3.9 h per fold. 12 h limit is 3x the longest.
set -u
ROOT=/work/ben/torsion_phase
PY=$HOME/miniforge3/envs/dpa3/bin/python
ARMS=(frozen finetuned_reference finetuned_per_angle frozen_per_angle)

FOLD=$(( SLURM_ARRAY_TASK_ID / 4 ))
ARM=${ARMS[$(( SLURM_ARRAY_TASK_ID % 4 ))]}
OUT=$ROOT/runs/${ARM}_fold${FOLD}
mkdir -p "$OUT"

echo "task $SLURM_ARRAY_TASK_ID -> arm $ARM fold $FOLD on $(hostname), $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT/code"
$PY finetune_torsion.py \
  --torsiondrives "$ROOT/data/torsiondrives_gen3_888.npz" \
  --checkpoint "$ROOT/models/DPA-3.1-3M/DPA-3.1-3M.pt" \
  --config "$ROOT/models/DPA-3.1-3M/input_pretrain.json" \
  --head SPICE2 \
  --rotors driven \
  --folds 5 --fold-index "$FOLD" --arms "$ARM" \
  --epochs 400 --patience 60 \
  --head-lr 0.003 --encoder-lr 3e-5 --ema 0.999 \
  --sampler scaffold --seed 20260911 \
  --profiles "$OUT/profiles.csv" \
  --per-molecule "$OUT/per_molecule.csv" \
  --quarantine "$OUT/quarantine.csv" \
  --history "$OUT/loss.csv"
