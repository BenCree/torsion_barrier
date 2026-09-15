#!/bin/bash
#SBATCH --job-name=torsxb
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --output=/work/ben/torsion_phase/logs/xb_%A_%a.out
#SBATCH --exclude=k0,k1,k2
#
# THE CROSS-BOND TEST. One array task is one (fold, arm).
#
# QUESTION. The head emits a profile for EVERY rotatable bond of a molecule from
# one encoder pass, which is where its 26x throughput advantage comes from. It
# was only ever trained on the bond a torsiondrive turned. Are the other bonds
# right? If holding out one bond of a molecule whose other bonds are in training
# scores near the driven-bond result, the all-bonds output is trustworthy and the
# throughput is real. If it degrades toward predicting the molecule's other bond
# verbatim, the model has learned one profile per molecule and the extra bonds
# are repetition.
#
# ON WHAT. 15,677 (molecule, driven bond) scans at B3LYP-D3BJ/DZVP, of which
# 3,677 belong to the 1,167 molecules that separate OpenFF datasets happened to
# scan at two or more distinct rotors. Only those 3,677 are holdable; the rest
# stay in training because they cannot answer the question.
#
# THE SPLIT IS OVER BONDS, NOT MOLECULES, which is the whole point: the held-out
# molecule is IN training, with its scaffold, elements and one of its own rotors
# seen. Only that bond is unseen.
#
# CONTROLS, computed at analysis time on the same rows: the molecule's other
# trained bond used as the prediction (if that scores as well, nothing
# bond-specific was learned), a random other molecule's profile, and the
# pretrained potentials, which have no notion of a driven bond and pay no penalty.
#
# UNITS. kcal/mol relative to each scan's own minimum. One row is one held-out
# (molecule, bond) scan.
#
# COST. 12,500 training scans against the 604 of the 888 run, so about 20x the
# per-epoch cost; 20 h limit, early stopping expected well inside it.
set -u
ROOT=/work/ben/torsion_phase
PY=$HOME/miniforge3/envs/dpa3/bin/python
ARMS=(frozen finetuned_per_angle)
FOLD=$(( SLURM_ARRAY_TASK_ID / 2 ))
ARM=${ARMS[$(( SLURM_ARRAY_TASK_ID % 2 ))]}
OUT=$ROOT/crossbond/${ARM}_fold${FOLD}
mkdir -p "$OUT"
echo "task $SLURM_ARRAY_TASK_ID -> arm $ARM fold $FOLD on $(hostname)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT/code"
$PY finetune_torsion.py \
  --torsiondrives "$ROOT/data/pool_scans.npz" \
  --checkpoint "$ROOT/models/DPA-3.1-3M/DPA-3.1-3M.pt" \
  --config "$ROOT/models/DPA-3.1-3M/input_pretrain.json" \
  --head SPICE2 --rotors driven \
  --split-mode cross_bond --folds 5 --fold-index "$FOLD" --arms "$ARM" \
  --epochs 250 --patience 50 \
  --head-lr 0.003 --encoder-lr 3e-5 --ema 0.999 --seed 20260911 \
  --profiles "$OUT/profiles.csv" \
  --per-molecule "$OUT/per_molecule.csv" \
  --quarantine "$OUT/quarantine.csv" \
  --history "$OUT/loss.csv"
