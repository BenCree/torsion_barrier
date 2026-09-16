#!/bin/bash
#SBATCH --job-name=tn500bench
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --output=/work/ben/torsion_phase/logs/tn500bench_%A_%a.out
#SBATCH --exclude=g0,g1
#
# THE FULL-POOL HEAD ON TorsionNet500. One array task is one arm.
#
# QUESTION. Our head's Jensen-Shannon distance on its own held-out pool is
# 0.281 [0.275, 0.289] at 500 K, which sits between Sage's 0.30 and espaloma's
# 0.20 in presto's Table 1. That comparison is across cohorts, so it is worth
# little. TorsionNet500 is the cohort presto, espaloma, Grappa and TorsionNet
# all validate on. Does the same trained model, evaluated there, produce a
# number that can be put in the same column?
#
# ON WHAT. data/test/torsionnet500.npz, 497 scans at 24 points over 345
# degrees. 50 of the 497 share an InChIKey with the training pool and are
# dropped by the run itself, leaving about 447. One independent unit is a
# molecule. Trained on pool_scans.npz, 15,677 scans over 13,167 molecules,
# exactly as job 79460 was: same seed, same hyperparameters, same split.
#
# WHY NOT REUSE JOB 79460. It saved no weights, so the model that produced
# 1.166 cannot be reloaded. Retraining under the same seed reproduces it, and
# the pool test set comes back in the same table, which is the check that the
# retrain matched: if its pool median is not 1.166 and 1.207, the two cohorts
# are not being compared under one model and nothing here is quotable.
#
# WHAT THIS STILL IS NOT. presto relaxes with the force field at every angle
# and this does not. Matching the cohort removes one of the two caveats on that
# comparison, not both, and the reported row says so.
#
# CONTROLS. MMFF94 and the flat profile travel in the same table on both
# cohorts, so a head that improves still has to beat a 1996 force field and
# doing nothing, on the benchmark as well as at home.
#
# UNITS. kcal/mol relative to each scan's own minimum. JS is dimensionless.
#
# COST. Two GPU tasks in parallel. 2.7 h frozen and 4.4 h fine-tuned when this
# last ran; predicting 447 extra scans after training is seconds.
#
# Not set -e: one arm failing must not take the other with it.

set -uo pipefail
ROOT=/work/ben/torsion_phase
PY=$HOME/miniforge3/envs/espaloma/bin/python
ARMS=(frozen finetuned_reference)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}
OUT=$ROOT/tn500_benchmark/$ARM
mkdir -p "$OUT"
echo "task $SLURM_ARRAY_TASK_ID -> arm $ARM on $(hostname)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NAGL_TORSION_ROOT=$ROOT/code_perbond/nagl
export SSL_CERT_FILE=$($PY -c "import certifi; print(certifi.where())" 2>/dev/null || true)
cd "$ROOT/code_perbond"
$PY finetune_torsion.py \
  --torsiondrives "$ROOT/data/pool_scans.npz" \
  --external-test "$ROOT/data/torsionnet500.npz" \
  --encoder espaloma \
  --espaloma-version 0.3.2 \
  --arms "$ARM" \
  --rotors driven \
  --epochs 400 --patience 60 \
  --head-lr 0.003 --encoder-lr 3e-5 --ema 0.999 --seed 20260911 \
  --sampler scaffold --split-by-identity \
  --profiles "$OUT/profiles.csv" \
  --per-molecule "$OUT/per_molecule.csv" \
  --quarantine "$OUT/quarantine.csv" \
  --history "$OUT/loss.csv"
echo "exit $?"
