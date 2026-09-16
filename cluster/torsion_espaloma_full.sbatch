#!/bin/bash
#SBATCH --job-name=espfull
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --output=/work/ben/torsion_phase/logs/espfull_%A_%a.out
#SBATCH --exclude=g0,g1
#
# THE ESPALOMA HEAD ON THE FULL POOL. One array task is one arm.
#
# QUESTION. Every espaloma arm so far trained on 855 molecules of the Gen3
# cohort and reached 1.595 frozen and 1.368 with the encoder fine-tuned. Does a
# cohort 18 times larger change either number, and does fine-tuning still help
# at that size? The scaling exponent measured earlier on this project was 0.047
# from 60 to 604 molecules, which predicts almost no gain; this is the test of
# that on a different encoder.
#
# ON WHAT. pool_scans.npz: 15,677 (molecule, driven bond) scans over 13,167
# molecules at B3LYP-D3BJ/DZVP, driven bond only, astartes scaffold split with
# a fallback to random that the run prints rather than hides. One independent
# unit is a molecule.
#
# WHY THE PER-BOND HEAD AND NOT THE PER-TORSION ONE. The two readouts disagreed
# about fine-tuning on the small cohort: per-bond gained 1.595 to 1.368 while
# per-torsion did not move, 1.220 to 1.288 unphased. This runs the one that
# gained, so the size question is asked where there is something to lose.
#
# CONTROLS. MMFF94 and the flat profile travel in the same table, so an arm that
# improves still has to beat a 1996 force field and doing nothing.
#
# UNITS. kcal/mol relative to each scan's own minimum. Training seconds measured.
#
# COST. Two GPU tasks in parallel on an idle cluster. espaloma stage 1 is 1.4 ms
# per molecule, so an epoch over about 11,700 training scans is minutes.
#
# Not set -e: one arm failing must not take the other with it.

set -uo pipefail
ROOT=/work/ben/torsion_phase
PY=$HOME/miniforge3/envs/espaloma/bin/python
ARMS=(frozen finetuned_reference)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}
OUT=$ROOT/espaloma_full/$ARM
mkdir -p "$OUT"
echo "task $SLURM_ARRAY_TASK_ID -> arm $ARM on $(hostname)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NAGL_TORSION_ROOT=$ROOT/code_perbond/nagl
# The checkpoint is pre-staged in ~/.cache/torch/hub/checkpoints so no compute
# node has to reach GitHub. If it ever does, thira's Python carries no CA bundle
# and fails with CERTIFICATE_VERIFY_FAILED, so certifi's is named explicitly.
export SSL_CERT_FILE=$($PY -c "import certifi; print(certifi.where())" 2>/dev/null || true)
cd "$ROOT/code_perbond"
$PY finetune_torsion.py \
  --torsiondrives "$ROOT/data/pool_scans.npz" \
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
