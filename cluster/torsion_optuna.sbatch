#!/bin/bash
#SBATCH --job-name=torsopt
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/work/ben/torsion_phase/logs/optuna_%A_%a.out
#SBATCH --exclude=k0,k1,k2
#
# One array task is one Optuna worker. They share a journal file and each pulls
# the next trial, so there is no coordinator and adding tasks adds throughput.
#
# QUESTION. The head is 2.3x worse than the pretrained potential it is built on,
# and every hyperparameter in it was chosen by hand. Is that a property of the
# architecture or of an untuned architecture? A search that cannot beat the hand
# choice answers it one way; one that closes a meaningful part of the gap answers
# it the other.
#
# ON WHAT. Fold 0 of the 870-molecule OpenFF Gen3 cohort, the same split and seed
# as the 20 completed runs, so a trial is comparable to them. The objective is the
# best validation RMSE, and validation comes from the training portion: the
# held-out set is never touched by the search.
#
# CONTROL. The hand-chosen configuration is trial 0, enqueued before the search
# starts, so "did the search beat the hand choice" is read off the same study.
#
# UNITS. kcal/mol. One trial is one training run to convergence or 150 epochs.
#
# COST. The frozen arm is 21 s/epoch on 604 molecules, so a 150-epoch trial is
# about 50 minutes. 12 workers x 10 trials = 120 trials in about 8 hours.
set -u
ROOT=/work/ben/torsion_phase
PY=$HOME/miniforge3/envs/dpa3/bin/python
cd "$ROOT/code"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "worker $SLURM_ARRAY_TASK_ID on $(hostname), $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
$PY optuna_search.py \
  --storage "$ROOT/optuna/study.journal" \
  --study "${STUDY:-torsion_head_frozen}" \
  --worker "$SLURM_ARRAY_TASK_ID" \
  --trials "${TRIALS:-10}" \
  --arm "${ARM:-frozen}" \
  --epochs "${EPOCHS:-150}" --patience "${PATIENCE:-40}" \
  --torsiondrives "$ROOT/data/torsiondrives_gen3_888.npz" \
  --checkpoint "$ROOT/models/DPA-3.1-3M/DPA-3.1-3M.pt" \
  --config "$ROOT/models/DPA-3.1-3M/input_pretrain.json" \
  --scratch "$ROOT/optuna/trials"
