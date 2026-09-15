#!/bin/bash
#SBATCH --job-name=themolimp
#SBATCH --output=/work/ben/torsion_phase/logs/themol_%A_%a.out
#SBATCH --error=/work/ben/torsion_phase/logs/themol_%A_%a.out
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=8:00:00
# One task per cohort size of G4: does more data move the error?
# CPU only. No --gres, so these never displace a GPU job.
# Nothing is set -e: a shard that will not read must not end the cohort.

set -uo pipefail
ROOT=/work/ben/torsion_phase
SHARDS=/work/ben/themol/TorsionScan
PY=/home/ben/miniforge3/envs/dpa3/bin/python

SIZES=(15000 150000 1500000)
NSHARDS=(1 3 25)
SIZE=${SIZES[$SLURM_ARRAY_TASK_ID]}
N=${NSHARDS[$SLURM_ARRAY_TASK_ID]}

FILES=""
for ((k=0; k<N; k++)); do
  f="$SHARDS/torsion_${k}.h5"
  [ -f "$f" ] && FILES="$FILES $f"
done

mkdir -p "$ROOT/data/themol" "$ROOT/logs"
echo "cohort $SIZE from $N shards:$FILES"
$PY "$ROOT/bin/import_themol.py" \
  --shards $FILES \
  --limit "$SIZE" \
  --out "$ROOT/data/themol/themol_${SIZE}.npz" \
  --quarantine "$ROOT/logs/themol_${SIZE}_quarantine.csv"
echo "exit $?"
