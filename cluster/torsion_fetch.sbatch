#!/bin/bash
#SBATCH --job-name=torsfetch
#SBATCH --partition=all
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=4:00:00
#SBATCH --output=/work/ben/torsion_phase/logs/fetch_%A_%a.out
#
# Fetch the QCArchive torsiondrive pool, SHARDED. One array task is one shard of
# one dataset, so the 8,737-entry XtalPi set does not become a 12.8 hour serial
# task while everything else waits.
#
# QUESTION. None on its own: this is acquisition, separated from anything that
# computes so that a change to the model does not cost another pass over the
# archive.
#
# ON WHAT. Every torsiondrive dataset carrying a b3lyp-d3bj/dzvp specification,
# excluding the four "OpenFF SMIRNOFF Sage" sets, which are Sage's own fitting
# data, and TorsionNet500, which is fetched separately and held out.
#
# UNITS. One output file is one shard: geometries in Angstrom and energies in
# kcal/mol relative to each scan's own minimum.
#
# COST. Measured 5.3 s per molecule. Shards are sized at 300 molecules, so about
# 26 minutes each; 4 h limit is 9x that. Throttled so the archive sees a bounded
# number of clients.
set -u
ROOT=/work/ben/torsion_phase
PY=$HOME/miniforge3/envs/dpa3/bin/python
SHARDS=$ROOT/data/shards.tsv          # slug <tab> offset <tab> limit, one per line

read -r SLUG OFFSET LIMIT <<< "$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$SHARDS" | tr '\t' ' ')"
if [ -z "${SLUG:-}" ]; then echo "no shard for task $SLURM_ARRAY_TASK_ID"; exit 0; fi

OUT=$ROOT/data/pool/${SLUG}_${OFFSET}_${LIMIT}.npz
if [ -f "$OUT" ]; then echo "$OUT already present, skipping"; exit 0; fi
mkdir -p "$ROOT/data/pool"

# Field 2 only. `cut -f2-` took the name PLUS the spec and count columns,
# so every task asked the archive for a dataset whose name had two tab-
# separated extras glued on and all 112 failed with HTTP 400.
NAME=$(awk -F'\t' -v s="$SLUG" '$1==s{print $2; exit}' "$ROOT/data/sources.tsv")
echo "task $SLURM_ARRAY_TASK_ID: $SLUG [$OFFSET, $((OFFSET+LIMIT))) of '$NAME' on $(hostname)"

cd "$ROOT/code"
$PY pull_torsiondrives.py \
  --dataset "$NAME" \
  --offset "$OFFSET" --limit "$LIMIT" --chunk 200 \
  --out "$OUT" \
  --quarantine "$ROOT/data/pool/${SLUG}_${OFFSET}_${LIMIT}_quarantine.csv"
