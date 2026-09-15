"""Acceptance check for the torsion fine-tuning array.

READS THE OUTPUT'S OWN CONTENT, never a file count. A task can exit 0 having
written a well-formed loss.csv full of NaNs, or having stopped at epoch 0, or
having produced a model worse than predicting a flat profile. ERRORS #285 is the
case where every row came back failed=True while the task exited 0 and the glob
was satisfied.

Four things are asserted per run, and any failure names the run:
  the loss file exists and has more than `--min-epochs` rows
  every validation RMSE is finite
  the kept epoch is not 0, which would mean nothing was learned
  the held-out RMSE beats the flat-profile control on a majority of molecules
"""

from __future__ import annotations

import argparse
import pathlib
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/work/ben/torsion_phase/runs")
    parser.add_argument("--min-epochs", type=int, default=20)
    parser.add_argument("--expect-runs", type=int, default=15)
    args = parser.parse_args()

    import numpy as np
    import pandas as pd

    root = pathlib.Path(args.root)
    runs = sorted(p for p in root.glob("*_fold*") if p.is_dir())
    problems, ok = [], 0
    for run in runs:
        loss, per_molecule = run / "loss.csv", run / "per_molecule.csv"
        if not loss.exists():
            problems.append(f"{run.name}: no loss.csv"); continue
        history = pd.read_csv(loss)
        if len(history) <= args.min_epochs:
            problems.append(f"{run.name}: only {len(history)} epochs"); continue
        if not np.isfinite(history["val_rmse"]).all():
            problems.append(f"{run.name}: non-finite validation RMSE"); continue
        if int(history["best_epoch"].iloc[-1]) == 0:
            problems.append(f"{run.name}: kept epoch 0, nothing was learned"); continue
        if not per_molecule.exists():
            problems.append(f"{run.name}: no per_molecule.csv"); continue
        table = pd.read_csv(per_molecule)
        if table.empty or not np.isfinite(table["rmse"]).all():
            problems.append(f"{run.name}: per-molecule RMSE missing or non-finite"); continue
        beats_flat = int((table["rmse"] < table["rmse_flat"]).sum())
        if beats_flat <= len(table) / 2:
            problems.append(
                f"{run.name}: beats a flat profile on only {beats_flat} of {len(table)}"
            ); continue
        ok += 1
        print(f"  PASS {run.name}: {len(history)} epochs, best val "
              f"{history['best_val_rmse'].min():.4f}, held-out RMSE median "
              f"{table['rmse'].median():.4f}, beats flat {beats_flat}/{len(table)}")

    print(f"\n{ok} of {len(runs)} runs pass; expected {args.expect_runs}")
    for problem in problems:
        print(f"  FAIL {problem}")
    if problems or ok < args.expect_runs:
        sys.exit(1)


if __name__ == "__main__":
    main()
