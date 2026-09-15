"""One row per completed Optuna trial, so the search has a table and not a log.

QUESTION. Is 1.25 kcal/mol the architecture or the hyperparameters? The search
answers it only if the hand-chosen configuration is IN the study as a trial, so
the comparison is like for like: same split, same epoch budget, same validation
set. It is enqueued first and identified here by its parameters, not by its
trial number, since a restarted study renumbers.

ON WHAT. The journal the distributed workers write, copied from the cluster.
One row is one completed trial.

UNITS. Validation RMSE in kcal/mol at the search's epoch budget. NOT the
held-out test RMSE the arms report, so it is comparable to the control in this
table and to nothing outside it.

COST. Seconds, CPU.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

CONTROL = {
    "head_lr": 0.003, "encoder_lr": 3e-05, "weight_decay": 0.01,
    "encoder_anchor": 1.0, "batch_size": 32, "clip": 5.0, "ema": 0.999,
    "periodicities": 6, "nonlinear_head": False,
}


def is_control(params):
    return all(abs(params.get(k, None) - v) < 1e-12
               if isinstance(v, float) else params.get(k, None) == v
               for k, v in CONTROL.items())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", required=True)
    parser.add_argument("--study", default=None,
                        help="which study; default is the one with most trials")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage = JournalStorage(JournalFileBackend(args.journal))
    names = optuna.get_all_study_names(storage=storage)
    if not names:
        print(f"no study in {args.journal}")
        return 1
    if args.study:
        chosen = [args.study]
    else:
        chosen = names

    rows = []
    for name in chosen:
        study = optuna.load_study(study_name=name, storage=storage)
        for trial in study.trials:
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue
            rows.append({
                "study": name,
                "trial": trial.number,
                "val_rmse": trial.value,
                "is_control": int(is_control(trial.params)),
                **{k: v for k, v in sorted(trial.params.items())},
            })
    if not rows:
        print("no completed trial in any study")
        return 1

    fields = sorted({k for row in rows for k in row},
                    key=lambda k: (k not in ("study", "trial", "val_rmse",
                                             "is_control"), k))
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)

    for name in sorted({r["study"] for r in rows}):
        mine = [r for r in rows if r["study"] == name]
        best = min(mine, key=lambda r: r["val_rmse"])
        control = [r for r in mine if r["is_control"]]
        line = (f"{name}: {len(mine)} complete, best {best['val_rmse']:.4f} "
                f"at trial {best['trial']}")
        if control:
            reference = min(control, key=lambda r: r["val_rmse"])["val_rmse"]
            line += (f", control {reference:.4f}, "
                     f"gain {reference - best['val_rmse']:+.4f} "
                     f"({(reference - best['val_rmse'])/0.767:.2f} of the "
                     f"label uncertainty)")
        else:
            line += ", NO CONTROL TRIAL FOUND in this study"
        print(line)
    print(f"  -> {out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
