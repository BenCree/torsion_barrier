"""Distributed hyperparameter search over the torsion head, with Optuna.

WHY, given the accuracy result is already negative. The head is 2.3x worse than
the pretrained potential it is built on, and every hyperparameter in it was
chosen by hand from the fine-tuning literature rather than searched. Before that
verdict is recorded as a property of the architecture it should be shown not to
be a property of an untuned architecture. This is the experiment that separates
those.

WHAT IS SEARCHED. Everything previously fixed by hand: the three network widths,
how many periodicities, the linear Fourier head against the flexible one, both
learning rates, weight decay, the anchor to the pretrained encoder, batch size,
gradient clipping and the EMA decay.

WHAT IS HELD. The split, the seed, the cohort, the stopping rule and the
objective. Trials differ by hyperparameters and by nothing else.

THE OBJECTIVE is the best validation RMSE on fold 0, which is the same quantity
the runs already report, so a trial is comparable to the 20 runs already done.
Validation comes from the training portion, so the held-out set is untouched by
the search; whatever wins is then run on all 5 folds before any number is quoted.
That second step is not optional. A search over 40 trials on one validation split
will find something that fits that split.

DISTRIBUTED THE WAY OPTUNA INTENDS. Workers share one journal file on /work and
each pulls the next trial; there is no coordinator. Adding workers adds
throughput and the study is resumable, which matters when a task hits its time
limit.

EACH WORKER NEEDS ITS OWN SAMPLER SEED. An earlier version gave every worker the
same one, so during the random startup phase all twelve drew the SAME point and
eleven of the first twelve trials were byte-identical duplicates. The study seed
is offset by the worker index, which keeps the run reproducible while making the
workers explore different points.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import tempfile


def objective(trial, args):
    import pandas as pd

    def widths(name, choices):
        return trial.suggest_categorical(name, choices)

    params = {
        "--torsion-hidden": widths("torsion_hidden", ["64,64", "128,128", "256,256", "128,128,128"]),
        "--coefficient-hidden": widths("coefficient_hidden", ["32", "64", "128", "64,64"]),
        "--energy-hidden": widths("energy_hidden", ["64", "128", "256", "128,128"]),
        "--periodicities": trial.suggest_int("periodicities", 3, 9),
        "--head-lr": trial.suggest_float("head_lr", 1e-4, 3e-2, log=True),
        "--encoder-lr": trial.suggest_float("encoder_lr", 1e-6, 1e-3, log=True),
        "--weight-decay": trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True),
        "--encoder-anchor": trial.suggest_float("encoder_anchor", 1e-3, 1e2, log=True),
        "--batch-size": trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128]),
        "--clip": trial.suggest_float("clip", 0.5, 20.0, log=True),
        "--ema": trial.suggest_categorical("ema", [0.0, 0.99, 0.999, 0.9999]),
    }
    nonlinear = trial.suggest_categorical("nonlinear_head", [False, True])

    workdir = pathlib.Path(tempfile.mkdtemp(prefix=f"trial{trial.number}_", dir=args.scratch))
    command = [
        sys.executable, str(pathlib.Path(__file__).parent / "finetune_torsion.py"),
        "--torsiondrives", args.torsiondrives,
        "--checkpoint", args.checkpoint, "--config", args.config, "--head", "SPICE2",
        "--rotors", "driven", "--sampler", "scaffold",
        "--folds", str(args.folds), "--fold-index", "0", "--arms", args.arm,
        "--epochs", str(args.epochs), "--patience", str(args.patience),
        "--seed", str(args.seed),
        "--profiles", str(workdir / "profiles.csv"),
        "--per-molecule", str(workdir / "per_molecule.csv"),
        "--quarantine", str(workdir / "quarantine.csv"),
        "--history", str(workdir / "loss.csv"),
    ]
    for flag, value in params.items():
        command += [flag, str(value)]
    if nonlinear:
        command.append("--nonlinear-head")

    result = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout)
    history = workdir / "loss.csv"
    if not history.exists() or len(pd.read_csv(history)) < 5:
        # A failed trial is pruned, not scored: returning a large number would
        # teach the sampler that the region is bad when it is only broken.
        print(f"trial {trial.number} produced no usable history\n{result.stderr[-600:]}",
              flush=True)
        raise __import__("optuna").TrialPruned()
    table = pd.read_csv(history)
    best = float(table.val_rmse.min())
    trial.set_user_attr("epochs_run", int(len(table)))
    trial.set_user_attr("kept_epoch", int(table.loc[table.val_rmse.idxmin(), "epoch"]))
    trial.set_user_attr("seconds", float(table.seconds.max()))
    print(f"trial {trial.number}: val RMSE {best:.4f} after {len(table)} epochs", flush=True)
    return best


def main():
    import optuna

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage", required=True, help="journal file shared by all workers")
    parser.add_argument("--study", default="torsion_head")
    parser.add_argument("--trials", type=int, default=10, help="trials THIS worker runs")
    parser.add_argument("--torsiondrives", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--arm", default="frozen")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--worker", type=int, default=0,
                        help="index of this worker; offsets the SAMPLER seed so that "
                             "parallel workers do not all draw the same startup point")
    parser.add_argument("--scratch", default="/work/ben/torsion_phase/optuna")
    parser.add_argument("--timeout", type=int, default=14400)
    args = parser.parse_args()

    pathlib.Path(args.scratch).mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.storage).parent.mkdir(parents=True, exist_ok=True)
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(args.storage)
    )
    study = optuna.create_study(
        study_name=args.study, storage=storage, direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(
            seed=args.seed + 1000 * args.worker, n_startup_trials=12
        ),
    )
    study.optimize(lambda t: objective(t, args), n_trials=args.trials,
                   catch=(Exception,), gc_after_trial=True)

    done = [t for t in study.trials if t.value is not None]
    print(f"\n{len(done)} completed trials in the study")
    if done:
        print(f"best value {study.best_value:.4f}")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
