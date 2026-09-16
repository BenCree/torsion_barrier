"""Did the TorsionNet500 benchmark run produce a comparable number, or only files?

THE CHECK READS CONTENT, NEVER A FILE COUNT. A task can exit 0 having written a
well-formed per-molecule table with no external cohort in it at all, which is
exactly what would happen if `--external-test` were dropped from the command
line or the cache path were wrong. Four things are therefore read out of the
table itself:

  1. the external cohort is PRESENT and holds enough molecules to resolve
     anything. 497 minus the 50 InChIKey overlaps is about 447; far fewer means
     the exclusion matched more than it should have.
  2. the pool cohort REPRODUCES the run this is compared against. Job 79460 gave
     a median pool RMSE of 1.166 frozen and 1.207 fine-tuned under this seed. If
     the retrain does not land there, the two cohorts are not being compared
     under one model and no row from it is quotable.
  3. no external molecule shares an InChIKey with the pool. The run drops them,
     so a survivor means the exclusion did not work.
  4. the predictions are finite and not constant. A head that emits one number
     everywhere scores a defined RMSE and a JS of whatever the QM happens to be,
     and neither is a prediction.
"""

from __future__ import annotations

import argparse
import pathlib
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", required=True,
                        help="comma separated; every one must pass")
    parser.add_argument("--root", default="/work/ben/torsion_phase/tn500_benchmark")
    parser.add_argument("--min-external", type=int, default=400)
    parser.add_argument("--expect-pool-rmse", default="frozen=1.166,finetuned_reference=1.207",
                        help="arm=median pairs: what each reached under this seed "
                             "in job 79460, which the retrain must reproduce")
    parser.add_argument("--tolerance", type=float, default=0.05)
    args = parser.parse_args()

    import numpy as np
    import pandas as pd

    expected = {}
    for item in args.expect_pool_rmse.split(","):
        if "=" in item:
            name, _, value = item.partition("=")
            expected[name.strip()] = float(value)

    failures = 0
    for arm in (a.strip() for a in args.arms.split(",")):
        failures += check_one(arm, args, expected.get(arm), np, pd)
    return 1 if failures else 0


def check_one(arm, args, expect_pool_rmse, np, pd):
    import pathlib
    path = pathlib.Path(args.root) / arm / "per_molecule.csv"
    if not path.exists():
        print(f"FAIL {arm}: {path} does not exist")
        return 1
    table = pd.read_csv(path)

    if "cohort" not in table.columns:
        print(f"FAIL {arm}: no `cohort` column, so the external set was "
              f"never evaluated. --external-test did not reach the run.")
        return 1

    external = table[table.cohort == "external"]
    pool = table[table.cohort == "pool"]
    print(f"{arm}: {len(pool)} pool rows, {len(external)} external rows")

    problems = []
    if len(external) < args.min_external:
        problems.append(f"only {len(external)} external molecules, "
                        f"expected at least {args.min_external}")

    shared = set(external.inchi_key.dropna()) & set(pool.inchi_key.dropna())
    if shared:
        problems.append(f"{len(shared)} external molecules share an InChIKey "
                        f"with the pool: the exclusion did not work")

    for name, block in (("pool", pool), ("external", external)):
        if block.empty:
            continue
        finite = np.isfinite(block.rmse).mean()
        print(f"  {name}: median RMSE {block.rmse.median():.3f} kcal/mol, "
              f"{100 * finite:.1f}% finite, "
              f"median rho {block.rho.median():+.3f}")
        if finite < 0.95:
            problems.append(f"{name}: only {100 * finite:.1f}% of RMSE finite")
        # A constant predictor has no rank correlation with anything, so an
        # all-NaN rho column means the head emitted one number everywhere.
        if block.rho.notna().mean() < 0.5:
            problems.append(f"{name}: rho undefined for "
                            f"{100 * block.rho.isna().mean():.0f}% of molecules, "
                            f"which is what a constant prediction looks like")

    if expect_pool_rmse is not None and not pool.empty:
        got = float(pool.rmse.median())
        if abs(got - expect_pool_rmse) > args.tolerance:
            problems.append(
                f"pool median RMSE {got:.3f} is more than {args.tolerance} from "
                f"the {expect_pool_rmse:.3f} this seed reached before, so "
                f"the retrain did not reproduce the model being compared")

    if problems:
        for problem in problems:
            print(f"FAIL {arm}: {problem}")
        return 1
    print(f"PASS {arm}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
