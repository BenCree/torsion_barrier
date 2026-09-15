"""Does giving a per-torsion readout a fitted phase help, all else identical?

QUESTION. espaloma emits one number per periodicity for each proper torsion,
K_n, with the phase pinned: "we do not fit phases and periodicities of torsions
as they are discrete". Its energy is therefore sum_t sum_n K_nt cos(n phi_t).
The phase census measured what that costs on 15,289 QM scans: removing the
asymmetric part of a profile shifts the minimum by 9.5 degrees at the median and
24 degrees on rotors beside a stereocentre, and moves the Boltzmann populations
by a Jensen-Shannon distance of 0.118 overall and 0.292 there, against Sage's
published 0.30 and presto's 0.12. And espaloma's own trained driven-bond term
reproduces only 0.220 of the QM odd magnitude at +0.059 direction similarity,
so it does not use the freedom its form technically has.

WHAT WOULD ANSWER IT EITHER WAY. Two arms differing in one thing: how many
numbers the readout emits per torsion per periodicity.

  unphased   one, a_n, giving sum_n a_n cos(n phi). espaloma's functional form
             exactly, reimplemented here so both arms share every other choice.
  phased     two, (a_n, b_n), giving sum_n [a_n cos(n phi) + b_n sin(n phi)],
             which is an amplitude and a phase.

Same encoder, same Janossy pooling, same data, same split, same seed, same
optimiser, same epochs, same loss. Comparing against espaloma AS PUBLISHED would
confound the phase with a different training set, a different loss and a
different everything; this does not.

THIS IS PER TORSION, NOT PER BOND, which is the difference from `model.py`'s
head and is deliberate: espaloma and Grappa parameterise the torsion, so an arm
that parameterises the bond would be testing two changes at once.

THE ENCODER CAN BE FINE-TUNED, and there is a reason to. espaloma's embedding
was trained under a readout that CANNOT express asymmetry, so it may never have
needed to encode it: a representation optimised against a fixed-phase target has
no gradient pushing it to retain the odd component. Freezing the encoder asks
whether that information survived anyway. Fine-tuning asks whether it is
recoverable. Those are different questions, and the four cells of
{frozen, fine-tuned} x {unphased, phased} separate them: the phase helping only
when the encoder moves would say the representation is the constraint, not the
readout.

THE LOSS CAN INCLUDE BOLTZMANN OVERLAP. espaloma trains on squared error, and
the census showed that is the wrong objective for this: removing the phase costs
0.292 kcal/mol of RMSE and a JS distance of 0.118, and presto's own paper says
the RMSE "is sensitive to barrier height errors, which do not affect equilibrium
distributions". `--js-weight` adds a differentiable Jensen-Shannon term at 500 K.
It defaults to 0 so the control is plain squared error and the phase is tested
on its own before the objective is also changed.

ON WHAT. The cached torsiondrives, driven bond only, split by scaffold. One
independent unit is a molecule.

CONTROLS. The flat profile on the same rows, and the unphased arm, which is the
whole point.

UNITS. kcal/mol for RMSE, degrees for how far the predicted minimum sits from
the QM one, and Jensen-Shannon distance in bits.

COST. espaloma's stage 1 is 1.4 ms per molecule on a card and the head is tiny,
so an epoch on 600 molecules is about a second.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import time

import numpy as np

BETA_500K = 1.0 / (0.0019872041 * 500.0)


class PerTorsionReadout:
    """Built in main so torch is imported once; see `build_readout`."""


def build_readout(n_features, periodicities, phased, hidden, torch):
    """Janossy pooling over the four atoms, then a_n or (a_n, b_n) per torsion."""
    nn = torch.nn

    def mlp(sizes):
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(nn.SiLU())
        return nn.Sequential(*layers)

    widths = [int(x) for x in str(hidden).split(",") if x.strip()]
    outputs = (2 if phased else 1) * periodicities

    class Readout(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = mlp([4 * n_features, *widths])
            self.out = mlp([widths[-1], widths[-1], outputs])
            self.periodicities = periodicities
            self.phased = phased

        def forward(self, h, indices, deltas):
            """h (n_atoms, F), indices (T, 4), deltas (P, T) -> (P,) energies.

            Reversing a torsion must not change it, so the pooling network is
            summed over both atom orderings rather than trained to ignore the
            difference. That is Janossy pooling and it is what espaloma does.
            """
            forward = h[indices].reshape(len(indices), -1)
            reverse = h[indices.flip(-1)].reshape(len(indices), -1)
            k = self.out(self.pool(forward) + self.pool(reverse))
            n = torch.arange(1, self.periodicities + 1, dtype=deltas.dtype,
                             device=deltas.device)
            angles = deltas.unsqueeze(-1) * n              # (P, T, N)
            if self.phased:
                a = k[:, :self.periodicities].unsqueeze(0)
                b = k[:, self.periodicities:].unsqueeze(0)
                terms = a * torch.cos(angles) + b * torch.sin(angles)
            else:
                terms = k.unsqueeze(0) * torch.cos(angles)
            energies = terms.sum(dim=(1, 2))
            return energies - energies.min()

    return Readout()


def js_loss(predicted, truth, torch, beta=BETA_500K):
    """Jensen-Shannon divergence between the two Boltzmann distributions.

    Differentiable, and it is the quantity presto's Table 1 reports because a
    squared error on energies is dominated by barrier heights that no simulation
    ever climbs.
    """
    p = torch.softmax(-beta * predicted, dim=0)
    q = torch.softmax(-beta * truth, dim=0)
    m = 0.5 * (p + q)
    kl = lambda x, y: (x * (torch.log(x + 1e-30) - torch.log(y + 1e-30))).sum()
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def evaluate(predicted, truth, grid):
    """RMSE, how far the minimum moved in degrees, and the JS distance."""
    predicted = np.asarray(predicted, dtype=float)
    predicted = predicted - predicted.min()
    truth = np.asarray(truth, dtype=float)
    truth = truth - truth.min()
    grid = np.asarray(grid, dtype=float)
    shift = float(grid[int(np.argmin(predicted))] - grid[int(np.argmin(truth))])

    def weights(curve):
        w = np.exp(-BETA_500K * np.clip(curve, 0.0, 700.0))
        return w / w.sum()

    p, q = weights(predicted), weights(truth)
    m = 0.5 * (p + q)
    kl = lambda x, y: float((x[(x > 0) & (y > 0)]
                             * np.log2(x[(x > 0) & (y > 0)] / y[(x > 0) & (y > 0)])).sum())
    return {
        "rmse": float(np.sqrt(np.mean((predicted - truth) ** 2))),
        "minimum_shift_degrees": abs((shift + 180.0) % 360.0 - 180.0),
        "js_distance": float(np.sqrt(max(0.5 * kl(p, m) + 0.5 * kl(q, m), 0.0))),
    }


def run_arm(phased, train, validation, held_out, args, torch, encoder_parts):
    build_encoder, _graph_of, embed_graph = encoder_parts
    from finetune_torsion import DEVICE

    device = DEVICE()
    encoder, n_features = build_encoder(args.espaloma_version)
    encoder = encoder.to(device)
    finetune = bool(args.finetune_encoder)
    encoder.train(finetune)
    for parameter in encoder.parameters():
        parameter.requires_grad_(finetune)
    # Detached clones, so the anchor is a fixed reference and not a second set
    # of parameters drifting alongside the first. Weight decay would pull the
    # encoder toward zero, which is not where a pretrained representation
    # should be pulled; this pulls it toward where it started.
    anchor = ({n: p.detach().clone() for n, p in encoder.named_parameters()}
              if finetune and args.encoder_anchor > 0 else None)
    for molecule in list(train) + list(validation) + list(held_out):
        if molecule["graph"].device != device:
            molecule["graph"] = molecule["graph"].to(device)

    torch.manual_seed(args.seed)
    head = build_readout(n_features, args.periodicities, phased,
                         args.hidden, torch).double().to(device)
    groups = [{"params": head.parameters(), "lr": args.lr,
               "weight_decay": args.weight_decay}]
    if finetune:
        groups.append({"params": encoder.parameters(),
                       "lr": args.encoder_lr, "weight_decay": 0.0})
    optimiser = torch.optim.AdamW(groups)
    schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=max(args.patience // 4, 5),
        min_lr=1e-7)

    def predict(molecule, grad):
        h = embed_graph(encoder, molecule["graph"], 1,
                        bool(grad and finetune))[0].double()
        return head(h, molecule["indices"], molecule["deltas"])

    def loss_of(molecule):
        predicted = predict(molecule, True)
        truth = molecule["qm_t"]
        loss = ((predicted - truth) ** 2).mean()
        if args.js_weight:
            loss = loss + args.js_weight * js_loss(predicted, truth, torch)
        return loss

    started, best = time.time(), {"rmse": float("inf"), "epoch": -1, "state": None}
    since = 0
    order_rng = np.random.default_rng(args.seed)
    batch_size = max(8, min(512, int(round(0.04 * len(train)))))
    for epoch in range(args.epochs):
        head.train()
        order = order_rng.permutation(len(train))
        total = 0.0
        for start in range(0, len(order), batch_size):
            batch = [train[i] for i in order[start:start + batch_size]]
            optimiser.zero_grad(set_to_none=True)
            for molecule in batch:
                loss = loss_of(molecule) / len(batch)
                loss.backward()
                total += float(loss.detach())
            if anchor is not None:
                # Scaled by the batch fraction so the anchor's weight per EPOCH
                # does not depend on how many batches an epoch happens to have.
                penalty = args.encoder_anchor * (len(batch) / len(train)) * sum(
                    ((p - anchor[n]) ** 2).sum()
                    for n, p in encoder.named_parameters())
                penalty.backward()
            parameters = [p for g in groups for p in g["params"]]
            torch.nn.utils.clip_grad_norm_(parameters, args.clip)
            optimiser.step()
        head.eval()
        encoder.eval()
        with torch.no_grad():
            current = float(np.mean([
                float(torch.sqrt(((predict(m, False) - m["qm_t"]) ** 2).mean()))
                for m in validation]))
        schedule.step(current)
        encoder.train(finetune)
        if current < best["rmse"] - 1e-6:
            best = {"rmse": current, "epoch": epoch,
                    "state": {k: v.detach().clone()
                              for k, v in head.state_dict().items()},
                    "encoder": ({k: v.detach().clone()
                                 for k, v in encoder.state_dict().items()}
                                if finetune else None)}
            since = 0
        else:
            since += 1
        if epoch % 25 == 0 or epoch == args.epochs - 1:
            print(f"  [{'phased' if phased else 'unphased'}] epoch {epoch:4d}  "
                  f"train {total:.4f}  val {current:.4f}  best {best['rmse']:.4f} "
                  f"@ {best['epoch']}", flush=True)
        if since >= args.patience:
            print(f"  [{'phased' if phased else 'unphased'}] stopped at {epoch}",
                  flush=True)
            break

    head.load_state_dict(best["state"])
    if best.get("encoder") is not None:
        encoder.load_state_dict(best["encoder"])
    head.eval()
    encoder.eval()
    rows = []
    with torch.no_grad():
        for molecule in held_out:
            predicted = predict(molecule, False).cpu().numpy()
            rows.append(dict(
                molecule_id=molecule["molecule_id"],
                arm=("phased" if phased else "unphased")
                    + ("_finetuned" if finetune else "_frozen"),
                smiles=molecule["smiles"], n_atoms=molecule["n_atoms"],
                qm_barrier_kcal_mol=float(np.ptp(molecule["qm"])),
                **evaluate(predicted, molecule["qm"], molecule["grid"])))
    print(f"  [{'phased' if phased else 'unphased'}] {time.time()-started:.0f} s, "
          f"kept epoch {best['epoch']} at validation {best['rmse']:.4f}", flush=True)
    return rows, {"state": best["state"], "phased": phased,
                  "args": {"espaloma_version": args.espaloma_version,
                           "periodicities": args.periodicities,
                           "hidden": args.hidden, "encoder": args.encoder,
                           "finetuned": finetune},
                  "encoder_state": best.get("encoder")}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--torsiondrives", required=True)
    p.add_argument("--encoder", default="espaloma", choices=("espaloma", "nagl"))
    p.add_argument("--espaloma-version", default="0.3.2")
    p.add_argument("--periodicities", type=int, default=6)
    p.add_argument("--hidden", default="128,128")
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--patience", type=int, default=60)
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--clip", type=float, default=5.0)
    p.add_argument("--finetune-encoder", action="store_true",
                   help="let the graph encoder move as well as the readout")
    p.add_argument("--encoder-lr", type=float, default=3e-5)
    p.add_argument("--encoder-anchor", type=float, default=1.0)
    p.add_argument("--js-weight", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=20260911)
    p.add_argument("--test-fraction", type=float, default=0.25)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--sampler", default="scaffold")
    p.add_argument("--split-mode", default="molecule")
    p.add_argument("--folds", type=int, default=1)
    p.add_argument("--checkpoint", default=None,
                   help="where to save the two readouts")
    p.add_argument("--out", required=True)
    p.add_argument("--quarantine", required=True)
    args = p.parse_args()

    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import torch
    from finetune_torsion import prepare, split_molecules, GRAPH_ENCODERS

    cache = np.load(args.torsiondrives, allow_pickle=False)
    entries = json.loads(str(cache["meta"]))
    prepared, skipped = prepare(cache, entries, rotors="driven",
                                encoder=args.encoder)
    if not prepared:
        print(f"nothing prepared; first skip: {skipped[:1]}")
        return 1
    rng = np.random.default_rng(args.seed)
    train_idx, val_idx, test_idx, how = split_molecules(prepared, args, rng)
    train = [prepared[i] for i in train_idx]
    validation = [prepared[i] for i in val_idx]
    held_out = [prepared[i] for i in test_idx]
    print(f"{len(prepared)} molecules, split by {how}: {len(train)} train, "
          f"{len(validation)} validation, {len(held_out)} held out", flush=True)

    parts = GRAPH_ENCODERS[args.encoder]()
    rows, checkpoints = [], {}
    for phased in (False, True):
        arm_rows, checkpoint = run_arm(phased, train, validation, held_out,
                                       args, torch, parts)
        rows += arm_rows
        checkpoints["phased" if phased else "unphased"] = checkpoint
    if args.checkpoint:
        # Saved so the pose scorer loads the readout rather than retraining it.
        pathlib.Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoints, args.checkpoint)
        print(f"  -> {args.checkpoint}")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pathlib.Path(args.quarantine).open("w") as fh:
        fh.write("molecule_id,reason\n")
        for name, reason in skipped:
            fh.write(f'{name},"{reason}"\n')

    print(f"\n{'arm':<12}{'RMSE':>8}{'min shift':>12}{'JS':>8}   n")
    by = {}
    for row in rows:
        by.setdefault(row["arm"], []).append(row)
    for arm, subset in by.items():
        r = np.array([x["rmse"] for x in subset])
        s = np.array([x["minimum_shift_degrees"] for x in subset])
        j = np.array([x["js_distance"] for x in subset])
        print(f"{arm:<12}{np.median(r):8.3f}{np.median(s):11.1f}d{np.median(j):8.3f}"
              f"   {len(subset)}")
    if len(by) == 2:
        pair = {a.split("_")[0]: {x["molecule_id"]: x for x in v}
                for a, v in by.items()}
        shared = sorted(set(pair["phased"]) & set(pair["unphased"]))
        for metric in ("rmse", "js_distance", "minimum_shift_degrees"):
            d = np.array([pair["unphased"][m][metric] - pair["phased"][m][metric]
                          for m in shared])
            print(f"  paired {metric:<22} phased better on "
                  f"{int((d > 0).sum())} of {len(d)}, median {np.median(d):+.4f}")
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
