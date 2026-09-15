"""The decision tree: what each test decides and what ends the work.

Drawn rather than listed because the structure is the point. Every gate has a
numeric criterion fixed before the data exists, and a failure branch that stops
the project rather than redirecting it.

EVERY NUMBER ON THE FIGURE IS READ FROM A FILE. A flowchart with its criteria
typed in is a flowchart that goes stale the first time a rule reruns, and the
criteria are the only part of it that matters. A gate whose inputs are not on
disk yet is drawn as pending and says so, rather than carrying a number from
memory.

The two criteria that recur:

  0.767 kcal/mol   how far two DFT references for the same 497 molecules sit
                   apart on barrier height. A difference smaller than this is
                   smaller than not knowing which reference was meant.
  0.724 kcal/mol   MACE-OFF23 small, the cheapest pretrained potential, on the
                   same held-out molecules. Anything above this is dominated.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib

import numpy as np

BLUE, ORANGE, GREEN, VERM, GREY = ("#0072B2", "#E69F00", "#009E73", "#D55E00",
                                   "#666666")
PASS_GREEN, FAIL_RED = "#065f46", "#991b1b"

# Fugaku, RIKEN: 158,976 nodes of 48 compute cores each.
FUGAKU_CORES = 158_976 * 48
ENAMINE_REAL = 48e9          # REAL Space, enumerable products, SMILES only


def read_csv(path):
    path = pathlib.Path(path)
    if not path.exists():
        return []
    with path.open() as fh:
        return list(csv.DictReader(fh))


def median_of(rows, column, where=None):
    values = []
    for row in rows:
        if where and not all(row.get(k) == v for k, v in where.items()):
            continue
        try:
            values.append(float(row[column]))
        except (KeyError, TypeError, ValueError):
            continue
    return float(np.median(values)) if values else None


def collect(root):
    """Everything the figure says, read from the project's own outputs."""
    root = pathlib.Path(root)
    facts = {}

    accuracy = read_csv(root / "data/accuracy_cost.csv")
    for row in accuracy:
        facts[f"rmse_{row['model']}"] = float(row["rmse"])
    facts["cheapest_potential"] = facts.get("rmse_mace_off23_small")
    facts["ours_best"] = min(
        (v for k, v in facts.items() if k.startswith("rmse_ours")), default=None
    )

    for device in ("cuda", "cpu"):
        rows = read_csv(root / f"data/cost_per_profile_{device}.csv")
        if not rows:
            rows = read_csv(root / "data/cost_per_profile.csv") if device == "cuda" else []
        encoder = [r for r in rows if r.get("path") == "encoder + head"]
        if encoder:
            facts[f"encoder_{device}"] = float(
                np.median([float(r["seconds_per_molecule"]) for r in encoder])
            )

    conformer = read_csv(root / "data/conformer_sensitivity.csv")
    for source in ("privileged", "qm_rotated", "etkdg", "noise", "etkdg_spread"):
        facts[f"conf_{source}"] = median_of(conformer, "rmse_vs_qm",
                                            {"source": source})
    if conformer:
        facts["conf_n"] = len({r["molecule_id"] for r in conformer})
        etkdg = [float(r["heavy_rmsd"]) for r in conformer
                 if r["source"] == "etkdg" and r["heavy_rmsd"] not in ("", "nan")]
        facts["conf_rmsd"] = float(np.median(etkdg)) if etkdg else None
    facts["etkdg_ms"] = None
    log = root / "logs/conformer_sensitivity.log"
    if log.exists():
        for line in log.read_text(errors="ignore").splitlines():
            if "ms per conformer" in line:
                facts["etkdg_ms"] = float(line.split("ETKDG")[1].split("ms")[0])

    npz = root / "data/test/torsiontest2000.npz"
    if npz.exists():
        meta = json.loads(str(np.load(npz, allow_pickle=False)["meta"]))
        facts["tt2000_n"] = len(meta)
        facts["tt2000_ionic"] = sum(1 for m in meta if m.get("charge"))
        facts["tt2000_elements"] = len({s for m in meta for s in m["symbols"]})
    return facts


def fugaku(facts):
    """Enamine REAL, from SMILES to a table of amplitudes, on one machine.

    Two costs, not one. A conformer has to be generated before the encoder can
    see anything, because Enamine REAL and ZINC22 are distributed as SMILES.
    Both are measured here: ETKDGv3 plus MMFF94s per conformer on one core, and
    one encoder pass per molecule. The encoder figure is a CPU figure or it is
    not an estimate for a machine with no GPUs.
    """
    etkdg = (facts.get("etkdg_ms") or 0) / 1e3
    encoder = facts.get("encoder_cpu")
    if not etkdg or not encoder:
        return None
    core_seconds = ENAMINE_REAL * (etkdg + encoder)
    return {
        "core_hours": core_seconds / 3600,
        "fugaku_hours": core_seconds / FUGAKU_CORES / 3600,
        "etkdg_ms": etkdg * 1e3,
        "encoder_ms": encoder * 1e3,
    }


def gates(facts):
    """Seven gates in two tracks. Text only; every number comes from `facts`."""
    def kc(value, digits=3):
        return "pending" if value is None else f"{value:.{digits}f}"

    label = facts.get("cheapest_potential")
    ours = facts.get("ours_best")
    plan = fugaku(facts)

    model_track = [
        ("G1", "All bonds, not just the driven one",
         "cross-bond split, held-out bonds of training molecules",
         "held-out bond RMSE within 20 percent of the driven-bond result",
         "the cost advantage is an advantage at producing wrong answers",
         "RUNNING  job 79351", GREEN),
        ("G2", "Architecture, or just tuning?",
         "Optuna TPE, 60+ trials, the hand-chosen configuration as control",
         f"tuning reaches {(ours or 1.25) - 0.767/6:.2f}, one sixth of the label "
         f"uncertainty below the control",
         f"{kc(ours, 2)} is the architecture, and k_n needs a different predictor",
         "RUNNING  job 79339", GREEN),
        ("G3", "Is the one win real?",
         (f"TorsionTest2000, {facts.get('tt2000_n', 0):,} scans, "
          f"{facts.get('tt2000_elements', 0)} elements, "
          f"{facts.get('tt2000_ionic', 0)} ionic, wB97X-D3BJ/def2-TZVPD"),
         "beat both potentials on the ionic and expanded-element scans",
         "the 46-molecule charged-species win was noise",
         "staged, ~2 h", ORANGE),
        ("G4", "Does more data move it?",
         "THEMol TorsionScan at 15k, 150k and 2.4M molecules, same theory",
         f"RMSE below {kc(label)}, the cheapest potential, at some cohort size",
         "the 0.047 scaling exponent holds and no amount of data fixes it",
         "345 GB staging", ORANGE),
    ]
    product_track = [
        ("G5", "Can it start from a SMILES?",
         (f"{facts['conf_n']} held-out molecules, ETKDGv3 conformers at "
          f"heavy-atom RMSD {kc(facts.get('conf_rmsd'), 2)} A from the QM minimum"
          if facts.get("conf_n") else
          "held-out molecules, ETKDGv3 conformers against the QM minimum"),
         (f"generated conformer reaches {kc(facts.get('conf_etkdg'), 2)} against "
          f"the privileged {kc(facts.get('conf_privileged'), 2)}, inside 0.767"
          if facts.get("conf_etkdg") else
          "the generated conformer stays within 0.767 of the privileged path"),
         "the published accuracy needed 24 QM geometries it will never have",
         "RUNNING  local", BLUE),
        ("G6", "Does the use case work?",
         "docking pose refinement, PoseBusters, ligand geometry after rescoring",
         "better poses, or the same poses at materially lower cost",
         "fast, and not useful",
         "GPU-days", BLUE),
        ("G7", "Fugaku: the artifact",
         (f"precompute every rotor of Enamine REAL, {ENAMINE_REAL/1e9:.0f} B "
          f"compounds, SMILES in and amplitudes out"),
         (f"{plan['core_hours']/1e3:,.0f}k core-hours, "
          f"{plan['fugaku_hours']:.2f} h on {FUGAKU_CORES/1e6:.1f} M cores"
          if plan else "pending: needs the CPU encoder rate and the ETKDG rate"),
         "", "allocation", VERM),
    ]
    return model_track, product_track


def _wrap(text, width):
    words, lines, line = text.split(), [], ""
    for word in words:
        trial = f"{line} {word}".strip()
        if len(trial) > width and line:
            lines.append(line)
            line = word
        else:
            line = trial
    if line:
        lines.append(line)
    return lines


def draw(path, facts):
    """Boxes are sized from their own content.

    A fixed box height and a title sharing its line with a status label is how
    the first version of this figure put a running job number through a gate's
    name and ran three lines of criteria out through the bottom edge. Each box
    is laid out first, into a list of (text, size, colour, indent) lines, and
    the rectangle is then drawn around what the layout actually produced.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    matplotlib.rcParams.update({
        "figure.dpi": 200, "savefig.dpi": 400, "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 7,
    })

    model_track, product_track = gates(facts)
    WRAP, LINE, PAD, GAP, WIDTH = 52, 1.62, 1.5, 4.2, 44.0

    def lines_of(gate):
        tag, title, what, passc, failc, status, colour = gate
        out = [(f"{tag}  {title}", 7.6, "black", 0.0, "bold")]
        out += [(l, 5.7, GREY, 0.0, None) for l in _wrap(what, WRAP)]
        for n, line in enumerate(_wrap(passc, WRAP - 6)):
            out.append((("PASS  " if n == 0 else "") + line, 5.7, PASS_GREEN,
                        0.0 if n == 0 else 3.2, None))
        if failc:
            for n, line in enumerate(_wrap(failc, WRAP - 6)):
                out.append((("FAIL   " if n == 0 else "") + line, 5.7, FAIL_RED,
                            0.0 if n == 0 else 3.2, None))
        out.append((status, 5.8, colour, 0.0, "bold"))
        return out, colour, bool(failc)

    laid = [[lines_of(g) for g in track] for track in (model_track, product_track)]
    heights = [[len(x[0]) * LINE + 2 * PAD + 0.8 for x in track] for track in laid]
    tallest = max(sum(h) + GAP * (len(h) - 1) for h in heights)

    fig, ax = plt.subplots(figsize=(7.4, 0.115 * (tallest + 34)))
    ax.set_xlim(0, 108)
    ax.set_ylim(0, tallest + 34)
    ax.axis("off")
    TOP = tallest + 12.0

    for column, (track, track_heights, heading) in enumerate(zip(
            laid, heights, ("Is the model right?", "Is the product real?"))):
        x0 = 1.0 + column * 54.0
        ax.text(x0, TOP + 5.2, heading, fontsize=8.5, fontweight="bold")
        ax.plot([x0, x0 + WIDTH], [TOP + 3.4, TOP + 3.4], lw=0.8, color="#cccccc")
        top = TOP
        for i, ((content, colour, has_fail), height) in enumerate(
                zip(track, track_heights)):
            ax.add_patch(FancyBboxPatch(
                (x0, top - height), WIDTH, height, boxstyle="round,pad=0.3",
                facecolor=colour, alpha=0.12, edgecolor=colour, lw=1.0))
            y = top - PAD - 1.0
            for text, size, text_colour, indent, weight in content[:-1]:
                ax.text(x0 + 1.3 + indent, y, text, fontsize=size,
                        color=text_colour, va="top",
                        fontweight=weight or "normal")
                y -= LINE
            status, size, text_colour, _, weight = content[-1]
            ax.text(x0 + WIDTH - 1.3, top - height + PAD - 0.2, status,
                    fontsize=size, color=text_colour, ha="right", va="bottom",
                    fontweight="bold")
            if has_fail:
                mid = top - height / 2
                ax.add_patch(FancyArrowPatch(
                    (x0 + WIDTH, mid), (x0 + WIDTH + 4.2, mid),
                    arrowstyle="-|>", mutation_scale=7, lw=0.8,
                    color=FAIL_RED, alpha=0.75))
                ax.text(x0 + WIDTH + 4.8, mid, "stop", fontsize=5.8,
                        color=FAIL_RED, va="center")
            if i < len(track) - 1:
                ax.add_patch(FancyArrowPatch(
                    (x0 + WIDTH / 3, top - height),
                    (x0 + WIDTH / 3, top - height - GAP + 0.5),
                    arrowstyle="-|>", mutation_scale=7, lw=0.8, color="black"))
                ax.text(x0 + WIDTH / 3 + 1.4, top - height - GAP / 2, "pass",
                        fontsize=5.4, color=GREY, va="center")
            top -= height + GAP

    ax.text(0, tallest + 33, "Torsion phase model: what each test decides",
            fontsize=10.5, fontweight="bold", va="top")
    caption = (
        "Criteria fixed before the data exists. Two recur: 0.767 kcal/mol is the "
        "label uncertainty, how far two DFT references for the same 497 molecules "
        "sit apart,\nand "
        f"{facts.get('cheapest_potential', float('nan')):.3f} is MACE-OFF23 small "
        "on the same held-out set, above which the model is dominated. The left "
        "track asks whether the numbers are\nright, the right track whether "
        "anything can be built from them. Failing either stops the work."
    )
    ax.text(0, tallest + 28.5, caption, fontsize=6.3, color=GREY, va="top",
            linespacing=1.55)

    plan = fugaku(facts)
    if plan:
        ax.text(0, 0.5,
                f"G7 arithmetic, both rates measured here on one core: ETKDGv3 "
                f"plus MMFF94s at {plan['etkdg_ms']:.1f} ms per conformer and one "
                f"encoder pass at {plan['encoder_ms']:.0f} ms.\nEnamine REAL and "
                f"ZINC22 ship SMILES with no coordinates, so the conformer is not "
                f"optional and the table is keyed on a molecule, not a pose. The "
                f"Fugaku figure\nscales by core count alone and assumes an A64FX "
                f"core matches this one, which is the part still to be measured.",
                fontsize=6.0, color=GREY, va="bottom", linespacing=1.55)

    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default="results/gates.png")
    args = parser.parse_args()
    facts = collect(args.root)
    for name in sorted(facts):
        print(f"  {name:<22}{facts[name]}")
    print(f"  -> {draw(args.out, facts)}")


if __name__ == "__main__":
    main()
