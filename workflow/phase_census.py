"""How much of a real torsion profile is odd in the dihedral, and whose chemistry is it?

QUESTION. A fixed-phase torsion form fits sum_n A_n cos(n phi). A free-phase one
fits sum_n A_n cos(n phi - delta_n). The difference is exactly the part of the
profile that is ODD in phi, so the most a continuous phase can ever buy is the
residual of a cosine-only least-squares fit. What chemistry lives in that part?

WHY THIS IS NOT A MODEL COMPARISON. It is least squares on the QM data. No
encoder, no training, no split. The number it produces is a CEILING on what any
phase-fitting architecture can gain over any fixed-phase one, so a small answer
closes the question for every such model at once, including ours.

THE CORRECTION THIS ENCODES. At the TORSION level espaloma and Grappa fix the
phase. At the BOND level they need not: summing k_ab cos(n(phi + delta_a +
eps_b)) over torsions whose substituent offsets differ yields
C cos(n phi) - S sin(n phi) with both free, so an asymmetrically substituted
rotor reaches an arbitrary bond phase. The constraint bites only where
permutation invariance forces equivalent substituents to share k: three
substituents at 120 degrees make S vanish for every n that is not a multiple of
3. So the odd part is split by whether the driven bond's substituents are
symmetry-equivalent, by RDKit canonical rank, and only the equivalent ones are
out of reach.

ON WHAT. Any cache in this project's npz format. One independent unit is a
molecule; one row is one (molecule, driven bond) scan.

HOW. Least squares for a_0 + sum_{n=1..6} [a_n cos(n phi) + b_n sin(n phi)] on
the scan's own grid. Power at periodicity n is a_n^2 + b_n^2 and is invariant to
where the angle origin was put; the split into a_n and b_n is not, which is why
the origin is the driven dihedral the archive recorded and is reported as such.

CONTROLS. Named molecules with textbook answers, looked up by InChIKey and
reported whether or not they are present. Ethane must put its power at n = 3 and
nowhere else with no odd part; butane, biphenyl and 1,2-difluoroethane must come
back even; an anomeric acetal and a rotor beside a stereocentre must come back
odd. A census without these cannot tell a real signal from a broken fit.

WHY RMSE IS NOT THE METRIC HERE, and presto's own paper says so: "the RMSE is
sensitive to barrier height errors, which do not affect equilibrium
distributions", so it reports the Jensen-Shannon distance between Boltzmann
distributions beside it. The phase does not set how deep a minimum is. It sets
WHERE it is. An even-only form places its minima at symmetric positions by
construction, so on a rotor whose profile is half odd it must put the minimum at
the wrong angle, which costs almost nothing in RMSE and everything in which
rotamer is predicted. Two numbers per scan settle it: how far the minimum moves
when the odd part is removed, and the Jensen-Shannon distance between the two
Boltzmann distributions at 500 K, which is directly comparable to Table 1 of
Clark et al., ChemRxiv 2026.

UNITS. kcal/mol for residuals, degrees for minimum displacement, dimensionless
for power fractions and for the Jensen-Shannon distance in bits.

COST. Seconds, CPU.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib

import numpy as np

PERIODICITIES = 6

# Textbook cases. The expectation is what the fit MUST return if it is working.
NAMED = {
    "ethane": ("CC", "pure n=3, no odd part"),
    "butane": ("CCCC", "anti and gauche, even"),
    "1,2-difluoroethane": ("FCCF", "gauche preferred, electronic, even"),
    "ethylene glycol": ("OCCO", "intramolecular H bond, gauche"),
    "dimethoxymethane": ("COCOC", "ANOMERIC, gauche-gauche"),
    "methanediol": ("OCO", "ANOMERIC, the minimal case"),
    "2-methoxytetrahydropyran": ("COC1CCCCO1", "ANOMERIC, the textbook one"),
    "N-methylacetamide": ("CC(=O)NC", "amide, trans favoured"),
    "biphenyl": ("c1ccccc1-c1ccccc1", "~44 degrees, even, 2-fold"),
    "styrene": ("C=Cc1ccccc1", "planar, 2-fold"),
    "2-butanol": ("CC(O)CC", "stereocentre beside the rotor, odd expected"),
    "fluoromethanol": ("OCF", "ANOMERIC, strongest simple case"),
}

# Chemistry of the DRIVEN BOND. Each is a predicate on (mol, j, k).
def _hetero(atom, symbols=("O", "N", "F", "S", "Cl")):
    return atom.GetSymbol() in symbols


def classify(mol, j, k):
    """Which stereoelectronic class the driven bond belongs to, if any."""
    a, b = mol.GetAtomWithIdx(int(j)), mol.GetAtomWithIdx(int(k))
    labels = []

    def neighbours(atom, exclude):
        return [n for n in atom.GetNeighbors() if n.GetIdx() != exclude]

    # ANOMERIC: a C-X bond whose carbon carries a SECOND heteroatom, so a lone
    # pair on one heteroatom can donate into the sigma* of the other C-X bond.
    for centre, partner in ((a, b), (b, a)):
        if centre.GetSymbol() == "C" and _hetero(partner, ("O", "N", "S", "F")):
            others = [n for n in neighbours(centre, partner.GetIdx())
                      if _hetero(n, ("O", "N", "S", "F"))]
            if others:
                labels.append("anomeric")
                break

    # GAUCHE EFFECT: C-C with an electronegative substituent on each carbon.
    if a.GetSymbol() == b.GetSymbol() == "C":
        if (any(_hetero(n, ("F", "O", "N")) for n in neighbours(a, b.GetIdx()))
                and any(_hetero(n, ("F", "O", "N")) for n in neighbours(b, a.GetIdx()))):
            labels.append("gauche_effect")

    # AMIDE: the carbonyl carbon to nitrogen bond.
    for centre, partner in ((a, b), (b, a)):
        if (centre.GetSymbol() == "C" and partner.GetSymbol() == "N"
                and any(n.GetSymbol() == "O" and
                        mol.GetBondBetweenAtoms(centre.GetIdx(), n.GetIdx()
                                                ).GetBondTypeAsDouble() == 2.0
                        for n in neighbours(centre, partner.GetIdx()))):
            labels.append("amide")
            break

    if a.GetIsAromatic() and b.GetIsAromatic():
        labels.append("biaryl")
    elif a.GetHybridization().name == "SP2" and b.GetHybridization().name == "SP2":
        labels.append("conjugated")

    # A stereocentre on either side breaks E(phi) = E(-phi) directly.
    from rdkit import Chem
    centres = {i for i, _ in Chem.FindMolChiralCenters(
        mol, includeUnassigned=True, useLegacyImplementation=False)}
    if {int(j), int(k)} & centres or any(
            n.GetIdx() in centres for atom, other in ((a, b), (b, a))
            for n in neighbours(atom, other.GetIdx())):
        labels.append("stereocentre_adjacent")

    return labels or ["plain"]


def substituents_equivalent(mol, j, k):
    """Are the driven bond's substituents symmetry-equivalent on BOTH sides?

    If they are, permutation invariance forces a fixed-phase per-torsion form to
    a zero bond-level phase, and the odd part of the profile is unreachable for
    it. If they are not, that form can reach an arbitrary bond phase and the odd
    part costs it nothing. This is the criterion that decides which of the two.
    """
    from rdkit import Chem

    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    for centre, partner in ((int(j), int(k)), (int(k), int(j))):
        others = [n.GetIdx() for n in mol.GetAtomWithIdx(centre).GetNeighbors()
                  if n.GetIdx() != partner]
        if len(others) > 1 and len({ranks[i] for i in others}) > 1:
            return False
    return True


def harmonics(grid_degrees, energies, n_max=PERIODICITIES):
    """(a, b, residuals). Least squares, no regularisation, no weighting."""
    phi = np.radians(np.asarray(grid_degrees, dtype=float))
    e = np.asarray(energies, dtype=float)
    n = np.arange(1, n_max + 1)
    design = np.column_stack(
        [np.ones_like(phi)]
        + [np.cos(phi[:, None] * n)[:, i] for i in range(n_max)]
        + [np.sin(phi[:, None] * n)[:, i] for i in range(n_max)]
    )
    full, *_ = np.linalg.lstsq(design, e, rcond=None)
    a, b = full[1:n_max + 1], full[n_max + 1:]

    def rms(columns, coefficients):
        return float(np.sqrt(np.mean((design[:, columns] @ coefficients - e) ** 2)))

    even_columns = list(range(0, n_max + 1))
    even, *_ = np.linalg.lstsq(design[:, even_columns], e, rcond=None)
    three = list(range(0, 4)) + list(range(n_max + 1, n_max + 4))
    low, *_ = np.linalg.lstsq(design[:, three], e, rcond=None)
    # Where each fit puts its minimum, and how different the two Boltzmann
    # distributions are. Evaluated on a dense grid rather than on the 24 scan
    # points, because a minimum that moves by 10 degrees is invisible at 15
    # degree spacing and 10 degrees is a real rotamer error.
    dense = np.linspace(-np.pi, np.pi, 721, endpoint=False)
    basis = np.column_stack(
        [np.ones_like(dense)]
        + [np.cos(dense[:, None] * n)[:, i] for i in range(n_max)]
        + [np.sin(dense[:, None] * n)[:, i] for i in range(n_max)]
    )
    curve_full = basis @ full
    even_padded = np.zeros_like(full)
    even_padded[:n_max + 1] = even
    curve_even = basis @ even_padded

    def boltzmann(curve, kelvin=500.0):
        beta = 1.0 / (0.0019872041 * kelvin)          # kcal/mol per K
        # A profile with a very large range underflows every weight but one, and
        # an underdetermined fit (a scan with fewer points than coefficients)
        # can produce a range of 1e10. Floored so the mixture below is strictly
        # positive wherever either distribution is, which is what makes the
        # Jensen-Shannon distance finite.
        w = np.exp(-beta * np.clip(curve - curve.min(), 0.0, 700.0))
        total = w.sum()
        return w / total if total > 0 else np.full_like(w, 1.0 / w.size)

    p_full, p_even = boltzmann(curve_full), boltzmann(curve_even)
    m = 0.5 * (p_full + p_even)

    def kl(p, q):
        mask = (p > 0) & (q > 0)
        return float((p[mask] * np.log2(p[mask] / q[mask])).sum())

    shift = np.degrees(dense[int(np.argmin(curve_full))]
                       - dense[int(np.argmin(curve_even))])
    return {
        "a": a, "b": b,
        "rms_full": rms(list(range(design.shape[1])), full),
        "rms_even_only": rms(even_columns, even),
        "rms_three_harmonics": rms(three, low),
        "rms_flat": float(np.sqrt(np.mean((e - e.mean()) ** 2))),
        # Wrapped to (-180, 180]: a rotor is periodic and a 350 degree shift is
        # a 10 degree one the other way.
        "minimum_shift_degrees": float(abs((shift + 180.0) % 360.0 - 180.0)),
        "js_distance": float(np.sqrt(max(0.5 * kl(p_full, m) + 0.5 * kl(p_even, m), 0.0))),
    }


def scans_from_npz(path, limit=0):
    """Yield (entry, energies, bonds, orders) from this project's cache format."""
    cache = np.load(path, allow_pickle=False)
    meta = json.loads(str(cache["meta"]))
    if limit:
        meta = meta[:limit]
    for entry in meta:
        key = entry["key"]
        yield (entry, cache[f"energy::{key}"], cache[f"bonds::{key}"],
               cache[f"bond_orders::{key}"])


def scans_from_themol(path, limit=0, min_points=20):
    """Yield the same tuples from a THEMol TorsionScan HDF5 shard.

    Read directly rather than converted to npz first: 4.2 million scans would be
    hundreds of gigabytes written once and read once, and the census needs the
    energies and the connectivity and nothing else.

    THEMol's energies are already kcal/mol, its `torsion_atom_indices` are
    0-based into the coordinate array rather than the 1-based atom map numbers of
    the SMILES beside them, and each scan's grid has to be computed from the
    geometries because no angle column is stored.
    """
    import h5py
    from import_themol import connectivity, dihedral

    with h5py.File(path, "r") as handle:
        taken = 0
        for name in handle:
            if limit and taken >= limit:
                return
            group = handle[name]
            points = sorted((k for k in group if k.startswith("constraint ")),
                            key=lambda k: int(k.split()[1]))
            if len(points) < min_points:
                continue
            smiles = group["mapped_isomeric_smiles"][()]
            smiles = smiles.decode() if isinstance(smiles, bytes) else str(smiles)
            bonds, orders, symbols = connectivity(smiles)
            atoms = [int(a) for a in
                     np.asarray(group["torsion_atom_indices"][()]).ravel()]
            xyz = np.stack([np.asarray(group[k]["coords"][()], dtype=float)
                            for k in points])
            energies = np.asarray([float(group[k]["energy"][()]) for k in points])
            grid = [dihedral(frame, atoms) for frame in xyz]
            order = np.argsort(grid)
            entry = {
                "molecule_id": name, "symbols": symbols, "n_atoms": len(symbols),
                "grid": [float(grid[int(i)]) for i in order],
                "driven_dihedral": atoms, "smiles": smiles, "key": name,
            }
            yield entry, (energies - energies.min())[order], bonds, orders
            taken += 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--torsiondrives", nargs="+", required=True)
    p.add_argument("--format", choices=("npz", "themol"), default="npz",
                   help="'themol' reads TorsionScan HDF5 shards directly")
    p.add_argument("--min-points", type=int, default=20,
                   help="six harmonics need thirteen points; twenty is the "
                        "filter THEMol's variable scan lengths need")
    p.add_argument("--out", required=True)
    p.add_argument("--named", required=True)
    p.add_argument("--quarantine", required=True)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from espaloma_encoder import rdkit_from_cache

    wanted = {}
    for name, (smiles, expectation) in NAMED.items():
        m = Chem.MolFromSmiles(smiles)
        wanted[Chem.MolToInchiKey(m)] = (name, smiles, expectation)

    reader = scans_from_themol if args.format == "themol" else scans_from_npz
    rows, found, quarantine = [], {}, []
    for path in args.torsiondrives:
        source = (reader(path, args.limit, args.min_points)
                  if args.format == "themol" else reader(path, args.limit))
        for entry, energies, bond_list, order_list in source:
            try:
                driven = entry.get("driven_dihedral")
                if not driven or len(driven) != 4:
                    raise ValueError("no driven dihedral")
                mol = rdkit_from_cache(entry["symbols"], bond_list, order_list)
                j, k = int(driven[1]), int(driven[2])
                fit = harmonics(entry["grid"], energies)
                power = fit["a"] ** 2 + fit["b"] ** 2
                total = float(power.sum()) or float("nan")
                inchi = Chem.MolToInchiKey(Chem.RemoveHs(mol))
                row = {
                    "source": pathlib.PurePath(path).stem,
                    "molecule_id": entry["molecule_id"],
                    "inchi_key": inchi,
                    "smiles": entry.get("smiles", ""),
                    "driven_bond": f"{j}-{k}",
                    "n_heavy": sum(1 for s in entry["symbols"] if s != "H"),
                    "classes": "|".join(classify(mol, j, k)),
                    "substituents_equivalent": int(substituents_equivalent(mol, j, k)),
                    "barrier_kcal_mol": float(np.ptp(energies)),
                    "odd_fraction": float((fit["b"] ** 2).sum() / total),
                    "power_above_n3": float(power[3:].sum() / total),
                    "rms_full": fit["rms_full"],
                    "rms_even_only": fit["rms_even_only"],
                    "rms_three_harmonics": fit["rms_three_harmonics"],
                    "rms_flat": fit["rms_flat"],
                    "phase_gain": fit["rms_even_only"] - fit["rms_full"],
                    "minimum_shift_degrees": fit["minimum_shift_degrees"],
                    "js_distance": fit["js_distance"],
                    **{f"power_n{i+1}": float(power[i] / total) for i in range(PERIODICITIES)},
                }
                rows.append(row)
                if inchi in wanted:
                    found.setdefault(inchi, []).append(row)
            except Exception as error:                   # noqa: BLE001
                quarantine.append((entry.get("molecule_id", "?"),
                                   f"{type(error).__name__}: {error}"))

    if not rows:
        print("no scan decomposed")
        return 1
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pathlib.Path(args.quarantine).open("w") as fh:
        fh.write("molecule_id,reason\n")
        for name, reason in quarantine:
            fh.write(f'{name},"{reason}"\n')

    with pathlib.Path(args.named).open("w", newline="") as fh:
        names = ["name", "smiles", "expectation", "present", "n_scans",
                 "odd_fraction", "minimum_shift_degrees", "js_distance",
                 "power_n3", "phase_gain", "barrier_kcal_mol"]
        writer = csv.DictWriter(fh, fieldnames=names)
        writer.writeheader()
        for inchi, (name, smiles, expectation) in wanted.items():
            hits = found.get(inchi, [])
            writer.writerow({
                "name": name, "smiles": smiles, "expectation": expectation,
                "present": int(bool(hits)), "n_scans": len(hits),
                "odd_fraction": np.median([h["odd_fraction"] for h in hits]) if hits else "",
                "minimum_shift_degrees": np.median([h["minimum_shift_degrees"] for h in hits]) if hits else "",
                "js_distance": np.median([h["js_distance"] for h in hits]) if hits else "",
                "power_n3": np.median([h["power_n3"] for h in hits]) if hits else "",
                "phase_gain": np.median([h["phase_gain"] for h in hits]) if hits else "",
                "barrier_kcal_mol": np.median([h["barrier_kcal_mol"] for h in hits]) if hits else "",
            })

    def summarise(label, subset):
        if not subset:
            return
        odd = np.array([r["odd_fraction"] for r in subset])
        gain = np.array([r["phase_gain"] for r in subset])
        shift = np.array([r["minimum_shift_degrees"] for r in subset])
        js = np.array([r["js_distance"] for r in subset])
        print(f"  {label:<26}{len(subset):>6}  odd {np.median(odd):.3f}"
              f"  gain {np.median(gain):5.3f}"
              f"  min shift {np.median(shift):5.1f} deg (90th {np.percentile(shift, 90):5.1f})"
              f"  JS {np.median(js):.3f}")

    print(f"{len(rows):,} scans decomposed, {len(quarantine):,} quarantined")
    print(f"\nWhat removing the odd part costs. The minimum shift is the rotamer")
    print(f"error a fixed-phase form must make; JS is Boltzmann overlap at 500 K,")
    print(f"comparable to presto Table 1 where Sage is 0.30 and presto 0.12.")
    print(f"\n{'cut':<26}{'n':>6}  odd    gain   minimum displacement        JS")
    summarise("ALL", rows)
    summarise("substituents equivalent", [r for r in rows if r["substituents_equivalent"]])
    summarise("substituents distinct", [r for r in rows if not r["substituents_equivalent"]])
    for cls in ("anomeric", "gauche_effect", "amide", "biaryl", "conjugated",
                "stereocentre_adjacent", "plain"):
        summarise(cls, [r for r in rows if cls in r["classes"].split("|")])
    print(f"\nceilings, median kcal/mol: six harmonics "
          f"{np.median([r['rms_full'] for r in rows]):.4f}, "
          f"three {np.median([r['rms_three_harmonics'] for r in rows]):.4f}, "
          f"flat {np.median([r['rms_flat'] for r in rows]):.4f}")
    print(f"  -> {out}\n  -> {args.named}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
