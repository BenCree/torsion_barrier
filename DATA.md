# Where the data comes from

No data is committed here. Every cohort is public and the workflow fetches it.

| cohort | what | how to get it |
|---|---|---|
| OpenFF torsiondrives | 15,677 scans, 13,167 molecules, B3LYP-D3(BJ)/DZVP | `workflow/pull_torsiondrives.py`, from QCArchive |
| THEMol TorsionScan | 4,192,791 scans, 2,436,985 molecules, 93,994,576 geometries, B3LYP-D3(BJ)/DZVP, 344.8 GB | `workflow/fetch_hf_shards.py --repo ByteDance-Seed/THEMol --subfolder TorsionScan`. **CC-BY-NC-4.0, non-commercial** |
| TorsionNet500, TorsionTest2000 | 497 and 2,109 scans, wB97X-D3BJ/def2-TZVPD and DLPNO-CCSD(T)/CBS | `github.com/leifjacobson/MLFF_test_data`, then `workflow/import_schrodinger_torsions.py` |
| OpenFE Industry Benchmark 2024 | 2,146 ligands, 146 systems, gnina poses at `--num_modes 10` | `github.com/OpenFreeEnergy/IndustryBenchmarks2024` |

Models: espaloma 0.3.2 via `esp.get_model`, Sage 2.3.0 and
`openff-gnn-am1bcc-1.0.0.pt` via `openff-toolkit` and `openff-nagl-models`,
DPA-3.1-3M via deepmd, MACE-OFF23 via `mace-torch`.

## Units that bite

espaloma's `n4.data["k"]` is in **Hartree**, not kcal/mol
(`espaloma/units.py:21`), and every proper torsion is listed **twice**,
`(a,b,c,d)` and `(d,c,b,a)`, which have identical dihedrals
(`espaloma/graphs/deploy.py:241`). Reading it as kcal/mol without
de-duplicating makes every predicted barrier 1255 times too small.

THEMol energies are already kcal/mol; `torsion_atom_indices` is 0-based into
the coordinate array, not the 1-based atom map numbers of the SMILES beside it;
and its mapped SMILES write every hydrogen explicitly, so RDKit must be given
`removeHs=False` or a 33-atom molecule comes back with 17 atoms.
