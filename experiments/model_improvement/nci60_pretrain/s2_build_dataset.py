"""NCI60-Pretrain-v1 step 2: aggregate -> (NSC, CellMiner cell, dose) points; NSC -> SMILES -> atom-v1 graphs.

Run isolated (reads the untrusted nsc_smiles.csv):
  .venv/Scripts/python.exe -I experiments/model_improvement/nci60_pretrain/s2_build_dataset.py
Rules (docs/NCI60_PRETRAIN_V1_PROTOCOL.md §2):
  * PREFIX 'S' only (nsc_smiles.csv is keyed by standard NSC numbers); CONCENTRATION_UNIT 'M' only (others dropped, counted);
  * cells mapped with the frozen nci60_cell_aliases.json rules and required to have a fold-0 CellMiner pathway graph;
  * endpoint y = sum(COUNT_PTC*AVERAGE_PTC)/sum(COUNT_PTC)/100 per (NSC, cell, round(log10 M, 4));
    points outside the ALMANAC D1 monotherapy bound [0, 3] are dropped and counted (CompactPointTable contract);
  * dose u = log10(1 + D/1e-12 M);
  * structures: ALMANAC standardisation policy (structure_manifest.json) + atom-v1 featuriser; failures dropped, counted.
Outputs: data/interim/nci60_pretrain_v1/{nci60_points.parquet, nci60_compounds.parquet, nci60_graphs.pt,
         cell_label_mapping.csv, structure_failures.csv, s2_stats.json}
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bisurfgnn.pretrain.nci60 import (ALMANAC_POLICY, DATA_ROOT, append_log, dose_model, forbid_test_responses,  # noqa: E402
                                      load_fold0_cell_graphs, map_cell_labels, sha256_file, standardize_batch)

RAW = ROOT / "data/raw/nci60_dtp_202610"
OUT = ROOT / "data/interim/nci60_pretrain_v1"
RUN = ROOT / "runs/model_improvement/nci60_pretrain_v1"
Y_LO, Y_HI = 0.0, 3.0
QS = [0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]


def quantiles(x) -> dict:
    x = np.asarray(x, float)
    return dict(n=int(len(x)), mean=float(x.mean()), sd=float(x.std()), min=float(x.min()), max=float(x.max()),
                **{f"q{q:g}": float(np.quantile(x, q)) for q in QS}, frac_gt1=float((x > 1).mean()))


def almanac_training_mono(ctx):
    """Training-role ALMANAC monotherapy (fold-0 train shard; same dedup identity as SparseMono monotherapy_table)."""
    path = DATA_ROOT / "data/processed/one_new_scaffold_benchmark_unclipped/learned_point_shards/fold_0/train.parquet"
    cols = ["source_combodrugseq", "cell_id", "drug_a_id", "drug_b_id", "dose_a_model", "dose_b_model",
            "mono_response_a", "mono_response_b", "mono_mask_a", "mono_mask_b"]
    f = pd.read_parquet(path, columns=cols)
    parts = [pd.DataFrame({"drug": f[f"drug_{a}_id"], "cell_id": f.cell_id, "dose": np.round(f[f"dose_{a}_model"], 6),
                           "mono": f[f"mono_response_{a}"].astype(float), "mask": f[f"mono_mask_{a}"].astype(bool),
                           "row": f.source_combodrugseq}) for a in ("a", "b")]
    t = pd.concat(parts); t = t[t["mask"]]
    t = t.sort_values("row").drop_duplicates(["drug", "cell_id", "dose", "mono"]).reset_index(drop=True)
    mp = ctx.mapping[ctx.mapping.status.astype(str) == "mapped"]
    t["bio"] = t.cell_id.map(dict(zip(mp.source_cell_id, mp.biological_cell_id)))
    return t, path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    targets = [OUT / n for n in ("nci60_points.parquet", "nci60_compounds.parquet", "nci60_graphs.pt", "s2_stats.json")]
    if any(p.exists() for p in targets):
        raise FileExistsError([str(p) for p in targets if p.exists()])
    t0 = time.monotonic(); st: dict = {"y_bounds": [Y_LO, Y_HI], "structure_policy": ALMANAC_POLICY}
    agg = pd.read_parquet(OUT / "doseresp_aggregate_raw.parquet")
    labels = pd.read_csv(OUT / "doseresp_cell_labels.csv", keep_default_na=False)
    with forbid_test_responses() as guard:
        graphs, ctx, _ = load_fold0_cell_graphs()
        alm, alm_path = almanac_training_mono(ctx)
    st["parquet_reads"] = guard.accesses
    cellmap = map_cell_labels(labels, ROOT / "data/mappings/nci60_cell_aliases.json", set(graphs))
    cellmap.to_csv(OUT / "cell_label_mapping.csv", index=False)
    st["raw"] = dict(keys=len(agg), rows=int(agg.n_rows.sum()), nsc=int(agg.nsc.nunique()))
    st["prefix"] = agg.groupby("prefix", observed=True).n_rows.sum().astype(int).to_dict()
    st["unit"] = {u: dict(keys=int((agg.unit == u).sum()), rows=int(agg.n_rows[agg.unit == u].sum()),
                          nsc=int(agg.nsc[agg.unit == u].nunique())) for u in agg.unit.cat.categories if (agg.unit == u).any()}
    keep = (agg.prefix == "S") & (agg.unit == "M")
    st["dropped_non_S_or_non_M"] = dict(keys=int((~keep).sum()), rows=int(agg.n_rows[~keep].sum()))
    a = agg[keep].merge(cellmap[["cell_code", "status", "biological_cell_id"]], on="cell_code", how="left", validate="many_to_one")
    del agg
    lab = cellmap.assign(rows=cellmap.raw_rows)
    st["cell_labels"] = dict(total=len(cellmap), by_status=cellmap.status.value_counts().to_dict(),
                             mapped_biological_cells=int(cellmap.loc[cellmap.status == "mapped", "biological_cell_id"].nunique()),
                             fold0_graph_cells=len(graphs))
    st["cell_drop"] = {s: dict(keys=int((a.status == s).sum()), rows=int(a.n_rows[a.status == s].sum()))
                       for s in sorted(a.status.unique()) if s != "mapped"}
    st["unmapped_labels"] = lab.loc[lab.status != "mapped", ["panel_name", "cell_name", "status", "rows"]].to_dict("records")
    a = a[a.status == "mapped"]
    cells = sorted(a.biological_cell_id.unique()); cidx = {c: i for i, c in enumerate(cells)}
    conc_int = np.round(a.log10_conc.to_numpy() * 1e4).astype(np.int64)
    key = (a.nsc.to_numpy().astype(np.int64) << 26) | (a.biological_cell_id.map(cidx).to_numpy().astype(np.int64) << 20) | (conc_int + (1 << 19))
    uk, inv = np.unique(key, return_inverse=True)
    w = np.bincount(inv, weights=a.w_sum.to_numpy()); wy = np.bincount(inv, weights=a.wy_sum.to_numpy())
    wyy = np.bincount(inv, weights=a.wyy_sum.to_numpy()); nr = np.bincount(inv, weights=a.n_rows.to_numpy())
    del a
    nsc = (uk >> 26).astype(np.int32); cell = ((uk >> 20) & 63).astype(np.int16); ci = (uk & ((1 << 20) - 1)) - (1 << 19)
    mean_pct = wy / w
    pts = pd.DataFrame({"nsc": nsc, "cell": pd.Categorical.from_codes(cell, cells), "log10_conc": ci / 1e4,
                        "y": mean_pct / 100.0, "y_between_exp_sd": np.sqrt(np.clip(wyy / w - mean_pct ** 2, 0, None)) / 100.0,
                        "n_exp": nr.astype(np.int32), "count_ptc": w.astype(np.float32)})
    pts["dose_u"] = dose_model(pts.log10_conc.to_numpy())
    st["mapped_points_before_bounds"] = dict(points=len(pts), nsc=int(pts.nsc.nunique()))
    st["y_before_bounds"] = quantiles(pts.y)
    inb = (pts.y >= Y_LO) & (pts.y <= Y_HI)
    st["dropped_out_of_bounds"] = dict(points=int((~inb).sum()), below0=int((pts.y < Y_LO).sum()), above3=int((pts.y > Y_HI).sum()))
    pts = pts[inb].reset_index(drop=True)
    # structures
    smi = pd.read_csv(RAW / "nsc_smiles.csv", dtype={"NSC": str, "SMILES": str}, keep_default_na=False)
    st["nsc_smiles_sha256"] = sha256_file(RAW / "nsc_smiles.csv"); st["nsc_smiles_rows"] = len(smi)
    smi = smi[smi.NSC.str.fullmatch(r"\d+")]
    smap = dict(zip(smi.NSC.astype(int), smi.SMILES.str.strip()))
    comp = np.unique(pts.nsc.to_numpy())
    no_smiles = [int(n) for n in comp if not smap.get(int(n), "")]
    st["compounds_without_smiles"] = dict(compounds=len(no_smiles), points=int(pts.nsc.isin(no_smiles).sum()))
    items = [(int(n), smap[int(n)]) for n in comp if smap.get(int(n), "")]
    batches = [items[i:i + 200] for i in range(0, len(items), 200)]
    t1 = time.monotonic(); recs = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for i, r in enumerate(ex.map(standardize_batch, batches)):
            recs.extend(r)
            if (i + 1) % 25 == 0:
                print(f"  standardised {len(recs):,}/{len(items):,} {time.monotonic() - t1:.0f}s", flush=True)
    st["standardize_seconds"] = time.monotonic() - t1
    ok = [r for r in recs if r["ok"] and r["diag_ok"] and r["sym_ok"] and r["binary_ok"]]
    bad = [r for r in recs if not (r["ok"] and r["diag_ok"] and r["sym_ok"] and r["binary_ok"])]
    fail = pd.DataFrame([dict(nsc=r["nsc"], raw_smiles=r["raw_smiles"], error_type=r.get("error_type", "graph_check"),
                              error=r.get("error", "")) for r in bad])
    fail.to_csv(OUT / "structure_failures.csv", index=False)
    st["structure"] = dict(attempted=len(items), ok=len(ok), failed=len(bad),
                           failed_by_type=fail.error_type.value_counts().to_dict() if len(fail) else {},
                           failed_points=int(pts.nsc.isin(set(fail.nsc) if len(fail) else set()).sum()))
    ok.sort(key=lambda r: r["nsc"])
    # CSR bank with an atom-feature vocabulary (exact float32 rows)
    feats = np.concatenate([r["node_features"] for r in ok]); vocab, atom_vocab = np.unique(feats, axis=0, return_inverse=True)
    atom_ptr = np.concatenate([[0], np.cumsum([r["n_atoms"] for r in ok])]).astype(np.int64)
    edges = np.concatenate([r["edges"] for r in ok]).astype(np.int16)
    edge_ptr = np.concatenate([[0], np.cumsum([len(r["edges"]) for r in ok])]).astype(np.int64)
    import torch
    bank = dict(nsc=torch.as_tensor([r["nsc"] for r in ok], dtype=torch.int64), atom_vocab=torch.as_tensor(vocab, dtype=torch.float32),
                atom_index=torch.as_tensor(atom_vocab.reshape(-1), dtype=torch.int32), atom_ptr=torch.as_tensor(atom_ptr),
                edges=torch.as_tensor(edges.astype(np.int32)), edge_ptr=torch.as_tensor(edge_ptr),
                atom_feature_version="atom-v1", policy=json.dumps(ALMANAC_POLICY))
    torch.save(bank, OUT / "nci60_graphs.pt")
    cdf = pd.DataFrame([dict(nsc=r["nsc"], graph_index=i, raw_smiles=r["raw_smiles"], canonical_smiles=r["canonical_smiles"],
                             identity_group=r["identity_group"], murcko_scaffold=r["murcko_scaffold"], inchikey=r["inchikey"],
                             flags=r["flags"], n_atoms=r["n_atoms"], n_bonds=len(r["edges"]), morgan2048=r["fp"])
                        for i, r in enumerate(ok)])
    cdf.to_parquet(OUT / "nci60_compounds.parquet", index=False)
    pts = pts[pts.nsc.isin(set(cdf.nsc))].reset_index(drop=True)
    pts.to_parquet(OUT / "nci60_points.parquet", index=False)
    # summary & sanity
    per_cd = pts.groupby(["nsc", "cell"], observed=True).size()
    st["final"] = dict(compounds=int(pts.nsc.nunique()), cells=int(pts.cell.nunique()), curves_nsc_cell=int(len(per_cd)),
                       points=len(pts), doses_per_curve=per_cd.describe().to_dict(), atoms_total=int(atom_ptr[-1]),
                       atom_vocab=int(len(vocab)), max_atoms=int(cdf.n_atoms.max()), median_atoms=float(cdf.n_atoms.median()),
                       compounds_with_metal_flag=int(cdf["flags"].str.contains("contains_metal").sum()))
    st["y_final"] = quantiles(pts.y)
    st["dose_log10M_final"] = quantiles(pts.log10_conc); st["dose_u_final"] = quantiles(pts.dose_u)
    st["almanac_training_mono"] = dict(source=str(alm_path), y=quantiles(alm.mono), dose_u=quantiles(alm.dose),
                                       observations=len(alm), drugs=int(alm.drug.nunique()))
    # matched sanity on TRAINING-role ALMANAC drugs only: same NSC, same cell, dose bin of 0.5 log10 units
    al = alm.assign(nsc=alm.drug.str[1:].astype(int), b=np.round(alm.dose * 2) / 2).groupby(["nsc", "bio", "b"]).mono.mean()
    nc = pts.assign(b=np.round(pts.dose_u * 2) / 2).groupby(["nsc", "cell", "b"], observed=True).y.mean()
    nc.index = nc.index.set_names(["nsc", "bio", "b"])
    j = pd.concat([al.rename("almanac"), nc.rename("nci60")], axis=1, join="inner")
    st["matched_training_drug_sanity"] = dict(groups=len(j), drugs=int(j.index.get_level_values(0).nunique()),
                                              pearson=float(j.corr().iloc[0, 1]) if len(j) > 2 else None,
                                              rmse=float(np.sqrt(((j.almanac - j.nci60) ** 2).mean())) if len(j) else None,
                                              mean_diff_nci60_minus_almanac=float((j.nci60 - j.almanac).mean()) if len(j) else None)
    st["outputs"] = {p.name: dict(sha256=sha256_file(p), bytes=p.stat().st_size)
                     for p in (OUT / "nci60_points.parquet", OUT / "nci60_compounds.parquet", OUT / "nci60_graphs.pt",
                               OUT / "cell_label_mapping.csv", OUT / "structure_failures.csv")}
    st["seconds"] = time.monotonic() - t0
    (OUT / "s2_stats.json").write_text(json.dumps(st, indent=1, default=float), encoding="utf-8")
    f = st["final"]
    msg = (f"STEP2 dataset: compounds {f['compounds']:,}, cells {f['cells']}, curves(NSC,cell) {f['curves_nsc_cell']:,}, points {f['points']:,}; "
           f"dropped: non-S/non-M rows {st['dropped_non_S_or_non_M']['rows']:,}, unmapped-cell rows "
           f"{sum(v['rows'] for v in st['cell_drop'].values()):,}, out-of-[0,3] points {st['dropped_out_of_bounds']['points']:,}, "
           f"no-SMILES compounds {st['compounds_without_smiles']['compounds']:,}, structure failures {st['structure']['failed']:,}; "
           f"y median {st['y_final']['q0.5']:.3f} (ALMANAC train mono {st['almanac_training_mono']['y']['q0.5']:.3f}); "
           f"points sha256 {st['outputs']['nci60_points.parquet']['sha256']}; graphs sha256 {st['outputs']['nci60_graphs.pt']['sha256']}; "
           f"{st['seconds']:.0f}s")
    append_log(RUN / "STEP_LOG.md", msg); print(msg)


if __name__ == "__main__":
    main()
