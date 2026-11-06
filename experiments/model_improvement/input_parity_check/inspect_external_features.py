"""Input-parity feasibility, step 1 (read-only): DeepSynBa's external drug/cell features against the
PublicSplitGate-v1 train and validation sets.

Opens only
  * the two author feature files (data.zip of github.com/hikuru/DeepSynBa, commit 43c16961, main checkout);
  * the reproduction-v2 protocol.json (for the recorded feature-file hashes);
  * S1 mapping files and the ID columns of the S1 train/validation point tables;
  * the local structure table (SMILES) and the frozen CellMiner RNA-seq matrix (identity sanity checks).
No response value, no split-3 row, no metadata CSV, nothing under s5/ is opened. Writes nothing: stdout only.

  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -I experiments/model_improvement/input_parity_check/inspect_external_features.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

if not sys.flags.isolated:
    raise SystemExit("run with `python -I` (the embedding file is a torch/pickle container)")
sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[3]                      # this worktree
MAIN = ROOT.parent / "bisurfgnn"                                 # read-only main checkout
FEAT = MAIN / "data/raw/external_benchmarks/deepsynba/data/features"
GEX = FEAT / "NCI-60_landmark_gex.csv"
EMB = FEAT / "NCI-ALMANAC_drug_molformer_embeddings.pt"
REPRO_PROTOCOL = MAIN / "runs/audit_revision/deepsynba_repro_v2_20261012/protocol.json"
S1 = ROOT / "runs/model_improvement/public_split_gate_v1/s1"
DRUGS = MAIN / "data/processed/nci_almanac_unclipped/drugs.csv"
CELLMINER = MAIN / "data/processed/cellminer/expression.parquet"
ID_COLUMNS = ["matrix_key", "deepsynba_cell", "cell_id", "biological_cell_id",
              "deepsynba_drug_a", "deepsynba_drug_b", "drug_a_id", "drug_b_id"]
FORBIDDEN = ("public_split_gate_v1/s5", "TEST_EVALUATED", "test_predictions", "final_predictions",
             "test.parquet", "nci_synergy_dose_response", "nci_synergy_metadata", "jaaks", "az_sanger")
RNG_SEED = 20261013


def guard(path: Path) -> Path:
    text = Path(path).resolve().as_posix()
    if any(token.lower() in text.lower() for token in FORBIDDEN):
        raise SystemExit(f"refusing to open a protected path: {text}")
    return path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with guard(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --- verbatim copies of the author's name normalisation (dataset.py L66-80) -------------------------------
def convert_cell_name(cell):
    cell = cell.replace('-', '_').replace('/ATCC', '')
    cell = cell.replace(' ', '').replace('(TB)', '').replace('MDA_MB_468', 'MDA_N').replace('NCI/', 'NCI_')
    cell = cell.replace('RXF393', 'RXF_393').replace('T_47D', 'T47D')
    return cell


def normalize_drug_name(drug_name):
    if drug_name[-1] == ' ':
        drug_name = drug_name[:-1]
    return drug_name


def q(values, ps=(0, 5, 25, 50, 75, 95, 100)):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return {f"p{p}": round(float(np.percentile(values, p)), 4) for p in ps}


def spearman(x, y):
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


def section_files():
    proto = json.loads(guard(REPRO_PROTOCOL).read_text(encoding="utf-8"))["inputs_sha256"]
    out = {}
    for name, path in (("gex", GEX), ("emb", EMB)):
        digest = sha256(path)
        recorded = proto[f"data/features/{path.name}"]
        out[name] = dict(path=str(path), bytes=path.stat().st_size, sha256=digest,
                         matches_repro_v2_protocol=digest == recorded)
    return out


def section_gex():
    g = pd.read_csv(guard(GEX))
    genes = g.iloc[:, 0].astype(str)
    x = g.iloc[:, 1:].astype(float)
    values = x.to_numpy()
    finite = values[np.isfinite(values)]
    nan_by_cell = {c: int(x[c].isna().sum()) for c in x.columns if x[c].isna().any()}
    almanac_cols = [c for c in x.columns if c != "MDA_N"]
    per_gene = {}
    for label, cols in (("all_60_columns", list(x.columns)), ("59_without_MDA_N", almanac_cols)):
        sub = x[cols]
        per_gene[label] = dict(mean=q(sub.mean(axis=1, skipna=True)), std_ddof1=q(sub.std(axis=1, ddof=1, skipna=True)),
                               share_abs_mean_below_0_05=round(float((sub.mean(axis=1).abs() < 0.05).mean()), 4))
    return dict(
        shape_genes_by_cells=[int(x.shape[0]), int(x.shape[1])], first_column=g.columns[0],
        genes_unique=int(genes.nunique()), first_genes=genes.head(8).tolist(), last_genes=genes.tail(3).tolist(),
        columns=list(x.columns), has_MDA_N=bool("MDA_N" in x.columns), has_MDA_MB_468=bool("MDA_MB_468" in x.columns),
        nan_total=int(np.isnan(values).sum()), nan_by_cell=nan_by_cell,
        genes_with_nan=int(x.isna().any(axis=1).sum()),
        share_values_with_2_decimals=round(float(np.mean(np.isclose(finite, np.round(finite, 2), atol=1e-9))), 6),
        value_range=[float(finite.min()), float(finite.max())], value_mean=float(finite.mean()), value_std=float(finite.std()),
        per_gene_across_cells=per_gene,
        per_cell_mean=q(x.mean(axis=0)), per_cell_std=q(x.std(axis=0)),
    ), g


def section_emb():
    emb = torch.load(guard(EMB), map_location="cpu", weights_only=True)
    if not isinstance(emb, dict):
        raise SystemExit(f"unexpected embedding container: {type(emb)}")
    keys = list(emb.keys())
    shapes = sorted({tuple(v.shape) for v in emb.values()})
    dtypes = sorted({str(v.dtype) for v in emb.values()})
    e = np.stack([v.detach().cpu().numpy().reshape(-1) for v in emb.values()]).astype(np.float64)
    groups = {}
    for k, row in zip(keys, e):
        groups.setdefault(row.tobytes(), []).append(k)
    duplicates = [ks for ks in groups.values() if len(ks) > 1]
    unit = e / np.linalg.norm(e, axis=1, keepdims=True)
    centred = e - e.mean(axis=0, keepdims=True)
    unit_c = centred / np.linalg.norm(centred, axis=1, keepdims=True)
    iu = np.triu_indices(len(e), 1)
    return dict(
        container=type(emb).__name__, n_keys=len(keys), key_types=sorted({type(k).__name__ for k in keys}),
        value_shapes=[list(s) for s in shapes], dtypes=dtypes, nan=int(np.isnan(e).sum()),
        norm=q(np.linalg.norm(e, axis=1)), value_range=[float(e.min()), float(e.max())],
        exact_duplicate_groups=duplicates,
        mean_cosine_raw=round(float((unit @ unit.T)[iu].mean()), 4),
        mean_cosine_after_centering=round(float((unit_c @ unit_c.T)[iu].mean()), 4),
        keys_with_trailing_space=[k for k in keys if k != k.strip()],
    ), emb, keys


def section_coverage(gex_columns, emb_keys):
    cells_used = pd.read_csv(guard(S1 / "cell_mapping_used.csv"), keep_default_na=False)
    drugs_used = pd.read_csv(guard(S1 / "drug_mapping_used.csv"), keep_default_na=False)
    emb_set, gex_set = set(emb_keys), set(gex_columns)
    result, drug_rows, cell_rows = {}, {}, {}
    for role in ("train", "validation"):
        f = pd.read_parquet(guard(S1 / f"points_{role}.parquet"), columns=ID_COLUMNS)
        mats = f.drop_duplicates("matrix_key")
        cells = mats[["deepsynba_cell", "cell_id", "biological_cell_id"]].drop_duplicates()
        drugs = pd.concat([mats[["deepsynba_drug_a", "drug_a_id"]].set_axis(["deepsynba_drug", "drug_id"], axis=1),
                           mats[["deepsynba_drug_b", "drug_b_id"]].set_axis(["deepsynba_drug", "drug_id"], axis=1)]
                          ).drop_duplicates()
        cells["gex_column"] = cells.deepsynba_cell.map(convert_cell_name)
        drugs["emb_key"] = drugs.deepsynba_drug.map(normalize_drug_name)
        miss_cells = sorted(set(cells.gex_column) - gex_set)
        miss_drugs = sorted(set(drugs.emb_key) - emb_set)
        sf539 = mats.deepsynba_cell.eq("SF-539")
        result[role] = dict(
            matrices=int(len(mats)), points=int(len(f)),
            cells=int(cells.cell_id.nunique()), drugs=int(drugs.drug_id.nunique()),
            cells_with_gex=int(cells.gex_column.isin(gex_set).sum()), drugs_with_embedding=int(drugs.emb_key.isin(emb_set).sum()),
            missing_cells=miss_cells, missing_drugs=miss_drugs,
            cell_id_to_gex_one_to_one=bool(cells.groupby("cell_id").gex_column.nunique().max() == 1
                                           and cells.groupby("gex_column").cell_id.nunique().max() == 1),
            drug_id_to_emb_one_to_one=bool(drugs.groupby("drug_id").emb_key.nunique().max() == 1
                                           and drugs.groupby("emb_key").drug_id.nunique().max() == 1),
            matrices_missing_any_feature=int((~(mats.deepsynba_cell.map(convert_cell_name).isin(gex_set)
                                                & mats.deepsynba_drug_a.map(normalize_drug_name).isin(emb_set)
                                                & mats.deepsynba_drug_b.map(normalize_drug_name).isin(emb_set))).sum()),
            sf539_matrices=int(sf539.sum()), sf539_points=int(f.deepsynba_cell.eq("SF-539").sum()),
            sr_matrices=int(mats.deepsynba_cell.eq("SR").sum()),
            rpmi_8226_matrices=int(mats.deepsynba_cell.eq("RPMI-8226").sum()),
            mda_mb_468_present=bool(mats.deepsynba_cell.eq("MDA-MB-468").any()),
        )
        for r in drugs.itertuples(index=False):
            drug_rows.setdefault(r.drug_id, set()).add(r.emb_key)
        for r in cells.itertuples(index=False):
            cell_rows.setdefault(r.cell_id, set()).add(r.gex_column)
    union_drugs = {k: sorted(v) for k, v in drug_rows.items()}
    union_cells = {k: sorted(v) for k, v in cell_rows.items()}
    mapped_names = {normalize_drug_name(n) for n in drugs_used.deepsynba_drug}
    renamed_cells = {r.deepsynba_cell: convert_cell_name(r.deepsynba_cell) for r in cells_used.itertuples(index=False)
                     if convert_cell_name(r.deepsynba_cell) != r.deepsynba_cell}
    result["union_train_validation"] = dict(
        drugs=len(union_drugs), cells=len(union_cells),
        drugs_mapping_unique=all(len(v) == 1 for v in union_drugs.values()),
        cells_mapping_unique=all(len(v) == 1 for v in union_cells.values()),
    )
    result["s1_mapping_files"] = dict(
        cell_rows=int(len(cells_used)), cell_status=cells_used.status.value_counts().to_dict(),
        drug_rows=int(len(drugs_used)), drugs_mapped=int(drugs_used.mapped.astype(str).eq("True").sum()),
        alias_merged=drugs_used.loc[drugs_used.alias_merged.astype(str).eq("True"), ["deepsynba_drug", "canonical_drug_id"]].values.tolist(),
        names_with_trailing_space=[n for n in drugs_used.deepsynba_drug if n != n.rstrip()],
        excluded_cell_author_column=convert_cell_name("MDA-MB-468"),
        cell_names_rewritten_by_author_rule=len(renamed_cells), cell_rename_examples=dict(list(renamed_cells.items())[:12]),
        embedding_keys_not_in_s1_drug_names=sorted(set(emb_keys) - mapped_names),
        gex_columns_not_in_s1_cells=sorted(gex_set - {convert_cell_name(c) for c in cells_used.deepsynba_cell}),
    )
    return result, cells_used, drugs_used


def section_drug_identity(emb, drugs_used):
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    table = pd.read_csv(guard(DRUGS), keep_default_na=False).set_index("drug_id")
    rows = drugs_used[drugs_used.mapped.astype(str).eq("True")].reset_index(drop=True)
    names = rows.deepsynba_drug.tolist()
    smiles = [table.loc[cid, "canonical_smiles"] for cid in rows.canonical_drug_id]
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles]
    n = len(fps)
    tan = np.array([DataStructs.BulkTanimotoSimilarity(fp, fps) for fp in fps])
    e = np.stack([emb[normalize_drug_name(nm)].numpy().reshape(-1) for nm in names]).astype(np.float64)
    ec = e - e.mean(axis=0, keepdims=True)
    unit = ec / np.linalg.norm(ec, axis=1, keepdims=True)
    cos = unit @ unit.T
    iu = np.triu_indices(n, 1)
    observed = spearman(tan[iu], cos[iu])
    rng = np.random.default_rng(RNG_SEED)
    null = []
    for _ in range(1000):
        p = rng.permutation(n)
        null.append(spearman(tan[iu], cos[np.ix_(p, p)][iu]))
    null = np.asarray(null)
    off = ~np.eye(n, dtype=bool)
    tan_nn = np.where(off, tan, -np.inf).argmax(1)
    cos_rank = (-np.where(off, cos, -np.inf)).argsort(1).argsort(1) + 1      # rank 1 = nearest by cosine
    nn_rank = cos_rank[np.arange(n), tan_nn]
    analogs = [("Imatinib mesylate", "Nilotinib"), ("Sirolimus", "Everolimus"), ("Thalidomide", "Lenalidomide"),
               ("Mercaptopurine", "Thioguanine"), ("Etoposide", "Teniposide"), ("Vinblastine sulfate", "Vincristine sulfate"),
               ("Doxorubicin hydrochloride", "Daunorubicin hydrochloride"), ("Paclitaxel", "Docetaxel"),
               ("Topotecan hydrochloride", "7-Ethyl-10-hydroxycamptothecin"), ("Gefitinib", "Erlotinib hydrochloride"),
               ("Letrozole", "Anastrozole"), ("Methotrexate", "pralatrexate"), ("Cisplatin", "Carboplatin"),
               ("Carmustine", "Lomustine"), ("Cytarabine hydrochloride", "Gemcitabine hydrochloride"),
               ("Cladribine", "Clofarabine"), ("Azacitidine", "Decitabine")]
    index = {nm: i for i, nm in enumerate(names)}
    pairs = []
    for a, b in analogs:
        i, j = index[a], index[b]
        pairs.append(dict(pair=f"{a} / {b}", tanimoto=round(float(tan[i, j]), 3), cosine_rank_of_partner=int(cos_rank[i, j]),
                          reverse_rank=int(cos_rank[j, i])))
    ranks = np.array([min(p["cosine_rank_of_partner"], p["reverse_rank"]) for p in pairs])
    return dict(
        drugs=n, fingerprint="Morgan r=2, 2048 bits (canonical_smiles of the S1 canonical IDs)",
        embedding_similarity="cosine after per-dimension centering over the 101 drugs",
        spearman_tanimoto_vs_cosine=round(observed, 4),
        permutation_null=dict(n=1000, mean=round(float(null.mean()), 4), sd=round(float(null.std()), 4),
                              max=round(float(null.max()), 4), p_one_sided=float((1 + (null >= observed).sum()) / 1001)),
        tanimoto_nearest_neighbour_cosine_rank=dict(median=float(np.median(nn_rank)), share_top1=round(float(np.mean(nn_rank == 1)), 3),
                                                    share_top5=round(float(np.mean(nn_rank <= 5)), 3),
                                                    chance_top5=round(5 / (n - 1), 3)),
        known_analog_pairs=pairs, analog_best_rank_median=float(np.median(ranks)),
        analog_best_rank_top3_share=round(float(np.mean(ranks <= 3)), 3),
    )


def section_cell_identity(g, cells_used):
    cm = pd.read_parquet(guard(CELLMINER))
    mapped = cells_used[cells_used.status.eq("mapped")]
    gex_cols = [convert_cell_name(c) for c in mapped.deepsynba_cell]
    cm_cols = mapped.biological_cell_id.tolist()
    missing_cm = [c for c in cm_cols if c not in cm.columns]
    genes = g.iloc[:, 0].astype(str).str.upper()
    a = g.set_index(genes).iloc[:, 0:].loc[:, gex_cols]
    cm = cm.assign(_sym=cm["Gene name"].astype(str).str.upper())
    dup = int(cm._sym.duplicated().sum())
    b = cm.drop_duplicates("_sym").set_index("_sym").loc[:, cm_cols]
    common = [s for s in a.index if s in b.index]
    a, b = a.loc[common].astype(float), b.loc[common].astype(float)
    keep = a.notna().all(axis=1) & b.notna().all(axis=1) & (b.std(axis=1) > 0)
    a, b = a[keep], b[keep]
    za = a.sub(a.mean(axis=1), axis=0).div(a.std(axis=1, ddof=0), axis=0).to_numpy()
    zb = b.sub(b.mean(axis=1), axis=0).div(b.std(axis=1, ddof=0), axis=0).to_numpy()
    per_gene = np.mean(za * zb, axis=1)                                     # Pearson across the 59 cells, per gene
    za_c = (za - za.mean(0)) / za.std(0)
    zb_c = (zb - zb.mean(0)) / zb.std(0)
    r = (za_c.T @ zb_c) / za.shape[0]                                       # 59 x 59: author column vs CellMiner column
    n = r.shape[0]
    self_r = np.diag(r)
    best_other = np.where(~np.eye(n, dtype=bool), r, -np.inf).max(1)
    top1 = r.argmax(1) == np.arange(n)
    worst = sorted(zip(self_r, gex_cols), key=lambda t: t[0])[:3]
    within = (za_c.T @ za_c) / za.shape[0]                                  # author column vs author column
    within_cm = (zb_c.T @ zb_c) / zb.shape[0]                               # CellMiner column vs CellMiner column
    known = [("SR", "RPMI_8226"), ("NCI_ADR_RES", "OVCAR_8"), ("SNB_19", "U251"), ("MDA_MB_435", "M14")]
    same_source_pairs = {f"{x} / {y}": dict(author_file=round(float(within[gex_cols.index(x), gex_cols.index(y)]), 3),
                                            cellminer_rnaseq=round(float(within_cm[gex_cols.index(x), gex_cols.index(y)]), 3))
                         for x, y in known}
    flagged = sorted({gex_cols[i] for i in np.where(~top1)[0]} | {c for _, c in worst})
    detail = {}
    for c in flagged:
        i = gex_cols.index(c)
        to_cm = np.argsort(-r[i])[:3]
        from_cm = np.argsort(-r[:, i])[:3]
        own = np.argsort(-np.where(np.arange(n) == i, -np.inf, within[i]))[:2]
        detail[c] = dict(self_r=round(float(r[i, i]), 3),
                         author_column_best_cellminer=[(cm_cols[j], round(float(r[i, j]), 3)) for j in to_cm],
                         cellminer_column_best_author=[(gex_cols[j], round(float(r[j, i]), 3)) for j in from_cm],
                         author_column_nearest_author_columns=[(gex_cols[j], round(float(within[i, j]), 3)) for j in own])
    return dict(
        cellminer_matrix="data/processed/cellminer/expression.parquet (RNA-seq composite, frozen)",
        cellminer_duplicate_symbols_dropped=dup, cells=n, missing_cellminer_columns=missing_cm,
        landmark_genes_in_cellminer=len(common), genes_used_complete=int(keep.sum()),
        per_gene_cross_source_pearson=q(per_gene),
        cell_self_match_top1=round(float(top1.mean()), 4), self_r=q(self_r), best_other_r=q(best_other),
        lowest_self_r=[(c, round(float(v), 3)) for v, c in worst],
        mismatched_cells=[gex_cols[i] for i in np.where(~top1)[0]],
        flagged_cell_detail=detail,
        same_source_pair_r=same_source_pairs,
    )


def main():
    report = {"files": section_files()}
    report["gex"], g = section_gex()
    report["emb"], emb, keys = section_emb()
    report["coverage"], cells_used, drugs_used = section_coverage(list(g.columns[1:]), keys)
    report["drug_identity"] = section_drug_identity(emb, drugs_used)
    report["cell_identity"] = section_cell_identity(g, cells_used)
    report["guards"] = dict(test_rows_read=0, response_values_read=0, files_written=0)
    print(json.dumps(report, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
