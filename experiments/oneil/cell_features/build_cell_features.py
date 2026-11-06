"""O'Neil cell features for the 30 DepMap cells (launch card E v2, docs/ONEIL_COMPARISON_CARD_20261020.md; D13).

A. 976-landmark vector (DeepSynBa input and Ours' gene projection). Genes and order are those of DeepSynBa's
   NCI-60_landmark_gex.csv; values are DepMap 24Q4 log2(TPM+1) (oneil_depmap_coverage_v1/expression.npz), z-scored
   per gene over the 30 cells (ddof 0; under the pair hold-out every cell is a training cell). A landmark symbol
   absent from DepMap is kept only through a one-to-one Entrez bridge: its CellMiner annotation (local, symbol ->
   Entrez) gives exactly one Entrez ID, that ID labels exactly one DepMap column, and the DepMap symbol is not a
   landmark symbol itself. All others are dropped for every cell. Zero-SD genes are written as NaN, which both
   loaders set to 0, as the NCI file's NaNs are (external_features.load_deepsynba_features, DeepSynBa forward).
   Output: oneil_landmark_gex.csv in DeepSynBa's layout (column ``genes`` = landmark symbol, then one column per
   cell_id in sorted order; convert_cell_name(cell_id) == cell_id), landmark_genes.csv (kept genes and sources).
B. Hallmark pathway graphs: the CellMiner pipeline's train-only transform on a DepMap ExpressionMatrix (gene IDs =
   upper-cased symbols; Entrez has duplicate columns and is not a key), Hallmark v2026.1 gmt, cached
   pathway_topology.pt reused, fit on the 30 cells. Output: cell_graphs.pt {cell_id: {node_features [50, 53],
   adjacency [50, 50]}}, the layout of the NCI fold caches that build_graph_feature_banks reads.

No O'Neil response file is opened; an audit hook refuses the files the card forbids.
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe experiments/oneil/cell_features/build_cell_features.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT.parent / "bisurfgnn"                       # read-only sibling checkout
FORBIDDEN = ("oneil_confirmation_eval_v1", "oneil_per_surface.csv", "external_combokr2_oneil_newcell_v1",
             "oneil_fitted_braids.npy")


def _refuse(event, args):
    if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
        text = os.fsdecode(args[0]).replace("\\", "/").lower()
        if any(token in text for token in FORBIDDEN):
            raise PermissionError(f"refusing to open a card-forbidden path: {text}")


sys.addaudithook(_refuse)
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from bisurfgnn.data.cellminer import (ExpressionMatrix, fit_train_only_pathway_transform, load_cached_pathway_topology,  # noqa: E402
                                      load_cellminer_config, load_hallmark_gene_sets, normalize_cell_label,
                                      normalize_gene_symbol, sha256_file, transform_cells_to_graphs)
from bisurfgnn.data.external_features import convert_cell_name  # noqa: E402

DEPMAP_NPZ = DATA / "runs/audit_revision/oneil_depmap_coverage_v1/expression.npz"
NCI_GEX = DATA / "data/raw/external_benchmarks/deepsynba/data/features/NCI-60_landmark_gex.csv"
CELLMINER_GENES = DATA / "data/processed/cellminer/expression_manifest.json"
CELL_CONFIG = DATA / "configs/cell_context.yaml"
ROLES = DATA / "runs/audit_revision/oneil_pairout_metadata_v1/surface_roles.jsonl"   # IDs and doses only
OUT = ROOT / "runs/oneil_v1/cell_features"
SPLIT_LABEL = "oneil-pairout-v1|30 DepMap 24Q4 cells, all training cells under pair hold-out"


def load_depmap():
    z = np.load(DEPMAP_NPZ, allow_pickle=False)
    cells = [normalize_cell_label(c) for c in z["cells"]]
    order = np.argsort(cells, kind="stable")
    labels = [str(v) for v in z["genes"]]
    symbols, entrez = zip(*(label.rsplit(" (", 1) for label in labels))
    values = z["expression"][order]
    assert values.shape == (30, len(labels)) and np.isfinite(values).all()
    return [cells[i] for i in order], labels, list(symbols), [e.rstrip(")") for e in entrez], values


def landmark_mapping(labels, symbols, entrez):
    landmarks = pd.read_csv(NCI_GEX, usecols=["genes"])["genes"].astype(str).tolist()
    assert len(landmarks) == len(set(landmarks)) == 976
    column = {s: i for i, s in enumerate(symbols)}
    assert len(column) == len(symbols), "DepMap symbols are not unique"
    by_entrez = defaultdict(list)
    for i, e in enumerate(entrez):
        by_entrez[e].append(i)
    manifest = json.loads(CELLMINER_GENES.read_text(encoding="utf-8"))
    cellminer = defaultdict(set)
    for s, e in zip(manifest["gene_ids"], manifest["entrez_ids"]):
        if e:
            cellminer[s].add(e)
    kept, dropped = [], []
    for g in landmarks:
        if g in column:
            kept.append(dict(landmark_symbol=g, depmap_label=labels[column[g]], method="symbol", column=column[g]))
            continue
        ids = cellminer.get(g, set())
        cols = by_entrez[next(iter(ids))] if len(ids) == 1 else []
        if len(cols) == 1 and symbols[cols[0]] not in landmarks:
            kept.append(dict(landmark_symbol=g, depmap_label=labels[cols[0]], method="cellminer_entrez", column=cols[0]))
        else:
            dropped.append(dict(landmark_symbol=g, cellminer_entrez=sorted(ids), depmap_columns=len(cols)))
    targets = [k["column"] for k in kept]
    assert len(set(targets)) == len(targets), "two landmark symbols map onto one DepMap column"
    return kept, dropped


def gene_table(cells, values, kept):
    x = values[:, [k["column"] for k in kept]]                       # [30, n], log2(TPM+1)
    mean, sd = x.mean(axis=0), x.std(axis=0, ddof=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        zscores = np.where(sd > 0, (x - mean) / sd, np.nan)
    frame = pd.DataFrame(zscores.T, columns=cells)
    frame.insert(0, "genes", [k["landmark_symbol"] for k in kept])
    return frame, int((sd == 0).sum())


def load_gene_rows(cell_ids, path=OUT / "oneil_landmark_gex.csv"):
    """Ours' cell_ext rows: float32 [cells, n] in bank order, NaN -> 0 as load_deepsynba_features does.

    load_deepsynba_features itself is NCI-only (author-file SHA, CELL_DIM 976, S1 mappings), so the O'Neil entry
    calls this instead and sets cell_ext_dim = n; DeepSynBa reads the CSV directly with gene_dim = n.
    """
    frame = pd.read_csv(path)
    values = frame.loc[:, list(cell_ids)].to_numpy(np.float64).T
    assert not np.isinf(values).any()
    return tuple(frame["genes"].astype(str)), torch.as_tensor(np.where(np.isnan(values), 0.0, values).astype(np.float32))


def pathway_graphs(cells, labels, symbols, entrez, values):
    cfg = load_cellminer_config(CELL_CONFIG)
    expression = ExpressionMatrix(
        cell_ids=tuple(cells), gene_ids=tuple(normalize_gene_symbol(s) for s in symbols), entrez_ids=tuple(entrez),
        values=torch.as_tensor(values, dtype=torch.float32), observed_mask=torch.ones(values.shape, dtype=torch.bool),
        source_release="DepMap 24Q4 Public (figshare v1) OmicsExpressionProteinCodingGenesTPMLogp1, 30 O'Neil cells",
        source_sha256=sha256_file(DEPMAP_NPZ), value_scale="log2(TPM+1)")
    catalog = load_hallmark_gene_sets(cfg)
    topology = load_cached_pathway_topology(cfg, expected_catalog_hash=catalog.catalog_hash)
    state = fit_train_only_pathway_transform(expression, catalog, topology, cells, SPLIT_LABEL,
                                             min_pathway_genes=cfg.min_pathway_genes, clip_value=cfg.clip_value,
                                             std_epsilon=cfg.std_epsilon)
    graphs = transform_cells_to_graphs(expression, topology, state, cells)
    present = set(expression.gene_ids)
    hallmark = {g for genes in catalog.genes for g in genes}
    coverage = dict(hallmark_unique_genes=len(hallmark), found=len(hallmark & present),
                    missing=sorted(hallmark - present),
                    found_per_pathway={p: [len(set(g) & present), len(g)] for p, g in zip(catalog.pathway_ids, catalog.genes)},
                    genes_inactive_zero_sd=int((~state.gene_active).sum()),
                    pathways_inactive=int((~state.pathway_active).sum()))
    binding = dict(state_hash=state.state_hash, expression_hash=expression.content_hash,
                   catalog_hash=catalog.catalog_hash, topology_hash=topology.topology_hash,
                   pathway_ids=list(state.pathway_ids), fit_cell_ids=list(state.fit_cell_ids), split_hash=SPLIT_LABEL,
                   min_pathway_genes=cfg.min_pathway_genes, clip_value=cfg.clip_value, std_epsilon=cfg.std_epsilon,
                   topology_file=str(cfg.processed_dir / "pathway_topology.pt"))
    return graphs, coverage, binding


def main():
    cells, labels, symbols, entrez, values = load_depmap()
    roles = [json.loads(line) for line in ROLES.read_text(encoding="utf-8").splitlines()]
    train_cells = {normalize_cell_label(r["cell"]) for r in roles if r["role"] == "train"}
    assert len(cells) == 30 and set(cells) <= train_cells, "every DepMap cell must be a training cell"
    assert all(convert_cell_name(c) == c for c in cells), "DeepSynBa's convert_cell_name would rename a cell"

    kept, dropped = landmark_mapping(labels, symbols, entrez)
    frame, zero_sd = gene_table(cells, values, kept)
    graphs, coverage, binding = pathway_graphs(cells, labels, symbols, entrez, values)

    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "oneil_landmark_gex.csv", index=False)
    pd.DataFrame([{k: v for k, v in row.items() if k != "column"} | {"position": i} for i, row in enumerate(kept)]
                 ).to_csv(OUT / "landmark_genes.csv", index=False)
    torch.save({c: {"node_features": g.node_features, "adjacency": g.adjacency} for c, g in graphs.items()},
               OUT / "cell_graphs.pt")
    record = dict(
        card="docs/ONEIL_COMPARISON_CARD_20261020.md (v2), D13", cell_ids=cells,
        sources=dict(depmap=str(DEPMAP_NPZ), landmark_genes=str(NCI_GEX), cellminer_entrez=str(CELLMINER_GENES),
                     cell_config=str(CELL_CONFIG), roles=str(ROLES)),
        landmark=dict(source_genes=976, kept=len(kept), by_symbol=sum(k["method"] == "symbol" for k in kept),
                      by_cellminer_entrez=sum(k["method"] == "cellminer_entrez" for k in kept), dropped=dropped,
                      zscore="per gene over the 30 cells, ddof 0, on DepMap log2(TPM+1)", zero_sd_genes_written_nan=zero_sd,
                      renamed={k["landmark_symbol"]: k["depmap_label"] for k in kept if k["method"] != "symbol"}),
        hallmark=coverage | binding | dict(node_feature_shape=list(next(iter(graphs.values())).node_features.shape)),
        responses_read=0)
    (OUT / "cell_features_record.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(f"cells={len(cells)} genes_kept={len(kept)} (symbol {record['landmark']['by_symbol']}, "
          f"entrez bridge {record['landmark']['by_cellminer_entrez']}) dropped={[d['landmark_symbol'] for d in dropped]} "
          f"zero_sd={zero_sd} hallmark_found={coverage['found']}/{coverage['hallmark_unique_genes']} "
          f"node_features={record['hallmark']['node_feature_shape']}")


if __name__ == "__main__":
    main()
