"""Self-check of runs/oneil_v1/cell_features (build_cell_features.py): shapes, gene count, NaNs, per-cell stats.

Recomputes the z-scores from expression.npz through landmark_genes.csv, reads the CSV the way DeepSynBa's
get_gene_expressions and Ours' loader do, and passes the graphs through _as_graph_item / GraphFeatureBanks.
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe experiments/oneil/cell_features/check_cell_features.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from bisurfgnn.independent_graph_benchmark import GraphFeatureBanks, _as_graph_item  # noqa: E402
from build_cell_features import (CELL_CONFIG, CELLMINER_GENES, DEPMAP_NPZ, NCI_GEX, OUT,  # noqa: E402  (also installs
                                 load_gene_rows)                                         # the audit hook)
from bisurfgnn.data.cellminer import load_cellminer_config, normalize_cell_label  # noqa: E402
from bisurfgnn.data.external_features import convert_cell_name  # noqa: E402


def main():
    record = json.loads((OUT / "cell_features_record.json").read_text(encoding="utf-8"))
    cells = record["cell_ids"]
    gex = pd.read_csv(OUT / "oneil_landmark_gex.csv")
    genes = pd.read_csv(OUT / "landmark_genes.csv")
    n = len(genes)
    print(f"[A] csv shape {gex.shape}  genes n={n}  first column {gex.columns[0]!r}")
    assert gex.columns[0] == "genes" and gex.shape == (n, 31) and list(gex.columns[1:]) == cells == sorted(cells)
    assert all(convert_cell_name(c) == c for c in cells)
    nci = pd.read_csv(NCI_GEX, usecols=["genes"])["genes"].tolist()
    assert gex["genes"].tolist() == genes["landmark_symbol"].tolist() == [g for g in nci if g in set(genes["landmark_symbol"])]
    print(f"    gene order = NCI order minus {976 - n}: {sorted(set(nci) - set(gex['genes']))}")
    print(f"    methods {genes['method'].value_counts().to_dict()}")

    # DeepSynBa dataset.get_gene_expressions + forward nan_to_num; Ours: load_gene_rows in bank (sorted) order
    cell2gex = {c: gex[c].values.astype("float32") for c in gex.columns[1:].values}
    ours_genes, ours = load_gene_rows(cells)
    print(f"    Ours rows {tuple(ours.shape)} {ours.dtype}  DeepSynBa cell2gex {len(cell2gex)} x {len(cell2gex[cells[0]])}")
    assert ours_genes == tuple(gex["genes"]) and ours.shape == (30, n) and bool(torch.isfinite(ours).all())
    assert all(np.array_equal(np.nan_to_num(cell2gex[c]), ours[i].numpy()) for i, c in enumerate(cells))

    # every kept landmark symbol has the same Entrez ID in CellMiner as its DepMap column (no reassigned symbols)
    manifest = json.loads(CELLMINER_GENES.read_text(encoding="utf-8"))
    cm = {}
    for s, e in zip(manifest["gene_ids"], manifest["entrez_ids"]):
        cm.setdefault(s, set()).add(e)
    mismatch = [(g, l) for g, l in zip(genes["landmark_symbol"], genes["depmap_label"])
                if l.rsplit(" (", 1)[1].rstrip(")") not in cm.get(g, set())]
    print(f"    Entrez agreement CellMiner(landmark symbol) vs DepMap column: {n - len(mismatch)}/{n}")
    assert not mismatch, mismatch

    z = np.load(DEPMAP_NPZ, allow_pickle=False)
    order = [list(map(normalize_cell_label, z["cells"])).index(c) for c in cells]
    labels = [str(v) for v in z["genes"]]
    x = z["expression"][order][:, [labels.index(l) for l in genes["depmap_label"]]]
    expected = (x - x.mean(0)) / x.std(0)
    values = gex.loc[:, cells].to_numpy(np.float64).T
    print(f"    NaN total {int(np.isnan(values).sum())}  per cell {np.isnan(values).sum(1).tolist()}")
    print(f"    max |csv - recomputed z| = {np.nanmax(np.abs(values - expected)):.3e}")
    print(f"    per-gene mean |max| {np.abs(np.nanmean(values, 0)).max():.2e}  per-gene sd range "
          f"[{np.nanstd(values, 0).min():.6f}, {np.nanstd(values, 0).max():.6f}]")
    assert np.allclose(values, expected, atol=1e-12, equal_nan=True)

    graphs = torch.load(OUT / "cell_graphs.pt", map_location="cpu", weights_only=True)
    items = [_as_graph_item(graphs[c], c) for c in cells]
    topology = torch.load(load_cellminer_config(CELL_CONFIG).processed_dir / "pathway_topology.pt", weights_only=True)
    nodes = torch.stack([g.node_features for g in items])
    print(f"[B] graphs {len(graphs)}  node_features {tuple(items[0].node_features.shape)}  adjacency "
          f"{tuple(items[0].adjacency.shape)}  finite {bool(torch.isfinite(nodes).all())}")
    assert sorted(graphs) == cells and nodes.shape == (30, 50, 53)
    assert all(torch.equal(g.adjacency, topology["adjacency"].float()) for g in items)
    assert torch.equal(nodes[:, :, 3:], torch.eye(50).expand(30, 50, 50)) and bool((nodes[:, :, 2] == nodes[0, :, 2]).all())
    act, frac = nodes[:, :, 0], nodes[:, :, 1]
    print(f"    hallmark genes found {record['hallmark']['found']}/{record['hallmark']['hallmark_unique_genes']}  "
          f"observed fraction range [{frac.min():.4f}, {frac.max():.4f}]  activity |pathway mean| max "
          f"{act.mean(0).abs().max():.2e}  activity clipped at 5: {int((act.abs() >= 5).sum())}")
    bank = GraphFeatureBanks(fold=0, drug_ids=("probe",), drug_structure_keys=("probe",), cell_ids=tuple(cells),
                             biological_cell_ids=tuple(cells), drug_graphs=(items[0],), cell_graphs=tuple(items),
                             drug_graph_hashes=("-",), cell_graph_hashes=tuple("-" for _ in cells),
                             context_state_hash=record["hallmark"]["state_hash"], source_assets={})
    print(f"    GraphFeatureBanks accepts the cells, cell_feature_dim={bank.cell_feature_dim}")

    print("[C] per cell: gene z mean / sd / min / max | pathway activity mean / sd | observed fraction mean")
    for i, c in enumerate(cells):
        v = values[i]
        print(f"    {c:<9} {np.nanmean(v):+.3f} {np.nanstd(v):.3f} {np.nanmin(v):+.2f} {np.nanmax(v):+.2f} | "
              f"{act[i].mean():+.3f} {act[i].std(unbiased=False):.3f} | {frac[i].mean():.4f}")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
