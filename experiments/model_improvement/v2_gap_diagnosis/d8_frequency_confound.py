"""V2 gap diagnosis, step D8: are the training-frequency slices confounded by pralatrexate? (VALIDATION only, CPU)

Recomputes the D2 frequency slices (same quintile edges, computed on all validation points) after removing the
18 validation pairs that contain pralatrexate.

  .venv/Scripts/python.exe -I experiments/model_improvement/v2_gap_diagnosis/d8_frequency_confound.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d2_gap_analysis as d2  # noqa: E402

OUT = d2.DIAG / "frequency_confound"


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    v, tr, y, preds, edges, _, _ = d2.load_inputs()
    hsa = d2.add_derived(v, y, preds, edges)
    drug_count = pd.concat([tr.drug_a_id, tr.drug_b_id]).value_counts()
    dmin = np.minimum(v.drug_a_id.map(drug_count).to_numpy(float), v.drug_b_id.map(drug_count).to_numpy(float))
    cell_drug = pd.concat([tr[["deepsynba_cell", "drug_a_id"]].rename(columns={"drug_a_id": "d"}),
                           tr[["deepsynba_cell", "drug_b_id"]].rename(columns={"drug_b_id": "d"})]).value_counts().to_dict()
    cdmin = np.minimum(np.array([cell_drug.get(k, 0) for k in zip(v.deepsynba_cell, v.drug_a_id)], float),
                       np.array([cell_drug.get(k, 0) for k in zip(v.deepsynba_cell, v.drug_b_id)], float))
    prala = ((v.deepsynba_drug_a == "pralatrexate") | (v.deepsynba_drug_b == "pralatrexate")).to_numpy()
    res = {"step": "D8", "test_rows_read": 0, "pralatrexate_points": int(prala.sum())}
    for name, x in (("drug_train_matrices_min_of_pair", dmin), ("cell_drug_train_matrices_min", cdmin)):
        q = np.unique(np.quantile(x, [0, .2, .4, .6, .8, 1.0]))
        lab = pd.cut(x, q, include_lowest=True).astype(str)
        keep = ~prala
        t_all = d2.slice_table(lab, y, hsa, preds, name)
        t_wo = d2.slice_table(lab[keep], y[keep], hsa[keep], {k: p[keep] for k, p in preds.items()}, name + "_without_pralatrexate")
        cols = ["slice", "points", "point_share", "rmse_BS", "rmse_DSB", "share_gap_BS_vs_DSB", "share_gap_BSens_vs_DSB", "rel_gap_in_slice"]
        res[name] = {"all": t_all[cols].to_dict("records"), "without_pralatrexate": t_wo[cols].to_dict("records")}
        low = x <= q[1]
        res[name]["lowest_bin_upper_edge"] = float(q[1])
        res[name]["share_of_lowest_bin_points_that_are_pralatrexate"] = float(prala[low].mean())
        res[name]["share_of_pralatrexate_points_in_lowest_bin"] = float(low[prala].mean())
        t_wo.to_csv(OUT / f"{name}_without_pralatrexate.csv", index=False)
    with (OUT / "d8_summary.json").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(res, indent=1, default=float) + "\n")
    for name in ("drug_train_matrices_min_of_pair", "cell_drug_train_matrices_min"):
        print(name)
        print(pd.DataFrame(res[name]["without_pralatrexate"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
