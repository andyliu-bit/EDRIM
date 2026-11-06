"""V2 gap diagnosis, step D3: diagnostic probes (validation only; not candidate models).

1. Entity-level deviation probes. Sparse ridge regressions fitted on TRAINING points only, target
   delta = y - HSA_lookup (percent), evaluated once on validation:
     P1  dose-position means                          mu[i,j]
     P2  P1 + drug x dose-level offsets              theta[drug_a, i] + theta[drug_b, j]   (one shared table)
     P3  P2 + drug x cell offsets                    eta[drug_a, cell] + eta[drug_b, cell]
     P4  P3 + cell x dose-position offsets            kappa[cell, i, j]
   Every val drug and every val (drug, cell) occurs in training; no val pair does. A fixed ridge penalty is used
   (lambda = 100 squared-error units per coefficient; sensitivity 10 / 1000 reported, not selected).
   The probes ask how much of the BiSurfGNN-DeepSynBa gap is reachable with additive per-entity deviations alone.
2. Drug- and pair-level tracking: per drug / per pair mean true deviation vs each model's mean correction.
3. BiSurfGNN checkpoint-selection metric: best pooled-RMSE epoch vs the selected surface-macro epoch.

  .venv/Scripts/python.exe -I experiments/model_improvement/v2_gap_diagnosis/d3_probes.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsqr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d2_gap_analysis as d2  # noqa: E402  (same loaders and alignment checks)

OUT_DEFAULT = d2.DIAG / "probes"
LAMBDAS = (10.0, 100.0, 1000.0)
MAIN_LAMBDA = 100.0


def design(frame, cells, drugs, blocks):
    """Sparse one-hot design. Drug blocks use one shared table for both arms (exchange symmetric in identity)."""
    n = len(frame)
    cols, rows, offset, names = [], [], 0, []
    r = np.arange(n)
    di = {d: k for k, d in enumerate(drugs)}
    ci = {c: k for k, c in enumerate(cells)}
    da = frame.drug_a_id.map(di).to_numpy()
    db = frame.drug_b_id.map(di).to_numpy()
    cc = frame.deepsynba_cell.map(ci).to_numpy()
    i, j = frame.row_i.to_numpy() - 1, frame.col_j.to_numpy() - 1
    if np.isnan(da.astype(float)).any() or np.isnan(db.astype(float)).any() or np.isnan(cc.astype(float)).any():
        raise RuntimeError("Unknown drug or cell in the design")
    for block in blocks:
        if block == "position":
            rows += [r]; cols += [offset + i * 3 + j]; offset += 9
        elif block == "drug_dose":
            rows += [r, r]; cols += [offset + da * 3 + i, offset + db * 3 + j]; offset += 3 * len(drugs)
        elif block == "drug_cell":
            rows += [r, r]; cols += [offset + da * len(cells) + cc, offset + db * len(cells) + cc]
            offset += len(drugs) * len(cells)
        elif block == "cell_position":
            rows += [r]; cols += [offset + cc * 9 + i * 3 + j]; offset += 9 * len(cells)
        names.append(block)
    rr, cc_ = np.concatenate(rows), np.concatenate(cols)
    return sparse.csr_matrix((np.ones(len(rr)), (rr, cc_)), shape=(n, offset))


def fit_probe(xtr, dtr, lam):
    mean = float(dtr.mean())
    sol = lsqr(xtr, dtr - mean, damp=np.sqrt(lam), atol=1e-10, btol=1e-10, iter_lim=20000)
    return sol[0], mean, int(sol[2])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    out = parser.parse_args().out
    out.mkdir(parents=True, exist_ok=False)
    v, _, y, preds, edges, _, _ = d2.load_inputs()
    hsa = d2.add_derived(v, y, preds, edges)
    manifest = json.loads((d2.S1 / "manifest.json").read_text(encoding="utf-8"))
    if d2.sha256(d2.S1 / "points_train.parquet") != manifest["points_train.parquet"]:
        raise RuntimeError("training table changed")
    tr = pd.read_parquet(d2.S1 / "points_train.parquet",
                         columns=["drug_a_id", "drug_b_id", "deepsynba_cell", "row_i", "col_j", "response_percent",
                                  "hsa_lookup_percent", "pair_id"])
    dtr = (tr.response_percent - tr.hsa_lookup_percent).to_numpy(float)
    drugs = sorted(set(tr.drug_a_id) | set(tr.drug_b_id))
    cells = sorted(set(tr.deepsynba_cell))
    if not (set(v.drug_a_id) | set(v.drug_b_id)) <= set(drugs) or not set(v.deepsynba_cell) <= set(cells):
        raise RuntimeError("validation entity missing from training")
    if v.pair_id.isin(set(tr.pair_id)).any():
        raise RuntimeError("validation pair present in training")
    specs = {"P1": ["position"], "P2": ["position", "drug_dose"], "P3": ["position", "drug_dose", "drug_cell"],
             "P4": ["position", "drug_dose", "drug_cell", "cell_position"]}
    e2 = lambda p: (p - y) ** 2  # noqa: E731
    bs_sse = np.mean([e2(preds[k]) for k in d2.BS_SEEDS], axis=0)
    dsb_sse = e2(preds["DSB_e484"])
    total_gap = float(np.sum(bs_sse - dsb_sse))
    rows, probe_preds = [], {}
    for name, blocks in specs.items():
        xtr = design(tr, cells, drugs, blocks)
        xva = design(v, cells, drugs, blocks)
        for lam in LAMBDAS:
            coef, mean, iters = fit_probe(xtr, dtr, lam)
            ptr = tr.hsa_lookup_percent.to_numpy(float) + mean + xtr @ coef
            pva = hsa + mean + xva @ coef
            rows.append(dict(probe=name, blocks="+".join(blocks), lam=lam, coefficients=xtr.shape[1], lsqr_iterations=iters,
                             train_rmse=d2.rmse(ptr - tr.response_percent.to_numpy(float)), val_rmse=d2.rmse(pva - y),
                             val_rel_to_DSB=d2.rmse(pva - y) / d2.rmse(preds["DSB_e484"] - y) - 1.0,
                             val_rel_to_BS_mean_seed=d2.rmse(pva - y) / np.mean([d2.rmse(preds[k] - y) for k in d2.BS_SEEDS]) - 1.0,
                             share_of_BS_DSB_gap_closed_if_replacing_BS=float(np.sum(bs_sse - e2(pva)) / total_gap)))
            if lam == MAIN_LAMBDA:
                probe_preds[name] = pva
        print(rows[-2], flush=True)
    probes = pd.DataFrame(rows)
    probes.to_csv(out / "entity_probes.csv", index=False)

    # complementarity of the entity probe with the learned models (cross-fitted blend on validation pairs)
    pairs = v.pair_id.to_numpy()
    blends = {f"BS_ens5+{k}": d2.blend_crossfit(y, preds["BS_ens5"], p, pairs) for k, p in probe_preds.items()}
    blends.update({f"DSB_e484+{k}": d2.blend_crossfit(y, preds["DSB_e484"], p, pairs) for k, p in probe_preds.items()})
    blends.update({f"XGB_ens5+{k}": d2.blend_crossfit(y, preds["XGB_ens5"], p, pairs) for k, p in probe_preds.items()})

    # per-slice view of the main probe (P3, lambda 100) on the slices where the gap sits
    dev = y - hsa
    dev_bin = d2.bins(dev, [-np.inf, -30, -10, -3, 3, 10, np.inf], ["<-30", "-30..-10", "-10..-3", "-3..3", "3..10", ">=10"])
    slice_rows = []
    for lab in ["<-30", "-30..-10", "-10..-3", "-3..3", "3..10", ">=10"]:
        m = dev_bin == lab
        slice_rows.append(dict(slice=lab, points=int(m.sum()), rmse_BS=float(np.sqrt(bs_sse[m].mean())),
                               rmse_DSB=float(np.sqrt(dsb_sse[m].mean())),
                               rmse_P2=float(np.sqrt(e2(probe_preds["P2"])[m].mean())),
                               rmse_P3=float(np.sqrt(e2(probe_preds["P3"])[m].mean())),
                               corr_P3=float((probe_preds["P3"] - hsa)[m].mean()), true_dev=float(dev[m].mean())))

    # drug- and pair-level tracking of systematic deviation
    tr_dev = pd.DataFrame({"d": np.concatenate([tr.drug_a_id, tr.drug_b_id]), "dev": np.concatenate([dtr, dtr])})
    train_drug_dev = tr_dev.groupby("d").dev.mean()
    long = pd.DataFrame({"d": np.concatenate([v.drug_a_id, v.drug_b_id]), "dev": np.concatenate([dev, dev])})
    models = ["BS_ens5", "XGB_ens5", "DSB_e484", "DSB_e047"]
    for k in models:
        long[k] = np.concatenate([preds[k] - hsa] * 2)
    long["P3"] = np.concatenate([probe_preds["P3"] - hsa] * 2)
    per_drug = long.groupby("d").mean()
    per_drug["train_dev"] = train_drug_dev.reindex(per_drug.index)
    drug_track = {"corr_val_dev_vs_train_dev": float(np.corrcoef(per_drug.dev, per_drug.train_dev)[0, 1])}
    for k in models + ["P3"]:
        drug_track[k] = dict(corr=float(np.corrcoef(per_drug.dev, per_drug[k])[0, 1]),
                             rmse_of_drug_means=float(np.sqrt(np.mean((per_drug.dev - per_drug[k]) ** 2))),
                             slope_true_on_pred=float(np.cov(per_drug[k], per_drug.dev)[0, 1] / np.var(per_drug[k], ddof=1)))
    per_drug.round(4).to_csv(out / "per_drug_mean_deviation.csv")
    pv = pd.DataFrame({"pair": pairs, "dev": dev, **{k: preds[k] - hsa for k in models}, "P3": probe_preds["P3"] - hsa})
    per_pair = pv.groupby("pair").mean()
    pair_track = {k: dict(corr=float(np.corrcoef(per_pair.dev, per_pair[k])[0, 1]),
                          rmse_of_pair_means=float(np.sqrt(np.mean((per_pair.dev - per_pair[k]) ** 2))),
                          slope_true_on_pred=float(np.cov(per_pair[k], per_pair.dev)[0, 1] / np.var(per_pair[k], ddof=1)))
                  for k in models + ["P3"]}
    pair_track["sd_true_pair_mean_dev"] = float(per_pair.dev.std())
    single = []
    for k in d2.BS_SEEDS:
        m = pd.DataFrame({"pair": pairs, "dev": dev, "c": preds[k] - hsa}).groupby("pair").mean()
        single.append(float(np.corrcoef(m.dev, m.c)[0, 1]))
    pair_track["BS_single_seed_corr"] = dict(zip(d2.BS_SEEDS, single))
    pair_track["BS_single_seed_corr_mean"] = float(np.mean(single))

    # checkpoint-selection metric (surface-macro) vs the evaluation metric (pooled)
    sel = {}
    for tag, path in d2.BISURF.items():
        hist = json.loads((path / "training_progress.json").read_text(encoding="utf-8"))["history"]
        h = [r for r in hist if r["epoch"] >= 0]
        by_macro = min(h, key=lambda r: r["surface_macro_rmse"])
        by_pooled = min(h, key=lambda r: r["pooled_rmse_percent"])
        sel[tag] = dict(selected_epoch=by_macro["epoch"], selected_pooled=by_macro["pooled_rmse_percent"],
                        best_pooled_epoch=by_pooled["epoch"], best_pooled=by_pooled["pooled_rmse_percent"],
                        last_epoch=h[-1]["epoch"], last_pooled=h[-1]["pooled_rmse_percent"])

    result = {"step": "D3", "test_rows_read": 0, "train_points": int(len(tr)), "validation_points": int(len(v)),
              "main_lambda": MAIN_LAMBDA, "probe_blends": blends, "probe_P3_by_deviation_bin": slice_rows,
              "drug_level_tracking": drug_track, "pair_level_tracking": pair_track, "selection_metric": sel}
    with (out / "probes_summary.json").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=1, ensure_ascii=False, default=float) + "\n")
    print(probes.round(4).to_string(index=False))
    print(json.dumps({k: result[k] for k in ("probe_blends", "drug_level_tracking", "pair_level_tracking", "selection_metric")},
                     indent=1, default=float))
    print(pd.DataFrame(slice_rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
