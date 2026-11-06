"""O'Neil XGBoost + identity codes (method "xgb_id", card E v2 method 2; DEVIATIONS D13), one seed per call.

Port of the NCI final setting experiments/model_improvement/identity_control/run_xgb_identity_final.py (D8):
residual target response - HSA, features of run_gate.xgb_inputs (anchors, dose u, min/product/|diff|, Morgan-512 ->
PCA16 fit on train drugs, pathway activity node_features[:, 0] -> PCA16 fit on train cells) plus categorical drug A,
drug B, cell (run_xgb_identity.identity_inputs, reused unchanged), drug-order augmentation with 1/size weights,
gate.yaml hyper-parameters, GPU, 20000 trees, then selection every 50 trees on validation surface-macro RMSE (no
patience); validation and test predictions average both drug orders. NCI -> O'Neil replacements:
  gate.load_points (S1 NCI tables)       -> common.load_points (runs/oneil_v1/data); surface id = matrix_key
  banks.drug_ids / banks.cell_ids        -> sorted drugs / cells of the train+validation frames (as build_banks)
  pilot.morgan512 (NCI drugs.csv)        -> runs/oneil_v1/drug_features/morgan512.csv (same generator)
  banks.cell_graphs (NCI fold-0 cache)   -> runs/oneil_v1/cell_features/cell_graphs.pt
  pilot.metric (ratio scale)             -> common.surface_macro_percent (%); min_delta 1e-5 ratio -> 1e-3 %
  load_test (NCI split 3)                -> common.load_points("test"), read after selection is final
  npz outputs / marker / test scoring    -> common.write_predictions; no scoring here (joint scorer)

  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe experiments/oneil/baselines/run_xgb_id.py --seed 41
Smoke (requires ONEIL_PRED_ROOT pointing outside runs/oneil_v1/preds): add --max-trees 200
Log: <PRED root>/../logs/xgb_id_seed<S>.log (runs/oneil_v1/logs for the full run).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

common.install_guard()
sys.path.insert(0, str(common.ROOT / "experiments/model_improvement/identity_control"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

import run_xgb_identity as rxi  # noqa: E402  (NCI identity_inputs, run_gate, gate config)

METHOD, MAX_TREES = "xgb_id", 20000
LOGS = common.PRED.parent / "logs"


def banks_and_fingerprints(frames):
    """Stand-in for gate.build_banks + gate.morgan with only the fields run_gate.xgb_inputs reads."""
    frame = pd.concat(frames.values())
    drug_ids = tuple(sorted(set(frame.drug_a_id) | set(frame.drug_b_id)))
    cell_ids = tuple(sorted(set(frame.cell_id)))
    fp = pd.read_csv(common.DRUG / "morgan512.csv").set_index("drug_id").loc[list(drug_ids)]
    graphs = torch.load(common.CELL / "cell_graphs.pt", weights_only=True)
    cell_graphs = tuple(SimpleNamespace(node_features=graphs[c]["node_features"]) for c in cell_ids)
    assert fp.shape == (len(drug_ids), 512) and all(g.node_features.shape == (50, 53) for g in cell_graphs)
    return SimpleNamespace(drug_ids=drug_ids, cell_ids=cell_ids, cell_graphs=cell_graphs), torch.tensor(fp.to_numpy(np.float32))


def predict(booster, types, k, build, frame):
    """Ratio-scale combination prediction at k trees, averaged over both drug orders (NCI predict_frame / 100)."""
    import xgboost as xgb
    hsa = np.minimum(frame.mono_response_a, frame.mono_response_b).to_numpy(float)
    if not k:
        return hsa
    d = [xgb.DMatrix(build(frame, s), feature_types=types, enable_categorical=True) for s in (False, True)]
    return hsa + 0.5 * (booster.predict(d[0], iteration_range=(0, k)) + booster.predict(d[1], iteration_range=(0, k)))


def train_select(frames, build, cfg, seed, device, max_trees, log):
    """run_xgb_identity_final.train_select with common.surface_macro_percent as the selection metric."""
    import xgboost as xgb
    xc = cfg["xgboost"]; tr, va = frames["train"], frames["validation"]
    min_delta = 100 * cfg["training"]["min_delta"]  # NCI threshold is on the ratio scale; the metric here is in %
    start = time.monotonic()
    hsa_tr = np.minimum(tr.mono_response_a, tr.mono_response_b).to_numpy(float)
    hsa_va = np.minimum(va.mono_response_a, va.mono_response_b).to_numpy(float)
    weight = 1.0 / tr.groupby("matrix_key").matrix_key.transform("size").to_numpy(float)
    x = np.vstack([build(tr, False), build(tr, True)])
    types = ["q"] * (x.shape[1] - 3) + ["c"] * 3
    dtrain = xgb.QuantileDMatrix(x, label=np.concatenate([tr.response.to_numpy(float) - hsa_tr] * 2),
                                 weight=np.concatenate([weight, weight]), feature_types=types, enable_categorical=True)
    log(f"TRAIN rows={x.shape[0]} features={x.shape[1]} max_trees={max_trees} device={device}")
    del x
    params = dict(objective="reg:squarederror", eta=xc["learning_rate"], max_depth=10, min_child_weight=1,
                  subsample=xc["subsample"], colsample_bytree=xc["colsample"], tree_method="hist", device=device,
                  seed=seed, nthread=xc["threads"])
    booster = xgb.train(params, dtrain, num_boost_round=max_trees)
    train_seconds = time.monotonic() - start
    log(f"TRAINED {max_trees} trees in {train_seconds:.1f}s ({1000 * train_seconds / max_trees:.1f} ms/tree)")
    del dtrain
    dva = [xgb.DMatrix(build(va, s), feature_types=types, enable_categorical=True) for s in (False, True)]
    base = rxi.run_gate.base_margin(booster)
    margin = [np.zeros(len(va)), np.zeros(len(va))]
    best_k, best = 0, common.surface_macro_percent(va, hsa_va)
    history, best_pred, step = [{"trees": 0, "surface_macro_percent": best}], hsa_va.copy(), xc["eval_every"]
    for k in range(step, max_trees + 1, step):
        for o in (0, 1):
            margin[o] += booster.predict(dva[o], iteration_range=(k - step, k), output_margin=True) - base
        pred = hsa_va + 0.5 * ((base + margin[0]) + (base + margin[1]))
        score = common.surface_macro_percent(va, pred)
        history.append({"trees": k, "surface_macro_percent": score,
                        "pooled_rmse_percent": common.pooled_rmse_percent(va.response, pred)})
        if score < best - min_delta:
            best, best_k, best_pred = score, k, pred.copy()
    direct = predict(booster, types, best_k, build, va)
    if np.max(np.abs(direct - best_pred)) > 1e-5:  # the block-wise sum must equal a direct prediction
        raise RuntimeError("Block-wise validation prediction differs from the direct prediction")
    return booster, types, best_k, history, direct, {
        "seed": seed, "best_trees": best_k, "max_trees": max_trees, "at_cap": best_k == max_trees, "device": device,
        "validation_surface_macro_percent": best, "validation_pooled_rmse_percent": common.pooled_rmse_percent(va.response, direct),
        "base_score": base, "train_seconds": train_seconds, "select_seconds": time.monotonic() - start - train_seconds,
        "params": params}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, choices=common.SEEDS, required=True)
    ap.add_argument("--max-trees", type=int, default=MAX_TREES, help="smoke only; the card fixes 20000")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    smoke = a.max_trees != MAX_TREES or a.device != "cuda"
    if smoke and "ONEIL_PRED_ROOT" not in os.environ:
        raise SystemExit("Non-card settings (smoke) require ONEIL_PRED_ROOT outside runs/oneil_v1/preds")
    existing = [p for p in (common.pred_path(METHOD, a.seed, r) for r in ("validation", "test")) if p.exists()]
    if existing:
        raise SystemExit(f"Refusing to overwrite: {existing}")
    LOGS.mkdir(parents=True, exist_ok=True)
    log_file = (LOGS / f"{METHOD}_seed{a.seed}.log").open("a", encoding="utf-8")

    def log(message):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
        print(line, flush=True); log_file.write(line + "\n"); log_file.flush()

    start = time.monotonic()
    cfg = rxi.gate.load_config(rxi.run_gate.CONFIG)
    cfg["xgboost"]["max_trees"] = a.max_trees  # gate.yaml says 3000 (gate cap); log/meta must show the cap used (D8: 20000)
    log(f"START {METHOD} seed={a.seed} smoke={smoke} pred_root={common.PRED} xgboost={cfg['xgboost']}")
    with threadpool_limits(limits=cfg["xgboost"]["threads"]):
        frames = {r: common.load_points(r) for r in ("train", "validation")}
        banks, fingerprints = banks_and_fingerprints(frames)
        build = rxi.identity_inputs(frames, banks, fingerprints, cfg)
        booster, types, best_k, history, va_pred, m = train_select(frames, build, cfg, a.seed, a.device, a.max_trees, log)
        log(f"SELECTED best_trees={best_k} at_cap={m['at_cap']} val_surface_macro={m['validation_surface_macro_percent']:.4f}% "
            f"val_pooled={m['validation_pooled_rmse_percent']:.4f}% select={m['select_seconds']:.0f}s")
        test = common.load_points("test")  # inputs only; read after this seed's selection is final
        te_pred = predict(booster, types, best_k, build, test)
    meta = {"method": METHOD, "smoke": smoke, **m, "wall_seconds": time.monotonic() - start, "history": history,
            "config": {"xgboost": cfg["xgboost"], "min_delta_percent": 100 * cfg["training"]["min_delta"],
                       "features": "run_gate.xgb_inputs + identity (drug A, drug B, cell categorical)",
                       "drugs": len(banks.drug_ids), "cells": len(banks.cell_ids),
                       "nci_reference": "experiments/model_improvement/identity_control/run_xgb_identity_final.py (D8)"}}
    for role, frame, pred in (("validation", frames["validation"], va_pred), ("test", test, te_pred)):
        path = common.write_predictions(METHOD, a.seed, role, frame[common.KEY].assign(prediction=pred), meta)
        log(f"WROTE {role} {len(frame)} points -> {path}")
    log(f"DONE wall={meta['wall_seconds']:.0f}s")
    log_file.close()
    print(json.dumps({k: meta[k] for k in ("best_trees", "validation_surface_macro_percent", "wall_seconds")}), flush=True)


if __name__ == "__main__":
    main()
