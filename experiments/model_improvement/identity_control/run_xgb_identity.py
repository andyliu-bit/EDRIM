"""Same-input control for BiSurfGNN v2: the gate's XGBoost residual comparator (xgb_residual_d10_m1, run_gate.train_xgb)
plus entity identity, i.e. drug A, drug B and cell index as native categorical features (A/B swapped with the arms).
Everything else is the gate setting: features, swap augmentation, 1/size weights, params, 3000-tree cap, block-wise
selection on validation surface-macro RMSE. Split 1 trains, split 2 evaluates; split 3 is never opened.

Reading (stated before the run): reference points on validation pooled RMSE % are gate XGB 10.74, DeepSynBa v2 9.966,
BiSurfGNN 'both' 9.195 and its random-vector control 9.01.
  * close to ~9.2  -> identity + monotherapy anchors explain the gain; BiSurfGNN's architecture adds little on this split;
  * near 9.97 or above -> BiSurfGNN uses the same identity information better; the architecture claim stands;
  * in between -> partial; report the gap.

  .venv/Scripts/python.exe -I -B experiments/model_improvement/identity_control/run_xgb_identity.py --seeds 41 42 43 44 45
  .venv/Scripts/python.exe -I -B experiments/model_improvement/identity_control/run_xgb_identity.py --seeds 41 --smoke
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/model_improvement/public_split_gate"))
import run_gate  # noqa: E402
import gate_common as gate  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

learned, pilot = gate.learned, run_gate.pilot
OUT = ROOT / "runs/model_improvement/bisurfgnn_v2_dev/xgb_identity"
NAME = "xgb_residual_d10_m1_identity"


def identity_inputs(frames, banks, fingerprints, cfg):
    base = run_gate.xgb_inputs(frames, banks, fingerprints, cfg)
    drug_index = {d: i for i, d in enumerate(banks.drug_ids)}
    cell_index = {c: i for i, c in enumerate(banks.cell_ids)}

    def build(frame, swap):
        a, b = ("b", "a") if swap else ("a", "b")
        ids = np.column_stack([frame[f"drug_{a}_id"].map(drug_index), frame[f"drug_{b}_id"].map(drug_index),
                               frame.cell_id.map(cell_index)]).astype(float)
        if np.isnan(ids).any():
            raise RuntimeError("Unmapped drug or cell id")
        return np.column_stack([base(frame, swap), ids])
    return build


def train_xgb_identity(frames, banks, fingerprints, cfg, seed):
    """run_gate.train_xgb with three categorical identity columns appended."""
    import xgboost as xgb
    xc = cfg["xgboost"]; tr, va = frames["train"], frames["validation"]
    start = time.monotonic()
    build = identity_inputs(frames, banks, fingerprints, cfg)
    hsa_tr = np.minimum(tr.mono_response_a, tr.mono_response_b).to_numpy(float)
    hsa_va = np.minimum(va.mono_response_a, va.mono_response_b).to_numpy(float)
    weight = 1.0 / tr.groupby("surface_replicate_id").surface_replicate_id.transform("size").to_numpy(float)
    x = np.vstack([build(tr, False), build(tr, True)])
    types = ["q"] * (x.shape[1] - 3) + ["c"] * 3
    matrix = lambda data, **k: xgb.DMatrix(data, feature_types=types, enable_categorical=True, **k)  # noqa: E731
    y = np.concatenate([tr.response.to_numpy(float) - hsa_tr] * 2)
    w = np.concatenate([weight, weight])
    params = dict(objective="reg:squarederror", eta=xc["learning_rate"], max_depth=10, min_child_weight=1,
                  subsample=xc["subsample"], colsample_bytree=xc["colsample"], tree_method="hist", seed=seed,
                  nthread=xc["threads"])
    booster = xgb.train(params, matrix(x, label=y, weight=w), num_boost_round=xc["max_trees"])
    train_seconds = time.monotonic() - start
    del x, y, w
    dva = (matrix(build(va, False)), matrix(build(va, True)))
    base = run_gate.base_margin(booster)
    margin = [np.zeros(len(va)), np.zeros(len(va))]
    target, surfaces = va.response.to_numpy(float), va.surface_replicate_id.to_numpy()
    best_k, best = 0, pilot.metric(target, hsa_va, surfaces)["surface_macro_rmse"]
    history = [{"trees": 0, "surface_macro_rmse": best}]
    step = xc["eval_every"]
    for k in range(step, xc["max_trees"] + 1, step):
        for o in (0, 1):
            margin[o] += booster.predict(dva[o], iteration_range=(k - step, k), output_margin=True) - base
        pred = hsa_va + 0.5 * ((base + margin[0]) + (base + margin[1]))
        score = pilot.metric(target, pred, surfaces)["surface_macro_rmse"]
        history.append({"trees": k, "surface_macro_rmse": score, "pooled_rmse_percent": gate.pooled_rmse_percent(target, pred)})
        if score < best - cfg["training"]["min_delta"]:
            best, best_k, best_pred = score, k, pred.copy()
    prediction = hsa_va.copy()
    if best_k:  # direct prediction at the selected size; must match the block-wise sum
        prediction = hsa_va + 0.5 * (booster.predict(dva[0], iteration_range=(0, best_k))
                                     + booster.predict(dva[1], iteration_range=(0, best_k)))
        if np.max(np.abs(prediction - best_pred)) > 1e-5:
            raise RuntimeError("Block-wise validation prediction differs from the direct prediction")
    return prediction, {"best_trees": best_k, "history": history, "params": params, "train_seconds": train_seconds,
                        "wall_seconds": time.monotonic() - start}


def main():
    if not (sys.flags.isolated and sys.flags.dont_write_bytecode):
        raise SystemExit("run with `python -I -B`")
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", choices=(41, 42, 43, 44, 45), required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg = gate.load_config(run_gate.CONFIG)
    with threadpool_limits(limits=cfg["xgboost"]["threads"]):
        frames, banks, tables = run_gate.prepare(cfg)
        if a.smoke:
            frames = run_gate.tiny_frames(frames, cfg["timing"]["subset_seed"] + 2)
            tables = gate.build_tables(frames, banks)
            cfg["xgboost"]["max_trees"] = 100
        fingerprints = gate.morgan(banks)
        va = tables["validation"]
        for seed in a.seeds:
            out = OUT / f"{'smoke_' if a.smoke else ''}{NAME}_seed_{seed}"
            if out.exists():
                raise SystemExit(f"Output directory already exists: {out}")
            out.mkdir(parents=True)
            print(f"START {NAME} seed={seed} train_points={len(frames['train'])} validation_points={len(va.target)}", flush=True)
            pred, detail = train_xgb_identity(frames, banks, fingerprints, cfg, seed)
            pilot.write_history(out, detail.pop("history"), best_trees=detail["best_trees"], variant=NAME)
            metrics = {"model": NAME, "seed": seed, "smoke": a.smoke,
                       "validation_surface_macro_rmse": gate.surface_macro(va, pred),
                       "validation_pooled_rmse_percent": gate.pooled_rmse_percent(va.target.numpy(), pred), **detail,
                       "test_response_rows_read": 0}
            pd.DataFrame(dict(source_row=va.source_rows, surface_id=va.surface_ids, target=va.target.numpy(),
                              prediction=pred)).to_csv(out / "validation_predictions.csv", index=False)
            learned._write_json_atomic(out / "metrics.json", metrics)
            print(f"DONE {NAME} seed={seed} val_pooled={metrics['validation_pooled_rmse_percent']:.4f} "
                  f"val_surface_macro={metrics['validation_surface_macro_rmse']:.6f} best_trees={detail['best_trees']} "
                  f"({detail['wall_seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
