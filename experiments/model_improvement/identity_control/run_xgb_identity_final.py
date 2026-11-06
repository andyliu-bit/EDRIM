"""XGBoost residual + entity identity as a test-table baseline for BiSurfGNN v2 (DEVIATIONS.md D8).

Same model as run_xgb_identity.py: gate xgb_residual_d10_m1 features and parameters plus categorical drug A, drug B
and cell identity. Two registered changes: up to 20000 trees (the 3000-tree validation run was still improving) and
training on the GPU. Selection: validation surface-macro RMSE in blocks of 50 trees (gate rule); split 1 trains,
split 2 selects. After a seed's selection is final, its test-input predictions at the selected size are saved and the
booster is dropped (no model files: disk). After all seeds the test set is scored once (marker) on the S5/D7 evaluable
set, with the same drug-pair bootstrap, against the saved BiSurfGNN v2 test predictions (D7), DeepSynBa v2, the gate
XGBoost and lookup HSA. Descriptive comparisons only; BiSurfGNN v2 stays frozen.

  .venv/Scripts/python.exe -I -B experiments/model_improvement/identity_control/run_xgb_identity_final.py --smoke
  .venv/Scripts/python.exe -I -B experiments/model_improvement/identity_control/run_xgb_identity_final.py
(--smoke: tiny split-1/2 subsets, 100 trees, CPU, the tiny validation frame stands in for test; split 3 is not read.)
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_xgb_identity as rxi  # noqa: E402  (gate imports, identity_inputs)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

gate, run_gate, learned, pilot = rxi.gate, rxi.run_gate, rxi.learned, rxi.pilot
eg = gate.load_module("evaluate_gate_for_xgb_identity", gate.ROOT / "experiments/model_improvement/public_split_gate/evaluate_gate.py")
V2 = gate.ROOT / "runs/model_improvement/bisurfgnn_v2_dev"
OUT, MARKER = V2 / "xgb_identity_final", V2 / "XGB_IDENTITY_TEST_EVALUATED.marker"
V2_PRED = V2 / "test_eval/test_combination_predictions_bisurf_v2.npz"
SEEDS, MAX_TREES = (41, 42, 43, 44, 45), 20000
COMPARE = [("bisurf_v2", "xgb_identity"), ("xgb_identity", "deepsynba_v2"), ("xgb_identity", "xgb"),
           ("xgb_identity", "lookup_hsa"), ("bisurf_v2", "deepsynba_v2")]


def train_select(frames, build, cfg, seed, device, max_trees):
    """run_xgb_identity.train_xgb_identity on `device` with the larger cap; returns the booster and the selection."""
    import xgboost as xgb
    xc = cfg["xgboost"]; tr, va = frames["train"], frames["validation"]
    start = time.monotonic()
    hsa_tr = np.minimum(tr.mono_response_a, tr.mono_response_b).to_numpy(float)
    hsa_va = np.minimum(va.mono_response_a, va.mono_response_b).to_numpy(float)
    weight = 1.0 / tr.groupby("surface_replicate_id").surface_replicate_id.transform("size").to_numpy(float)
    x = np.vstack([build(tr, False), build(tr, True)])
    types = ["q"] * (x.shape[1] - 3) + ["c"] * 3
    dtrain = xgb.QuantileDMatrix(x, label=np.concatenate([tr.response.to_numpy(float) - hsa_tr] * 2),
                                 weight=np.concatenate([weight, weight]), feature_types=types, enable_categorical=True)
    del x
    params = dict(objective="reg:squarederror", eta=xc["learning_rate"], max_depth=10, min_child_weight=1,
                  subsample=xc["subsample"], colsample_bytree=xc["colsample"], tree_method="hist", device=device,
                  seed=seed, nthread=xc["threads"])
    booster = xgb.train(params, dtrain, num_boost_round=max_trees)
    train_seconds = time.monotonic() - start
    del dtrain
    matrix = lambda data: xgb.DMatrix(data, feature_types=types, enable_categorical=True)  # noqa: E731
    dva = (matrix(build(va, False)), matrix(build(va, True)))
    base = run_gate.base_margin(booster)
    margin = [np.zeros(len(va)), np.zeros(len(va))]
    target, surfaces = va.response.to_numpy(float), va.surface_replicate_id.to_numpy()
    best_k, best = 0, pilot.metric(target, hsa_va, surfaces)["surface_macro_rmse"]
    history, best_pred, step = [{"trees": 0, "surface_macro_rmse": best}], hsa_va.copy(), xc["eval_every"]
    for k in range(step, max_trees + 1, step):
        for o in (0, 1):
            margin[o] += booster.predict(dva[o], iteration_range=(k - step, k), output_margin=True) - base
        pred = hsa_va + 0.5 * ((base + margin[0]) + (base + margin[1]))
        score = pilot.metric(target, pred, surfaces)["surface_macro_rmse"]
        history.append({"trees": k, "surface_macro_rmse": score, "pooled_rmse_percent": gate.pooled_rmse_percent(target, pred)})
        if score < best - cfg["training"]["min_delta"]:
            best, best_k, best_pred = score, k, pred.copy()
    if best_k:  # the block-wise sum must equal a direct prediction at the selected size
        direct = hsa_va + 0.5 * (booster.predict(dva[0], iteration_range=(0, best_k)) + booster.predict(dva[1], iteration_range=(0, best_k)))
        if np.max(np.abs(direct - best_pred)) > 1e-5:
            raise RuntimeError("Block-wise validation prediction differs from the direct prediction")
    return booster, types, best_k, history, best_pred, {
        "seed": seed, "best_trees": best_k, "max_trees": max_trees, "at_cap": best_k == max_trees, "device": device,
        "validation_surface_macro_rmse": best, "validation_pooled_rmse_percent": gate.pooled_rmse_percent(target, best_pred),
        "train_seconds": train_seconds, "wall_seconds": time.monotonic() - start, "params": params}


def predict_frame(booster, types, best_k, build, frame):
    """Percent-scale combination predictions at the selected size, averaged over both drug orders."""
    import xgboost as xgb
    hsa = np.minimum(frame.mono_response_a, frame.mono_response_b).to_numpy(float)
    if not best_k:
        return 100.0 * hsa
    d = [xgb.DMatrix(build(frame, s), feature_types=types, enable_categorical=True) for s in (False, True)]
    return 100.0 * (hsa + 0.5 * (booster.predict(d[0], iteration_range=(0, best_k)) + booster.predict(d[1], iteration_range=(0, best_k))))


def load_test(frames):
    """split-3 evaluable frame exactly as evaluate_gate (S5) and evaluate_v2 (D7) build it."""
    s1 = gate.S1
    cells_tab, drugs_tab = pd.read_csv(s1 / "cell_mapping_used.csv"), pd.read_csv(s1 / "drug_mapping_used.csv")
    cells = {r.deepsynba_cell: {"cell_id": r.cell_id, "biological_cell_id": r.biological_cell_id}
             for r in cells_tab.itertuples() if r.status == "mapped"}
    drugs = {r.deepsynba_drug: r.canonical_drug_id for r in drugs_tab.itertuples() if bool(r.mapped)}
    meta = pd.read_csv(eg.META)
    meta["key"] = meta.drug1 + "_" + meta.drug2 + "_" + meta.cell
    train_rows = eg.evaluable_rows(meta, 1, cells, drugs)
    data = pickle.load(eg.RESP.open("rb"))
    train_entries = {k: data[k] for k in train_rows.key}
    entries = {k: data[k] for k in meta.key[meta.split == 3]}
    del data
    lookup = eg.Lookup.from_matrices(train_rows, train_entries)
    del train_entries
    eg.check_lookup(lookup, frames, tolerance=0.0)
    return eg.test_points(meta, entries, lookup, cells, drugs)


def evaluate(frame, matrices, xgb_id, frames, banks, cfg):
    import xgboost as xgb
    saved = np.load(V2_PRED, allow_pickle=False)
    if not np.array_equal(saved["point_ids"], frame.source_combodrugseq.to_numpy()):
        raise RuntimeError("Saved BiSurfGNN v2 predictions are not aligned with the evaluable test frame")
    comb = {"lookup_hsa": {0: np.minimum(frame.mono_a_percent, frame.mono_b_percent).to_numpy()}, "xgb": {},
            "xgb_identity": xgb_id, "bisurf_v2": {s: saved[f"seed_{s}"] for s in SEEDS}}
    runs = eg.locate_runs("additive")["xgb"]
    build = run_gate.xgb_inputs(frames, banks, gate.morgan(banks), cfg)
    dtest = (xgb.DMatrix(build(frame, False)), xgb.DMatrix(build(frame, True)))
    hsa = np.minimum(frame.mono_response_a, frame.mono_response_b).to_numpy(float)
    for seed, path in runs.items():
        k = json.loads((path / "metrics.json").read_text(encoding="utf-8"))["best_epoch"]
        booster = xgb.Booster(); booster.load_model(path / "model.json")
        comb["xgb"][seed] = 100.0 * (hsa + 0.5 * (booster.predict(dtest[0], iteration_range=(0, k)) + booster.predict(dtest[1], iteration_range=(0, k))))
    dsb16, dsb_check = eg.deepsynba_combination(matrices)
    comb["deepsynba_v2"] = {10: np.concatenate([dsb16[s, 1:, 1:].ravel() for s in range(len(matrices))])}
    lookup16 = np.stack([p16 for _, _, p16 in matrices])
    preds16 = {}
    for method, seeds in comb.items():
        preds16[method] = {}
        for seed, p in seeds.items():
            if method == "deepsynba_v2":
                preds16[method][seed] = dsb16
            else:
                full = lookup16.copy(); full[:, 1:, 1:] = np.asarray(p).reshape(-1, 3, 3); preds16[method][seed] = full
    rows, sse, n, pairs = eg.evaluate(frame, matrices, preds16, comb)
    boots = eg.bootstrap(sse, n, list(comb))
    point = {m: float(rows[rows.method == m].combination_rmse_percent.mean()) for m in comb}
    def interval(a, b):
        d = (boots[a] - boots[b]) / boots[b]; return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
    comparisons = {f"{a}_vs_{b}": {"relative_change": (point[a] - point[b]) / point[b], "ci95": interval(a, b)} for a, b in COMPARE}
    return rows, point, comparisons, dsb_check, {"matrices": len(matrices), "combination_points": int(len(frame)), "pairs": int(len(pairs))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if not (sys.flags.isolated and sys.flags.dont_write_bytecode):
        raise SystemExit("run with `python -I -B`")
    sys.stdout.reconfigure(encoding="utf-8")
    out = OUT.parent / "smoke_xgb_identity_final" if a.smoke else OUT
    if out.exists() or (not a.smoke and MARKER.exists()):
        raise SystemExit(f"Already run: {out if out.exists() else MARKER}")
    cfg = gate.load_config(run_gate.CONFIG)
    device, max_trees, seeds = ("cpu", 100, SEEDS[:1]) if a.smoke else ("cuda", MAX_TREES, SEEDS)
    out.mkdir(parents=True)
    with threadpool_limits(limits=cfg["xgboost"]["threads"]):
        frames, banks, _ = run_gate.prepare(cfg)
        if a.smoke:
            frames = run_gate.tiny_frames(frames, cfg["timing"]["subset_seed"] + 2)
        build = rxi.identity_inputs(frames, banks, gate.morgan(banks), cfg)
        test = None
        xgb_id, summary = {}, []
        for seed in seeds:
            booster, types, best_k, history, va_pred, m = train_select(frames, build, cfg, seed, device, max_trees)
            seed_dir = out / f"seed_{seed}"; seed_dir.mkdir()
            pilot.write_history(seed_dir, history, **{k: v for k, v in m.items() if k != "params"})
            np.savez_compressed(seed_dir / "validation_predictions.npz", prediction=va_pred.astype(np.float32))
            print(f"SELECTED seed={seed} best_trees={best_k} at_cap={m['at_cap']} val_pooled={m['validation_pooled_rmse_percent']:.4f} "
                  f"({m['wall_seconds']:.0f}s)", flush=True)
            if test is None:  # selection of this seed is final before any split-3 row is read
                test = (frames["validation"], None) if a.smoke else load_test(frames)
            xgb_id[seed] = predict_frame(booster, types, best_k, build, test[0])
            np.savez_compressed(seed_dir / "test_predictions.npz", prediction=xgb_id[seed],
                                point_ids=test[0].source_combodrugseq.to_numpy())
            summary.append(m)
            del booster
        learned._write_json_atomic(out / "selection_summary.json", {"seeds": summary, "smoke": a.smoke})
        if a.smoke:
            print("SMOKE done (no test scoring)", flush=True)
            return
        with MARKER.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps({"created_local_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "registered": "DEVIATIONS.md D8"}) + "\n")
        frame, matrices = test
        rows, point, comparisons, dsb_check, evaluable = evaluate(frame, matrices, xgb_id, frames, banks, cfg)
        rows.to_csv(out / "test_per_seed_metrics.csv", index=False)
        result = {"registered": "DEVIATIONS.md D8", "evaluable": evaluable, "point_estimates_combination_rmse_percent": point,
                  "seed_sd_combination_rmse_percent": {m: float(rows[rows.method == m].combination_rmse_percent.std(ddof=1))
                                                       if (rows.method == m).sum() > 1 else None for m in point},
                  "comparisons": comparisons, "selection": summary, "deepsynba_check": dsb_check,
                  "bootstrap": {"unit": "test drug pair", "draws": eg.DRAWS, "seed": eg.BOOT_SEED},
                  "note": "BiSurfGNN v2 frozen in D7; this run adds a baseline only", "test_response_reads": 1}
        with (out / "test_evaluation.json").open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=1, default=float)
        print(json.dumps({k: result[k] for k in ("evaluable", "point_estimates_combination_rmse_percent", "comparisons")},
                         indent=1, default=float), flush=True)


if __name__ == "__main__":
    main()
