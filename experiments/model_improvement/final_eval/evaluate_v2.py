"""BiSurfGNN v2 final evaluation (dev protocol section 1 and section 3 phase 3; DEVIATIONS.md D7): the single test
evaluation of the frozen configuration ('both' inputs + D6 long schedule, seeds 41-45).

  --check   preconditions, model loading, and an exact re-prediction of every v2 seed's saved validation predictions;
            never opens the split-3 responses or the DeepSynBa test predictions.
  (default) create V2_TEST_EVALUATED.marker exclusively, read split 3 once and score every method on the S5 evaluable
            set: lookup HSA/Bliss, gate XGBoost residual (5 seeds), original BiSurfGNN (gate additive, 5 seeds),
            BiSurfGNN v2 (5 seeds, plus their ensemble), DeepSynBa reproduction v2. Refuses to run if the marker exists.

Decision rule (protocol section 1): drug-pair bootstrap (2000 draws, seed 20261009) of the v2 five-seed mean
combination RMSE relative to DeepSynBa v2, 95% upper bound < 0, and at least 4 of 5 seeds individually below DeepSynBa.
Additional: the 5-seed ensemble (DeepSynBa has none). Sensitivity: without pralatrexate pairs (primary not trimmed).
Reuses the gate's S5 functions (public_split_gate/evaluate_gate.py) without modifying them.

  .venv/Scripts/python.exe -I -B experiments/model_improvement/final_eval/evaluate_v2.py --check
  .venv/Scripts/python.exe -I -B experiments/model_improvement/final_eval/evaluate_v2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "input_parity"))
import run_parity as rp  # noqa: E402  (sets CUBLAS_WORKSPACE_CONFIG before torch is imported)

import argparse  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

gate, learned = rp.gate, rp.learned
eg = gate.load_module("evaluate_gate_for_v2", gate.ROOT / "experiments/model_improvement/public_split_gate/evaluate_gate.py")

V2RUN = rp.V2 / "phase2_long_both"
OUT, MARKER = rp.V2 / "test_eval", rp.V2 / "V2_TEST_EVALUATED.marker"
SEEDS = eg.SEEDS
# docs/PUBLIC_SPLIT_GATE_V1_RESULTS.md section 2 (S5, rounded to 3 decimals); recomputed here on the same set
S5_REPORTED = {"lookup_hsa": 12.009, "lookup_bliss": 11.976, "xgb": 10.408, "bisurf": 10.424, "deepsynba_v2": 9.725}
S5_EVALUABLE = {"matrices": 54196, "combination_points": 487764, "pairs": 979}
COMPARE = [("bisurf_v2", "deepsynba_v2"), ("bisurf_v2", "xgb"), ("bisurf_v2", "bisurf"), ("bisurf_v2", "lookup_hsa"),
           ("bisurf_v2_ensemble", "deepsynba_v2"), ("bisurf", "deepsynba_v2"), ("xgb", "deepsynba_v2"),
           ("deepsynba_v2", "lookup_hsa")]


def v2_runs() -> dict:
    runs = {}
    for seed in SEEDS:
        path = V2RUN / f"long_both_seed_{seed}"
        m = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
        if m.get("smoke") or m.get("first_epochs_identical_to_phase1") is not True or m.get("stopped_by") != "max_epochs" \
                or not (path / "model.pt").exists():
            raise RuntimeError(f"v2 seed {seed}: not a finished, checked formal run")
        runs[seed] = path
    return runs


def load_v2(runs, banks, cfg):
    models = {}
    for seed, path in runs.items():
        model = rp.build_parity_model(banks, "both", seed, cfg["device"])
        model.load_state_dict(torch.load(path / "model.pt", weights_only=True, map_location="cpu"))
        model.eval(); models[seed] = model
    return models


def check_validation(models, runs, frames, banks, ext, cfg) -> dict:
    """Each reloaded v2 model must reproduce the validation predictions saved at the end of its training run."""
    va = gate.build_tables({"validation": frames["validation"]}, banks)["validation"]
    out = {}
    for seed, model in models.items():
        saved = pd.read_csv(runs[seed] / "validation_predictions.csv")
        if not np.array_equal(saved.source_row.to_numpy(), np.asarray(va.source_rows)):
            raise RuntimeError(f"v2 seed {seed}: validation row order differs")
        diff = float(np.max(np.abs(rp.predict_ext(model, va, banks, ext, cfg) - saved.prediction.to_numpy())))
        if diff > 1e-6:
            raise RuntimeError(f"v2 seed {seed}: reloaded model does not reproduce its validation predictions ({diff})")
        out[seed] = diff
    return out


def interval(a, b):
    d = (a - b) / b
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def rmse(p, y):
    return float(np.sqrt(np.mean((np.asarray(p, float) - y) ** 2)))


def score(frame, matrices, comb, prala):
    lookup16 = np.stack([p16 for _, _, p16 in matrices])
    dsb16 = comb.pop("_dsb16")
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
    comparisons = {f"{a}_vs_{b}": {"relative_change": (point[a] - point[b]) / point[b], "ci95": interval(boots[a], boots[b])}
                   for a, b in COMPARE}
    per_seed = rows[rows.method == "bisurf_v2"].set_index("seed").combination_rmse_percent
    below = int((per_seed < point["deepsynba_v2"]).sum())
    main = comparisons["bisurf_v2_vs_deepsynba_v2"]
    decision = {"rule": "pair bootstrap 95% upper bound of v2 five-seed mean RMSE vs DeepSynBa v2 < 0, and >= 4/5 seeds below",
                "relative_change": main["relative_change"], "ci95": main["ci95"], "seeds_below_deepsynba": below,
                "passed": bool(main["ci95"][1] < 0 and below >= 4)}
    # sensitivity: drop every pair that contains pralatrexate (pre-declared; the primary metric is not trimmed)
    keep = ~(frame.drug_a_id.isin(prala) | frame.drug_b_id.isin(prala)).to_numpy()
    sub, y = frame[keep].reset_index(drop=True), frame.response_percent.to_numpy()[keep]
    sse_sub, n_sub = {}, None
    for method, seeds in comb.items():
        for seed, p in seeds.items():
            sse_sub[(method, seed)], n_sub, _ = eg.pair_sse(sub, np.asarray(p)[keep])
    boots_sub = eg.bootstrap(sse_sub, n_sub, list(comb))
    point_sub = {m: float(np.mean([rmse(np.asarray(p)[keep], y) for p in seeds.values()])) for m, seeds in comb.items()}
    seeds_sub = {s: rmse(comb["bisurf_v2"][s][keep], y) for s in SEEDS}
    sensitivity = {"excluded_drug_ids": sorted(prala), "pairs": int(len(n_sub)), "combination_points": int(keep.sum()),
                   "point_estimates_combination_rmse_percent": point_sub, "bisurf_v2_per_seed": seeds_sub,
                   "bisurf_v2_vs_deepsynba_v2": {
                       "relative_change": (point_sub["bisurf_v2"] - point_sub["deepsynba_v2"]) / point_sub["deepsynba_v2"],
                       "ci95": interval(boots_sub["bisurf_v2"], boots_sub["deepsynba_v2"]),
                       "seeds_below_deepsynba": int(sum(v < point_sub["deepsynba_v2"] for v in seeds_sub.values()))}}
    return rows, point, comparisons, decision, sensitivity, pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if not (sys.flags.isolated and sys.flags.dont_write_bytecode):
        raise SystemExit("run with `python -I -B` (pickle and torch containers)")
    if MARKER.exists():
        raise SystemExit("The v2 test evaluation has already been run; the test set is evaluated only once")
    bindings, dsb_eval, selection, form, runs, cfg = eg.preconditions()
    if form != "additive" or not dsb_eval.get("reproduced"):
        raise RuntimeError("Unexpected gate form selection or DeepSynBa acceptance state")
    v2cfg = rp.effective_config()
    vruns = v2_runs()
    frames, banks = eg.dev_setup(cfg)
    csv_lookup_check = eg.check_lookup(eg.Lookup(), frames, tolerance=1e-9)
    cells_tab = pd.read_csv(gate.S1 / "cell_mapping_used.csv")
    drugs_tab = pd.read_csv(gate.S1 / "drug_mapping_used.csv")
    cells = {r.deepsynba_cell: {"cell_id": r.cell_id, "biological_cell_id": r.biological_cell_id}
             for r in cells_tab.itertuples() if r.status == "mapped"}
    drugs = {r.deepsynba_drug: r.canonical_drug_id for r in drugs_tab.itertuples() if bool(r.mapped)}
    prala = {r.canonical_drug_id for r in drugs_tab.itertuples() if str(r.deepsynba_drug).strip().lower() == "pralatrexate"}
    if len(prala) != 1:
        raise RuntimeError(f"Expected one pralatrexate ID, found {prala}")
    with threadpool_limits(limits=cfg["xgboost"]["threads"]):
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        features = rp.load_deepsynba_features(banks.drug_ids, banks.cell_ids, frames,
                                              feature_dir=rp.guard(rp.FEATURE_DIR), mapping_dir=rp.guard(gate.S1))
        ext = rp.DeviceExternal(features, "both", torch.device(v2cfg["device"]))
        gate_models, trees = eg.load_models(form, runs, banks, cfg)
        v2_models = load_v2(vruns, banks, v2cfg)
        if a.check:
            validation_check = check_validation(v2_models, vruns, frames, banks, ext, v2cfg)
            print(json.dumps({"check": "passed", "gate_form": form, "xgb_trees": trees, "csv_lookup_check": csv_lookup_check,
                              "v2_runs": {s: str(p) for s, p in vruns.items()}, "v2_validation_max_abs_diff": validation_check,
                              "pralatrexate_ids": sorted(prala)}, indent=1, default=str))
            return
        OUT.mkdir()
        with MARKER.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps({"created_local_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "registered": "DEVIATIONS.md D7"}) + "\n")
        start = time.monotonic()
        try:
            meta = pd.read_csv(eg.META)
            meta["key"] = meta.drug1 + "_" + meta.drug2 + "_" + meta.cell
            train_rows = eg.evaluable_rows(meta, 1, cells, drugs)
            data = pickle.load(eg.RESP.open("rb"))
            train_entries = {k: data[k] for k in train_rows.key}
            entries = {k: data[k] for k in meta.key[meta.split == 3]}
            del data
            lookup = eg.Lookup.from_matrices(train_rows, train_entries)
            del train_entries
            lookup_check = eg.check_lookup(lookup, frames, tolerance=0.0)
            frame, matrices = eg.test_points(meta, entries, lookup, cells, drugs)
            table = learned.build_compact_point_table(frame, banks, role="test", fold=0, monotherapy_upper_bound=3.0)
            comb = {"lookup_hsa": {0: np.minimum(frame.mono_a_percent, frame.mono_b_percent).to_numpy()},
                    "lookup_bliss": {0: (frame.mono_a_percent * frame.mono_b_percent / 100.0).to_numpy()},
                    "xgb": {}, "bisurf": {}, "bisurf_v2": {}}
            for seed, model in gate_models.items():
                comb["bisurf"][seed] = 100.0 * np.asarray(eg.search.predict(model, table, banks, cfg), float)
            for seed, model in v2_models.items():
                comb["bisurf_v2"][seed] = 100.0 * np.asarray(rp.predict_ext(model, table, banks, ext, v2cfg), float)
            comb["bisurf_v2_ensemble"] = {0: np.mean([comb["bisurf_v2"][s] for s in SEEDS], axis=0)}
            import xgboost as xgb
            build = eg.run_gate.xgb_inputs(frames, banks, gate.morgan(banks), cfg)
            dtest = (xgb.DMatrix(build(frame, False)), xgb.DMatrix(build(frame, True)))
            hsa_ratio = np.minimum(frame.mono_response_a, frame.mono_response_b).to_numpy(float)
            for seed, path in runs["xgb"].items():
                booster = xgb.Booster(); booster.load_model(path / "model.json")
                k = trees[seed]
                delta = 0.5 * (booster.predict(dtest[0], iteration_range=(0, k)) + booster.predict(dtest[1], iteration_range=(0, k))) if k else 0.0
                comb["xgb"][seed] = 100.0 * (hsa_ratio + delta)
            dsb16, dsb_check = eg.deepsynba_combination(matrices)
            comb["deepsynba_v2"] = {10: np.concatenate([dsb16[s, 1:, 1:].ravel() for s in range(len(matrices))])}
            np.savez_compressed(OUT / "test_combination_predictions_bisurf_v2.npz",
                                **{f"seed_{s}": comb["bisurf_v2"][s] for s in SEEDS}, ensemble=comb["bisurf_v2_ensemble"][0],
                                point_ids=frame.source_combodrugseq.to_numpy(), matrix_keys=frame.matrix_key.to_numpy().astype(str),
                                pair_ids=frame.pair_id.to_numpy().astype(str))
            comb["_dsb16"] = dsb16
            rows, point, comparisons, decision, sensitivity, pairs = score(frame, matrices, comb, prala)
        except BaseException:
            (OUT / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
            raise
        rows.to_csv(OUT / "per_seed_metrics.csv", index=False)
        evaluable = {"matrices": len(matrices), "combination_points": int(len(frame)), "pairs": int(len(pairs))}
        consistency = {m: {"recomputed": point[m], "s5_reported": v, "abs_diff": abs(point[m] - v)} for m, v in S5_REPORTED.items()}
        result = {"protocol": "BiSurfGNN-v2-dev section 1 / phase 3", "registered": "DEVIATIONS.md D7",
                  "final_configuration": "gate additive BiSurfGNN + MoLFormer 768 + 976 genes (zero-init projections), "
                                         "60-epoch cosine schedule (D6), seeds 41-45",
                  "evaluable": evaluable, "point_estimates_combination_rmse_percent": point,
                  "seed_sd_combination_rmse_percent": {m: float(rows[rows.method == m].combination_rmse_percent.std(ddof=1))
                                                       if (rows.method == m).sum() > 1 else None for m in point},
                  "all16_rmse_percent": {m: float(rows[rows.method == m].all16_rmse_percent.mean()) for m in point},
                  "decision": decision, "comparisons": comparisons, "sensitivity_without_pralatrexate": sensitivity,
                  "s5_consistency": {"evaluable_identical": evaluable == S5_EVALUABLE, "methods": consistency,
                                     "all_within_rounding": bool(all(c["abs_diff"] <= 6e-4 for c in consistency.values()))},
                  "deepsynba_check": dsb_check, "lookup_check": lookup_check, "xgb_trees": trees,
                  "bootstrap": {"unit": "test drug pair", "draws": eg.DRAWS, "seed": eg.BOOT_SEED},
                  "disclosure": "split 3 was used once before, in PublicSplitGate-v1 S5, where the original version failed",
                  "test_response_reads": 1, "wall_seconds": time.monotonic() - start}
        with (OUT / "evaluation.json").open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=1, default=float)
        print(json.dumps({k: result[k] for k in ("evaluable", "point_estimates_combination_rmse_percent", "decision",
                                                 "s5_consistency")}, indent=1, default=float), flush=True)


if __name__ == "__main__":
    main()
