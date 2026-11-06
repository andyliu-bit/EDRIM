"""Shared test-side helpers for the D10 comparators (card C): rebuild the S5/D7 evaluable test frame and score
per-seed comparator predictions once against the saved BiSurfGNN v2 (D7), XGBoost+identity (D8), DeepSynBa v2 and
lookup HSA, with the same drug-pair bootstrap as D7/D8. Descriptive only; nothing here selects or trains a model."""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "input_parity"))
import run_parity as rp  # noqa: E402

gate = rp.gate
eg = gate.load_module("evaluate_gate_for_comparators", gate.ROOT / "experiments/model_improvement/public_split_gate/evaluate_gate.py")
V2 = rp.V2
SEEDS = (41, 42, 43, 44, 45)


def load_test():
    """split-3 evaluable frame exactly as evaluate_gate (S5), evaluate_v2 (D7) and D8 build it."""
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
    frame, matrices = eg.test_points(meta, entries, lookup, cells, drugs)
    saved = np.load(V2 / "test_eval/test_combination_predictions_bisurf_v2.npz", allow_pickle=False)
    if not np.array_equal(saved["point_ids"], frame.source_combodrugseq.to_numpy()):
        raise RuntimeError("Rebuilt test frame differs from the D7 frame")
    return frame, matrices


def score_once(name: str, preds: dict, frame, matrices, out: Path, marker: Path, extra: dict) -> dict:
    """preds: {seed: combination predictions in percent, aligned with frame}. Writes out/test_evaluation.json."""
    with marker.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps({"created_local_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "method": name}) + "\n")
    ids = frame.source_combodrugseq.to_numpy()
    saved = np.load(V2 / "test_eval/test_combination_predictions_bisurf_v2.npz", allow_pickle=False)
    xid = {}
    for s in SEEDS:
        z = np.load(V2 / f"xgb_identity_final/seed_{s}/test_predictions.npz", allow_pickle=False)
        if not np.array_equal(z["point_ids"], ids):
            raise RuntimeError("XGB+identity predictions misaligned")
        xid[s] = z["prediction"]
    comb = {"lookup_hsa": {0: np.minimum(frame.mono_a_percent, frame.mono_b_percent).to_numpy()},
            "xgb_identity": xid, "bisurf_v2": {s: saved[f"seed_{s}"] for s in SEEDS}, name: preds}
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
                full = lookup16.copy()
                full[:, 1:, 1:] = np.asarray(p, float).reshape(-1, 3, 3)
                preds16[method][seed] = full
    rows, sse, n, pairs = eg.evaluate(frame, matrices, preds16, comb)
    boots = eg.bootstrap(sse, n, list(comb))
    point = {m: float(rows[rows.method == m].combination_rmse_percent.mean()) for m in comb}

    def interval(a, b):
        d = (boots[a] - boots[b]) / boots[b]
        return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
    compare = [(name, "deepsynba_v2"), (name, "lookup_hsa"), (name, "xgb_identity"), ("bisurf_v2", name)]
    comparisons = {f"{a}_vs_{b}": {"relative_change": (point[a] - point[b]) / point[b], "ci95": interval(a, b)}
                   for a, b in compare}
    per_seed = rows[rows.method == name]
    result = dict(method=name, point=point, comparisons=comparisons, per_seed=per_seed.to_dict(orient="records"),
                  seed_sd=float(per_seed.combination_rmse_percent.std(ddof=1)) if len(per_seed) > 1 else None,
                  deepsynba_check=dsb_check,
                  evaluable={"matrices": len(matrices), "points": int(len(frame)), "pairs": int(len(pairs))},
                  bootstrap={"unit": "test drug pair", "draws": eg.DRAWS, "seed": eg.BOOT_SEED}, **extra)
    out.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out / "test_per_seed_metrics.csv", index=False)
    (out / "test_evaluation.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    return result
