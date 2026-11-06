"""D9 (card A), test part: synergy scores from ALREADY SAVED test predictions, computed once after the validation
result was written (docs/BISURFGNN_V2_SYNERGY_RECHECK.md section 1). No model is run or selected.

Inputs: v2 saved test predictions (D7), XGBoost+identity saved test predictions (D8, stands in for the 'XGB' series),
DeepSynBa v2 test predictions, and the test matrices rebuilt exactly as evaluate_v2.py does (truth and observed edges).
Same pre-declared metrics and bootstrap as the validation run.

  .venv/Scripts/python.exe -I -B experiments/model_improvement/synergy_check_v2/run_v2_test.py
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "synergy_check"))
sys.path.insert(0, str(HERE.parent / "input_parity"))
import synergy_check as sc  # noqa: E402
import run_parity as rp  # noqa: E402

gate = rp.gate
eg = gate.load_module("evaluate_gate_for_synergy", gate.ROOT / "experiments/model_improvement/public_split_gate/evaluate_gate.py")
V2 = rp.V2
OUT = V2 / "synergy_check_v2/test"
MARKER = V2 / "SYNERGY_TEST_EVALUATED.marker"
SEEDS = (41, 42, 43, 44, 45)

sc.BISURF = {f"BS_s{s}": None for s in SEEDS}
sc.XGB = {f"XGB_s{s}": None for s in SEEDS}
sc.BS_SEEDS, sc.XGB_SEEDS = list(sc.BISURF), list(sc.XGB)
sc.COMPARISONS = {k.replace("BS - XGB", "BS - XGB_identity").replace("XGB - DSB", "XGB_identity - DSB"): v
                  for k, v in sc.COMPARISONS.items()}
sc.allowed = lambda p: (sc.READ_LOG.append(str(p)), Path(p))[1]   # test files are intended inputs here


def load_test():
    cells_tab = pd.read_csv(gate.S1 / "cell_mapping_used.csv")
    drugs_tab = pd.read_csv(gate.S1 / "drug_mapping_used.csv")
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
    n = len(matrices)
    if not ((frame.row_i.to_numpy() == np.tile(np.repeat([1, 2, 3], 3), n)).all()
            and (frame.col_j.to_numpy() == np.tile([1, 2, 3], 3 * n)).all()):
        raise RuntimeError("Unexpected test point order")
    ids = frame.source_combodrugseq.to_numpy()
    saved = np.load(V2 / "test_eval/test_combination_predictions_bisurf_v2.npz", allow_pickle=False)
    if not ((saved["point_ids"] == ids).all() and (saved["matrix_keys"] == frame.matrix_key.to_numpy().astype(str)).all()):
        raise RuntimeError("Saved v2 test predictions are not aligned with the rebuilt test table")
    truth = np.stack([m for _, m, _ in matrices])
    y = truth[:, 1:, 1:]
    if np.max(np.abs(y.reshape(-1) - frame.response_percent.to_numpy())) > 0:
        raise RuntimeError("Truth grid mismatch")
    preds, checks = {}, {}
    for s in SEEDS:
        preds[f"BS_s{s}"] = saved[f"seed_{s}"].reshape(n, 3, 3)
        x = np.load(V2 / f"xgb_identity_final/seed_{s}/test_predictions.npz", allow_pickle=False)
        if not (x["point_ids"] == ids).all():
            raise RuntimeError("XGB+identity test predictions are not aligned")
        preds[f"XGB_s{s}"] = x["prediction"].reshape(n, 3, 3)
        checks[f"BS_s{s}"] = dict(combination_rmse=float(np.sqrt(np.mean((preds[f'BS_s{s}'] - y) ** 2))))
        checks[f"XGB_s{s}"] = dict(combination_rmse=float(np.sqrt(np.mean((preds[f'XGB_s{s}'] - y) ** 2))))
    dsb, dsb_check = eg.deepsynba_combination(matrices)
    checks["DSB"] = dict(dsb_check, combination_rmse=float(np.sqrt(np.mean((dsb[:, 1:, 1:] - y) ** 2))))
    obs_a, obs_b = truth[:, 1:, 0], truth[:, 0, 1:]
    lk = np.stack([p for _, _, p in matrices])
    lk_a, lk_b = lk[:, 1:, 0], lk[:, 0, 1:]
    first = frame.iloc[::9].reset_index(drop=True)
    meta_out = first[["matrix_key", "pair_id", "deepsynba_cell", "deepsynba_drug_a", "deepsynba_drug_b"]].copy()
    preds["DSB"] = dsb[:, 1:, 1:]
    preds["LK_HSA"] = np.minimum(lk_a[:, :, None], lk_b[:, None, :])
    preds["LK_Bliss"] = lk_a[:, :, None] * lk_b[:, None, :] / 100.0
    preds["BS_ens5"] = np.mean([preds[k] for k in sc.BS_SEEDS], axis=0)
    preds["XGB_ens5"] = np.mean([preds[k] for k in sc.XGB_SEEDS], axis=0)
    checks["note"] = "XGB series = XGBoost residual + identity codes (D8), not the gate XGBoost"
    return meta_out, y, obs_a, obs_b, lk_a, lk_b, dsb, preds, checks


sc.load_matrices = load_test

if __name__ == "__main__":
    if not (sys.flags.isolated and sys.flags.dont_write_bytecode):
        raise SystemExit("run with `python -I -B`")
    with MARKER.open("x", encoding="utf-8") as handle:     # computed once
        handle.write(json.dumps({"created_local_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "registered": "DEVIATIONS.md D9"}) + "\n")
    sys.argv += ["--out", str(OUT)]
    sc.main()
