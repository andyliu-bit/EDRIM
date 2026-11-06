"""V2 gap diagnosis, step D7: snapshot ensembles from the D5 annealing probe (VALIDATION only, CPU, no training).

For each probed seed: average the saved best-state validation prediction (gate) with the D5 epoch-1..4 predictions,
and compare with single seeds and with 2-/5-seed ensembles of the gate runs.

  .venv/Scripts/python.exe -I experiments/model_improvement/v2_gap_diagnosis/d7_snapshot_ensemble.py
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d2_gap_analysis as d2  # noqa: E402

PROBE = d2.DIAG / "anneal_probe"
OUT = d2.DIAG / "snapshot_ensemble"


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    v = pd.read_parquet(d2.S1 / "points_validation.parquet", columns=["response_percent"])
    y = v.response_percent.to_numpy(float)
    gate = {}
    for tag, path in d2.BISURF.items():
        frame = pd.read_csv(path / "validation_predictions.csv")
        if not (frame.source_row.to_numpy() == d2.VAL_OFFSET + np.arange(len(y))).all():
            raise RuntimeError("order")
        gate[tag] = frame.prediction.to_numpy(float) * 100.0
    res = {"step": "D7", "test_rows_read": 0, "seeds": {}}
    snaps = {}
    for seed in (42, 44):
        run = PROBE / f"bisurf_additive_seed_{seed}"
        ep = [np.load(run / f"validation_prediction_epoch{k}.npy").astype(float) * 100.0 for k in range(1, 5)]
        base = gate[f"BS_add_s{seed}"]
        snap5 = np.mean([base] + ep, axis=0)
        snap4 = np.mean(ep, axis=0)
        snaps[seed] = snap5
        res["seeds"][seed] = {"gate_best_state": d2.rmse(base - y),
                              "anneal_epochs": [d2.rmse(p - y) for p in ep],
                              "snapshot_best_plus_4_anneal": d2.rmse(snap5 - y),
                              "snapshot_4_anneal_only": d2.rmse(snap4 - y)}
    res["two_seed_gate_ensemble_42_44"] = d2.rmse((gate["BS_add_s42"] + gate["BS_add_s44"]) / 2 - y)
    res["two_seed_snapshot_ensemble_42_44"] = d2.rmse((snaps[42] + snaps[44]) / 2 - y)
    res["five_seed_gate_ensemble"] = d2.rmse(np.mean([gate[k] for k in d2.BS_SEEDS], axis=0) - y)
    res["mean_two_seed_gate_ensembles"] = float(np.mean([d2.rmse((gate[a] + gate[b]) / 2 - y)
                                                         for a, b in itertools.combinations(d2.BS_SEEDS, 2)]))
    with (OUT / "d7_summary.json").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
