"""D11 (card D): gate XGBoost residual (xgb_residual_d10_m1, configs/gate.yaml unchanged) on the new-drug split.
run_gate.train_xgb with train = 'train' role and validation = 'cold_val' (block-wise tree selection, gate rule); the
selected booster then predicts 'cold_test'. No cold_test metric here (score_cold.py scores once).

  .venv/Scripts/python.exe -I -B experiments/model_improvement/cold_drug/run_cold_xgb.py --seed 41 [--smoke]
  --max-trees 20000: the larger-cap version registered in DEVIATIONS.md (D11 supplement; output xgb_cap20000_seed_<s>).
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "input_parity"))
import run_parity as rp  # noqa: E402
import cold_split  # noqa: E402

import argparse  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

gate, run_gate = rp.gate, rp.run_gate


def main():
    if not sys.flags.isolated:
        raise SystemExit("run with python -I")
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, choices=rp.SEEDS, required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-trees", type=int, default=None)
    a = ap.parse_args()
    tag = "xgb" if a.max_trees is None else f"xgb_cap{a.max_trees}"
    out = cold_split.OUT / f"{'smoke_' if a.smoke else ''}{tag}_seed_{a.seed}"
    if out.exists():
        raise SystemExit(f"{out} exists")
    cfg = rp.effective_config()
    if a.max_trees is not None:
        cfg["xgboost"]["max_trees"] = a.max_trees
    if a.smoke:
        cfg["xgboost"]["max_trees"] = 100
    out.mkdir(parents=True)
    frames = cold_split.load_frames()
    if a.smoke:
        frames = run_gate.tiny_frames(frames, 7)
    banks = gate.build_banks(frames)
    with threadpool_limits(limits=cfg["xgboost"]["threads"]):
        import xgboost as xgb
        fp = gate.morgan(banks)
        val_pred, detail = run_gate.train_xgb({"train": frames["train"], "validation": frames["cold_val"]}, banks, fp, cfg, a.seed, out)
        build = run_gate.xgb_inputs({"train": frames["train"]}, banks, fp, cfg)
        te = frames["cold_test"]
        hsa = np.minimum(te.mono_response_a, te.mono_response_b).to_numpy(float)
        k = detail["best_epoch"]
        booster = xgb.Booster(); booster.load_model(out / "model.json")
        delta = 0.5 * (booster.predict(xgb.DMatrix(build(te, False)), iteration_range=(0, k))
                       + booster.predict(xgb.DMatrix(build(te, True)), iteration_range=(0, k))) if k else 0.0
        test_pred = hsa + delta
    for role, frame, pred in (("cold_val", frames["cold_val"], val_pred), ("cold_test", te, test_pred)):
        np.savez_compressed(out / f"{role}_predictions.npz", source=frame.source_combodrugseq.to_numpy(),
                            prediction_percent=100.0 * np.asarray(pred, float))
    (out / "metrics.json").write_text(json.dumps(
        {"seed": a.seed, "best_trees": k, "max_trees": cfg["xgboost"]["max_trees"], "at_cap": k == cfg["xgboost"]["max_trees"],
         "cold_val_pooled_rmse_percent": gate.pooled_rmse_percent(frames["cold_val"].response.to_numpy(float), val_pred),
         "train_seconds": detail["train_seconds"]}, indent=1), encoding="utf-8")
    print(f"DONE cold xgb seed={a.seed} best_trees={k}", flush=True)


if __name__ == "__main__":
    main()
