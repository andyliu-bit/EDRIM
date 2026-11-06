"""D11 (card D): score the new-drug TEST role once (marker COLD_TEST_SCORED.marker) after every run has finished.

Main rule (pre-declared in the card): BiSurfGNN v2 (seeds 41-45, mean of per-seed combination RMSE) vs the gate
XGBoost residual (mean of seeds 41-45): relative change, drug-pair bootstrap 2000 draws (seed 20261009, as D7/D8),
95% upper bound < 0, and at least 4 v2 seeds below the XGBoost mean. Descriptive: lookup HSA, the v2 random-vector
control, DeepSynBa (seed 10, different information level). Sensitivity: excluding pralatrexate pairs.

  .venv/Scripts/python.exe -I -B experiments/model_improvement/cold_drug/score_cold.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "input_parity"))
import run_parity as rp  # noqa: E402
import cold_split  # noqa: E402

D = cold_split.OUT
MARKER = rp.V2 / "COLD_TEST_SCORED.marker"
CACHE = Path("C:/LHZ/research/bisurfgnn/runs/audit_revision/deepsynba_repro_v2_20261012/inputs/trainval_cache.npz")
BOOT_SEED, DRAWS = 20261009, 2000
PRALA = "S754230"


def aligned(path: Path, frame: pd.DataFrame) -> np.ndarray:
    z = np.load(path)
    if not np.array_equal(z["source"], frame.source_combodrugseq.to_numpy()):
        raise RuntimeError(f"{path} not aligned with the cold_test frame")
    return z["prediction_percent"]


def deepsynba(frame: pd.DataFrame) -> tuple[np.ndarray, dict]:
    z = np.load(D / "deepsynba_seed_10/cold_test_predictions.npz")
    index = {k: i for i, k in enumerate(z["keys"].tolist())}
    c = np.load(CACHE)
    ckeys = ["_".join(k) for k in zip(c["drug1"].tolist(), c["drug2"].tolist(), c["cell"].tolist())]
    cpos = {k: i for i, k in enumerate(ckeys)}
    keys = frame.matrix_key.to_numpy()[::9]
    rows = np.array([index[k] for k in keys])
    truth = c["matrix"][[cpos[k] for k in keys]][:, 1:, 1:].astype(float)
    y = frame.response_percent.to_numpy(float).reshape(-1, 3, 3)
    direct, transposed = np.abs(truth - y).max(), np.abs(truth.transpose(0, 2, 1) - y).max()
    p = z["predictions"][rows][:, 1:, 1:].astype(float)
    if direct <= 1e-3:
        layout = "as-is"
    elif transposed <= 1e-3:
        layout, p = "transposed", p.transpose(0, 2, 1)
    else:
        raise RuntimeError(f"DeepSynBa truth layout mismatch ({direct}, {transposed})")
    return p.reshape(-1), {"layout": layout, "truth_max_abs": float(min(direct, transposed))}


def main():
    if not (sys.flags.isolated and sys.flags.dont_write_bytecode):
        raise SystemExit("run with python -I -B")
    frame = cold_split.load_frames()["cold_test"]
    preds = {"v2": {s: aligned(D / f"v2_both_seed_{s}/cold_test_predictions.npz", frame) for s in rp.SEEDS},
             "xgb_cap3000": {s: aligned(D / f"xgb_seed_{s}/cold_test_predictions.npz", frame) for s in rp.SEEDS},
             "v2_random": {s: aligned(D / f"v2_random_seed_{s}/cold_test_predictions.npz", frame) for s in (41, 42, 43)},
             "xgb": {s: aligned(D / f"xgb_cap20000_seed_{s}/cold_test_predictions.npz", frame) for s in rp.SEEDS},
             "lookup_hsa": {0: np.minimum(frame.mono_a_percent, frame.mono_b_percent).to_numpy(float)}}
    dsb, dsb_check = deepsynba(frame)
    preds["deepsynba"] = {10: dsb}
    with MARKER.open("x", encoding="utf-8") as h:
        h.write(json.dumps({"created_local_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "registered": "DEVIATIONS.md D11"}) + "\n")
    y = frame.response_percent.to_numpy(float)
    pairs, codes = np.unique(frame.pair_id.to_numpy(), return_inverse=True)
    n = np.bincount(codes).astype(float)
    sse = {(m, s): np.bincount(codes, weights=(p - y) ** 2) for m, d in preds.items() for s, p in d.items()}
    rng = np.random.default_rng(BOOT_SEED)
    W = np.stack([np.bincount(rng.integers(0, len(pairs), len(pairs)), minlength=len(pairs)) for _ in range(DRAWS)]).astype(float)
    prala = np.array([PRALA in p.split("|") for p in pairs])
    out = {}
    for label, keep in (("all", np.ones(len(pairs), bool)), ("excluding_pralatrexate", ~prala)):
        w = W * keep
        point = {m: float(np.mean([np.sqrt(sse[(m, s)][keep].sum() / n[keep].sum()) for s in d])) for m, d in preds.items()}
        seeds = {m: {int(s): float(np.sqrt(sse[(m, s)][keep].sum() / n[keep].sum())) for s in d} for m, d in preds.items()}
        boot = {m: np.mean([np.sqrt((w @ sse[(m, s)]) / (w @ n)) for s in d], axis=0) for m, d in preds.items()}

        def cmp(a, b):
            r = (boot[a] - boot[b]) / boot[b]
            return {"relative_change": (point[a] - point[b]) / point[b],
                    "ci95": [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]}
        comps = {f"{a}_vs_{b}": cmp(a, b) for a, b in (("v2", "xgb"), ("v2", "lookup_hsa"), ("v2", "deepsynba"),
                                                       ("v2", "v2_random"), ("v2", "xgb_cap3000"), ("xgb", "lookup_hsa"),
                                                       ("deepsynba", "lookup_hsa"))}
        main_cmp = comps["v2_vs_xgb"]
        below = sum(v < point["xgb"] for v in seeds["v2"].values())
        out[label] = dict(pairs=int(keep.sum()), points=int(n[keep].sum()), point=point, per_seed=seeds, comparisons=comps,
                          decision=dict(upper_bound_below_zero=main_cmp["ci95"][1] < 0, v2_seeds_below_xgb=int(below),
                                        passed=bool(main_cmp["ci95"][1] < 0 and below >= 4)))
    res = dict(registered="DEVIATIONS.md D11", split=json.loads((D / "split.json").read_text(encoding="utf-8"))["counts"],
               deepsynba_check=dsb_check, bootstrap={"unit": "drug pair", "draws": DRAWS, "seed": BOOT_SEED}, results=out)
    (D / "cold_test_evaluation.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
