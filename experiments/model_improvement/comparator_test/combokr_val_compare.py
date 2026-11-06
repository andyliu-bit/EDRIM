"""D10 C2, descriptive: put comboKR 2.0 (public split, 10 iterations) next to the other models on split 2 with the
same quantities, so its Pearson can be read against ours. Validation only, nothing trained, split 3 not read.

  response Pearson : Pearson of predicted vs measured response over all combination cells (the D7 table column)
  Bliss Pearson    : Pearson of the derived Bliss-excess score per matrix, observed monotherapy reference
                     (the card-A primary metric; mean over the 9 combination cells of a_i*b_j/100 - response)
Seeded models: metric per seed, then the mean (single-model convention).

  .venv/Scripts/python.exe -I -B experiments/model_improvement/comparator_test/combokr_val_compare.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "input_parity"))
import run_parity as rp  # noqa: E402

ROOT = rp.gate.ROOT
GATE = ROOT / "runs/model_improvement/public_split_gate_v1"
PUB = ROOT / "runs/model_improvement/combokr2_original_v1/public_split"
DSB = ROOT / "runs/model_improvement/v2_gap_diagnosis/deepsynba_val/full_epoch484_selected_val_predictions.npz"
VAL_OFFSET = 10_000_000


def main():
    v = pd.read_parquet(rp.gate.S1 / "points_validation.parquet",
                        columns=["matrix_key", "row_i", "col_j", "response_percent", "hsa_lookup_percent"])
    n = len(v) // 9
    y = v.response_percent.to_numpy(float).reshape(n, 3, 3)
    keys = v.matrix_key.to_numpy()[::9]
    z = np.load(DSB, allow_pickle=False)
    pos = pd.Index(z["matrix_key"]).get_indexer(keys)
    if (pos < 0).any():
        raise RuntimeError("DeepSynBa validation predictions missing for some matrices")
    truth = z["truth"].astype(float)[pos]
    if np.max(np.abs(truth[:, 1:, 1:] - y)) > 1e-3:
        raise RuntimeError("DeepSynBa truth differs from the point table")
    a, b = truth[:, 1:, 0], truth[:, 0, 1:]
    bliss_ref = a[:, :, None] * b[:, None, :] / 100.0
    s_obs = (bliss_ref - y).mean(axis=(1, 2))

    def csv_grid(path):
        f = pd.read_csv(path / "validation_predictions.csv")
        if not (f.source_row.to_numpy() == VAL_OFFSET + np.arange(len(v))).all():
            raise RuntimeError(f"{path} not aligned")
        return 100.0 * f.prediction.to_numpy(float).reshape(n, 3, 3)

    ck = np.load(PUB / "validation/pred_iters10.npz")
    if not np.array_equal(ck["matrix_key"], keys.astype(str)):
        raise RuntimeError("comboKR predictions not aligned")
    models = {
        "BiSurfGNN v2": [csv_grid(ROOT / f"runs/model_improvement/bisurfgnn_v2_dev/phase2_long_both/long_both_seed_{s}")
                         for s in (41, 42, 43, 44, 45)],
        "DeepSynBa v2": [z["predictions"].astype(float)[pos][:, 1:, 1:]],
        "XGBoost residual (gate)": [csv_grid(GATE / f"s4_xgb_r2/xgb_residual_d10_m1_seed_{s}") for s in (41, 42, 43, 44, 45)],
        "lookup HSA": [v.hsa_lookup_percent.to_numpy(float).reshape(n, 3, 3)],
        "comboKR 2.0 (10 iterations)": [ck["pred"].astype(float).reshape(n, 3, 3)],
    }
    rows = []
    for name, grids in models.items():
        r = [float(np.sqrt(np.mean((p - y) ** 2))) for p in grids]
        pr = [float(np.corrcoef(p.ravel(), y.ravel())[0, 1]) for p in grids]
        bl = [float(np.corrcoef((bliss_ref - p).mean(axis=(1, 2)), s_obs)[0, 1]) for p in grids]
        rows.append(dict(model=name, seeds=len(grids), combination_rmse=np.mean(r), response_pearson=np.mean(pr),
                         bliss_excess_pearson=np.mean(bl)))
    table = pd.DataFrame(rows)
    out = PUB / "validation/same_quantity_comparison.csv"
    table.to_csv(out, index=False)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(json.dumps({"matrices": n, "written": str(out)}))


if __name__ == "__main__":
    main()
