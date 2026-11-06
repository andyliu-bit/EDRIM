"""D10 C2: choose the comboKR 2.0 CGKronRLS iteration count on split 2 (validation) by combination-cell surface-macro
RMSE, the selection metric used for every model in the test table (registered in DEVIATIONS.md, D10 supplement).
The authors' criterion (CV Pearson, select_iters.py) is recorded alongside. Split 3 is not read.

  .venv/Scripts/python.exe -I -B experiments/model_improvement/comparator_test/combokr_select.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "input_parity"))
import run_parity as rp  # noqa: E402

PUB = rp.gate.ROOT / "runs/model_improvement/combokr2_original_v1/public_split"
ITERS = (3, 5, 10)


def main():
    v = pd.read_parquet(rp.gate.S1 / "points_validation.parquet",
                        columns=["matrix_key", "row_i", "col_j", "response_percent", "hsa_lookup_percent"])
    n = len(v) // 9
    y = v.response_percent.to_numpy(float).reshape(n, 9)
    keys = v.matrix_key.to_numpy()[::9]
    rows = {}
    for k in ITERS:
        z = np.load(PUB / f"validation/pred_iters{k}.npz")
        if not np.array_equal(z["matrix_key"], keys):
            raise RuntimeError(f"iters{k}: predictions not aligned with the split-2 point table")
        p = z["pred"].astype(float)
        e = p - y
        rows[k] = dict(combination_rmse_percent=float(np.sqrt(np.mean(e ** 2))),
                       surface_macro_rmse_percent=float(np.mean(np.sqrt(np.mean(e ** 2, axis=1)))),
                       pearson=float(np.corrcoef(p.ravel(), y.ravel())[0, 1]),
                       nonfinite_matrices=int((~np.isfinite(p).all(1)).sum()))
    hsa = v.hsa_lookup_percent.to_numpy(float).reshape(n, 9)
    ref = dict(combination_rmse_percent=float(np.sqrt(np.mean((hsa - y) ** 2))),
               surface_macro_rmse_percent=float(np.mean(np.sqrt(np.mean((hsa - y) ** 2, axis=1)))))
    chosen = min(ITERS, key=lambda k: rows[k]["surface_macro_rmse_percent"])
    author = max(ITERS, key=lambda k: rows[k]["pearson"])
    info = dict(rule="min split-2 combination-cell surface-macro RMSE", selected_iters=chosen,
                authors_rule_max_pearson=author, agree=chosen == author, per_iteration=rows,
                lookup_hsa_validation=ref, matrices=n, registered="DEVIATIONS.md D10 supplement")
    (PUB / "validation/selection.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
