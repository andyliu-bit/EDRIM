"""O'Neil stage 2 comboKR 2.0 (DEVIATIONS D13 supplement), host side: iteration selection and prediction files.

--role validation: for each iteration in {3,5,10}, validation surface-macro RMSE (common.surface_macro_percent, the
  selection metric of every method), pooled RMSE and Pearson (the author's criterion); selects the iteration with the
  lowest surface-macro (ties -> fewer iterations), writes runs/oneil_v1/combokr2/selection.json and the validation
  predictions of the selected iteration through common.write_predictions("combokr2", None, ...).
--role test: writes the test predictions of the selected iteration (the container ran only that iteration).
  Test is never scored here: the test point table has no responses.
Predictions are converted from the author's percent scale to the ratio scale (/100), no clipping.

  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe experiments/oneil/combokr2/select_write.py --role validation
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common  # noqa: E402

common.install_guard()
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RUN = common.OUT / "combokr2"
ITERS = (3, 5, 10)


def to_frame(npz_path):
    z = np.load(npz_path)
    n = len(z["matrix_key"])
    k = np.tile(np.arange(16), n)
    return pd.DataFrame(dict(matrix_key=np.repeat(z["matrix_key"], 16), row_i=k % 4 + 1, col_j=k // 4 + 1,
                             prediction=z["pred"].astype(float).ravel() / 100.0))


def meta(role, k, extra):
    info = json.loads((RUN / role / "info.json").read_text(encoding="utf-8"))
    return dict(method="combokr2", seed=None, role=role, selected_iters=k, registered="DEVIATIONS.md D13 supplement",
                card="ONEIL_COMPARISON_CARD_20261020 v2 stage 2", scale="author percent / 100, not clipped",
                braid_fit_failed_author_fallback=info[f"iters{k}_braid_fit_failed"], train_used=info["train_used"],
                created_local_time=time.strftime("%Y-%m-%dT%H:%M:%S%z"), **extra)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True, choices=["validation", "test"])
    role = ap.parse_args().role
    points = common.load_points(role)
    if role == "validation":
        rows, frames = {}, {}
        for k in ITERS:
            f = points[common.KEY + ["response"]].merge(to_frame(RUN / f"validation/pred_iters{k}.npz"), on=common.KEY,
                                                        how="left", validate="one_to_one")
            p = f.prediction.to_numpy()
            rows[k] = dict(surface_macro_rmse_percent=common.surface_macro_percent(f, p),
                           pooled_rmse_percent=common.pooled_rmse_percent(f.response, p),
                           pearson=float(np.corrcoef(f.response, p)[0, 1]),
                           nonfinite_points=int((~np.isfinite(p)).sum()))
            frames[k] = f
        chosen = min(ITERS, key=lambda k: rows[k]["surface_macro_rmse_percent"])
        author = max(ITERS, key=lambda k: rows[k]["pearson"])
        sel = dict(rule="min validation surface-macro RMSE over the 16 combination cells (ties -> fewer iterations)",
                   selected_iters=chosen, authors_rule_max_pearson=author, agree=chosen == author, per_iteration=rows,
                   surfaces=int(points.matrix_key.nunique()), registered="DEVIATIONS.md D13 supplement")
        (RUN / "selection.json").write_text(json.dumps(sel, indent=1), encoding="utf-8")
        print(json.dumps(sel, indent=1))
        m = meta(role, chosen, dict(validation_surface_macro_percent=rows[chosen]["surface_macro_rmse_percent"],
                                    validation_pooled_rmse_percent=rows[chosen]["pooled_rmse_percent"]))
        path = common.write_predictions("combokr2", None, role, frames[chosen], m)
    else:
        chosen = json.loads((RUN / "selection.json").read_text(encoding="utf-8"))["selected_iters"]
        info = json.loads((RUN / "test/info.json").read_text(encoding="utf-8"))
        assert info["args"]["n_iters"] == str(chosen), (info["args"]["n_iters"], chosen)
        path = common.write_predictions("combokr2", None, role, to_frame(RUN / f"test/pred_iters{chosen}.npz"),
                                        meta(role, chosen, {}))
    print(f"{role}: iterations {chosen} -> {path}", flush=True)


if __name__ == "__main__":
    main()
