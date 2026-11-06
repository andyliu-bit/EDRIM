"""Self-check of runs/oneil_v1/data (prepare_points.py): counts, keys, doses, anchors, and no test responses.

  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe experiments/oneil/data/check_points.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "runs/oneil_v1/data"
EXPRESSION = ROOT.parent / "bisurfgnn/runs/audit_revision/oneil_depmap_coverage_v1/expression.npz"
EXPECTED = {"train": (349, 10590), "validation": (116, 3528), "test": (118, 3624)}
TEST_COLUMNS = {"matrix_key", "surface_index", "pair_id", "deepsynba_cell", "cell_id", "biological_cell_id",
                "deepsynba_drug_a", "deepsynba_drug_b", "drug_a_id", "drug_b_id", "row_i", "col_j", "dose_a_M", "dose_b_M",
                "mono_a_ratio_raw", "mono_b_ratio_raw", "batch", "mono_a_percent", "mono_b_percent", "dose_a_model",
                "dose_b_model", "mono_response_a", "mono_response_b"}
sys.stdout.reconfigure(encoding="utf-8")


def main():
    t = {r: pd.read_parquet(DATA / f"points_{'test_inputs' if r == 'test' else r}.parquet") for r in EXPECTED}
    report = json.loads((DATA / "prepare_report.json").read_text(encoding="utf-8"))
    checks = {}
    test_cols = set(t["test"].columns)
    checks["test_columns_are_inputs_only"] = test_cols == TEST_COLUMNS and not any(
        ("response" in c and not c.startswith("mono_response")) or "hsa" in c or "viab" in c for c in test_cols)
    checks["dev_columns_are_test_plus_targets"] = all(
        set(t[r].columns) == TEST_COLUMNS | {"response", "response_percent", "hsa_lookup_percent"} for r in ("train", "validation"))
    for r, f in t.items():
        checks[f"{r}_pairs_surfaces"] = (f.pair_id.nunique(), f.matrix_key.nunique()) == EXPECTED[r]
        g = f.groupby("matrix_key")
        checks[f"{r}_16_cells_4x4"] = bool((g.size() == 16).all() and (g.row_i.nunique() == 4).all()
                                           and (g.col_j.nunique() == 4).all()
                                           and not f.duplicated(["matrix_key", "row_i", "col_j"]).any()
                                           and f.row_i.between(1, 4).all() and f.col_j.between(1, 4).all())
        # numeric dose order: dose rises with row_i / col_j, and is constant along the other axis
        da = f.pivot_table(index="matrix_key", columns="row_i", values="dose_a_M", aggfunc=["min", "max"])
        db = f.pivot_table(index="matrix_key", columns="col_j", values="dose_b_M", aggfunc=["min", "max"])
        checks[f"{r}_dose_numeric_order"] = bool(
            (da["min"].to_numpy() == da["max"].to_numpy()).all() and (np.diff(da["min"].to_numpy(), axis=1) > 0).all()
            and (db["min"].to_numpy() == db["max"].to_numpy()).all() and (np.diff(db["min"].to_numpy(), axis=1) > 0).all())
        checks[f"{r}_ids"] = bool((f.drug_a_id < f.drug_b_id).all() and (f.pair_id == f.drug_a_id + "|" + f.drug_b_id).all()
                                  and (f.matrix_key == f.batch + "|" + f.cell_id + "|" + f.pair_id).all())
        checks[f"{r}_dose_model"] = bool(np.allclose(f.dose_a_model, np.log10(1 + f.dose_a_M / 1e-12), rtol=0, atol=1e-12)
                                         and np.allclose(f.dose_b_model, np.log10(1 + f.dose_b_M / 1e-12), rtol=0, atol=1e-12)
                                         and f[["dose_a_model", "dose_b_model"]].stack().between(2.04, 8.41).all())
        checks[f"{r}_anchors"] = bool(all(f[f"mono_response_{x}"].between(0, 3).all()
                                          and (f[f"mono_response_{x}"] == f[f"mono_{x}_ratio_raw"].clip(0, 3)).all()
                                          and np.isfinite(f[f"mono_{x}_ratio_raw"]).all() for x in ("a", "b")))
        if r != "test":
            checks[f"{r}_response_finite"] = bool(np.isfinite(f.response).all() and np.allclose(f.response_percent, 100 * f.response))
    keys = {r: set(f.matrix_key) for r, f in t.items()}
    pairs = {r: set(f.pair_id) for r, f in t.items()}
    checks["roles_disjoint"] = not (keys["train"] & keys["validation"] or keys["train"] & keys["test"]
                                    or keys["validation"] & keys["test"] or pairs["train"] & pairs["validation"]
                                    or pairs["train"] & pairs["test"] or pairs["validation"] & pairs["test"])
    allf = pd.concat(t.values())
    anchors = pd.concat([allf[["batch", "cell_id", f"drug_{x}_id", f"dose_{x}_M", f"mono_{x}_ratio_raw"]].set_axis(
        ["batch", "cell", "drug", "dose", "mono"], axis=1) for x in ("a", "b")]).drop_duplicates()
    checks["anchor_single_valued_across_roles"] = not anchors.duplicated(["batch", "cell", "drug", "dose"]).any()
    cells, drugs = pd.read_csv(DATA / "cells_oneil.csv"), pd.read_csv(DATA / "drugs_oneil.csv")
    expr_cells = set(np.load(EXPRESSION, allow_pickle=False)["cells"].tolist())
    checks["cells_30_depmap"] = len(cells) == 30 and set(cells.cell_id) == set(allf.cell_id) == expr_cells
    checks["drugs_38"] = len(drugs) == 38 and set(drugs.drug_id) == set(allf.drug_a_id) | set(allf.drug_b_id) \
        and drugs.canonical_smiles.notna().all()
    pr = pd.read_csv(DATA / "pairs_by_role.csv")
    checks["pairs_by_role"] = all((pr.role == r).sum() == EXPECTED[r][0] and pr[pr.role == r].surfaces.sum() == EXPECTED[r][1]
                                  for r in EXPECTED)
    checks["test_batches_3498_42_84"] = t["test"].drop_duplicates("matrix_key").batch.value_counts().to_dict() == {"1": 3498, "2": 42, "3": 84}
    a = report["anchor_path_reproduces_v2_dev"]
    checks["anchor_path_reproduces_v2_dev"] = a["points"] == a["identical"] == 16 * (10590 + 3528)
    # ID-only context statistic, recomputed
    ctx = {r: set(map(tuple, pd.concat([f[["batch", "cell_id", f"drug_{x}_id"]].set_axis(["b", "c", "d"], axis=1)
                                         for x in ("a", "b")]).drop_duplicates().to_numpy())) for r, f in t.items()}
    unseen = ctx["test"] - ctx["train"]
    ts = t["test"].drop_duplicates("matrix_key")
    hit = sum((b, c, x) in unseen or (b, c, y) in unseen for b, c, x, y in ts[["batch", "cell_id", "drug_a_id", "drug_b_id"]].to_numpy())
    stat = (len(unseen), len(ctx["test"]), hit)
    checks["context_statistic_matches_report"] = stat == tuple(report["test_contexts_unseen_in_train"][k]
                                                               for k in ("unseen", "total", "surfaces_affected"))
    for k, v in checks.items():
        print(f"{'PASS' if v else 'FAIL'}  {k}")
    print("test contexts unseen in train (unseen, total, surfaces affected):", stat)
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
