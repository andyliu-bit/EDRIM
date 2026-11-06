"""Shared paths, guard and prediction I/O for the O'Neil comparison (card E v2, DEVIATIONS D13).

Every method writes one prediction file per (method, seed, role) through `write_predictions`; the joint test scorer
(the only code allowed to decode test responses) reads them through `read_predictions`.
Predictions are on the viability ratio scale; metrics are reported x100 (%).
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OLD = Path("C:/LHZ/research/bisurfgnn")
OUT = ROOT / "runs/oneil_v1"
DRUG, CELL = OUT / "drug_features", OUT / "cell_features"
# D14 cause checks: val-only runs under runs/oneil_v1/analysis (never scored on test); H1 swaps the point tables
ANALYSIS = OUT / "analysis"
DATA = Path(os.environ.get("ONEIL_DATA_ROOT") or OUT / "data")
# smoke tests set ONEIL_PRED_ROOT to a scratch folder so they never write into the real prediction tree
PRED = Path(os.environ.get("ONEIL_PRED_ROOT") or OUT / "preds")
MOLFORMER = OUT / "molformer/ONEIL_drug_molformer_embeddings.pt"
GENE_CSV = CELL / "oneil_landmark_gex.csv"
SEEDS = (41, 42, 43, 44, 45)
METHODS = ("hsa", "xgb_id", "ours", "ours_e500", "deepsynba_4x4", "deepsynba_5x5", "ddprism", "combokr2", "combofm",
           "deepdrp", "cdcrrank")  # combofm: D21; deepdrp, cdcrrank: docs/DEEPDRP_/CDCRRANK_ADAPTATION_CARD_20261103.md
KEY = ["matrix_key", "row_i", "col_j"]
ROLES = ("train", "validation", "test")
# training / selection code must never open these (card E v2, "test 读取与一次性评分")
FORBIDDEN = ("oneil_confirmation_eval_v1", "oneil_per_surface.csv", "external_combokr2_oneil_newcell_v1",
             "oneil_fitted_Braids.npy", "supp_1_w2lrww.xls", "ONEIL_TEST_EVALUATED.marker")


def _guard(event, args):
    if event == "open" and isinstance(args[0], (str, bytes, Path)):
        path = str(args[0]).replace("\\", "/")
        if any(f in path for f in FORBIDDEN):
            raise PermissionError(f"Card E forbids opening {path} outside the joint test scorer")


def install_guard():
    """Call first thing in every training / selection / feature script (not in the joint test scorer)."""
    sys.addaudithook(_guard)


def analysis_run():
    """True when predictions go under runs/oneil_v1/analysis (D14): full-length runs allowed, test never scored."""
    return PRED.resolve().is_relative_to(ANALYSIS.resolve())


def load_points(role):
    """train / validation: full point table with `response`; test: inputs only (no response column)."""
    import pandas as pd
    if role not in ROLES:
        raise ValueError(role)
    frame = pd.read_parquet(DATA / ("points_test_inputs.parquet" if role == "test" else f"points_{role}.parquet"))
    if role == "test" and any(c.startswith("response") or c == "hsa_lookup_percent" for c in frame.columns):
        raise RuntimeError("test inputs must not carry responses")
    return frame


def pooled_rmse_percent(target, prediction):
    import numpy as np
    target, prediction = np.asarray(target, float), np.asarray(prediction, float)
    return float(np.sqrt(np.mean((prediction - target) ** 2)) * 100)


def surface_macro_percent(points, prediction):
    """Validation selection metric for every method: mean within-surface RMSE over the 16 combination cells, %."""
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    from bisurfgnn.learned_benchmark import surface_macro_rmse
    return surface_macro_rmse(points["response"].to_numpy(), prediction, points["matrix_key"].to_numpy()) * 100


def pred_path(method, seed, role):
    if method not in METHODS or role not in ("validation", "test"):
        raise ValueError((method, role))
    return PRED / method / ("deterministic" if seed is None else f"seed_{seed}") / f"{role}_predictions.parquet"


def write_predictions(method, seed, role, frame, meta=None):
    """frame: KEY columns + `prediction` (ratio scale). Must cover exactly the role's points, all finite."""
    import numpy as np
    points = load_points(role)[KEY]
    got = frame[KEY + ["prediction"]].copy()
    if len(got) != len(points) or got.duplicated(KEY).any():
        raise RuntimeError(f"{method}/{seed}/{role}: {len(got)} rows for {len(points)} points or duplicate keys")
    merged = points.merge(got, on=KEY, how="left", validate="one_to_one")
    if merged["prediction"].isna().any() or not np.isfinite(merged["prediction"]).all():
        raise RuntimeError(f"{method}/{seed}/{role}: missing or non-finite predictions")
    path = pred_path(method, seed, role)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(path, index=False)
    if meta is not None:
        path.with_name(f"{role}_meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    return path


def read_predictions(method, seed, role):
    import pandas as pd
    return pd.read_parquet(pred_path(method, seed, role))
