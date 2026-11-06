"""Comparator diagnostic: which feature groups carry the XGBoost residual baseline.

Runs the frozen xgb_residual_d10_m1 configuration through the unchanged search
runner with one feature group retained. Diagnostic only; not a candidate and not
a target. Feature columns follow `run_algorithm_search.xgb_features`:
0-6 monotherapy/dose/pair scalars, 7-22 drug A PCA, 23-38 drug B PCA, 39-54 cell PCA.

Usage: python diagnose_xgb_features.py <subset> --config ... --output-dir ...
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("search_runner", HERE / "run_algorithm_search.py")
runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)

SUBSETS = {
    "monodose": list(range(0, 7)),
    "monodose_drug": list(range(0, 39)),
    "monodose_cell": list(range(0, 7)) + list(range(39, 55)),
}

if __name__ == "__main__":
    subset = sys.argv.pop(1)
    keep = SUBSETS[subset]
    original = runner.xgb_features

    def restricted(*args, **kwargs):
        x = original(*args, **kwargs)
        if x.shape[1] != 55:
            raise RuntimeError(f"Unexpected feature width {x.shape[1]}")
        return np.ascontiguousarray(x[:, keep])

    runner.xgb_features = restricted
    runner.main()
