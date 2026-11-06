"""JointFeature-v1 summary from saved per-point validation predictions (no test access).

Usage: python summarize_joint_feature.py '<json: {model_label: [[run_dir, prefix], ...]}>' cand1 [cand2 ...]
References (T, numeric TabM, previous full model, low-dim XGB, mdx, v2) come from NumericSurface-v1 sources.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
TABLES = ROOT / "reports/tables"
spec = importlib.util.spec_from_file_location("nss", Path(__file__).with_name("summarize_numeric_surface.py"))
NSS = importlib.util.module_from_spec(spec); spec.loader.exec_module(NSS)
S = NSS.S
REFS = ["xgb_full_T", "C2_num_tabm_o1", "C3_numsurf_tabm_o1_g2", "xgb_lowdim", "mdx", "v2"]


def main():
    extra = {k: [(ROOT / r, p) for r, p in v] for k, v in json.loads(sys.argv[1]).items()}
    cands = sys.argv[2:]
    NSS.SOURCES = {k: NSS.SOURCES[k] for k in REFS} | extra
    va = S.validation_frame()
    merged, dups = NSS.collect(va)
    dups.to_csv(TABLES / "joint_feature_v1_seed_reproduction.csv", index=False)
    pd.DataFrame([dict(model=m, seed=s, surface_macro_rmse=float(v.mean())) for m, d in merged.items() for s, v in d.items()]) \
        .sort_values(["model", "seed"]).to_csv(TABLES / "joint_feature_v1_per_seed.csv", index=False)
    stats = pd.DataFrame([dict(model=m, **S.seed_stats([v.mean() for v in d.values()])) for m, d in merged.items() if d]).sort_values("mean")
    stats.to_csv(TABLES / "joint_feature_v1_models.csv", index=False); print(stats.to_string())
    comps = pd.DataFrame([S.compare(va, merged, c, r) for c in cands for r in REFS if len(merged[c]) >= 2])
    comps.to_csv(TABLES / "joint_feature_v1_comparisons.csv", index=False)
    print(comps[["candidate", "reference", "paired_seeds", "candidate_mean", "reference_mean", "relative_improvement",
                 "improved_seeds", "group_bootstrap_rel_ci95_low", "group_bootstrap_rel_ci95_high"]].round(6).to_string())


if __name__ == "__main__":
    main()
