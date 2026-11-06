"""CrossScreenAdditive-v1 summary (docs/CROSS_SCREEN_ADDITIVE_V1_PROTOCOL.md section 5).

Reads saved per-point predictions only. Existing cells of the 2x2 design come from earlier runs:
  CS  HSA_cs, XGB_cs           runs/model_improvement/cross_screen_mono_v1/cpu/cs
      anchored_cs              runs/model_improvement/cross_screen_mono_v1/gpu_r2/cs (bisurf_mono_v2)
      additive_cs              runs/model_improvement/cross_screen_additive_v1/gpu/cs (bisurf_additive_v2)
  SP  XGB_sp                   runs/model_improvement/algorithm_search_v1/xgb_final
      anchored_sp              runs/model_improvement/factorial_exchange_v1 (bisurf_mono_v2)
      additive_sp              runs/model_improvement/cross_screen_additive_v1/gpu/sp (bisurf_additive_v2)
CS rows are scored on the CS-covered validation subset (8442 points); SP rows on the full frozen validation set
(9000 points) and, as a supplement, on the covered subset. Statistics reuse summarize_algorithm_search
(per_surface, seed_stats, compare -> group_bootstrap, 2000 draws, seed 20261009), as in CrossScreenMono-v1.

Relative change = (RMSE_candidate - RMSE_reference) / RMSE_reference (negative = lower error).

Usage: .venv/Scripts/python.exe experiments/model_improvement/cross_screen_additive/summarize.py
Outputs: runs/model_improvement/cross_screen_additive_v1/summary/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cross_screen_mono"))
from common import CS_DIR, ROOT, RUN_ROOT as CSM_ROOT, append_log, sha256_file, summ  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bisurfgnn.pretrain.nci60 import forbid_test_responses  # noqa: E402

RUN_ROOT = ROOT / "runs/model_improvement/cross_screen_additive_v1"
NEW = RUN_ROOT / "gpu"
OUT = RUN_ROOT / "summary"
SEEDS = ["41", "42"]
CS_JOBS = {"XGB_cs": CSM_ROOT / "cpu/cs/xgb_residual_d10_m1_seed_{s}_primary",
           "anchored_cs": CSM_ROOT / "gpu_r2/cs/bisurf_mono_v2_seed_{s}_primary",
           "additive_cs": NEW / "cs/bisurf_additive_v2_seed_{s}_primary"}
SP_JOBS = {"XGB_sp": ROOT / "runs/model_improvement/algorithm_search_v1/xgb_final/xgb_residual_d10_m1_seed_{s}_primary",
           "anchored_sp": ROOT / "runs/model_improvement/factorial_exchange_v1/bisurf_mono_v2_seed_{s}_primary",
           "additive_sp": NEW / "sp/bisurf_additive_v2_seed_{s}_primary"}
HSA_BINS = [0.0, 0.1, 0.3, 0.6, np.inf]
HSA_LABELS = ["<0.1", "0.1-0.3", "0.3-0.6", ">=0.6"]


def predictions(path: Path, order: np.ndarray) -> np.ndarray:
    p = pd.read_csv(path / "predictions.csv")
    if not (p.source_row.to_numpy() == order).all():
        raise RuntimeError(f"Prediction order mismatch: {path}")
    return p.prediction.to_numpy(np.float64)


def rel_change(c: dict, scope: str) -> dict:
    """compare() reports improvement (ref - cand)/ref; convert to relative change (cand - ref)/ref."""
    return dict(scope=scope, candidate=c["candidate"], reference=c["reference"], paired_seeds=c["paired_seeds"],
                candidate_mean=c["candidate_mean"], reference_mean=c["reference_mean"],
                relative_change=-c["relative_improvement"], seeds_candidate_lower=c["improved_seeds"],
                group_bootstrap_rel_change_ci95_low=-c["group_bootstrap_rel_ci95_high"],
                group_bootstrap_rel_change_ci95_high=-c["group_bootstrap_rel_ci95_low"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with forbid_test_responses() as guard:
        va_full = summ.validation_frame()   # frozen 9000-point validation frame (comparator_gpu_v2 order)
    cov_ids = pd.read_csv(CS_DIR / "covered_validation_ids.csv")
    covered = va_full.source_combodrugseq.isin(set(cov_ids.source_combodrugseq)).to_numpy()
    va = va_full.loc[covered].reset_index(drop=True)
    if not (va.source_combodrugseq.to_numpy() == cov_ids.source_combodrugseq.to_numpy()).all():
        raise RuntimeError("Covered validation order mismatch")
    # saved predictions are keyed by source_combodrugseq (column source_row), as in CrossScreenMono-v1
    order_cov, order_full = va.source_combodrugseq.to_numpy(), va_full.source_combodrugseq.to_numpy()
    # ---------------- point predictions
    point_cs = {"HSA_cs": {s: predictions(CSM_ROOT / "cpu/cs/hsa_cs_seed_None_primary", order_cov) for s in SEEDS}}
    for key, pattern in CS_JOBS.items():
        point_cs[key] = {s: predictions(Path(str(pattern).format(s=s)), order_cov) for s in SEEDS}
    hsa_sp_full = np.minimum(va_full.mono_response_a, va_full.mono_response_b).to_numpy(float)
    point_sp = {"HSA_sp": {s: hsa_sp_full for s in SEEDS}}
    for key, pattern in SP_JOBS.items():
        point_sp[key] = {s: predictions(Path(str(pattern).format(s=s)), order_full) for s in SEEDS}
    # ---------------- per-surface RMSE
    surf_cs = {k: {s: summ.per_surface(va, p) for s, p in v.items()} for k, v in point_cs.items()}
    surf_sp = {k: {s: summ.per_surface(va_full, p) for s, p in v.items()} for k, v in point_sp.items()}
    surf_sp_cov = {f"{k}_cov": {s: summ.per_surface(va, p[covered]) for s, p in v.items()} for k, v in point_sp.items()}
    checks = {}
    # consistency with CrossScreenMono-v1 (summary/models.csv): anchored_cs and XGB_cs seeds 41/42
    v1 = pd.read_csv(CSM_ROOT / "summary/models.csv").set_index("model")
    for key, v1_key in (("anchored_cs", "BiSurfGNN_cs"), ("XGB_cs", "XGB_cs"), ("HSA_cs", "HSA_cs")):
        for s in SEEDS if key != "HSA_cs" else ["41"]:
            diff = abs(float(surf_cs[key][s].mean()) - float(v1.loc[v1_key, f"seed_{s}"]))
            checks[f"{key}_seed{s}_vs_v1_summary_abs_diff"] = diff
            if diff > 1e-9:
                raise RuntimeError(f"{key} seed {s} differs from the CrossScreenMono-v1 summary")
    # ---------------- model table
    rows = []
    for scope, table, hsa_key in (("CS covered (8442)", surf_cs, "HSA_cs"), ("SP full (9000)", surf_sp, "HSA_sp"),
                                  ("SP covered (8442)", surf_sp_cov, "HSA_sp_cov")):
        hsa = float(table[hsa_key]["41"].mean())
        for key, seeds in table.items():
            values = [float(v.mean()) for v in seeds.values()]
            stats = summ.seed_stats(values) if not key.startswith("HSA") else dict(mean=values[0], sd=float("nan"))
            rows.append(dict(scope=scope, model=key, seed_41=values[0], seed_42=values[1], mean=stats["mean"],
                             sd=stats["sd"], rel_change_vs_HSA=(stats["mean"] - hsa) / hsa))
    models = pd.DataFrame(rows)
    models.to_csv(OUT / "models.csv", index=False)
    # ---------------- comparisons (seeds 41/42 paired; group bootstrap on seed-averaged per-surface RMSE)
    comp = []
    for cand, ref in [("additive_cs", "anchored_cs"), ("additive_cs", "XGB_cs"), ("additive_cs", "HSA_cs"),
                      ("anchored_cs", "XGB_cs"), ("anchored_cs", "HSA_cs"), ("XGB_cs", "HSA_cs")]:
        comp.append(rel_change(summ.compare(va, surf_cs, cand, ref), "CS covered (8442)"))
    for cand, ref in [("additive_sp", "anchored_sp"), ("additive_sp", "XGB_sp"), ("additive_sp", "HSA_sp"),
                      ("anchored_sp", "XGB_sp"), ("anchored_sp", "HSA_sp")]:
        comp.append(rel_change(summ.compare(va_full, surf_sp, cand, ref), "SP full (9000)"))
    for cand, ref in [("additive_sp_cov", "anchored_sp_cov"), ("additive_sp_cov", "XGB_sp_cov")]:
        comp.append(rel_change(summ.compare(va, surf_sp_cov, cand, ref), "SP covered (8442)"))
    comparisons = pd.DataFrame(comp)
    comparisons.to_csv(OUT / "comparisons.csv", index=False)
    # ---------------- pre-registered readout (protocol section 5)
    m = {k: {s: float(v.mean()) for s, v in seeds.items()} for k, seeds in {**surf_cs, **surf_sp}.items()}
    per_seed = [dict(seed=s, anchored_cs=m["anchored_cs"][s], additive_cs=m["additive_cs"][s], XGB_cs=m["XGB_cs"][s],
                     additive_vs_anchored_cs=(m["additive_cs"][s] - m["anchored_cs"][s]) / m["anchored_cs"][s],
                     additive_vs_XGB_cs=(m["additive_cs"][s] - m["XGB_cs"][s]) / m["XGB_cs"][s],
                     anchored_sp=m["anchored_sp"][s], additive_sp=m["additive_sp"][s],
                     additive_vs_anchored_sp=(m["additive_sp"][s] - m["anchored_sp"][s]) / m["anchored_sp"][s])
                for s in SEEDS]
    pd.DataFrame(per_seed).to_csv(OUT / "readout_per_seed.csv", index=False)
    mean = {k: float(np.mean(list(v.values()))) for k, v in m.items()}
    closure = (mean["anchored_cs"] - mean["additive_cs"]) / (mean["anchored_cs"] - mean["XGB_cs"])
    lower_both = all(r["additive_cs"] < r["anchored_cs"] for r in per_seed)
    if lower_both and closure >= 0.5:
        verdict = "major cause"
    elif not lower_both or closure < 0.2:
        verdict = "not the main cause"
    else:
        verdict = "partial cause"
    sp_worse_both = all(r["additive_vs_anchored_sp"] >= 0.01 for r in per_seed)
    follow_up = all(r["additive_vs_XGB_cs"] <= -0.01 for r in per_seed)
    readout = dict(
        Q1_additive_vs_anchored_cs_per_seed={r["seed"]: r["additive_vs_anchored_cs"] for r in per_seed},
        Q2_gap_closure=closure, additive_cs_lower_on_both_seeds=lower_both, verdict=verdict,
        verdict_rule="major: lower on both seeds and closure >= 50%; not main: not lower on both seeds or closure < 20%; else partial",
        Q3_additive_vs_anchored_sp_per_seed={r["seed"]: r["additive_vs_anchored_sp"] for r in per_seed},
        sp_additive_worse_by_1pct_on_both_seeds=sp_worse_both,
        follow_up_5_seed_protocol=follow_up,
        follow_up_rule="additive_cs better than XGB_cs by >= 1% on both seeds")
    # ---------------- Q4 mechanism: bins of the CS anchor x sign of the true residual (seed-averaged predictions)
    y = va.response.to_numpy(float)
    hsa = point_cs["HSA_cs"]["41"]
    avg = {k: np.mean([v[s] for s in SEEDS], axis=0) for k, v in point_cs.items()}
    bins = pd.cut(hsa, HSA_BINS, right=False, labels=HSA_LABELS)
    sign = np.where(y - hsa > 0, "y>HSA_cs", "y<=HSA_cs")
    mech = []
    for b in HSA_LABELS:
        for sg in ("y>HSA_cs", "y<=HSA_cs", "all"):
            mask = (bins == b) & ((sign == sg) | (sg == "all"))
            if mask.sum() == 0:
                continue
            row = dict(hsa_cs_bin=b, residual_sign=sg, points=int(mask.sum()),
                       mean_true_minus_anchor=float(np.mean(y[mask] - hsa[mask])),
                       mean_abs_true_minus_anchor=float(np.mean(np.abs(y[mask] - hsa[mask]))))
            for k, p in avg.items():
                row[f"{k}_point_rmse"] = float(np.sqrt(np.mean((p[mask] - y[mask]) ** 2)))
                if k != "HSA_cs":
                    row[f"{k}_mean_correction"] = float(np.mean(p[mask] - hsa[mask]))
            mech.append(row)
    pd.DataFrame(mech).to_csv(OUT / "mechanism_bins.csv", index=False)
    # ---------------- descriptive: range of the unclipped additive predictions
    ranges = {k: dict(min=float(min(v[s].min() for s in SEEDS)), max=float(max(v[s].max() for s in SEEDS)),
                      fraction_below_0=float(np.mean([np.mean(v[s] < 0) for s in SEEDS])),
                      fraction_above_3=float(np.mean([np.mean(v[s] > 3) for s in SEEDS])))
              for k, v in {**point_cs, **point_sp}.items() if k.startswith(("additive", "anchored"))}
    decision = dict(protocol="CrossScreenAdditive-v1", readout=readout, per_seed=per_seed, prediction_ranges=ranges,
                    checks=checks, evaluation=dict(cs_points=int(len(va)), cs_surfaces=int(va.surface_replicate_id.nunique()),
                                                   sp_points=int(len(va_full)), sp_surfaces=int(va_full.surface_replicate_id.nunique())),
                    parquet_reads=guard.accesses, test_response_rows_read=0,
                    inputs={str(p.relative_to(ROOT)): sha256_file(p) for p in
                            [CS_DIR / "covered_validation_ids.csv", CS_DIR / "manifest.json", NEW / "summary.json"]})
    (OUT / "decision.json").write_text(json.dumps(decision, indent=1, default=float), encoding="utf-8")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(models.to_string()); print(comparisons.to_string()); print(pd.DataFrame(mech).to_string())
    print(json.dumps(decision, indent=1, default=float))
    append_log(RUN_ROOT / "STEP_LOG.md", "SUMMARY written: " + "; ".join(
        f"{r.scope}:{r.model}={r.mean:.6f}" for r in models.itertuples()) + f"; verdict={verdict}; closure={closure:.3f}")


if __name__ == "__main__":
    main()
