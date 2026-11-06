"""CrossScreenMono-v1 step 3 summary (docs/CROSS_SCREEN_MONO_V1_PROTOCOL.md section 4).

Reads saved per-point predictions only (validation shard for targets and the same-plate HSA reference row).
Statistics reuse experiments/model_improvement/analysis/summarize_algorithm_search.py (per_surface, seed_stats,
compare -> group_bootstrap: prediction groups resampled within held-out clusters, 2000 draws, seed 20261009).

Relative change = (RMSE_method - RMSE_reference) / RMSE_reference (negative = lower error than the reference).

Usage: .venv/Scripts/python.exe experiments/model_improvement/cross_screen_mono/summarize.py
Outputs: runs/model_improvement/cross_screen_mono_v1/summary/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CS_DIR, ROOT, RUN_ROOT, append_log, sha256_file, summ  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bisurfgnn.pretrain.nci60 import forbid_test_responses  # noqa: E402

OUT = RUN_ROOT / "summary"
SEARCH = ROOT / "runs/model_improvement/algorithm_search_v1"
SEEDS = ["41", "42"]
REFERENCE_RUNS = {  # same-plate reference rows (saved predictions, full frozen validation order)
    "XGB_sp": [SEARCH / "xgb_final" / f"xgb_residual_d10_m1_seed_{s}_primary" for s in range(41, 46)],
    "v2_sp": [ROOT / "runs/model_improvement/factorial_exchange_v1" / f"bisurf_mono_v2_seed_{s}_primary" for s in range(41, 46)],
}
V2_SCREEN = [SEARCH / "screen_b1" / f"bisurf_mono_v2_seed_{s}_primary" for s in (41, 42)]
# gpu/ completed every job but failed its end-of-run source check (summarize.py was edited during the run);
# the clean rerun gpu_r2/ is the reported run (FROZEN_RECORD.md, 17:15 entry)
CS_RUNS = {"XGB_cs": RUN_ROOT / "cpu/cs", "BiSurfGNN_cs": RUN_ROOT / "gpu_r2/cs"}
CS_MODEL = {"XGB_cs": "xgb_residual_d10_m1", "BiSurfGNN_cs": "bisurf_mono_v2"}


def predictions(path: Path, order: np.ndarray) -> np.ndarray:
    p = pd.read_csv(path / "predictions.csv")
    p = p.set_index("source_row").loc[order].reset_index() if len(p) != len(order) else p
    if not (p.source_row.to_numpy() == order).all():
        raise RuntimeError(f"Prediction order mismatch: {path}")
    return p.prediction.to_numpy(np.float64)


def rel_change(c: dict) -> dict:
    """compare() reports improvement (ref - cand)/ref; convert to relative change (cand - ref)/ref."""
    return dict(candidate=c["candidate"], reference=c["reference"], paired_seeds=c["paired_seeds"],
                candidate_mean=c["candidate_mean"], reference_mean=c["reference_mean"],
                relative_change=-c["relative_improvement"], seeds_candidate_lower=c["improved_seeds"],
                group_bootstrap_rel_change_ci95_low=-c["group_bootstrap_rel_ci95_high"],
                group_bootstrap_rel_change_ci95_high=-c["group_bootstrap_rel_ci95_low"],
                paired_rmse_difference_cand_minus_ref=-c["paired_rmse_reduction"],
                seed_t_ci95_low=-c["seed_ci95_high"], seed_t_ci95_high=-c["seed_ci95_low"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with forbid_test_responses() as guard:
        va_full = summ.validation_frame()           # frozen 9000-point validation frame (comparator_gpu_v2 order)
        assign = pd.read_parquet(ROOT.parent / "bisurfgnn/data/processed/one_new_scaffold_benchmark_unclipped/surface_assignments.parquet",
                                 columns=["surface_replicate_id", "heldout_drug_id"])
    cov_ids = pd.read_csv(CS_DIR / "covered_validation_ids.csv")
    covered = va_full.source_combodrugseq.isin(set(cov_ids.source_combodrugseq)).to_numpy()
    va = va_full.loc[covered].reset_index(drop=True)
    if not (va.source_combodrugseq.to_numpy() == cov_ids.source_combodrugseq.to_numpy()).all():
        raise RuntimeError("Covered validation order mismatch")
    order_full, order = va_full.source_combodrugseq.to_numpy(), va.source_combodrugseq.to_numpy()
    surfaces, macro_rows, checks = {}, [], {}
    # -------- CS rows (this pilot)
    hsa_cs = predictions(RUN_ROOT / "cpu/cs/hsa_cs_seed_None_primary", order)
    surfaces["HSA_cs"] = {s: summ.per_surface(va, hsa_cs) for s in SEEDS}
    for key, run in CS_RUNS.items():
        surfaces[key] = {}
        for s in SEEDS:
            job = run / f"{CS_MODEL[key]}_seed_{s}_primary"
            if (job / "predictions.csv").exists():
                surfaces[key][s] = summ.per_surface(va, predictions(job, order))
        if not surfaces[key]:
            del surfaces[key]
    # -------- same-plate reference rows, re-scored on the covered subset
    hsa_sp = np.minimum(va.mono_response_a, va.mono_response_b).to_numpy(float)
    surfaces["HSA_sp"] = {s: summ.per_surface(va, hsa_sp) for s in SEEDS}
    full_hsa = summ.per_surface(va_full, np.minimum(va_full.mono_response_a, va_full.mono_response_b)).mean()
    checks["HSA_sp_full_validation"] = float(full_hsa)
    for key, jobs in REFERENCE_RUNS.items():
        surfaces[f"{key}_5seed"] = {}
        full = []
        for job in jobs:
            seed = job.name.split("_seed_")[1].split("_")[0]
            pred_full = predictions(job, order_full)
            full.append(summ.per_surface(va_full, pred_full).mean())
            surfaces[f"{key}_5seed"][seed] = summ.per_surface(va, pred_full[covered])
        surfaces[key] = {s: surfaces[f"{key}_5seed"][s] for s in SEEDS}
        checks[f"{key}_full_validation_5seed_mean"] = float(np.mean(full))
    for job in V2_SCREEN:  # screen_b1 v2 seeds 41/42 must equal factorial_exchange_v1 seeds 41/42
        seed = job.name.split("_seed_")[1].split("_")[0]
        diff = float(np.max(np.abs(summ.per_surface(va, predictions(job, order_full)[covered]).to_numpy()
                                   - surfaces["v2_sp"][seed].to_numpy())))
        checks[f"v2_screen_b1_vs_factorial_seed{seed}_max_surface_rmse_diff"] = diff
        if diff > 1e-9:
            raise RuntimeError("v2 seed provenance mismatch")
    # -------- per-model table
    info = {"HSA_cs": "CS", "XGB_cs": "CS", "BiSurfGNN_cs": "CS", "HSA_sp": "同板（参照）", "XGB_sp": "同板（参照）",
            "v2_sp": "同板（参照）", "XGB_sp_5seed": "同板（参照）", "v2_sp_5seed": "同板（参照）"}
    hsa_cs_mean = float(surfaces["HSA_cs"]["41"].mean())
    for key, seeds in surfaces.items():
        values = [float(v.mean()) for v in seeds.values()]
        stats = summ.seed_stats(values) if not key.startswith("HSA") else dict(seeds=0, mean=values[0], sd=float("nan"))
        macro_rows.append(dict(model=key, information=info[key], **{f"seed_{s}": float(v.mean()) for s, v in seeds.items()},
                               **{k: stats[k] for k in ("seeds", "mean", "sd")},
                               rel_change_vs_HSA_cs=(stats["mean"] - hsa_cs_mean) / hsa_cs_mean,
                               rel_change_vs_HSA_sp=(stats["mean"] - float(surfaces["HSA_sp"]["41"].mean()))
                               / float(surfaces["HSA_sp"]["41"].mean())))
    models = pd.DataFrame(macro_rows)
    models.to_csv(OUT / "models.csv", index=False)
    # -------- comparisons (seeds 41/42 paired; HSA rows are seed-independent)
    pairs = [("XGB_cs", "HSA_cs"), ("BiSurfGNN_cs", "HSA_cs"), ("BiSurfGNN_cs", "XGB_cs"),
             ("HSA_cs", "HSA_sp"), ("XGB_cs", "XGB_sp"), ("BiSurfGNN_cs", "v2_sp"),
             ("XGB_sp", "HSA_sp"), ("v2_sp", "HSA_sp"), ("v2_sp", "XGB_sp"),
             ("XGB_sp_5seed", "HSA_sp"), ("v2_sp_5seed", "HSA_sp"), ("v2_sp_5seed", "XGB_sp_5seed"),
             # gap in the AlgorithmSearch-v1 convention (XGB relative to BiSurfGNN; -4.63% on the full sample)
             ("XGB_cs", "BiSurfGNN_cs"), ("XGB_sp", "v2_sp"), ("XGB_sp_5seed", "v2_sp_5seed")]
    comp = []
    for cand, ref in pairs:
        if cand in surfaces and ref in surfaces:
            if ref == "HSA_sp" and cand.endswith("_5seed"):
                hsa5 = {s: surfaces["HSA_sp"]["41"] for s in surfaces[cand]}
                c = summ.compare(va, {**surfaces, "HSA_sp_5": hsa5}, cand, "HSA_sp_5"); c["reference"] = "HSA_sp"
            else:
                c = summ.compare(va, surfaces, cand, ref)
            comp.append(rel_change(c))
    comparisons = pd.DataFrame(comp)
    comparisons.to_csv(OUT / "comparisons.csv", index=False)
    # -------- per-seed paired values and the section-4 readout
    per_seed = []
    if "BiSurfGNN_cs" in surfaces and "XGB_cs" in surfaces:
        for s in SEEDS:
            if s in surfaces["BiSurfGNN_cs"] and s in surfaces["XGB_cs"]:
                b, x = float(surfaces["BiSurfGNN_cs"][s].mean()), float(surfaces["XGB_cs"][s].mean())
                per_seed.append(dict(seed=s, BiSurfGNN_cs=b, XGB_cs=x, improvement_of_BiSurfGNN_over_XGB=(x - b) / x))
    sp_seed = [dict(seed=s, v2_sp=float(surfaces["v2_sp"][s].mean()), XGB_sp=float(surfaces["XGB_sp"][s].mean()),
                    improvement_of_v2_over_XGB=(float(surfaces["XGB_sp"][s].mean()) - float(surfaces["v2_sp"][s].mean()))
                    / float(surfaces["XGB_sp"][s].mean())) for s in SEEDS]
    pd.DataFrame(per_seed).to_csv(OUT / "readout_per_seed.csv", index=False)
    pd.DataFrame(sp_seed).to_csv(OUT / "same_plate_per_seed.csv", index=False)
    readout = None
    if len(per_seed) == 2:
        both = all(r["improvement_of_BiSurfGNN_over_XGB"] >= 0.01 for r in per_seed)
        readout = dict(rule="BiSurfGNN_cs better than XGB_cs by >=1% on both seeds (paired by seed)",
                       per_seed=per_seed, satisfied=both,
                       consequence="register a separate 5-seed protocol" if both else "record as is; no follow-up implied")
    # -------- descriptive: per held-out target drug and by extrapolation status (seed-averaged predictions)
    meta = va[["surface_replicate_id"]].drop_duplicates().merge(assign, on="surface_replicate_id", how="left").set_index("surface_replicate_id")
    by_drug = pd.DataFrame({k: pd.concat(list(v.values()), axis=1).mean(axis=1) for k, v in surfaces.items()
                            if not k.endswith("_5seed")})
    by_drug["heldout_drug_id"] = meta.loc[by_drug.index, "heldout_drug_id"].to_numpy()
    drug_table = by_drug.groupby("heldout_drug_id").mean()
    drug_table.insert(0, "surfaces", by_drug.groupby("heldout_drug_id").size())
    drug_table.to_csv(OUT / "per_heldout_drug.csv")
    cs = pd.read_csv(CS_DIR / "cs_points_validation.csv")
    cs = cs[cs.covered].reset_index(drop=True)
    extrap = (cs.extrapolated_a | cs.extrapolated_b).to_numpy()
    point_pred = {"HSA_cs": hsa_cs, "HSA_sp": hsa_sp}
    for key, run in CS_RUNS.items():
        if key in surfaces:
            point_pred[key] = np.mean([predictions(run / f"{CS_MODEL[key]}_seed_{s}_primary", order) for s in surfaces[key]], axis=0)
    for key, jobs in REFERENCE_RUNS.items():
        point_pred[key] = np.mean([predictions(j, order_full)[covered] for j in jobs[:2]], axis=0)
    y = va.response.to_numpy(float)
    split = [dict(model=k, subset=name, points=int(m.sum()), point_rmse=float(np.sqrt(np.mean((p[m] - y[m]) ** 2))))
             for k, p in point_pred.items() for name, m in (("no_arm_extrapolated", ~extrap), ("any_arm_extrapolated", extrap))]
    pd.DataFrame(split).to_csv(OUT / "by_extrapolation_point_rmse.csv", index=False)
    decision = dict(protocol="CrossScreenMono-v1", evaluation_points=int(len(va)), evaluation_surfaces=int(va.surface_replicate_id.nunique()),
                    evaluation_groups=int(va.prediction_group_id.nunique()), evaluation_clusters=int(va.heldout_cluster_id.nunique()),
                    readout=readout, checks=checks, parquet_reads=guard.accesses, test_response_rows_read=0,
                    inputs={str(p.relative_to(ROOT)): sha256_file(p) for p in [CS_DIR / "covered_validation_ids.csv", CS_DIR / "manifest.json"]})
    (OUT / "decision.json").write_text(json.dumps(decision, indent=1, default=float), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(models.to_string()); print(comparisons.to_string()); print(json.dumps(decision, indent=1, default=float))
    append_log(RUN_ROOT / "STEP_LOG.md", "SUMMARY written: " + "; ".join(f"{r.model}={r.mean:.6f}" for r in models.itertuples())
               + f"; readout={readout['satisfied'] if readout else 'pending'}")


if __name__ == "__main__":
    main()
