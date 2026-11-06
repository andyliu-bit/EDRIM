"""NumericSurface-v1 summary from saved per-point validation predictions (no test access)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "runs/model_improvement"
NS = RUNS / "numeric_surface_v1"
TABLES = ROOT / "reports/tables"
spec = importlib.util.spec_from_file_location("as_summary", Path(__file__).with_name("summarize_algorithm_search.py"))
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)


def snap(tag, stage):
    return NS / f"snap_{tag}" / "runs/model_improvement" / stage


SOURCES = {  # model -> list of (run dir, model directory prefix)
    "xgb_full_T": [(RUNS / "algorithm_search_v1/xgb_final", "xgb_residual_d10_m1")],
    "xgb_lowdim": [(NS / "xgb_lowdim_5seed", "xgb_residual_d10_m1")],
    "mdx": [(RUNS / "algorithm_search_v1/screen_b1", "bisurf_search_mdx_v2"),
            (RUNS / "algorithm_search_v1/stability_mdx", "bisurf_search_mdx_v2")],
    "v2": [(RUNS / "factorial_exchange_v1", "bisurf_mono_v2")],
    "C2_num_tabm_o1": [(snap("c78b22d69f15", "screen_num"), "bisurf_num_tabm_o1"),
                       (snap("ae92da88a034", "stab_num"), "bisurf_num_tabm_o1")],
    "C3_numsurf_tabm_o1_g2": [(snap("33509857888e", "screen_full_g2"), "bisurf_numsurf_tabm_o1"),
                              (snap("ae92da88a034", "stab_full_g2"), "bisurf_numsurf_tabm_o1")],
    "C1_num_ple_o1": [(snap("c78b22d69f15", "screen_num"), "bisurf_num_ple_o1")],
    "C1_num_ple_o2": [(snap("c78b22d69f15", "screen_num"), "bisurf_num_ple_o2")],
    "C2_num_tabm_o2": [(snap("c78b22d69f15", "screen_num"), "bisurf_num_tabm_o2")],
    "C3_numsurf_tabm_o1_g1": [(snap("33509857888e", "screen_full_g1"), "bisurf_numsurf_tabm_o1")],
    "C4_numsurf_tabm_o1_lowdim": [(snap("5b9fe1796a6d", "screen_c4_lowdim"), "bisurf_numsurf_tabm_o1_lowdim")],
}


def collect(va):
    merged, dups = {}, []
    for model, sources in SOURCES.items():
        seeds = {}
        for run, prefix in sources:
            for job in sorted(run.glob(f"{prefix}_seed_*_primary")):
                if job.name.split("_seed_")[0] != prefix or not (job / "predictions.csv").exists():
                    continue
                seed = job.name.split("_seed_")[1].split("_")[0]
                p = pd.read_csv(job / "predictions.csv")
                assert (p.source_row.to_numpy() == va.source_combodrugseq.to_numpy()).all()
                ps = S.per_surface(va, p.prediction.to_numpy())
                if seed in seeds:
                    diff = float(np.max(np.abs(seeds[seed].to_numpy() - ps.to_numpy())))
                    dups.append(dict(model=model, seed=seed, run=str(run.relative_to(ROOT)), max_abs_surface_rmse_diff=diff))
                    if diff > 1e-9:
                        raise RuntimeError(f"Cross-run seed mismatch {model} {seed}: {diff}")
                else:
                    seeds[seed] = ps
        merged[model] = seeds
    return merged, pd.DataFrame(dups)


def main():
    va = S.validation_frame()
    merged, dups = collect(va)
    dups.to_csv(TABLES / "numeric_surface_v1_seed_reproduction.csv", index=False)
    per_seed = pd.DataFrame([dict(model=m, seed=s, surface_macro_rmse=float(v.mean())) for m, d in merged.items() for s, v in d.items()])
    per_seed.sort_values(["model", "seed"]).to_csv(TABLES / "numeric_surface_v1_per_seed.csv", index=False)
    stats = pd.DataFrame([dict(model=m, **S.seed_stats([v.mean() for v in d.values()])) for m, d in merged.items() if d])
    stats.sort_values("mean").to_csv(TABLES / "numeric_surface_v1_models.csv", index=False)
    print(stats.sort_values("mean").to_string())
    comps = []
    for cand in ("C3_numsurf_tabm_o1_g2", "C2_num_tabm_o1"):
        for ref in ("xgb_full_T", "mdx", "v2", "xgb_lowdim", "C2_num_tabm_o1"):
            if cand != ref and len(merged[cand]) >= 2:
                comps.append(S.compare(va, merged, cand, ref))
    comps = pd.DataFrame(comps); comps.to_csv(TABLES / "numeric_surface_v1_comparisons.csv", index=False)
    print(comps[["candidate", "reference", "paired_seeds", "candidate_mean", "reference_mean", "relative_improvement",
                 "improved_seeds", "seed_ci95_low", "seed_ci95_high", "group_bootstrap_rel_ci95_low",
                 "group_bootstrap_rel_ci95_high"]].round(6).to_string())
    # descriptive error-complementarity (not a candidate, no weight fitted)
    common = sorted(set(merged["C3_numsurf_tabm_o1_g2"]) & set(merged["xgb_full_T"]), key=int)
    if common:
        a = pd.concat([merged["C3_numsurf_tabm_o1_g2"][s] for s in common], axis=1).mean(1)
        b = pd.concat([merged["xgb_full_T"][s] for s in common], axis=1).mean(1)
        print("per-surface RMSE correlation C3 vs T:", round(float(np.corrcoef(a, b)[0, 1]), 4))


if __name__ == "__main__":
    main()
