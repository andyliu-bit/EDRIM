"""Summaries for AlgorithmSearch-v1 (validation-only, reads saved per-point predictions).

Outputs (reports/tables):
  competitive_position_v1.csv   comparable models on the frozen fold-0 protocol
  algorithm_search_v1_registry.csv  every attempted run incl. failures
  algorithm_search_v1_candidates.csv  per-candidate decision vs v2 and vs T
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "runs/model_improvement"
SEARCH = RUNS / "algorithm_search_v1"
TABLES = ROOT / "reports/tables"
SHARD = ROOT.parent / "bisurfgnn/data/processed/one_new_scaffold_benchmark_unclipped/learned_point_shards/fold_0/validation.parquet"


def validation_frame():
    ids = pd.read_csv(RUNS / "comparator_gpu_v2/validation_ids.csv")
    cols = ["source_combodrugseq", "surface_replicate_id", "prediction_group_id", "heldout_cluster_id",
            "response", "mono_response_a", "mono_response_b"]
    va = pd.read_parquet(SHARD, columns=cols).set_index("source_combodrugseq").loc[ids.source_combodrugseq].reset_index()
    assert (va.surface_replicate_id.to_numpy() == ids.surface_replicate_id.to_numpy()).all()
    return va


def per_surface(va, prediction):
    err = pd.DataFrame({"s": va.surface_replicate_id, "e": (np.asarray(prediction) - va.response.to_numpy()) ** 2})
    return np.sqrt(err.groupby("s", sort=True).e.mean())


def load_runs(va):
    """All primary predictions in the given run directories → {model: {seed: per-surface RMSE}}."""
    surfaces, macro, status = {}, [], []
    dirs = {"historical_factorial_exchange_v1": RUNS / "factorial_exchange_v1",
            "historical_comparator_gpu_v2": RUNS / "comparator_gpu_v2"}
    for d in sorted(SEARCH.iterdir()):
        if d.is_dir() and d.name not in {"launchers", "regression"}:
            dirs[d.name] = d
    for run, path in dirs.items():
        summary = path / "summary.json"
        state = json.loads(summary.read_text())["status"] if summary.exists() else (
            "failed" if (path / "failure.json").exists() else "incomplete_or_placeholder")
        status.append(dict(run=run, status=state, path=str(path.relative_to(ROOT))))
        for job in sorted(path.glob("*_primary")):
            if not (job / "predictions.csv").exists():
                continue
            name, seed = job.name.split("_seed_")[0], job.name.split("_seed_")[1].split("_")[0]
            if run.startswith("historical") and name not in {"bisurf_mono_v2", "ddprism_hallmark_observed_mono",
                                                              "combokr2_dense_minres", "bisurf_legacy"}:
                continue
            if run == "historical_comparator_gpu_v2" and name == "bisurf_mono_v2":
                name = "bisurf_mono_v2_batch64"
            p = pd.read_csv(job / "predictions.csv")
            assert (p.source_row.to_numpy() == va.source_combodrugseq.to_numpy()).all()
            ps = per_surface(va, p.prediction.to_numpy())
            key = f"{run}:{name}"
            surfaces.setdefault(key, {})[seed] = ps
            macro.append(dict(run=run, model=name, seed=seed, surface_macro_rmse=float(ps.mean())))
    return surfaces, pd.DataFrame(macro), pd.DataFrame(status)


def merge_seeds(surfaces):
    """Merge each model's seeds across non-preflight runs; duplicated seeds must agree exactly."""
    merged, provenance = {}, []
    for key, seeds in surfaces.items():
        run, model = key.split(":", 1)
        if run in {"preflight", "historical_comparator_gpu_v2"} or run.startswith("xgbdiag"):
            continue
        if model.startswith("xgb") and run != "xgb_final":
            continue  # tuning-grid seeds are not the frozen T configuration
        if model.startswith("ddprism") and run != "dd_final":
            continue
        if run == "screen_b2_e80":
            model = f"{model}__epochs80"  # K6: same module as K5, different training budget
        target = merged.setdefault(model, {})
        for seed, values in seeds.items():
            if seed in target:
                diff = float(np.max(np.abs(target[seed].to_numpy() - values.to_numpy())))
                provenance.append(dict(model=model, seed=seed, duplicate_run=run, max_abs_surface_rmse_diff=diff))
                if diff > 1e-9:
                    raise RuntimeError(f"Non-reproducible duplicate seed {model} {seed} {run}: {diff}")
            else:
                target[seed] = values
    return merged, pd.DataFrame(provenance)


def seed_stats(values):
    v = np.asarray(values, float); n = len(v)
    sd = float(v.std(ddof=1)) if n > 1 else float("nan")
    r = float(student_t.ppf(.975, n - 1) * sd / np.sqrt(n)) if n > 1 else float("nan")
    return dict(seeds=n, mean=float(v.mean()), sd=sd, seed_ci95_low=float(v.mean()) - r, seed_ci95_high=float(v.mean()) + r)


def group_bootstrap(va, cand, ref, iterations=2000, seed=20261009):
    """Relative improvement of seed-averaged per-surface RMSE; resample prediction groups within clusters."""
    meta = va.drop_duplicates("surface_replicate_id").set_index("surface_replicate_id").loc[cand.index]
    groups = meta.prediction_group_id.to_numpy(); clusters = meta.heldout_cluster_id.to_numpy()
    frame = pd.DataFrame({"g": groups, "k": clusters, "c": cand.to_numpy(), "r": ref.to_numpy()})
    agg = frame.groupby(["k", "g"]).agg(c=("c", "sum"), r=("r", "sum"), n=("c", "size")).reset_index()
    rng = np.random.default_rng(seed); draws = []
    by_cluster = [part for _, part in agg.groupby("k")]
    for _ in range(iterations):
        c = r = n = 0.0
        for part in by_cluster:
            pick = part.iloc[rng.integers(0, len(part), len(part))]
            c += pick.c.sum(); r += pick.r.sum(); n += pick.n.sum()
        draws.append((r / n - c / n) / (r / n))
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(low), float(high)


def compare(va, surfaces, cand_key, ref_key):
    c, r = surfaces[cand_key], surfaces[ref_key]
    common = sorted(set(c) & set(r), key=int)
    cm = np.array([c[s].mean() for s in common]); rm = np.array([r[s].mean() for s in common])
    diff = rm - cm; n = len(common)
    radius = float(student_t.ppf(.975, n - 1) * diff.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    cs = pd.concat([c[s] for s in common], axis=1).mean(axis=1)
    rs = pd.concat([r[s] for s in common], axis=1).mean(axis=1)
    low, high = group_bootstrap(va, cs, rs) if n > 1 else (float("nan"), float("nan"))
    return dict(candidate=cand_key, reference=ref_key, paired_seeds=n,
                candidate_mean=float(cm.mean()), reference_mean=float(rm.mean()),
                relative_improvement=float((rm.mean() - cm.mean()) / rm.mean()),
                improved_seeds=int((cm < rm).sum()), paired_rmse_reduction=float(diff.mean()),
                seed_ci95_low=float(diff.mean() - radius), seed_ci95_high=float(diff.mean() + radius),
                group_bootstrap_rel_ci95_low=low, group_bootstrap_rel_ci95_high=high)


def main():
    va = validation_frame()
    surfaces, macro, status = load_runs(va)
    hsa = per_surface(va, np.minimum(va.mono_response_a, va.mono_response_b)).mean()
    macro.to_csv(TABLES / "algorithm_search_v1_registry_runs.csv", index=False)
    status.to_csv(TABLES / "algorithm_search_v1_registry_status.csv", index=False)
    rows = [dict(key="HSA (zero-parameter)", seeds=0, mean=float(hsa))]
    for key, seeds in surfaces.items():
        rows.append(dict(key=key, **seed_stats([v.mean() for v in seeds.values()])))
    pd.DataFrame(rows).sort_values("mean").to_csv(TABLES / "algorithm_search_v1_all_models.csv", index=False)
    merged, duplicates = merge_seeds(surfaces)
    duplicates.to_csv(TABLES / "algorithm_search_v1_seed_reproduction.csv", index=False)
    per_seed = [dict(model=m, seed=s, surface_macro_rmse=float(v.mean())) for m, seeds in merged.items() for s, v in seeds.items()]
    pd.DataFrame(per_seed).sort_values(["model", "seed"]).to_csv(TABLES / "algorithm_search_v1_final_per_seed.csv", index=False)
    final = [dict(model=m, **seed_stats([v.mean() for v in seeds.values()])) for m, seeds in merged.items()]
    pd.DataFrame(final).sort_values("mean").to_csv(TABLES / "algorithm_search_v1_final_models.csv", index=False)
    print(pd.DataFrame(final).sort_values("mean").to_string())
    if len(sys.argv) > 2:  # candidate comparisons on merged seeds: <T model> <v2 model> <candidate models...>
        t_key, v2_key, *cands = sys.argv[1:]
        out = [compare(va, merged, c, ref) for c in cands for ref in (t_key, v2_key)]
        pd.DataFrame(out).to_csv(TABLES / "algorithm_search_v1_comparisons.csv", index=False)
        print(pd.DataFrame(out).T.to_string())


if __name__ == "__main__":
    main()
