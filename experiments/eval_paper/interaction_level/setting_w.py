"""Eval-paper interaction-level reanalysis, setting W (warm new combination, development).

Frozen card: bisurfgnn/docs/EVAL_INTERACTION_LEVEL_20261029.md. DeepSynBa public split 2 validation set, MDA-MB-468
excluded (gate set, 54,615 matrices / 987 pairs). Percent growth, c = 100; reference from each matrix's own measured
mono edges. Rows: lookup HSA / Bliss (degenerate, RMSE only), mono-implied ranking (yhat = min(a, b)), P4 (lambda 100,
refit on split 1 only, deterministic), gate XGBoost seeds 41-45 (single-model average + range), DeepSynBa seed 10
(epoch 484 saved validation predictions). No BiSurfGNN row; BiSurfGNN prediction files are not opened.
Nothing is trained or tuned apart from the pre-fixed deterministic P4 refit; no split-3 / test file is opened
(synergy_check.allowed guards every read).

  .venv/Scripts/python.exe -I experiments/eval_paper/interaction_level/setting_w.py
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

T0 = time.perf_counter()
HERE = Path(__file__).resolve().parent
EXP = HERE.parents[2]
sys.path[:0] = [str(HERE), str(EXP / "experiments/model_improvement/synergy_check"),
                str(EXP / "experiments/model_improvement/v2_gap_diagnosis")]
import interaction_metrics as im  # noqa: E402
import synergy_check as sc  # noqa: E402  (loader, alignment checks, test-file guard)
import d3_probes as d3  # noqa: E402  (P4 design and ridge fit)

OUT = EXP / "runs/eval_paper/interaction_level/W"
SC_POINTS = EXP / "runs/model_improvement/bisurfgnn_v2_dev/synergy_check/results/point_estimates.csv"
PROBES_CSV = EXP / "runs/model_improvement/v2_gap_diagnosis/probes/entity_probes.csv"
C = 100.0
P4_BLOCKS, P4_LAMBDA = ["position", "drug_dose", "drug_cell", "cell_position"], 100.0
SCORES = ("HSA", "Bliss", "Bliss_clip")          # Bliss_clip = sensitivity: monos clipped to [0, 100]
METRICS = ("pearson", "spearman", "precision_top5", "rmse")
XGB_SEEDS = list(sc.XGB)
ROWS = ["LK_HSA", "LK_Bliss", "MONO", "P4", "XGB", "DSB"]
COMPARISONS = [("XGB", "P4"), ("DSB", "P4")]


def degenerate(s: str, k: str) -> bool:
    return s.startswith("LK_") or (s == "MONO" and k == "HSA")


def fit_p4(m_count: int, y_grid: np.ndarray) -> tuple[np.ndarray, dict]:
    manifest = json.loads(sc.allowed(sc.S1 / "manifest.json").read_text(encoding="utf-8"))
    if sc.sha256(sc.S1 / "points_train.parquet") != manifest["points_train.parquet"]:
        raise RuntimeError("S1 training table changed")
    cols = ["drug_a_id", "drug_b_id", "deepsynba_cell", "row_i", "col_j", "response_percent", "hsa_lookup_percent", "pair_id"]
    tr = pd.read_parquet(sc.allowed(sc.S1 / "points_train.parquet"), columns=cols)
    v = pd.read_parquet(sc.allowed(sc.S1 / "points_validation.parquet"), columns=cols + ["surface_index"])
    drugs = sorted(set(tr.drug_a_id) | set(tr.drug_b_id))
    cells = sorted(set(tr.deepsynba_cell))
    if not (set(v.drug_a_id) | set(v.drug_b_id)) <= set(drugs) or not set(v.deepsynba_cell) <= set(cells):
        raise RuntimeError("validation entity missing from training")
    if v.pair_id.isin(set(tr.pair_id)).any():
        raise RuntimeError("validation pair present in training")
    dtr = (tr.response_percent - tr.hsa_lookup_percent).to_numpy(float)
    xtr = d3.design(tr, cells, drugs, P4_BLOCKS)
    coef, mean, iters = d3.fit_probe(xtr, dtr, P4_LAMBDA)
    n_train, n_coef = len(tr), xtr.shape[1]
    del xtr, tr, dtr
    pva = v.hsa_lookup_percent.to_numpy(float) + mean + d3.design(v, cells, drugs, P4_BLOCKS) @ coef
    yv = v.response_percent.to_numpy(float)
    r = im.rmse(pva, yv)
    probes = pd.read_csv(sc.allowed(PROBES_CSV))
    logged = float(probes[(probes.probe == "P4") & (probes.lam == P4_LAMBDA)].val_rmse.iloc[0])
    if abs(r - logged) > 1e-6:
        raise RuntimeError(f"P4 split-2 combination RMSE {r} differs from the logged {logged}")
    grid = np.full((m_count, 3, 3), np.nan)
    grid[v.surface_index.to_numpy(), v.row_i.to_numpy() - 1, v.col_j.to_numpy() - 1] = pva
    ygrid = np.full((m_count, 3, 3), np.nan)
    ygrid[v.surface_index.to_numpy(), v.row_i.to_numpy() - 1, v.col_j.to_numpy() - 1] = yv
    if np.isnan(grid).any() or np.max(np.abs(ygrid - y_grid)) > 0:
        raise RuntimeError("P4 grid does not align with the gate validation matrices")
    return grid, dict(train_points=n_train, coefficients=n_coef, lsqr_iterations=iters, train_mean_delta=mean,
                      split2_combination_rmse=r, logged_rmse=logged, abs_diff=abs(r - logged))


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a non-empty output directory: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    # reuse the synergy-check loader without opening BiSurfGNN files (its BS_ens5 becomes an XGB copy and is dropped)
    sc.BISURF, sc.BS_SEEDS = {}, sc.XGB_SEEDS
    meta, y, obs_a, obs_b, lk_a, lk_b, dsb, preds, checks = sc.load_matrices()
    m_count = len(meta)
    pair_ids = meta.pair_id.to_numpy()
    if m_count != 54_615 or len(np.unique(pair_ids)) != 987:
        raise RuntimeError(f"unexpected gate set: {m_count} matrices, {len(np.unique(pair_ids))} pairs")
    p4, p4_info = fit_p4(m_count, y)
    print(f"P4 split-2 combination RMSE {p4_info['split2_combination_rmse']:.6f} (logged {p4_info['logged_rmse']:.6f})", flush=True)

    shape = (m_count, 3, 3)
    codes = np.repeat(np.arange(m_count), 9)
    a = np.broadcast_to(obs_a[:, :, None], shape).ravel()
    b = np.broadcast_to(obs_b[:, None, :], shape).ravel()
    ref = {"HSA": im.reference(a, b, "HSA", C), "Bliss": im.reference(a, b, "Bliss", C),
           "Bliss_clip": im.reference(np.clip(a, 0, 100), np.clip(b, 0, 100), "Bliss", C)}
    yhat = {"LK_HSA": preds["LK_HSA"].ravel(), "LK_Bliss": preds["LK_Bliss"].ravel(), "MONO": np.minimum(a, b),
            "P4": p4.ravel(), **{s: preds[s].ravel() for s in XGB_SEEDS}, "DSB": preds["DSB"].ravel()}
    observed = {k: im.matrix_score(ref[k], y.ravel(), codes) for k in SCORES}
    derived = {s: {k: im.matrix_score(ref[k], p, codes) for k in SCORES} for s, p in yhat.items()}
    for s, p in yhat.items():   # card corollary: same score error under every reference = matrix-mean cell error
        e = im.matrix_mean(y.ravel() - p, codes)
        for k in SCORES:
            if np.max(np.abs(derived[s][k] - observed[k] - e)) > 1e-9:
                raise RuntimeError(f"{s}/{k}: score error is not the matrix-mean combination error")
    if np.max(np.abs(derived["MONO"]["HSA"])) != 0:
        raise RuntimeError("mono-implied HSA score should be identically 0")

    # self-check: SYNERGY_CHECK table 3 (DeepSynBa, XGBoost per seed, lookup RMSE)
    sc_pts = pd.read_csv(sc.allowed(SC_POINTS))
    sc_pts = sc_pts[sc_pts.subset == "all"].set_index(["series", "score"])
    self_check = {}
    for s in XGB_SEEDS + ["DSB", "LK_HSA", "LK_Bliss"]:
        for k in ("HSA", "Bliss"):
            ref_row = sc_pts.loc[(s, k)]
            mine = im.all_metrics(derived[s][k], observed[k], degenerate=bool(np.isnan(ref_row.pearson)))
            for m, col in (("rmse", "rmse"), ("pearson", "pearson"), ("spearman", "spearman"),
                           ("precision_top5", "precision_top5"), ("slope", "slope_pred_on_obs")):
                if np.isnan(ref_row[col]):
                    continue
                d = abs(mine[m] - ref_row[col])
                self_check[f"{s}|{k}|{m}"] = dict(mine=mine[m], synergy_check=float(ref_row[col]), abs_diff=d)
                if d > 1e-9:
                    raise RuntimeError(f"self-check failed {s}/{k}/{m}: {mine[m]} vs {ref_row[col]}")
    print("self-check vs SYNERGY_CHECK table 3 passed", flush=True)

    # point estimates
    series = list(yhat)
    point = {(s, k): im.all_metrics(derived[s][k], observed[k], degenerate(s, k)) for s in series for k in SCORES}
    for k in SCORES:
        for m in ("rmse", "pearson", "spearman", "precision_top5", "slope"):
            vals = [point[(s, k)][m] for s in XGB_SEEDS]
            point.setdefault(("XGB", k), {})[m] = float(np.mean(vals))
            point[("XGB", k)][m + "_seed_min"], point[("XGB", k)][m + "_seed_max"] = float(np.min(vals)), float(np.max(vals))

    # drug-pair cluster bootstrap, same draws for every statistic
    def stat(idx):
        out = {}
        for k in SCORES:
            o = observed[k][idx]
            for s in series:
                p = derived[s][k][idx]
                out[f"{s}|{k}|rmse"] = im.rmse(p, o)
                if not degenerate(s, k):
                    out[f"{s}|{k}|pearson"] = im.pearson(p, o)
                    out[f"{s}|{k}|spearman"] = im.spearman(p, o)
                    out[f"{s}|{k}|precision_top5"] = im.precision_top(p, o)
        return out

    t_boot = time.perf_counter()
    boot = im.bootstrap(stat, pair_ids)
    boot_seconds = time.perf_counter() - t_boot
    for k in SCORES:
        for m in METRICS:
            if f"XGB_s41|{k}|{m}" in boot:
                boot[f"XGB|{k}|{m}"] = np.mean([boot[f"{s}|{k}|{m}"] for s in XGB_SEEDS], axis=0)
    print(f"bootstrap {boot_seconds:.0f} s", flush=True)

    rows = []
    for s in ROWS:
        for k in SCORES:
            r = dict(row=s, score=k, matrices=m_count, pairs=987, degenerate=degenerate(s, k), **point[(s, k)])
            for m in METRICS:
                key = f"{s}|{k}|{m}"
                r[m + "_ci_low"], r[m + "_ci_high"] = im.percentile_ci(boot[key]) if key in boot else (np.nan, np.nan)
            rows.append(r)
    points = pd.DataFrame(rows)
    points.to_csv(OUT / "point_estimates.csv", index=False)

    comp_rows = []
    for k in SCORES:
        for m in METRICS:
            for a_name, b_name in COMPARISONS:
                d = boot[f"{a_name}|{k}|{m}"] - boot[f"{b_name}|{k}|{m}"]
                lo, hi = im.percentile_ci(d)
                comp_rows.append(dict(comparison=f"{a_name} - {b_name}", score=k, metric=m,
                                      a_value=point[(a_name, k)][m], b_value=point[(b_name, k)][m],
                                      difference=point[(a_name, k)][m] - point[(b_name, k)][m], ci_low=lo, ci_high=hi,
                                      boot_p=im.boot_p(d), better_is="lower" if m == "rmse" else "higher",
                                      family={"Bliss": "primary", "Bliss_clip": "sensitivity"}.get(k, "secondary")
                                      if m == "pearson" else "secondary"))
    comps = pd.DataFrame(comp_rows)
    comps["holm_p"] = np.nan
    for fam in ("primary", "sensitivity"):
        sel = comps.family == fam
        comps.loc[sel, "holm_p"] = im.holm(comps.loc[sel, "boot_p"].to_numpy())
    comps.to_csv(OUT / "paired_comparisons.csv", index=False)

    def verdict(r):
        if r.holm_p >= 0.05:
            return "交互层上未见学习增益"
        return "学习在交互层有增益" if r.difference > 0 else "学习模型低于参照（显著为负）"

    wording = {f"{r.comparison} ({r.family})": dict(difference=r.difference, ci=[r.ci_low, r.ci_high], holm_p=r.holm_p,
                                                    wording=verdict(r))
               for r in comps[comps.family.isin(["primary", "sensitivity"])].itertuples()}
    for k in ("Bliss", "Bliss_clip"):
        lo = points[(points.row == "P4") & (points.score == k)].pearson_ci_low.iloc[0]
        wording[f"P4 Bliss Pearson CI lower ({k})"] = dict(
            ci_low=float(lo), wording="不训练的参照已带有可检出的交互信号" if lo > 0 else "区间下限不大于 0，不作该表述")
    clip_edges = np.concatenate([obs_a.ravel(), obs_b.ravel()])
    summary = dict(
        setting="W", card="bisurfgnn/docs/EVAL_INTERACTION_LEVEL_20261029.md", split_read=[1, 2], test_rows_read=0,
        matrices=m_count, pairs=987, c=C, k_top5=im.top_k(m_count), checks=checks, p4=p4_info,
        self_check_vs_synergy_check=dict(max_abs_diff=max(v["abs_diff"] for v in self_check.values()), items=self_check),
        observed_scores={k: dict(mean=float(observed[k].mean()), sd=float(observed[k].std())) for k in SCORES},
        mono_edges=dict(total=int(clip_edges.size), below_0=int((clip_edges < 0).sum()), above_100=int((clip_edges > 100).sum())),
        wording=wording, bootstrap=dict(seed=im.BOOT_SEED, draws=im.BOOT_N, unit="validation drug pair", seconds=round(boot_seconds, 1)),
        files_read=sorted(set(sc.READ_LOG)),
        versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__),
        runtime_seconds=round(time.perf_counter() - T0, 1), finished_local_time=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    with (OUT / "summary.json").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, indent=1, ensure_ascii=False, default=float) + "\n")
    pd.set_option("display.width", 250)
    print(points[["row", "score", "pearson", "pearson_ci_low", "pearson_ci_high", "spearman", "precision_top5", "rmse",
                  "slope"]].round(4).to_string(index=False))
    print(comps[["comparison", "score", "metric", "difference", "ci_low", "ci_high", "boot_p", "holm_p", "family"]]
          .round(4).to_string(index=False))
    print(json.dumps(wording, ensure_ascii=False, indent=1, default=float))
    print(f"done in {summary['runtime_seconds']} s")


if __name__ == "__main__":
    main()
