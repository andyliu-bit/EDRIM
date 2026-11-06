"""Eval-paper interaction-level reanalysis, setting D (new drug, development).

Frozen card: bisurfgnn/docs/EVAL_INTERACTION_LEVEL_20261029.md. The five new-drug partitions of
bisurfgnn/docs/EVAL_COLD_DRUG_PARTITIONS_20261018.md (D11 = 20261016, 20261018..20261021), new-drug test only.
Percent growth, c = 100; reference from the S1 lookup monos (mono_a/b_percent, what lookup HSA, P4 and XGBoost use).
Rows: lookup HSA (degenerate, RMSE only), mono-implied ranking (yhat = lookup HSA = min(a, b), Bliss score only),
P4 (saved p4_predictions.npz), XGBoost seeds 41-43 (single-model average + range; D11 also seeds 41-45).
No BiSurfGNN / DeepSynBa rows; their files are not opened. Nothing is trained or tuned; split 3 is never read
(common.read_points reads S1 points_train + points_validation only).

  .venv/Scripts/python.exe -I experiments/eval_paper/interaction_level/setting_d.py
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from scipy.stats import rankdata

T0 = time.perf_counter()
HERE = Path(__file__).resolve().parent
EXP = HERE.parents[2]
sys.path[:0] = [str(HERE), str(EXP / "experiments/eval_paper/cold_drug_partitions")]
import interaction_metrics as im  # noqa: E402
import common as cd  # noqa: E402  (S1 reader, partition frames, lookup HSA, pralatrexate id)
import score as cs  # noqa: E402  (xgb_dir, aligned)

OUT = EXP / "runs/eval_paper/interaction_level/D"
PARTS = cd.PARTS
N_BOOT = im.BOOT_N
C = 100.0
SCORES = ("HSA", "Bliss", "Bliss_clip")          # Bliss_clip = sensitivity: monos clipped to [0, 100]
SETS = ("all", "excluding_pralatrexate")
METRICS = ("pearson", "spearman", "precision_top5", "rmse")
COLS = ["matrix_key", "pair_id", "drug_a_id", "drug_b_id", "mono_a_percent", "mono_b_percent", "response_percent"]
FILES_READ: list[str] = []


def degenerate(s: str, k: str) -> bool:
    return s == "LK_HSA" or (s == "MONO" and k == "HSA")


def load(path: Path):
    FILES_READ.append(str(path.relative_to(EXP)))
    return np.load(path)


def run_partition(part: int, both: pd.DataFrame, evaluation: dict, final_table: pd.DataFrame):
    te = cd.frames(part, both)["cold_test"]
    FILES_READ.append(str(cd.split_path(part).relative_to(EXP)))
    src = te.source_combodrugseq.to_numpy()
    y = te.response_percent.to_numpy(float)
    a, b = te.mono_a_percent.to_numpy(float), te.mono_b_percent.to_numpy(float)
    codes, keys = pd.factorize(te.matrix_key)
    m_count = len(keys)
    if te.pair_id.groupby(codes).nunique().max() != 1:
        raise RuntimeError("a matrix spans more than one drug pair")
    pair_m = te.pair_id.groupby(codes).first().to_numpy()
    prala_m = np.array([cd.PRALA in p.split("|") for p in pair_m])
    keep = {"all": np.ones(m_count, bool), "excluding_pralatrexate": ~prala_m}
    cells_per_matrix = np.bincount(codes)

    xgb_seeds = (41, 42, 43, 44, 45) if part == cd.D11_SEED else cd.XGB_SEEDS
    yhat = {"MONO": cd.lookup_hsa(te),
            "P4": cs.aligned(load(cd.OUT / f"p{part}/p4_predictions.npz"), "cold_test_source", src, "cold_test_p4_percent")}
    for s in xgb_seeds:
        yhat[f"XGB_s{s}"] = cs.aligned(load(cs.xgb_dir(part, s) / "cold_test_predictions.npz"), "source", src, "prediction_percent")
    series = list(yhat)
    groups = {"XGB": [f"XGB_s{s}" for s in cd.XGB_SEEDS]}
    if part == cd.D11_SEED:
        groups["XGB5"] = [f"XGB_s{s}" for s in xgb_seeds]

    # self-check: pooled combination-cell RMSE = evaluation.json per seed and final_table.csv
    ev = evaluation["partitions"][str(part)]["test"]
    rmse_check = {}
    for label in SETS:
        kp = keep[label][codes]
        mine = {s: im.rmse(p[kp], y[kp]) for s, p in yhat.items()}
        want = {"MONO": ev[label]["per_seed"]["lookup_hsa"]["0"], "P4": ev[label]["per_seed"]["p4"]["0"],
                **{f"XGB_s{s}": ev[label]["per_seed"]["xgb_5seed" if part == cd.D11_SEED else "xgb"][str(s)] for s in xgb_seeds}}
        ft = final_table[(final_table.set == label) & (final_table.partition == part)].set_index("method")
        want_ft = {"MONO": ft.loc["xgb", "lookup_hsa"], "P4": ft.loc["p4", "rmse"], "XGB": ft.loc["xgb", "rmse"]}
        mine["XGB"] = float(np.mean([mine[s] for s in groups["XGB"]]))
        for s, w in list(want.items()) + [(f"{k}|final_table", v) for k, v in want_ft.items()]:
            d = abs(mine[s.split("|")[0]] - w)
            rmse_check[f"{label}|{s}"] = dict(mine=mine[s.split("|")[0]], saved=float(w), abs_diff=d)
            if d > 1e-9:
                raise RuntimeError(f"{part} {label} {s}: combination RMSE {mine[s.split('|')[0]]} != saved {w}")
        if ev[label]["points"] != int(kp.sum()) or ev[label]["pairs"] != len(np.unique(pair_m[keep[label]])):
            raise RuntimeError(f"{part} {label}: point / pair count differs from evaluation.json")

    ref = {"HSA": im.reference(a, b, "HSA", C), "Bliss": im.reference(a, b, "Bliss", C),
           "Bliss_clip": im.reference(np.clip(a, 0, 100), np.clip(b, 0, 100), "Bliss", C)}
    observed = {k: im.matrix_score(ref[k], y, codes) for k in SCORES}
    derived = {s: {k: im.matrix_score(ref[k], p, codes) for k in SCORES} for s, p in yhat.items()}
    corollary = 0.0
    for s, p in yhat.items():   # card corollary: same score error under every reference = matrix-mean cell error
        e = im.matrix_mean(y - p, codes)
        for k in SCORES:
            corollary = max(corollary, float(np.max(np.abs(derived[s][k] - observed[k] - e))))
    if corollary > 1e-9:
        raise RuntimeError(f"score error is not the matrix-mean combination error ({corollary})")
    if np.max(np.abs(derived["MONO"]["HSA"])) != 0:
        raise RuntimeError("mono-implied HSA score should be identically 0")

    def stat(idx):
        """All metrics for one draw; observed ranks and top-k are computed once per score and reused for every series."""
        out = {}
        for label in SETS:
            sel = idx[keep[label][idx]]
            k_top = im.top_k(len(sel))
            for k in SCORES:
                o = observed[k][sel]
                ro, to = rankdata(o), np.argsort(-o, kind="stable")[:k_top]
                for s in series:
                    p = derived[s][k][sel]
                    key = f"{label}|{s}|{k}"
                    out[key + "|rmse"] = im.rmse(p, o)
                    if not degenerate(s, k):
                        out[key + "|pearson"] = im.pearson(p, o)
                        out[key + "|spearman"] = im.pearson(rankdata(p), ro)   # = im.spearman (asserted below)
                        out[key + "|precision_top5"] = len(np.intersect1d(np.argsort(-p, kind="stable")[:k_top], to)) / k_top
        return out

    # point estimates with the module functions; the cached draw path must reproduce them exactly
    point = {(label, s, k): im.all_metrics(derived[s][k][keep[label]], observed[k][keep[label]], degenerate(s, k))
             for label in SETS for s in series for k in SCORES}
    ident = stat(np.arange(m_count))
    cache_diff = max(abs(ident[f"{l}|{s}|{k}|{m}"] - point[(l, s, k)][m]) for l in SETS for s in series for k in SCORES
                     for m in METRICS if f"{l}|{s}|{k}|{m}" in ident)
    if cache_diff > 1e-12:
        raise RuntimeError(f"cached bootstrap metrics differ from interaction_metrics ({cache_diff})")
    for label in SETS:
        for k in SCORES:
            point[(label, "LK_HSA", k)] = im.all_metrics(derived["MONO"][k][keep[label]], observed[k][keep[label]], True)
            for g, members in groups.items():
                row = {}
                for m in ("rmse", "pearson", "spearman", "precision_top5", "slope"):
                    vals = [point[(label, s, k)][m] for s in members]
                    row[m], row[m + "_seed_min"], row[m + "_seed_max"] = float(np.mean(vals)), float(np.min(vals)), float(np.max(vals))
                point[(label, g, k)] = row

    t_boot = time.perf_counter()
    boot = im.bootstrap(stat, pair_m, N_BOOT)
    boot_seconds = time.perf_counter() - t_boot
    for key in [k for k in boot if k.split("|")[1] == "MONO"]:
        boot[key.replace("|MONO|", "|LK_HSA|")] = boot[key]          # same prediction; only RMSE is reported
    for g, members in groups.items():
        for label in SETS:
            for k in SCORES:
                for m in METRICS:
                    boot[f"{label}|{g}|{k}|{m}"] = np.mean([boot[f"{label}|{s}|{k}|{m}"] for s in members], axis=0)

    rows_out = ["LK_HSA", "MONO", "P4", *groups]
    rows = []
    for label in SETS:
        for s in rows_out:
            for k in SCORES:
                r = dict(partition=part, set=label, row=s, score=k, matrices=int(keep[label].sum()),
                         pairs=len(np.unique(pair_m[keep[label]])), degenerate=degenerate(s, k), **point[(label, s, k)])
                for m in METRICS:
                    key = f"{label}|{s}|{k}|{m}"
                    r[m + "_ci_low"], r[m + "_ci_high"] = im.percentile_ci(boot[key]) if key in boot and not (
                        m != "rmse" and degenerate(s, k)) else (np.nan, np.nan)
                rows.append(r)
    comps = []
    for label in SETS:
        for k in SCORES:
            for m in METRICS:
                for g in groups:
                    d = boot[f"{label}|{g}|{k}|{m}"] - boot[f"{label}|P4|{k}|{m}"]
                    lo, hi = im.percentile_ci(d)
                    fam = "secondary"
                    if m == "pearson" and g == "XGB":
                        fam = {("all", "Bliss"): "primary", ("excluding_pralatrexate", "Bliss"): "sensitivity_pralatrexate",
                               ("all", "Bliss_clip"): "sensitivity_clip"}.get((label, k), "secondary")
                    comps.append(dict(partition=part, set=label, comparison=f"{g} - P4", score=k, metric=m,
                                      a_value=point[(label, g, k)][m], b_value=point[(label, "P4", k)][m],
                                      difference=point[(label, g, k)][m] - point[(label, "P4", k)][m], ci_low=lo, ci_high=hi,
                                      boot_p=im.boot_p(d), better_is="lower" if m == "rmse" else "higher", family=fam))
    a_all = np.r_[a, b]
    diag = dict(matrices=m_count, pairs=int(len(np.unique(pair_m))),
                points=int(len(y)), cells_per_matrix=[int(cells_per_matrix.min()), int(cells_per_matrix.max())],
                pralatrexate_matrices=int(prala_m.sum()), pralatrexate_pairs=int(len(np.unique(pair_m[prala_m]))),
                k_top5={l: im.top_k(int(keep[l].sum())) for l in SETS},
                observed_scores={l: {k: dict(mean=float(observed[k][keep[l]].mean()), sd=float(observed[k][keep[l]].std()))
                                     for k in SCORES} for l in SETS},
                mono_values=dict(total=int(a_all.size), below_0=int((a_all < 0).sum()), above_100=int((a_all > 100).sum())),
                xgb_seeds=list(xgb_seeds), rmse_self_check_max_abs_diff=max(v["abs_diff"] for v in rmse_check.values()),
                rmse_self_check=rmse_check, corollary_max_abs_diff=corollary, cached_metric_max_abs_diff=cache_diff,
                bootstrap_seconds=round(boot_seconds, 1))
    print(f"{part}: {m_count} matrices, {diag['pairs']} pairs, bootstrap {boot_seconds:.0f} s, "
          f"RMSE self-check max diff {diag['rmse_self_check_max_abs_diff']:.2e}", flush=True)
    return rows, comps, diag


def verdict(r) -> str:
    if r.holm_p >= 0.05:
        return "交互层上未见学习增益"
    return "学习在交互层有增益" if r.difference > 0 else "学习模型低于参照（显著为负）"


def main() -> None:
    if not sys.flags.isolated:
        raise SystemExit("run with python -I")
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a non-empty output directory: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    evaluation = json.loads((cd.OUT / "evaluation.json").read_text(encoding="utf-8"))
    final_table = pd.read_csv(cd.OUT / "final_table.csv")
    FILES_READ.extend(str(p.relative_to(EXP)) for p in (cd.OUT / "evaluation.json", cd.OUT / "final_table.csv",
                                                        cd.S1 / "points_train.parquet", cd.S1 / "points_validation.parquet"))
    both = cd.read_points(COLS)
    rows, comps, diags = [], [], {}
    for part in PARTS:
        r, c_, d = run_partition(part, both, evaluation, final_table)
        rows += r; comps += c_; diags[part] = d
    del both
    points = pd.DataFrame(rows)
    comps = pd.DataFrame(comps)
    comps["holm_p"] = np.nan
    for fam in ("primary", "sensitivity_pralatrexate", "sensitivity_clip"):
        sel = comps.family == fam
        if sel.sum() != len(PARTS):
            raise RuntimeError(f"family {fam} has {sel.sum()} tests, expected {len(PARTS)}")
        comps.loc[sel, "holm_p"] = im.holm(comps.loc[sel, "boot_p"].to_numpy())
    comps["wording"] = [verdict(r) if r.family != "secondary" else "" for r in comps.itertuples()]
    points.to_csv(OUT / "point_estimates.csv", index=False)
    comps.to_csv(OUT / "paired_comparisons.csv", index=False)

    wording = {f"{r.partition} {r.comparison} ({r.family})": dict(difference=r.difference, ci=[r.ci_low, r.ci_high],
                                                                  holm_p=r.holm_p, wording=r.wording)
               for r in comps[comps.family != "secondary"].itertuples()}
    for r in points[(points.row == "P4") & (points.score != "HSA")].itertuples():
        wording[f"{r.partition} P4 Bliss Pearson CI lower ({r.set}, {r.score})"] = dict(
            ci_low=r.pearson_ci_low, wording="不训练的参照已带有可检出的交互信号" if r.pearson_ci_low > 0 else "区间下限不大于 0，不作该表述")
    summary = dict(
        setting="D", card="bisurfgnn/docs/EVAL_INTERACTION_LEVEL_20261029.md", partitions=list(PARTS), split_read=[1, 2],
        c=C, reference="S1 lookup monos (mono_a_percent, mono_b_percent)", partitions_diag=diags, wording=wording,
        bootstrap=dict(seed=im.BOOT_SEED, draws=N_BOOT, unit="test drug pair; pralatrexate set masks the same draws"),
        files_read=sorted(set(FILES_READ)),
        versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__),
        peak_working_set_mb=round(psutil.Process().memory_info().peak_wset / 2 ** 20),
        runtime_seconds=round(time.perf_counter() - T0, 1), finished_local_time=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    with (OUT / "summary.json").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, indent=1, ensure_ascii=False, default=float) + "\n")
    pd.set_option("display.width", 250)
    print(points[["partition", "set", "row", "score", "pearson", "pearson_ci_low", "pearson_ci_high", "spearman",
                  "precision_top5", "rmse", "slope"]].round(4).to_string(index=False))
    print(comps[comps.family != "secondary"][["partition", "set", "comparison", "score", "difference", "ci_low", "ci_high",
                                              "boot_p", "holm_p", "family", "wording"]].round(4).to_string(index=False))
    print(f"done in {summary['runtime_seconds']} s, peak {summary['peak_working_set_mb']} MB")


if __name__ == "__main__":
    main()
