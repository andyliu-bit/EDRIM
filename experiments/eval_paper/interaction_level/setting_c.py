"""Eval-paper interaction-level reanalysis, setting C (new cell, development).

Frozen card: bisurfgnn/docs/EVAL_INTERACTION_LEVEL_20261029.md. O'Neil and NCI five comboKR new-cell splits (Zenodo
14509938) exactly as in C4 / task B (bisurfgnn/docs/EVAL_NEWCELL_STATS_UNIFICATION_20261018.md): the frozen loader and
rules (run_cross_dataset_mean_controls.load/predict, run_learned_hsa_calibration.fit/predict) are imported unchanged.
The only addition: the measured target-cell monos a, b behind each point's HSA anchor are captured in-process from the
frozen builders (asserted min(a, b) == anchor). Viability fraction, c = 1; NCI NoTZ/100 unclipped. Ridge uses the
saved selected lambda; every prediction array must match its saved SHA256 and the per-surface errors must equal task
B's CSVs. No AZ, Jaaks or DeepSynBa file is opened. Nothing is trained or tuned.

Needs the bisurfgnn venv (the frozen modules run there); from bisurfgnn-model-experiments:
  ../bisurfgnn/.venvs/strong-baseline-replay/Scripts/python.exe -I experiments/eval_paper/interaction_level/setting_c.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

T0 = time.perf_counter()
HERE = Path(__file__).resolve().parent
EXP = HERE.parents[2]
BS = EXP.parent / "bisurfgnn"
AR = BS / "experiments/audit_revision"
sys.path[:0] = [str(HERE), str(AR)]  # -I drops the script dir; frozen modules import each other by bare name
import interaction_metrics as im  # noqa: E402
import run_cross_dataset_mean_controls as controls  # noqa: E402
import run_learned_hsa_calibration as learned  # noqa: E402
import external_combokr2_oneil_newcell as oneil_src  # noqa: E402
import external_combokr2_almanac_newcell as nci_src  # noqa: E402

OUT = EXP / "runs/eval_paper/interaction_level/C"
TASK_B = BS / "runs/audit_revision/eval_newcell_stats"
C = 1.0
METHODS = ("H", "R-p", "R-pd", "ridge")            # H = min(a, b): also the mono-implied prediction
SERIES = ("MONO", "R-p", "R-pd", "ridge")          # scored series; the H row is MONO's RMSE only
ROWS = ("H",) + SERIES
SCORES = ("HSA", "Bliss", "Bliss_clip")            # Bliss_clip = sensitivity: monos clipped to [0, 1]
METRICS = ("pearson", "spearman", "precision_top5", "rmse", "cell_spearman")
COMPARISONS = (("ridge", "R-pd"), ("R-pd", "R-p"))
TOL = 1e-12


def degenerate(s: str, k: str) -> bool:
    return s == "H" or (s == "MONO" and k == "HSA")


def digest(a) -> str:
    return hashlib.sha256(np.asarray(a, dtype="<f8").tobytes()).hexdigest()


def peak_mb() -> float:
    return psutil.Process().memory_info().peak_wset / 2 ** 20


# ---------------------------------------------------------------------------------- per-point predictions
def load(dataset):
    """Frozen controls.load, plus the measured monos a, b captured from its builder in the same row order."""
    cap = {}
    mod, name = (oneil_src.oneil, "build_surfaces") if dataset == "oneil" else (nci_src, "build_points")
    orig = getattr(mod, name)

    def capture(*args, **kw):
        cap["out"] = orig(*args, **kw)
        return cap["out"]

    setattr(mod, name, capture)
    try:
        df, splits, _ = controls.load(dataset)
    finally:
        setattr(mod, name, orig)
    if dataset == "oneil":   # same iteration order as controls.load
        mono = np.array([(p["mono_a"], p["mono_b"]) for role in cap["out"][0].values() for s in role
                         if s["primary_eligible"] for p in s["points"]], float)
    else:
        mono = cap["out"][0][["ma", "mb"]].to_numpy(float)
    del cap
    if mono.shape != (len(df), 2) or not np.array_equal(np.minimum(mono[:, 0], mono[:, 1]), df.h.to_numpy()):
        raise RuntimeError(f"{dataset}: captured monos do not reproduce the frozen HSA anchor")
    df["a"], df["b"] = mono[:, 0], mono[:, 1]
    return df, splits


def regenerate(dataset: str, out: Path):
    ctrl = controls.read(BS / f"docs/results/cross_dataset_mean_controls_v1/{dataset}/result.json")["seeds"]
    lrn = controls.read(BS / f"docs/results/learned_hsa_calibration_v1/{dataset}/result.json")["folds"]
    df, splits = load(dataset)
    df["unit"] = df.dose  # as in run_learned_hsa_calibration.load for O'Neil / NCI
    cells = np.array(sorted(df.cell.unique()))
    parts, lambdas = [], {}
    for seed, (tr, te) in splits.items():
        s = str(seed)
        train, test = df.loc[tr], df.loc[te]
        pred, _ = controls.predict(train, test)
        lam = lrn[s]["selected_lambda"]            # saved selection; no inner CV re-run, hash below proves identity
        ridge = learned.predict(learned.fit(train), test, lam)
        if digest(pred) != ctrl[s]["prediction_sha256"] or digest(ridge) != lrn[s]["prediction_sha256"]:
            raise RuntimeError(f"{dataset} seed {seed}: prediction SHA256 differs from the saved results")
        lambdas[s] = lam
        parts.append(pd.DataFrame({"seed": np.int8(seed), "surface": test.surface.to_numpy(np.int64),
                                   "cell_code": np.searchsorted(cells, test.cell.to_numpy()).astype(np.int32),
                                   "y": test.y.to_numpy(), "a": test.a.to_numpy(), "b": test.b.to_numpy(),
                                   "H": pred[:, 0], "R-p": pred[:, 1], "R-pd": pred[:, 2], "ridge": ridge}))
        del train, test, pred, ridge
    del df, splits
    pts = pd.concat(parts, ignore_index=True)
    del parts
    if not np.array_equal(pts.H.to_numpy(), np.minimum(pts.a, pts.b).to_numpy()):
        raise RuntimeError("H prediction is not min(a, b)")
    check = check_task_b(dataset, pts, cells)
    np.savez_compressed(out / f"{dataset}_points.npz", cells=cells,
                        **{c.replace("-", "_"): pts[c].to_numpy() for c in pts.columns})
    return pts, cells, dict(lambdas=lambdas, prediction_sha256_match=True, task_b_per_surface=check)


def check_task_b(dataset, pts, cells) -> dict:
    """Card self-check: per-surface errors of the regenerated points equal task B's per-surface CSV."""
    e = pts[["seed", "surface", "cell_code"]].copy()
    for m in METHODS:
        e["sse_" + m] = (pts[m] - pts.y) ** 2
    e["n_points"] = 1
    mine = e.groupby(["seed", "surface"], sort=True).agg(
        cell_code=("cell_code", "first"), n_points=("n_points", "sum"),
        **{"sse_" + m: ("sse_" + m, "sum") for m in METHODS}).reset_index()
    ref = pd.read_csv(TASK_B / f"{dataset}_per_surface.csv", dtype={"cell": str},
                      usecols=["seed", "cell", "surface", "n_points"] + ["sse_" + m for m in METHODS])
    ref = ref.sort_values(["seed", "surface"]).reset_index(drop=True)
    same_keys = (len(ref) == len(mine) and np.array_equal(ref.seed, mine.seed) and np.array_equal(ref.surface, mine.surface)
                 and np.array_equal(ref.n_points, mine.n_points) and np.array_equal(ref.cell.to_numpy(), cells[mine.cell_code]))
    diffs = {m: float(np.max(np.abs(mine["sse_" + m].to_numpy() - ref["sse_" + m].to_numpy()))) for m in METHODS}
    if not same_keys or max(diffs.values()) > TOL:
        raise RuntimeError(f"{dataset}: per-surface errors differ from task B ({same_keys}, {diffs})")
    return dict(surfaces=int(len(ref)), keys_cells_npoints_identical=True, max_abs_sse_diff=diffs)


# ---------------------------------------------------------------------------------------------- scoring
def analyse_split(dataset: str, seed: int, sp: pd.DataFrame, n_boot: int):
    codes, _ = pd.factorize(sp.surface.to_numpy(), sort=True)
    y, a, b = sp.y.to_numpy(), sp.a.to_numpy(), sp.b.to_numpy()
    ref = {"HSA": im.reference(a, b, "HSA", C), "Bliss": im.reference(a, b, "Bliss", C),
           "Bliss_clip": im.reference(np.clip(a, 0, 1), np.clip(b, 0, 1), "Bliss", C)}
    yhat = {"MONO": sp.H.to_numpy(), "R-p": sp["R-p"].to_numpy(), "R-pd": sp["R-pd"].to_numpy(), "ridge": sp.ridge.to_numpy()}
    observed = {k: im.matrix_score(ref[k], y, codes) for k in SCORES}
    derived = {s: {k: im.matrix_score(ref[k], p, codes) for k in SCORES} for s, p in yhat.items()}
    corollary = 0.0
    for s, p in yhat.items():   # card corollary: same score error under every reference = surface-mean point error
        e = im.matrix_mean(y - p, codes)
        corollary = max(corollary, *(float(np.max(np.abs(derived[s][k] - observed[k] - e))) for k in SCORES))
    if corollary > 1e-9 or np.max(np.abs(derived["MONO"]["HSA"])) != 0:
        raise RuntimeError(f"{dataset} {seed}: score corollary failed ({corollary})")
    # R-pd's dose residuals average to R-p's pair residual over a balanced surface: equal surface scores up to
    # float noise there; snap those to R-p so 1e-16 noise cannot produce a spurious bootstrap p
    gap = np.abs(im.matrix_mean(yhat["R-p"] - yhat["R-pd"], codes))
    tie = gap <= TOL
    for k in SCORES:
        derived["R-pd"][k] = np.where(tie, derived["R-p"][k], derived["R-pd"][k])

    surf_cell = np.zeros(codes.max() + 1, np.int64)
    surf_cell[codes] = sp.cell_code.to_numpy()
    _, cl = np.unique(surf_cell, return_inverse=True)
    n_cells = cl.max() + 1
    per_cell = np.bincount(cl)
    rho = {(s, k): np.array([im.spearman(derived[s][k][cl == j], observed[k][cl == j]) for j in range(n_cells)])
           for s in SERIES for k in SCORES if not degenerate(s, k)}
    if any(np.isnan(v).any() for v in rho.values()):
        raise RuntimeError("undefined within-cell Spearman")

    def stat(idx):
        out = {}
        w = np.bincount(cl[idx], minlength=n_cells) / per_cell     # whole-cell multiplicities
        for k in SCORES:
            o = observed[k][idx]
            for s in SERIES:
                p = derived[s][k][idx]
                out[f"{s}|{k}|rmse"] = im.rmse(p, o)
                if not degenerate(s, k):
                    out[f"{s}|{k}|pearson"] = im.pearson(p, o)
                    out[f"{s}|{k}|spearman"] = im.spearman(p, o)
                    out[f"{s}|{k}|precision_top5"] = im.precision_top(p, o)
                    out[f"{s}|{k}|cell_spearman"] = float(w @ rho[(s, k)] / w.sum())
        return out

    t = time.perf_counter()
    boot = im.bootstrap(stat, surf_cell, n_boot=n_boot)
    boot_s = time.perf_counter() - t
    for k in SCORES:
        boot[f"H|{k}|rmse"] = boot[f"MONO|{k}|rmse"]

    n_surf = len(surf_cell)
    rows = []
    for s in ROWS:
        src = "MONO" if s == "H" else s
        for k in SCORES:
            met = im.all_metrics(derived[src][k], observed[k], degenerate(s, k))
            met["cell_spearman"] = np.nan if degenerate(s, k) else float(rho[(src, k)].mean())
            r = dict(dataset=dataset, seed=seed, row=s, score=k, cells=int(n_cells), surfaces=n_surf,
                     k_top5=im.top_k(n_surf), degenerate=degenerate(s, k), **met)
            for m in METRICS:
                key = f"{s}|{k}|{m}"
                r[m + "_ci_low"], r[m + "_ci_high"] = im.percentile_ci(boot[key]) if key in boot else (np.nan, np.nan)
            rows.append(r)
    pe = pd.DataFrame(rows)
    comps = []
    for k in SCORES:
        for m in METRICS:
            for A, B in COMPARISONS:
                d = boot[f"{A}|{k}|{m}"] - boot[f"{B}|{k}|{m}"]
                av = pe[(pe.row == A) & (pe.score == k)][m].iloc[0]
                bv = pe[(pe.row == B) & (pe.score == k)][m].iloc[0]
                lo, hi = im.percentile_ci(d)
                fam = {"Bliss": "primary", "Bliss_clip": "sensitivity"}.get(k, "secondary") if m == "pearson" else "secondary"
                comps.append(dict(dataset=dataset, seed=seed, comparison=f"{A} - {B}", score=k, metric=m, a_value=av,
                                  b_value=bv, difference=av - bv, ci_low=lo, ci_high=hi, boot_p=im.boot_p(d),
                                  better_is="lower" if m == "rmse" else "higher", family=fam))
    edges = np.concatenate([a, b])
    info = dict(dataset=dataset, seed=seed, cells=int(n_cells), surfaces=n_surf, points=int(len(sp)),
                observed={k: dict(mean=float(observed[k].mean()), sd=float(observed[k].std())) for k in SCORES},
                mono_edges=dict(total=int(edges.size), below_0=int((edges < 0).sum()), above_1=int((edges > 1).sum())),
                corollary_max_abs=corollary, bootstrap_seconds=round(boot_s, 1),
                rpd_vs_rp_surface_score=dict(surfaces_differing=int((~tie).sum()), max_abs_diff=float(gap.max()),
                                             max_abs_diff_snapped=float(gap[tie].max()) if tie.any() else 0.0))
    return pe, pd.DataFrame(comps), info


def verdict(r) -> str:
    sig = r.holm_p < 0.05
    if r.comparison == "ridge - R-pd":   # learned model vs untrained reference: card wording
        if not sig:
            return "交互层上未见学习增益"
        return "学习在交互层有增益" if r.difference > 0 else "学习模型低于参照（显著为负）"
    if r.difference == 0 and r.ci_low == 0 and r.ci_high == 0:
        return "R-pd 与 R-p 的曲面分数逐曲面相同，差恒为 0"
    if not sig:                          # R-pd vs R-p: both untrained references, neutral wording
        return "R-pd 与 R-p 的差未检出"
    return "R-pd 高于 R-p（Holm 后显著）" if r.difference > 0 else "R-pd 低于 R-p（Holm 后显著）"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["oneil", "almanac"], choices=["oneil", "almanac"])
    ap.add_argument("--boot", type=int, default=im.BOOT_N)   # smoke tests only; the card fixes 2000
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # -I ignores PYTHONIOENCODING; the log carries Chinese wording
    out = args.out
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a non-empty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    im._self_test()

    pes, cps, infos, regen, mem = [], [], [], {}, {}
    for dataset in args.datasets:
        t = time.perf_counter()
        pts, cells, regen[dataset] = regenerate(dataset, out)
        print(f"{dataset}: per-point predictions regenerated, hashes and task-B per-surface errors match "
              f"({time.perf_counter() - t:.0f} s, peak {peak_mb():.0f} MB)", flush=True)
        for seed, sp in pts.groupby("seed", sort=True):
            pe, cp, info = analyse_split(dataset, int(seed), sp, args.boot)
            pes.append(pe); cps.append(cp); infos.append(info)
            b = pe[(pe.score == "Bliss")].set_index("row").pearson
            print(f"{dataset} seed {seed}: Bliss Pearson MONO {b['MONO']:.3f} R-p {b['R-p']:.3f} R-pd {b['R-pd']:.3f} "
                  f"ridge {b['ridge']:.3f}; bootstrap {info['bootstrap_seconds']} s", flush=True)
        mem[dataset] = round(peak_mb())
        del pts

    points = pd.concat(pes, ignore_index=True)
    points.to_csv(out / "point_estimates.csv", index=False)
    comps = pd.concat(cps, ignore_index=True)
    comps["holm_p"] = np.nan
    for fam in ("primary", "sensitivity"):
        sel = comps.family == fam
        comps.loc[sel, "holm_p"] = im.holm(comps.loc[sel, "boot_p"].to_numpy())
    comps["wording"] = [verdict(r) if r.family in ("primary", "sensitivity") else "" for r in comps.itertuples()]
    comps.to_csv(out / "paired_comparisons.csv", index=False)

    untrained = {}
    for k in ("Bliss", "Bliss_clip"):
        for r in points[(points.row.isin(["R-p", "R-pd"])) & (points.score == k)].itertuples():
            untrained[f"{r.dataset}/{r.seed}/{r.row}/{k}"] = dict(
                pearson=r.pearson, ci_low=r.pearson_ci_low,
                wording="不训练的参照已带有可检出的交互信号" if r.pearson_ci_low > 0 else "区间下限不大于 0，不作该表述")
    summary = dict(
        setting="C", card="bisurfgnn/docs/EVAL_INTERACTION_LEVEL_20261029.md", c=C, scale="viability fraction; NCI NoTZ/100 unclipped",
        bootstrap=dict(seed=im.BOOT_SEED, draws=args.boot, unit="test cell (whole cell, all its surfaces)", reinitialised_per_split=True),
        self_checks=dict(interaction_metrics_self_test=True, regenerate=regen,
                         monos_reproduce_frozen_hsa_anchor_exactly=True, h_prediction_equals_min_a_b=True,
                         score_corollary_max_abs=max(i["corollary_max_abs"] for i in infos), mono_hsa_score_identically_zero=True),
        splits=infos, primary_holm_members=int((comps.family == "primary").sum()),
        primary=comps[comps.family == "primary"][["dataset", "seed", "comparison", "difference", "ci_low", "ci_high", "boot_p",
                                                   "holm_p", "wording"]].to_dict(orient="records"),
        untrained_reference_wording=untrained,
        files_read=dict(
            frozen_code=[str(Path(m.__file__)) for m in (controls, learned, oneil_src, nci_src, oneil_src.oneil)],
            oneil_source="O'Neil v2 builder inputs (workbooks listed in prepare_oneil_development_v2.SOURCE, surface_roles.jsonl); "
                         "all roles incl. the whole-pair confirmation role, as in task B",
            nci_source=f"{nci_src.RAW} columns CELLNAME, NSC1, CONC1, NSC2, CONC2, PERCENTGROWTHNOTZ (as in task B)",
            splits=[str(oneil_src.SPLITS), str(nci_src.SPLITS)],
            saved_results=["docs/results/cross_dataset_mean_controls_v1/{oneil,almanac}/result.json",
                           "docs/results/learned_hsa_calibration_v1/{oneil,almanac}/result.json",
                           "runs/audit_revision/external_combokr2_{oneil,almanac}_newcell_v1/result.json",
                           "runs/audit_revision/eval_newcell_stats/{oneil,almanac}_per_surface.csv"],
            not_opened="AZ (any), Jaaks (any), DeepSynBa files incl. public split 3"),
        peak_working_set_mb=mem,
        versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__),
        runtime_seconds=round(time.perf_counter() - T0, 1), finished_local_time=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    with (out / "summary.json").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, indent=1, ensure_ascii=False, default=float) + "\n")
    pd.set_option("display.width", 250)
    print(comps[comps.family == "primary"][["dataset", "seed", "comparison", "difference", "ci_low", "ci_high", "boot_p",
                                            "holm_p", "wording"]].round(4).to_string(index=False))
    print(f"done in {summary['runtime_seconds']} s, peak {mem} MB")


if __name__ == "__main__":
    main()
