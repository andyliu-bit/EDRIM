"""Joint O'Neil scorer (card E v2, DEVIATIONS D13): the ONLY code allowed to decode O'Neil test responses.

  --role validation  dry run: scores every method's saved validation predictions against the validation responses
                     (common.load_points("validation")). No marker. Writes oneil_validation_dryrun_results.{json,md}.
  --role test        once, after every method is selected on validation and has written its test predictions:
                     requires all test prediction files (hsa deterministic; xgb_id, ours, ours_e500, deepsynba_4x4,
                     deepsynba_5x5, ddprism seeds 41-45) and deepsynba_primary.json, creates
                     runs/oneil_v1/eval/ONEIL_TEST_EVALUATED.marker exclusively, and only then decodes the test
                     responses in memory through the old project's build_surfaces(("confirmation",)). Responses are
                     never written to disk. Refuses if the marker or an output exists, or if ONEIL_PRED_ROOT is set.
                     Does not call common.install_guard (build_surfaces reads the combination workbook); a narrower
                     audit hook still refuses the other forbidden files.

Metrics (one code path for both roles), x100 (%): pooled RMSE over the 16 combination cells per method/seed (primary),
surface-macro RMSE, 5-seed mean and SD, 5-seed ensemble (extra row); r = mean(Ours 5 seeds RMSE) / mean(comparator
5 seeds RMSE) - 1 with a drug-pair bootstrap (2000 draws, seed 20261009, the NCI final-evaluation mechanism: each draw
resamples the pairs once, every method uses the same draw, both 5-seed means are recomputed); verdict = 95% upper bound
< 0 and >= 4 Ours seeds below the comparator's 5-seed mean, for (1) the DeepSynBa primary version and (2) xgb_id;
HSA and the rest descriptive; per-batch breakdown (interval only for batch 1); Bliss synergy-score Pearson with the
NCI synergy_check definition on 4x4 surfaces and same-batch interpolated anchors.
Ours (D13, 10-20 20:30): the primary Ours for every statistic and the H-E verdicts is ours_e500 (500 epochs + early
stopping); the card's 60-epoch ours is scored in the same run with the same contrasts, verdict rule and Bliss contrasts,
reported as a separate block "卡原设定（60 轮）" (verdicts reported, not the headline); ours_e500 vs ours is descriptive.

  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe experiments/oneil/eval/evaluate_oneil.py --role validation
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe experiments/oneil/eval/evaluate_oneil.py --role test   # once
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import common  # noqa: E402
import select_deepsynba_version as sel  # noqa: E402

sys.path.insert(0, str(common.ROOT / "experiments/model_improvement/synergy_check"))
from synergy_check import matrix_scores, references, w_pearson_fixed  # noqa: E402  (NCI Bliss definition, reused)

# combokr2 (D13 supplement), combofm (D21) and deepdrp are scored by their own one-shot scripts, not here
METHODS = tuple(m for m in common.METHODS if m not in ("combokr2", "combofm", "deepdrp"))
SEEDED = tuple(m for m in METHODS if m != "hsa")
NAMES = {"hsa": "HSA", "xgb_id": "XGBoost 加身份码", "ours": "Ours（卡原设定 60 轮）",
         "ours_e500": "Ours（500 轮＋早停，D13）", "deepsynba_4x4": "DeepSynBa 原设定（4×4）",
         "deepsynba_5x5": "DeepSynBa 5×5 适配版", "ddprism": "DD-PRiSM（实测单药适配版）"}
OURS, OURS_CARD = "ours_e500", "ours"     # D13: primary Ours for the H-E verdicts / card's pre-registered 60 epochs
DRAWS, BOOT_SEED = 2000, 20261009
REL_TOL = 1e-12
EVAL = sel.EVAL
MARKER = EVAL / "ONEIL_TEST_EVALUATED.marker"
STEM = {"validation": "oneil_validation_dryrun_results", "test": "oneil_test_results"}
TEST_FORBIDDEN = tuple(f for f in common.FORBIDDEN if f not in ("supp_1_w2lrww.xls", "ONEIL_TEST_EVALUATED.marker"))


def _test_guard(event, args):
    if event == "open" and isinstance(args[0], (str, bytes, Path)):
        path = str(args[0]).replace("\\", "/")
        if any(f in path for f in TEST_FORBIDDEN):
            raise PermissionError(f"Card E forbids opening {path}")


def tag(seed):
    return "deterministic" if seed is None else str(seed)


def runs():
    return [("hsa", None)] + [(m, s) for m in SEEDED for s in common.SEEDS]


def load_predictions(role, points):
    """{(method, seed): ratio-scale predictions aligned row by row with `points`}; refuses if any file is missing."""
    missing = [str(common.pred_path(m, s, role)) for m, s in runs() if not common.pred_path(m, s, role).exists()]
    if missing:
        raise SystemExit("STOP: missing prediction files:\n  " + "\n  ".join(missing))
    out = {}
    for m, s in runs():
        p = points[common.KEY].merge(common.read_predictions(m, s, role)[common.KEY + ["prediction"]], on=common.KEY,
                                     how="left", validate="one_to_one")["prediction"].to_numpy(float)
        if not np.isfinite(p).all():
            raise SystemExit(f"STOP: {m}/{tag(s)}/{role}: predictions missing or non-finite for some points")
        out[(m, s)] = p
    return out


def bootstrap_weights(n_pairs):
    """(DRAWS, n_pairs) pair multiplicities; same generator and call order as public_split_gate/evaluate_gate.py."""
    rng = np.random.default_rng(BOOT_SEED)
    return np.stack([np.bincount(rng.integers(0, n_pairs, n_pairs), minlength=n_pairs)
                     for _ in range(DRAWS)]).astype(float)


def interval(x):
    return [float(v) for v in np.percentile(x, [2.5, 97.5])]


def rmse_part(points, preds, primary, with_ci, ours=OURS):
    y = points["response"].to_numpy(float)
    codes, pairs = pd.factorize(points["pair_id"], sort=True)
    n = np.bincount(codes, minlength=len(pairs)).astype(float)
    W = bootstrap_weights(len(pairs)) if with_ci else None
    seed_rmse, boot = {}, {}
    for (m, s), p in preds.items():
        seed_rmse.setdefault(m, {})[tag(s)] = common.pooled_rmse_percent(y, p)
        if with_ci:
            sse = np.bincount(codes, weights=((p - y) * 100.0) ** 2, minlength=len(pairs))
            boot.setdefault(m, []).append(np.sqrt((W @ sse) / (W @ n)))
    mean = {m: float(np.mean(list(v.values()))) for m, v in seed_rmse.items()}
    boot = {m: np.mean(v, axis=0) for m, v in boot.items()}
    contrasts = {}
    rest = [m for m in METHODS if m not in (primary, "xgb_id", "hsa", OURS, OURS_CARD)]
    for b in [primary, "xgb_id", "hsa"] + rest + ([OURS_CARD] if ours == OURS else []):   # ours_e500 vs ours: descriptive
        c = dict(ours_method=ours, ours_mean_rmse_percent=mean[ours], comparator_mean_rmse_percent=mean[b],
                 relative_difference=mean[ours] / mean[b] - 1.0,
                 ours_seeds_below_comparator_mean=int(sum(v < mean[b] for v in seed_rmse[ours].values())))
        if with_ci:
            c["ci95"] = interval(boot[ours] / boot[b] - 1.0)
        contrasts[b] = c
    return dict(surfaces=int(points["matrix_key"].nunique()), pairs=len(pairs), points=len(points),
                seed_rmse_percent=seed_rmse, mean_rmse_percent=mean, contrasts=contrasts)


def bliss_part(points, preds, primary):
    """NCI synergy_check: per surface S = mean_ij[a_i b_j / 100 - y_ij] (observed), S^ with the prediction in place of y,
    same observed anchors for every model; Pearson(S^, S) over surfaces; bootstrap draws identical to the RMSE ones."""
    surf, keys = pd.factorize(points["matrix_key"])
    i, j = points["row_i"].to_numpy() - 1, points["col_j"].to_numpy() - 1
    count = np.zeros((len(keys), 4, 4), int)
    np.add.at(count, (surf, i, j), 1)
    assert (count == 1).all(), "each surface must hold each of the 16 combination cells exactly once"

    def grid(values):
        g = np.full((len(keys), 4, 4), np.nan)
        g[surf, i, j] = np.asarray(values, float)
        return g

    y = grid(points["response"].to_numpy(float) * 100.0)
    ga, gb = grid(points["mono_a_percent"]), grid(points["mono_b_percent"])
    a, b = ga[:, :, 0], gb[:, 0, :]
    assert (ga == a[:, :, None]).all() and (gb == b[:, None, :]).all(), "anchors vary along the other drug's dose"
    ref = references(a, b)
    observed = matrix_scores(ref, y)
    pcodes, pairs = pd.factorize(points.drop_duplicates("matrix_key")["pair_id"], sort=True)
    W = bootstrap_weights(len(pairs))[:, pcodes]
    seed_r, boot, derived = {}, {}, {}
    for (m, s), p in preds.items():
        g = grid(p * 100.0)
        d = matrix_scores(ref, g)
        e = (y - g).mean(axis=(1, 2))      # 4x4 check of the NCI invariance: both score errors = surface-mean error
        assert max(np.abs(d[x] - observed[x] - e).max() for x in ("HSA", "Bliss")) < 1e-9, (m, s)
        seed_r.setdefault(m, {})[tag(s)] = float(np.corrcoef(d["Bliss"], observed["Bliss"])[0, 1])
        boot.setdefault(m, []).append(w_pearson_fixed(d["Bliss"], observed["Bliss"], W))
        derived.setdefault(m, []).append(d["Bliss"])
    methods = {m: dict(seed_pearson=seed_r[m], mean_pearson=float(np.mean(list(seed_r[m].values()))),
                       # score of the 5-seed ensemble = mean of the seed scores (the score is linear in the prediction)
                       ensemble_pearson=float(np.corrcoef(np.mean(derived[m], axis=0), observed["Bliss"])[0, 1]))
               for m in METHODS}
    def contrasts(ours):
        return {c: dict(ours_method=ours, difference=methods[ours]["mean_pearson"] - methods[c]["mean_pearson"],
                        ci95=interval(np.mean(boot[ours], axis=0) - np.mean(boot[c], axis=0)),
                        ours_seeds_above_comparator_mean=int(sum(v > methods[c]["mean_pearson"]
                                                                 for v in seed_r[ours].values())))
                for c in (primary, "xgb_id")}
    return dict(definition="Bliss excess per surface, observed same-batch interpolated anchors for every model; "
                           "single model = mean of the 5 seed Pearsons; ensemble = Pearson of the mean seed score",
                surfaces=len(keys), pairs=len(pairs), observed_score_mean=float(observed["Bliss"].mean()),
                observed_score_sd=float(observed["Bliss"].std()), methods=methods, contrasts=contrasts(OURS),
                card_setting_contrasts=contrasts(OURS_CARD))


def score(points, preds, primary):
    main = rmse_part(points, preds, primary, with_ci=True)
    methods = {}
    for m in METHODS:
        keys = [k for k in preds if k[0] == m]
        macro = {tag(s): common.surface_macro_percent(points, preds[(m, s)]) for _, s in keys}
        x = dict(seeds={t: dict(pooled_rmse_percent=main["seed_rmse_percent"][m][t], surface_macro_percent=macro[t])
                        for t in macro},
                 mean_rmse_percent=main["mean_rmse_percent"][m],
                 sd_rmse_percent=float(np.std(list(main["seed_rmse_percent"][m].values()), ddof=1)) if len(keys) > 1 else None,
                 mean_surface_macro_percent=float(np.mean(list(macro.values()))))
        if len(keys) > 1:
            ens = np.mean([preds[k] for k in keys], axis=0)
            x["ensemble"] = dict(pooled_rmse_percent=common.pooled_rmse_percent(points["response"], ens),
                                 surface_macro_percent=common.surface_macro_percent(points, ens))
        methods[m] = x

    def judged(part, ours):
        ens_r = {b: methods[ours]["ensemble"]["pooled_rmse_percent"] / part["mean_rmse_percent"][b] - 1.0
                 for b in part["contrasts"]}
        verdicts = {}
        for label, b in (("(1) vs DeepSynBa primary", primary), ("(2) vs xgb_id", "xgb_id")):
            c = part["contrasts"][b]
            verdicts[label] = dict(comparator=b, relative_difference=c["relative_difference"], ci95=c["ci95"],
                                   ours_seeds_below_comparator_mean=c["ours_seeds_below_comparator_mean"],
                                   passed=bool(c["ci95"][1] < 0 and c["ours_seeds_below_comparator_mean"] >= 4))
        return dict(ours_method=ours, contrasts=part["contrasts"], ours_ensemble_relative_difference=ens_r,
                    verdicts=verdicts)

    head = judged(main, OURS)
    v1, v2 = (v["passed"] for v in head["verdicts"].values())
    interpretation = ("(1)(2) 都成立：新组合主结论在 O'Neil 上复现" if v1 and v2 else
                      "只有 (1) 成立：相对 DeepSynBa 的优势可由同批次单药锚点解释，模型增量没有迁移" if v1 else
                      "(1) 不成立：按原样写成负结果并分析条件，不换指标")
    batch = points["batch"].astype(str).to_numpy()
    per_batch = {}
    for b in sorted(set(batch)):
        mask = batch == b
        per_batch[b] = rmse_part(points[mask], {k: v[mask] for k, v in preds.items()}, primary, with_ci=(b == "1"))
    bliss = bliss_part(points, preds, primary)
    card = dict(label=NAMES[OURS_CARD], note="卡原设定（预先登记），同一次联合评分、同一统计；判定只作报告，H-E 判定以 ours_e500 为准",
                **judged(rmse_part(points, preds, primary, with_ci=True, ours=OURS_CARD), OURS_CARD),
                bliss_contrasts=bliss["card_setting_contrasts"])
    return dict(surfaces=main["surfaces"], pairs=main["pairs"], points=main["points"], methods=methods,
                **head, interpretation=interpretation, card_setting_60_epochs=card, per_batch=per_batch,
                bliss=bliss, bootstrap=dict(unit="drug pair", draws=DRAWS, seed=BOOT_SEED, interval="95% percentile"))


def primary_record(required):
    """deepsynba_primary.json, checked against a recomputation from the saved validation predictions."""
    recomputed = sel.choose()
    if not sel.PRIMARY_JSON.exists():
        if required:
            raise SystemExit(f"STOP: {sel.PRIMARY_JSON} missing; run select_deepsynba_version.py first")
        return dict(recomputed, source="provisional: deepsynba_primary.json not written yet")
    saved = json.loads(sel.PRIMARY_JSON.read_text(encoding="utf-8"))
    if saved["primary"] != recomputed["primary"] or any(
            abs(saved["versions"][v]["mean_val_surface_macro_percent"]
                - recomputed["versions"][v]["mean_val_surface_macro_percent"]) > 1e-9 for v in sel.VERSIONS):
        raise SystemExit("STOP: deepsynba_primary.json no longer matches the saved DeepSynBa validation predictions")
    return dict(saved, source=str(sel.PRIMARY_JSON))


def decode_test(build_surfaces):
    """Decoded confirmation surfaces -> common.KEY rows on the 30 cells; response half of the v2 eligibility rule."""
    sys.path.insert(0, str(common.ROOT / "src"))
    from bisurfgnn.data.cellminer import normalize_cell_label
    inputs = common.load_points("test")
    cells = set(inputs["cell_id"])
    built, counters = build_surfaces(("confirmation",))
    surfaces = built["confirmation"]
    rows, outside, dropped = [], 0, {}
    for s in surfaces:
        cell = normalize_cell_label(s["cell"])
        if cell not in cells:
            outside += 1
            continue
        a, b = s["drugs"]
        key = f"{s['batch']}|{cell}|{a}|{b}"
        pts = []
        for p in s["points"]:
            flip = p["drug_a"] != a
            assert ((p["drug_b"], p["drug_a"]) if flip else (p["drug_a"], p["drug_b"])) == (a, b), key
            da, db = (p["dose_b_uM"], p["dose_a_uM"]) if flip else (p["dose_a_uM"], p["dose_b_uM"])
            ma, mb = (p["mono_b"], p["mono_a"]) if flip else (p["mono_a"], p["mono_b"])
            pts.append((Decimal(da), Decimal(db), ma, mb, p["response"]))
        rank_a = {x: k + 1 for k, x in enumerate(sorted({q[0] for q in pts}))}
        rank_b = {x: k + 1 for k, x in enumerate(sorted({q[1] for q in pts}))}
        assert len(pts) == 16 and len(rank_a) == len(rank_b) == 4 and len({q[:2] for q in pts}) == 16, key
        if not all(q[4] is not None and math.isfinite(q[4]) for q in pts):
            dropped[key] = str(s["batch"])
        rows += [dict(matrix_key=key, row_i=rank_a[da], col_j=rank_b[db], dose_a_dec=float(da * Decimal("1e-6")),
                      dose_b_dec=float(db * Decimal("1e-6")), mono_a_dec=ma, mono_b_dec=mb, response=r)
                 for da, db, ma, mb, r in pts]
    mapped = inputs.merge(pd.DataFrame(rows), on=common.KEY, how="outer", validate="one_to_one", indicator=True)
    assert (mapped["_merge"] == "both").all(), \
        f"decoded points do not map one-to-one onto points_test_inputs: {mapped['_merge'].value_counts().to_dict()}"
    for dec, ref in (("dose_a_dec", "dose_a_M"), ("dose_b_dec", "dose_b_M"),
                     ("mono_a_dec", "mono_a_ratio_raw"), ("mono_b_dec", "mono_b_ratio_raw")):
        got, want = mapped[dec].astype(float).to_numpy(), mapped[ref].to_numpy(float)
        assert (np.abs(got - want) <= REL_TOL * np.abs(want)).all(), f"{dec} differs from {ref} beyond 1e-12 relative"
    mapped["response"] = mapped["response"].astype(float)
    points = mapped[~mapped["matrix_key"].isin(dropped)].drop(
        columns=["_merge", "dose_a_dec", "dose_b_dec", "mono_a_dec", "mono_b_dec"]).reset_index(drop=True)
    assert np.isfinite(points["response"]).all()
    report = dict(confirmation_surfaces_decoded=len(surfaces), outside_30_cells=outside,
                  in_30_cells=len(surfaces) - outside, mapped_points=len(mapped),
                  dropped_surfaces_response_not_finite=len(dropped), dropped_by_batch=dict(Counter(dropped.values())),
                  pairs_lost_entirely=sorted(set(inputs["pair_id"]) - set(points["pair_id"])),
                  eligible_surfaces=int(points["matrix_key"].nunique()), eligible_pairs=int(points["pair_id"].nunique()),
                  eligible_surfaces_by_batch=dict(Counter(points.drop_duplicates("matrix_key")["batch"].astype(str))),
                  builder_counters=counters)
    return points, report


def write_outputs(result, role, exclusive):
    EVAL.mkdir(parents=True, exist_ok=True)
    mode = "x" if exclusive else "w"
    with (EVAL / f"{STEM[role]}.json").open(mode, encoding="utf-8") as handle:
        json.dump(result, handle, indent=1, ensure_ascii=False)
    with (EVAL / f"{STEM[role]}.md").open(mode, encoding="utf-8") as handle:
        handle.write(markdown(result))


def run_validation():
    t0 = time.perf_counter()
    primary = primary_record(required=False)
    points = common.load_points("validation")
    preds = load_predictions("validation", points)
    result = dict(role="validation", nature="dry run, not a result: validation also selected every epoch/tree count",
                  pred_root=str(common.PRED), deepsynba_primary=primary, **score(points, preds, primary["primary"]))
    result["runtime_seconds"] = round(time.perf_counter() - t0, 1)
    write_outputs(result, "validation", exclusive=False)
    return result


def run_test(build_surfaces):
    t0 = time.perf_counter()
    outputs = [EVAL / f"{STEM['test']}.{x}" for x in ("json", "md")]
    if MARKER.exists():
        raise SystemExit(f"STOP: {MARKER} exists; the O'Neil test set is scored only once")
    if any(p.exists() for p in outputs):
        raise SystemExit(f"STOP: test outputs already exist in {EVAL}")
    primary = primary_record(required=True)
    load_predictions("test", common.load_points("test"))     # every test prediction file present and complete
    EVAL.mkdir(parents=True, exist_ok=True)
    with MARKER.open("x", encoding="utf-8") as handle:
        json.dump(dict(created_local=time.strftime("%Y-%m-%dT%H:%M:%S%z"), pred_root=str(common.PRED),
                       deepsynba_primary=primary["primary"]), handle)
    points, decode = decode_test(build_surfaces)
    preds = load_predictions("test", points)
    result = dict(role="test", nature="single joint test scoring (card E v2); responses decoded in memory only",
                  pred_root=str(common.PRED), marker=str(MARKER), decode=decode, deepsynba_primary=primary,
                  **score(points, preds, primary["primary"]))
    result["runtime_seconds"] = round(time.perf_counter() - t0, 1)
    write_outputs(result, "test", exclusive=True)
    return result


def _pct(x):
    return f"{x * 100:+.1f}%"


def markdown(r):
    p = r["deepsynba_primary"]["primary"]
    head = ("O'Neil 联合评分：test（一次性）" if r["role"] == "test" else
            "O'Neil 联合评分：validation 预演（不是结果；val 同时用于选轮/选树）")
    L = [f"# {head}", "",
         f"- 曲面 {r['surfaces']}，药对 {r['pairs']}，联用格 {r['points']}；单位 %，RMSE 为 16 个联用格合并。",
         f"- DeepSynBa 主比较版本：{NAMES[p]}（{r['deepsynba_primary']['source']}）。",
         f"- 预测目录：`{r['pred_root']}`。"]
    if r["role"] == "test":
        d = r["decode"]
        L.append(f"- 解码：{d['confirmation_surfaces_decoded']} 张确认曲面，30 个细胞外 {d['outside_30_cells']} 张；"
                 f"响应不全有限而整面排除 {d['dropped_surfaces_response_not_finite']} 张（{d['dropped_by_batch']}），"
                 f"整对丢失 {len(d['pairs_lost_entirely'])} 对。")
    L += ["", "## 表 1　联用格 RMSE（%）", "",
          "| 方法 | 种子数 | 合并 RMSE，种子均值 | 种子 SD | surface-macro，种子均值 | 5 种子集成（附加） |",
          "|---|---:|---:|---:|---:|---:|"]
    for m in METHODS:
        x = r["methods"][m]
        sd = "—" if x["sd_rmse_percent"] is None else f"{x['sd_rmse_percent']:.3f}"
        ens = f"{x['ensemble']['pooled_rmse_percent']:.3f}" if "ensemble" in x else "—"
        L.append(f"| {NAMES[m]} | {len(x['seeds'])} | {x['mean_rmse_percent']:.3f} | {sd} | "
                 f"{x['mean_surface_macro_percent']:.3f} | {ens} |")
    for m in (OURS, OURS_CARD):
        L.append(f"\n{NAMES[m]} 各种子：" + "、".join(f"{t} {v['pooled_rmse_percent']:.3f}"
                                                   for t, v in r["methods"][m]["seeds"].items()))

    def contrast_table(block):
        rows = ["| 对照 | r | 95% 区间 | Ours 低于对照均值的种子 | 判定 | 集成的 r（附加） |", "|---|---:|---|---:|---|---:|"]
        judged = {v["comparator"]: v["passed"] for v in block["verdicts"].values()}
        for b, c in block["contrasts"].items():
            verdict = ("成立" if judged[b] else "不成立") if b in judged else "仅描述"
            rows.append(f"| {NAMES[b]} | {_pct(c['relative_difference'])} | [{_pct(c['ci95'][0])}, {_pct(c['ci95'][1])}] | "
                        f"{c['ours_seeds_below_comparator_mean']}/5 | {verdict} | {_pct(block['ours_ensemble_relative_difference'][b])} |")
        return rows

    def bliss_rows(contrasts):
        return [f"| {NAMES[c]} | {x['difference']:+.3f} | [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}] | "
                f"{x['ours_seeds_above_comparator_mean']}/5 |" for c, x in contrasts.items()]

    L += ["", f"## 表 2　相对差 r = {NAMES[OURS]} 5 种子均值 ÷ 对照 5 种子均值 − 1（H-E 判定）", "",
          f"按药对 bootstrap {r['bootstrap']['draws']} 次，种子 {r['bootstrap']['seed']}。判定：区间上界 < 0 且至少 4 个 Ours 种子低于对照的 5 种子均值。"
          f"与 {NAMES[OURS_CARD]} 的对比只作描述。", ""] + contrast_table(r)
    L += ["", f"**判读**：{r['interpretation']}", "",
          f"## 表 3　按 batch（只作描述；batch 2、3 不报区间；r 为 {NAMES[OURS]}）", "",
          "| batch | 曲面 | 药对 | " + " | ".join(NAMES[m] for m in METHODS) + f" | r vs {NAMES[p]} | r vs XGBoost 加身份码 |",
          "|---|---:|---:|" + "---:|" * len(METHODS) + "---|---|"]
    for b, x in r["per_batch"].items():
        def rel(c):
            ci = f" [{_pct(c['ci95'][0])}, {_pct(c['ci95'][1])}]" if "ci95" in c else ""
            return _pct(c["relative_difference"]) + ci
        L.append(f"| {b} | {x['surfaces']} | {x['pairs']} | " + " | ".join(f"{x['mean_rmse_percent'][m]:.3f}" for m in METHODS)
                 + f" | {rel(x['contrasts'][p])} | {rel(x['contrasts']['xgb_id'])} |")
    bl = r["bliss"]
    L += ["", "## 表 4　Bliss 协同分数 Pearson（次要指标，NCI 表 2 同口径，参照用同批次插值单药）", "",
          f"Ours = {NAMES[OURS]}。", "",
          "| 方法 | Pearson，5 种子均值 | 集成 Pearson（附加） | Ours − 该方法 | 95% 区间 | Ours 更高的种子 |",
          "|---|---:|---:|---:|---|---:|"]
    for m in METHODS:
        x, c = bl["methods"][m], bl["contrasts"].get(m)
        tail = (f"{c['difference']:+.3f} | [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] | {c['ours_seeds_above_comparator_mean']}/5"
                if c else "— | — | —")
        L.append(f"| {NAMES[m]} | {x['mean_pearson']:.3f} | {x['ensemble_pearson']:.3f} | {tail} |")
    cs = r["card_setting_60_epochs"]
    verdicts = "；".join(f"{k}：{'成立' if v['passed'] else '不成立'}" for k, v in cs["verdicts"].items())
    L += ["", f"## 表 5　卡原设定（60 轮）：{NAMES[OURS_CARD]}", "",
          f"{cs['note']}。表 1 已列其 RMSE；同一 bootstrap 抽样、同一判定规则。按规则：{verdicts}。", "",
          f"r = {NAMES[OURS_CARD]} 5 种子均值 ÷ 对照 5 种子均值 − 1：", ""] + contrast_table(cs)
    L += ["", f"Bliss Pearson：{NAMES[OURS_CARD]} − 该方法", "",
          "| 对照 | 差 | 95% 区间 | Ours 更高的种子 |", "|---|---:|---|---:|"] + bliss_rows(cs["bliss_contrasts"])
    L += ["", f"运行 {r['runtime_seconds']} s。", ""]
    return "\n".join(L)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True, choices=("validation", "test"))
    role = parser.parse_args().role
    if role == "validation":
        common.install_guard()
        result = run_validation()
    else:
        if "ONEIL_PRED_ROOT" in os.environ:
            raise SystemExit("STOP: --role test scores the real prediction tree; unset ONEIL_PRED_ROOT")
        sys.addaudithook(_test_guard)
        sys.dont_write_bytecode = True                       # the old repo is read-only
        sys.path.insert(0, str(common.OLD / "experiments/audit_revision"))
        import evaluate_oneil_confirmation as old            # import only; nothing is decoded before the marker exists
        result = run_test(old.build_surfaces)
    print(json.dumps(dict(role=role, surfaces=result["surfaces"], pairs=result["pairs"],
                          mean_rmse_percent={m: round(v["mean_rmse_percent"], 4) for m, v in result["methods"].items()},
                          verdicts={k: v["passed"] for k, v in result["verdicts"].items()},
                          card_setting_60_epochs_verdicts={k: v["passed"] for k, v in
                                                           result["card_setting_60_epochs"]["verdicts"].items()},
                          interpretation=result["interpretation"], out=str(EVAL / STEM[role])),
                     ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
