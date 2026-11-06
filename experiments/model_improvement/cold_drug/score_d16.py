"""D16 (card H, docs/COLD_DRUG_PARTITIONS_OURS_CARD_20261023.md): score Ours on the 5 new-drug partitions, once.

Per partition (D11 p20261016 + p20261018-21), new-drug test role, combination-cell pooled RMSE (%) on identical points:
Ours (mean of per-seed RMSE, seeds 41-43), XGBoost cap 20000 (seeds 41-43), P4, lookup HSA. Relative change Ours vs
XGBoost / lookup HSA / P4 with the eval-paper code (common.bootstrap: drug-pair bootstrap, 2000 shared draws, seed
20261009; common.compare: 95% percentile interval, two-sided bootstrap p; common.holm), Holm within each comparison
family across the 5 partitions, verdict better / no_difference / worse (Holm p < 0.05, as eval-paper score.py).
Card wording rule: Ours not better than XGBoost (interval not entirely < 0) in >= 4/5 partitions -> the fixed sentence;
otherwise the pattern. D11 also gets a descriptive 5-seed row (Ours 41-45 vs XGBoost 41-45), checked against
cold_test_evaluation.json (v2_vs_xgb); HSA / P4 / XGBoost points are checked against baselines.json / evaluation.json.

Refuses unless all 12 D16 runs and D11 Ours seeds 41-45 have finished (non-smoke metrics.json + predictions); creates
runs/eval_paper/cold_drug_partitions/D16_OURS_TEST_SCORED.marker exclusively before any test response is read.
--dry-run: the same pipeline on the cold_val role (no marker, no test row used); --ours-root DIR (dry-run only) reads
Ours from DIR/p<partition>/ours_both_seed_<s> and writes the dry-run JSON/markdown there.

  .venv/Scripts/python.exe -I -B experiments/model_improvement/cold_drug/score_d16.py [--dry-run [--ours-root DIR]]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/eval_paper/cold_drug_partitions"))
import common as c  # noqa: E402
import score as ep  # noqa: E402  (eval-paper scorer: aligned, xgb_dir, COLS)

MARKER = c.OUT / "D16_OURS_TEST_SCORED.marker"
SEEDS, D11_SEEDS = c.XGB_SEEDS, (41, 42, 43, 44, 45)
FAMILIES = ("xgb", "lookup_hsa", "p4")
SETS = ("all", "excluding_pralatrexate")
FIXED = "Ours 相对 XGBoost 的优势不延伸到新药"
ZH = {"better": "更好", "no_difference": "无差别", "worse": "更差", "xgb": "XGBoost", "lookup_hsa": "查表 HSA", "p4": "P4"}


def ours_dir(part: int, seed: int, root: Path | None) -> Path:
    if root:
        return root / f"p{part}/ours_both_seed_{seed}"
    return c.D11_DIR / f"v2_both_seed_{seed}" if part == c.D11_SEED else c.OUT / f"p{part}/ours_both_seed_{seed}"


def pattern_text(pattern: dict) -> str:
    return " / ".join(f"{ZH[k]} {pattern[k]}" for k in ("better", "no_difference", "worse")) + " 个分区"


def cell(v: dict) -> str:
    lo, hi = v["ci95"]
    return f"{100 * v['relative_change']:+.2f}% [{100 * lo:+.2f}, {100 * hi:+.2f}]，p_Holm={v['p_holm']:.3f}，{ZH[v['verdict']]}"


def markdown(res: dict, role: str) -> str:
    head = "D16 dry-run（cold_val 演练，不是 test 结果）" if role == "cold_val" else "D16：Ours 在 5 个新药分区上的 test 结果"
    lines = [f"# {head}", "",
             "组合格合并 RMSE（%），各方法用同一组点；Ours 与 XGBoost 为种子 41–43 逐种子 RMSE 的均值。相对变化 < 0 表示 Ours 更好；"
             "区间为药对 bootstrap 95% 区间（2000 次，种子 20261009，各比较共用同一组抽样），p 为双侧 bootstrap p，"
             "按比较族（对 XGBoost / 对查表 HSA / 对 P4）在 5 个分区内做 Holm 校正。去掉 pralatrexate 药对的敏感性结果见 JSON。", "",
             "| 分区 | 药对 | 点数 | Ours | XGBoost | P4 | 查表 HSA | Ours 对 XGBoost | Ours 对查表 HSA | Ours 对 P4 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for part, row in res["partitions"].items():
        r = row["all"]
        pt, cm = r["point"], r["comparisons"]
        name = f"p{part}" + ("（D11）" if part == c.D11_SEED else "")
        lines.append(f"| {name} | {r['pairs']} | {r['points']} | {pt['ours']:.3f} | {pt['xgb']:.3f} | {pt['p4']:.3f} | "
                     f"{pt['lookup_hsa']:.3f} | {cell(cm['ours_vs_xgb'])} | {cell(cm['ours_vs_lookup_hsa'])} | {cell(cm['ours_vs_p4'])} |")
    d11 = res["partitions"][c.D11_SEED]["all"]
    v = d11["comparisons"]["ours_5seed_vs_xgb_5seed"]
    lines += ["", f"D11 5 种子（描述，不进 Holm）：Ours {d11['point']['ours_5seed']:.3f}，XGBoost {d11['point']['xgb_5seed']:.3f}，"
              f"相对变化 {100 * v['relative_change']:+.2f}% [{100 * v['ci95'][0]:+.2f}, {100 * v['ci95'][1]:+.2f}]。", ""]
    d = res["decision"]["all"]
    lines += [f"**写法（事先固定）**：{d['sentence_xgb']}", "",
              f"- 对 XGBoost：区间不全 < 0 的分区 {d['xgb_not_better_partitions']}/5（卡片口径）；Holm 判定 {pattern_text(d['pattern']['xgb'])}。",
              f"- 对查表 HSA：Holm 判定 {pattern_text(d['pattern']['lookup_hsa'])}。",
              f"- 对 P4：Holm 判定 {pattern_text(d['pattern']['p4'])}。", ""]
    return "\n".join(lines)


def main():
    if not (sys.flags.isolated and sys.flags.dont_write_bytecode):
        raise SystemExit("run with python -I -B")
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="cold_val role only; no marker, no test row used")
    ap.add_argument("--ours-root", type=Path, help="dry-run only: Ours runs under DIR/p<partition>/ours_both_seed_<s>")
    a = ap.parse_args()
    if a.ours_root and not a.dry_run:
        ap.error("--ours-root needs --dry-run")
    role = "cold_val" if a.dry_run else "cold_test"
    seeds = {p: D11_SEEDS if p == c.D11_SEED else SEEDS for p in c.PARTS}
    runs = {(p, s): ours_dir(p, s, a.ours_root) for p in c.PARTS for s in seeds[p]}
    need = [d / f for d in runs.values() for f in ("metrics.json", f"{role}_predictions.npz")]
    need += [ep.xgb_dir(p, s) / f"{role}_predictions.npz" for p in c.PARTS for s in seeds[p]]
    need += [c.OUT / f"p{p}/p4_predictions.npz" for p in c.PARTS] + [c.split_path(p) for p in c.PARTS]
    need += [c.OUT / "baselines.json"] + ([] if a.dry_run else [c.OUT / "evaluation.json", c.D11_DIR / "cold_test_evaluation.json"])
    missing = [str(f) for f in need if not f.exists()]
    if missing:
        raise SystemExit("Refused, inputs missing (all runs must have finished):\n  " + "\n  ".join(missing))
    bad = []
    for (p, s), d in runs.items():
        mj = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        if mj.get("smoke") is not False or mj.get("seed") != s or mj.get("partition", c.D11_SEED) != p:
            bad.append(str(d))
    if bad:
        raise SystemExit("Refused, smoke or mislabelled runs:\n  " + "\n  ".join(bad))
    if not a.dry_run:
        with MARKER.open("x", encoding="utf-8") as h:   # one-shot: a second scoring fails here
            h.write(json.dumps({"created_local_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "registered": "DEVIATIONS.md D16"}) + "\n")

    base = json.loads((c.OUT / "baselines.json").read_text(encoding="utf-8"))["partitions"]
    ev = None if a.dry_run else json.loads((c.OUT / "evaluation.json").read_text(encoding="utf-8"))["partitions"]
    d11 = None if a.dry_run else json.loads((c.D11_DIR / "cold_test_evaluation.json").read_text(encoding="utf-8"))["results"]
    both = c.read_points(ep.COLS)
    res = {"card": "docs/COLD_DRUG_PARTITIONS_OURS_CARD_20261023.md", "registered": "DEVIATIONS.md D16", "role": role,
           "dry_run": a.dry_run, "ours_root": str(a.ours_root) if a.ours_root else None,
           "bootstrap": {"unit": "test drug pair", "draws": c.DRAWS, "seed": c.BOOT_SEED}, "partitions": {}}
    for part in c.PARTS:
        fr = c.frames(part, both)[role]
        src = fr.source_combodrugseq.to_numpy()

        def load(d: Path) -> np.ndarray:
            return ep.aligned(np.load(d / f"{role}_predictions.npz"), "source", src, "prediction_percent")
        ours = {s: load(runs[(part, s)]) for s in seeds[part]}
        xgb = {s: load(ep.xgb_dir(part, s)) for s in seeds[part]}
        preds = {"ours": {s: ours[s] for s in SEEDS}, "xgb": {s: xgb[s] for s in SEEDS},
                 "p4": {0: ep.aligned(np.load(c.OUT / f"p{part}/p4_predictions.npz"), f"{role}_source", src, f"{role}_p4_percent")},
                 "lookup_hsa": {0: c.lookup_hsa(fr)}}
        if part == c.D11_SEED:
            preds.update(ours_5seed=ours, xgb_5seed=xgb)
        boot = c.bootstrap(fr, preds)
        row = {}
        for label in SETS:
            b = boot[label]
            for m in ("lookup_hsa", "p4"):   # identical points as the eval paper
                assert abs(b["point"][m] - base[str(part)][role][m][label]) < 1e-9, (part, m, label)
            comps = {f"ours_vs_{m}": c.compare(b, "ours", m) for m in FAMILIES}
            if part == c.D11_SEED:
                comps["ours_5seed_vs_xgb_5seed"] = c.compare(b, "ours_5seed", "xgb_5seed")
            if ev is not None:
                assert abs(b["point"]["xgb"] - ev[str(part)]["test"][label]["point"]["xgb"]) < 1e-9, (part, label)
                if part == c.D11_SEED:   # D11 5-seed Ours vs XGBoost must reproduce score_cold.py
                    want, got = d11[label]["comparisons"]["v2_vs_xgb"], comps["ours_5seed_vs_xgb_5seed"]
                    assert abs(got["relative_change"] - want["relative_change"]) < 1e-12 and np.allclose(got["ci95"], want["ci95"], atol=1e-12, rtol=0), (got, want)
            row[label] = dict(pairs=b["pairs"], points=b["points"], point=b["point"], per_seed=b["per_seed"], comparisons=comps)
        res["partitions"][part] = row
        print(part, json.dumps({m: round(v, 4) for m, v in row["all"]["point"].items()}), flush=True)
        del fr, preds, boot
    res["decision"] = {}
    for label in SETS:
        pattern = {}
        for m in FAMILIES:
            tests = {p: res["partitions"][p][label]["comparisons"][f"ours_vs_{m}"] for p in c.PARTS}
            adj = c.holm({p: v["p"] for p, v in tests.items()})
            for p, v in tests.items():
                v["p_holm"] = adj[p]
                v["verdict"] = "better" if adj[p] < 0.05 and v["relative_change"] < 0 else "worse" if adj[p] < 0.05 else "no_difference"
            pattern[m] = {k: sum(v["verdict"] == k for v in tests.values()) for k in ("better", "no_difference", "worse")}
        not_better = sum(res["partitions"][p][label]["comparisons"]["ours_vs_xgb"]["ci95"][1] >= 0 for p in c.PARTS)
        res["decision"][label] = dict(
            primary=label == "all", pattern=pattern, xgb_not_better_partitions=int(not_better), wording_rule_met=bool(not_better >= 4),
            sentence_xgb=FIXED if not_better >= 4 else f"Ours 相对 XGBoost：{pattern_text(pattern['xgb'])}（Holm）",
            sentence_hsa=f"Ours 相对查表 HSA：{pattern_text(pattern['lookup_hsa'])}（Holm）")
    stem, mode, out = (("D16_dryrun_cold_val", "w", a.ours_root or c.OUT) if a.dry_run else ("D16_ours_test", "x", c.OUT))
    with (out / f"{stem}_evaluation.json").open(mode, encoding="utf-8") as h:
        json.dump(res, h, indent=1, ensure_ascii=False, default=str)
    with (out / f"{stem}_table.md").open(mode, encoding="utf-8") as h:
        h.write(markdown(res, role))
    print(markdown(res, role))
    print("written", out / f"{stem}_evaluation.json", out / f"{stem}_table.md", flush=True)


if __name__ == "__main__":
    main()
