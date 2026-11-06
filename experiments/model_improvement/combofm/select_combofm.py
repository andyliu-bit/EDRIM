"""D21 (docs/COMBOFM_CARD_20261027.md): selection and summary for the NCI comboFM runs (validation only).

  --pick-reg   best reg among the rank-50, seed-41, 100-epoch runs (lowest val surface-macro; ties -> larger reg)
  --pick-best  best (reg, rank) among all seed-41, 100-epoch tuning runs, printed as "<reg> <rank>"
  (no flag)    summary.json + summary.md: tuning table, 5-seed table at the chosen setting, card L wording
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "runs/model_improvement/combofm/nci"
OURS_VAL = (8.667, 8.844, 8.755, 8.853, 8.848)  # Ours (D7) validation pooled RMSE %, seeds 41-45
SEEDS = (41, 42, 43, 44, 45)


def runs(epochs):
    for m in sorted(OUT.glob(f"reg*_rank*/seed_*_ep{epochs}/metrics.json")):
        yield json.loads(m.read_text())


def tuning():
    return [r for r in runs(100) if r["seed"] == 41]


def key(r):
    return (r["validation_surface_macro_rmse"], -r["reg"], r["rank"])


def wording(gap):
    if gap >= 5:
        return f"在验证集上，Ours 比 comboFM 低 {gap:.2f}%"
    if gap >= 0:
        return f"验证集上与 comboFM 相当（差 {gap:.2f}%）"
    return f"comboFM 在验证集上比 Ours 低 {-gap:.2f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pick-reg", action="store_true")
    ap.add_argument("--pick-best", action="store_true")
    a = ap.parse_args()
    t = tuning()
    if a.pick_reg:
        regs = [r for r in t if r["rank"] == 50]
        assert len(regs) == 4, f"expected 4 rank-50 runs, found {len(regs)}"
        print(f"{min(regs, key=key)['reg']:g}")
        return
    best = min(t, key=key)
    if a.pick_best:
        assert len(t) == 7, f"expected 7 tuning runs, found {len(t)}"
        print(f"{best['reg']:g} {best['rank']}")
        return
    final = {r["seed"]: r for r in runs(200) if r["reg"] == best["reg"] and r["rank"] == best["rank"]}
    assert sorted(final) == list(SEEDS), f"final seeds {sorted(final)}"
    pooled = [final[s]["validation_pooled_rmse_percent"] for s in SEEDS]
    mean, sd = statistics.mean(pooled), statistics.stdev(pooled)
    ours = statistics.mean(OURS_VAL)
    gap = 100 * (mean - ours) / mean
    summary = dict(task="D21", setting=dict(reg=best["reg"], rank=best["rank"], order=5),
                   tuning=[{k: r[k] for k in ("reg", "rank", "best_epoch", "validation_surface_macro_rmse",
                                               "validation_pooled_rmse_percent")} for r in sorted(t, key=key)],
                   final={s: {k: final[s][k] for k in ("best_epoch", "validation_surface_macro_rmse",
                                                       "validation_pooled_rmse_percent")} for s in SEEDS},
                   combofm_val_pooled_mean=mean, combofm_val_pooled_sd=sd, ours_val_pooled_mean=ours,
                   relative_gap_percent=gap, wording=wording(gap))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    md = ["# D21 comboFM（PyTorch 移植版）在 NCI 验证集上的结果", "",
          "启动卡：docs/COMBOFM_CARD_20261027.md。只用验证集：split 1 训练，split 2 选参与报告，split 3 未读取。", "",
          "## 调参（种子 41，最多 100 轮，每 10 轮按 val surface-macro 选轮）", "",
          "| reg | rank | 选中轮 | val surface-macro | val 合并 RMSE % |", "|---|---|---|---|---|"]
    md += [f"| {r['reg']:g} | {r['rank']} | {r['best_epoch']} | {r['validation_surface_macro_rmse']:.6f} | "
           f"{r['validation_pooled_rmse_percent']:.3f} |" for r in sorted(t, key=key)]
    md += ["", f"## 选定 reg {best['reg']:g}、rank {best['rank']}：种子 41–45，最多 200 轮", "",
           "| 种子 | 选中轮 | val surface-macro | val 合并 RMSE % |", "|---|---|---|---|"]
    md += [f"| {s} | {final[s]['best_epoch']} | {final[s]['validation_surface_macro_rmse']:.6f} | "
           f"{final[s]['validation_pooled_rmse_percent']:.3f} |" for s in SEEDS]
    md += ["", f"- comboFM 5 种子均值 {mean:.3f}，SD {sd:.3f}；Ours {ours:.3f}；相对差 (comboFM − Ours)/comboFM = {gap:.1f}%",
           "", "## 事先固定的写法（卡 L）", "", f"> {wording(gap)}。", "", "无论结果如何都报告，不追加网格。"]
    (OUT / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"chosen reg={best['reg']:g} rank={best['rank']} mean={mean:.3f} gap={gap:.1f}%")


if __name__ == "__main__":
    main()
