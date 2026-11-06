"""D20 summary (docs/XGB_GRID_CARD_20261026.md). Reads only run_grid.py's metrics.json files (validation numbers).

  summarize.py --list    the 16 grid config ids, one per line (queue order)
  summarize.py --pick    the best seed-41 config: lowest val surface-macro, then fewer trees, then config id
  summarize.py           writes summary.json + summary.md (refuses to overwrite)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_grid as rg  # noqa: E402

import numpy as np  # noqa: E402

OURS_VAL = (8.667, 8.844, 8.755, 8.853, 8.848)  # Ours validation pooled combination RMSE %, seeds 41-45
OURS_MEAN = 8.793
THRESHOLD = 0.05  # card: gap >= 5% vs < 5% decides the pre-fixed wording


def load(root, config, seed):
    path = root / config / f"seed_{seed}" / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def pick(root):
    rows = {c: load(root, c, 41) for c in rg.GRID}
    missing = [c for c, r in rows.items() if r is None]
    if missing:
        raise SystemExit(f"seed-41 runs missing: {missing}")
    best = min(rows, key=lambda c: (rows[c]["validation_surface_macro_rmse"], rows[c]["best_trees"], c))
    return best, rows


def wording(gap):
    x = f"{100 * gap:.1f}%"
    if gap < THRESHOLD and x == f"{100 * THRESHOLD:.1f}%":  # 4.95%-5% would print as "5.0%" in the < 5% sentence
        x = f"{np.floor(10000 * gap) / 100:.2f}%"
    if gap >= THRESHOLD:
        return f"在验证集上调参后，Ours 仍比 XGBoost 加身份码低 {x}。", None
    note = "（负值表示调参后的 XGBoost 加身份码在验证集上低于 Ours。）" if gap < 0 else ""
    return f"调参后验证集差距缩小到 {x}。{note}", "局限：test 上报告的 XGBoost 加身份码（D8）未经调参；调参后的 XGBoost 未上 test（NCI 公开 test 已在 T 批次后封存）。"


def summarize(root):
    best, rows = pick(root)
    seeds = {s: load(root, best, s) for s in rg.fin.SEEDS}
    missing = [s for s, r in seeds.items() if r is None]
    if missing:
        raise SystemExit(f"{best}: seeds missing {missing}")
    pooled = np.array([seeds[s]["validation_pooled_rmse_percent"] for s in rg.fin.SEEDS])
    mean, sd = float(pooled.mean()), float(pooled.std(ddof=1))
    gap = (mean - OURS_MEAN) / mean  # how much lower Ours is, relative to XGBoost (the paper's relative-change convention)
    d8 = np.array([s["validation_pooled_rmse_percent"] for s in json.loads(rg.D8_SUMMARY.read_text(encoding="utf-8"))["seeds"]])
    sentence, limitation = wording(gap)
    result = {"task": "D20", "card": "docs/XGB_GRID_CARD_20261026.md", "split": "validation only (split 2); split 3 not read",
              "selection_rule": "seed 41, min val surface-macro RMSE, tie: fewer trees, then config id",
              "grid": {c: {k: r[k] for k in ("grid_values", "best_trees", "at_cap", "validation_surface_macro_rmse",
                                             "validation_pooled_rmse_percent", "wall_seconds")} for c, r in rows.items()},
              "chosen": best, "chosen_values": rg.GRID[best],
              "chosen_seeds": {s: {k: r[k] for k in ("best_trees", "at_cap", "validation_surface_macro_rmse",
                                                      "validation_pooled_rmse_percent")} for s, r in seeds.items()},
              "val_pooled_rmse_percent": {"mean": mean, "sd": sd},
              "ours_val_pooled_rmse_percent": {"seeds": OURS_VAL, "mean": OURS_MEAN},
              "d8_untuned_val_pooled_rmse_percent": {"mean": float(d8.mean()), "sd": float(d8.std(ddof=1))},
              "relative_gap_ours_lower": gap, "relative_gap_ours_lower_d8_untuned": float((d8.mean() - OURS_MEAN) / d8.mean()),
              "wording": sentence, "limitation": limitation,
              "any_smoke": any(r["smoke"] for r in [*rows.values(), *seeds.values()])}
    lines = ["# D20 XGBoost 加身份码验证集调参结果", "",
             "启动卡：docs/XGB_GRID_CARD_20261026.md。只用验证集：split 1 训练，split 2 选参与报告，split 3 未读取。"
             "选参也用 val，数值对 XGBoost 偏乐观，对 Ours 的结论是保守的。", "",
             "## 16 组网格（种子 41）", "",
             "| 配置 | max_depth | eta | min_child_weight | 选中树数 | 到上限 | val surface-macro | val 合并 RMSE % |",
             "|---|---|---|---|---|---|---|---|"]
    for c, r in sorted(rows.items(), key=lambda kv: (kv[1]["validation_surface_macro_rmse"], kv[1]["best_trees"], kv[0])):
        v = r["grid_values"]
        lines.append(f"| {'**' + c + '**' if c == best else c} | {v['max_depth']} | {v['eta']} | {v['min_child_weight']} | "
                     f"{r['best_trees']} | {'是' if r['at_cap'] else '否'} | {r['validation_surface_macro_rmse']:.6f} | "
                     f"{r['validation_pooled_rmse_percent']:.3f} |")
    lines += ["", f"## 选中配置：{best}（{rg.GRID[best]}）", "",
              "| 种子 | 选中树数 | 到上限 | val surface-macro | val 合并 RMSE % |", "|---|---|---|---|---|"]
    lines += [f"| {s} | {r['best_trees']} | {'是' if r['at_cap'] else '否'} | {r['validation_surface_macro_rmse']:.6f} | "
              f"{r['validation_pooled_rmse_percent']:.3f} |" for s, r in seeds.items()]
    lines += ["", "## 与 Ours 比较（val 联用格合并 RMSE %）", "",
              f"- 调参后 XGBoost 加身份码：5 种子均值 {mean:.3f}，SD {sd:.3f}",
              f"- Ours：{' / '.join(f'{v:.3f}' for v in OURS_VAL)}，均值 {OURS_MEAN:.3f}",
              f"- 未调参 D8（参照）：均值 {d8.mean():.3f}，SD {d8.std(ddof=1):.3f}",
              f"- 相对差（Ours 比 XGBoost 低多少，(XGB − Ours) / XGB）：调参后 {100 * gap:.1f}%，"
              f"未调参 D8 为 {100 * result['relative_gap_ours_lower_d8_untuned']:.1f}%",
              "", "## 事先固定的写法（按 5% 门槛自动选取）", "", f"> {sentence}"]
    if limitation:
        lines += ["", f"> {limitation}"]
    lines += ["", "无论结果如何都报告，不追加网格。"]
    if result["any_smoke"]:
        lines += ["", "**注意：含冒烟运行，不是正式结果。**"]
    with (root / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=1, ensure_ascii=False)
    with (root / "summary.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=rg.OUT)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--list", action="store_true")
    g.add_argument("--pick", action="store_true")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if a.list:
        print("\n".join(rg.GRID))
    elif a.pick:
        print(pick(a.root)[0])
    else:
        r = summarize(a.root)
        print(f"chosen={r['chosen']} mean={r['val_pooled_rmse_percent']['mean']:.3f} gap={100 * r['relative_gap_ours_lower']:.1f}%")


if __name__ == "__main__":
    main()
