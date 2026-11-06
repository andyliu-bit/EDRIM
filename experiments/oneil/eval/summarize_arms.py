"""O'Neil 消融两臂的验证集汇总（docs/ONEIL_ABLATION_CARD_20261104.md）。

主指标：验证集联用格 pooled RMSE（O'Neil 的 4x4 网格每格都是联用），与同种子 Ours 配对，
报相对变化的均值和配对 t 的 95% 区间（口径同 docs/BISURFGNN_V2_ABLATION_RESULTS.md）。
次要：surface-macro，只作描述。test 不读。

基准是 preds/ours（60 轮，卡上事先登记的设定），与 NCI 消融同一套训练协议；
论文 O'Neil 的主模型是 preds/ours_e500，两者在验证集上差约 1.0%。

  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe experiments/oneil/eval/summarize_arms.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

ARMS = ("random", "noanchor")
# NCI-ALMANAC 上同名臂的结果，只作对照打印（论文表 5）
NCI = {"random": (-1.41, -3.85, 1.03), "noanchor": (-4.15, -5.11, -3.18)}


def load_prediction(path, points):
    frame = pd.read_parquet(path)
    merged = points.merge(frame[common.KEY + ["prediction"]], on=common.KEY, how="left", validate="one_to_one")
    if merged.prediction.isna().any() or len(merged) != len(points):
        raise RuntimeError(f"预测没有正好覆盖验证集的点：{path}")
    return merged.prediction.to_numpy(float)


def scores(points, path):
    pred = load_prediction(path, points)
    return (common.pooled_rmse_percent(points.response.to_numpy(float), pred),
            common.surface_macro_percent(points, pred))


def paired_interval(changes):
    """相对变化（%）的均值与配对 t 的 95% 区间；正数表示该臂更差。"""
    a = np.asarray(changes, float)
    mean = a.mean()
    if len(a) < 2:
        return mean, float("nan"), float("nan")
    half = stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a))
    return mean, mean - half, mean + half


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    points = common.load_points("validation")
    base = {s: scores(points, common.OUT / f"preds/ours/seed_{s}/validation_predictions.parquet")
            for s in common.SEEDS}

    print(f"Ours 基准（验证集，{len(points)} 个点）")
    print("| 种子 | pooled RMSE | surface macro |")
    print("|---|---:|---:|")
    for s in common.SEEDS:
        print(f"| {s} | {base[s][0]:.3f} | {base[s][1]:.3f} |")
    pooled_base = [base[s][0] for s in common.SEEDS]
    print(f"均值 {np.mean(pooled_base):.3f}，SD {np.std(pooled_base, ddof=1):.3f}\n")

    for arm in ARMS:
        root = common.ANALYSIS / f"ablation_{arm}/preds/ours"
        got = {s: root / f"seed_{s}/validation_predictions.parquet" for s in common.SEEDS}
        have = [s for s in common.SEEDS if got[s].exists()]
        if not have:
            print(f"## {arm}：还没有结果\n")
            continue
        rows, changes = [], []
        for s in have:
            pooled, macro = scores(points, got[s])
            change = 100 * (pooled - base[s][0]) / base[s][0]
            rows.append((s, pooled, macro, change))
            changes.append(change)
        mean, lo, hi = paired_interval(changes)
        print(f"## {arm}（{len(have)}/{len(common.SEEDS)} 个种子）")
        print("| 种子 | pooled RMSE | surface macro | 相对 Ours |")
        print("|---|---:|---:|---:|")
        for s, pooled, macro, change in rows:
            print(f"| {s} | {pooled:.3f} | {macro:.3f} | {change:+.2f}% |")
        if len(have) == len(common.SEEDS):
            print(f"\n均值 {mean:+.2f}%，95% 区间 [{lo:+.2f}, {hi:+.2f}]，"
                  f"变差的种子 {sum(c > 0 for c in changes)}/{len(changes)}")
            n, (m, l, h) = arm, NCI[arm]
            print(f"NCI-ALMANAC 上同名臂：{m:+.2f}% [{l:+.2f}, {h:+.2f}]")
            same = (mean > 0) == (m > 0)
            print(f"方向{'一致' if same else '相反'}；区间{'不含' if lo > 0 or hi < 0 else '含'} 0")
        else:
            print(f"\n（未跑满 5 个种子，暂不报区间；当前均值 {mean:+.2f}%）")
        print()


if __name__ == "__main__":
    main()
