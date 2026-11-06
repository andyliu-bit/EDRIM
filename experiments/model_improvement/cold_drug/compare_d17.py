"""D17: arm A (run_cold_v2.py --ablation B11) vs Ours on the D11 cold_val, same seeds 41-43, pooled RMSE.
Relative change = 100 * (A - Ours) / Ours per seed (positive = removing the anchor is worse); paired t 95% interval.
Reads only the two cold_val metrics.json files per seed; cold_test is never touched.

  .venv/Scripts/python.exe -I -B experiments/model_improvement/cold_drug/compare_d17.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

D = Path(__file__).resolve().parents[3] / "runs/model_improvement/bisurfgnn_v2_dev/cold_drug"
SEEDS = (41, 42, 43)
KEY = "cold_val_pooled_rmse_percent"


def reading(mean: float, low: float) -> str:
    """The D17 pre-declared reading (DEVIATIONS.md D17)."""
    if mean >= 1.0 and low > 0:
        return "锚定结构在新药上带来稳健性，精度与稳健性之间有取舍"
    if mean <= 0:
        return "锚定结构未带来新药稳健性"
    return "方向一致但未确定"


assert reading(1.0, 0.01).startswith("锚定结构在新药上") and reading(0.0, -1).endswith("未带来新药稳健性")
assert reading(0.5, 0.1) == reading(2.0, -0.1) == "方向一致但未确定"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    rows = []
    for s in SEEDS:
        a_dir, ours = D / f"ablation_B11_seed_{s}", json.loads((D / f"v2_both_seed_{s}/metrics.json").read_text(encoding="utf-8"))
        if not (a_dir / "metrics.json").exists():
            raise SystemExit(f"missing {a_dir / 'metrics.json'} (D17 run not finished)")
        a = json.loads((a_dir / "metrics.json").read_text(encoding="utf-8"))
        if a.get("registered") != "DEVIATIONS.md D17" or a["smoke"] or ours["smoke"] or (a_dir / "cold_test_predictions.npz").exists():
            raise SystemExit(f"{a_dir}: not a full D17 run, or a cold_test file exists")
        rows.append((s, ours[KEY], a[KEY], 100.0 * (a[KEY] - ours[KEY]) / ours[KEY], a["best_epoch"], ours["best_epoch"]))
    rel = np.array([r[3] for r in rows])
    n, mean = len(rel), float(rel.mean())
    radius = float(student_t.ppf(.975, n - 1) * rel.std(ddof=1) / np.sqrt(n))
    print("seed  Ours_cold_val  A_cold_val  rel_change_%  best_epoch_A  best_epoch_Ours")
    for s, o, a, r, ea, eo in rows:
        print(f"{s:>4}  {o:13.4f}  {a:10.4f}  {r:+12.2f}  {ea:>12}  {eo:>15}")
    print(f"mean relative change {mean:+.2f}%  paired t 95% interval [{mean - radius:+.2f}, {mean + radius:+.2f}]  "
          f"A worse in {int((rel > 0).sum())}/{n} seeds")
    print(f"D17 reading: {reading(mean, mean - radius)}  (3 seeds, descriptive only; cold_val also selected the epoch)")


if __name__ == "__main__":
    main()
