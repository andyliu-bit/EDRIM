"""Self-check of summarize.py on fake metrics (no training, no data): pick tie-break, 5-seed stats, gap and wording.

  .venv/Scripts/python.exe -I -B experiments/model_improvement/xgb_grid/test_summarize.py <empty-or-new dir>
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize as su  # noqa: E402

root = Path(sys.argv[1])


def put(config, seed, surface, trees, pooled):
    d = root / config / f"seed_{seed}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "metrics.json").write_text(json.dumps({
        "grid_values": su.rg.GRID[config], "best_trees": trees, "at_cap": trees == 20000, "smoke": False,
        "validation_surface_macro_rmse": surface, "validation_pooled_rmse_percent": pooled, "wall_seconds": 1.0}), encoding="utf-8")


for i, c in enumerate(su.rg.GRID):
    put(c, 41, 0.08 + 0.001 * i, 5000, 10.0 + i)
put("d8_e0.1_m10", 41, 0.07, 900, 9.30)   # tie on surface-macro ...
put("d6_e0.1_m1", 41, 0.07, 800, 9.30)    # ... fewer trees wins
put("d12_e0.1_m1", 41, 0.07, 800, 9.30)   # ... same trees: config id decides ("d12..." < "d6...")
for s, p in zip((42, 43, 44, 45), (9.2, 9.4, 9.3, 9.25)):
    put("d12_e0.1_m1", s, 0.07, 800, p)
assert su.pick(root)[0] == "d12_e0.1_m1", su.pick(root)[0]
r = su.summarize(root)
assert abs(r["val_pooled_rmse_percent"]["mean"] - 9.29) < 1e-9
assert abs(r["relative_gap_ours_lower"] - (9.29 - 8.793) / 9.29) < 1e-9  # 5.35% >= 5% -> first wording
assert r["wording"].startswith("在验证集上调参后") and r["limitation"] is None, r["wording"]
assert su.wording(0.03)[1] and su.wording(-0.01)[0].startswith("调参后验证集差距缩小到 -1.0%")
assert su.wording(0.04996)[0].startswith("调参后验证集差距缩小到 4.99%") and su.wording(0.0494)[0].startswith("调参后验证集差距缩小到 4.9%")
try:
    su.summarize(root)
    raise AssertionError("summary overwritten")
except FileExistsError:
    pass
print("test_summarize OK")
