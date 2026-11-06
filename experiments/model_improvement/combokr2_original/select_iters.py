"""Select CGKronRLS early-stopping iterations from the CV folds (run INSIDE the container).

Criterion (fixed before looking at test data): mean over CV folds of the comboKR 2.0 Pearson on the
4x4 raw response matrices (NoTZ clipped to [0,100]) -- the ground truth that reproduces the paper's
neutral-baseline numbers. Ties -> fewer iterations (stronger early-stopping regularisation).
Usage: python -I select_iters.py SCENARIO SEED   -> prints the selected value, writes /out/cv/<..>_selection.json
"""
import glob
import json
import sys

import numpy as np

scen, seed = sys.argv[1], int(sys.argv[2])
runs = sorted(glob.glob(f"/out/cv/{scen}_r{seed}_f*/posthoc_metrics.json"))
table = {}
for r in runs:
    d = json.load(open(r))
    for k, v in d["per_iters"].items():
        for t in ["raw4x4_notz_clip0_100", "raw3x3_notz_clip0_100", "author_braid_fit_4x4_clip0_100"]:
            table.setdefault(t, {}).setdefault(int(k), []).append(v[t]["pearson_combokr2"])
crit = table["raw4x4_notz_clip0_100"]
means = {k: float(np.mean(v)) for k, v in crit.items()}
best = sorted(means.items(), key=lambda kv: (-round(kv[1], 6), kv[0]))[0][0]
out = {"scenario": scen, "seed": seed, "n_folds": len(runs), "criterion": "raw4x4_notz_clip0_100 Pearson",
       "selected_n_iters": best,
       "fold_means": {t: {k: float(np.mean(v)) for k, v in tab.items()} for t, tab in table.items()},
       "fold_values": {t: {k: v for k, v in tab.items()} for t, tab in table.items()}}
json.dump(out, open(f"/out/cv/{scen}_r{seed}_selection.json", "w"), indent=1)
print(json.dumps(out["fold_means"]), file=sys.stderr)
print(best)
