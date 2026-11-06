"""Direction A zero-training check: HSA anchor per arm = w*SP + (1-w)*M (M = other-plate median).

Card (decision rule written before running): docs/ANCHOR_FUSION_CHECK_20261029.md
Reuses the 10-18 same-plate analysis (parsing, unit anchors, point set P) unchanged by executing its source.
w is chosen on one random half of drug pairs and scored on the other (cross-fit); drug-pair cluster bootstrap CI.
Usage (from bisurfgnn-model-experiments):
  .venv/Scripts/python.exe -I experiments/eval_paper/anchor_fusion_check/run.py
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "same_plate_anchor_mechanism" / "run.py"
OUT = Path(__file__).resolve().parents[3] / "runs/eval_paper/anchor_fusion_check"
SPLIT_SEED = 20261029
GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
SELF_CHECK = {("notz", "med5", 1.0): 0.07294489390123736, ("notz", "med5", 0.0): 0.10283307088853119,
              ("dd", "med4", 0.0): 0.0763210696504572}  # values from runs/eval_paper/same_plate_anchor_mechanism/anchor_table.csv

src = SRC.read_text(encoding="utf-8")
head, rest = src.split("def main():", 1)
body = rest.split("    def hsa(e, ia, ib):", 1)[0]
g = {"__file__": str(SRC), "__name__": "same_plate_src"}
exec(head, g)
exec(textwrap.dedent(body), g)

m = g["sets"]["P"]
sa, sb, A, val, y, sid, n_s, pair_of_s = (g[k] for k in ("sa", "sb", "A", "val", "y", "sid", "n_s", "pair_of_s"))
ia, ib, sidm = sa[m], sb[m], sid[m]
s_here = np.unique(sidm)
pairs = pair_of_s[s_here]
upairs = np.unique(pairs)
half_p = np.zeros(len(upairs), np.int8)
half_p[np.random.default_rng(SPLIT_SEED).permutation(len(upairs))[len(upairs) // 2:]] = 1
half = half_p[np.searchsorted(upairs, pairs)]
W, cl, C = g["boot_weights"](pairs)

res = {"n_points": int(m.sum()), "n_surfaces": int(len(s_here)), "n_drug_pairs": int(C), "grid": GRID.tolist(), "variants": {}}
for e in g["ENDPOINTS"]:
    for med in ("med5", "med4"):
        R = np.empty((len(GRID), len(s_here)))
        for k, w in enumerate(GRID):
            fa = w * val[e][ia] + (1 - w) * A[f"{med}_{e}"][ia]
            fb = w * val[e][ib] + (1 - w) * A[f"{med}_{e}"][ib]
            p = np.minimum(fa, fb)
            assert np.isfinite(p).all()
            R[k] = g["surface_rmse"]((p - y[e][m]) ** 2, sidm, n_s)[0][s_here]
        curve = R.mean(1)
        for (ee, mm, w), ref in SELF_CHECK.items():
            if (ee, mm) == (e, med):
                assert abs(curve[np.flatnonzero(GRID == w)[0]] - ref) < 1e-9, (e, med, w, curve[GRID == w], ref)
        k0, k1 = (int(R[:, half == h].mean(1).argmin()) for h in (0, 1))
        cf = np.where(half == 0, R[k1], R[k0])  # each half scored with the w chosen on the other half
        sp = R[np.flatnonzero(GRID == 1.0)[0]]
        s_cf, s_sp = np.bincount(cl, weights=cf, minlength=C), np.bincount(cl, weights=sp, minlength=C)
        boot = (W @ s_cf) / (W @ s_sp) - 1.0
        res["variants"][f"{e}/{med}"] = dict(
            curve=curve.tolist(), w_best_full=float(GRID[curve.argmin()]), rmse_best_full=float(curve.min()),
            w_chosen_half0=float(GRID[k0]), w_chosen_half1=float(GRID[k1]),
            rmse_sp=float(sp.mean()), rmse_crossfit=float(cf.mean()), rel_crossfit_vs_sp=float(s_cf.sum() / s_sp.sum() - 1.0),
            ci95=[float(x) for x in np.percentile(boot, [2.5, 97.5])])
v = res["variants"]["notz/med5"]
res["decision_primary"] = ("supported" if v["rel_crossfit_vs_sp"] <= -0.01 and v["ci95"][1] < 0 else "not supported")
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
for name, v in res["variants"].items():
    print(f"{name}: SP {v['rmse_sp']:.6f} | best w (full) {v['w_best_full']} -> {v['rmse_best_full']:.6f} | "
          f"cross-fit w {v['w_chosen_half0']}/{v['w_chosen_half1']} -> {v['rmse_crossfit']:.6f} "
          f"({100 * v['rel_crossfit_vs_sp']:+.2f}% [{100 * v['ci95'][0]:+.2f}, {100 * v['ci95'][1]:+.2f}])")
print("primary decision (notz/med5):", res["decision_primary"])
