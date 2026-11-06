"""Variant A': plate offset estimated from other single-drug wells on the same plate.

Card (decision rules written before running): docs/ANCHOR_FUSION_CHECK_20261029.md, section A'.
Per arm: anchor = w*SP + (1-w)*(M + delta); M = cross-screener median excl. own plate; delta = median residual
(value - own M) of the other single-drug units on the same plate (delta_all: all but the arm's own unit;
delta_oth: only units of other drugs). Same point set, grid, cross-fit halves and bootstrap as run.py.
Usage (from bisurfgnn-model-experiments):
  .venv/Scripts/python.exe -I experiments/eval_paper/anchor_fusion_check/run_offset.py
"""
from __future__ import annotations

import json
import textwrap
import warnings
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "same_plate_anchor_mechanism" / "run.py"
OUT = Path(__file__).resolve().parents[3] / "runs/eval_paper/anchor_fusion_check"
SPLIT_SEED = 20261029
GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
SP_NOTZ, M_NOTZ = 0.07294489390123736, 0.10283307088853119  # 10-18 anchor_table.csv, P / notz

src = SRC.read_text(encoding="utf-8")
head, rest = src.split("def main():", 1)
body = rest.split("    def hsa(e, ia, ib):", 1)[0]
g = {"__file__": str(SRC), "__name__": "same_plate_src"}
exec(head, g)
exec(textwrap.dedent(body), g)

u, A, val, y, sid, n_s, pair_of_s = (g[k] for k in ("u", "A", "val", "y", "sid", "n_s", "pair_of_s"))
m = g["sets"]["P"]
ia, ib, sidm = g["sa"][m], g["sb"][m], sid[m]
s_here = np.unique(sidm)
pairs = pair_of_s[s_here]
upairs = np.unique(pairs)
half_p = np.zeros(len(upairs), np.int8)
half_p[np.random.default_rng(SPLIT_SEED).permutation(len(upairs))[len(upairs) // 2:]] = 1
half = half_p[np.searchsorted(upairs, pairs)]
W, cl, C = g["boot_weights"](pairs)

plate_u, nsc_u = u.plate.to_numpy(), u.nsc.to_numpy()
order = np.argsort(plate_u, kind="stable")
ps = plate_u[order]
starts = np.r_[0, np.flatnonzero(np.diff(ps)) + 1]
ends = np.r_[starts[1:], len(ps)]
sizes = ends - starts
n_drugs = np.array([len(np.unique(nsc_u[order[a:b]])) for a, b in zip(starts, ends)])


def offsets(r):
    d_all, d_oth = np.full(len(r), np.nan), np.full(len(r), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for a, b in zip(starts, ends):
            if b - a < 2:
                continue
            idx = order[a:b]
            v = np.broadcast_to(r[idx], (b - a, b - a))
            d_all[idx] = np.nanmedian(np.where(~np.eye(b - a, dtype=bool), v, np.nan), axis=1)
            nd = nsc_u[idx]
            d_oth[idx] = np.nanmedian(np.where(nd[:, None] != nd[None, :], v, np.nan), axis=1)
    return d_all, d_oth


def score(p, e):
    return g["surface_rmse"]((p - y[e][m]) ** 2, sidm, n_s)[0][s_here]


res = {"n_points": int(m.sum()), "n_surfaces": int(len(s_here)), "n_drug_pairs": int(C), "grid": GRID.tolist(),
       "plates": dict(n=int(len(sizes)), units_per_plate_quantiles=np.percentile(sizes, [5, 25, 50, 75, 95]).tolist(),
                      drugs_per_plate_counts={int(k): int(v) for k, v in zip(*np.unique(n_drugs, return_counts=True))}),
       "variants": {}}
for e in g["ENDPOINTS"]:
    M_u = A[f"med5_{e}"]
    d_all, d_oth = offsets(val[e] - M_u)
    for dname, d in (("delta_all", d_all), ("delta_oth", d_oth)):
        zero = np.isnan(np.r_[d[ia], d[ib]]).mean()
        d = np.nan_to_num(d)
        R = np.empty((len(GRID), len(s_here)))
        for k, w in enumerate(GRID):
            fa = w * val[e][ia] + (1 - w) * (M_u[ia] + d[ia])
            fb = w * val[e][ib] + (1 - w) * (M_u[ib] + d[ib])
            p = np.minimum(fa, fb)
            assert np.isfinite(p).all()
            R[k] = score(p, e)
        sp, r_m = R[-1], score(np.minimum(M_u[ia], M_u[ib]), e)
        if e == "notz":
            assert abs(sp.mean() - SP_NOTZ) < 1e-9 and abs(r_m.mean() - M_NOTZ) < 1e-9
        k0, k1 = (int(R[:, half == h].mean(1).argmin()) for h in (0, 1))
        cf = np.where(half == 0, R[k1], R[k0])
        s_cf, s_sp = np.bincount(cl, weights=cf, minlength=C), np.bincount(cl, weights=sp, minlength=C)
        boot = (W @ s_cf) / (W @ s_sp) - 1.0
        res["variants"][f"{e}/{dname}"] = dict(
            arm_share_delta_zero=float(zero), curve=R.mean(1).tolist(),
            w_chosen_half0=float(GRID[k0]), w_chosen_half1=float(GRID[k1]),
            rmse_sp=float(sp.mean()), rmse_m=float(r_m.mean()), rmse_m_plus_delta=float(R[0].mean()),
            recovery=float((r_m.mean() - R[0].mean()) / (r_m.mean() - sp.mean())),
            rmse_crossfit=float(cf.mean()), rel_crossfit_vs_sp=float(s_cf.sum() / s_sp.sum() - 1.0),
            ci95=[float(x) for x in np.percentile(boot, [2.5, 97.5])])
v = res["variants"]["notz/delta_all"]
res["decision_primary"] = "supported" if v["rel_crossfit_vs_sp"] <= -0.01 and v["ci95"][1] < 0 else "not supported"
res["secondary_notz"] = {k: ("usable without SP" if res["variants"][f"notz/{k}"]["recovery"] >= 0.5 else "below 50%")
                         for k in ("delta_all", "delta_oth")}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "results_offset.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print("plates:", json.dumps(res["plates"]))
for name, v in res["variants"].items():
    print(f"{name}: delta=0 share {v['arm_share_delta_zero']:.3f} | SP {v['rmse_sp']:.6f} M {v['rmse_m']:.6f} "
          f"M+delta {v['rmse_m_plus_delta']:.6f} (recovery {100 * v['recovery']:.1f}%) | cross-fit w {v['w_chosen_half0']}/"
          f"{v['w_chosen_half1']} -> {v['rmse_crossfit']:.6f} ({100 * v['rel_crossfit_vs_sp']:+.2f}% "
          f"[{100 * v['ci95'][0]:+.2f}, {100 * v['ci95'][1]:+.2f}])")
print("primary decision (notz/delta_all):", res["decision_primary"], "| secondary:", res["secondary_notz"])
