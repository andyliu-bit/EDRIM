"""Rebuild the author's NCI-ALMANAC test-measurement arrays and calibrate the endpoint.

The comboKR 2.0 Zenodo record ships the triplets, splits, fitted Hill and BRAID functions, but NOT the
ALMANAC response matrix. We rebuild it from ComboDrugGrowth_Nov2017 and select, among candidate
reconstructions, the one consistent with the author's own fitted BRAID surfaces (fit residual) and with
the paper's label-free neutral-BRAID baseline Pearson. Run INSIDE the container.

Outputs (/out/inputs/):
  almanac_conc.npz                 c1, c2 (n_triplets x 9) for the author's 311,527 (cell, d1, d2) rows
  almanac_y_<endpoint>.npy         responses (n_triplets x 9), one file per candidate endpoint
  measurement_build_audit.json     orientation / design / duplicate counts
  endpoint_calibration.json        BRAID residuals + neutral-baseline Pearson per candidate & split
"""
import glob
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "/mnt/scripts")
sys.path.insert(0, "/mnt/code")
from profile_almanac_raw import load_raw  # noqa: E402
from comboKR2.braid_surface_model_for_fit import braid_model_with_raw_c_input  # noqa: E402

Z = "/mnt/zenodo"
OUT = "/out/inputs"
os.makedirs(OUT, exist_ok=True)
ALIAS = {"753082": "761431"}  # both vemurafenib; the author's triplet list only uses 761431
t0 = time.time()

raw = load_raw()
combo = raw[raw["NSC2"].notna()].copy()
for col in ["NSC1", "NSC2"]:
    combo[col] = combo[col].astype(int).astype(str).replace(ALIAS)
combo = combo.reset_index(drop=True)
cdd = np.load(f"{Z}/almanac_cv_splits/almanac_cell_drug_drug_all.npy")
n = len(cdd)

groups = combo.groupby(["CELLNAME", "NSC1", "NSC2"]).indices
C1 = combo["CONC1"].to_numpy()
C2 = combo["CONC2"].to_numpy()
I1 = combo["CONCINDEX1"].to_numpy()
I2 = combo["CONCINDEX2"].to_numpy()
PG = combo["PERCENTGROWTH"].to_numpy()
NZ = combo["PERCENTGROWTHNOTZ"].to_numpy()
PLATE = combo["PLATE"].to_numpy()
STUDY = combo["STUDY"].to_numpy()

# subset rules for 5x3 designs (drug with 5 concentrations): keep 3 of the 5 concentration indices
SUBSETS = {"idx123": (1, 2, 3), "idx345": (3, 4, 5), "idx135": (1, 3, 5)}

c1 = {k: np.full((n, 9), np.nan) for k in SUBSETS}
c2 = {k: np.full((n, 9), np.nan) for k in SUBSETS}
ypg = {k: np.full((n, 9), np.nan) for k in SUBSETS}
ynz = {k: np.full((n, 9), np.nan) for k in SUBSETS}
orient = np.zeros(n, dtype=np.int8)       # 1 same order, -1 swapped, 0 missing
design = np.zeros(n, dtype=np.int8)       # 9 or 15 points per experiment
n_exp = np.zeros(n, dtype=np.int8)        # experiments (studies) merged into the triplet
for i, (cell, a, b) in enumerate(cdd.tolist()):
    pos = groups.get((cell, a, b))
    sw = False
    if pos is None:
        pos = groups.get((cell, b, a))
        sw = pos is not None
    if pos is None:
        continue
    orient[i] = -1 if sw else 1
    ia, ib = (I2[pos], I1[pos]) if sw else (I1[pos], I2[pos])
    ca, cb = (C2[pos], C1[pos]) if sw else (C1[pos], C2[pos])
    pg, nz = PG[pos], NZ[pos]
    studies = np.unique(STUDY[pos])
    n_exp[i] = len(studies)
    is15 = (len(np.unique(ia)) == 5) or (len(np.unique(ib)) == 5)
    design[i] = 15 if is15 else 9
    for name, keep in SUBSETS.items():
        m = np.ones(len(pos), dtype=bool)
        if len(np.unique(ia)) == 5:
            m &= np.isin(ia, keep)
        if len(np.unique(ib)) == 5:
            m &= np.isin(ib, keep)
        if m.sum() == 9:
            # one experiment, 3x3 grid: order by (index of drug a, index of drug b)
            o = np.lexsort((ib[m], ia[m]))
            c1[name][i] = ca[m][o]
            c2[name][i] = cb[m][o]
            ypg[name][i] = pg[m][o]
            ynz[name][i] = nz[m][o]
            continue
        # duplicated grid cells across experiments: average per (conc-index a, conc-index b) cell
        df = pd.DataFrame({"ia": ia[m], "ib": ib[m], "ca": ca[m], "cb": cb[m], "pg": pg[m], "nz": nz[m]})
        agg = df.groupby(["ia", "ib"], as_index=False).mean().sort_values(["ia", "ib"])
        if len(agg) != 9:
            continue  # not a 3x3 grid: leave NaN (counted in the audit)
        c1[name][i] = agg["ca"].to_numpy()
        c2[name][i] = agg["cb"].to_numpy()
        ypg[name][i] = agg["pg"].to_numpy()
        ynz[name][i] = agg["nz"].to_numpy()
print("assembled", time.time() - t0)

audit = {"n_triplets": int(n), "orientation_same": int((orient == 1).sum()),
         "orientation_swapped": int((orient == -1).sum()), "missing": int((orient == 0).sum()),
         "design_9": int((design == 9).sum()), "design_15": int((design == 15).sum()),
         "n_experiments_hist": {int(k): int(v) for k, v in zip(*np.unique(n_exp, return_counts=True))},
         "alias": ALIAS}
for name in SUBSETS:
    audit[f"complete_rows_{name}"] = int((~np.isnan(ypg[name]).any(1)).sum())
    audit[f"notz_nan_rows_{name}"] = int(np.isnan(ynz[name]).any(1).sum())

# ---------------- author BRAIDs aligned to cdd -------------------------------------------------
frames = []
for bf in sorted(glob.glob(f"{Z}/almanac_fitted_braids_and_hills/*_braid.df.pkl")):
    with open(bf, "rb") as f:
        df = pickle.load(f)
    df = df.copy()
    df["cell"] = os.path.basename(bf)[: -len("_braid.df.pkl")]
    frames.append(df)
br = pd.concat(frames, ignore_index=True)
br["NSC1"] = br["NSC1"].astype(str)
br["NSC2"] = br["NSC2"].astype(str)
bkey = {(c, a, b): i for i, (c, a, b) in enumerate(zip(br["cell"], br["NSC1"], br["NSC2"]))}
PCOLS = ["E0", "E1", "E2", "E3", "h1", "h2", "C1", "C2", "kappa"]
bpar = br[PCOLS].to_numpy()
order = np.array([bkey.get(tuple(r), -1) for r in cdd.tolist()])
audit["braid_rows_matched"] = int((order >= 0).sum())
audit["braid_rows_in_cdd_order"] = bool(np.all(order == np.arange(n)))
braids = bpar[order]
audit["braid_nan_rows"] = int(np.isnan(braids).any(1).sum())

# ---------------- Hills -> neutral BRAID (author's predict_with_neutral_braid arithmetic) ------
with open(f"{Z}/almanac_fitted_braids_and_hills/almanac_fitted_hills.pkl", "rb") as f:
    hills = pickle.load(f)
hk = {(c, str(d)): (e, h, cc) for c, d, e, h, cc in zip(hills["cell"], hills["drug"], hills["Emax"],
                                                        hills["h"], hills["C"])}
missing_hill = np.array([((c, a) not in hk) or ((c, b) not in hk) for c, a, b in cdd.tolist()])
audit["triplets_missing_hill"] = int(missing_hill.sum())


def neutral(i, cc1, cc2):
    cell, a, b = cdd[i]
    e1, h1, k1 = hk[(cell, a)]
    e2, h2, k2 = hk[(cell, b)]
    p = [100, e1, e2, np.minimum(e1, e2), h1, h2, k1, k2, 0]
    return braid_model_with_raw_c_input([cc1, cc2], *p)


import warnings  # noqa: E402
warnings.filterwarnings("ignore")
res = {"audit": audit, "residual": {}, "neutral_pearson": {}}
endpoints = {
    "pg_raw": lambda y: y,
    "pg_clip0_100": lambda y: np.clip(y, 0, 100),
    "pg_clip0": lambda y: np.clip(y, 0, None),
    "notz_raw": None,
    "notz_clip0_100": None,
}
for sname in SUBSETS:
    ok = ~np.isnan(ypg[sname]).any(1) & ~np.isnan(braids).any(1)
    fit = np.full((n, 9), np.nan)
    neu = np.full((n, 9), np.nan)
    for i in np.where(~np.isnan(ypg[sname]).any(1))[0]:
        if ok[i]:
            fit[i] = braid_model_with_raw_c_input([c1[sname][i], c2[sname][i]], *braids[i])
        if not missing_hill[i]:
            neu[i] = neutral(i, c1[sname][i], c2[sname][i])
    for ename in endpoints:
        if ename.startswith("notz"):
            y = ynz[sname] if ename == "notz_raw" else np.clip(ynz[sname], 0, 100)
        else:
            y = endpoints[ename](ypg[sname])
        good = ok & ~np.isnan(y).any(1) & ~np.isnan(fit).any(1)
        r = np.sqrt(np.nanmean((fit[good] - y[good]) ** 2, axis=1))
        d9 = good & (design == 9)
        d15 = good & (design == 15)
        r9 = np.sqrt(np.mean((fit[d9] - y[d9]) ** 2, axis=1))
        r15 = np.sqrt(np.mean((fit[d15] - y[d15]) ** 2, axis=1)) if d15.any() else np.array([np.nan])
        res["residual"][f"{sname}|{ename}"] = {
            "n": int(good.sum()), "median_surface_rmse": float(np.median(r)), "mean_surface_rmse": float(np.mean(r)),
            "median_surface_rmse_3x3": float(np.median(r9)), "median_surface_rmse_5x3": float(np.median(r15)),
            "pearson_fit_vs_y": float(np.corrcoef(fit[good].ravel(), y[good].ravel())[0, 1])}
        # neutral-baseline Pearson on the final test splits
        for scen in ["newcell", "newcombo", "newdrug"]:
            vals = []
            for r_ in range(1, 6):
                te = np.load(f"{Z}/almanac_cv_splits/almanac_{scen}_rseed{r_}_tst_cdd_inds.npy")
                m = ~np.isnan(neu[te]).any(1) & ~np.isnan(y[te]).any(1)
                vals.append(float(np.corrcoef(y[te][m].ravel(), neu[te][m].ravel())[0, 1]))
            res["neutral_pearson"][f"{sname}|{ename}|{scen}"] = {
                "per_seed": vals, "mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1))}
    if sname == "idx123":
        np.save(f"{OUT}/neutral_pred_idx123.npy", neu)
        np.save(f"{OUT}/author_braid_fit_idx123.npy", fit)
    print("subset", sname, "done", time.time() - t0)

for sname in SUBSETS:
    np.savez_compressed(f"{OUT}/almanac_conc_{sname}.npz", c1=c1[sname], c2=c2[sname])
    np.save(f"{OUT}/almanac_y_pg_{sname}.npy", ypg[sname])
    np.save(f"{OUT}/almanac_y_notz_{sname}.npy", ynz[sname])
np.save(f"{OUT}/almanac_design.npy", design)
np.save(f"{OUT}/almanac_orientation.npy", orient)
with open(f"{OUT}/endpoint_calibration.json", "w") as f:
    json.dump(res, f, indent=1)
print(json.dumps(audit, indent=1))
print("PAPER neutral Pearson: newcombo 0.902+-0.000, newdrug 0.904+-0.003, newcell 0.897+-0.003")
for k, v in res["residual"].items():
    print("RESID", k, {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()})
for k, v in res["neutral_pearson"].items():
    print("NEUTRAL", k, round(v["mean"], 4), "+-", round(v["sd"], 4), [round(x, 4) for x in v["per_seed"]])
print("total", time.time() - t0)
