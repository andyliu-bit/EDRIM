"""Second endpoint-calibration pass (run INSIDE the container).

Questions: (1) can NoTZ be recomputed as 100*TESTVALUE/CONTROLVALUE (also for screener 1A, where the file
leaves PERCENTGROWTHNOTZ empty)? (2) do the author's surfaces include the on-plate monotherapy points?
(3) which monotherapy endpoint do the author's Hill fits describe?
Writes /out/inputs/endpoint_calibration_v2.json and, for the selected candidates, measurement arrays.
"""
import json
import os
import pickle
import sys
import time
import warnings
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, "/mnt/scripts")
sys.path.insert(0, "/mnt/code")
from comboKR2.braid_surface_model_for_fit import braid_model_with_raw_c_input  # noqa: E402
from profile_almanac_raw import CTRL  # noqa: E402

warnings.filterwarnings("ignore")
Z = "/mnt/zenodo"
OUT = "/out/inputs"
ALIAS = {"753082": "761431"}
t0 = time.time()
cols = ["SCREENER", "STUDY", "PLATE", "NSC1", "CONCINDEX1", "CONC1", "NSC2", "CONCINDEX2", "CONC2",
        "PERCENTGROWTH", "PERCENTGROWTHNOTZ", "TESTVALUE", "CONTROLVALUE", "TZVALUE", "CELLNAME"]
with zipfile.ZipFile("/mnt/almanac/ComboDrugGrowth_Nov2017.zip").open("ComboDrugGrowth_Nov2017.csv") as f:
    raw = pd.read_csv(f, usecols=cols, dtype={"SCREENER": str, "STUDY": str, "PLATE": str, "CELLNAME": str})
raw["CELLNAME"] = raw["CELLNAME"].map(lambda x: CTRL.sub("", x).strip().replace("/", "_"))
raw["NSC1"] = raw["NSC1"].astype(int).astype(str).replace(ALIAS)
raw["NSC2"] = raw["NSC2"].map(lambda x: str(int(x)) if pd.notna(x) else None).replace(ALIAS)
raw["NZCALC"] = 100.0 * raw["TESTVALUE"] / raw["CONTROLVALUE"]
res = {}
have = raw["PERCENTGROWTHNOTZ"].notna()
d = (raw.loc[have, "NZCALC"] - raw.loc[have, "PERCENTGROWTHNOTZ"]).abs()
res["nzcalc_vs_file"] = {"n": int(have.sum()), "max_abs_diff": float(d.max()), "median_abs_diff": float(d.median()),
                         "frac_abs_diff_gt_0.01": float((d > 0.011).mean())}
for s in ["FF", "FG", "1A"]:
    sub = raw[raw.SCREENER == s]
    res[f"screener_{s}"] = {"test_nan": int(sub.TESTVALUE.isna().sum()), "control_nan": int(sub.CONTROLVALUE.isna().sum()),
                            "tz_nan": int(sub.TZVALUE.isna().sum()), "rows": int(len(sub))}
# PG recomputation check: PG = 100*(T-TZ)/(C-TZ) if T>=TZ else 100*(T-TZ)/TZ
pgc = np.where(raw.TESTVALUE >= raw.TZVALUE, 100 * (raw.TESTVALUE - raw.TZVALUE) / (raw.CONTROLVALUE - raw.TZVALUE),
               100 * (raw.TESTVALUE - raw.TZVALUE) / raw.TZVALUE)
dd = np.abs(pgc - raw.PERCENTGROWTH)
res["pgcalc_vs_file"] = {"median_abs_diff": float(np.nanmedian(dd)), "frac_gt_0.01": float(np.nanmean(dd > 0.011))}
print(json.dumps(res, indent=1), time.time() - t0)

combo = raw[raw.NSC2.notna()].reset_index(drop=True)
mono = raw[raw.NSC2.isna()].reset_index(drop=True)
cdd = np.load(f"{Z}/almanac_cv_splits/almanac_cell_drug_drug_all.npy")
n = len(cdd)
groups = combo.groupby(["CELLNAME", "NSC1", "NSC2"]).indices
# monotherapy lookup on the same plate: (SCREENER, STUDY, PLATE, NSC, CONC) -> NoTZcalc / PG
mono_key = {}
for scr, st, pl, nsc, c, nz, pg in zip(mono.SCREENER, mono.STUDY, mono.PLATE, mono.NSC1, mono.CONC1, mono.NZCALC,
                                        mono.PERCENTGROWTH):
    mono_key.setdefault((scr, st, pl, nsc, float(f"{c:.6e}")), []).append((nz, pg))

A = {k: combo[k].to_numpy() for k in ["SCREENER", "STUDY", "PLATE", "CONCINDEX1", "CONC1", "CONCINDEX2", "CONC2",
                                       "NZCALC", "PERCENTGROWTH"]}
c1 = np.full((n, 9), np.nan); c2 = np.full((n, 9), np.nan)
ynz = np.full((n, 9), np.nan); ypg = np.full((n, 9), np.nan)
m1c = np.full((n, 3), np.nan); m1nz = np.full((n, 3), np.nan); m1pg = np.full((n, 3), np.nan)
m2c = np.full((n, 3), np.nan); m2nz = np.full((n, 3), np.nan); m2pg = np.full((n, 3), np.nan)
for i, (cell, a, b) in enumerate(cdd.tolist()):
    pos = groups[(cell, a, b)]
    ia, ib = A["CONCINDEX1"][pos], A["CONCINDEX2"][pos]
    m = np.ones(len(pos), bool)
    if len(np.unique(ia)) == 5:
        m &= np.isin(ia, (1, 2, 3))
    p = pos[m]
    df = pd.DataFrame({"ia": A["CONCINDEX1"][p], "ib": A["CONCINDEX2"][p], "ca": A["CONC1"][p], "cb": A["CONC2"][p],
                       "nz": A["NZCALC"][p], "pg": A["PERCENTGROWTH"][p]}) if len(p) != 9 else None
    if df is None:
        o = np.lexsort((A["CONCINDEX2"][p], A["CONCINDEX1"][p]))
        p = p[o]
        c1[i], c2[i] = A["CONC1"][p], A["CONC2"][p]
        ynz[i], ypg[i] = A["NZCALC"][p], A["PERCENTGROWTH"][p]
    else:
        agg = df.groupby(["ia", "ib"], as_index=False).mean().sort_values(["ia", "ib"])
        c1[i], c2[i] = agg.ca.to_numpy(), agg.cb.to_numpy()
        ynz[i], ypg[i] = agg.nz.to_numpy(), agg.pg.to_numpy()
    # monotherapy of each drug at the same concentrations, same plate(s)
    plates = set(zip(A["SCREENER"][pos], A["STUDY"][pos], A["PLATE"][pos]))
    for which, (nsc, concs, mc, mnz, mpg) in enumerate([(a, np.unique(c1[i]), m1c, m1nz, m1pg),
                                                         (b, np.unique(c2[i]), m2c, m2nz, m2pg)]):
        for j, cc in enumerate(concs[:3]):
            vals = []
            for scr, st, pl in plates:
                vals += mono_key.get((scr, st, pl, nsc, float(f"{cc:.6e}")), [])
            if vals:
                mc[i, j] = cc
                mnz[i, j] = np.mean([v[0] for v in vals])
                mpg[i, j] = np.mean([v[1] for v in vals])
print("assembled", time.time() - t0)
res["mono_found_frac"] = float(np.mean(~np.isnan(m1nz).any(1) & ~np.isnan(m2nz).any(1)))

# author Hills and BRAIDs
with open(f"{Z}/almanac_fitted_braids_and_hills/almanac_fitted_hills.pkl", "rb") as f:
    hills = pickle.load(f)
hk = {(c, str(dd_)): (e, h, cc) for c, dd_, e, h, cc in zip(hills.cell, hills.drug, hills.Emax, hills.h, hills.C)}
braids = None
import glob  # noqa: E402
frames = []
for bf in sorted(glob.glob(f"{Z}/almanac_fitted_braids_and_hills/*_braid.df.pkl")):
    with open(bf, "rb") as f:
        df = pickle.load(f)
    df = df.copy(); df["cell"] = os.path.basename(bf)[: -len("_braid.df.pkl")]
    frames.append(df)
br = pd.concat(frames, ignore_index=True)
bk = {(c, str(a), str(b)): i for i, (c, a, b) in enumerate(zip(br.cell, br.NSC1, br.NSC2))}
bpar = br[["E0", "E1", "E2", "E3", "h1", "h2", "C1", "C2", "kappa"]].to_numpy()[[bk[tuple(r)] for r in cdd.tolist()]]


def hill(c, e, h, k):
    return 100 + (e - 100) * c ** h / (k ** h + c ** h)


# Hill residual vs monotherapy endpoint (drug-1 and drug-2 monotherapy at plate concentrations)
hr = {"nz": [], "nz_clip": [], "pg": [], "pg_clip": []}
neu9 = np.full((n, 9), np.nan); fit9 = np.full((n, 9), np.nan)
neuM1 = np.full((n, 3), np.nan); neuM2 = np.full((n, 3), np.nan)
fitM1 = np.full((n, 3), np.nan); fitM2 = np.full((n, 3), np.nan)
for i, (cell, a, b) in enumerate(cdd.tolist()):
    e1, h1, k1 = hk[(cell, a)]; e2, h2, k2 = hk[(cell, b)]
    p = [100, e1, e2, min(e1, e2), h1, h2, k1, k2, 0]
    neu9[i] = braid_model_with_raw_c_input([c1[i], c2[i]], *p)
    if not np.isnan(bpar[i]).any():
        fit9[i] = braid_model_with_raw_c_input([c1[i], c2[i]], *bpar[i])
        fitM1[i] = braid_model_with_raw_c_input([np.nan_to_num(m1c[i]), np.zeros(3)], *bpar[i])
        fitM2[i] = braid_model_with_raw_c_input([np.zeros(3), np.nan_to_num(m2c[i])], *bpar[i])
    neuM1[i] = hill(m1c[i], e1, h1, k1)
    neuM2[i] = hill(m2c[i], e2, h2, k2)
print("evaluated", time.time() - t0)

ok = ~np.isnan(m1nz).any(1) & ~np.isnan(m2nz).any(1)
for nm, Y1, Y2 in [("nz", m1nz, m2nz), ("pg", m1pg, m2pg)]:
    for clip in [False, True]:
        y1 = np.clip(Y1, 0, 100) if clip else Y1
        y2 = np.clip(Y2, 0, 100) if clip else Y2
        r = np.sqrt(np.nanmean(np.hstack([(neuM1 - y1) ** 2, (neuM2 - y2) ** 2])[ok], axis=1))
        rb = np.sqrt(np.nanmean(np.hstack([(fitM1 - y1) ** 2, (fitM2 - y2) ** 2])[ok], axis=1))
        res[f"hill_resid_mono_{nm}{'_clip' if clip else ''}"] = {"median": float(np.nanmedian(r)),
                                                                  "braid_mono_median": float(np.nanmedian(rb))}


def pearson_sets(Y, P, label):
    out = {}
    for scen in ["newcell", "newcombo", "newdrug"]:
        vals = []
        for r_ in range(1, 6):
            te = np.load(f"{Z}/almanac_cv_splits/almanac_{scen}_rseed{r_}_tst_cdd_inds.npy")
            yy, pp = Y[te], P[te]
            mm = ~np.isnan(yy).any(1) & ~np.isnan(pp).any(1)
            vals.append(float(np.corrcoef(yy[mm].ravel(), pp[mm].ravel())[0, 1]))
        out[scen] = {"mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)), "per_seed": vals}
    res[f"neutral_pearson|{label}"] = out
    print("NEUTRAL", label, {k: (round(v["mean"], 4), round(v["sd"], 4)) for k, v in out.items()})


for nm, Y in [("nzcalc", ynz), ("pg", ypg)]:
    for clip in [(None, None), (0, 100), (0, None)]:
        yy = np.clip(Y, clip[0], clip[1]) if clip[0] is not None else Y
        lab = f"{nm}_{'raw' if clip[0] is None else ('clip0_100' if clip[1] else 'clip0')}"
        pearson_sets(yy, neu9, "9pt|" + lab)
        r = np.sqrt(np.nanmean((fit9 - yy) ** 2, axis=1))
        res[f"braid_resid_9pt|{lab}"] = float(np.nanmedian(r))
        # 15 points: 3x3 combos + monotherapy at the same plate concentrations
        M1 = m1nz if nm == "nzcalc" else m1pg
        M2 = m2nz if nm == "nzcalc" else m2pg
        Y15 = np.hstack([yy, np.clip(M1, clip[0], clip[1]) if clip[0] is not None else M1,
                         np.clip(M2, clip[0], clip[1]) if clip[0] is not None else M2])
        P15 = np.hstack([neu9, neuM1, neuM2])
        pearson_sets(Y15, P15, "15pt_with_mono|" + lab)
        F15 = np.hstack([fit9, fitM1, fitM2])
        res[f"braid_resid_15pt|{lab}"] = float(np.nanmedian(np.sqrt(np.nanmean((F15 - Y15) ** 2, axis=1))))
        # 16 points: + (0,0) == 100
        Y16 = np.hstack([Y15, np.full((n, 1), 100.0)])
        P16 = np.hstack([P15, np.full((n, 1), 100.0)])
        pearson_sets(Y16, P16, "16pt_with_mono_and_zero|" + lab)

np.savez_compressed(f"{OUT}/almanac_measurements_v2.npz", c1=c1, c2=c2, y_nzcalc=ynz, y_pg=ypg,
                    m1c=m1c, m1nz=m1nz, m1pg=m1pg, m2c=m2c, m2nz=m2nz, m2pg=m2pg)
with open(f"{OUT}/endpoint_calibration_v2.json", "w") as f:
    json.dump(res, f, indent=1)
for k, v in res.items():
    if not k.startswith("neutral"):
        print(k, v)
print("total", time.time() - t0)
