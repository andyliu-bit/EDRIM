"""Fourth calibration pass: 4x4 response matrices with study-level monotherapy matching (run INSIDE container).

v2 matched monotherapy by plate and found both drugs' monotherapy for only 53% of triplets, so its
'with monotherapy' Pearson was computed on a biased subset. Here monotherapy is matched by
(SCREENER, STUDY, CELLNAME, NSC, CONC) and averaged.
"""
import json
import pickle
import sys
import warnings
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, "/mnt/scripts")
sys.path.insert(0, "/mnt/code")
from comboKR2.braid_surface_model_for_fit import braid_model_with_raw_c_input  # noqa: E402,F401
from profile_almanac_raw import CTRL  # noqa: E402

warnings.filterwarnings("ignore")
Z = "/mnt/zenodo"
I = "/out/inputs"
ALIAS = {"753082": "761431"}
cols = ["SCREENER", "STUDY", "NSC1", "CONC1", "NSC2", "TESTVALUE", "CONTROLVALUE", "CELLNAME"]
with zipfile.ZipFile("/mnt/almanac/ComboDrugGrowth_Nov2017.zip").open("ComboDrugGrowth_Nov2017.csv") as f:
    raw = pd.read_csv(f, usecols=cols, dtype={"SCREENER": str, "STUDY": str, "CELLNAME": str})
raw["CELLNAME"] = raw["CELLNAME"].map(lambda x: CTRL.sub("", x).strip().replace("/", "_"))
raw["NSC1"] = raw["NSC1"].astype(int).astype(str).replace(ALIAS)
mono = raw[raw["NSC2"].isna()].copy()
mono["NZ"] = 100.0 * mono["TESTVALUE"] / mono["CONTROLVALUE"]
mono["CK"] = mono["CONC1"].map(lambda c: float(f"{c:.6e}"))
mstudy = mono.groupby(["CELLNAME", "NSC1", "CK", "STUDY"])["NZ"].mean().to_dict()
mall = mono.groupby(["CELLNAME", "NSC1", "CK"])["NZ"].median().to_dict()
combo = raw[raw["NSC2"].notna()].copy()
combo["NSC2"] = combo["NSC2"].astype(int).astype(str).replace(ALIAS)
study_of = combo.groupby(["CELLNAME", "NSC1", "NSC2"])["STUDY"].first().to_dict()

cdd = np.load(f"{Z}/almanac_cv_splits/almanac_cell_drug_drug_all.npy")
v2 = np.load(f"{I}/almanac_measurements_v2.npz")
c1, c2, y9 = v2["c1"], v2["c2"], np.clip(v2["y_nzcalc"], 0, 100)
neu9 = np.load(f"{I}/neutral_pred_idx123.npy")
n = len(cdd)
with open(f"{Z}/almanac_fitted_braids_and_hills/almanac_fitted_hills.pkl", "rb") as f:
    hills = pickle.load(f)
hk = {(c, str(d)): (e, h, k) for c, d, e, h, k in zip(hills.cell, hills.drug, hills.Emax, hills.h, hills.C)}


def hill(c, e, h, k):
    return 100 + (e - 100) * c ** h / (k ** h + c ** h)


res = {}
for mode in ["same_study", "any_study_median"]:
    M = np.full((n, 6), np.nan)
    P = np.full((n, 6), np.nan)
    for i, (cell, a, b) in enumerate(cdd.tolist()):
        st = study_of[(cell, a, b)]
        for j, (nsc, cc) in enumerate([(a, x) for x in np.unique(c1[i])] + [(b, x) for x in np.unique(c2[i])]):
            ck = float(f"{cc:.6e}")
            v = mstudy.get((cell, nsc, ck, st)) if mode == "same_study" else mall.get((cell, nsc, ck))
            if v is not None and j < 6:
                M[i, j] = v
                e, h, k = hk[(cell, nsc)]
                P[i, j] = hill(cc, e, h, k)
    found = ~np.isnan(M).any(1)
    res[f"{mode}|mono_found_frac"] = float(found.mean())
    Y15 = np.hstack([y9, np.clip(M, 0, 100)])
    P15 = np.hstack([neu9, P])
    Y16 = np.hstack([Y15, np.full((n, 1), 100.0)])
    P16 = np.hstack([P15, np.full((n, 1), 100.0)])
    for lab, Y, PP in [("9pt", y9, neu9), ("15pt", Y15, P15), ("16pt", Y16, P16)]:
        out = {}
        for scen in ["newcell", "newcombo", "newdrug"]:
            vals = []
            for r in range(1, 6):
                te = np.load(f"{Z}/almanac_cv_splits/almanac_{scen}_rseed{r}_tst_cdd_inds.npy")
                m = found[te] & ~np.isnan(Y[te]).any(1) & ~np.isnan(PP[te]).any(1)
                vals.append(float(np.corrcoef(Y[te][m].ravel(), PP[te][m].ravel())[0, 1]))
            out[scen] = [round(float(np.mean(vals)), 4), round(float(np.std(vals, ddof=1)), 4)]
        res[f"{mode}|{lab}|neutral_pearson(on mono-found subset)"] = out
        print(mode, lab, out, "found", round(float(found.mean()), 4), flush=True)
with open(f"{I}/endpoint_calibration_v4.json", "w") as f:
    json.dump(res, f, indent=1)
print("paper: newcell 0.897, newcombo 0.902, newdrug 0.904")
