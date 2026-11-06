"""Third calibration pass: what 'ground truth' reproduces the paper's neutral-baseline Pearson?

Uses arrays written by build_measurements.py / calibrate_endpoint_v2.py (run INSIDE the container).
"""
import json

import numpy as np

Z = "/mnt/zenodo"
I = "/out/inputs"
v2 = np.load(f"{I}/almanac_measurements_v2.npz")
neu = np.load(f"{I}/neutral_pred_idx123.npy")
fit = np.load(f"{I}/author_braid_fit_idx123.npy")
design = np.load(f"{I}/almanac_design.npy")
y_nz = np.clip(v2["y_nzcalc"], 0, 100)
y_nz_raw = v2["y_nzcalc"]
PAPER = {"newcell": 0.897, "newcombo": 0.902, "newdrug": 0.904}
res = {}


def pearson(Y, P, mask_fn=None):
    out = {}
    for scen in ["newcell", "newcombo", "newdrug"]:
        vals = []
        for r in range(1, 6):
            te = np.load(f"{Z}/almanac_cv_splits/almanac_{scen}_rseed{r}_tst_cdd_inds.npy")
            m = ~np.isnan(Y[te]).any(1) & ~np.isnan(P[te]).any(1)
            if mask_fn is not None:
                m &= mask_fn(te)
            vals.append(float(np.corrcoef(Y[te][m].ravel(), P[te][m].ravel())[0, 1]))
        out[scen] = [round(float(np.mean(vals)), 4), round(float(np.std(vals, ddof=1)), 4)]
    return out


cases = {
    "y=NoTZclip, all": (y_nz, neu, None),
    "y=NoTZclip, 3x3 designs only": (y_nz, neu, lambda te: design[te] == 9),
    "y=NoTZclip, exclude NaN-BRAID rows": (y_nz, neu, lambda te: ~np.isnan(fit[te]).any(1)),
    "y=author BRAID fit at measured conc": (fit, neu, None),
    "y=clip(author BRAID fit,0,100)": (np.clip(fit, 0, 100), neu, None),
    "y=NoTZ raw, clip pred to [0,100]": (y_nz_raw, np.clip(neu, 0, 100), None),
    "y=NoTZclip, pred clipped [0,100]": (y_nz, np.clip(neu, 0, 100), None),
}
for name, (Y, P, mf) in cases.items():
    res[name] = pearson(Y, P, mf)
    print(f"{name:45s}", res[name], " paper:", PAPER)
# sanity: fit vs data
res["fit_vs_y_pearson_all"] = float(np.corrcoef(fit[~np.isnan(fit).any(1)].ravel(),
                                                y_nz[~np.isnan(fit).any(1)].ravel())[0, 1])
with open(f"{I}/endpoint_calibration_v3.json", "w") as f:
    json.dump(res, f, indent=1)
print(res["fit_vs_y_pearson_all"])
