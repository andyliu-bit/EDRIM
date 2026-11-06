"""Label-free baselines on all author splits (5 seeds x 3 scenarios); run INSIDE the container.

BRAID neutral (author's predict_with_neutral_braid arithmetic from the Zenodo Hills) and HSA from measured
same-study monotherapy, scored on (a) the paper-consistent truth (4x4 matrices, NoTZ clipped [0,100], pooled
Pearson) and (b) our project's metric family (3x3 combination points, NoTZ unclipped, surface-macro RMSE).
Output: /out/baselines_all_seeds.json
"""
import json
import pickle
import sys
import warnings

import numpy as np

sys.path.insert(0, "/mnt/code")
sys.path.insert(0, "/mnt/scripts")
warnings.filterwarnings("ignore")
from comboKR2.braid_surface_model_for_fit import braid_model_with_raw_c_input  # noqa: E402
from posthoc_eval import hsa_3x3  # noqa: E402

Z = "/mnt/zenodo"
I = "/out/inputs"
cdd = np.load(f"{Z}/almanac_cv_splits/almanac_cell_drug_drug_all.npy")
m4 = np.load(f"{I}/almanac_measurements_4x4.npz")
C1, C2, Y16, cp = m4["c1"], m4["c2"], m4["y_nzcalc"], m4["combo_pos"]
with open(f"{Z}/almanac_fitted_braids_and_hills/almanac_fitted_hills.pkl", "rb") as f:
    hills = pickle.load(f)
hk = {(c, str(d)): (e, h, k) for c, d, e, h, k in zip(hills.cell, hills.drug, hills.Emax, hills.h, hills.C)}
N = np.full(Y16.shape, np.nan)
for i, (cell, a, b) in enumerate(cdd.tolist()):
    e1, h1, k1 = hk[(cell, a)]
    e2, h2, k2 = hk[(cell, b)]
    N[i] = braid_model_with_raw_c_input([C1[i], C2[i]], 100, e1, e2, min(e1, e2), h1, h2, k1, k2, 0)
H9 = hsa_3x3(Y16)
res = {}
for scen in ["newcell", "newcombo", "newdrug"]:
    rows = []
    for r in range(1, 6):
        te = np.load(f"{Z}/almanac_cv_splits/almanac_{scen}_rseed{r}_tst_cdd_inds.npy")
        y16, n16 = Y16[te], N[te]
        m = ~np.isnan(y16).any(1)
        pear = float(np.corrcoef(np.clip(y16[m], 0, 100).ravel(), n16[m].ravel())[0, 1])
        y9, n9, h9 = y16[m][:, cp], n16[m][:, cp], H9[te][m]
        rn = np.sqrt(np.mean((n9 - y9) ** 2, axis=1))
        rh = np.sqrt(np.mean((h9 - y9) ** 2, axis=1))
        rows.append({"seed": r, "n": int(m.sum()), "neutral_pearson_4x4_clip": pear,
                     "neutral_surface_macro_rmse_3x3_unclipped": float(rn.mean() / 100),
                     "hsa_surface_macro_rmse_3x3_unclipped": float(rh.mean() / 100),
                     "neutral_pearson_3x3_unclipped": float(np.corrcoef(y9.ravel(), n9.ravel())[0, 1]),
                     "hsa_pearson_3x3_unclipped": float(np.corrcoef(y9.ravel(), h9.ravel())[0, 1]),
                     "winrate_neutral_lt_hsa": float(np.mean(rn < rh))})
    summ = {k: [float(np.mean([x[k] for x in rows])), float(np.std([x[k] for x in rows], ddof=1))]
            for k in rows[0] if k not in ("seed", "n")}
    res[scen] = {"per_seed": rows, "mean_sd": summ}
    print(scen, {k: [round(v[0], 4), round(v[1], 4)] for k, v in summ.items()}, flush=True)
json.dump(res, open("/out/baselines_all_seeds.json", "w"), indent=1)
