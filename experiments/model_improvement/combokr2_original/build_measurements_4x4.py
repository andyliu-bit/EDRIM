"""Build the 4x4 NCI-ALMANAC response matrices that reproduce the paper's neutral-baseline Pearson.

grid = {0, a1, a2, a3} x {0, b1, b2, b3}: (0,0) = 100 (vehicle control; the author's BRAID code sets
res[D == 0] = 100), (ai, 0) / (0, bj) = same-study monotherapy of each drug at the combination
concentrations (mean of replicate wells/plates), (ai, bj) = the 3x3 combination measurements.
Endpoint: NoTZ = 100*TESTVALUE/CONTROLVALUE (clipping to [0,100] is applied at evaluation time).
5x3 designs (screener 1A): the three lowest concentrations of the 5-dose drug (idx123), see calibration.
Run INSIDE the container. Output: /out/inputs/almanac_measurements_4x4.npz
"""
import json
import sys
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, "/mnt/scripts")
from profile_almanac_raw import CTRL  # noqa: E402

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
mstudy = mono.groupby(["CELLNAME", "NSC1", "CK", "STUDY"])["NZ"].agg(["mean", "size"])
mmean, msize = mstudy["mean"].to_dict(), mstudy["size"].to_dict()
combo = raw[raw["NSC2"].notna()].copy()
combo["NSC2"] = combo["NSC2"].astype(int).astype(str).replace(ALIAS)
study_of = combo.groupby(["CELLNAME", "NSC1", "NSC2"])["STUDY"].first().to_dict()

cdd = np.load(f"{Z}/almanac_cv_splits/almanac_cell_drug_drug_all.npy")
v2 = np.load(f"{I}/almanac_measurements_v2.npz")
c1, c2, y9 = v2["c1"], v2["c2"], v2["y_nzcalc"]
n = len(cdd)
C1 = np.zeros((n, 16)); C2 = np.zeros((n, 16)); Y = np.full((n, 16), np.nan)
combo_pos = np.zeros(16, dtype=bool)
mono_reps = np.zeros((n, 6))
for i, (cell, a, b) in enumerate(cdd.tolist()):
    st = study_of[(cell, a, b)]
    la = np.unique(c1[i]); lb = np.unique(c2[i])
    if len(la) != 3 or len(lb) != 3:
        continue
    ga = np.concatenate([[0.0], la]); gb = np.concatenate([[0.0], lb])
    k = 0
    for ia, ca in enumerate(ga):
        for ib, cb in enumerate(gb):
            C1[i, k], C2[i, k] = ca, cb
            if ia == 0 and ib == 0:
                Y[i, k] = 100.0
            elif ib == 0:
                key = (cell, a, float(f"{ca:.6e}"), st)
                Y[i, k] = mmean.get(key, np.nan); mono_reps[i, ia - 1] = msize.get(key, 0)
            elif ia == 0:
                key = (cell, b, float(f"{cb:.6e}"), st)
                Y[i, k] = mmean.get(key, np.nan); mono_reps[i, 2 + ib] = msize.get(key, 0)
            else:
                sel = np.where((c1[i] == ca) & (c2[i] == cb))[0]
                Y[i, k] = y9[i, sel[0]]
                combo_pos[k] = True
            k += 1
audit = {"rows_complete": int((~np.isnan(Y).any(1)).sum()), "rows": int(n),
         "combo_positions": np.where(combo_pos)[0].tolist(),
         "mono_replicates_median": float(np.median(mono_reps[mono_reps > 0])),
         "mono_replicates_max": float(mono_reps.max())}
np.savez_compressed(f"{I}/almanac_measurements_4x4.npz", c1=C1, c2=C2, y_nzcalc=Y, combo_pos=combo_pos)
with open(f"{I}/measurement_4x4_audit.json", "w") as f:
    json.dump(audit, f, indent=1)
print(json.dumps(audit, indent=1))
