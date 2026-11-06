"""Exploratory: 5x3 designs, missing drug, orientation, 1A screener (run inside container)."""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/mnt/scripts")
from profile_almanac_raw import load_raw  # noqa: E402

raw = load_raw()
cdd = np.load("/mnt/zenodo/almanac_cv_splits/almanac_cell_drug_drug_all.npy")
combo = raw[raw["NSC2"].notna()].copy()
combo["NSC1"] = combo["NSC1"].astype(int).astype(str)
combo["NSC2"] = combo["NSC2"].astype(int).astype(str)
mono = raw[raw["NSC2"].isna()].copy()
mono["NSC1"] = mono["NSC1"].astype(int).astype(str)

nsc_raw = set(combo["NSC1"]) | set(combo["NSC2"])
nsc_auth = set(cdd[:, 1]) | set(cdd[:, 2])
print("raw NSCs", len(nsc_raw), "author NSCs", len(nsc_auth))
print("raw-only NSCs", sorted(nsc_raw - nsc_auth), "author-only", sorted(nsc_auth - nsc_raw))
for n in sorted(nsc_raw - nsc_auth):
    print(" rows with", n, int(((combo.NSC1 == n) | (combo.NSC2 == n)).sum()))

# screener 1A
for s in ["FF", "FG", "1A"]:
    sub = raw[raw.SCREENER == s]
    print("screener", s, "rows", len(sub), "combo rows", int(sub.NSC2.notna().sum()),
          "NoTZ nan", int(sub.PERCENTGROWTHNOTZ.isna().sum()))

g = combo.groupby(["CELLNAME", "NSC1", "NSC2"])
size = g.size()
big = size[size == 15].index
print("15-point triplets", len(big))
ex = combo.set_index(["CELLNAME", "NSC1", "NSC2"]).loc[big[:3]].reset_index()
print(ex[["CELLNAME", "NSC1", "NSC2", "PLATE", "CONCINDEX1", "CONC1", "CONCINDEX2", "CONC2",
          "PERCENTGROWTH"]].sort_values(["CELLNAME", "NSC1", "NSC2", "CONC1", "CONC2"]).to_string())
pat = combo.set_index(["CELLNAME", "NSC1", "NSC2"]).loc[big].groupby(level=[0, 1, 2])["CONCINDEX1"].apply(
    lambda s: tuple(sorted(set(s))))
print("CONCINDEX1 patterns among 15-pt:", pat.value_counts().head(10).to_dict())
pat2 = combo.set_index(["CELLNAME", "NSC1", "NSC2"]).loc[big].groupby(level=[0, 1, 2])["PLATE"].nunique()
print("plates among 15-pt:", pat2.value_counts().to_dict())
drugs15 = pd.Series([b[1] for b in big]).value_counts()
print("drug1 in 15-pt designs:", drugs15.head(10).to_dict())
# Concentration index -1 rows
neg = combo[combo.CONCINDEX1 == -1]
print("CONCINDEX1==-1 rows", len(neg), "example:\n", neg.head(5)[["CELLNAME", "NSC1", "CONC1", "NSC2", "CONC2",
                                                                    "CONCINDEX2", "PLATE"]].to_string())
c4 = combo[combo.CONCINDEX2 == 4]
print("CONCINDEX2==4 rows", len(c4), c4.head(3)[["CELLNAME", "NSC1", "CONC1", "NSC2", "CONC2", "PLATE"]].to_string())

# orientation of author triplets
ordered = set(size.index)
same = np.array([(r[0], r[1], r[2]) in ordered for r in cdd.tolist()])
swap = np.array([(r[0], r[2], r[1]) in ordered for r in cdd.tolist()])
print("author: same", same.sum(), "swap", swap.sum(), "both", (same & swap).sum(), "neither", (~same & ~swap).sum())
neither = cdd[~same & ~swap]
print("neither examples", neither[:10].tolist())
# raw triplets not in author list (either orientation)
auth = set(map(tuple, cdd.tolist()))
auth_sw = {(c, b, a) for c, a, b in auth}
missing = [t for t in ordered if t not in auth and t not in auth_sw]
print("raw ordered triplets not in author list (either orientation):", len(missing), missing[:10])
# 15-point triplets in author list?
big_set = set(big)
print("author triplets whose raw match is 15-pt:",
      sum(((c, a, b) in big_set) or ((c, b, a) in big_set) for c, a, b in cdd.tolist()))
# mono: points per (cell, drug)
mg = mono.groupby(["CELLNAME", "NSC1"])
print("mono rows per cell-drug: median", mg.size().median(), "max", mg.size().max(),
      "unique conc per cell-drug median", mg.CONC1.nunique().median())
