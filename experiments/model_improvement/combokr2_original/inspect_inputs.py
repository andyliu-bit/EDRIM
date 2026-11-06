"""Inspect the author's Zenodo files (run INSIDE the container only; pickles are loaded here).

Mounts (read-only): /mnt/zenodo (Zenodo 14509938 unpacked), /mnt/almanac (NCI-ALMANAC Nov2017 raw).
Writes: /out/inputs/zenodo_inventory.json
"""
import glob
import hashlib
import json
import os
import pickle
import sys
import zipfile

import numpy as np
import pandas as pd

Z = "/mnt/zenodo"
OUT = "/out/inputs"
os.makedirs(OUT, exist_ok=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def describe_df(df, n=3):
    out = {"shape": list(df.shape), "columns": [str(c) for c in df.columns],
           "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
           "head": df.head(n).astype(str).to_dict(orient="records")}
    num = df.select_dtypes(include=[np.number])
    if num.shape[1]:
        out["numeric_summary"] = {c: {"min": float(num[c].min()), "median": float(num[c].median()),
                                      "max": float(num[c].max()), "nan": int(num[c].isna().sum())}
                                  for c in num.columns}
    return out


inv = {"pandas": pd.__version__, "numpy": np.__version__}

# --- ALMANAC splits ---------------------------------------------------------------------------
cdd_all = np.load(f"{Z}/almanac_cv_splits/almanac_cell_drug_drug_all.npy")
inv["almanac_cdd_all"] = {"shape": list(cdd_all.shape), "dtype": str(cdd_all.dtype),
                          "head": cdd_all[:5].tolist(),
                          "n_cells": int(len(np.unique(cdd_all[:, 0]))),
                          "n_drugs": int(len(np.unique(cdd_all[:, 1:]))),
                          "cells": sorted(np.unique(cdd_all[:, 0]).tolist()),
                          "n_unique_rows": int(len(np.unique(cdd_all, axis=0))),
                          "n_unordered_dupes": None}
key_sorted = np.sort(cdd_all[:, 1:], axis=1)
und = pd.DataFrame({"c": cdd_all[:, 0], "a": key_sorted[:, 0], "b": key_sorted[:, 1]})
inv["almanac_cdd_all"]["n_unordered_dupes"] = int(und.duplicated().sum())
inv["almanac_cdd_all"]["n_d1_eq_d2"] = int((cdd_all[:, 1] == cdd_all[:, 2]).sum())

splits = {}
for f in sorted(glob.glob(f"{Z}/almanac_cv_splits/almanac_new*_inds.npy")):
    a = np.load(f)
    splits[os.path.basename(f)] = {"n": int(len(a)), "min": int(a.min()), "max": int(a.max()),
                                   "unique": int(len(np.unique(a)))}
inv["almanac_split_files"] = splits
# disjointness + entity checks for the final splits
checks = {}
for scen in ["newcombo", "newdrug", "newcell"]:
    for r in range(1, 6):
        tr = np.load(f"{Z}/almanac_cv_splits/almanac_{scen}_rseed{r}_tr_cdd_inds.npy")
        te = np.load(f"{Z}/almanac_cv_splits/almanac_{scen}_rseed{r}_tst_cdd_inds.npy")
        c_tr, c_te = set(cdd_all[tr, 0]), set(cdd_all[te, 0])
        d_tr = set(cdd_all[tr, 1]) | set(cdd_all[tr, 2])
        d_te = set(cdd_all[te, 1]) | set(cdd_all[te, 2])
        cd_tr = set(zip(cdd_all[tr, 0], cdd_all[tr, 1])) | set(zip(cdd_all[tr, 0], cdd_all[tr, 2]))
        cd_te = set(zip(cdd_all[te, 0], cdd_all[te, 1])) | set(zip(cdd_all[te, 0], cdd_all[te, 2]))
        pair_tr = {tuple(sorted(x)) for x in zip(cdd_all[tr, 1], cdd_all[tr, 2])}
        pair_te = {tuple(sorted(x)) for x in zip(cdd_all[te, 1], cdd_all[te, 2])}
        trip_tr = {(c,) + tuple(sorted(x)) for c, x in zip(cdd_all[tr, 0], zip(cdd_all[tr, 1], cdd_all[tr, 2]))}
        trip_te = {(c,) + tuple(sorted(x)) for c, x in zip(cdd_all[te, 0], zip(cdd_all[te, 1], cdd_all[te, 2]))}
        checks[f"{scen}_rseed{r}"] = {
            "n_tr": int(len(tr)), "n_te": int(len(te)), "index_overlap": int(len(set(tr) & set(te))),
            "union_covers_all": int(len(set(tr) | set(te))) == len(cdd_all),
            "test_cells_unseen": int(len(c_te - c_tr)), "test_cells": int(len(c_te)),
            "test_drugs_unseen": int(len(d_te - d_tr)),
            "test_cell_drug_pairs_unseen": int(len(cd_te - cd_tr)), "test_cell_drug_pairs": int(len(cd_te)),
            "test_drugpairs_unseen": int(len(pair_te - pair_tr)), "test_drugpairs": int(len(pair_te)),
            "unordered_triplet_overlap": int(len(trip_tr & trip_te)),
        }
inv["almanac_split_checks"] = checks

# --- ALMANAC fitted hills / braids --------------------------------------------------------------
with open(f"{Z}/almanac_fitted_braids_and_hills/almanac_fitted_hills.pkl", "rb") as f:
    hills = pickle.load(f)
inv["almanac_hills"] = describe_df(hills)
inv["almanac_hills"]["n_cells"] = int(hills["cell"].nunique()) if "cell" in hills else None
inv["almanac_hills"]["n_drugs"] = int(hills["drug"].nunique()) if "drug" in hills else None
inv["almanac_hills"]["dup_cell_drug"] = int(hills.duplicated(["cell", "drug"]).sum()) if "cell" in hills else None
inv["almanac_hills"]["cells"] = sorted(map(str, hills["cell"].unique()))[:80]
inv["almanac_hills"]["drug_dtype_example"] = repr(hills["drug"].iloc[0])

my_cdd = np.load(f"{Z}/almanac_fitted_braids_and_hills/my_cdd_full.npy")
inv["almanac_my_cdd_full"] = {"shape": list(my_cdd.shape), "head": my_cdd[:5].tolist(),
                              "n_cells": int(len(np.unique(my_cdd[:, 0])))}
set_all = set(map(tuple, cdd_all.tolist()))
set_my = set(map(tuple, my_cdd.tolist()))
inv["almanac_my_cdd_full"]["in_cdd_all"] = int(len(set_my & set_all))
inv["almanac_my_cdd_full"]["cdd_all_not_in_my"] = int(len(set_all - set_my))
sw = {(c, b, a) for c, a, b in set_my}
inv["almanac_my_cdd_full"]["cdd_all_in_my_swapped_only"] = int(len((set_all - set_my) & sw))

braid_files = sorted(glob.glob(f"{Z}/almanac_fitted_braids_and_hills/*_braid.df.pkl"))
inv["almanac_braid_files"] = len(braid_files)
frames = []
for bf in braid_files:
    with open(bf, "rb") as f:
        df = pickle.load(f)
    df = df.copy()
    df["__file_cell"] = os.path.basename(bf)[: -len("_braid.df.pkl")]
    frames.append(df)
braids = pd.concat(frames, ignore_index=True)
inv["almanac_braids"] = describe_df(braids)
braids.to_pickle("/tmp/braids_all.pkl")

# --- O'Neil & Jaaks (structure only) -------------------------------------------------------------
with open(f"{Z}/oneil_fitted_braids_and_hills/oneil_fitted_Hills.pkl", "rb") as f:
    oh = pickle.load(f)
inv["oneil_hills"] = describe_df(oh)
with open(f"{Z}/jaaks_fitted_braids_and_hills/hills.df.pkl", "rb") as f:
    jh = pickle.load(f)
inv["jaaks_hills"] = describe_df(jh)
jm = np.load(f"{Z}/jaaks_fitted_braids_and_hills/measurements_v1.npy")
inv["jaaks_measurements_v1"] = {"shape": list(jm.shape), "min": float(np.nanmin(jm)), "max": float(np.nanmax(jm)),
                                "nan": int(np.isnan(jm).sum())}

# --- raw ALMANAC zip listing ------------------------------------------------------------------
zf = zipfile.ZipFile("/mnt/almanac/ComboDrugGrowth_Nov2017.zip")
inv["almanac_zip_members"] = [(i.filename, i.file_size) for i in zf.infolist()]

with open(f"{OUT}/zenodo_inventory.json", "w") as f:
    json.dump(inv, f, indent=1, default=str)
print(json.dumps({k: (v if k not in ("almanac_split_checks", "almanac_split_files") else "...") for k, v in inv.items()},
                 indent=1, default=str)[:20000])
print("split checks (rseed1):")
for k, v in checks.items():
    if k.endswith("rseed1"):
        print(k, v)
