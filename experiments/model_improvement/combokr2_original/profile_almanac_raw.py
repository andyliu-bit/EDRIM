"""Profile NCI-ALMANAC ComboDrugGrowth_Nov2017 (read-only) against the author's triplet list.

Run INSIDE the container. Mounts: /mnt/almanac (ro), /mnt/zenodo (ro). Writes /out/inputs/.
Produces a compact parquet-free cache (/out/inputs/almanac_raw_slim.pkl.gz is NOT written; the slim
frame is rebuilt on demand to keep outputs small) and a JSON profile.
"""
import json
import os
import re
import zipfile

import numpy as np
import pandas as pd

OUT = "/out/inputs"
os.makedirs(OUT, exist_ok=True)
CTRL = re.compile(r"[\x00-\x1f]")
USECOLS = ["SCREENER", "STUDY", "TESTDATE", "PLATE", "NSC1", "CONCINDEX1", "CONC1", "CONCUNIT1",
           "NSC2", "CONCINDEX2", "CONC2", "CONCUNIT2", "PERCENTGROWTH", "PERCENTGROWTHNOTZ",
           "VALID", "PANEL", "CELLNAME"]


def load_raw():
    zf = zipfile.ZipFile("/mnt/almanac/ComboDrugGrowth_Nov2017.zip")
    with zf.open("ComboDrugGrowth_Nov2017.csv") as f:
        df = pd.read_csv(f, usecols=USECOLS, dtype={"SCREENER": str, "STUDY": str, "PLATE": str,
                                                    "CONCUNIT1": str, "CONCUNIT2": str, "VALID": str,
                                                    "PANEL": str, "CELLNAME": str, "TESTDATE": str})
    # author naming: "/" -> "_" (e.g. A549/ATCC -> A549_ATCC); drop control characters (SF-539 + 0x1a)
    df["CELLNAME_RAW"] = df["CELLNAME"]
    df["CELLNAME"] = df["CELLNAME"].map(lambda x: CTRL.sub("", x).strip().replace("/", "_")
                                        if isinstance(x, str) else x)
    return df


if __name__ == "__main__":
    raw = load_raw()
    prof = {"rows": int(len(raw)), "columns": list(raw.columns)}
    prof["valid_counts"] = raw["VALID"].value_counts(dropna=False).to_dict()
    prof["concunit1"] = raw["CONCUNIT1"].value_counts(dropna=False).to_dict()
    prof["concunit2"] = raw["CONCUNIT2"].value_counts(dropna=False).to_dict()
    prof["concindex1"] = raw["CONCINDEX1"].value_counts(dropna=False).sort_index().to_dict()
    prof["concindex2"] = raw["CONCINDEX2"].value_counts(dropna=False).sort_index().to_dict()
    prof["screeners"] = raw["SCREENER"].value_counts(dropna=False).to_dict()
    prof["n_cells"] = int(raw["CELLNAME"].nunique())
    prof["cells"] = sorted(raw["CELLNAME"].dropna().unique().tolist())
    prof["raw_cell_labels"] = sorted(map(repr, raw["CELLNAME_RAW"].dropna().unique().tolist()))
    mono = raw[raw["NSC2"].isna()]
    combo = raw[raw["NSC2"].notna()]
    prof["mono_rows"] = int(len(mono))
    prof["combo_rows"] = int(len(combo))
    prof["combo_rows_both_conc"] = int(((combo["CONCINDEX1"] > 0) & (combo["CONCINDEX2"] > 0)).sum())
    prof["combo_rows_concindex_zero"] = int(((combo["CONCINDEX1"] == 0) | (combo["CONCINDEX2"] == 0)).sum())
    for col in ["PERCENTGROWTH", "PERCENTGROWTHNOTZ"]:
        v = raw[col]
        prof[col] = {"min": float(v.min()), "q01": float(v.quantile(0.01)), "median": float(v.median()),
                     "q99": float(v.quantile(0.99)), "max": float(v.max()), "nan": int(v.isna().sum()),
                     "frac_lt0": float((v < 0).mean()), "frac_gt100": float((v > 100).mean())}
    c = combo[(combo["CONCINDEX1"] > 0) & (combo["CONCINDEX2"] > 0)].copy()
    c["NSC1"] = c["NSC1"].astype(int).astype(str)
    c["NSC2"] = c["NSC2"].astype(int).astype(str)
    trip = c.groupby(["CELLNAME", "NSC1", "NSC2"])
    sizes = trip.size()
    prof["ordered_triplets_raw"] = int(len(sizes))
    prof["points_per_ordered_triplet"] = sizes.value_counts().sort_index().head(30).to_dict()
    nconc = trip.agg(n1=("CONC1", "nunique"), n2=("CONC2", "nunique"), nplate=("PLATE", "nunique"),
                     nscreener=("SCREENER", "nunique"), nstudy=("STUDY", "nunique"))
    prof["n_unique_conc1_per_triplet"] = nconc["n1"].value_counts().sort_index().to_dict()
    prof["n_unique_conc2_per_triplet"] = nconc["n2"].value_counts().sort_index().to_dict()
    prof["n_plates_per_triplet"] = nconc["nplate"].value_counts().sort_index().head(20).to_dict()
    prof["n_screeners_per_triplet"] = nconc["nscreener"].value_counts().sort_index().to_dict()
    prof["n_studies_per_triplet"] = nconc["nstudy"].value_counts().sort_index().head(20).to_dict()
    # unordered
    a = np.where(c["NSC1"].astype(int) < c["NSC2"].astype(int), c["NSC1"], c["NSC2"])
    b = np.where(c["NSC1"].astype(int) < c["NSC2"].astype(int), c["NSC2"], c["NSC1"])
    c["A"], c["B"] = a, b
    und = c.groupby(["CELLNAME", "A", "B"]).size()
    prof["unordered_triplets_raw"] = int(len(und))
    both_orders = c.groupby(["CELLNAME", "A", "B"])["NSC1"].nunique()
    prof["unordered_triplets_with_both_orders"] = int((both_orders > 1).sum())

    # author triplets
    cdd = np.load("/mnt/zenodo/almanac_cv_splits/almanac_cell_drug_drug_all.npy")
    acells = sorted(np.unique(cdd[:, 0]).tolist())
    prof["author_cells"] = acells
    rawcells = set(prof["cells"])
    prof["author_cells_not_in_raw"] = [x for x in acells if x not in rawcells]
    prof["raw_cells_not_in_author"] = [x for x in prof["cells"] if x not in set(acells)]
    ordered = set(sizes.index)
    hit_same = sum((r[0], r[1], r[2]) in ordered for r in cdd.tolist())
    hit_swap = sum((r[0], r[2], r[1]) in ordered for r in cdd.tolist())
    prof["author_triplets_found_same_order"] = int(hit_same)
    prof["author_triplets_found_swapped_order"] = int(hit_swap)
    with open(f"{OUT}/almanac_raw_profile.json", "w") as f:
        json.dump(prof, f, indent=1, default=str)
    print(json.dumps(prof, indent=1, default=str)[:12000])
