"""D4: zero-parameter references on DD-PRiSM's own endpoint and split style (public NCI-ALMANAC).

Re-implements the NCI-ALMANAC part of DD-PRiSM's 01_Preprocessing.ipynb (cached under
data/raw/model_comparators/ddprism) on the local ComboDrugGrowth_Nov2017.zip and scores
zero-parameter references (observed-monotherapy HSA, NCI "ExpectedGrowth"-style Bliss, a naive
product, the training mean) on the "UnseenPair" test portion of that split.

Notebook steps reproduced:
  y = (PERCENTGROWTH + 100) / 200; rows with y >= 1.5 dropped (before any aggregation)
  mono = rows with NSC2 missing; conc = log10(CONC1) + 6; median by (NSC, CONC, CELLNAME)
  comb = rows with NSC2 present; log concs; median by (NSC1, CONC1, NSC2, CONC2, CELLNAME)
  unseen cells / drugs = every int(1/(1-sqrt(0.95)))-th = 39th entry of the mono count ranking
  "seen" combinations = both drugs and the cell not unseen
  UnseenPair test = (CELLNAME, min NSC, max NSC) triplets sampled with frac = 1/9 from "seen"
Deviations (documented in the report):
  * the triplet sample uses a FIXED random_state (the notebook calls .sample() unseeded);
  * DD's cell filter (NCI60 -> DepMap mapping) and drug filter (NSCs RDKit-parsable in
    Chem2D_Jun2016.sdf) need files that are not local; no cell/drug filter is applied;
  * references use observed monotherapy medians, not DD's predicted monotherapy;
  * this is public data and not the project's locked test set (no project test shard is read).
Writes only under reports/diagnostics_20261011/.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import pandas as pd

T0 = time.perf_counter()
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "reports/diagnostics_20261011"
ZIP = ROOT.parent / "bisurfgnn/data/raw/nci_almanac/ComboDrugGrowth_Nov2017.zip"
EXPECTED_ZIP_SHA256 = "69367a8b654908ebf70a752548c8d240fda2b751336eff0241a5986628696100"  # data/raw_manifests/nci_almanac.json
PRIMARY_SEED = 20261011
EXTRA_SEEDS = list(range(10))
DD_PAPER = dict(unseen_pair_rmse=0.0854, unseen_pair_pearson=0.9063)  # values supplied by the audit


def log(*args):
    print(*args, flush=True)


def sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pearson(a, b):
    return float(np.corrcoef(np.asarray(a, float), np.asarray(b, float))[0, 1])


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


if not ZIP.exists():
    log(f"STOP: {ZIP} not found; D4 not run"); sys.exit(2)
zip_hash = sha256(ZIP)
log(f"zip sha256 {zip_hash} matches manifest: {zip_hash == EXPECTED_ZIP_SHA256}")
assert zip_hash == EXPECTED_ZIP_SHA256

PLATE_COLS = ["SCREENER", "STUDY", "TESTDATE", "PLATE"]
with zipfile.ZipFile(ZIP) as archive, archive.open("ComboDrugGrowth_Nov2017.csv") as handle:
    raw = pd.read_csv(handle, usecols=PLATE_COLS + ["NSC1", "SAMPLE1", "CONC1", "NSC2", "SAMPLE2", "CONC2", "CELLNAME",
                                                    "PERCENTGROWTH", "PERCENTGROWTHNOTZ"],
                      dtype={c: str for c in PLATE_COLS})  # PLATE has mixed int/str tokens; keep text keys
log(f"raw rows {len(raw)} read in {time.perf_counter() - T0:.1f} s")

# ---- notebook cells 46-48: endpoint and outlier filter (applied to every record) ----------------
raw["y"] = (raw.PERCENTGROWTH + 100.0) / 200.0
kept = raw[raw.y < 1.5].copy()
n_drop = len(raw) - len(kept)
kept["notz"] = kept.PERCENTGROWTHNOTZ / 100.0  # side endpoint for the scale comparison only
kept["pg"] = kept.PERCENTGROWTH

# ---- cell 49: monotherapy medians ---------------------------------------------------------------
mono = kept[kept.NSC2.isna()].copy()
mono["CONC"] = np.log10(mono.CONC1) + 6.0
mono_med = mono.groupby(["NSC1", "CONC", "CELLNAME"]).agg(y=("y", "median"), pg=("pg", "median"),
                                                         notz=("notz", "median"), n=("y", "size")).reset_index()
mono_med = mono_med.rename(columns={"NSC1": "NSC"})

# ---- cell 53: combination medians ---------------------------------------------------------------
comb = kept[kept.NSC2.notna()].copy()
comb["NSC2"] = comb.NSC2.astype(int); comb["SAMPLE2"] = comb.SAMPLE2.astype(int)
# Supplementary (not in DD): same-plate monotherapy anchors, matched like the project benchmark
# (screener, study, test date, plate, NSC, sample, raw concentration), before the log transform.
plate_mono = mono.groupby(PLATE_COLS + ["NSC1", "SAMPLE1", "CONC1"]).y.median()
for arm in ("1", "2"):
    idx = pd.MultiIndex.from_arrays([comb[c].to_numpy() for c in PLATE_COLS]
                                    + [comb[f"NSC{arm}"].to_numpy(), comb[f"SAMPLE{arm}"].to_numpy(), comb[f"CONC{arm}"].to_numpy()])
    comb[f"pm{arm}"] = plate_mono.reindex(idx).to_numpy()
comb["plate_hsa"] = np.minimum(comb.pm1, comb.pm2)
plate_cov = float(comb.plate_hsa.notna().mean())
comb["CONC1"] = np.log10(comb.CONC1) + 6.0
comb["CONC2"] = np.log10(comb.CONC2) + 6.0
comb_med = comb.groupby(["NSC1", "CONC1", "NSC2", "CONC2", "CELLNAME"]).agg(
    y=("y", "median"), notz=("notz", "median"), plate_hsa=("plate_hsa", "median"), n=("y", "size")).reset_index()
log(f"rows dropped by y>=1.5: {n_drop}; mono medians {len(mono_med)}; combination medians {len(comb_med)} "
    f"(groups with >1 record: {int((comb_med.n > 1).sum())}); same-plate mono coverage (rows) {plate_cov:.6f}")

# ---- cells 58-60: unseen cells / drugs from the monotherapy table -------------------------------
mono_dd = mono_med.rename(columns={"y": "VIABILITY"})
ratio_test = 1 - 0.9
factor_test = 1 - (1 - ratio_test / 2) ** 0.5
cells = list(set(mono_dd.CELLNAME)); drugs = list(set(mono_dd.NSC))
step_cell = int(len(cells) / (factor_test * len(cells))); step_drug = int(len(drugs) / (factor_test * len(drugs)))
cell_rank = mono_dd.groupby("CELLNAME").VIABILITY.count().sort_values(ascending=False)
drug_rank = mono_dd.groupby("NSC").VIABILITY.count().sort_values(ascending=False)
unseen_cells = list(cell_rank.iloc[np.arange(0, len(cells), step_cell)].index)
unseen_drugs = list(drug_rank.iloc[np.arange(0, len(drugs), step_drug)].index)
log(f"cells {len(cells)} drugs {len(drugs)} step {step_cell}/{step_drug}; unseen cells {unseen_cells}; unseen drugs {unseen_drugs}")

# ---- cells 65-66: "seen" combinations and the UnseenPair triplet holdout ------------------------
seen = comb_med[~comb_med.NSC1.isin(unseen_drugs) & ~comb_med.NSC2.isin(unseen_drugs)
                & ~comb_med.CELLNAME.isin(unseen_cells)].copy()
seen["min_NSC"] = np.minimum(seen.NSC1, seen.NSC2); seen["max_NSC"] = np.maximum(seen.NSC1, seen.NSC2)
triplets = seen[["CELLNAME", "min_NSC", "max_NSC"]].drop_duplicates()

# monotherapy lookup for both arms (log conc rounded to 5 dp, as the training notebook does)
mono_key = mono_med.assign(CONC=mono_med.CONC.round(5)).set_index(["NSC", "CONC", "CELLNAME"])
assert mono_key.index.is_unique


def attach_mono(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for arm in ("1", "2"):
        idx = pd.MultiIndex.from_arrays([out[f"NSC{arm}"].to_numpy(), out[f"CONC{arm}"].round(5).to_numpy(),
                                         out.CELLNAME.to_numpy()])
        looked = mono_key.reindex(idx)
        out[f"m{arm}"] = looked.y.to_numpy(); out[f"pg{arm}"] = looked.pg.to_numpy(); out[f"nz{arm}"] = looked.notz.to_numpy()
    return out


seen = attach_mono(seen)
coverage = float((seen.m1.notna() & seen.m2.notna()).mean())
log(f"seen combination medians {len(seen)}; triplets {len(triplets)}; both-mono coverage {coverage:.6f}")


def nci_expected(pg1, pg2):
    """ALMANAC_DataFields.txt ExpectedGrowth rule, on median single-agent percent growth."""
    neg = (pg1 < 0) | (pg2 < 0)
    bliss = np.minimum(pg1, 100.0) * np.minimum(pg2, 100.0) / 100.0
    return np.where(neg, np.minimum(pg1, pg2), bliss)


def evaluate(seed: int) -> tuple[dict, pd.DataFrame]:
    test_triplets = triplets.sample(frac=1 / 9, random_state=seed)
    key = ["CELLNAME", "min_NSC", "max_NSC"]
    merged = seen[key].merge(test_triplets.assign(_t=1), on=key, how="left", validate="many_to_one")
    flag = merged._t.notna().to_numpy()
    test, trainval = seen[flag], seen[~flag]
    ok = test.m1.notna() & test.m2.notna()
    t = test[ok]
    tv = trainval[trainval.m1.notna() & trainval.m2.notna()]
    slope, intercept = np.polyfit(np.minimum(tv.m1, tv.m2).to_numpy(), tv.y.to_numpy(), 1)  # 2 params, TrainVal only
    tp = t[t.plate_hsa.notna()]
    scored = [
        ("HSA min(m1,m2) [observed mono medians]", t, np.minimum(t.m1, t.m2)),
        ("NCI ExpectedGrowth-style Bliss -> DD scale", t, (nci_expected(t.pg1.to_numpy(), t.pg2.to_numpy()) + 100.0) / 200.0),
        ("naive product m1*m2 on DD scale", t, t.m1 * t.m2),
        ("TrainVal mean (constant, 1 param)", t, np.full(len(t), trainval.y.mean())),
        ("SUPPL: HSA linearly recalibrated on TrainVal (2 params)", t, intercept + slope * np.minimum(t.m1, t.m2)),
        ("SUPPL: HSA same-plate mono (project-benchmark style anchors)", tp, tp.plate_hsa),
        ("SUPPL: HSA mono medians, same subset as same-plate row", tp, np.minimum(tp.m1, tp.m2)),
    ]
    rows = []
    for name, frame, p in scored:
        p = np.asarray(p, float)
        triplet_err = pd.DataFrame({"k": frame.CELLNAME + "|" + frame.min_NSC.astype(str) + "|" + frame.max_NSC.astype(str),
                                    "e": (p - frame.y.to_numpy()) ** 2}).groupby("k").e.mean()
        rows.append(dict(seed=seed, reference=name, endpoint="DD (PG+100)/200", test_points=int(len(frame)),
                         rmse=rmse(p, frame.y), pearson=pearson(p, frame.y) if np.ptp(p) > 1e-12 else float("nan"),
                         bias=float(np.mean(p - frame.y.to_numpy())),
                         triplet_macro_rmse=float(np.sqrt(triplet_err).mean()),
                         calib_slope=float(slope) if "recalibrated" in name else float("nan"),
                         calib_intercept=float(intercept) if "recalibrated" in name else float("nan")))
    tz = t[t.notz.notna() & t.nz1.notna() & t.nz2.notna()]  # same points on the NOTZ/100 scale (unclipped)
    p = np.minimum(tz.nz1, tz.nz2).to_numpy()
    rows.append(dict(seed=seed, reference="HSA min(m1,m2) [observed mono medians]", endpoint="PERCENTGROWTHNOTZ/100 (unclipped)",
                     test_points=int(len(tz)), rmse=rmse(p, tz.notz), pearson=pearson(p, tz.notz),
                     bias=float(np.mean(p - tz.notz.to_numpy())), triplet_macro_rmse=float("nan")))
    pair_tv = set(zip(trainval.min_NSC, trainval.max_NSC))
    tt = test.drop_duplicates(key)
    info = dict(seed=seed, test_triplets=int(len(test_triplets)), test_points_all=int(len(test)), test_points_scored=int(len(t)),
                trainval_points=int(len(trainval)), test_fraction_of_seen=float(len(test) / len(seen)),
                frac_test_triplets_drug_pair_seen_in_other_cell=float(np.mean([(a, b) in pair_tv for a, b in zip(tt.min_NSC, tt.max_NSC)])))
    return info, pd.DataFrame(rows)


infos, frames = [], []
for seed in [PRIMARY_SEED] + EXTRA_SEEDS:
    info, frame = evaluate(seed); infos.append(info); frames.append(frame)
results = pd.concat(frames, ignore_index=True)
results.to_csv(OUT / "d4_ddprism_rung0_all_seeds.csv", index=False)
primary = results[results.seed == PRIMARY_SEED]
log("\n[D4] primary seed results\n" + primary.to_string(index=False))
log("[D4] primary split info:", json.dumps(infos[0]))
spread = results[results.seed.isin(EXTRA_SEEDS)].groupby(["reference", "endpoint"]).agg(
    rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std"), rmse_min=("rmse", "min"), rmse_max=("rmse", "max"),
    pearson_mean=("pearson", "mean"), pearson_sd=("pearson", "std")).reset_index()
spread.to_csv(OUT / "d4_ddprism_rung0_seed_spread.csv", index=False)
log("[D4] spread over 10 extra split seeds (0-9)\n" + spread.to_string(index=False))

hsa_p = primary[(primary.reference.str.startswith("HSA")) & (primary.endpoint.str.startswith("DD"))].iloc[0]
summary = dict(
    zip_sha256=zip_hash, raw_rows=int(len(raw)), rows_dropped_y_ge_1p5=int(n_drop), mono_medians=int(len(mono_med)),
    combination_medians=int(len(comb_med)), unseen_cells=[str(c) for c in unseen_cells], unseen_drugs=[int(d) for d in unseen_drugs],
    seen_combination_medians=int(len(seen)), seen_triplets=int(len(triplets)), both_mono_coverage_seen=coverage,
    same_plate_mono_coverage_combination_rows=plate_cov, combination_groups_with_multiple_records=int((comb_med.n > 1).sum()),
    primary_seed=PRIMARY_SEED, primary_split=infos[0], extra_split_infos=infos[1:], dd_paper=DD_PAPER,
    hsa_primary=dict(rmse=float(hsa_p.rmse), pearson=float(hsa_p.pearson)),
    hsa_minus_dd_paper_rmse=float(hsa_p.rmse - DD_PAPER["unseen_pair_rmse"]),
    hsa_minus_dd_paper_pearson=float(hsa_p.pearson - DD_PAPER["unseen_pair_pearson"]),
    runtime_seconds=round(time.perf_counter() - T0, 2), python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__)
(OUT / "d4_ddprism_rung0_summary.json").write_text(json.dumps(summary, indent=2))
log(f"[D4] HSA primary RMSE {hsa_p.rmse:.6f} Pearson {hsa_p.pearson:.6f} vs DD paper RMSE 0.0854 Pearson 0.9063")
log(f"done in {summary['runtime_seconds']} s")
