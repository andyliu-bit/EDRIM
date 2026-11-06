"""Eval-paper supplement C: mechanism of the same-plate monotherapy anchor (C2) and the BMC 2019 reconciliation.

Zero-parameter descriptive analysis on public NCI-ALMANAC ComboDrugGrowth records; nothing is fitted or tuned.
Card (written before any response column was read): bisurfgnn/docs/EVAL_SAME_PLATE_ANCHOR_MECHANISM_20261018.md
  0. self-check: D1's same-plate (0.069950) and median (0.083943) HSA on D1's own 9,000 points, D1 code path
  1. DeepSynBa split-3 drug pairs (979) -> NSC pairs; their combination rows are skipped at parse time
  2. anchors (1) SP, (2) same-screener other-plate single x3, (3') same-screener same-day single x3,
     (4) same-screener median excl. own plate, (5) cross-screener median excl. own plate,
     (5r) median incl. own plate (the D1/D4 'median')
  3. surface-macro RMSE on NOTZ/100 and (PG+100)/200 on the same points, drug-pair cluster bootstrap,
     shared-plate-error estimates, mono-estimation errors against a held-out other-plate proxy.
Usage (from bisurfgnn-model-experiments):
  .venv/Scripts/python.exe -I experiments/eval_paper/same_plate_anchor_mechanism/run.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

T0 = time.perf_counter()
EXP = Path(__file__).resolve().parents[3]
BIS = EXP.parent / "bisurfgnn"
ZIP = BIS / "data/raw/nci_almanac/ComboDrugGrowth_Nov2017.zip"
NAMES = BIS / "data/raw/nci_almanac/ComboCompoundNames_all.txt"
META = BIS / "data/raw/external_benchmarks/deepsynba/data/nci_synergy_metadata.csv"
DRUGS = BIS / "data/processed/nci_almanac_unclipped/drugs.csv"
D1_SHARDS = BIS / "data/processed/one_new_scaffold_benchmark_unclipped/learned_point_shards/fold_0"
D1_IDS = EXP / "runs/model_improvement/comparator_gpu_v2"
D1_CSV = EXP / "reports/diagnostics_20261011/d1_representation_vs_learner.csv"
OUT = EXP / "runs/eval_paper/same_plate_anchor_mechanism"
PLATE_COLS = ["SCREENER", "STUDY", "TESTDATE", "PLATE"]
DRAW_SEEDS = (20261018, 20261019, 20261020)
PROXY_SEED, COV_SEED, BOOT_SEED, BOOT_DRAWS = 20261021, 20261022, 20261018, 2000
ENDPOINTS = ("notz", "dd")  # NOTZ/100 (D1 style, unclipped) and (PG+100)/200 (D4 style)


def log(*a):
    print(*a, flush=True)


def rel_err(a, b):
    return float(a / b - 1.0)


# ------------------------------------------------------------------ 0. D1 self-check (D1's own code path)
def d1_self_check(excluded_keys: np.ndarray) -> dict:
    cols = ["source_combodrugseq", "surface_replicate_id", "cell_id", "drug_a_id", "drug_b_id",
            "dose_a_model", "dose_b_model", "mono_response_a", "mono_response_b", "response"]

    def load_role(shard: Path, ids_csv: Path) -> pd.DataFrame:  # comparator_diagnostics_d1_d3.load_role
        assert shard.name in {"train.parquet", "validation.parquet"}, shard
        ids = pd.read_csv(ids_csv)
        frame = pd.read_parquet(shard, columns=cols).set_index("source_combodrugseq").loc[ids.source_combodrugseq].reset_index()
        assert (frame.surface_replicate_id.to_numpy() == ids.surface_replicate_id.to_numpy()).all()
        return frame

    va = load_role(D1_SHARDS / "validation.parquet", D1_IDS / "validation_ids.csv")
    tr = load_role(D1_SHARDS / "train.parquet", D1_IDS / "train_ids.csv")
    molar = lambda d: (10.0 ** np.asarray(d, float) - 1.0) * 1e-12
    y, surf = va.response.to_numpy(float), va.surface_replicate_id.to_numpy()
    macro = lambda p: float(np.sqrt(pd.DataFrame({"s": surf, "e": (p - y) ** 2}).groupby("s").e.mean()).mean())
    parts = [pd.DataFrame({"drug": f[f"drug_{a}_id"], "cell": f.cell_id, "dose": molar(f[f"dose_{a}_model"]),
                           "mono": f[f"mono_response_{a}"] * 100.0}) for f in (tr, va) for a in ("a", "b")]
    cov = pd.concat(parts).groupby(["cell", "drug", "dose"], as_index=False).mono.median()
    look = {(c, d, x): m / 100.0 for c, d, x, m in cov.itertuples(index=False)}
    med = {a: np.array([look[(c, d, x)] for c, d, x in zip(va.cell_id, va[f"drug_{a}_id"], molar(va[f"dose_{a}_model"]))])
           for a in ("a", "b")}
    sp = macro(np.minimum(va.mono_response_a.to_numpy(float), va.mono_response_b.to_numpy(float)))
    md = macro(np.minimum(med["a"], med["b"]))
    saved = pd.read_csv(D1_CSV)
    s_sp, s_md = float(saved.surface_macro_rmse[0]), float(saved.surface_macro_rmse[2])
    assert saved.model[0].startswith("observed HSA") and saved.model[2].startswith("median-mono HSA")
    assert abs(sp - 0.069950) < 5e-7 and abs(sp - s_sp) < 1e-12, (sp, s_sp)
    assert abs(md - 0.083943) < 5e-7 and abs(md - s_md) < 1e-12, (md, s_md)
    nsc = pd.read_csv(DRUGS, usecols=["drug_id", "nsc"]).set_index("drug_id").nsc
    na, nb = nsc.reindex(va.drug_a_id).to_numpy(np.int64), nsc.reindex(va.drug_b_id).to_numpy(np.int64)
    in_dsb = np.isin(np.minimum(na, nb) * 10**7 + np.maximum(na, nb), excluded_keys)
    res = dict(points=int(len(va)), surfaces=int(len(np.unique(surf))), same_plate_hsa=sp, median_hsa=md,
               saved_same_plate=s_sp, saved_median=s_md, reduction_vs_median=1 - sp / md,
               points_in_deepsynba_split3_pairs=int(in_dsb.sum()))
    log("[self-check 1] D1 reproduced:", json.dumps(res))
    return res


# ------------------------------------------------------------------ 1. DeepSynBa split-3 pairs -> NSC pairs
def deepsynba_test_pairs() -> tuple[np.ndarray, dict]:
    m = pd.read_csv(META, dtype=str, keep_default_na=False, usecols=["cell", "drug1", "drug2", "split"])
    t = m[m.split == "3"]
    pairs = {frozenset((a.strip(), b.strip())) for a, b in zip(t.drug1, t.drug2)}
    assert len(pairs) == 979, len(pairs)
    names = defaultdict(set)
    for line in NAMES.read_text(encoding="utf-8", errors="replace").splitlines():
        p = line.split(None, 1)
        if len(p) == 2 and p[0].isdigit():
            names[p[1].strip().lower()].add(int(p[0]))
    keys, multi = set(), {}
    for pair in pairs:
        a, b = sorted(pair)
        na, nb = names[a.lower()], names[b.lower()]
        assert na and nb, (a, b)  # every name maps to >= 1 NSC
        for d, s in ((a, na), (b, nb)):
            if len(s) > 1:
                multi[d] = sorted(s)
        keys |= {min(x, y) * 10**7 + max(x, y) for x in na for y in nb}
    info = dict(split3_matrices=int(len(t)), split3_pairs=len(pairs), nsc_pair_keys=len(keys),
                names_with_multiple_nsc=multi)
    log("[exclusion]", json.dumps(info))
    return np.array(sorted(keys), np.int64), info


def pair_key(n1: np.ndarray, n2: np.ndarray) -> np.ndarray:
    out = np.full(len(n1), -1, np.int64)
    c = ~np.isnan(n2)
    out[c] = np.minimum(n1[c], n2[c]).astype(np.int64) * 10**7 + np.maximum(n1[c], n2[c]).astype(np.int64)
    return out


def load_raw(excluded_keys: np.ndarray) -> tuple[pd.DataFrame, dict]:
    with zipfile.ZipFile(ZIP) as z:  # pass 1: drug keys only, no response column
        k = pd.read_csv(z.open("ComboDrugGrowth_Nov2017.csv"), usecols=["NSC1", "NSC2"])
    drop = np.isin(pair_key(k.NSC1.to_numpy(float), k.NSC2.to_numpy(float)), excluded_keys)
    n_total, n_drop = len(k), int(drop.sum())
    kept_n1 = k.NSC1.to_numpy(float)[~drop]
    del k
    skip = set((np.flatnonzero(drop) + 1).tolist())  # +1: line 0 is the header
    cols = PLATE_COLS + ["NSC1", "SAMPLE1", "CONC1", "NSC2", "SAMPLE2", "CONC2", "CELLNAME",
                         "PERCENTGROWTH", "PERCENTGROWTHNOTZ"]
    with zipfile.ZipFile(ZIP) as z:  # pass 2: excluded combination rows are never parsed
        raw = pd.read_csv(z.open("ComboDrugGrowth_Nov2017.csv"), usecols=cols, skiprows=skip,
                          dtype={c: "category" for c in PLATE_COLS + ["CELLNAME"]})
    # self-check 2: excluded pairs really absent, rows aligned with pass 1
    assert len(raw) == n_total - n_drop
    assert np.array_equal(raw.NSC1.to_numpy(float), kept_n1)
    assert not np.isin(pair_key(raw.NSC1.to_numpy(float), raw.NSC2.to_numpy(float)), excluded_keys).any()
    assert not raw[PLATE_COLS + ["CELLNAME"]].isna().any().any()
    pg, nz = raw.PERCENTGROWTH.to_numpy(float), raw.PERCENTGROWTHNOTZ.to_numpy(float)
    ok = ~np.isnan(pg) & ~np.isnan(nz) & ((pg + 100.0) / 200.0 < 1.5)
    info = dict(raw_rows=n_total, skipped_deepsynba_split3_rows=n_drop, loaded_rows=int(len(raw)),
                dropped_missing_pg_or_notz=int((np.isnan(pg) | np.isnan(nz)).sum()),
                dropped_dd_ge_1p5=int((~np.isnan(pg) & ~np.isnan(nz) & ((pg + 100.0) / 200.0 >= 1.5)).sum()),
                screener_rows_dropped=raw.SCREENER[~ok].value_counts().to_dict())
    raw = raw[ok].reset_index(drop=True)
    info["analysed_rows"] = int(len(raw))
    info["analysed_rows_by_screener"] = raw.SCREENER.value_counts().to_dict()
    log("[load]", json.dumps(info, default=int))
    return raw, info


# ------------------------------------------------------------------ 2. per-unit anchors
def pick(rng, mask):
    r = rng.random(mask.shape)
    r[~mask] = -1.0
    return np.where(mask.any(1), r.argmax(1), -1)


def mmed(v, mask):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(np.where(mask, v[None, :], np.nan), axis=1)


def unit_anchors(u: pd.DataFrame) -> dict:
    U = len(u)
    order = np.argsort(u.k.to_numpy(), kind="stable")
    ks = u.k.to_numpy()[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ks)) + 1]
    ends = np.r_[starts[1:], U]
    scr, date = u.scr.to_numpy(), u.date.to_numpy()
    val = {e: u[e].to_numpy(float) for e in ENDPOINTS}
    A = dict(n_other_any=np.zeros(U, np.int32), n_other_scr=np.zeros(U, np.int32), n_other_day=np.zeros(U, np.int32),
             d2=np.full((3, U), -1), d3=np.full((3, U), -1), o1=np.full(U, -1), o2=np.full(U, -1),
             proxy=np.full(U, -1), d2p=np.full((3, U), -1), d3p=np.full((3, U), -1))
    for e in ENDPOINTS:
        for name in ("med4", "med5", "med5r", "med4p", "med5p", "med5rp"):
            A[f"{name}_{e}"] = np.full(U, np.nan)
        A[f"med5r_{e}"][:] = val[e]  # singleton groups: the median incl. own plate is the own value
    rng_d2 = [np.random.default_rng(s) for s in DRAW_SEEDS]
    rng_d3 = [np.random.default_rng([s, 1]) for s in DRAW_SEEDS]
    rng_d2p = [np.random.default_rng([s, 2]) for s in DRAW_SEEDS]
    rng_d3p = [np.random.default_rng([s, 3]) for s in DRAW_SEEDS]
    rng_cov, rng_proxy = np.random.default_rng(COV_SEED), np.random.default_rng(PROXY_SEED)
    for a, b in zip(starts, ends):
        n = b - a
        if n < 2:
            continue
        idx = order[a:b]
        s, d = scr[idx], date[idx]
        other = ~np.eye(n, dtype=bool)  # units in one k group are distinct plates by construction
        m4 = other & (s[:, None] == s[None, :])
        md = m4 & (d[:, None] == d[None, :])
        A["n_other_any"][idx] = n - 1
        A["n_other_scr"][idx] = m4.sum(1)
        A["n_other_day"][idx] = md.sum(1)
        for e in ENDPOINTS:
            v = val[e][idx]
            A[f"med4_{e}"][idx] = mmed(v, m4)
            A[f"med5_{e}"][idx] = mmed(v, other)
            A[f"med5r_{e}"][idx] = np.median(v)
        for j in range(3):
            p = pick(rng_d2[j], m4); A["d2"][j, idx] = np.where(p >= 0, idx[np.maximum(p, 0)], -1)
            p = pick(rng_d3[j], md); A["d3"][j, idx] = np.where(p >= 0, idx[np.maximum(p, 0)], -1)
        two = m4.sum(1) >= 2
        r = rng_cov.random((n, n)); r[~m4] = -1.0
        top = np.argsort(-r, axis=1, kind="stable")[:, :2]
        A["o1"][idx] = np.where(two, idx[top[:, 0]], -1)
        A["o2"][idx] = np.where(two, idx[top[:, 1]], -1)
        # mono-estimation proxy: one same-screener other plate, removed from every estimator's pool
        pr = pick(rng_proxy, m4)
        pr = np.where(two, pr, -1)
        rows = np.flatnonzero(pr >= 0)
        if len(rows) == 0:
            continue
        P = np.zeros((n, n), bool); P[rows, pr[rows]] = True
        live = np.zeros((n, 1), bool); live[rows] = True
        A["proxy"][idx[rows]] = idx[pr[rows]]
        m4p, m5p, mdp, mallp = m4 & ~P & live, other & ~P & live, md & ~P & live, ~P & live
        for e in ENDPOINTS:
            v = val[e][idx]
            for name, mk in (("med4p", m4p), ("med5p", m5p), ("med5rp", mallp)):
                out = mmed(v, mk)
                A[f"{name}_{e}"][idx[rows]] = out[rows]
        for j in range(3):
            p = pick(rng_d2p[j], m4p); A["d2p"][j, idx] = np.where(p >= 0, idx[np.maximum(p, 0)], -1)
            p = pick(rng_d3p[j], mdp); A["d3p"][j, idx] = np.where(p >= 0, idx[np.maximum(p, 0)], -1)
    return A


def check_units(u: pd.DataFrame, A: dict, n_check=500) -> dict:
    """Self-check 3/4: naive per-unit recomputation on random units."""
    rng = np.random.default_rng(7)
    cand = np.flatnonzero(A["n_other_scr"] >= 2)
    groups = u.groupby("k").indices
    k, scr, plate, v = u.k.to_numpy(), u.scr.to_numpy(), u.plate.to_numpy(), u["notz"].to_numpy(float)
    for i in rng.choice(cand, size=min(n_check, len(cand)), replace=False):
        g = groups[k[i]]
        oth = [j for j in g if plate[j] != plate[i]]
        same = [j for j in oth if scr[j] == scr[i]]
        assert np.isclose(np.median(v[oth]), A["med5_notz"][i]) and np.isclose(np.median(v[same]), A["med4_notz"][i])
        assert np.isclose(np.median(v[g]), A["med5r_notz"][i])
        for j in range(3):
            assert A["d2"][j, i] in same and A["d2p"][j, i] in same and A["d2p"][j, i] != A["proxy"][i]
        assert A["o1"][i] in same and A["o2"][i] in same and A["o1"][i] != A["o2"][i]
        p = A["proxy"][i]
        assert p in same
        pool = [j for j in same if j != p]
        assert np.isclose(np.median(v[pool]), A["med4p_notz"][i])
        assert np.isclose(np.median(v[[j for j in oth if j != p]]), A["med5p_notz"][i])
        assert np.isclose(np.median(v[[j for j in g if j != p]]), A["med5rp_notz"][i])
    return dict(units_checked=int(min(n_check, len(cand))), passed=True)


# ------------------------------------------------------------------ 3. scoring
def surface_rmse(err2, sid, n_s):
    sse = np.bincount(sid, weights=err2, minlength=n_s)
    cnt = np.bincount(sid, minlength=n_s)
    return np.sqrt(sse / np.maximum(cnt, 1)), cnt > 0


def boot_weights(cl_of_surface, seed=BOOT_SEED, draws=BOOT_DRAWS):
    _, cl = np.unique(cl_of_surface, return_inverse=True)
    C = cl.max() + 1
    rng = np.random.default_rng(seed)
    W = np.empty((draws, C), np.int32)
    for b in range(draws):
        W[b] = np.bincount(rng.integers(0, C, C), minlength=C)
    return W, cl, C


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    excluded_keys, excl_info = deepsynba_test_pairs()
    d1 = d1_self_check(excluded_keys)
    raw, load_info = load_raw(excluded_keys)

    raw["plate"] = raw.groupby(PLATE_COLS, observed=True, sort=True).ngroup().astype(np.int32)
    raw["cell"] = raw.CELLNAME.cat.codes.astype(np.int32)
    raw["notz"] = raw.PERCENTGROWTHNOTZ.to_numpy(float) / 100.0
    raw["dd"] = (raw.PERCENTGROWTH.to_numpy(float) + 100.0) / 200.0
    plates = raw.drop_duplicates("plate").set_index("plate").sort_index()
    p_scr = plates.SCREENER.cat.codes.to_numpy()
    p_date = plates.TESTDATE.cat.codes.to_numpy()
    scr_names = list(raw.SCREENER.cat.categories)
    raw = raw.drop(columns=PLATE_COLS + ["CELLNAME", "PERCENTGROWTH", "PERCENTGROWTHNOTZ"])

    is_mono = raw.NSC2.isna().to_numpy()
    mono = raw[is_mono].copy()
    mono["lc"] = np.round(np.log10(mono.CONC1.to_numpy(float)) + 6.0, 5)
    ukeys = ["cell", "NSC1", "lc", "plate"]
    mono["unit"] = mono.groupby(ukeys, sort=True).ngroup()
    u = mono.groupby("unit").agg(cell=("cell", "first"), nsc=("NSC1", "first"), lc=("lc", "first"),
                                 plate=("plate", "first"), notz=("notz", "mean"), dd=("dd", "mean"), n_rec=("notz", "size"))
    u["k"] = u.groupby(["cell", "nsc", "lc"], sort=True).ngroup()
    u["scr"], u["date"] = p_scr[u.plate.to_numpy()], p_date[u.plate.to_numpy()]
    unit_info = dict(mono_records=int(len(mono)), mono_units=int(len(u)), units_with_gt1_record=int((u.n_rec > 1).sum()),
                     mono_keys_cell_nsc_conc=int(u.k.nunique()))
    # exact same-plate key of the project benchmark: (plate, cell, NSC, SAMPLE, CONC) -> unit
    ex = mono.drop_duplicates(["plate", "cell", "NSC1", "SAMPLE1", "CONC1"])
    ex_index = pd.MultiIndex.from_arrays([ex.plate.to_numpy(), ex.cell.to_numpy(), ex.NSC1.to_numpy(np.int64),
                                          ex.SAMPLE1.to_numpy(np.int64), ex.CONC1.to_numpy(float)])
    assert ex_index.is_unique
    ex_unit = ex.unit.to_numpy()
    del mono, ex

    c = raw[~is_mono].reset_index(drop=True)
    del raw
    arm = {}
    for a, (n, s, x) in {"a": ("NSC1", "SAMPLE1", "CONC1"), "b": ("NSC2", "SAMPLE2", "CONC2")}.items():
        pos = ex_index.get_indexer(pd.MultiIndex.from_arrays([c.plate.to_numpy(), c.cell.to_numpy(), c[n].to_numpy(np.int64),
                                                              c[s].to_numpy(np.int64), c[x].to_numpy(float)]))
        arm[a] = np.where(pos >= 0, ex_unit[np.maximum(pos, 0)], -1)
    n1, n2 = c.NSC1.to_numpy(np.int64), c.NSC2.to_numpy(np.int64)
    lo, hi = np.minimum(n1, n2), np.maximum(n1, n2)
    y = {e: c[e].to_numpy(float) for e in ENDPOINTS}
    cell, plate = c.cell.to_numpy(), c.plate.to_numpy()
    n_combo = len(c)
    del c

    t = time.perf_counter()
    A = unit_anchors(u)
    log(f"[units] anchors computed in {time.perf_counter() - t:.1f} s; {json.dumps(unit_info)}")
    chk = check_units(u, A)
    log("[self-check 3/4]", chk)
    val = {e: u[e].to_numpy(float) for e in ENDPOINTS}

    ua, ub = arm["a"], arm["b"]
    both = (ua >= 0) & (ub >= 0)
    sa, sb = np.maximum(ua, 0), np.maximum(ub, 0)
    nn = lambda key, m: (A[key][sa] >= m) & (A[key][sb] >= m)
    sets = {"P0": both & nn("n_other_any", 1)}
    sets["P"] = sets["P0"] & nn("n_other_scr", 1)
    sets["P_day"] = sets["P"] & nn("n_other_day", 1)
    sets["P2"] = sets["P"] & nn("n_other_scr", 2)
    # surfaces (cell, unordered pair, plate) and drug-pair clusters, defined on P0
    i0 = np.flatnonzero(sets["P0"])
    sf = pd.DataFrame({"cell": cell[i0], "lo": lo[i0], "hi": hi[i0], "plate": plate[i0]})
    sid = np.full(n_combo, -1, np.int64); sid[i0] = sf.groupby(["cell", "lo", "hi", "plate"], sort=True).ngroup().to_numpy()
    n_s = int(sid.max() + 1)
    pair_of_s = np.empty(n_s, np.int64); pair_of_s[sid[i0]] = lo[i0] * 10**7 + hi[i0]
    count_info = dict(combination_rows=n_combo, both_arms_same_plate=int(both.sum()),
                      **{f"{k}_points": int(v.sum()) for k, v in sets.items()},
                      **{f"{k}_surfaces": int(len(np.unique(sid[v]))) for k, v in sets.items()},
                      **{f"{k}_drug_pairs": int(len(np.unique(lo[v] * 10**7 + hi[v]))) for k, v in sets.items()},
                      P_points_by_screener={scr_names[k]: int(v) for k, v in
                                            zip(*np.unique(p_scr[plate[sets["P"]]], return_counts=True))})
    assert not np.isin(lo[sets["P0"]] * 10**7 + hi[sets["P0"]], excluded_keys).any()
    log("[points]", json.dumps(count_info))

    def hsa(e, ia, ib):
        return np.minimum(val[e][ia], val[e][ib])

    def anchors_for(e, m):
        ia, ib = sa[m], sb[m]
        H = {"(1) SP": [hsa(e, ia, ib)],
             "(2) same-screener other-plate single x3": [hsa(e, A["d2"][j, ia], A["d2"][j, ib]) for j in range(3)],
             "(4) same-screener median excl. own plate": [np.minimum(A[f"med4_{e}"][ia], A[f"med4_{e}"][ib])],
             "(5) cross-screener median excl. own plate": [np.minimum(A[f"med5_{e}"][ia], A[f"med5_{e}"][ib])],
             "(5r) median incl. own plate (D1/D4 'median')": [np.minimum(A[f"med5r_{e}"][ia], A[f"med5r_{e}"][ib])]}
        if (A["d3"][:, ia] >= 0).all() and (A["d3"][:, ib] >= 0).all():
            H["(3') same-screener same-day other-plate single x3"] = [hsa(e, A["d3"][j, ia], A["d3"][j, ib]) for j in range(3)]
        return H

    rows, boots, t0_rows = [], {}, []
    for set_name, keep in (("P", ["(1)", "(2)", "(4)", "(5)", "(5r)"]), ("P_day", ["(1)", "(2)", "(3')", "(4)", "(5)"]),
                           ("P0", ["(1)", "(5)", "(5r)"]), ("P2", ["(1)", "(2)", "(4)", "(5)", "(5r)"])):
        m = sets[set_name]
        s_here = np.unique(sid[m])
        W, cl, C = boot_weights(pair_of_s[s_here])
        per_e = {}
        for e in ENDPOINTS:
            H = {k: v for k, v in anchors_for(e, m).items() if k.split()[0] in keep}
            assert set(k.split()[0] for k in H) == set(keep), (set_name, list(H))
            ps = {}
            for name, preds in H.items():
                draws = []
                for p in preds:
                    assert np.isfinite(p).all(), name
                    r, _ = surface_rmse((p - y[e][m]) ** 2, sid[m], n_s)
                    draws.append(r[s_here])
                ps[name] = np.mean(draws, axis=0)
                macro_draws = [float(d.mean()) for d in draws]
                rows.append(dict(point_set=set_name, endpoint=e, anchor=name, n_points=int(m.sum()), n_surfaces=int(len(s_here)),
                                 n_drug_pairs=int(C), surface_macro_rmse=float(np.mean(macro_draws)),
                                 draw_sd=float(np.std(macro_draws, ddof=1)) if len(draws) > 1 else float("nan"),
                                 point_rmse=float(np.mean([np.sqrt(np.mean((p - y[e][m]) ** 2)) for p in preds])),
                                 bias=float(np.mean([np.mean(p - y[e][m]) for p in preds]))))
            sums = {k: np.bincount(cl, weights=v, minlength=C) for k, v in ps.items()}
            ref = sums["(1) SP"]
            for name in ps:
                if name == "(1) SP":
                    continue
                stat = (W @ sums[name]) / (W @ ref) - 1.0
                lo95, hi95 = np.percentile(stat, [2.5, 97.5])
                boots[(set_name, e, name)] = (float(sums[name].sum() / ref.sum() - 1.0), float(lo95), float(hi95))
            per_e[e] = (sums, W)
        # time-zero test: Delta_E = 1 - SP / median, DD minus NOTZ, same cluster draws
        for med in ("(5r) median incl. own plate (D1/D4 'median')", "(5) cross-screener median excl. own plate"):
            if med.split()[0] not in keep:
                continue
            dl = {e: 1.0 - per_e[e][0]["(1) SP"].sum() / per_e[e][0][med].sum() for e in ENDPOINTS}
            db = {e: 1.0 - (W @ per_e[e][0]["(1) SP"]) / (W @ per_e[e][0][med]) for e in ENDPOINTS}
            diff = db["dd"] - db["notz"]
            t0_rows.append(dict(point_set=set_name, median=med, delta_notz=float(dl["notz"]), delta_dd=float(dl["dd"]),
                                dd_minus_notz_pp=float(100 * (dl["dd"] - dl["notz"])),
                                ci95_pp=[float(100 * x) for x in np.percentile(diff, [2.5, 97.5])],
                                delta_notz_ci95=[float(x) for x in np.percentile(db["notz"], [2.5, 97.5])],
                                delta_dd_ci95=[float(x) for x in np.percentile(db["dd"], [2.5, 97.5])]))
    tab = pd.DataFrame(rows)
    tab["rel_vs_sp"] = [boots.get((r.point_set, r.endpoint, r.anchor), (0.0, 0.0, 0.0))[0] for r in tab.itertuples()]
    tab["ci95_low"] = [boots.get((r.point_set, r.endpoint, r.anchor), (np.nan,) * 3)[1] for r in tab.itertuples()]
    tab["ci95_high"] = [boots.get((r.point_set, r.endpoint, r.anchor), (np.nan,) * 3)[2] for r in tab.itertuples()]
    tab.to_csv(OUT / "anchor_table.csv", index=False)
    log("\n[anchors] surface-macro RMSE, rel vs SP (drug-pair cluster bootstrap 95% CI)\n" + tab.drop(columns=["n_points", "n_surfaces", "n_drug_pairs"]).round(6).to_string(index=False))
    log("[T0 test]", json.dumps(t0_rows, indent=1))

    # ---- shared plate error -----------------------------------------------------------------------------
    shared = {}
    for e in ENDPOINTS:
        m = sets["P"]; ia, ib = sa[m], sb[m]; yy = y[e][m]
        hsp = hsa(e, ia, ib)
        corr2 = [float(np.corrcoef(yy - h, hsp - h)[0, 1]) for h in (hsa(e, A["d2"][j, ia], A["d2"][j, ib]) for j in range(3))]
        h5 = np.minimum(A[f"med5_{e}"][ia], A[f"med5_{e}"][ib])
        m2 = sets["P2"]; ja, jb = sa[m2], sb[m2]; y2 = y[e][m2]
        hsp2 = hsa(e, ja, jb)
        h1, h2 = hsa(e, A["o1"][ja], A["o1"][jb]), hsa(e, A["o2"][ja], A["o2"][jb])
        assert (A["o1"][ja] != A["o2"][ja]).all() and (A["o1"][jb] != A["o2"][jb]).all()
        orient = []
        for ha, hb in ((h1, h2), (h2, h1)):
            C_hat = float(np.mean((y2 - ha) * (hsp2 - hb)))
            V_hat = float(np.mean((hsp2 - hb) ** 2) - np.mean((ha - hb) ** 2))
            obs = float(np.mean((y2 - ha) ** 2) - np.mean((y2 - hsp2) ** 2))
            orient.append(dict(two_C=2 * C_hat, V=V_hat, predicted_mse_gap=2 * C_hat - V_hat, observed_mse_gap=obs,
                               centred_cov=float(np.cov(y2 - ha, hsp2 - hb)[0, 1]),
                               corr=float(np.corrcoef(y2 - ha, hsp2 - hb)[0, 1])))
        avg = {k: float(np.mean([o[k] for o in orient])) for k in orient[0]}
        # cluster bootstrap of 2C and V (point level, P2), both orientations averaged
        s2 = np.unique(sid[m2]); W, cl, C = boot_weights(pair_of_s[s2])
        pt_cl = cl[np.searchsorted(s2, sid[m2])]
        prod = 0.5 * ((y2 - h1) * (hsp2 - h2) + (y2 - h2) * (hsp2 - h1))
        vterm = 0.5 * ((hsp2 - h2) ** 2 + (hsp2 - h1) ** 2) - (h1 - h2) ** 2
        cnt = np.bincount(pt_cl, minlength=C).astype(float)
        bC = 2 * (W @ np.bincount(pt_cl, weights=prod, minlength=C)) / (W @ cnt)
        bV = (W @ np.bincount(pt_cl, weights=vterm, minlength=C)) / (W @ cnt)
        shared[e] = dict(P_points=int(m.sum()), corr_y_minus_Ho_vs_HSP_minus_Ho_draw2=corr2,
                         corr_draw2_mean=float(np.mean(corr2)), corr_with_Ho_median5=float(np.corrcoef(yy - h5, hsp - h5)[0, 1]),
                         P2_points=int(m2.sum()), clean_two_orientations=orient, clean_mean=avg,
                         two_C_ci95=[float(x) for x in np.percentile(bC, [2.5, 97.5])],
                         V_ci95=[float(x) for x in np.percentile(bV, [2.5, 97.5])],
                         two_C_minus_V_ci95=[float(x) for x in np.percentile(bC - bV, [2.5, 97.5])],
                         sp_wins_condition_2C_gt_V=bool(avg["two_C"] > avg["V"]))
    log("[shared plate error]", json.dumps(shared, indent=1))

    # ---- mono-estimation errors against a held-out other-plate proxy (BMC reconciliation) ---------------
    arm_units = np.unique(np.r_[sa[sets["P"]], sb[sets["P"]]])
    mu = arm_units[A["proxy"][arm_units] >= 0]
    pr = A["proxy"][mu]
    assert (pr != mu).all() and (u.plate.to_numpy()[pr] != u.plate.to_numpy()[mu]).all()
    assert all((A[k][j, mu] != pr).all() and (A[k][j, mu] != mu).all() for k in ("d2p", "d3p") for j in range(3))
    mono_rows = []
    day = (A["d3p"][:, mu] >= 0).all(0)
    for e in ENDPOINTS:
        tgt = val[e][pr]
        est = {"(1) SP": [val[e][mu]], "(2) same-screener other-plate single x3": [val[e][A["d2p"][j, mu]] for j in range(3)],
               "(4) same-screener median excl. own plate": [A[f"med4p_{e}"][mu]],
               "(5) cross-screener median excl. own plate": [A[f"med5p_{e}"][mu]],
               "(5r) median incl. own plate (D1/D4 'median')": [A[f"med5rp_{e}"][mu]]}
        for name, ps in est.items():
            assert all(np.isfinite(p).all() for p in ps), name
            r = [float(np.sqrt(np.mean((p - tgt) ** 2))) for p in ps]
            mono_rows.append(dict(unit_set="arm units of P with >=2 same-screener other plates", endpoint=e, estimator=name,
                                  n_units=int(len(mu)), rmse_vs_proxy=float(np.mean(r)), mse_vs_proxy=float(np.mean(np.square(r))),
                                  bias_vs_proxy=float(np.mean([np.mean(p - tgt) for p in ps]))))
        sub = {"(1) SP": [val[e][mu[day]]], "(3') same-screener same-day other-plate single x3": [val[e][A["d3p"][j, mu[day]]] for j in range(3)],
               "(2) same-screener other-plate single x3": [val[e][A["d2p"][j, mu[day]]] for j in range(3)]}
        for name, ps in sub.items():
            r = [float(np.sqrt(np.mean((p - tgt[day]) ** 2))) for p in ps]
            mono_rows.append(dict(unit_set="same, restricted to units with a same-day other plate besides the proxy", endpoint=e,
                                  estimator=name, n_units=int(day.sum()), rmse_vs_proxy=float(np.mean(r)),
                                  mse_vs_proxy=float(np.mean(np.square(r))), bias_vs_proxy=float(np.mean([np.mean(p - tgt[day]) for p in ps]))))
    mono_tab = pd.DataFrame(mono_rows)
    mono_tab.to_csv(OUT / "mono_estimation_table.csv", index=False)
    log("\n[mono estimation vs held-out other-plate proxy]\n" + mono_tab.round(6).to_string(index=False))

    two_col = []
    for e in ENDPOINTS:
        mt = mono_tab[(mono_tab.endpoint == e) & mono_tab.unit_set.str.startswith("arm")].set_index("estimator")
        for set_name in ("P", "P2"):
            at = tab[(tab.endpoint == e) & (tab.point_set == set_name)].set_index("anchor")
            for name in mt.index:
                two_col.append(dict(endpoint=e, combination_point_set=set_name, anchor_source=name,
                                    mono_estimation_rmse=float(mt.rmse_vs_proxy[name]),
                                    combination_anchor_surface_macro_rmse=float(at.surface_macro_rmse[name]),
                                    combination_rel_vs_sp=float(at.rel_vs_sp[name])))
    two_col = pd.DataFrame(two_col)
    two_col.to_csv(OUT / "two_column_table.csv", index=False)

    # ---- pre-specified decisions (NOTZ primary, P) -------------------------------------------------------
    def decide(e):
        r = lambda key: float(tab[(tab.point_set == "P") & (tab.endpoint == e) & tab.anchor.str.startswith(key)].rel_vs_sp.iloc[0])
        r2, r4, r5 = r("(2)"), r("(4)"), r("(5)")
        if r4 <= 0 or r5 <= 0:
            code = "D-a: C2 as stated does not hold"
        elif r2 < 0.05:
            code = "D-b: same-plate advantage not from shared plate error; rewrite C2 as 'median/Hill anchors are biased'"
        else:
            code = "D-c: C2 supported; mechanism = shared plate/batch error"
        return dict(rel_2=r2, rel_4=r4, rel_5=r5, decision=code)

    decisions = {e: decide(e) for e in ENDPOINTS}
    t0p = next(r for r in t0_rows if r["point_set"] == "P" and r["median"].startswith("(5r)"))
    decisions["t0"] = ("|DD-NOTZ| < 5 pp: time-zero normalisation does not explain the D1/D4 gap" if abs(t0p["dd_minus_notz_pp"]) < 5
                       else ("DD larger by >= 5 pp: endpoint contributes" if t0p["dd_minus_notz_pp"] >= 5 else "NOTZ larger by >= 5 pp"))
    decisions["endpoint_dependent"] = decisions["notz"]["decision"][:3] != decisions["dd"]["decision"][:3]
    log("[decisions]", json.dumps(decisions, indent=1))

    result = dict(card="bisurfgnn/docs/EVAL_SAME_PLATE_ANCHOR_MECHANISM_20261018.md", exclusion=excl_info, d1_self_check=d1,
                  load=load_info, units=unit_info, unit_check=chk, points=count_info, t0_test=t0_rows, shared_plate_error=shared,
                  decisions=decisions, two_column_table=two_col.to_dict("records"),
                  seeds=dict(draws=DRAW_SEEDS, proxy=PROXY_SEED, cov=COV_SEED, bootstrap=BOOT_SEED, bootstrap_draws=BOOT_DRAWS),
                  same_study_anchor_3="unavailable: within one STUDY each (cell, NSC, conc) occurs on exactly one plate",
                  runtime_seconds=round(time.perf_counter() - T0, 1), python=sys.version.split()[0],
                  numpy=np.__version__, pandas=pd.__version__)
    (OUT / "results.json").write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    log(f"done in {result['runtime_seconds']} s")


if __name__ == "__main__":
    main()
