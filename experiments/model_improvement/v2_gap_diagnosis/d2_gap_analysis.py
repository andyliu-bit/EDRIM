"""V2 gap diagnosis, step D2: where does DeepSynBa v2 beat BiSurfGNN on the VALIDATION split?

Validation only (split 2). Reads:
  * the gate's S1 validation point table (hash-checked against s1/manifest.json) and, for training-frequency counts,
    the S1 training point table (identifiers only, no responses used);
  * the gate's saved validation predictions: BiSurfGNN additive seeds 41-45 + anchored 41 (S3/S4), XGBoost seeds 41-45;
  * DeepSynBa v2 validation predictions written by d1_deepsynba_val_inference.py (epoch 484 = selected checkpoint,
    epoch 47 = pilot checkpoint, identical to the full run's epoch-47 state).
Nothing under s5/, no marker file, no test predictions, no test points are opened. Nothing is trained.

Writes tables and a JSON summary to runs/model_improvement/v2_gap_diagnosis/analysis/ (fresh directory).

  .venv/Scripts/python.exe -I experiments/model_improvement/v2_gap_diagnosis/d2_gap_analysis.py
"""
from __future__ import annotations

import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
GATE = ROOT / "runs/model_improvement/public_split_gate_v1"
S1 = GATE / "s1"
DIAG = ROOT / "runs/model_improvement/v2_gap_diagnosis"
DSB_DIR = DIAG / "deepsynba_val"
OUT = DIAG / "analysis"
VAL_OFFSET = 10_000_000
BOOT_SEED, BOOT_N = 20261009, 2000

BISURF = {
    "BS_add_s41": GATE / "s3_additive_r2/bisurf_additive_seed_41",
    "BS_add_s42": GATE / "s4_bisurf_additive_a/bisurf_additive_seed_42",
    "BS_add_s43": GATE / "s4_bisurf_additive_b/bisurf_additive_seed_43",
    "BS_add_s44": GATE / "s4_bisurf_additive_a/bisurf_additive_seed_44",
    "BS_add_s45": GATE / "s4_bisurf_additive_b/bisurf_additive_seed_45",
    "BS_anch_s41": GATE / "s3_anchored_r2/bisurf_anchored_seed_41",
}
XGB = {f"XGB_s{s}": GATE / f"s4_xgb_r2/xgb_residual_d10_m1_seed_{s}" for s in (41, 42, 43, 44, 45)}
BS_SEEDS = [f"BS_add_s{s}" for s in (41, 42, 43, 44, 45)]
XGB_SEEDS = list(XGB)
DSB_FILES = {"DSB_e484": "full_epoch484_selected_val_predictions.npz", "DSB_e047": "pilot_epoch047_val_predictions.npz"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def rmse(e) -> float:
    return float(np.sqrt(np.mean(np.square(e))))


def load_inputs():
    manifest = json.loads((S1 / "manifest.json").read_text(encoding="utf-8"))
    for name in ("points_validation.parquet", "points_train.parquet"):
        if sha256(S1 / name) != manifest[name]:
            raise RuntimeError(f"S1 table changed: {name}")
    v = pd.read_parquet(S1 / "points_validation.parquet")
    tr = pd.read_parquet(S1 / "points_train.parquet",
                         columns=["matrix_key", "pair_id", "deepsynba_cell", "drug_a_id", "drug_b_id", "row_i"])
    tr = tr[tr.row_i == 1].drop_duplicates("matrix_key").drop(columns="row_i").reset_index(drop=True)
    y = v.response_percent.to_numpy(float)
    preds, checks = {}, {}
    for tag, path in {**BISURF, **XGB}.items():
        frame = pd.read_csv(path / "validation_predictions.csv")
        metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
        ok_order = bool((frame.source_row.to_numpy() == VAL_OFFSET + np.arange(len(v))).all())
        ok_surface = bool((frame.surface_id.to_numpy() == v.matrix_key.to_numpy()).all())
        target_diff = float(np.max(np.abs(frame.target.to_numpy(float) * 100.0 - y)))
        p = frame.prediction.to_numpy(float) * 100.0
        logged = metrics["validation_pooled_rmse_percent"]
        if not (ok_order and ok_surface and target_diff < 1e-4 and abs(rmse(p - y) - logged) < 1e-6):
            raise RuntimeError(f"{tag}: alignment failure {ok_order} {ok_surface} {target_diff} {rmse(p - y)} {logged}")
        preds[tag] = p
        checks[tag] = dict(rows=len(frame), order_ok=ok_order, surface_ok=ok_surface, target_max_abs_diff=target_diff,
                           rmse_recomputed=rmse(p - y), rmse_logged=logged, best_epoch=metrics.get("best_epoch"),
                           stopped_by=metrics.get("stopped_by"))
    key_to_row = {}
    edges = {}
    for tag, name in DSB_FILES.items():
        z = np.load(DSB_DIR / name, allow_pickle=False)
        keys = z["matrix_key"]
        index = pd.Index(keys)
        if not index.is_unique:
            raise RuntimeError("DeepSynBa validation keys are not unique")
        pos = index.get_indexer(v.matrix_key)
        if (pos < 0).any():
            raise RuntimeError(f"{tag}: {int((pos < 0).sum())} gate validation points have no DeepSynBa prediction")
        i, j = v.row_i.to_numpy(), v.col_j.to_numpy()
        pm, tm = z["predictions"].astype(float), z["truth"].astype(float)
        truth_diff = float(np.max(np.abs(tm[pos, i, j] - y)))
        if truth_diff > 1e-3:
            raise RuntimeError(f"{tag}: truth differs from the gate table ({truth_diff})")
        preds[tag] = pm[pos, i, j]
        edges[tag] = dict(mono_a=pm[pos, i, 0], mono_b=pm[pos, 0, j], control=pm[pos, 0, 0])
        if not key_to_row:
            key_to_row = dict(pos=pos, truth=tm, pred_full={}, keys=keys)
            edges["truth"] = dict(mono_a=tm[pos, i, 0], mono_b=tm[pos, 0, j], control=tm[pos, 0, 0])
        key_to_row["pred_full"][tag] = pm
        checks[tag] = dict(matrices_in_file=int(len(keys)), gate_matrices_matched=int(v.matrix_key.nunique()),
                           truth_max_abs_diff=truth_diff, rmse_combination=rmse(preds[tag] - y))
    return v, tr, y, preds, edges, key_to_row, checks


def add_derived(v, y, preds, edges):
    hsa = v.hsa_lookup_percent.to_numpy(float)
    bliss = v.mono_a_percent.to_numpy(float) * v.mono_b_percent.to_numpy(float) / 100.0
    preds["HSA"], preds["Bliss"] = hsa, bliss
    preds["BS_ens5"] = np.mean([preds[k] for k in BS_SEEDS], axis=0)
    preds["XGB_ens5"] = np.mean([preds[k] for k in XGB_SEEDS], axis=0)
    for tag in ("DSB_e484", "DSB_e047"):
        e = edges[tag]
        own_hsa = np.minimum(e["mono_a"], e["mono_b"])
        own_bliss = e["mono_a"] * e["mono_b"] / 100.0
        preds[f"{tag}_ownHSA"] = own_hsa
        preds[f"{tag}_reanchor_HSA"] = preds[tag] - own_hsa + hsa
        preds[f"{tag}_reanchor_Bliss"] = preds[tag] - own_bliss + bliss
    return hsa


def group_metrics(y, preds, surfaces):
    rows = []
    codes, uniq = pd.factorize(surfaces)
    for tag, p in preds.items():
        e = p - y
        sse_s = np.bincount(codes, weights=e * e)
        n_s = np.bincount(codes)
        rows.append(dict(model=tag, rmse=rmse(e), pearson=float(np.corrcoef(p, y)[0, 1]), bias=float(np.mean(e)),
                         surface_macro_rmse=float(np.mean(np.sqrt(sse_s / n_s)))))
    return pd.DataFrame(rows)


def pair_bootstrap(y, preds, pairs, comparisons):
    """Pair-level bootstrap (gate convention): 2000 resamples of validation pairs, seed 20261009; each resample
    recomputes the mean of per-seed RMSEs of both methods and their relative difference."""
    codes, uniq = pd.factorize(pairs)
    n_p = np.bincount(codes).astype(float)
    sse = {k: np.bincount(codes, weights=(p - y) ** 2) for k, p in preds.items()}
    rng = np.random.default_rng(BOOT_SEED)
    w = rng.multinomial(len(uniq), np.full(len(uniq), 1.0 / len(uniq)), size=BOOT_N).astype(float)
    n_b = w @ n_p

    def stat(members, weights=None):
        if weights is None:
            return np.mean([np.sqrt(sse[m].sum() / n_p.sum()) for m in members])
        return np.mean([np.sqrt((weights @ sse[m]) / n_b) for m in members], axis=0)

    out = []
    for name, (a, b) in comparisons.items():
        point = stat(a) / stat(b) - 1.0
        boot = stat(a, w) / stat(b, w) - 1.0
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out.append(dict(comparison=name, a=" + ".join(a), b=" + ".join(b), rmse_a=stat(a), rmse_b=stat(b),
                        relative=point, ci_low=float(lo), ci_high=float(hi), pairs=len(uniq)))
    return pd.DataFrame(out)


def slice_table(labels, y, hsa, preds, name):
    """Per-slice RMSE, bias, mean correction and share of the squared-error gap. The BiSurfGNN and XGBoost columns
    average squared errors over their 5 seeds (consistent with the gate's 'mean over seeds'); *_ens columns use the
    seed-averaged prediction."""
    labels = pd.Series(np.asarray(labels).astype(str))
    e2 = {k: (preds[k] - y) ** 2 for k in BS_SEEDS + XGB_SEEDS + ["HSA", "BS_ens5", "XGB_ens5", "DSB_e484", "DSB_e047"]}
    bs_sse = np.mean([e2[k] for k in BS_SEEDS], axis=0)
    xgb_sse = np.mean([e2[k] for k in XGB_SEEDS], axis=0)
    gap = bs_sse - e2["DSB_e484"]
    gap_ens = e2["BS_ens5"] - e2["DSB_e484"]
    gap_xgb = xgb_sse - e2["DSB_e484"]
    gap_bx = bs_sse - xgb_sse
    tot, tot_ens, tot_xgb, tot_bx = gap.sum(), gap_ens.sum(), gap_xgb.sum(), gap_bx.sum()
    rows = []
    for lab, idx in labels.groupby(labels).groups.items():
        idx = np.asarray(idx)
        yy = y[idx]
        rows.append(dict(
            slice_type=name, slice=lab, points=len(idx), point_share=len(idx) / len(y),
            mean_y=float(yy.mean()), mean_hsa=float(hsa[idx].mean()), true_dev_vs_hsa=float((yy - hsa[idx]).mean()),
            rmse_HSA=float(np.sqrt(e2["HSA"][idx].mean())), rmse_XGB=float(np.sqrt(xgb_sse[idx].mean())),
            rmse_BS=float(np.sqrt(bs_sse[idx].mean())), rmse_BS_ens5=float(np.sqrt(e2["BS_ens5"][idx].mean())),
            rmse_DSB=float(np.sqrt(e2["DSB_e484"][idx].mean())), rmse_DSB_e047=float(np.sqrt(e2["DSB_e047"][idx].mean())),
            corr_BS=float((preds["BS_ens5"][idx] - hsa[idx]).mean()), corr_XGB=float((preds["XGB_ens5"][idx] - hsa[idx]).mean()),
            corr_DSB=float((preds["DSB_e484"][idx] - hsa[idx]).mean()),
            corr_DSB_e047=float((preds["DSB_e047"][idx] - hsa[idx]).mean()),
            bias_BS=float((preds["BS_ens5"][idx] - yy).mean()), bias_XGB=float((preds["XGB_ens5"][idx] - yy).mean()),
            bias_DSB=float((preds["DSB_e484"][idx] - yy).mean()),
            gap_sse_BS_minus_DSB=float(gap[idx].sum()), share_gap_BS_vs_DSB=float(gap[idx].sum() / tot),
            share_gap_BSens_vs_DSB=float(gap_ens[idx].sum() / tot_ens),
            share_gap_XGB_vs_DSB=float(gap_xgb[idx].sum() / tot_xgb),
            share_gap_BS_vs_XGB=float(gap_bx[idx].sum() / tot_bx) if abs(tot_bx) > 0 else None,
            rel_gap_in_slice=float(np.sqrt(bs_sse[idx].mean()) / np.sqrt(e2["DSB_e484"][idx].mean()) - 1.0)))
    return pd.DataFrame(rows)


def bins(x, edges, labels):
    return pd.cut(x, edges, right=False, labels=labels).astype(str)


def decomposition(y, preds, v, members):
    """Orthogonal SSE decomposition of each model's error: pair mean + (matrix mean - pair mean) + within-matrix."""
    pc, _ = pd.factorize(v.pair_id)
    mc, _ = pd.factorize(v.matrix_key)
    n_p, n_m = np.bincount(pc), np.bincount(mc)
    res = {}
    for k in members:
        e = preds[k] - y
        ep = (np.bincount(pc, weights=e) / n_p)[pc]
        em = (np.bincount(mc, weights=e) / n_m)[mc]
        res[k] = dict(pair_level=float(np.sum(ep ** 2)), matrix_level=float(np.sum((em - ep) ** 2)),
                      within_matrix=float(np.sum((e - em) ** 2)), total=float(np.sum(e ** 2)))
    return res


def concentration(group_ids, gap):
    codes, uniq = pd.factorize(group_ids)
    g = np.bincount(codes, weights=gap)
    order = np.sort(g)[::-1]
    total = g.sum()
    out = {"groups": int(len(uniq)), "groups_where_DSB_better": int((g > 0).sum()),
           "share_groups_DSB_better": float((g > 0).mean())}
    for q in (0.01, 0.05, 0.10, 0.25, 0.50):
        k = max(1, int(round(q * len(g))))
        out[f"top_{int(q * 100)}pct_share_of_gap"] = float(order[:k].sum() / total)
    cum = np.cumsum(order) / total
    out["groups_needed_for_100pct_of_net_gap"] = int(np.argmax(cum >= 1.0) + 1) if (cum >= 1.0).any() else None
    return out


def calibration(y, hsa, preds, members, mask=None):
    rows = []
    for k in members:
        d, dh = y - hsa, preds[k] - hsa
        if mask is not None:
            d, dh = d[mask], dh[mask]
        slope = float(np.cov(dh, d)[0, 1] / np.var(dh, ddof=1)) if np.var(dh) > 0 else None
        rows.append(dict(model=k, corr_pred_dev_true_dev=float(np.corrcoef(dh, d)[0, 1]) if np.var(dh) > 0 else None,
                         slope_true_on_pred=slope, sd_pred_dev=float(np.std(dh)), sd_true_dev=float(np.std(d)),
                         mean_pred_dev=float(np.mean(dh)), mean_true_dev=float(np.mean(d)),
                         r2_of_dev=float(1 - np.sum((dh - d) ** 2) / np.sum((d - d.mean()) ** 2))))
    return rows


def blend_crossfit(y, a, b, pairs, seed=7):
    """w*a + (1-w)*b with w fitted (least squares) on one random half of the validation pairs and scored on the other."""
    uniq = np.array(sorted(set(pairs)))
    rng = np.random.default_rng(seed)
    half = set(rng.choice(uniq, size=len(uniq) // 2, replace=False))
    in_a = np.array([p in half for p in pairs])
    out, sse = [], 0.0
    for fit, score in ((in_a, ~in_a), (~in_a, in_a)):
        da, db = a[fit] - b[fit], y[fit] - b[fit]
        w = float(np.dot(da, db) / np.dot(da, da))
        e = w * a[score] + (1 - w) * b[score] - y[score]
        sse += float(np.sum(e ** 2))
        out.append(w)
    return dict(weights_per_fold=out, crossfit_rmse=float(np.sqrt(sse / len(y))),
                fixed_half_rmse=rmse(0.5 * a + 0.5 * b - y))


def main():
    import argparse
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT, help="fresh output directory (default: the diagnosis run dir)")
    OUT = parser.parse_args().out
    OUT.mkdir(parents=True, exist_ok=False)
    v, tr, y, preds, edges, dsb_full, checks = load_inputs()
    hsa = add_derived(v, y, preds, edges)
    summary = {"step": "D2", "validation_points": int(len(v)), "validation_matrices": int(v.matrix_key.nunique()),
               "validation_pairs": int(v.pair_id.nunique()), "test_rows_read": 0, "alignment_checks": checks}

    # ---- overall metrics -------------------------------------------------------------------------------------
    overall = group_metrics(y, preds, v.matrix_key.to_numpy())
    overall.to_csv(OUT / "overall_metrics.csv", index=False)
    bs = [rmse(preds[k] - y) for k in BS_SEEDS]
    xg = [rmse(preds[k] - y) for k in XGB_SEEDS]
    summary["overall"] = {
        "HSA": rmse(preds["HSA"] - y), "Bliss": rmse(preds["Bliss"] - y),
        "BS_add_per_seed": dict(zip(BS_SEEDS, bs)), "BS_add_mean_of_seed_rmse": float(np.mean(bs)),
        "BS_add_seed_sd": float(np.std(bs, ddof=1)), "BS_ens5": rmse(preds["BS_ens5"] - y),
        "BS_anch_s41": rmse(preds["BS_anch_s41"] - y),
        "XGB_per_seed": dict(zip(XGB_SEEDS, xg)), "XGB_mean_of_seed_rmse": float(np.mean(xg)),
        "XGB_seed_sd": float(np.std(xg, ddof=1)), "XGB_ens5": rmse(preds["XGB_ens5"] - y),
        "DSB_e484": rmse(preds["DSB_e484"] - y), "DSB_e047": rmse(preds["DSB_e047"] - y)}
    boot = pair_bootstrap(y, preds, v.pair_id.to_numpy(), {
        "BS_add(5 seeds) vs DSB_e484": (BS_SEEDS, ["DSB_e484"]),
        "XGB(5 seeds) vs DSB_e484": (XGB_SEEDS, ["DSB_e484"]),
        "BS_add(5 seeds) vs XGB(5 seeds)": (BS_SEEDS, XGB_SEEDS),
        "BS_ens5 vs DSB_e484": (["BS_ens5"], ["DSB_e484"]),
        "XGB_ens5 vs DSB_e484": (["XGB_ens5"], ["DSB_e484"]),
        "BS_ens5 vs BS_add(5 seeds)": (["BS_ens5"], BS_SEEDS),
        "DSB_e047 vs DSB_e484": (["DSB_e047"], ["DSB_e484"]),
        "BS_add(5 seeds) vs DSB_e047": (BS_SEEDS, ["DSB_e047"]),
        "DSB_e484 vs HSA": (["DSB_e484"], ["HSA"]),
        "BS_add(5 seeds) vs HSA": (BS_SEEDS, ["HSA"]),
        "DSB_e484_reanchor_HSA vs DSB_e484": (["DSB_e484_reanchor_HSA"], ["DSB_e484"]),
        "DSB_e484_reanchor_Bliss vs DSB_e484": (["DSB_e484_reanchor_Bliss"], ["DSB_e484"])})
    boot.to_csv(OUT / "bootstrap_comparisons.csv", index=False)

    # ---- DeepSynBa's own monotherapy edges vs the lookup anchors ---------------------------------------------
    t_edges = edges["truth"]
    mono_rows = {}
    first = ~v.duplicated(["matrix_key", "row_i"]).to_numpy()      # one row per (matrix, drug-A dose)
    first_b = ~v.duplicated(["matrix_key", "col_j"]).to_numpy()
    for tag in ("DSB_e484", "DSB_e047"):
        e = edges[tag]
        da = e["mono_a"][first] - t_edges["mono_a"][first]
        db = e["mono_b"][first_b] - t_edges["mono_b"][first_b]
        ctrl = e["control"][~v.duplicated("matrix_key").to_numpy()] - 100.0
        mono_rows[tag] = dict(edge_rmse_vs_own_matrix=rmse(np.concatenate([da, db])), control_rmse_vs_100=rmse(ctrl),
                              edge_bias=float(np.mean(np.concatenate([da, db]))),
                              ownHSA_vs_lookupHSA_rmse=rmse(preds[f"{tag}_ownHSA"] - hsa))
    la = v.mono_a_percent.to_numpy(float)[first] - t_edges["mono_a"][first]
    lb = v.mono_b_percent.to_numpy(float)[first_b] - t_edges["mono_b"][first_b]
    mono_rows["lookup"] = dict(edge_rmse_vs_own_matrix=rmse(np.concatenate([la, lb])),
                               share_edges_exact_1e4=float(np.mean(np.abs(np.concatenate([la, lb])) < 1e-4)))
    summary["monotherapy_edges"] = mono_rows

    # ---- slice features --------------------------------------------------------------------------------------
    fine_edges = [-np.inf, 10, 30, 60, 80, 90, 100, np.inf]
    fine_labels = ["<10", "10-30", "30-60", "60-80", "80-90", "90-100", ">=100"]
    coarse = bins(hsa, [-np.inf, 10, 30, 60, np.inf], ["<10", "10-30", "30-60", ">=60"])
    below = np.where(y < hsa, "y<HSA", "y>=HSA")
    dev = y - hsa
    drug_count = pd.concat([tr.drug_a_id, tr.drug_b_id]).value_counts()
    partners = pd.concat([tr[["drug_a_id", "drug_b_id"]].rename(columns={"drug_a_id": "d", "drug_b_id": "p"}),
                          tr[["drug_b_id", "drug_a_id"]].rename(columns={"drug_b_id": "d", "drug_a_id": "p"})])
    partner_count = partners.drop_duplicates().groupby("d").p.nunique()
    dc_a = v.drug_a_id.map(drug_count).to_numpy(float)
    dc_b = v.drug_b_id.map(drug_count).to_numpy(float)
    pc_a = v.drug_a_id.map(partner_count).to_numpy(float)
    pc_b = v.drug_b_id.map(partner_count).to_numpy(float)
    train_pairs = set(tr.pair_id)
    pair_in_train = v.pair_id.isin(train_pairs).to_numpy()
    cell_drug = pd.concat([tr[["deepsynba_cell", "drug_a_id"]].rename(columns={"drug_a_id": "d"}),
                           tr[["deepsynba_cell", "drug_b_id"]].rename(columns={"drug_b_id": "d"})]).value_counts().to_dict()
    cd_a = np.array([cell_drug.get(k, 0) for k in zip(v.deepsynba_cell, v.drug_a_id)], float)
    cd_b = np.array([cell_drug.get(k, 0) for k in zip(v.deepsynba_cell, v.drug_b_id)], float)
    pair_cells = v.groupby("pair_id").matrix_key.nunique()
    q_min = np.unique(np.quantile(np.minimum(dc_a, dc_b), [0, .2, .4, .6, .8, 1.0]))
    q_cd = np.unique(np.quantile(np.minimum(cd_a, cd_b), [0, .2, .4, .6, .8, 1.0]))
    q_pa = np.unique(np.quantile(np.minimum(pc_a, pc_b), [0, .2, .4, .6, .8, 1.0]))
    lookup = pd.read_csv(S1 / "lookup_split1_exact.csv", dtype={"cell": str, "drug": str, "dose_key": str})
    lk = {(c, d, k) for c, d, k in zip(lookup.cell, lookup.drug, lookup.dose_key)}
    dk = lambda x: f"{float(x):.6g}"  # noqa: E731  same key rule as s1_prepare
    exact_a = np.array([(c, d, dk(x)) in lk for c, d, x in zip(v.deepsynba_cell, v.deepsynba_drug_a, v.dose_a_M)])
    exact_b = np.array([(c, d, dk(x)) in lk for c, d, x in zip(v.deepsynba_cell, v.deepsynba_drug_b, v.dose_b_M)])

    def qbin(x, q):
        return pd.cut(x, q, include_lowest=True).astype(str)

    slices = {
        "dose_position(i,j)": (v.row_i.astype(str) + "," + v.col_j.astype(str)).to_numpy(),
        "dose_level_drugA(i)": v.row_i.to_numpy(),
        "dose_level_drugB(j)": v.col_j.to_numpy(),
        "max_dose_level": np.maximum(v.row_i, v.col_j).to_numpy(),
        "anchor_HSA_fine": bins(hsa, fine_edges, fine_labels),
        "anchor_HSA_coarse": coarse,
        "anchor_coarse_x_sign": pd.Series(coarse).str.cat(pd.Series(below), sep="|").to_numpy(),
        "observed_y_fine": bins(y, fine_edges, fine_labels),
        "true_dev_y_minus_HSA": bins(dev, [-np.inf, -30, -10, -3, 3, 10, np.inf],
                                     ["<-30", "-30..-10", "-10..-3", "-3..3", "3..10", ">=10"]),
        "drug_train_matrices_min_of_pair": qbin(np.minimum(dc_a, dc_b), q_min),
        "drug_train_partners_min_of_pair": qbin(np.minimum(pc_a, pc_b), q_pa),
        "cell_drug_train_matrices_min": qbin(np.minimum(cd_a, cd_b), q_cd),
        "pair_val_cells": qbin(v.pair_id.map(pair_cells).to_numpy(float),
                               np.unique(np.quantile(v.pair_id.map(pair_cells), [0, .25, .5, .75, 1.0]))),
        "pair_in_train": np.where(pair_in_train, "yes", "no"),
        "anchor_source": np.where(exact_a & exact_b, "both_exact", "fallback"),
        "tissue": v.biological_cell_id.str.split(":").str[0].to_numpy(),
        "cell": v.deepsynba_cell.to_numpy(),
    }
    tables = [slice_table(lab, y, hsa, preds, name) for name, lab in slices.items()]
    # per-drug (not a partition: every point belongs to two drugs)
    e2bs = np.mean([(preds[k] - y) ** 2 for k in BS_SEEDS], axis=0)
    gap = e2bs - (preds["DSB_e484"] - y) ** 2
    drug_rows = []
    for d in sorted(set(v.drug_a_id) | set(v.drug_b_id)):
        m = ((v.drug_a_id == d) | (v.drug_b_id == d)).to_numpy()
        drug_rows.append(dict(drug_id=d, name=(v.deepsynba_drug_a[v.drug_a_id == d].iloc[0] if (v.drug_a_id == d).any()
                                               else v.deepsynba_drug_b[v.drug_b_id == d].iloc[0]),
                              points=int(m.sum()), train_matrices=int(drug_count.get(d, 0)),
                              rmse_BS=float(np.sqrt(e2bs[m].mean())), rmse_DSB=float(np.sqrt(((preds["DSB_e484"] - y) ** 2)[m].mean())),
                              rmse_XGB=float(np.sqrt(np.mean([(preds[k] - y) ** 2 for k in XGB_SEEDS], axis=0)[m].mean())),
                              gap_sse=float(gap[m].sum()), share_gap_half_attributed=float(0.5 * gap[m].sum() / gap.sum())))
    pd.DataFrame(drug_rows).sort_values("gap_sse", ascending=False).to_csv(OUT / "per_drug.csv", index=False)
    pv = pd.DataFrame({"pair_id": v.pair_id, "drug_a": v.deepsynba_drug_a, "drug_b": v.deepsynba_drug_b,
                       "e2_BS": e2bs, "e2_DSB": (preds["DSB_e484"] - y) ** 2, "e2_HSA": (hsa - y) ** 2,
                       "e2_XGB": np.mean([(preds[k] - y) ** 2 for k in XGB_SEEDS], axis=0), "dev": y - hsa,
                       "corr_BS": preds["BS_ens5"] - hsa, "corr_DSB": preds["DSB_e484"] - hsa, "gap": gap})
    per_pair = pv.groupby("pair_id").agg(drug_a=("drug_a", "first"), drug_b=("drug_b", "first"), points=("gap", "size"),
                                         mse_BS=("e2_BS", "mean"), mse_DSB=("e2_DSB", "mean"), mse_HSA=("e2_HSA", "mean"),
                                         mse_XGB=("e2_XGB", "mean"), true_mean_dev=("dev", "mean"),
                                         corr_BS_ens5=("corr_BS", "mean"), corr_DSB=("corr_DSB", "mean"), gap_sse=("gap", "sum"))
    for k in ("BS", "DSB", "HSA", "XGB"):
        per_pair[f"rmse_{k}"] = np.sqrt(per_pair.pop(f"mse_{k}"))
    per_pair["share_gap"] = per_pair.gap_sse / gap.sum()
    per_pair.sort_values("gap_sse", ascending=False).to_csv(OUT / "per_pair.csv")
    allslices = pd.concat(tables, ignore_index=True)
    allslices.to_csv(OUT / "slices.csv", index=False)
    summary["pair_in_train_points"] = int(pair_in_train.sum())
    summary["anchor_fallback_points"] = int((~(exact_a & exact_b)).sum())
    summary["drug_train_matrices_range"] = [float(drug_count.min()), float(drug_count.max())]

    # ---- error decomposition & concentration -----------------------------------------------------------------
    members = BS_SEEDS + XGB_SEEDS + ["BS_ens5", "XGB_ens5", "DSB_e484", "DSB_e047", "HSA"]
    dec = decomposition(y, preds, v, members)
    comp = {}
    for part in ("pair_level", "matrix_level", "within_matrix", "total"):
        bsv = np.mean([dec[k][part] for k in BS_SEEDS])
        xgv = np.mean([dec[k][part] for k in XGB_SEEDS])
        comp[part] = dict(BS_mean_seeds=bsv, BS_ens5=dec["BS_ens5"][part], XGB_mean_seeds=xgv, DSB_e484=dec["DSB_e484"][part],
                          DSB_e047=dec["DSB_e047"][part], HSA=dec["HSA"][part])
    tot_gap = comp["total"]["BS_mean_seeds"] - comp["total"]["DSB_e484"]
    tot_gap_ens = comp["total"]["BS_ens5"] - comp["total"]["DSB_e484"]
    tot_gap_x = comp["total"]["XGB_mean_seeds"] - comp["total"]["DSB_e484"]
    for part in ("pair_level", "matrix_level", "within_matrix"):
        c = comp[part]
        c["share_gap_BS_vs_DSB"] = (c["BS_mean_seeds"] - c["DSB_e484"]) / tot_gap
        c["share_gap_BSens_vs_DSB"] = (c["BS_ens5"] - c["DSB_e484"]) / tot_gap_ens
        c["share_gap_XGB_vs_DSB"] = (c["XGB_mean_seeds"] - c["DSB_e484"]) / tot_gap_x
        for k in ("BS_mean_seeds", "BS_ens5", "XGB_mean_seeds", "DSB_e484", "DSB_e047", "HSA"):
            c[f"{k}_rmse_contrib"] = float(np.sqrt(c[k] / len(y)))
    summary["error_decomposition_sse"] = comp
    summary["concentration_matrix"] = concentration(v.matrix_key.to_numpy(), gap)
    summary["concentration_pair"] = concentration(v.pair_id.to_numpy(), gap)
    summary["concentration_cell"] = concentration(v.deepsynba_cell.to_numpy(), gap)

    # ---- deviation-from-HSA calibration ----------------------------------------------------------------------
    cal_members = ["BS_ens5", "BS_add_s42", "BS_anch_s41", "XGB_ens5", "DSB_e484", "DSB_e047", "DSB_e484_reanchor_HSA"]
    summary["deviation_calibration_all"] = calibration(y, hsa, preds, cal_members)
    hi_below = (hsa >= 60) & (y < hsa)
    summary["deviation_calibration_high_anchor_below"] = calibration(y, hsa, preds, cal_members, hi_below)

    # ---- complementarity -------------------------------------------------------------------------------------
    pairs = v.pair_id.to_numpy()
    summary["blend"] = {
        "BS_ens5+DSB_e484": blend_crossfit(y, preds["BS_ens5"], preds["DSB_e484"], pairs),
        "XGB_ens5+DSB_e484": blend_crossfit(y, preds["XGB_ens5"], preds["DSB_e484"], pairs),
        "BS_ens5+XGB_ens5": blend_crossfit(y, preds["BS_ens5"], preds["XGB_ens5"], pairs),
        "DSB_e484_reanchor_HSA+DSB_e484": blend_crossfit(y, preds["DSB_e484_reanchor_HSA"], preds["DSB_e484"], pairs)}
    err = {k: preds[k] - y for k in ("BS_ens5", "XGB_ens5", "DSB_e484", "BS_add_s41", "BS_add_s42")}
    summary["error_correlation"] = {f"{a}~{b}": float(np.corrcoef(err[a], err[b])[0, 1])
                                    for a, b in itertools.combinations(err, 2)}

    # ---- seed ensemble ----------------------------------------------------------------------------------------
    ens = {}
    for k in range(1, 6):
        vals = [rmse(np.mean([preds[s] for s in sub], axis=0) - y) for sub in itertools.combinations(BS_SEEDS, k)]
        ens[k] = dict(n_subsets=len(vals), mean=float(np.mean(vals)), min=float(np.min(vals)), max=float(np.max(vals)))
    stack = np.stack([preds[s] for s in BS_SEEDS])
    spread = float(np.mean(np.var(stack, axis=0, ddof=0)))          # mean per-seed MSE - ensemble MSE (exact)
    mse_seeds = float(np.mean([np.mean((preds[s] - y) ** 2) for s in BS_SEEDS]))
    mse_ens = float(np.mean((preds["BS_ens5"] - y) ** 2))
    sigma2 = spread * 5 / 4                                           # unbiased per-seed variance
    xstack = np.stack([preds[s] for s in XGB_SEEDS])
    summary["seed_ensemble"] = {
        "bisurf_k_subsets": ens, "mean_seed_mse": mse_seeds, "ens5_mse": mse_ens, "between_seed_variance_mean": spread,
        "identity_check": mse_seeds - mse_ens - spread,
        "extrapolated_infinite_seed_rmse": float(np.sqrt(max(mse_ens - sigma2 / 5, 0.0))),
        "ens6_incl_anchored_rmse": rmse(np.mean([preds[s] for s in BS_SEEDS + ["BS_anch_s41"]], axis=0) - y),
        "ens_best3_by_val_rmse": rmse(np.mean([preds[s] for s in sorted(BS_SEEDS, key=lambda s: rmse(preds[s] - y))[:3]], axis=0) - y),
        "xgb_between_seed_variance_mean": float(np.mean(np.var(xstack, axis=0, ddof=0))),
        "xgb_ens5_rmse": rmse(preds["XGB_ens5"] - y)}

    # ---- training curves --------------------------------------------------------------------------------------
    curves = {}
    for tag, path in BISURF.items():
        hist = json.loads((path / "training_progress.json").read_text(encoding="utf-8"))["history"]
        curves[tag] = [dict(epoch=h["epoch"], val_pooled=h["pooled_rmse_percent"], val_macro=h["surface_macro_rmse"],
                            train_loss=h.get("train_batch_mean_loss")) for h in hist]
    pd.DataFrame([dict(run=k, **r) for k, rows in curves.items() for r in rows]).to_csv(OUT / "bisurf_training_curves.csv", index=False)
    late = {}
    for tag, rows in curves.items():
        ep = np.array([r["epoch"] for r in rows if r["epoch"] >= 10], float)
        vp = np.array([r["val_pooled"] for r in rows if r["epoch"] >= 10], float)
        tl = np.array([r["train_loss"] for r in rows if r["epoch"] >= 10], float)
        late[tag] = dict(val_pooled_slope_per_epoch_ep10_on=float(np.polyfit(ep, vp, 1)[0]),
                         train_loss_ratio_last_over_ep10=float(tl[-1] / tl[0]),
                         epoch_to_epoch_val_sd_ep10_on=float(np.std(np.diff(vp), ddof=1)),
                         best_epoch=int(min(rows, key=lambda r: r["val_macro"])["epoch"]), epochs_run=int(max(ep)) + 1)
    summary["bisurf_late_training"] = late
    summary["bisurf_mean_epoch_seconds"] = None
    walls = []
    for path in BISURF.values():
        hist = json.loads((path / "training_progress.json").read_text(encoding="utf-8"))["history"]
        walls += [h["wall_seconds"] for h in hist if "wall_seconds" in h]
    summary["bisurf_mean_epoch_seconds"] = float(np.mean(walls))
    summary["bisurf_steps_per_epoch_approx"] = float(tr.matrix_key.nunique() / 128)
    summary["train_matrices"] = int(tr.matrix_key.nunique())

    with (OUT / "summary.json").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, indent=1, ensure_ascii=False, default=float) + "\n")
    print(json.dumps({k: summary[k] for k in ("overall", "monotherapy_edges", "error_decomposition_sse", "seed_ensemble", "blend")},
                     indent=1, ensure_ascii=False, default=float))
    print(boot.to_string())


if __name__ == "__main__":
    main()
