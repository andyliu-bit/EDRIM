"""Post-hoc evaluation of saved comboKR 2.0 runs on all ground-truth definitions (run INSIDE the container).

The pre-image output of predict_with_gd_nadam_and_braid is a BRAID parameter vector per test surface; the
author evaluates it with braid_model_with_raw_c_input at the test concentrations. We re-evaluate the saved
parameters with the same function on (a) the 4x4 response-matrix grid and (b) the 3x3 combination grid, and
the author's neutral baseline (predict_with_neutral_braid arithmetic) on the same grids.

Usage: python -I posthoc_eval.py /out/<run_dir> [/out/<run_dir> ...]
"""
import glob
import json
import os
import pickle
import sys
import warnings

import numpy as np

sys.path.insert(0, "/mnt/code")
sys.path.insert(0, "/mnt/scripts")
warnings.filterwarnings("ignore")
from comboKR2.braid_surface_model_for_fit import braid_model_with_raw_c_input  # noqa: E402
from run_combokr2 import load_braids, metrics  # noqa: E402

Z = "/mnt/zenodo"
I = "/out/inputs"
cdd = np.load(f"{Z}/almanac_cv_splits/almanac_cell_drug_drug_all.npy")
m4 = np.load(f"{I}/almanac_measurements_4x4.npz")
C1, C2, Y16, combo_pos = m4["c1"], m4["c2"], m4["y_nzcalc"], m4["combo_pos"]
braids_all = load_braids(cdd)
with open(f"{Z}/almanac_fitted_braids_and_hills/almanac_fitted_hills.pkl", "rb") as f:
    hills = pickle.load(f)
hk = {(c, str(d)): (e, h, k) for c, d, e, h, k in zip(hills.cell, hills.drug, hills.Emax, hills.h, hills.C)}


def evaluate_params(params, idx):
    out = np.full((len(idx), 16), np.nan)
    for j, i in enumerate(idx):
        try:
            out[j] = braid_model_with_raw_c_input([C1[i], C2[i]], *params[j])
        except Exception:  # the author's fallback for failed fits
            out[j] = 100.0
    return out


def neutral(idx):
    out = np.full((len(idx), 16), np.nan)
    for j, i in enumerate(idx):
        cell, a, b = cdd[i]
        e1, h1, k1 = hk[(cell, a)]
        e2, h2, k2 = hk[(cell, b)]
        out[j] = braid_model_with_raw_c_input([C1[i], C2[i]], 100, e1, e2, np.minimum(e1, e2), h1, h2, k1, k2, 0)
    return out


def author_fit(idx):
    out = np.full((len(idx), 16), np.nan)
    for j, i in enumerate(idx):
        if not np.isnan(braids_all[i]).any():
            out[j] = braid_model_with_raw_c_input([C1[i], C2[i]], *braids_all[i])
    return out


def hsa_3x3(y16):
    """HSA from measured same-study monotherapy, growth scale: min(mono_a(ca), mono_b(cb)).

    4x4 grid index k = 4*ia + ib with ia, ib in {0 (zero dose), 1, 2, 3}."""
    out = []
    for ia in (1, 2, 3):
        for ib in (1, 2, 3):
            out.append(np.minimum(y16[:, 4 * ia], y16[:, ib]))
    return np.stack(out, axis=1)  # same order as combo_pos (5,6,7,9,10,11,13,14,15)


def bridge(p9, b9, h9, y9):
    """Our project's metric family on the author's split: surface-macro RMSE in [0,1] units, unclipped truth."""
    m = ~np.isnan(y9).any(1) & np.isfinite(p9).all(1) & np.isfinite(b9).all(1) & ~np.isnan(h9).any(1)
    y, p, b, h = y9[m], p9[m], b9[m], h9[m]
    rs = {nm: np.sqrt(np.mean((x - y) ** 2, axis=1)) for nm, x in [("combokr2", p), ("braid_neutral", b), ("hsa_measured_mono", h)]}
    gt = y > 100
    out = {"n_surfaces": int(m.sum()), "frac_points_gt100": float(gt.mean())}
    for nm, x in [("combokr2", p), ("braid_neutral", b), ("hsa_measured_mono", h)]:
        out[nm] = {"surface_macro_rmse_frac": float(rs[nm].mean() / 100), "pearson": float(np.corrcoef(y.ravel(), x.ravel())[0, 1]),
                   "point_rmse_gt100_frac": float(np.sqrt(np.mean((x[gt] - y[gt]) ** 2)) / 100) if gt.any() else None,
                   "point_rmse_le100_frac": float(np.sqrt(np.mean((x[~gt] - y[~gt]) ** 2)) / 100)}
    out["winrate_combokr2_lt_hsa"] = float(np.mean(rs["combokr2"] < rs["hsa_measured_mono"]))
    out["winrate_braid_neutral_lt_hsa"] = float(np.mean(rs["braid_neutral"] < rs["hsa_measured_mono"]))
    out["winrate_combokr2_lt_braid_neutral"] = float(np.mean(rs["combokr2"] < rs["braid_neutral"]))
    return out


def main(run_dir):
    info = json.load(open(f"{run_dir}/metrics.json"))
    files = sorted(glob.glob(f"{run_dir}/pred_iters*.npz"))
    te = np.load(files[0])["test_idx"]
    base = neutral(te)
    fit = author_fit(te)
    y = Y16[te]
    truths = {
        "raw4x4_notz_clip0_100": (np.clip(y, 0, 100), slice(None)),
        "raw4x4_notz": (y, slice(None)),
        "raw3x3_notz_clip0_100": (np.clip(y, 0, 100)[:, combo_pos], combo_pos),
        "author_braid_fit_4x4_clip0_100": (np.clip(fit, 0, 100), slice(None)),
        "author_braid_fit_3x3_clip0_100": (np.clip(fit, 0, 100)[:, combo_pos], combo_pos),
    }
    res = {"n_test": int(len(te)), "per_iters": {}}
    excl = None
    if info["args"]["scenario"] == "newcell":
        excl = cdd[te, 0] != "MDA-MB-468"
    for f in files:
        k = os.path.basename(f)[len("pred_iters"):-len(".npz")]
        d = np.load(f)
        assert np.array_equal(d["test_idx"], te)
        p16 = evaluate_params(d["params"], te)
        # consistency with the author's own 3x3 output saved by run_combokr2.py
        p9_saved = d["pred"]
        p9 = p16[:, combo_pos]
        ok = np.isfinite(p9_saved).all(1)
        out = {"max_abs_diff_vs_author_3x3_output": float(np.nanmax(np.abs(np.sort(p9[ok], 1) - np.sort(p9_saved[ok], 1))))}
        for tname, (yt, pos) in truths.items():
            pp, bb = p16[:, pos], base[:, pos]
            m = ~np.isnan(yt).any(1) & np.isfinite(pp).all(1) & np.isfinite(bb).all(1)
            out[tname] = metrics(pp, bb, yt, m)
            if excl is not None and (~excl).any():
                out[tname + "|excl_MDA-MB-468"] = metrics(pp, bb, yt, m & excl)
        h9 = hsa_3x3(y)
        out["bridge_unclipped_3x3"] = bridge(p16[:, combo_pos], base[:, combo_pos], h9, y[:, combo_pos])
        out["bridge_clipped_3x3"] = bridge(p16[:, combo_pos], base[:, combo_pos], np.clip(h9, 0, 100),
                                           np.clip(y[:, combo_pos], 0, 100))
        res["per_iters"][k] = out
        print(run_dir, "iters", k, "diff", out["max_abs_diff_vs_author_3x3_output"],
              {t: (round(out[t]["pearson_combokr2"], 4), round(out[t]["pearson_baseline"], 4),
                   round(out[t]["winrate_combokr2_lt_baseline"], 4)) for t in truths}, flush=True)
    with open(f"{run_dir}/posthoc_metrics.json", "w") as fh:
        json.dump(res, fh, indent=1)
    np.savez_compressed(f"{run_dir}/posthoc_baseline_4x4.npz", base=base, test_idx=te)


if __name__ == "__main__":
    for rd in sys.argv[1:]:
        main(rd)
