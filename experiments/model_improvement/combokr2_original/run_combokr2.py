"""comboKR 2.0 on NCI-ALMANAC in the authors' original setting (run INSIDE the container).

Follows the author's demo.py pipeline verbatim (difference-surface Nystrom output map with tol=0.1 and RBF
output kernel, training set doubled with both drug orders, ScalablecomboKRdiff with RLScore CGKronRLS early
stopping, projected-gradient (NAdam) pre-image followed by BRAID fitting), with max_is_100=True as the
author code prescribes for ALMANAC. The author's per-surface pre-image loop is unchanged; it is only run on
chunks of the test set in parallel worker processes, each receiving the precomputed raw predictions of the
author's own _predict_raw (CGKronRLS training + prediction) for its chunk.

Example:
  python -I /mnt/scripts/run_combokr2.py --scenario newcell --seed 1 --n-iters 5 --workers 11 --tag newcell_r1
"""
import argparse
import contextlib
import glob
import io
import json
import multiprocessing as mp
import os
import pickle
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, "/mnt/code")
warnings.filterwarnings("ignore")  # as in the author's demo.py
from comboKR2.scalable_comboKR import SurfaceKernelNystromDifference, ScalablecomboKRdiff  # noqa: E402
from comboKR2.braid_surface_model_for_fit import braid_model_with_raw_c_input  # noqa: E402

Z = "/mnt/zenodo"
I = "/out/inputs"
G = {}


def load_braids(cdd):
    frames = []
    for bf in sorted(glob.glob(f"{Z}/almanac_fitted_braids_and_hills/*_braid.df.pkl")):
        with open(bf, "rb") as f:
            df = pickle.load(f)
        df = df.copy()
        df["cell"] = os.path.basename(bf)[: -len("_braid.df.pkl")]
        frames.append(df)
    br = pd.concat(frames, ignore_index=True)
    key = {(c, str(a), str(b)): i for i, (c, a, b) in enumerate(zip(br["cell"], br["NSC1"], br["NSC2"]))}
    par = br[["E0", "E1", "E2", "E3", "h1", "h2", "C1", "C2", "kappa"]].to_numpy()
    return par[[key[tuple(r)] for r in cdd.tolist()]]


def split_indices(scen, seed, cv_fold):
    stem = f"{Z}/almanac_cv_splits/almanac_{scen}_rseed{seed}"
    if cv_fold is None:
        return np.load(f"{stem}_tr_cdd_inds.npy"), np.load(f"{stem}_tst_cdd_inds.npy")
    return np.load(f"{stem}_cvfold{cv_fold}_tr_cdd_inds.npy"), np.load(f"{stem}_cvfold{cv_fold}_tst_cdd_inds.npy")


def _gd_worker(task):
    k, idx = task
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(1)
    threadpool_limits(1)  # parent may use multi-threaded BLAS for the CGKronRLS phase; workers stay at 1
    model = G["model"]
    raw = G["raw"]
    model._predict_raw = lambda cdd, difference=False: raw[idx]  # author's raw predictions, precomputed
    buf = io.StringIO()
    t = time.time()
    with contextlib.redirect_stdout(buf):
        preds, params = model.predict_with_gd_nadam_and_braid(G["cdd_tst"][idx], G["c1"][idx], G["c2"][idx])
    m = re.search(r"FAILED TO FIT BRAID (\d+)/(\d+)", buf.getvalue())
    return k, preds, np.asarray(params, dtype=float), int(m.group(1)) if m else -1, time.time() - t


def _raw_worker(ii):
    """Body of the author's ScalableComboKR._predict_raw loop for output dimension ii (unchanged calls)."""
    from threadpoolctl import threadpool_limits
    from rlscore.learner import CGKronRLS
    threadpool_limits(1)
    m = G["model"]
    x1 = G["x1"]  # == np.kron(m.Kc, m.Kd), recomputed identically inside the author's loop
    learner = CGKronRLS(K1=x1, K2=m.Kd, Y=m.Phiy_tr[:, ii], label_row_inds=m.mat1_inds,
                        label_col_inds=m.mat2_inds, regparam=0, maxiter=m.maxiter)
    return ii, learner.predict(x1, m.Kd, G["q1"], G["q2"])


def parallel_predict_raw(model, cdd_tst, workers):
    with contextlib.redirect_stdout(io.StringIO()):
        q1, q2 = model._get_inds_from_cdd(cdd_tst)
    G.update(x1=np.kron(model.Kc, model.Kd), q1=q1, q2=q2)
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        res = dict(pool.imap_unordered(_raw_worker, range(model.Phiy_tr.shape[1])))
    G.pop("x1")
    return np.array([res[ii] for ii in range(model.Phiy_tr.shape[1])]).T


def _cand_worker(task):
    """Author's candidate-set pre-image (ScalablecomboKRdiff.predict_with_candidates), unchanged."""
    k, idx = task
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(1)
    threadpool_limits(1)
    model = G["model"]
    raw = G["raw"]
    model._predict_raw = lambda cdd, difference=False: raw[idx]
    t = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        preds, params = model.predict_with_candidates(G["cdd_tst"][idx], G["c1"][idx], G["c2"][idx])
    return k, np.asarray(preds), np.asarray(params, dtype=float), 0, time.time() - t


def _neutral_worker(task):
    k, idx = task
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    model = G["model"]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        preds, params = model.predict_with_neutral_braid(G["cdd_tst"][idx], G["c1"][idx], G["c2"][idx])
    return k, preds


def run_pool(fn, idx_all, workers, chunk):
    tasks = [(k, idx_all[s:s + chunk]) for k, s in enumerate(range(0, len(idx_all), chunk))]
    out = {}
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        for res in pool.imap_unordered(fn, tasks):
            out[res[0]] = res[1:]
    return [out[k] for k in range(len(tasks))]


def surface_rmse(p, y):
    return np.sqrt(np.mean((p - y) ** 2, axis=1))


def metrics(pred, base, y, mask):
    p, b, t = pred[mask], base[mask], y[mask]
    rp, rb = surface_rmse(p, t), surface_rmse(b, t)
    return {
        "n_surfaces": int(mask.sum()),
        "pearson_combokr2": float(np.corrcoef(t.ravel(), p.ravel())[0, 1]),
        "pearson_baseline": float(np.corrcoef(t.ravel(), b.ravel())[0, 1]),
        "spearman_combokr2": float(spearmanr(t.ravel(), p.ravel())[0]),
        "spearman_baseline": float(spearmanr(t.ravel(), b.ravel())[0]),
        "winrate_combokr2_lt_baseline": float(np.mean(rp < rb)),
        "winrate_baseline_lt_combokr2": float(np.mean(rb < rp)),
        "ties": float(np.mean(rp == rb)),
        "surface_macro_rmse_combokr2_frac": float(np.mean(rp) / 100.0),
        "surface_macro_rmse_baseline_frac": float(np.mean(rb) / 100.0),
        "point_rmse_combokr2_frac": float(np.sqrt(np.mean((p - t) ** 2)) / 100.0),
        "point_rmse_baseline_frac": float(np.sqrt(np.mean((b - t) ** 2)) / 100.0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="newcell", choices=["newcell", "newcombo", "newdrug"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cv-fold", type=int, default=None)
    ap.add_argument("--n-iters", default="5", help="comma-separated CGKronRLS early-stopping iterations")
    ap.add_argument("--workers", type=int, default=11)
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--max-test", type=int, default=None, help="random subsample of test surfaces (fixed rng)")
    ap.add_argument("--subsample-seed", type=int, default=0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--max-train", type=int, default=None, help="ablation: random subsample of training triplets")
    ap.add_argument("--train-subsample-seed", type=int, default=0)
    ap.add_argument("--preimage", default="gd", choices=["gd", "candidates"],
                    help="gd: predict_with_gd_nadam_and_braid (paper default); candidates: predict_with_candidates")
    ap.add_argument("--raw-mode", default="parallel", choices=["parallel", "author"],
                    help="author: sequential ScalableComboKR._predict_raw; parallel: same loop body per dimension")
    ap.add_argument("--check-raw-against", default=None, help="npz with 'raw' from an author-mode run")
    args = ap.parse_args()
    out_dir = f"/out/{args.tag}"
    os.makedirs(out_dir, exist_ok=True)
    T = {"start": time.time()}

    cdd = np.load(f"{Z}/almanac_cv_splits/almanac_cell_drug_drug_all.npy")
    tr, te = split_indices(args.scenario, args.seed, args.cv_fold)
    if args.max_test is not None and args.max_test < len(te):
        rng = np.random.default_rng(args.subsample_seed)
        te = np.sort(rng.choice(te, size=args.max_test, replace=False))
    if args.max_train is not None and args.max_train < len(tr):
        rng_tr = np.random.default_rng(args.train_subsample_seed)
        tr = np.sort(rng_tr.choice(tr, size=args.max_train, replace=False))
    ker = np.load(f"{I}/kernels.npz")
    Kc, Kd, cell_ids, drug_ids = ker["Kc"], ker["Kd"], ker["cell_ids"], ker["drug_ids"]
    with open(f"{Z}/almanac_fitted_braids_and_hills/almanac_fitted_hills.pkl", "rb") as f:
        hills = pickle.load(f)
    braids_all = load_braids(cdd)
    meas = np.load(f"{I}/almanac_measurements_v2.npz")
    c1_all, c2_all, ynz_all, ypg_all = meas["c1"], meas["c2"], meas["y_nzcalc"], meas["y_pg"]
    T["loaded"] = time.time()

    # ---- training data exactly as in demo.py: both drug orders --------------------------------------
    cdd_tr = cdd[tr]
    braids_tr = braids_all[tr, :]
    cdd_tr = np.vstack((cdd_tr, np.hstack((cdd_tr[:, 0][:, None], cdd_tr[:, 2][:, None], cdd_tr[:, 1][:, None]))))
    braids_tr2 = braids_tr[:, [0, 2, 1, 3, 5, 4, 7, 6, 8]]
    braids_tr = np.vstack((braids_tr, braids_tr2))
    cdd_tst, c1_tst, c2_tst = cdd[te], c1_all[te], c2_all[te]

    output_nystrom = SurfaceKernelNystromDifference(tol=1e-1, kernel="rbf", max_is_100=True)
    output_nystrom.create_approximation()
    model = ScalablecomboKRdiff(Kc, Kd, cell_ids, drug_ids, output_nystrom, hills=hills, max_is_100=True)
    iters = [int(x) for x in args.n_iters.split(",")]
    model.train(cdd_tr, braids_tr, iters[0])
    T["trained_features"] = time.time()
    info = {"args": vars(args), "n_train_triplets": int(len(tr)), "n_train_doubled": int(len(cdd_tr)),
            "n_test": int(len(te)), "nystrom_landmarks": int(output_nystrom.Wsqinv.shape[0]),
            "train_braid_nan_rows": int(np.isnan(braids_tr).any(1).sum()),
            "test_cells": sorted(set(cdd_tst[:, 0].tolist())), "timings": {}}
    print("landmarks", info["nystrom_landmarks"], "train doubled", len(cdd_tr), "test", len(te), flush=True)

    # ---- truths -----------------------------------------------------------------------------------------
    braid_truth = np.full((len(te), 9), np.nan)
    for j, i in enumerate(te):
        if not np.isnan(braids_all[i]).any():
            braid_truth[j] = braid_model_with_raw_c_input([c1_tst[j], c2_tst[j]], *braids_all[i])
    truths = {"raw_notz_clip0_100": np.clip(ynz_all[te], 0, 100), "raw_notz": ynz_all[te],
              "author_braid_fit_clip0_100": np.clip(braid_truth, 0, 100),
              "raw_pg_clip0_100": np.clip(ypg_all[te], 0, 100)}

    G.update(model=model, cdd_tst=cdd_tst, c1=c1_tst, c2=c2_tst)
    idx_all = np.arange(len(te))
    base = None
    if not args.skip_baseline:
        t0 = time.time()
        base = np.vstack([r[0] for r in run_pool(_neutral_worker, idx_all, args.workers, args.chunk)])
        info["timings"]["baseline_s"] = time.time() - t0
    results = {}
    for k in iters:
        model.maxiter = k
        t0 = time.time()
        if args.raw_mode == "author":
            with contextlib.redirect_stdout(io.StringIO()):
                raw = model._predict_raw(cdd_tst)
        else:
            raw = parallel_predict_raw(model, cdd_tst, args.workers)
        t_raw = time.time() - t0
        if args.check_raw_against:
            ref = np.load(args.check_raw_against)["raw"]
            info[f"raw_check_iters{k}"] = {"max_abs_diff": float(np.max(np.abs(ref - raw))),
                                           "max_abs_ref": float(np.max(np.abs(ref)))}
            print("raw check", info[f"raw_check_iters{k}"], flush=True)
        G["raw"] = raw
        t0 = time.time()
        res = run_pool(_gd_worker if args.preimage == "gd" else _cand_worker, idx_all, args.workers, args.chunk)
        t_gd = time.time() - t0
        pred = np.vstack([r[0] for r in res])
        params = np.vstack([r[1] for r in res])
        failed = int(sum(r[2] for r in res))
        cpu_gd = float(sum(r[3] for r in res))
        info["timings"][f"iters{k}"] = {"raw_predict_s": t_raw, "preimage_wall_s": t_gd,
                                        "preimage_worker_sum_s": cpu_gd,
                                        "preimage_s_per_surface_worker": cpu_gd / len(te)}
        out = {"failed_braid_fits": failed, "nonfinite_pred_rows": int((~np.isfinite(pred)).any(1).sum())}
        for tname, y in truths.items():
            m = ~np.isnan(y).any(1) & np.isfinite(pred).all(1)
            if base is not None:
                m &= np.isfinite(base).all(1)
                out[tname] = metrics(pred, base, y, m)
                if args.scenario == "newcell" and "MDA-MB-468" in info["test_cells"]:
                    m2 = m & (cdd_tst[:, 0] != "MDA-MB-468")
                    out[tname + "|excl_MDA-MB-468"] = metrics(pred, base, y, m2)
            else:
                out[tname] = {"pearson_combokr2": float(np.corrcoef(y[m].ravel(), pred[m].ravel())[0, 1])}
        results[f"iters{k}"] = out
        np.savez_compressed(f"{out_dir}/pred_iters{k}.npz", pred=pred, params=params, test_idx=te,
                            raw=raw)
        print(f"iters={k} raw {t_raw:.0f}s preimage {t_gd:.0f}s failed {failed}", json.dumps(
            {t: {kk: round(v, 4) for kk, v in out[t].items() if kk.startswith(("pearson", "winrate"))}
             for t in truths}), flush=True)
    if base is not None:
        np.savez_compressed(f"{out_dir}/baseline.npz", pred=base, test_idx=te)
    np.savez_compressed(f"{out_dir}/truths.npz", test_idx=te, **truths)
    info["timings"]["total_s"] = time.time() - T["start"]
    info["timings"]["load_s"] = T["loaded"] - T["start"]
    info["timings"]["train_features_s"] = T["trained_features"] - T["loaded"]
    info["results"] = results
    with open(f"{out_dir}/metrics.json", "w") as f:
        json.dump(info, f, indent=1)
    print(json.dumps(info["timings"], indent=1))


if __name__ == "__main__":
    main()
