"""O'Neil stage 2 comboKR 2.0 (DEVIATIONS D13 supplement), INSIDE the container: Hill fits, train BRAID targets, kernels.

Inputs (/mnt/inputs, from prep_inputs.py): mono_curves.csv, surfaces_train.npz (16 combination doses + 100*response).
Validation and test responses are not available here and nothing is fitted to them.

1. Hill per (batch|cell, drug) on the same-batch monotherapy curve (8 doses, mean of finite replicates), with the
   author's MyHill and synergy 0.5.1 Hill.fit. Bounds = the defaults MyBRAID hands to its own single-drug MyHill fits
   (E0 90-110, Emax 0-120, C 1e-16-1 M) except h in [0.5, 4.8], the range of the author's published NCI-ALMANAC Hill
   fits (Zenodo almanac_fitted_hills.pkl: h min 0.5, max 4.8). With the default h >= 1e-12, 30/1,482 O'Neil contexts
   converged to the degenerate flat fit h = 1e-12, whose BRAID residuals are non-finite (smoke test, D13 supplement
   note). Initial guess = MyHill's [100, min E, 1, median d].
   synergy silently returns the initial guess when curve_fit fails (converged=False). Such contexts are refitted with
   curve_fit's evaluation budget raised (max_nfev=1e5), then also without the Jacobian (synergy's use_jacobian=False);
   every attempt is counted, and if all three fail the run stops (no fallback).
   Stored in the author's hills format (cell, drug, C, h, Emax); the model uses E0 = 100 for every Hill.
2. BRAID per train surface: the author's only published BRAID fitting routine,
   ScalableComboKR._fit_braid_to_normalised_grid_prediction (Hill-centred bounds from _get_braid_e_bounds with
   error 10 -> 20 -> 30 on ValueError, kappa in [-1.5, 10], MyBRAID initial guess, least_squares on
   braid_model_with_log_c_input), run on the 16 measured combination doses instead of the Hill-normalised grid.
   Vectorisation is Fortran order, so preds[-1] (E3 bound) is the max-max dose as on the author grid.
   Failed fits and fits whose Nystrom grid sample stays non-finite after the author's fix_Ytr are dropped from training.
3. Kernels as experiments/model_improvement/combokr2_original/build_kernels.py: drugs MACCS (RDKit) Tanimoto from the
   canonical SMILES of drugs.csv; cells RBF on the 968 z-scored DepMap genes, gamma = 1/(2 sigma^2), sigma = mean
   pairwise Euclidean distance over the 30 cells; the comboKR cell unit batch|cell copies its cell's row.

  bash experiments/oneil/combokr2/drun.sh python -I /mnt/scripts/fit_inputs.py [--max-train N]
"""
import argparse
import contextlib
import io
import json
import multiprocessing as mp
import os
import sys
import time
import types
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, "/mnt/code")
warnings.filterwarnings("ignore")  # as in the author's demo.py
from comboKR2.braid_surface_model_for_fit import MyHill, MyBraidWithOptimisers, braid_model_with_log_c_input  # noqa
from comboKR2.braid_surface_model_for_fit import braid_model_with_raw_c_input  # noqa: E402
from comboKR2.helpers import mat_to_vec  # noqa: E402
from comboKR2.scalable_comboKR import ScalableComboKR, SurfaceKernelNystromDifference  # noqa: E402

IN, OUT = "/mnt/inputs", "/out/fits"
HILL_BOUNDS = dict(E0_bounds=(90, 110), Emax_bounds=(0, 120), h_bounds=(0.5, 4.8), C_bounds=(1e-16, 1))
MAXRESP = types.SimpleNamespace(max_responseval=100)  # max_is_100=True
G = {}


def fit_hill(d, e):
    out = {}
    for attempt, (jac, kw) in enumerate([(True, {}), (True, {"max_nfev": 100000}), (False, {"max_nfev": 100000})], 1):
        m = MyHill(**HILL_BOUNDS)
        with contextlib.redirect_stdout(io.StringIO()):
            m.fit(d, e, use_jacobian=jac, **kw)
        out.update(converged=bool(m.converged), attempt=attempt)
        if m.converged:
            E0, Emax, h, C = m.get_parameters()
            pred = m.E(d)
            out.update(E0=E0, Emax=Emax, h=h, C=C, fit_rmse=float(np.sqrt(np.mean((pred - e) ** 2))))
            break
    return out


def fit_braid(h1_np, h2_np, c1, c2, y):
    """Body of ScalableComboKR._fit_braid_to_normalised_grid_prediction with measured doses (c1, c2) and responses y."""
    for error in [10, 20, 30]:
        try:
            bb = ScalableComboKR._get_braid_e_bounds(MAXRESP, h1_np, h2_np, y, error=error)
            braid = MyBraidWithOptimisers(E0_bounds=bb["E0_bounds"], E1_bounds=bb["E1_bounds"], E2_bounds=bb["E2_bounds"],
                                          E3_bounds=bb["E3_bounds"], h1_bounds=bb["h1_bounds"], h2_bounds=bb["h2_bounds"],
                                          C1_bounds=bb["C1_bounds"], C2_bounds=bb["C2_bounds"], kappa_bounds=[-1.5, 10])
            x0_tofit = braid._get_initial_guess(c1, c2, mat_to_vec(y))
            p = braid.fit_with_scipy_least_squares(np.vstack((c1[None, :], c2[None, :])), mat_to_vec(y),
                                                   braid_model_with_log_c_input, x0_tofit, loss="linear")
            return np.asarray(braid._transform_params_from_fit(p), float), error
        except ValueError:
            pass
    return None, None


def _braid_worker(i):
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    s = G["s"]
    u, a, b = s["cdd"][i]
    h = G["hills"]
    h1, h2 = [100, *h[(u, a)]], [100, *h[(u, b)]]  # [E0, Emax, h, C]
    with contextlib.redirect_stdout(io.StringIO()):
        p, err = fit_braid(h1, h2, s["c1"][i], s["c2"][i], s["y"][i])
    if p is None:
        return i, np.full(9, np.nan), -1, np.nan, False
    surf = braid_model_with_raw_c_input([s["c1"][i], s["c2"][i]], *p)
    rmse = float(np.sqrt(np.mean((surf - s["y"][i]) ** 2)))
    nys = G["nys"]
    grid = nys.fix_Ytr(nys._get_samples_on_grid(p[None, :]), int(np.sqrt(len(nys.c1))))
    return i, p, err, rmse, bool(np.isfinite(grid).all() and np.isfinite(p).all())


def build_kernels(units):
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import MACCSkeys
    from sklearn.metrics.pairwise import pairwise_distances, rbf_kernel
    RDLogger.DisableLog("rdApp.*")
    drugs = pd.read_csv("/mnt/drug/drugs.csv")
    drug_ids = np.array(sorted(drugs.drug_id.astype(str)))
    smi = dict(zip(drugs.drug_id.astype(str), drugs.canonical_smiles))
    fps = [MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(smi[d])) for d in drug_ids]
    Kd = np.array([DataStructs.BulkTanimotoSimilarity(f, fps) for f in fps])
    g = pd.read_csv("/mnt/cell/oneil_landmark_gex.csv", index_col=0)
    X = g.T.to_numpy(float)  # 30 cells x 968 genes, already z-scored per gene over the 30 cells
    X = X[:, X.std(0) > 0]
    sigma = float(np.mean(pairwise_distances(X)))
    gamma = 1.0 / (2.0 * sigma ** 2)
    K30 = rbf_kernel(X, gamma=gamma)
    pos = {c: i for i, c in enumerate(g.columns)}
    cell_ids = np.array(sorted(units))
    ix = [pos[u.split("|", 1)[1]] for u in cell_ids]
    Kc = K30[np.ix_(ix, ix)]
    off = ~np.eye(len(Kd), dtype=bool)
    audit = dict(n_drugs=len(drug_ids), n_cells=len(g.columns), n_units=len(cell_ids), n_genes_used=int(X.shape[1]),
                 rbf_sigma_mean_pairwise_distance=sigma, rbf_gamma=gamma, Kd_shape=list(Kd.shape), Kc_shape=list(Kc.shape),
                 Kd_offdiag_mean=float(Kd[off].mean()), Kd_identical_pairs=int(((Kd > 0.999999) & off).sum() // 2),
                 Kd_min_eig=float(np.linalg.eigvalsh(Kd).min()), Kc30_offdiag_mean=float(K30[~np.eye(30, dtype=bool)].mean()),
                 Kc_min_eig=float(np.linalg.eigvalsh(Kc).min()))
    return dict(Kc=Kc, Kd=Kd, cell_ids=cell_ids, drug_ids=drug_ids), audit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-train", type=int, default=None, help="smoke only: random subset of train surfaces")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=False)
    t0 = time.time()
    audit = dict(args=vars(args), hill_bounds={k: list(v) for k, v in HILL_BOUNDS.items()})

    mono = pd.read_csv(f"{IN}/mono_curves.csv")
    rows = []
    for (u, d), f in mono.groupby(["unit", "drug"], sort=True):
        rows.append(dict(cell=u, drug=d, n_doses=len(f), **fit_hill(f.dose_M.to_numpy(float), 100.0 * f.response_ratio.to_numpy(float))))
    hf = pd.DataFrame(rows)
    hf.to_csv(f"{OUT}/hill_fits.csv", index=False)
    if not hf.converged.all():
        raise SystemExit(f"Hill fit failed for {int((~hf.converged).sum())} contexts even without the Jacobian: "
                         + hf.loc[~hf.converged, ["cell", "drug"]].to_string())
    hf[["cell", "drug", "C", "h", "Emax"]].to_pickle(f"{OUT}/hills.pkl")
    audit["hill"] = dict(contexts=len(hf), converged_by_attempt=hf.attempt.value_counts().sort_index().to_dict(), failed=0,
                         describe=hf[["E0", "Emax", "h", "C", "fit_rmse"]].describe().to_dict(),
                         h_at_bound=int(((hf.h < 0.5 + 1e-6) | (hf.h > 4.8 - 1e-6)).sum()), Emax_at_bound=int(((hf.Emax < 1e-6) | (hf.Emax > 120 - 1e-6)).sum()),
                         seconds=time.time() - t0)
    print("hill", json.dumps({k: v for k, v in audit["hill"].items() if k != "describe"}), flush=True)

    s = dict(np.load(f"{IN}/surfaces_train.npz"))
    n_all = len(s["cdd"])
    if args.max_train:
        sel = np.sort(np.random.default_rng(0).choice(n_all, size=args.max_train, replace=False))
        s = {k: v[sel] for k, v in s.items()}
    G.update(s=s, hills={(c, d): (e, h, C) for c, d, e, h, C in hf[["cell", "drug", "Emax", "h", "C"]].itertuples(index=False)},
             nys=SurfaceKernelNystromDifference(tol=1e-1, kernel="rbf", max_is_100=True))
    t1 = time.time()
    with mp.get_context("fork").Pool(args.workers) as pool:
        res = sorted(pool.imap_unordered(_braid_worker, range(len(s["cdd"])), chunksize=20))
    params = np.vstack([r[1] for r in res])
    err = np.array([r[2] for r in res])
    rmse = np.array([r[3] for r in res])
    ok = np.array([r[4] for r in res])
    np.savez(f"{OUT}/train_braids.npz", params=params, error_level=err, fit_rmse=rmse, ok=ok,
             matrix_key=s["matrix_key"], cdd=s["cdd"])
    audit["braid"] = dict(train_surfaces=int(n_all), fitted_surfaces=len(res), ok=int(ok.sum()),
                          failed_all_error_levels=int((err == -1).sum()),
                          nonfinite_grid_after_fix_Ytr=int(((err != -1) & ~ok).sum()),
                          error_level_counts={str(k): int(v) for k, v in zip(*np.unique(err, return_counts=True))},
                          fit_rmse_percent=dict(mean=float(np.nanmean(rmse)), median=float(np.nanmedian(rmse)),
                                                p95=float(np.nanpercentile(rmse, 95)), max=float(np.nanmax(rmse))),
                          seconds=time.time() - t1)
    print("braid", json.dumps(audit["braid"]), flush=True)

    units = set(pd.read_csv(f"{IN}/mono_curves.csv", usecols=["unit"]).unit)
    ker, audit["kernels"] = build_kernels(units)
    np.savez(f"{OUT}/kernels.npz", **ker)
    print("kernels", json.dumps(audit["kernels"]), flush=True)
    audit["total_s"] = time.time() - t0
    with open(f"{OUT}/fit_audit.json", "w") as f:
        json.dump(audit, f, indent=1, default=float)


if __name__ == "__main__":
    main()
