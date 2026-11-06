"""Smoke 0: compiled RLScore CGKronRLS (generalized vec trick, C extension) vs explicit dense MINRES.

Same linear system as the author's _predict_raw: (R (Kc (x) Kd (x) Kd) R^T) a = y, regparam=0, early stopping
after `maxiter` MINRES iterations. Uses a small random subset of the ALMANAC new-cell rseed1 training triplets
and real Nystrom output features. Run INSIDE the container.
"""
import json
import sys
import time
import warnings

import numpy as np
from rlscore.learner import CGKronRLS
from scipy.sparse.linalg import minres

sys.path.insert(0, "/mnt/code")
sys.path.insert(0, "/mnt/scripts")
warnings.filterwarnings("ignore")
from comboKR2.scalable_comboKR import SurfaceKernelNystromDifference, ScalablecomboKRdiff  # noqa: E402
from run_combokr2 import load_braids  # noqa: E402

Z = "/mnt/zenodo"
ker = np.load("/out/inputs/kernels.npz")
Kc, Kd, cell_ids, drug_ids = ker["Kc"], ker["Kd"], ker["cell_ids"], ker["drug_ids"]
cdd = np.load(f"{Z}/almanac_cv_splits/almanac_cell_drug_drug_all.npy")
tr = np.load(f"{Z}/almanac_cv_splits/almanac_newcell_rseed1_tr_cdd_inds.npy")
te = np.load(f"{Z}/almanac_cv_splits/almanac_newcell_rseed1_tst_cdd_inds.npy")
rng = np.random.default_rng(0)
tr_s, te_s = np.sort(rng.choice(tr, 1500, replace=False)), np.sort(rng.choice(te, 300, replace=False))
braids = load_braids(cdd)
nys = SurfaceKernelNystromDifference(tol=1e-1, kernel="rbf", max_is_100=True)
nys.create_approximation()
import pickle  # noqa: E402
with open(f"{Z}/almanac_fitted_braids_and_hills/almanac_fitted_hills.pkl", "rb") as f:
    hills = pickle.load(f)
m = ScalablecomboKRdiff(Kc, Kd, cell_ids, drug_ids, nys, hills=hills, max_is_100=True)
m.train(cdd[tr_s], braids[tr_s], 5)
r1, r2 = m.mat1_inds, m.mat2_inds
q1, q2 = m._get_inds_from_cdd(cdd[te_s])
f1, f2, g1, g2 = r1.ravel(), r2.ravel(), q1.ravel(), q2.ravel()
K1 = np.kron(Kc, Kd)
nd = Kd.shape[0]
Ktr = K1[np.ix_(f1, f1)] * Kd[np.ix_(f2, f2)]
Kte = K1[np.ix_(g1, f1)] * Kd[np.ix_(g2, f2)]
# Nystrom landmark count sensitivity (paper reports 27 landmarks for eigenvalue threshold 0.1)
from comboKR2.scalable_comboKR import SurfaceKernelNystrom  # noqa: E402
from sklearn.metrics.pairwise import pairwise_distances, rbf_kernel  # noqa: E402
spec = {}
for cls, nm in [(SurfaceKernelNystromDifference, "difference"), (SurfaceKernelNystrom, "direct")]:
    for mx in [True, False]:
        o = cls(tol=1e-1, kernel="rbf", max_is_100=mx)
        o.create_approximation()
        spec[f"{nm}|max_is_100={mx}"] = int(o.Wsqinv.shape[0])
# eigenvalue spectrum of the generated landmark kernel (difference variant, max_is_100=True)
o = SurfaceKernelNystromDifference(tol=1e-1, kernel="rbf", max_is_100=True)
C2g, C1g = np.meshgrid(np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1))
o.create_approximation()
report_spec = {"landmarks": spec}
report = {"nystrom": report_spec, "n_train": int(len(tr_s)), "n_test": int(len(te_s)), "landmarks": int(m.Phiy_tr.shape[1]), "cases": []}
for maxiter in [3, 5, 10]:
    for col in [0, 1, m.Phiy_tr.shape[1] - 1]:
        y = m.Phiy_tr[:, col]
        t0 = time.time()
        lr = CGKronRLS(K1=K1, K2=Kd, Y=y, label_row_inds=r1, label_col_inds=r2, regparam=0, maxiter=maxiter)
        p_rl = lr.predict(K1, Kd, q1, q2)
        t_rl = time.time() - t0
        a, _ = minres(Ktr, y, maxiter=maxiter, tol=1e-20)
        p_dense = Kte @ a
        report["cases"].append({"maxiter": maxiter, "col": int(col),
                                "max_abs_diff": float(np.max(np.abs(p_rl - p_dense))),
                                "rel_diff": float(np.linalg.norm(p_rl - p_dense) / np.linalg.norm(p_dense)),
                                "rlscore_s": t_rl})
with open("/out/smoke/rlscore_vs_dense_minres.json", "w") as f:
    json.dump(report, f, indent=1)
print(json.dumps(report, indent=1))
