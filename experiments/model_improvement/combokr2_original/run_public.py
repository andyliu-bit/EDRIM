"""D10 C2 (inside the container): comboKR 2.0 on the DeepSynBa public new-combination split.

Author pipeline unchanged (run_combokr2.py: difference-surface Nystrom output map, both drug orders, CGKronRLS early
stopping, projected-gradient pre-image + BRAID fit, max_is_100=True). Only the data change:
  train = author BRAID fits of the split-1 triplets (prep_public_split.py; every split-1 matrix mapped);
  query = split-2 (role validation, iterations {3,5,10}) or the evaluable test frame (role test, selected iteration),
          predicted at the matrix's own 3x3 combination doses (molar), drug order as in the point table.
Writes /out/public_split/<role>/pred_iters<k>.npz with pred (n, 9) in percent.

  bash drun.sh python -I /mnt/scripts/run_public.py --role validation --n-iters 3,5,10
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, "/mnt/scripts")
import run_combokr2 as rk  # noqa: E402

P = "/out/public_split"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True, choices=["validation", "test"])
    ap.add_argument("--n-iters", default="3,5,10")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--max-query", type=int, default=None, help="smoke only: first n query matrices")
    args = ap.parse_args()
    out_dir = f"{P}/{args.role}" + ("_smoke" if args.max_query else "")
    os.makedirs(out_dir, exist_ok=False)
    t0 = time.time()
    cdd = np.load(f"{rk.Z}/almanac_cv_splits/almanac_cell_drug_drug_all.npy")
    tr = np.load(f"{P}/train.npz")["cdd_rows"]
    q = np.load(f"{P}/query_{args.role}.npz")
    cdd_q, c1, c2 = q["cdd"].astype(cdd.dtype), q["c1"], q["c2"]
    if args.max_query:
        cdd_q, c1, c2 = cdd_q[:args.max_query], c1[:args.max_query], c2[:args.max_query]
    ker = np.load(f"{rk.I}/kernels.npz")
    Kc, Kd, cell_ids, drug_ids = ker["Kc"], ker["Kd"], ker["cell_ids"], ker["drug_ids"]
    with open(f"{rk.Z}/almanac_fitted_braids_and_hills/almanac_fitted_hills.pkl", "rb") as f:
        hills = pickle.load(f)
    braids_all = rk.load_braids(cdd)
    cdd_tr, braids_tr = cdd[tr], braids_all[tr, :]
    cdd_tr = np.vstack((cdd_tr, np.hstack((cdd_tr[:, 0][:, None], cdd_tr[:, 2][:, None], cdd_tr[:, 1][:, None]))))
    braids_tr = np.vstack((braids_tr, braids_tr[:, [0, 2, 1, 3, 5, 4, 7, 6, 8]]))
    nys = rk.SurfaceKernelNystromDifference(tol=1e-1, kernel="rbf", max_is_100=True)
    nys.create_approximation()
    model = rk.ScalablecomboKRdiff(Kc, Kd, cell_ids, drug_ids, nys, hills=hills, max_is_100=True)
    iters = [int(x) for x in args.n_iters.split(",")]
    model.train(cdd_tr, braids_tr, iters[0])
    info = dict(args=vars(args), train_triplets=int(len(tr)), train_doubled=int(len(cdd_tr)), query=int(len(cdd_q)),
                train_braid_nan_rows=int(np.isnan(braids_tr).any(1).sum()), timings={"train_s": time.time() - t0})
    rk.G.update(model=model, cdd_tst=cdd_q, c1=c1, c2=c2)
    idx_all = np.arange(len(cdd_q))
    for k in iters:
        model.maxiter = k
        t1 = time.time()
        raw = rk.parallel_predict_raw(model, cdd_q, args.workers)
        rk.G["raw"] = raw
        res = rk.run_pool(rk._gd_worker, idx_all, args.workers, args.chunk)
        pred = np.vstack([r[0] for r in res])
        params = np.vstack([r[1] for r in res])
        failed = int(sum(r[2] for r in res))
        np.savez_compressed(f"{out_dir}/pred_iters{k}.npz", pred=pred, params=params, matrix_key=q["matrix_key"][:len(cdd_q)])
        info["timings"][f"iters{k}_s"] = time.time() - t1
        info[f"iters{k}_braid_fit_failed"] = failed
        print(f"iters={k} done {time.time() - t1:.0f}s failed {failed}", flush=True)
    info["timings"]["total_s"] = time.time() - t0
    with open(f"{out_dir}/info.json", "w") as f:
        json.dump(info, f, indent=1)
    print(json.dumps(info["timings"]), flush=True)


if __name__ == "__main__":
    main()
