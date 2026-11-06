"""O'Neil stage 2 comboKR 2.0 (DEVIATIONS D13 supplement), INSIDE the container. Mirrors run_public.py (NCI D10 C2).

Author pipeline unchanged (run_combokr2.py: difference-surface Nystrom output map, both drug orders, CGKronRLS early
stopping, projected-gradient pre-image + BRAID fit, max_is_100=True). Only the data change:
  train = this project's BRAID fits of the train surfaces (fit_inputs.py, failed fits dropped), cell unit batch|cell;
  query = every validation surface (iterations {3,5,10}) or every test surface (selected iteration only), predicted at
          its own 16 combination doses (molar) in the point-table drug order. Query inputs are IDs and doses only.
Writes /out/<role>/pred_iters<k>.npz with pred (n, 16) in percent, Fortran order (k = (row_i-1) + 4*(col_j-1)).

  bash experiments/oneil/combokr2/drun.sh python -I /mnt/scripts/run_oneil.py --role validation --n-iters 3,5,10
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "/mnt/nci")
import run_combokr2 as rk  # noqa: E402

IN, FITS = "/mnt/inputs", "/out/fits"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True, choices=["validation", "test"])
    ap.add_argument("--n-iters", default="3,5,10")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--max-query", type=int, default=None, help="smoke only: first n query surfaces")
    args = ap.parse_args()
    out_dir = f"/out/{args.role}" + ("_smoke" if args.max_query else "")
    os.makedirs(out_dir, exist_ok=False)
    t0 = time.time()
    b = np.load(f"{FITS}/train_braids.npz")
    keep = b["ok"]
    cdd_tr, braids_tr = b["cdd"][keep], b["params"][keep]
    cdd_tr = np.vstack((cdd_tr, np.hstack((cdd_tr[:, 0][:, None], cdd_tr[:, 2][:, None], cdd_tr[:, 1][:, None]))))
    braids_tr = np.vstack((braids_tr, braids_tr[:, [0, 2, 1, 3, 5, 4, 7, 6, 8]]))
    q = np.load(f"{IN}/surfaces_{args.role}.npz")
    assert "y" not in q.files, "query inputs must not carry responses"
    n = args.max_query or len(q["cdd"])
    cdd_q, c1, c2, keys = q["cdd"][:n], q["c1"][:n], q["c2"][:n], q["matrix_key"][:n]
    ker = np.load(f"{FITS}/kernels.npz")
    Kc, Kd, cell_ids, drug_ids = ker["Kc"], ker["Kd"], ker["cell_ids"], ker["drug_ids"]
    hills = pd.read_pickle(f"{FITS}/hills.pkl")
    nys = rk.SurfaceKernelNystromDifference(tol=1e-1, kernel="rbf", max_is_100=True)
    nys.create_approximation()
    model = rk.ScalablecomboKRdiff(Kc, Kd, cell_ids, drug_ids, nys, hills=hills, max_is_100=True)
    iters = [int(x) for x in args.n_iters.split(",")]
    model.train(cdd_tr, braids_tr, iters[0])
    info = dict(args=vars(args), train_surfaces=int(len(b["ok"])), train_used=int(keep.sum()), train_doubled=int(len(cdd_tr)),
                query=int(n), nystrom_landmarks=int(nys.Wsqinv.shape[0]), timings={"train_s": time.time() - t0})
    rk.G.update(model=model, cdd_tst=cdd_q, c1=c1, c2=c2)
    idx_all = np.arange(n)
    for k in iters:
        model.maxiter = k
        t1 = time.time()
        raw = rk.parallel_predict_raw(model, cdd_q, args.workers)
        t_raw = time.time() - t1
        rk.G["raw"] = raw
        res = rk.run_pool(rk._gd_worker, idx_all, args.workers, args.chunk)
        pred = np.vstack([r[0] for r in res])
        params = np.vstack([r[1] for r in res])
        failed = int(sum(r[2] for r in res))
        np.savez_compressed(f"{out_dir}/pred_iters{k}.npz", pred=pred, params=params, matrix_key=keys)
        info["timings"][f"iters{k}"] = dict(raw_predict_s=t_raw, total_s=time.time() - t1)
        info[f"iters{k}_braid_fit_failed"] = failed  # author's fallback: neutral BRAID from the Hills
        info[f"iters{k}_nonfinite_surfaces"] = int((~np.isfinite(pred)).any(1).sum())
        print(f"iters={k} done {time.time() - t1:.0f}s failed {failed} nonfinite {info[f'iters{k}_nonfinite_surfaces']}",
              flush=True)
    info["timings"]["total_s"] = time.time() - t0
    with open(f"{out_dir}/info.json", "w") as f:
        json.dump(info, f, indent=1)
    print(json.dumps(info["timings"]), flush=True)


if __name__ == "__main__":
    main()
