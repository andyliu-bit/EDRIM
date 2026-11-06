"""SparseMono-v1 summary (validation predictions only; hidden truth only for reconstruction metrics).

Usage: python summarize_sparse_mono.py <run_dir> [<run_dir> ...] --compare CAND:REF[,CAND:REF...]
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[3]
TABLES = ROOT / "reports/tables"
JF = ROOT / "runs/model_improvement/joint_feature_v1"
CLUSTERS = None  # optional target drug -> held-out cluster (set by --clusters)


def per_surface(df, pred, mask):
    f = pd.DataFrame({"s": df.surface_id[mask], "d": df.target_drug[mask], "e": (pred[mask] - df.target[mask]) ** 2})
    g = f.groupby("s").agg(e=("e", "mean"), d=("d", "first"))
    return pd.DataFrame({"rmse": np.sqrt(g.e), "drug": g.d})


def collect(runs):
    rows, surf = [], {}
    for run in runs:
        for job in sorted(Path(run).glob("*_primary")):
            m = json.loads((job / "metrics.json").read_text())
            p = pd.read_csv(job / "predictions.csv")
            key = (m["pipeline"], m["k"])
            ps = per_surface(p, p.prediction.to_numpy(), p.primary_query.to_numpy())
            surf.setdefault(key, {})[m["seed"]] = ps
            rows.append(dict(pipeline=m["pipeline"], k=m["k"], seed=m["seed"], primary_rmse=m["primary_rmse"],
                             all_points_rmse=m["all_points_rmse"], recon_rmse_hidden_query=m["recon_rmse_hidden_query"], run=Path(run).name))
    return pd.DataFrame(rows), surf


def full_info_reference(query):
    """Previous round's best full-monotherapy model (JointFeature C) on the SAME query points."""
    out = {}
    for d in sorted(JF.glob("snap_*/runs/model_improvement/*/bisurf_joint_pleall_rmse_seed_*_primary")):
        seed = int(d.name.split("_seed_")[1].split("_")[0])
        p = pd.read_csv(d / "predictions.csv").merge(query, left_on="source_row", right_on="source_row")
        out.setdefault(seed, per_surface(p, p.prediction.to_numpy(), p.primary_query.to_numpy()))
    return out


def compare(surf, cand, ref, iters=2000, seed=20261010):
    c, r = surf[cand], surf[ref]
    common = sorted(set(c) & set(r))
    cm = np.array([c[s].rmse.mean() for s in common]); rm = np.array([r[s].rmse.mean() for s in common])
    diff = rm - cm; n = len(common)
    rad = float(student_t.ppf(.975, n - 1) * diff.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    ca = pd.concat([c[s].rmse for s in common], axis=1).mean(1); ra = pd.concat([r[s].rmse for s in common], axis=1).mean(1)
    drug = c[common[0]].drug.loc[ca.index]
    by = pd.DataFrame({"c": ca, "r": ra, "d": drug}).groupby("d").agg(c=("c", "sum"), r=("r", "sum"), n=("c", "size"))
    rng = np.random.default_rng(seed); draws = []
    for _ in range(iters):
        pick = by.iloc[rng.integers(0, len(by), len(by))]
        cr, rr = pick.c.sum() / pick.n.sum(), pick.r.sum() / pick.n.sum(); draws.append((rr - cr) / rr)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    per_drug = ((by.r - by.c) / by.r)
    extra = {}
    if CLUSTERS is not None:  # resample held-out chemical clusters (drugs of a cluster move together)
        bc = by.assign(k=by.index.map(CLUSTERS)).groupby("k").agg(c=("c", "sum"), r=("r", "sum"), n=("n", "sum"))
        rng2 = np.random.default_rng(seed + 1); cd = []
        for _ in range(iters):
            pick = bc.iloc[rng2.integers(0, len(bc), len(bc))]
            cr, rr = pick.c.sum() / pick.n.sum(), pick.r.sum() / pick.n.sum(); cd.append((rr - cr) / rr)
        clo, chi = np.percentile(cd, [2.5, 97.5])
        extra = dict(cluster_bootstrap_rel_ci95_low=float(clo), cluster_bootstrap_rel_ci95_high=float(chi), clusters=len(bc))
    return dict(candidate=f"{cand[0]}_k{cand[1]}", reference=f"{ref[0]}_k{ref[1]}", paired_seeds=n, candidate_mean=float(cm.mean()),
                reference_mean=float(rm.mean()), relative_improvement=float((rm.mean() - cm.mean()) / rm.mean()),
                improved_seeds=int((cm < rm).sum()), seed_ci95_low=float(diff.mean() - rad), seed_ci95_high=float(diff.mean() + rad),
                drug_bootstrap_rel_ci95_low=float(lo), drug_bootstrap_rel_ci95_high=float(hi), drugs=len(by),
                drugs_improved=int((per_drug > 0).sum()), **extra)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--compare", default=""); ap.add_argument("--prefix", default="sparse_mono_v1"); ap.add_argument("--clusters", default=None)
    a = ap.parse_args()
    global CLUSTERS
    if a.clusters:
        cl = pd.read_csv(a.clusters); CLUSTERS = dict(zip(cl.target_drug, cl.cluster))
    rows, surf = collect(a.runs)
    dup = rows[rows.duplicated(["pipeline", "k", "seed"], keep=False)]
    for (pl, k, s), g in dup.groupby(["pipeline", "k", "seed"]):  # same seed in two runs must reproduce exactly
        if g.primary_rmse.max() - g.primary_rmse.min() > 1e-9:
            raise RuntimeError(f"Cross-run mismatch {pl} k{k} seed {s}")
    print("cross-run duplicate seeds identical:", sorted(set(map(tuple, dup[["pipeline", "k", "seed"]].values.tolist()))))
    rows = rows.drop_duplicates(["pipeline", "k", "seed"])
    rows.sort_values(["pipeline", "k", "seed"]).to_csv(TABLES / f"{a.prefix}_per_seed.csv", index=False)
    agg = rows.groupby(["pipeline", "k"]).agg(seeds=("seed", "nunique"), primary_mean=("primary_rmse", "mean"), primary_sd=("primary_rmse", "std"),
                                              recon_mean=("recon_rmse_hidden_query", "mean"), all_points_mean=("all_points_rmse", "mean")).reset_index()
    agg.to_csv(TABLES / f"{a.prefix}_models.csv", index=False); print(agg.round(6).to_string())
    any_job = next(Path(a.runs[0]).glob("*_primary"))
    q = pd.read_csv(any_job / "predictions.csv")[["source_row", "surface_id", "target_drug", "primary_query"]].rename(columns={"surface_id": "s2"})
    fi = full_info_reference(q.drop(columns="s2"))
    if fi:
        surf[("full_mono_jointC", -1)] = fi
        print("full-monotherapy reference (JointFeature C) on same query points:", round(float(np.mean([v.rmse.mean() for v in fi.values()])), 6), "seeds", sorted(fi))
    if a.compare:
        out = []
        for pair in a.compare.split(","):
            cs, rs = pair.split(":")
            parse = lambda x: (x.rsplit("_k", 1)[0], int(x.rsplit("_k", 1)[1]))
            out.append(compare(surf, parse(cs), parse(rs)))
        out = pd.DataFrame(out); out.to_csv(TABLES / f"{a.prefix}_comparisons.csv", index=False); print(out.round(6).T.to_string())


if __name__ == "__main__":
    main()
