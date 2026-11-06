"""NCI60-Pretrain-v1 step 4: pretrain MonoPriorNet (SparseMono D_gnn architecture g1 or g2) on NCI-60.

  .venv/Scripts/python.exe experiments/model_improvement/nci60_pretrain/s4_pretrain.py \
      --config experiments/model_improvement/nci60_pretrain/configs/pretrain.yaml --arch g2 \
      --output-dir runs/model_improvement/nci60_pretrain_v1/pretrain_g2 [--benchmark-steps N]
Before touching the data the script asserts the frozen exclusion-list SHA-256 (protocol §3.3) and every input hash.
``--benchmark-steps`` = timing-only smoke run (no checkpoint kept for transfer).
"""
from __future__ import annotations

import argparse
import json
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from bisurfgnn.data.batch import pad_graph_items  # noqa: E402
from bisurfgnn.learned_benchmark import _seed_everything  # noqa: E402
from bisurfgnn.models.sparse_mono import MonoPriorNet  # noqa: E402
from bisurfgnn.pretrain.nci60 import GraphBank, append_log, forbid_test_responses, load_fold0_cell_graphs, sha256_file  # noqa: E402
from bisurfgnn.pretrain.trainer import PointTensors, evaluate, graph_hash, holdout_mask, pretrain  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True); ap.add_argument("--arch", choices=["g1", "g2"], required=True)
    ap.add_argument("--output-dir", type=Path, required=True); ap.add_argument("--benchmark-steps", type=int, default=0)
    a = ap.parse_args()
    cfg = yaml.safe_load(a.config.read_text(encoding="utf-8"))
    out = a.output_dir.resolve()
    if out.exists() or not out.is_relative_to(ROOT / "runs/model_improvement/nci60_pretrain_v1"):
        raise ValueError("Output must be a fresh subdirectory of runs/model_improvement/nci60_pretrain_v1")
    out.mkdir(parents=True)
    t_start = time.monotonic()
    # ---- frozen exclusion hash first (protocol §3.3)
    excl_path = ROOT / cfg["exclusion_list"]
    got = sha256_file(excl_path)
    frozen = json.loads((ROOT / cfg["exclusion_freeze"]).read_text(encoding="utf-8"))
    if got != cfg["exclusion_sha256"] or got != frozen["exclusion_list_sha256"]:
        raise RuntimeError(f"Exclusion list hash mismatch: {got}")
    for key in ("points", "graphs", "compounds"):
        h = sha256_file(ROOT / cfg[key])
        if h != cfg[f"{key}_sha256"]:
            raise RuntimeError(f"{key} hash mismatch {h}")
    sm_cfg = yaml.safe_load((ROOT / cfg["sparse_mono_config"]).read_text(encoding="utf-8"))
    hp = dict(sm_cfg["net_prior"][a.arch]); sched = dict(cfg["schedule"])
    if a.benchmark_steps:
        sched.update(max_steps=a.benchmark_steps, eval_every=max(1, a.benchmark_steps // 2), patience_evals=10 ** 9)
    dev = torch.device(cfg["device"]); torch.set_num_threads(cfg["threads"])
    torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    excluded = set(pd.read_csv(excl_path).nsc.astype(int))
    pts = pd.read_parquet(ROOT / cfg["points"], columns=["nsc", "cell", "dose_u", "y"])
    n_all = len(pts); pts = pts[~pts.nsc.isin(excluded)].reset_index(drop=True)
    if pts.nsc.isin(excluded).any():
        raise RuntimeError("Excluded compound survived filtering")
    bank, raw = GraphBank.load(ROOT / cfg["graphs"], device=dev)
    gindex = {int(n): i for i, n in enumerate(raw["nsc"].tolist())}
    if not set(pts.nsc.unique()) <= set(gindex):
        raise RuntimeError("Point compound without graph")
    with forbid_test_responses() as guard:
        graphs, _, _ = load_fold0_cell_graphs()
    cells = sorted(graphs); cindex = {c: i for i, c in enumerate(cells)}
    if not set(pts.cell.astype(str).unique()) <= set(cells):
        raise RuntimeError("Point cell without fold-0 graph")
    cell_items = [graphs[c] for c in cells]
    cell_in = pad_graph_items(cell_items).to(dev)
    comp = np.sort(pts.nsc.unique()); hmask = holdout_mask(comp, cfg["holdout_salt"], cfg["holdout_modulus"])
    hold_c = set(comp[hmask].tolist())
    is_hold = pts.nsc.isin(hold_c).to_numpy()
    tr_df, ho_df = pts[~is_hold], pts[is_hold]
    train, hold = PointTensors(tr_df, gindex, cindex, dev), PointTensors(ho_df, gindex, cindex, dev)
    # trivial references on the hold-out (no learning of compound structure)
    glob = float(tr_df.y.mean())
    b = lambda f: f.assign(b=np.round(f.dose_u * 2) / 2)
    cb = b(tr_df).groupby(["cell", "b"], observed=True).y.mean()
    hb = b(ho_df).join(cb.rename("ref"), on=["cell", "b"])
    hb["ref"] = hb.ref.fillna(glob)
    refs = dict(global_mean=float(np.sqrt(((ho_df.y - glob) ** 2).mean())), cell_dose_bin_mean=float(np.sqrt(((hb.y - hb.ref) ** 2).mean())))
    seed = int(cfg["seed"])
    _seed_everything(seed)
    model = MonoPriorNet("gnn", int(bank.vocab.shape[1]), int(cell_in.node_features.shape[-1]), hidden=hp["hidden"],
                         dropout=hp["dropout"], dose_mean=float(tr_df.dose_u.mean()), dose_sd=float(tr_df.dose_u.std()),
                         mono_mean=float(tr_df.y.mean())).to(dev)
    info = dict(arch=a.arch, hp=hp, schedule=sched, seed=seed, points_all=n_all, points_after_exclusion=len(pts),
                compounds_after_exclusion=int(len(comp)), train_points=len(tr_df), holdout_points=len(ho_df),
                train_compounds=int(len(comp) - hmask.sum()), holdout_compounds=int(hmask.sum()), cells=len(cells),
                holdout_reference_rmse=refs, exclusion_sha256=got, parquet_reads=guard.accesses,
                parameters=int(sum(p.numel() for p in model.parameters())), benchmark_steps=a.benchmark_steps)
    (out / "pretrain_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in info.items() if k != "parquet_reads"}), flush=True)
    n_tr, n_ho = len(tr_df), len(ho_df)
    del pts, tr_df, ho_df, hb, cb, raw, is_hold   # the device tensors hold everything needed; free ~3 GB of host RAM
    import gc; gc.collect()
    res = pretrain(model, bank, cell_in, train, hold, hp, sched, seed, out, log=lambda s: print(s, flush=True))
    res["holdout_rmse_reloaded"] = evaluate(model, bank, cell_in, hold)
    meta = dict(protocol="NCI60-Pretrain-v1", arch=a.arch, hidden=hp["hidden"], dropout=hp["dropout"],
                atom_feature_dim=int(bank.vocab.shape[1]), cell_feature_dim=int(cell_in.node_features.shape[-1]),
                best_step=res["best_step"], best_holdout_rmse=res["best_holdout_rmse"], exclusion_sha256=got,
                cell_graph_hashes={c: graph_hash(g) for c, g in zip(cells, cell_items)},
                dose_stats=[float(x) for x in model.dose_stats.tolist()])
    ckpt = dict(drug_encoder=model.drug_encoder.state_dict(), cell_encoder=model.cell_encoder.state_dict(),
                full_model=model.state_dict(), meta=meta)
    torch.save(ckpt, out / ("pretrained_encoders.pt" if not a.benchmark_steps else "benchmark_only.pt"))
    res.update(info=info, total_seconds=time.monotonic() - t_start,
               checkpoint_sha256=sha256_file(out / ("pretrained_encoders.pt" if not a.benchmark_steps else "benchmark_only.pt")))
    curve = res.pop("curve"); pd.DataFrame(curve).to_csv(out / "curve.csv", index=False)
    (out / "pretrain_metrics.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    if not a.benchmark_steps:
        append_log(ROOT / "runs/model_improvement/nci60_pretrain_v1/STEP_LOG.md",
                   f"STEP4 pretrain {a.arch}: best hold-out RMSE {res['best_holdout_rmse']:.5f} at step {res['best_step']} "
                   f"(stop {res['stop_reason']}, last step {res['last_step']}); references global {refs['global_mean']:.5f}, "
                   f"cell x dose-bin {refs['cell_dose_bin_mean']:.5f}; train {n_tr:,} pts / {info['train_compounds']:,} cpds, "
                   f"hold-out {n_ho:,} pts / {info['holdout_compounds']:,} cpds; {res['total_seconds'] / 60:.1f} min; "
                   f"checkpoint sha256 {res['checkpoint_sha256']}")
    print(f"DONE {a.arch} best {res['best_holdout_rmse']:.5f}@{res['best_step']} stop={res['stop_reason']} {res['total_seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
