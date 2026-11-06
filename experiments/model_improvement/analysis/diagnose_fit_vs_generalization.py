"""Training- vs validation-set surface-macro RMSE for frozen AlgorithmSearch-v1 checkpoints.

Reloads saved checkpoints / boosters, recomputes predictions on the frozen sampled
training and validation tables, and reports both errors. No training, no test
access (same parquet guard as the runners). Validation numbers must reproduce
the saved predictions exactly; this is asserted.

Usage: python diagnose_fit_vs_generalization.py <run_dir>:<model>:<seed> ... --output OUT.csv
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import torch
import yaml

HERE = ROOT / "experiments/model_improvement"
def load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module
search = load("search_runner", "run_algorithm_search.py")
b2 = load("search_b2", "run_algorithm_search_b2.py")
pilot = search.pilot
learned, graph = search.learned, search.graph


def frozen_tables(cfg):
    from bisurfgnn.benchmark_config import load_benchmark_config
    from bisurfgnn.data.benchmark_context import load_benchmark_cell_context
    from bisurfgnn.data.cellminer import load_cellminer_config
    from bisurfgnn.data.chemical_splits import load_one_new_scaffold_benchmark
    data_root = (ROOT / cfg["data_root"]).resolve()
    benchmark = load_benchmark_config(data_root / cfg["benchmark_config"])
    cell = load_cellminer_config(data_root / cfg["cell_config"])
    split = load_one_new_scaffold_benchmark(benchmark)
    context = load_benchmark_cell_context(benchmark, cell)
    shards = learned.load_learned_point_shards(benchmark, verify_roles=("train", "validation"))
    full = {r: learned.load_learned_role_frame(shards, split.surfaces, 0, r, sample_seed=cfg["sampling_seed"])
            for r in ("train", "validation")}
    frames = pilot.sample_supported_frames(full["train"], full["validation"], split.surfaces, cfg)
    reference = ROOT / cfg["reference_run"]
    for role, frame in frames.items():
        ids = pd.read_csv(reference / f"{role}_ids.csv")
        assert (ids.source_combodrugseq.to_numpy() == frame.source_combodrugseq.to_numpy()).all(), role
    frame = pd.concat(frames.values())
    banks = graph.build_graph_feature_banks(sorted(set(frame.drug_a_id) | set(frame.drug_b_id)),
                                            sorted(set(frame.cell_id)), context, 0, benchmark)
    tables = {r: learned.build_compact_point_table(f, banks, role=r, fold=0, monotherapy_upper_bound=3.0)
              for r, f in frames.items()}
    return frames, banks, tables, pilot.morgan512(data_root, banks)


def neural(run, model, seed, tables, banks, cfg):
    path = ROOT / run / f"{model}_seed_{seed}_primary"
    if model in b2.VARIANTS:
        net = b2.build(banks, b2.VARIANTS[model], int(seed), cfg)
    else:
        net = search.build_model(banks, model, int(seed), cfg)
    net.load_state_dict(torch.load(path / "model.pt", weights_only=True, map_location="cpu"))
    return {r: search.predict(net, t, banks, cfg) for r, t in tables.items()}, path


def boosted(run, model, seed, frames, banks, fingerprints, cfg):
    import xgboost as xgb
    from sklearn.decomposition import PCA
    path = ROOT / run / f"{model}_seed_{seed}_primary"
    trees = json.loads((path / "metrics.json").read_text())["best_epoch"]
    tr = frames["train"]; xc = cfg["xgboost"]
    drug_index = {d: i for i, d in enumerate(banks.drug_ids)}; cell_index = {c: i for i, c in enumerate(banks.cell_ids)}
    fp = fingerprints.numpy().astype(float)
    drug_feat = PCA(xc["drug_pca"], random_state=0).fit(
        fp[[drug_index[d] for d in sorted(set(tr.drug_a_id) | set(tr.drug_b_id))]]).transform(fp)
    cell_vec = np.stack([g.node_features[:, 0].numpy() for g in banks.cell_graphs]).astype(float)
    cell_feat = PCA(xc["cell_pca"], random_state=0).fit(
        cell_vec[[cell_index[c] for c in sorted(set(tr.cell_id))]]).transform(cell_vec)
    booster = xgb.Booster(); booster.load_model(path / "model.json")
    out = {}
    for role, frame in frames.items():
        hsa = np.minimum(frame.mono_response_a, frame.mono_response_b).to_numpy(float)
        x = [xgb.DMatrix(search.xgb_features(frame, drug_feat, cell_feat, drug_index, cell_index, s)) for s in (False, True)]
        out[role] = hsa + 0.5 * sum(booster.predict(m, iteration_range=(0, trees)) for m in x)
    return out, path


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("jobs", nargs="+"); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/model_improvement/algorithm_search_v1/screen_b1.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")); cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.setdefault("xgboost", yaml.safe_load((ROOT / "configs/model_improvement/algorithm_search_v1/final_xgb_final.yaml").read_text())["xgboost"])
    original = pd.read_parquet
    def guard(path, *a, **k):
        if Path(path).name.lower() in {"test.parquet", "points.parquet"}:
            raise RuntimeError("Forbidden response source")
        return original(path, *a, **k)
    rows = []
    with patch.object(pd, "read_parquet", guard):
        frames, banks, tables, fingerprints = frozen_tables(cfg)
        hsa = {r: np.minimum(t.mono_a.numpy(), t.mono_b.numpy()) for r, t in tables.items()}
        for role, t in tables.items():
            rows.append(dict(job="HSA", role=role, surface_macro_rmse=pilot.metric(t.target.numpy(), hsa[role], t.surface_ids)["surface_macro_rmse"]))
        for job in args.jobs:
            run, model, seed = job.split(":")
            if model.startswith("xgb"):
                pred, path = boosted(run, model, seed, frames, banks, fingerprints, cfg)
            else:
                pred, path = neural(run, model, seed, tables, banks, cfg)
            saved = pd.read_csv(path / "predictions.csv").prediction.to_numpy()
            np.testing.assert_allclose(pred["validation"], saved, atol=1e-6, rtol=0)
            for role, t in tables.items():
                rows.append(dict(job=job, role=role, surface_macro_rmse=pilot.metric(t.target.numpy(), pred[role], t.surface_ids)["surface_macro_rmse"]))
    frame = pd.DataFrame(rows).pivot(index="job", columns="role", values="surface_macro_rmse")
    frame["generalization_gap"] = frame["validation"] - frame["train"]
    frame.to_csv(args.output); print(frame.to_string())


if __name__ == "__main__":
    main()
