"""Write the AlgorithmSearch-v1 stage configs from one frozen template."""
from __future__ import annotations

import copy
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "configs/model_improvement/algorithm_search_v1"

BASE = dict(
    protocol="AlgorithmSearch-v1", protocol_document="docs/ALGORITHM_SEARCH_V1_PROTOCOL.md",
    claim_scope="engineering_pilot_no_performance_ranking",
    reference_run="runs/model_improvement/comparator_gpu_v2", data_root="../bisurfgnn",
    benchmark_config="configs/benchmark_bisurf_unclipped.yaml", cell_config="configs/cell_context.yaml",
    fold=0, seeds=[41, 42], sampling_seed=41, sampling_salt="model-comparator-gpu-v2",
    preserve_partner_cell_support=True, train_max_points=18000, validation_max_points=9000,
    device="cuda", threads=2, max_wall_seconds=10800, auxiliary_weight=0.5,
    training=dict(epochs=40, mono_epochs=0, batch_size=512, evaluation_batch_size=256,
                  batch_surfaces=128, learning_rate=0.0003, weight_decay=0.0001,
                  gradient_clip=1.0, min_delta=0.00001),
    xgboost=dict(learning_rate=0.03, max_trees=3000, eval_every=50, subsample=0.8,
                 colsample=0.8, drug_pca=16, cell_pca=16, threads=6),
)
ALL_GRAPH = ["bisurf_mono_v2", "bisurf_search_mdx_v2", "bisurf_search_pr_v2", "bisurf_search_aux_v2",
             "bisurf_search_mdx_pr_v2", "bisurf_search_mdx_aux_v2", "bisurf_search_mdx_pr_aux_v2",
             "bisurf_search_pr_aux_v2", "bisurf_search_dcd_v2", "bisurf_search_dcd_aux_v2"]
STAGES = {
    "preflight": dict(models=ALL_GRAPH, seeds=[41], training=dict(epochs=2), max_wall_seconds=3600),
    "screen_b1": dict(models=["bisurf_mono_v2", "bisurf_search_mdx_v2", "bisurf_search_pr_v2",
                              "bisurf_search_aux_v2"]),
    "screen_b2_mdx": dict(models=["bisurf_search_mdx_pr_v2", "bisurf_search_mdx_aux_v2",
                                  "bisurf_search_mdx_pr_aux_v2"]),
    "screen_b2_fallback": dict(models=["bisurf_search_pr_aux_v2", "bisurf_search_dcd_v2",
                                       "bisurf_search_dcd_aux_v2"]),
    "xgb_d10": dict(models=["xgb_residual_d10_m1"], device="cpu", max_wall_seconds=7200),
}


def merge(base, update):
    result = copy.deepcopy(base)
    for key, value in update.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and key in result else value
    return result


def write(name, update):
    cfg = merge(BASE, update)
    cfg["protocol"] = f"AlgorithmSearch-v1-{name}"
    cfg["report_prefix"] = f"algosearch_v1_{name}"
    path = OUT / f"{name}.yaml"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite frozen config {path}")
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    names = sys.argv[1:] or list(STAGES)
    for name in names:
        if name.startswith("final_"):
            # final_<label>=model1,model2 ; seeds 41-45
            label, models = name.split("=")
            write(label, dict(models=models.split(","), seeds=[41, 42, 43, 44, 45],
                              **({"device": "cpu"} if models.startswith("xgb") else {})))
        else:
            write(name, STAGES[name])
