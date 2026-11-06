"""AlgorithmSearch-v1 batch 2: monotherapy–dose branch with branch learning rate and weight EMA.

Training-procedure changes are part of each candidate's definition (protocol
amendment in runs/model_improvement/algorithm_search_v1/FROZEN_RECORD.md):
  * the branch parameters use lr x `branch_lr_multiplier`;
  * an exponential moving average of all parameters (decay per optimizer step)
    is the model that is evaluated, selected per epoch, checkpointed and reloaded.
Everything else (data path, IDs, guards, repeat checks) is the frozen pilot path.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import torch
import yaml
from threadpoolctl import threadpool_limits

from bisurfgnn import independent_graph_benchmark as graph
from bisurfgnn import learned_benchmark as learned
from bisurfgnn.models.search_v1b import SearchV1bBiSurfGNN

spec = importlib.util.spec_from_file_location("search_base", Path(__file__).with_name("run_algorithm_search.py"))
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
pilot = base.pilot

VARIANTS = {
    "bisurf_search_mdx_opt_v2": dict(contextual_branch=False, branch_lr_multiplier=10.0, ema_decay=0.99),
    "bisurf_search_mdx_opt_ctx_v2": dict(contextual_branch=True, branch_lr_multiplier=10.0, ema_decay=0.99),
}


def build(banks, spec_, seed, cfg):
    learned._seed_everything(seed)
    model = SearchV1bBiSurfGNN(banks.atom_feature_dim, banks.cell_feature_dim, 136,
                               contextual_branch=spec_["contextual_branch"], graph_layers=2,
                               attention_heads=4, dropout=0.1, monotherapy_upper_bound=3.0,
                               residual_monotherapy_input=True)
    learned._seed_everything(seed)
    return model.to(torch.device(cfg["device"]))


@torch.no_grad()
def ema_update(ema, model, decay):
    for (name, target), source in zip(ema.state_dict().items(), model.state_dict().values()):
        if target.dtype.is_floating_point:
            target.mul_(decay).add_(source, alpha=1.0 - decay)
        else:
            target.copy_(source)


def train_variant(tables, banks, cfg, seed, mono, out, deadline):
    del mono
    variant = out.name.split("_seed_")[0]
    if variant not in VARIANTS:
        raise ValueError(f"Unregistered batch-2 variant: {variant}")
    spec_ = VARIANTS[variant]
    bindings = base.check_reference(out, cfg, banks)
    model = build(banks, spec_, seed, cfg)
    device = torch.device(cfg["device"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    branch = {id(p) for p in model.branch_parameters()}
    lr = cfg["training"]["learning_rate"]
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if id(p) not in branch], "lr": lr},
        {"params": [p for p in model.parameters() if id(p) in branch], "lr": lr * spec_["branch_lr_multiplier"]},
    ], weight_decay=cfg["training"]["weight_decay"])
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    tr, va = tables["train"], tables["validation"]
    initial = base.predict(ema, va, banks, cfg)
    np.testing.assert_allclose(initial, np.minimum(va.mono_a.numpy(), va.mono_b.numpy()), atol=3e-6, rtol=3e-6)
    best = pilot.metric(va.target.numpy(), initial, va.surface_ids)["surface_macro_rmse"]
    state, best_epoch = pilot.frozen_state(ema), -1
    history = [{"epoch": -1, "surface_macro_rmse": best}]
    for epoch in range(cfg["training"]["epochs"]):
        model.train(); losses = []; start = time.monotonic()
        for idx in graph._surface_batch_indices(tr, cfg["training"]["batch_surfaces"], shuffle_seed=seed*1000003+epoch):
            if time.monotonic() > deadline:
                raise TimeoutError("Frozen search wall budget exceeded")
            optimizer.zero_grad(set_to_none=True)
            output = graph._model_forward(model, tr, banks, idx, device)
            loss = pilot.loss_equal_surface(output.prediction, tr.target[idx].to(device), tr.surface_ids[idx.numpy()])
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Non-finite search loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["gradient_clip"])
            if not bool(torch.isfinite(norm)):
                raise RuntimeError("Non-finite search gradient")
            optimizer.step(); ema_update(ema, model, spec_["ema_decay"]); losses.append(float(loss.detach()))
        raw = pilot.metric(va.target.numpy(), base.predict(model, va, banks, cfg), va.surface_ids)["surface_macro_rmse"]
        score = pilot.metric(va.target.numpy(), base.predict(ema, va, banks, cfg), va.surface_ids)["surface_macro_rmse"]
        if score < best - cfg["training"]["min_delta"]:
            best, state, best_epoch = score, pilot.frozen_state(ema), epoch
        history.append({"epoch": epoch, "surface_macro_rmse": score, "raw_weights_surface_macro_rmse": raw,
                        "train_batch_mean_mse": float(np.mean(losses)), "wall_seconds": time.monotonic() - start,
                        "branch_head_weight_norm": float(model.monotherapy_dose_branch.head.weight.detach().norm())})
        pilot.write_history(out, history, best_epoch=best_epoch, variant=variant, device=str(device))
        print(f"SEARCHB2 {variant} seed={seed} epoch={epoch} ema={score:.7f} raw={raw:.7f}", flush=True)
    ema.load_state_dict(state)
    prediction = base.predict(ema, va, banks, cfg)
    torch.save(state, out / "model.pt")
    reloaded = build(banks, spec_, seed, cfg).eval()
    reloaded.load_state_dict(torch.load(out / "model.pt", weights_only=True, map_location="cpu"))
    np.testing.assert_allclose(base.predict(reloaded, va, banks, cfg), prediction, atol=0, rtol=0)
    return prediction, {"best_epoch": best_epoch, "history": history, "variant": variant, **spec_,
        "parameters": sum(p.numel() for p in model.parameters()), "reference_ids": bindings,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(); config_path = args.config.resolve(); output = args.output_dir.resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not set(cfg["models"]) <= set(VARIANTS) or cfg["fold"] != 0 \
            or cfg["claim_scope"] != "engineering_pilot_no_performance_ranking":
        raise ValueError("Unregistered or non-development batch-2 configuration")
    allowed = ROOT / "runs/model_improvement"
    if not output.is_relative_to(allowed) or output == allowed:
        raise ValueError("Output must be a fresh experiment subdirectory")
    output.mkdir(parents=True, exist_ok=False)
    reference = ROOT / cfg["reference_run"]
    sources = [learned._file_record(reference / n) for n in ("train_ids.csv", "validation_ids.csv", "data_binding.json", "summary.json")]
    learned._write_json_atomic(output / "reference_binding.json", {"sources": sources, "written_before_response_loading": True})
    accesses = []; original = pd.read_parquet
    def guard(path, *a, **k):
        resolved = Path(path).resolve()
        if resolved.name.lower() in {"test.parquet", "points.parquet"}:
            raise RuntimeError(f"Forbidden response source: {resolved}")
        accesses.append(str(resolved)); return original(path, *a, **k)
    try:
        with threadpool_limits(limits=cfg["threads"]), patch.object(pd, "read_parquet", guard), \
                patch.object(pilot, "train_graph", train_variant):
            result = pilot.execute(output, cfg, config_path)
        for record in sources:
            if learned._file_record(Path(record["path"]))["sha256"] != record["sha256"]:
                raise RuntimeError("Reference run drifted during training")
        base.assess(output, cfg)
        print(json.dumps({k: v for k, v in result.items() if k != "results"}), flush=True)
    except Exception as error:
        learned._write_json_atomic(output / "failure.json", {"status": "failed", "error": repr(error)})
        learned._write_json_atomic(output / "progress.json", {"status": "failed", "error": repr(error)})
        raise
    finally:
        learned._write_json_atomic(output / "input_access.json", {"parquet_reads": accesses, "test_response_rows_read": 0})


if __name__ == "__main__":
    main()
