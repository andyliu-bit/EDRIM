"""Validation-only remaining algorithm experiments using the established frozen pilot data path."""
from __future__ import annotations

import argparse
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
from scipy.stats import t as student_t
from threadpoolctl import threadpool_limits

from bisurfgnn import independent_graph_benchmark as graph
from bisurfgnn import learned_benchmark as learned
from bisurfgnn.models.residual_parameterization import ResidualParameterizationBiSurf
from bisurfgnn.models.factorial_bisurf import FactorialBiSurfGNN
from bisurfgnn.models.compact_bisurf import CompactBiSurfGNN

spec = importlib.util.spec_from_file_location("remaining_trial_pilot", Path(__file__).with_name("run_comparator_pilot.py"))
pilot = importlib.util.module_from_spec(spec); spec.loader.exec_module(pilot)
VARIANTS = {"bisurf_mono_v2": "base", "bisurf_additive_v2": "additive", "bisurf_gain4_v2": "logit_gain4"}
for exchange in ("pooled", "bixt", "cross_attention"):
    for fusion in ("concat", "gate"):
        if (exchange,fusion) != ("bixt","gate"):
            VARIANTS[f"bisurf_{exchange}_{fusion}_v2"] = (exchange,fusion)


def build_model(banks, mode, seed, cfg):
    learned._seed_everything(seed)
    kwargs = dict(graph_layers=2, attention_heads=4, dropout=0.1,
                  monotherapy_upper_bound=3.0, residual_monotherapy_input=True)
    if isinstance(mode, tuple):
        kwargs.pop("residual_monotherapy_input")
        model = FactorialBiSurfGNN(banks.atom_feature_dim, banks.cell_feature_dim, 136,
                                  exchange=mode[0], fusion=mode[1], **kwargs)
    elif mode == "base":
        model = CompactBiSurfGNN(banks.atom_feature_dim, banks.cell_feature_dim, 136, **kwargs)
    else:
        model = ResidualParameterizationBiSurf(banks.atom_feature_dim, banks.cell_feature_dim,
                                               136, output_mode=mode, **kwargs)
    learned._seed_everything(seed)
    return model.to(torch.device(cfg["device"]))


def predict(model, table, banks, cfg):
    return graph.predict_graph_compact(model, table, banks,
        batch_size=cfg["training"]["evaluation_batch_size"], device=torch.device(cfg["device"]))[0]


def train_variant(tables, banks, cfg, seed, mono, out, deadline):
    del mono
    variant = out.name.split("_seed_")[0]
    if variant not in VARIANTS:
        raise ValueError("Unregistered remaining variant")
    # Compare full point IDs, not just group counts. A changed task stops the run.
    reference = ROOT / cfg["reference_run"]
    bindings = []
    if cfg["fold"] == 0:
        for role in tables:
            old, current = reference / f"{role}_ids.csv", out.parent / f"{role}_ids.csv"
            record = learned._file_record(old)
            if learned._file_record(current)["sha256"] != record["sha256"]:
                raise RuntimeError(f"Point ID/order drift relative to GPU v2: {role}")
            bindings.append(record)
        old_binding = json.loads((reference / "data_binding.json").read_text())
        if banks.context_state_hash != old_binding["context_state_hash"]:
            raise RuntimeError("Training-fitted expression state drift")
        current_binding = json.loads((out.parent / "data_binding.json").read_text())
        current_inputs = {r["path"]: r["sha256"] for r in current_binding["inputs"]}
        if any(current_inputs.get(r["path"]) != r["sha256"] for r in old_binding["inputs"]):
            raise RuntimeError("Frozen data or preprocessing changed since GPU v2")
    model = build_model(banks, VARIANTS[variant], seed, cfg)
    device = torch.device(cfg["device"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"],
                                  weight_decay=cfg["training"]["weight_decay"])
    tr, va = tables["train"], tables["validation"]
    pd.DataFrame(dict(source_row=va.source_rows, surface_id=va.surface_ids,
                      cluster_id=va.cluster_ids)).to_csv(out.parent / "validation_units.csv", index=False)
    initial = predict(model, va, banks, cfg)
    np.testing.assert_allclose(initial, np.minimum(va.mono_a.numpy(), va.mono_b.numpy()), atol=3e-6, rtol=3e-6)
    best = pilot.metric(va.target.numpy(), initial, va.surface_ids)["surface_macro_rmse"]
    state, best_epoch = pilot.frozen_state(model), -1
    history = [{"epoch": -1, "surface_macro_rmse": best}]
    for epoch in range(cfg["training"]["epochs"]):
        model.train(); values = []; epoch_start = time.monotonic()
        for idx in graph._surface_batch_indices(tr, cfg["training"]["batch_surfaces"], shuffle_seed=seed*1000003+epoch):
            if time.monotonic() > deadline:
                raise TimeoutError("Frozen cell interaction development budget exceeded")
            optimizer.zero_grad(set_to_none=True)
            output = graph._model_forward(model, tr, banks, idx, device)
            loss = pilot.loss_equal_surface(output.prediction, tr.target[idx].to(device), tr.surface_ids[idx.numpy()])
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Non-finite cell interaction loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["gradient_clip"])
            if not bool(torch.isfinite(norm)):
                raise RuntimeError("Non-finite cell interaction gradient")
            optimizer.step(); values.append(float(loss.detach()))
        score = pilot.metric(va.target.numpy(), predict(model, va, banks, cfg), va.surface_ids)["surface_macro_rmse"]
        if score < best-cfg["training"]["min_delta"]:
            best, state, best_epoch = score, pilot.frozen_state(model), epoch
        history.append({"epoch": epoch, "surface_macro_rmse": score,
            "train_batch_mean_mse": float(np.mean(values)), "wall_seconds": time.monotonic()-epoch_start})
        pilot.write_history(out, history, best_epoch=best_epoch, variant=variant, device=str(device))
        print(f"REMAINING {variant} seed={seed} epoch={epoch} val={score:.7f}", flush=True)
    model.load_state_dict(state)
    prediction = predict(model, va, banks, cfg)
    torch.save(state, out / "model.pt")
    model.load_state_dict(torch.load(out / "model.pt", weights_only=True, map_location="cpu"))
    np.testing.assert_allclose(predict(model, va, banks, cfg), prediction, atol=0, rtol=0)
    return prediction, {"best_epoch": best_epoch, "history": history, "variant": variant,
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "reference_ids": bindings,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None}


def assess(output, cfg):
    if cfg.get("stage") == "preflight":
        return
    frame = pd.read_csv(output / "results.csv"); frame = frame[frame.run == "primary"]
    paired = frame.pivot(index="seed", columns="model", values="surface_macro_rmse")
    statistics = []
    for name in paired:
        values = paired[name].dropna(); n = len(values)
        mean, sd = float(values.mean()), float(values.std(ddof=1))
        radius = float(student_t.ppf(.975, n-1)*sd/np.sqrt(n))
        statistics.append(dict(model=name, seeds=n, mean=mean, sd=sd,
            seed_mean_ci95_low=mean-radius, seed_mean_ci95_high=mean+radius))
    statistics = pd.DataFrame(statistics)
    statistics.to_csv(output / "seed_statistics.csv", index=False)
    if cfg.get("report_prefix"):
        statistics.to_csv(ROOT / "reports/tables" / f"{cfg['report_prefix']}_seed_statistics.csv", index=False)
    decisions=[]
    base=paired.bisurf_mono_v2
    if len(paired)!=5 or paired.isna().any().any():
        raise RuntimeError("Development decision requires all five paired seeds")
    for name in paired:
        if name == "bisurf_mono_v2":
            continue
        full=paired[name]; difference=base-full
        radius=float(student_t.ppf(.975,len(difference)-1)*difference.std(ddof=1)/np.sqrt(len(difference)))
        reduction=float((base.mean()-full.mean())/base.mean())
        checks={"four_of_five_seeds_improve": int((full<base).sum())>=cfg["go_required_seeds"],
                "mean_relative_reduction_at_least_one_percent": reduction>=cfg["go_relative_improvement"]}
        decisions.append(dict(model=name, checks=checks, advance_to_multifold=all(checks.values()),
            relative_rmse_reduction_percent=100*reduction, improved_seeds=int((full<base).sum()),
            paired_seed_mean_rmse_reduction=float(difference.mean()),
            paired_seed_ci95_low=float(difference.mean()-radius), paired_seed_ci95_high=float(difference.mean()+radius)))
    learned._write_json_atomic(output/"development_decision.json", dict(
        scope=f"fixed_fold{cfg['fold']}_development_only", candidates=decisions,
        not_independent_confirmation=True,
        ci_scope="training_seed_variability_conditional_on_fixed_development_sample"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/model_improvement/remaining_suite_v1.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(); config_path = args.config.resolve(); output = args.output_dir.resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not set(cfg["models"]).issubset(VARIANTS) or "bisurf_mono_v2" not in cfg["models"] or cfg["fold"] not in range(5) or cfg["claim_scope"] != "engineering_pilot_no_performance_ranking":
        raise ValueError("Unregistered validation-only experiment variants")
    allowed = ROOT / "runs/model_improvement"
    if not output.is_relative_to(allowed) or output == allowed:
        raise ValueError("Output must be a fresh experiment subdirectory")
    output.mkdir(parents=True, exist_ok=False)
    reference = ROOT / cfg["reference_run"]
    sources = [learned._file_record(reference / name) for name in (
        "train_ids.csv", "validation_ids.csv", "data_binding.json", "summary.json")]
    learned._write_json_atomic(output / "reference_binding.json", {"sources": sources, "written_before_response_loading": True})
    accesses = []; original = pd.read_parquet
    def guard(path, *args, **kwargs):
        resolved = Path(path).resolve()
        if resolved.name.lower() in {"test.parquet", "points.parquet"}:
            raise RuntimeError(f"Forbidden response source: {resolved}")
        accesses.append(str(resolved)); return original(path, *args, **kwargs)
    try:
        with threadpool_limits(limits=cfg["threads"]), patch.object(pd, "read_parquet", guard), patch.object(pilot, "train_graph", train_variant):
            result = pilot.execute(output, cfg, config_path)
        for record in sources:
            if learned._file_record(Path(record["path"]))["sha256"] != record["sha256"]:
                raise RuntimeError("Reference run drifted during training")
        assess(output, cfg)
        print(json.dumps({k: v for k, v in result.items() if k != "results"}), flush=True)
    except Exception as error:
        learned._write_json_atomic(output / "failure.json", {"status": "failed", "error": repr(error)})
        learned._write_json_atomic(output / "progress.json", {"status": "failed", "error": repr(error)})
        if (output / "summary.json").exists():
            summary = json.loads((output / "summary.json").read_text())
            summary.update(status="failed", postprocessing_error=repr(error))
            learned._write_json_atomic(output / "summary.json", summary)
        raise
    finally:
        learned._write_json_atomic(output / "input_access.json", {"parquet_reads": accesses, "test_response_rows_read": 0})


if __name__ == "__main__":
    main()
