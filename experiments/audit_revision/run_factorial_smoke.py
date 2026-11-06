"""Fixed six-arm real-data engineering preflight; never ranks or selects arms.

Uses existing role-verified graph/data/loss kernels, not the legacy factory's
parameter-matched selection contract. Full five-fold training is deliberately
not exposed by this script. See docs/FACTORIAL_EXPERIMENT_PROTOCOL.md.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
from time import perf_counter
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import yaml

import bisurfgnn.independent_graph_benchmark as graph
import bisurfgnn.learned_benchmark as learned
from bisurfgnn.benchmark_config import load_benchmark_config
from bisurfgnn.data.benchmark_context import load_benchmark_cell_context
from bisurfgnn.data.cellminer import load_cellminer_config
from bisurfgnn.data.chemical_splits import load_one_new_scaffold_benchmark
from bisurfgnn.models.factorial_bisurf import FactorialBiSurfGNN, VARIANTS, common_initial_state_hash

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/audit_revision/factorial_v1.yaml"


def load_matrix(path=CONFIG):
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config["protocol_version"] != "common_decoder_factorial_development_v1":
        raise ValueError("Unknown factorial protocol")
    if tuple(config["variants"]) != VARIANTS:
        raise ValueError("All six arms in declared order are mandatory; no smoke-based filtering")
    if config["capacity_design"] != "constant_width_not_parameter_matched":
        raise ValueError("This runner does not implement parameter matching")
    smoke = config["smoke"]
    if smoke != dict(fold=0, seed=41, max_epochs=3, train_max_points=900,
                     validation_max_points=1800, group_order_salt="P1-v1", decision_use="forbidden"):
        raise ValueError("Smoke budget/fold/seed is fixed; full training needs a reviewed runner")
    if config["model"]["monotherapy_upper_bound"] != 3.0:
        raise ValueError("D1 preflight requires B=3")
    if config["training"]["residual_shrinkage"] != 0.0:
        raise ValueError("This version freezes zero residual shrinkage")
    return config


def sample_groups(frame, max_points, *, seed, salt):
    sizes = frame.groupby("prediction_group_id", sort=True).size()
    order = sorted(sizes.index, key=lambda g: hashlib.sha256(f"{salt}|{seed}|{g}".encode()).hexdigest())
    selected, total = [], 0
    for group_id in order:
        if total + int(sizes[group_id]) <= max_points:
            selected.append(group_id)
            total += int(sizes[group_id])
    if not selected:
        raise ValueError("No complete group fits the budget")
    return frame[frame.prediction_group_id.isin(selected)].copy()


def guarded_read(original, accesses):
    def read(path, *args, **kwargs):
        resolved = Path(path).resolve()
        if resolved.name.lower() in {"test.parquet", "points.parquet"}:
            raise RuntimeError(f"Forbidden test/global response source: {resolved}")
        accesses.append(str(resolved))
        return original(path, *args, **kwargs)
    return read


def build_model(banks, config, variant):
    exchange, fusion = variant.split("__")
    learned._seed_everything(config["smoke"]["seed"])
    return FactorialBiSurfGNN(banks.atom_feature_dim, banks.cell_feature_dim,
                             exchange=exchange, fusion=fusion, **config["model"])


def fit_smoke(model, banks, train, validation, config, device):
    training, smoke = config["training"], config["smoke"]
    model.to(device)
    # Construction consumes different numbers of random draws in each arm.
    learned._seed_everything(smoke["seed"])
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                 lr=training["learning_rate"], weight_decay=training["weight_decay"])

    def predict():
        return graph.predict_graph_compact(model, validation, banks,
            batch_size=training["evaluation_batch_points"], device=device)[0]

    target = validation.target.numpy()
    anchor = np.minimum(validation.mono_a.numpy(), validation.mono_b.numpy())
    initial = predict()
    np.testing.assert_allclose(initial, anchor, rtol=2e-6, atol=2e-7)
    best_rmse = learned.surface_macro_rmse(target, anchor, validation.surface_ids)
    best_epoch = -1
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    history = [{"epoch": -1, "checkpoint_kind": "anchor_no_learning",
                "validation_surface_macro_rmse": best_rmse, "train_surface_mse": None}]
    gradient_max = {name: 0.0 for name in ("drug_encoder", "context_encoder", "bixt",
                                         "context_fusion", "interaction_decoder")}
    for epoch in range(smoke["max_epochs"]):
        model.train()
        fit_sum, surface_count = 0.0, 0
        for indices in graph._surface_batch_indices(train, training["batch_surfaces"],
                shuffle_seed=smoke["seed"] * 1_000_003 + epoch):
            optimizer.zero_grad(set_to_none=True)
            output = graph._model_forward(model, train, banks, indices, device)
            index_cpu = indices.to(dtype=torch.int64, device="cpu")
            surfaces = train.surface_ids[index_cpu.numpy()]
            loss = graph._surface_mean_mse(output.prediction, train.target[index_cpu].to(device), surfaces)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training["gradient_clip"])
            if not bool(torch.isfinite(norm)):
                raise RuntimeError("Non-finite gradient")
            for name in gradient_max:
                gradients = [p.grad for p in getattr(model, name).parameters() if p.grad is not None]
                square_sum = sum(float(g.detach().square().sum()) for g in gradients)
                gradient_max[name] = max(gradient_max[name], square_sum ** 0.5)
            optimizer.step()
            n_surfaces = len(set(surfaces.tolist()))
            fit_sum += float(loss.detach()) * n_surfaces
            surface_count += n_surfaces
        prediction = predict()
        if not np.isfinite(prediction).all():
            raise RuntimeError("Non-finite validation prediction")
        rmse = learned.surface_macro_rmse(target, prediction, validation.surface_ids)
        improved = rmse < best_rmse - training["min_delta"]
        if improved:
            best_epoch, best_rmse = epoch, rmse
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        history.append({"epoch": epoch, "train_surface_mse": fit_sum / surface_count,
                        "validation_surface_macro_rmse": rmse, "is_best": improved})
    if any(value <= 0 or not np.isfinite(value) for value in gradient_max.values()):
        raise RuntimeError(f"A required module never received a finite nonzero gradient: {gradient_max}")
    if not all(torch.isfinite(value).all() for value in best_state.values()):
        raise RuntimeError("Non-finite selected state")
    model.load_state_dict(best_state, strict=True)
    selected = predict()
    reproduced = learned.surface_macro_rmse(target, selected, validation.surface_ids)
    np.testing.assert_allclose(reproduced, best_rmse, atol=2e-6, rtol=0)
    return ({"best_epoch": best_epoch, "epochs_ran": smoke["max_epochs"],
             "best_validation_surface_macro_rmse": best_rmse, "history": history,
             "module_max_gradient_norm": gradient_max}, best_state, selected)


def execute(output, device, accesses):
    config = load_matrix()
    benchmark_path = ROOT / config["benchmark_config"]
    benchmark = load_benchmark_config(benchmark_path)
    if benchmark.endpoint != config["endpoint"]:
        raise RuntimeError("Endpoint mismatch")
    # Capture the entire package, runner, protocol and all input configurations.
    # Original files are checked again at completion, as are every saved snapshot.
    sources = sorted((ROOT / "src/bisurfgnn").rglob("*.py")) + [
        Path(__file__), CONFIG, benchmark_path, ROOT / "configs/cell_context.yaml",
        ROOT / "docs/FACTORIAL_EXPERIMENT_PROTOCOL.md"]
    records = []
    for source in sources:
        destination = output / "source_snapshot" / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        before = learned._file_record(source)
        shutil.copyfile(source, destination)
        snapshot = learned._file_record(destination)
        if before["sha256"] != snapshot["sha256"]:
            raise RuntimeError("Source changed during snapshot")
        records.append({"original": before, "snapshot": snapshot})
    protocol = {"run_mode": "p1_factorial_engineering_smoke_no_branch_decision",
                "resolved_config": config, "sources": records,
                "written_before_response_loading_and_training": True}
    protocol["manifest_sha256"] = learned._canonical_hash(protocol)
    learned._write_json_atomic(output / "preflight_protocol.json", protocol)
    split = load_one_new_scaffold_benchmark(benchmark)
    context = load_benchmark_cell_context(benchmark, load_cellminer_config(ROOT / "configs/cell_context.yaml"))
    if context.split_hash != split.split_hash:
        raise RuntimeError("Context/split mismatch")
    shards = learned.load_learned_point_shards(benchmark, verify_roles=("train", "validation"))
    smoke = config["smoke"]
    frames = {role: sample_groups(learned.load_learned_role_frame(
        shards, split.surfaces, smoke["fold"], role, sample_seed=smoke["seed"]),
        smoke[f"{role}_max_points"], seed=smoke["seed"], salt=smoke["group_order_salt"])
        for role in ("train", "validation")}
    if set(frames["train"].prediction_group_id) & set(frames["validation"].prediction_group_id):
        raise RuntimeError("Train/validation group overlap")
    assignment = learned._fold_surface_roles(split.surfaces, smoke["fold"])
    drugs = sorted(set(assignment.drug_a_id) | set(assignment.drug_b_id))
    cells = sorted(set(assignment.cell_id))
    banks = graph.build_graph_feature_banks(drugs, cells, context, smoke["fold"], benchmark)
    tables = {role: learned.build_compact_point_table(frame, banks, role=role, fold=smoke["fold"],
              monotherapy_upper_bound=config["model"]["monotherapy_upper_bound"])
              for role, frame in frames.items()}
    inventory = []
    for variant in VARIANTS:
        model = build_model(banks, config, variant)
        inventory.append({"variant": variant, "common_initial_state_sha256": common_initial_state_hash(model),
                          "parameters_total": sum(p.numel() for p in model.parameters()),
                          "parameters_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)})
    if len({item["common_initial_state_sha256"] for item in inventory}) != 1:
        raise RuntimeError("Shared initialization differs across arms")
    learned._write_json_atomic(output / "prefit_inventory.json", {"inventory": inventory,
                              "graph_bank_hash": graph._graph_bank_hash(banks),
                              "sample": {role: {"points": len(table), "table_hash": learned._point_table_hash(table),
                                   "group_ids": sorted(set(table.group_ids.tolist()))}
                                   for role, table in tables.items()}})
    val = tables["validation"]
    assay_lookup = split.surfaces.set_index("surface_replicate_id").assay_block_id
    metadata = pd.DataFrame(dict(fold=smoke["fold"], seed=smoke["seed"], source_id=val.source_rows,
        surface_id=val.surface_ids, group_id=val.group_ids, cluster_id=val.cluster_ids,
        assay_block_id=pd.Series(val.surface_ids).map(assay_lookup).to_numpy(), target=val.target.numpy()))
    if metadata.isna().any().any():
        raise RuntimeError("Missing point metadata")
    parts = [metadata.assign(model="hsa", prediction=np.minimum(val.mono_a.numpy(), val.mono_b.numpy()))]
    jobs = []
    for item in inventory:
        variant = item["variant"]
        print(f"Training fixed preflight arm: {variant}", flush=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = perf_counter()
        model = build_model(banks, config, variant)
        if common_initial_state_hash(model) != item["common_initial_state_sha256"]:
            raise RuntimeError("Initial state changed since prefit inventory")
        selection, state, prediction = fit_smoke(model, banks, tables["train"], val, config, device)
        checkpoint = output / variant / "checkpoint.pt"
        learned._write_checkpoint_atomic(checkpoint, {"variant": variant, "state_dict": state,
                                                      "model_config": config["model"]})
        reloaded = build_model(banks, config, variant).to(device)
        saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
        if saved["variant"] != variant or saved["model_config"] != config["model"]:
            raise RuntimeError("Checkpoint model identity mismatch")
        reloaded.load_state_dict(saved["state_dict"], strict=True)
        replay = graph.predict_graph_compact(reloaded, val, banks,
            batch_size=config["training"]["evaluation_batch_points"], device=device)[0]
        np.testing.assert_allclose(prediction, replay, atol=1e-7, rtol=0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        jobs.append({**item, "selection": selection, "checkpoint": learned._file_record(checkpoint),
                     "prediction_reload_check": "passed", "elapsed_seconds": perf_counter() - started,
                     "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None})
        parts.append(metadata.assign(model=variant, prediction=prediction))
        del model, reloaded
    predictions = pd.concat(parts, ignore_index=True)
    errors = predictions.assign(squared_error=(predictions.prediction - predictions.target) ** 2).groupby(
        ["fold", "seed", "model", "surface_id"], sort=True).agg(
            group_id=("group_id", "first"), cluster_id=("cluster_id", "first"),
            assay_block_id=("assay_block_id", "first"), n_points=("source_id", "size"),
            rmse=("squared_error", "mean")).reset_index()
    errors["rmse"] = np.sqrt(errors.rmse)
    files = {}
    for name, frame in (("point_predictions", predictions), ("surface_errors", errors)):
        path = output / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        pd.testing.assert_frame_equal(frame, pd.read_parquet(path))
        files[name] = learned._file_record(path)
    for record in records:
        for label in ("original", "snapshot"):
            if learned._file_record(Path(record[label]["path"])) != record[label]:
                raise RuntimeError("Source/snapshot changed during preflight")
    report = {"run_mode": protocol["run_mode"], "claim_scope": "engineering_only_not_performance_evidence",
              "protocol": learned._file_record(output / "preflight_protocol.json"),
              "prefit_inventory": learned._file_record(output / "prefit_inventory.json"),
              "endpoint": benchmark.endpoint, "split_hash": split.split_hash, "device": str(device),
              "source_inputs": {role: learned._file_record(shards.role_path(smoke["fold"], role)) for role in frames},
              "sample": {role: {"points": len(table), "groups": len(set(table.group_ids)),
                   "surfaces": len(set(table.surface_ids)), "clusters": len(set(table.cluster_ids))}
                   for role, table in tables.items()}, "jobs": jobs, **files,
              "runtime": learned._runtime_environment(), "parquet_read_trace": accesses,
              "test_response_files_read": 0, "branch_selection_permitted": False}
    report["manifest_sha256"] = learned._canonical_hash(report)
    learned._write_json_atomic(output / "factorial_smoke_manifest.json", report)
    print("Six-arm training, gradient, common initialization, source and reload checks passed. No arm selected.", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    allowed = (ROOT / "runs/audit_revision").resolve()
    if not output.is_relative_to(allowed) or output == allowed:
        raise SystemExit("Output must be a fresh child of runs/audit_revision")
    output.mkdir(parents=True, exist_ok=False)
    accesses = []
    with patch.object(pd, "read_parquet", guarded_read(pd.read_parquet, accesses)):
        execute(output, torch.device("cuda" if torch.cuda.is_available() else "cpu"), accesses)


if __name__ == "__main__":
    main()
