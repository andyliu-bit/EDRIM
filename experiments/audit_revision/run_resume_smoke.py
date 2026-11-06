"""Explicit CPU/CUDA exact-replay acceptance on real development data."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import bisurfgnn.independent_graph_benchmark as graph
import bisurfgnn.learned_benchmark as learned
from bisurfgnn.benchmark_config import load_benchmark_config
from bisurfgnn.data.benchmark_context import load_benchmark_cell_context
from bisurfgnn.data.cellminer import load_cellminer_config
from bisurfgnn.data.chemical_splits import load_one_new_scaffold_benchmark
from bisurfgnn.resumable_training import train_resumable
from run_factorial_smoke import ROOT, CONFIG, load_matrix, build_model, sample_groups, guarded_read


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        if not torch.equal(a, b):
            raise RuntimeError("Recovered tensor differs from uninterrupted reference")
    elif isinstance(a, dict):
        if a.keys() != b.keys():
            raise RuntimeError("Recovered dictionary inventory differs")
        for key in a:
            equal_tree(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            raise RuntimeError("Recovered sequence length differs")
        for left, right in zip(a, b):
            equal_tree(left, right)
    elif a != b:
        raise RuntimeError("Recovered scalar differs")


def execute(output, accesses, device_name="cpu", phase=None):
    torch.set_num_threads(1)
    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device is unavailable")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise RuntimeError("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting the CUDA process")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    config = load_matrix()
    benchmark_path = ROOT / config["benchmark_config"]
    benchmark = load_benchmark_config(benchmark_path)
    sources = sorted((ROOT/"src/bisurfgnn").rglob("*.py")) + [Path(__file__),
        ROOT/"experiments/audit_revision/run_factorial_smoke.py", CONFIG, benchmark_path,
        ROOT/"configs/cell_context.yaml"]
    records, snapshots = [], []
    for source in sources:
        record = learned._file_record(source)
        destination = output/"source_snapshot"/source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copyfile(source, destination)
        snapshot = learned._file_record(destination)
        if record["sha256"] != snapshot["sha256"]:
            raise RuntimeError("Source snapshot mismatch")
        records.append(record)
        snapshots.append(snapshot)
    protocol = dict(
        claim_scope=f"{device.type}_exact_epoch_recovery_only", sources=records, snapshots=snapshots,
        config=config, variant="bixt__gate", epochs=3, pause_after_epochs=1,
        training_batch_surfaces=32, frozen_before_response_loading=True)
    protocol_path = output/"preflight_protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise RuntimeError("Existing preflight protocol changed")
    else:
        learned._write_json_exclusive(protocol_path, protocol)
    split = load_one_new_scaffold_benchmark(benchmark)
    context = load_benchmark_cell_context(benchmark, load_cellminer_config(ROOT/"configs/cell_context.yaml"))
    if split.split_hash != context.split_hash:
        raise RuntimeError("Split mismatch")
    shards = learned.load_learned_point_shards(benchmark, verify_roles=("train", "validation"))
    inputs = [learned._file_record(shards.role_path(0, role)) for role in ("train", "validation")]
    frames = {role: sample_groups(learned.load_learned_role_frame(shards, split.surfaces, 0, role,
        sample_seed=41), config["smoke"][f"{role}_max_points"], seed=41, salt="P1-v1")
        for role in ("train", "validation")}
    assignment = learned._fold_surface_roles(split.surfaces, 0)
    banks = graph.build_graph_feature_banks(sorted(set(assignment.drug_a_id)|set(assignment.drug_b_id)),
        sorted(set(assignment.cell_id)), context, 0, benchmark)
    tables = {role: learned.build_compact_point_table(frame, banks, role=role, fold=0,
        monotherapy_upper_bound=3) for role, frame in frames.items()}
    if set(tables["train"].group_ids) & set(tables["validation"].group_ids):
        raise RuntimeError("Group overlap")
    contract = dict(identity=dict(variant="bixt__gate", fold=0, seed=41, device=str(device),
        split_hash=split.split_hash, endpoint=benchmark.endpoint, model=config["model"],
        graph_bank_hash=graph._graph_bank_hash(banks),
        table_hashes={role: learned._point_table_hash(table) for role, table in tables.items()},
        batch_surfaces=32, gradient_clip=1.0), files=records+inputs,
        training=dict(max_epochs=3, min_epochs=3, patience=3, min_delta=1e-5))
    def build():
        model = build_model(banks, config, "bixt__gate").to(device)
        learned._seed_everything(41)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                      lr=3e-4, weight_decay=1e-4)
        return model, optimizer
    def predict(model):
        return graph.predict_graph_compact(model, tables["validation"], banks, batch_size=512, device=device)[0]
    def evaluate(model):
        return learned.surface_macro_rmse(tables["validation"].target.numpy(), predict(model),
                                          tables["validation"].surface_ids)
    def train_epoch(model, optimizer, epoch):
        model.train()
        losses, counts = [], []
        for indices in graph._surface_batch_indices(tables["train"], 32, shuffle_seed=41*1_000_003+epoch):
            optimizer.zero_grad(set_to_none=True)
            result = graph._model_forward(model, tables["train"], banks, indices, device)
            surfaces = tables["train"].surface_ids[indices.numpy()]
            loss = graph._surface_mean_mse(result.prediction, tables["train"].target[indices].to(device), surfaces)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(norm)):
                raise RuntimeError("Non-finite real-data update")
            optimizer.step()
            losses.append(float(loss.detach()))
            counts.append(len(set(surfaces)))
        return dict(surface_mse=float(np.average(losses, weights=counts)), optimizer_updates=len(losses))
    model, optimizer = build()
    anchor = np.minimum(tables["validation"].mono_a.numpy(), tables["validation"].mono_b.numpy())
    np.testing.assert_allclose(predict(model), anchor, rtol=2e-6, atol=2e-7)
    if phase is not None:
        if phase not in ("continuous", "pause", "resume", "replay"):
            raise ValueError("Unknown process phase")
        phase_path = output/f"phase_{phase}.json"
        if phase_path.exists():
            raise RuntimeError("Phase already has a receipt; refusing overwrite")
        route = "continuous" if phase == "continuous" else "resumed"
        if phase in ("resume", "replay") and not (output/route/"contract.json").exists():
            raise RuntimeError("Resume/replay requires an existing committed job")
        if phase in ("continuous", "pause") and (output/route/"contract.json").exists():
            raise RuntimeError("Initial phase requires a fresh route")
        before = [learned._file_record(p) for p in sorted((output/route).glob("epoch_*"))]
        print(f"Real data process phase={phase} device={device} pid={os.getpid()}", flush=True)
        state = train_resumable(model, optimizer, output/route, contract, train_epoch, evaluate,
                                 stop_after_epochs=1 if phase == "pause" else None)
        expected_epochs = 1 if phase == "pause" else 3
        if state["completed_epochs"] != expected_epochs or state["finished"] != (phase != "pause"):
            raise RuntimeError("Phase completion boundary mismatch")
        if phase == "replay":
            after = [learned._file_record(p) for p in sorted((output/route).glob("epoch_*"))]
            if before != after:
                raise RuntimeError("Completed process replay rewrote committed artifacts")
        model.load_state_dict(state["best_model"], strict=True)
        prediction_path = output/f"predictions_{phase}.pt"
        if prediction_path.exists():
            raise RuntimeError("Prediction receipt already exists")
        learned._write_checkpoint_atomic(prediction_path, dict(
            source_rows=torch.from_numpy(tables["validation"].source_rows.copy()),
            prediction=torch.from_numpy(predict(model))))
        for record in records+snapshots+inputs:
            if learned._file_record(Path(record["path"])) != record:
                raise RuntimeError("Bound file changed during worker phase")
        report = dict(phase=phase, pid=os.getpid(), device=str(device),
            completed_epochs=state["completed_epochs"], finished=state["finished"],
            checkpoint=learned._file_record(output/route/f"epoch_{expected_epochs:06d}.pt"),
            prediction=learned._file_record(prediction_path),
            contract=learned._file_record(output/route/"contract.json"),
            test_response_files_read=0, parquet_read_trace=accesses,
            completed_job_unchanged=phase == "replay")
        report["manifest_sha256"] = learned._canonical_hash(report)
        learned._write_json_exclusive(phase_path, report)
        return
    print(f"Uninterrupted {device} reference: 3 epochs", flush=True)
    full = train_resumable(model, optimizer, output/"continuous", contract, train_epoch, evaluate)
    model, optimizer = build()
    print("Interrupted route: commit epoch 1, then reconstruct model and optimizer", flush=True)
    paused = train_resumable(model, optimizer, output/"resumed", contract, train_epoch, evaluate, stop_after_epochs=1)
    if paused["completed_epochs"] != 1 or paused["finished"]:
        raise RuntimeError("Pause boundary not respected")
    model, optimizer = build()
    resumed = train_resumable(model, optimizer, output/"resumed", contract, train_epoch, evaluate)
    equal_tree(full, resumed)
    model.load_state_dict(full["best_model"], strict=True)
    first = predict(model)
    model.load_state_dict(resumed["best_model"], strict=True)
    np.testing.assert_array_equal(first, predict(model))
    checkpoint_records = [learned._file_record(p) for p in sorted(output.glob("*/epoch_*"))]
    model, optimizer = build()
    replay = train_resumable(model, optimizer, output/"resumed", contract, train_epoch, evaluate)
    equal_tree(replay, resumed)
    for record in records+snapshots+inputs+checkpoint_records:
        if learned._file_record(Path(record["path"])) != record:
            raise RuntimeError("Bound artifact changed")
    report = dict(claim_scope=f"{device.type}_exact_epoch_recovery_only", completed_epochs=resumed["completed_epochs"],
        history=resumed["history"], full_state_exact_match=True, best_prediction_exact_match=True,
        completed_job_idempotent=True, checkpoint_records=checkpoint_records,
        protocol=learned._file_record(output/"preflight_protocol.json"),
        contracts=[learned._file_record(output/route/"contract.json") for route in ("continuous", "resumed")],
        test_response_files_read=0, parquet_read_trace=accesses, runtime=learned._runtime_environment(),
        not_validated=["new-process GPU numerical replay", "process kill during write", "mid-epoch recovery", "full-run scheduling"])
    report["manifest_sha256"] = learned._canonical_hash(report)
    learned._write_json_atomic(output/"resume_smoke_manifest.json", report)
    print(f"Real-data {device} full-state replay is exact; completed job replay made no checkpoint changes.", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--phase", choices=("continuous", "pause", "resume", "replay"))
    args = parser.parse_args()
    output = args.output_dir.resolve()
    allowed = (ROOT/"runs/audit_revision").resolve()
    if not output.is_relative_to(allowed) or output == allowed:
        raise SystemExit("Output must be a fresh child of runs/audit_revision")
    output.mkdir(parents=True, exist_ok=args.phase is not None)
    accesses = []
    with patch.object(pd, "read_parquet", guarded_read(pd.read_parquet, accesses)):
        execute(output, accesses, args.device, args.phase)


if __name__ == "__main__":
    main()
