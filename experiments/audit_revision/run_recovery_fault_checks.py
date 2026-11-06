"""Fresh-directory subprocess fault fixtures; no real data and no automatic cleanup."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from bisurfgnn import resumable_training as recovery
from bisurfgnn.learned_benchmark import _canonical_hash, _file_record, _seed_everything, _write_json_atomic
from run_resume_smoke import equal_tree

ROOT = Path(__file__).resolve().parents[2]
STAGES = ("partial_temp", "checkpoint_without_commit", "after_commit")


def child(directory, stage, device_name):
    torch.set_num_threads(1)
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    _seed_everything(41)
    model = torch.nn.Sequential(torch.nn.Linear(4, 16), torch.nn.GELU(), torch.nn.Dropout(.3),
                                torch.nn.Linear(16, 1)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    contract = dict(identity=dict(kind="synthetic_process_fixture", device=str(device), seed=41),
        files=[_file_record(Path(__file__)), _file_record(Path(recovery.__file__))],
        training=dict(max_epochs=3, min_epochs=3, patience=3, min_delta=0.0))
    original_checkpoint, original_json, original_save = recovery._write_checkpoint_atomic, recovery._write_json_atomic, torch.save
    def checkpoint(path, payload):
        original_checkpoint(path, payload)
        if stage == "checkpoint_without_commit" and path.name == "epoch_000001.pt":
            os._exit(73)
    def manifest(path, payload):
        original_json(path, payload)
        if stage == "after_commit" and path.name == "epoch_000001.json":
            os._exit(73)
    def save(payload, filename, *args, **kwargs):
        if stage == "partial_temp" and Path(filename).name.startswith(".epoch_000001.pt."):
            # Deliberately incomplete NEW temporary fixture; never a user artifact.
            Path(filename).write_bytes(b"INTENTIONALLY_INCOMPLETE_TEST_CHECKPOINT")
            os._exit(73)
        return original_save(payload, filename, *args, **kwargs)
    recovery._write_checkpoint_atomic, recovery._write_json_atomic, torch.save = checkpoint, manifest, save
    def train(model, optimizer, epoch):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        value = model(torch.rand(17, 4, device=device)).square().mean()
        value.backward()
        optimizer.step()
        return dict(loss=float(value.detach()))
    def evaluate(model):
        model.eval()
        with torch.no_grad():
            return float(model(torch.ones(17, 4, device=device)).square().mean())
    recovery.train_resumable(model, optimizer, directory, contract, train, evaluate,
                             stop_after_epochs=1 if stage == "pause" else None)


def launch(directory, stage, device="cpu"):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT/"src")
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker",
        "--output-dir", str(directory), "--stage", stage, "--device", device],
        env=env, capture_output=True, text=True, timeout=120,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    return dict(stage=stage, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)


def inspect_latest(directory):
    contract = json.loads((directory/"contract.json").read_text(encoding="utf-8"))
    return recovery._latest(directory, _canonical_hash(contract))


def check_fault(directory, stage):
    result = launch(directory, stage)
    if result["returncode"] != 73:
        raise RuntimeError(f"Fault injection did not reach its declared boundary: {result}")
    if not (directory/"writer.lock").exists():
        raise RuntimeError("Abrupt exit did not leave the expected lock")
    before = {str(p): _file_record(p) for p in directory.iterdir() if p.is_file()}
    refused = False
    try:
        with recovery.exclusive_job(directory):
            pass
    except ValueError:
        refused = True
    if not refused:
        raise RuntimeError("Stale lock did not block a new writer")
    # Verify generation zero independently even when an orphan blocks latest.
    record = json.loads((directory/"epoch_000000.json").read_text(encoding="utf-8"))
    if _file_record(directory/"epoch_000000.pt") != record["checkpoint"]:
        raise RuntimeError("Previously committed checkpoint changed")
    if stage == "checkpoint_without_commit":
        try:
            inspect_latest(directory)
        except ValueError as exc:
            if "Orphan/partial" not in str(exc):
                raise
        else:
            raise RuntimeError("Uncommitted checkpoint was accepted")
        last_epoch = None
    else:
        last_epoch = inspect_latest(directory)["completed_epochs"]
        if last_epoch != (1 if stage == "after_commit" else 0):
            raise RuntimeError("Wrong committed generation")
    if before != {str(p): _file_record(p) for p in directory.iterdir() if p.is_file()}:
        raise RuntimeError("Read-only fault inspection mutated artifacts")
    return dict(**result, writer_refused=True, previous_commit_intact=True,
                readable_completed_epochs=last_epoch, artifacts=list(before.values()),
                orphan_refused=stage == "checkpoint_without_commit", inspection_read_only=True)


def execute(output):
    sources = [_file_record(Path(__file__)), _file_record(Path(recovery.__file__)),
               _file_record(ROOT/"experiments/audit_revision/run_resume_smoke.py")]
    results = []
    for stage in STAGES:
        print(f"Subprocess abrupt-exit fixture: {stage}", flush=True)
        results.append(check_fault(output/stage, stage))
    gpu = None
    if torch.cuda.is_available():
        print("Synthetic CUDA fresh-process replay: uninterrupted / pause / resume", flush=True)
        launches = [launch(output/"gpu_full", "continuous", "cuda"),
                    launch(output/"gpu_resumed", "pause", "cuda"),
                    launch(output/"gpu_resumed", "resume", "cuda")]
        if any(run["returncode"] != 0 for run in launches):
            raise RuntimeError(f"Fresh-process CUDA execution failed: {launches}")
        equal_tree(inspect_latest(output/"gpu_full"), inspect_latest(output/"gpu_resumed"))
        gpu = dict(scope="synthetic_fixture_not_real_drug_data", fresh_process_full_state_exact=True,
                   launches=launches, artifacts=[_file_record(p) for route in ("gpu_full", "gpu_resumed")
                       for p in sorted((output/route).iterdir()) if p.is_file()])
    for record in sources:
        if _file_record(Path(record["path"])) != record:
            raise RuntimeError("Source changed during fault tests")
    report = dict(scope="subprocess_fault_detection_and_synthetic_cuda_replay", sources=sources,
        fault_results=results, gpu_process_replay=gpu, data_response_files_read=0,
        limitations=["os._exit injection, not external OS kill or power loss", "no stale-lock removal or automatic recovery",
                     "fresh-process GPU replay uses synthetic data", "no mid-epoch recovery"])
    report["manifest_sha256"] = _canonical_hash(report)
    _write_json_atomic(output/"recovery_fault_manifest.json", report)
    print("All declared fault boundaries rejected unsafe writes; retained fixture artifacts for inspection.", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--stage", choices=STAGES+("continuous", "pause", "resume"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.worker:
        child(output, args.stage, args.device)
        return
    allowed = (ROOT/"runs/audit_revision").resolve()
    if not output.is_relative_to(allowed) or output == allowed:
        raise SystemExit("Output must be a fresh child of runs/audit_revision")
    output.mkdir(parents=True, exist_ok=False)
    execute(output)


if __name__ == "__main__":
    main()
