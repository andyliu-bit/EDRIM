"""Immutable epoch-boundary recovery kernel, not a full experiment scheduler.

A JSON commit record is written after its checkpoint. Partial/orphan writes
fail closed. One writer per directory; stale locks require manual inspection.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch

from .learned_benchmark import _canonical_hash, _file_record, _write_checkpoint_atomic, _write_json_atomic


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def capture_rng():
    numpy_state = np.random.get_state()
    return dict(python=random.getstate(), torch=torch.get_rng_state(),
        numpy=[numpy_state[0], numpy_state[1].tolist(), int(numpy_state[2]),
               int(numpy_state[3]), float(numpy_state[4])],
        cuda=[state.cpu() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA topology changed")
        torch.cuda.set_rng_state_all(state["cuda"])


@contextmanager
def exclusive_job(path):
    path.mkdir(parents=True, exist_ok=True)
    lock = path / "writer.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise ValueError("Writer lock exists; inspect active/stale job before resuming") from exc
    try:
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        os.close(descriptor)
        yield
    finally:
        lock.unlink()


def _verify_bindings(contract):
    for record in contract["files"]:
        if _file_record(Path(record["path"])) != record:
            raise ValueError("Bound source/input file changed")


def _commit(path, state, contract_hash):
    generation = len(list(path.glob("epoch_*.json")))
    checkpoint = path / f"epoch_{generation:06d}.pt"
    manifest = path / f"epoch_{generation:06d}.json"
    if checkpoint.exists() or manifest.exists():
        raise ValueError("Refusing to overwrite an epoch generation")
    _write_checkpoint_atomic(checkpoint, state)
    record = dict(generation=generation, completed_epochs=state["completed_epochs"],
                  contract_hash=contract_hash, checkpoint=_file_record(checkpoint))
    record["manifest_sha256"] = _canonical_hash(record)
    _write_json_atomic(manifest, record)


def _latest(path, contract_hash):
    manifests = sorted(path.glob("epoch_*.json"))
    checkpoints = sorted(path.glob("epoch_*.pt"))
    if len(manifests) != len(checkpoints):
        raise ValueError("Orphan/partial checkpoint requires inspection")
    last = None
    for generation, manifest in enumerate(manifests):
        checkpoint = path / f"epoch_{generation:06d}.pt"
        if manifest.name != f"epoch_{generation:06d}.json" or checkpoints[generation] != checkpoint:
            raise ValueError("Epoch generations are not contiguous")
        record = json.loads(manifest.read_text(encoding="utf-8"))
        digest = record.pop("manifest_sha256")
        if digest != _canonical_hash(record) or record["contract_hash"] != contract_hash:
            raise ValueError("Epoch manifest integrity/contract mismatch")
        if record["generation"] != generation or _file_record(checkpoint) != record["checkpoint"]:
            raise ValueError("Checkpoint integrity mismatch")
        last = checkpoint
    return None if last is None else torch.load(last, weights_only=True, map_location="cpu")


def train_resumable(model, optimizer, directory, contract, train_epoch, evaluate, *, stop_after_epochs=None):
    """Run/resume at committed epoch boundaries; callbacks must be source-bound.

    contract requires identity, files (source/input records), and training with
    max_epochs/min_epochs/patience/min_delta. Epoch zero candidate is the caller's
    initial model (generation 0 means epoch -1); the caller must separately
    validate that it equals HSA.
    stop_after_epochs is an execution pause, not a changed scientific budget.
    """
    path = Path(directory).resolve()
    budget = contract["training"]
    stride = budget.get("checkpoint_stride", 1)
    time_budget = budget.get("max_train_seconds")
    if type(stride) is not int or stride < 1:
        raise ValueError("checkpoint_stride must be a positive integer")
    if time_budget is not None and (not np.isfinite(time_budget) or time_budget <= 0):
        raise ValueError("Invalid training time budget")
    for key in ("max_epochs", "min_epochs", "patience"):
        if type(budget[key]) is not int or budget[key] < 1:
            raise ValueError("Training epoch budgets must be positive integers")
    if budget["min_epochs"] > budget["max_epochs"] or not np.isfinite(budget["min_delta"]) or budget["min_delta"] < 0:
        raise ValueError("Invalid early stopping budget")
    if not contract.get("identity") or not contract.get("files"):
        raise ValueError("Explicit job identity and source/input bindings are required")
    if stop_after_epochs is not None and (type(stop_after_epochs) is not int or stop_after_epochs < 1):
        raise ValueError("Pause boundary must be a positive epoch count")
    envelope = dict(version="epoch_recovery_v1", contract=contract,
        runtime=dict(python=sys.version, numpy=np.__version__, torch=str(torch.__version__), cuda=torch.version.cuda,
            device=str(next(model.parameters()).device), cuda_devices=torch.cuda.device_count(),
            threads=torch.get_num_threads(), deterministic=torch.are_deterministic_algorithms_enabled(),
            cudnn_benchmark=torch.backends.cudnn.benchmark, cudnn_deterministic=torch.backends.cudnn.deterministic,
            cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            matmul_precision=torch.get_float32_matmul_precision(),
            cuda_device_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]),
        model_schema=[dict(name=name, shape=list(p.shape), dtype=str(p.dtype), trainable=p.requires_grad)
                      for name, p in model.named_parameters()],
        optimizer_type=f"{type(optimizer).__module__}.{type(optimizer).__name__}",
        optimizer_options=[{k: v for k, v in group.items() if k != "params"} for group in optimizer.param_groups])
    contract_hash = _canonical_hash(envelope)
    with exclusive_job(path):
        _verify_bindings(contract)
        frozen = path / "contract.json"
        if frozen.exists():
            if json.loads(frozen.read_text(encoding="utf-8")) != json.loads(json.dumps(envelope)):
                raise ValueError("Frozen job contract changed")
        else:
            if list(path.glob("epoch_*")):
                raise ValueError("Epoch artifacts exist without a frozen contract")
            _write_json_atomic(frozen, envelope)
        state = _latest(path, contract_hash)
        if state is None:
            score = float(evaluate(model))
            if not np.isfinite(score):
                raise ValueError("Non-finite initial score")
            state = dict(completed_epochs=0, best_epoch=-1, best_score=score, stale_epochs=0,
                last_model=cpu_tree(model.state_dict()), best_model=cpu_tree(model.state_dict()),
                optimizer=cpu_tree(optimizer.state_dict()), rng=capture_rng(), finished=False, training_seconds=0.0,
                history=[dict(epoch=-1, validation_rmse=score, train_metrics=None, improved=True)])
            _verify_bindings(contract)
            _commit(path, state, contract_hash)
        else:
            model.load_state_dict(state["last_model"], strict=True)
            optimizer.load_state_dict(state["optimizer"])
            restore_rng(state["rng"])
        while not state["finished"]:
            epoch = state["completed_epochs"]
            if stop_after_epochs is not None and epoch >= stop_after_epochs:
                break
            metrics = (train_epoch(model, optimizer, epoch) if time_budget is None else
                       train_epoch(model, optimizer, epoch, time_budget-state["training_seconds"]))
            seconds = metrics.get("training_seconds", 0.0)
            if not np.isfinite(seconds) or seconds < 0 or (time_budget is not None and seconds <= 0):
                raise ValueError("Invalid measured training time")
            state["training_seconds"] += seconds
            score = float(evaluate(model))
            if not np.isfinite(score) or not all(torch.isfinite(v).all() for v in model.state_dict().values()):
                raise ValueError("Non-finite epoch state/score")
            improved = score < state["best_score"]-budget["min_delta"]
            if improved:
                state["best_score"], state["best_epoch"] = score, epoch
                state["best_model"] = cpu_tree(model.state_dict())
            state["stale_epochs"] = 0 if improved else state["stale_epochs"]+1
            state["completed_epochs"] = epoch+1
            state["finished"] = epoch+1 >= budget["max_epochs"] or (
                time_budget is None and epoch+1 >= budget["min_epochs"] and state["stale_epochs"] >= budget["patience"]) or (
                time_budget is not None and state["training_seconds"] >= time_budget)
            state["history"].append(dict(epoch=epoch, validation_rmse=score, train_metrics=metrics, improved=improved))
            state.update(last_model=cpu_tree(model.state_dict()), optimizer=cpu_tree(optimizer.state_dict()), rng=capture_rng())
            _verify_bindings(contract)
            if state["finished"] or state["completed_epochs"] % stride == 0 or state["completed_epochs"] == stop_after_epochs:
                _commit(path, state, contract_hash)
        return state
