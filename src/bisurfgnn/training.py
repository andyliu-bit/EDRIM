from __future__ import annotations

import copy
import csv
import json
import math
import platform
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from .config import ExperimentConfig, resolve_device
from .data import (
    SyntheticSurfaceDataset,
    collate_surface_samples,
    split_by_surface,
    swap_drug_order,
)
from .losses import BiSurfLoss
from .metrics import summarize_predictions
from .models import BiSurfGNN


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader(
    dataset: SyntheticSurfaceDataset,
    indices: list[int],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_surface_samples,
        generator=generator,
        persistent_workers=num_workers > 0,
    )


def _train_epoch(
    model: BiSurfGNN,
    loader: DataLoader,
    optimizer: AdamW,
    criterion: BiSurfLoss,
    device: torch.device,
    grad_clip_norm: float,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    examples = 0
    for cpu_batch in loader:
        batch = cpu_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        breakdown = criterion(output, batch)
        breakdown.total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        size = int(batch.response.shape[0])
        examples += size
        for key, value in breakdown.detached().items():
            totals[key] = totals.get(key, 0.0) + value * size
    return {key: value / max(examples, 1) for key, value in totals.items()}


@torch.no_grad()
def evaluate(
    model: BiSurfGNN,
    loader: DataLoader,
    criterion: BiSurfLoss,
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    targets: list[float] = []
    predictions: list[float] = []
    surface_ids: list[int] = []
    rows: list[dict[str, Any]] = []
    total_loss = 0.0
    total_examples = 0
    maximum_symmetry_error = 0.0
    boundary_error_sum = 0.0
    boundary_count = 0

    for cpu_batch in loader:
        batch = cpu_batch.to(device)
        output = model(batch)
        swapped_output = model(swap_drug_order(batch))
        maximum_symmetry_error = max(
            maximum_symmetry_error,
            float((output.prediction - swapped_output.prediction).abs().max().cpu()),
        )
        boundary_mask = output.boundary_gate == 0
        if bool(boundary_mask.any()):
            boundary_error_sum += float(
                (output.prediction[boundary_mask] - output.reference[boundary_mask])
                .abs()
                .sum()
                .cpu()
            )
            boundary_count += int(boundary_mask.sum().item())

        loss = criterion(output, batch).total
        size = int(batch.response.shape[0])
        total_loss += float(loss.cpu()) * size
        total_examples += size

        prediction_values = output.prediction.cpu().tolist()
        target_values = batch.response.cpu().tolist()
        reference_values = output.reference.cpu().tolist()
        uncertainty_values = torch.exp(0.5 * output.log_variance).cpu().tolist()
        dose_values = batch.doses.cpu().tolist()
        current_surfaces = batch.surface_id.cpu().tolist()
        drug_a_ids = batch.drug_a_id.cpu().tolist()
        drug_b_ids = batch.drug_b_id.cpu().tolist()
        cell_ids = batch.cell_id.cpu().tolist()

        targets.extend(target_values)
        predictions.extend(prediction_values)
        surface_ids.extend(current_surfaces)
        for index in range(size):
            rows.append(
                {
                    "surface_id": current_surfaces[index],
                    "drug_a_id": drug_a_ids[index],
                    "drug_b_id": drug_b_ids[index],
                    "cell_id": cell_ids[index],
                    "dose_a": dose_values[index][0],
                    "dose_b": dose_values[index][1],
                    "target": target_values[index],
                    "prediction": prediction_values[index],
                    "reference": reference_values[index],
                    "predicted_std": uncertainty_values[index],
                }
            )

    metrics = summarize_predictions(targets, predictions, surface_ids)
    metrics["loss"] = total_loss / max(total_examples, 1)
    metrics["symmetry_max_abs_error"] = maximum_symmetry_error
    metrics["boundary_mae"] = boundary_error_sum / max(boundary_count, 1)
    return metrics, rows


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, ensure_ascii=False, indent=2, allow_nan=False)


def _write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No predictions were produced")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    seed_everything(config.seed)
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = SyntheticSurfaceDataset(config.data, seed=config.seed)
    split = split_by_surface(
        dataset.samples,
        train_fraction=config.split.train_fraction,
        validation_fraction=config.split.validation_fraction,
        seed=config.seed,
    )
    loaders = {
        "train": _loader(
            dataset,
            split.train,
            config.training.batch_size,
            True,
            config.training.num_workers,
            config.seed,
        ),
        "validation": _loader(
            dataset,
            split.validation,
            config.training.batch_size,
            False,
            config.training.num_workers,
            config.seed + 1,
        ),
        "test": _loader(
            dataset,
            split.test,
            config.training.batch_size,
            False,
            config.training.num_workers,
            config.seed + 2,
        ),
    }

    model = BiSurfGNN(config.data, config.model).to(device)
    criterion = BiSurfLoss(
        monotherapy_weight=config.training.monotherapy_weight,
        variance_weight=config.training.variance_weight,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    _write_json(output_dir / "config.json", config.to_dict())
    _write_json(
        output_dir / "split_manifest.json",
        {
            "train_surfaces": split.train_surfaces,
            "validation_surfaces": split.validation_surfaces,
            "test_surfaces": split.test_surfaces,
            "train_points": len(split.train),
            "validation_points": len(split.validation),
            "test_points": len(split.test),
        },
    )
    _write_json(
        output_dir / "environment.json",
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
        },
    )

    best_validation_rmse = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    for epoch in range(1, config.training.epochs + 1):
        train_loss = _train_epoch(
            model,
            loaders["train"],
            optimizer,
            criterion,
            device,
            config.training.grad_clip_norm,
        )
        validation_metrics, _ = evaluate(model, loaders["validation"], criterion, device)
        record = {
            "epoch": epoch,
            "train": train_loss,
            "validation": validation_metrics,
        }
        history.append(record)
        print(json.dumps(_json_safe(record), ensure_ascii=False), flush=True)
        if validation_metrics["surface_rmse_mean"] < best_validation_rmse:
            best_validation_rmse = validation_metrics["surface_rmse_mean"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint")
    model.load_state_dict(best_state)

    final_metrics: dict[str, dict[str, float]] = {}
    prediction_rows: dict[str, list[dict[str, Any]]] = {}
    for name, loader in loaders.items():
        final_metrics[name], prediction_rows[name] = evaluate(model, loader, criterion, device)

    checkpoint = {
        "model_state": {key: value.cpu() for key, value in best_state.items()},
        "config": config.to_dict(),
        "best_epoch": best_epoch,
        "best_validation_surface_rmse": best_validation_rmse,
    }
    torch.save(checkpoint, output_dir / "model.pt")
    _write_json(output_dir / "history.json", history)
    result = {
        "status": "completed",
        "best_epoch": best_epoch,
        "best_validation_surface_rmse": best_validation_rmse,
        "metrics": final_metrics,
        "output_dir": str(output_dir),
    }
    _write_json(output_dir / "metrics.json", result)
    for split_name, rows in prediction_rows.items():
        _write_predictions(output_dir / f"predictions_{split_name}.csv", rows)
    return result

