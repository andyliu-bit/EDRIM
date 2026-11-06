from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable

import numpy as np


def _pearson(target: np.ndarray, prediction: np.ndarray) -> float:
    if target.size < 2 or np.std(target) < 1e-12 or np.std(prediction) < 1e-12:
        return float("nan")
    return float(np.corrcoef(target, prediction)[0, 1])


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _spearman(target: np.ndarray, prediction: np.ndarray) -> float:
    return _pearson(_average_ranks(target), _average_ranks(prediction))


def _finite_mean(values: Iterable[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - target
    return {
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(np.abs(error))),
        "pearson": _pearson(target, prediction),
        "spearman": _spearman(target, prediction),
    }


def summarize_predictions(
    targets: Iterable[float],
    predictions: Iterable[float],
    surface_ids: Iterable[int],
) -> dict[str, float]:
    target = np.asarray(list(targets), dtype=np.float64)
    prediction = np.asarray(list(predictions), dtype=np.float64)
    surfaces = np.asarray(list(surface_ids), dtype=np.int64)
    if target.shape != prediction.shape or target.shape != surfaces.shape:
        raise ValueError("Targets, predictions, and surface IDs must have identical shapes")
    point = regression_metrics(target, prediction)

    grouped: dict[int, list[int]] = defaultdict(list)
    for index, surface_id in enumerate(surfaces.tolist()):
        grouped[surface_id].append(index)
    per_surface = [
        regression_metrics(target[indexes], prediction[indexes])
        for indexes in grouped.values()
    ]
    result = {f"point_{key}": value for key, value in point.items()}
    for key in ("rmse", "mae", "pearson", "spearman"):
        result[f"surface_{key}_mean"] = _finite_mean(
            metric[key] for metric in per_surface
        )
    result["num_points"] = float(target.size)
    result["num_surfaces"] = float(len(grouped))
    return result

