"""Parameter-free observed-Bliss benchmark for the frozen scaffold protocol.

This module deliberately separates evaluation from split construction.  The core
functions accept in-memory point and assignment tables, making the metric contract
testable without loading the NCI-ALMANAC assets.  The file-backed runner imports the
chemical split loader lazily so importing this module never prepares or mutates a
split.

The endpoint currently has an explicit decision status supplied by the benchmark
configuration.  Results emitted here are engineering benchmark evidence only; they
do not imply causal validity, scientific endpoint validity, or state of the art.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .benchmark_config import BenchmarkConfig, load_benchmark_config


class BenchmarkEvaluationError(ValueError):
    """Raised when a benchmark ledger violates the fail-closed contract."""


GROUP_COLUMN = "prediction_group_id"
SURFACE_COLUMN = "surface_replicate_id"
CLUSTER_COLUMNS = ("heldout_cluster_id", "designated_cluster_id", "cluster_id")
TARGET_COLUMN = "response"
SOURCE_POINT_COLUMN = "source_combodrugseq"
DOSE_A_COLUMN = "dose_a_model"
DOSE_B_COLUMN = "dose_b_model"
MONO_A_COLUMN = "mono_response_a"
MONO_B_COLUMN = "mono_response_b"
MONO_MASK_A_COLUMN = "mono_mask_a"
MONO_MASK_B_COLUMN = "mono_mask_b"
RESPONSE_MASK_COLUMN = "response_mask"
PRIMARY_MASK_COLUMN = "primary_metric_mask"

POINT_COLUMNS = (
    SOURCE_POINT_COLUMN,
    GROUP_COLUMN,
    SURFACE_COLUMN,
    DOSE_A_COLUMN,
    DOSE_B_COLUMN,
    TARGET_COLUMN,
    MONO_A_COLUMN,
    MONO_B_COLUMN,
    MONO_MASK_A_COLUMN,
    MONO_MASK_B_COLUMN,
    RESPONSE_MASK_COLUMN,
    PRIMARY_MASK_COLUMN,
)

PREDICTION_LEDGER_COLUMNS = (
    "fold",
    "reference_model",
    SOURCE_POINT_COLUMN,
    GROUP_COLUMN,
    SURFACE_COLUMN,
    "cluster_id",
    DOSE_A_COLUMN,
    DOSE_B_COLUMN,
    "target",
    "prediction",
    "residual",
)

REFERENCE_MODELS: dict[str, dict[str, str | int]] = {
    "observed_bliss": {
        "name": "ObservedProductReference",
        "compatibility_alias": "ObservedBlissReference",
        "formula": "mono_response_a * mono_response_b",
        "interpretation": (
            "product_reference_on_clipped_growth_ratios; scientific Bliss "
            "interpretation is provisional"
        ),
        "trainable_parameters": 0,
    },
    "observed_hsa": {
        "name": "ObservedHSAReference",
        "formula": "min(mono_response_a, mono_response_b)",
        "interpretation": (
            "minimum-single-agent reference on clipped growth ratios; endpoint "
            "interpretation is provisional"
        ),
        "trainable_parameters": 0,
    },
}


def _require_columns(frame: pd.DataFrame, required: Sequence[str], label: str) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise BenchmarkEvaluationError(f"{label} is missing required columns: {missing}")


def _boolean_mask(values: pd.Series, label: str) -> pd.Series:
    """Parse a mask without allowing truthy non-empty strings."""

    if values.isna().any():
        raise BenchmarkEvaluationError(f"{label} contains missing values")
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.astype(bool)
    if pd.api.types.is_numeric_dtype(values.dtype):
        numeric = pd.to_numeric(values, errors="coerce")
        if numeric.isna().any() or not numeric.isin((0, 1)).all():
            raise BenchmarkEvaluationError(f"{label} must contain only boolean or 0/1 values")
        return numeric.astype(bool)
    normalized = values.astype(str).str.strip().str.casefold()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    if not normalized.isin(mapping).all():
        raise BenchmarkEvaluationError(f"{label} must contain only boolean or 0/1 values")
    return normalized.map(mapping).astype(bool)


def observed_bliss_prediction(points: pd.DataFrame) -> np.ndarray:
    """Return the observed Bliss reference, failing closed on either mono arm.

    The reference is the product of the two *observed* monotherapy responses.  No
    clipping is applied here: endpoint transformations belong to data preparation,
    and silently applying another transform would change the frozen endpoint.
    """

    _require_columns(
        points,
        (MONO_A_COLUMN, MONO_B_COLUMN, MONO_MASK_A_COLUMN, MONO_MASK_B_COLUMN),
        "points",
    )
    mask_a = _boolean_mask(points[MONO_MASK_A_COLUMN], MONO_MASK_A_COLUMN)
    mask_b = _boolean_mask(points[MONO_MASK_B_COLUMN], MONO_MASK_B_COLUMN)
    if not bool((mask_a & mask_b).all()):
        missing = int((~(mask_a & mask_b)).sum())
        raise BenchmarkEvaluationError(
            "Observed-Bliss requires both monotherapy observations for every "
            f"evaluated point; {missing} rows are incomplete"
        )
    mono_a = pd.to_numeric(points[MONO_A_COLUMN], errors="coerce").to_numpy(float)
    mono_b = pd.to_numeric(points[MONO_B_COLUMN], errors="coerce").to_numpy(float)
    if not np.isfinite(mono_a).all() or not np.isfinite(mono_b).all():
        raise BenchmarkEvaluationError("Observed monotherapy values must all be finite")
    return mono_a * mono_b


def observed_hsa_prediction(points: pd.DataFrame) -> np.ndarray:
    """Return the minimum observed single-agent response for every point."""

    # Reuse the product implementation for strict mask, numeric, and finite checks.
    observed_bliss_prediction(points)
    mono_a = pd.to_numeric(points[MONO_A_COLUMN], errors="coerce").to_numpy(float)
    mono_b = pd.to_numeric(points[MONO_B_COLUMN], errors="coerce").to_numpy(float)
    return np.minimum(mono_a, mono_b)


def _reference_prediction(points: pd.DataFrame, reference_model: str) -> np.ndarray:
    if reference_model == "observed_bliss":
        return observed_bliss_prediction(points)
    if reference_model == "observed_hsa":
        return observed_hsa_prediction(points)
    supported = ", ".join(sorted(REFERENCE_MODELS))
    raise BenchmarkEvaluationError(
        f"Unsupported reference model {reference_model!r}; expected one of: {supported}"
    )


def _pearson(target: np.ndarray, prediction: np.ndarray) -> float:
    if target.size < 2 or np.std(target) < 1e-12 or np.std(prediction) < 1e-12:
        return float("nan")
    return float(np.corrcoef(target, prediction)[0, 1])


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ordered = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and ordered[end] == ordered[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    if target.size == 0:
        raise BenchmarkEvaluationError("Cannot evaluate an empty prediction set")
    error = prediction - target
    return {
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(np.abs(error))),
        "pearson": _pearson(target, prediction),
        "spearman": _pearson(_average_ranks(target), _average_ranks(prediction)),
    }


def _finite_mean(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def _macro_metrics(
    ledger: pd.DataFrame,
    group_column: str,
) -> dict[str, float | int]:
    grouped_metrics: list[dict[str, float]] = []
    for _, group in ledger.groupby(group_column, sort=True, dropna=False):
        grouped_metrics.append(
            _regression_metrics(
                group["target"].to_numpy(float),
                group["prediction"].to_numpy(float),
            )
        )
    result: dict[str, float | int] = {
        key: _finite_mean([metric[key] for metric in grouped_metrics])
        for key in ("rmse", "mae", "pearson", "spearman")
    }
    result["count"] = len(grouped_metrics)
    for metric in ("pearson", "spearman"):
        valid = sum(math.isfinite(float(values[metric])) for values in grouped_metrics)
        result[f"{metric}_valid_count"] = valid
        result[f"{metric}_undefined_count"] = len(grouped_metrics) - valid
    return result


def summarize_reference_ledger(ledger: pd.DataFrame) -> dict[str, Any]:
    """Summarize one fold or a disjoint pooled ledger.

    ``surface_macro`` is the primary metric family: a metric is calculated within
    each surface and then averaged with equal surface weight. ``cluster_macro``
    analogously gives each held-out chemical cluster equal weight after calculating
    its metric across all of that cluster's points.
    """

    _require_columns(ledger, PREDICTION_LEDGER_COLUMNS, "prediction ledger")
    target = ledger["target"].to_numpy(float)
    prediction = ledger["prediction"].to_numpy(float)
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise BenchmarkEvaluationError("Prediction ledger contains non-finite values")
    point = _regression_metrics(target, prediction)
    point["count"] = int(len(ledger))
    point["pearson_valid_count"] = int(math.isfinite(point["pearson"]))
    point["pearson_undefined_count"] = 1 - point["pearson_valid_count"]
    point["spearman_valid_count"] = int(math.isfinite(point["spearman"]))
    point["spearman_undefined_count"] = 1 - point["spearman_valid_count"]
    surface = _macro_metrics(ledger, SURFACE_COLUMN)
    prediction_group = _macro_metrics(ledger, GROUP_COLUMN)
    cluster = _macro_metrics(ledger, "cluster_id")
    return {
        "primary_metric": {
            "name": "surface_macro.rmse",
            "value": surface["rmse"],
            "lower_is_better": True,
        },
        "surface_macro": surface,
        "prediction_group_macro": prediction_group,
        "point_micro": point,
        "cluster_macro": cluster,
    }


def summarize_observed_bliss_ledger(ledger: pd.DataFrame) -> dict[str, Any]:
    """Compatibility wrapper for the original public function name."""

    return summarize_reference_ledger(ledger)


def _assignment_cluster_column(assignments: pd.DataFrame) -> str:
    available = [column for column in CLUSTER_COLUMNS if column in assignments.columns]
    if not available:
        raise BenchmarkEvaluationError(
            "assignments must contain heldout_cluster_id, designated_cluster_id, or cluster_id"
        )
    return available[0]


def _assignment_join_columns(assignments: pd.DataFrame) -> list[str]:
    _require_columns(assignments, (GROUP_COLUMN,), "assignments")
    return (
        [GROUP_COLUMN, SURFACE_COLUMN]
        if SURFACE_COLUMN in assignments.columns
        else [GROUP_COLUMN]
    )


def _validate_fold_test_disjointness(
    assignments: pd.DataFrame,
    folds: int,
) -> None:
    role_columns = [f"role_fold_{fold}" for fold in range(folds)]
    _require_columns(assignments, role_columns, "assignments")
    join_columns = _assignment_join_columns(assignments)
    normalized = assignments[join_columns + role_columns].drop_duplicates()
    for identity_columns, label in (
        ([GROUP_COLUMN], "prediction group"),
        (join_columns, "surface" if len(join_columns) == 2 else "prediction group"),
    ):
        fold_sets: dict[tuple[Any, ...], set[int]] = {}
        for fold, role_column in enumerate(role_columns):
            test_rows = normalized.loc[
                normalized[role_column].astype(str).str.casefold().eq("test"),
                identity_columns,
            ].drop_duplicates()
            for values in test_rows.itertuples(index=False, name=None):
                fold_sets.setdefault(tuple(values), set()).add(fold)
        overlaps = [key for key, seen in fold_sets.items() if len(seen) > 1]
        if overlaps:
            raise BenchmarkEvaluationError(
                f"Test {label} memberships are not pairwise disjoint across folds; "
                f"first overlap: {overlaps[0]}"
            )
        if len(join_columns) == 1:
            break


def _fold_test_ledger(
    points: pd.DataFrame,
    assignments: pd.DataFrame,
    fold: int,
    reference_model: str,
) -> pd.DataFrame:
    _require_columns(points, POINT_COLUMNS, "points")
    role_column = f"role_fold_{fold}"
    _require_columns(assignments, (role_column,), "assignments")
    join_columns = _assignment_join_columns(assignments)
    cluster_column = _assignment_cluster_column(assignments)

    assignment_columns = join_columns + [cluster_column, role_column]
    selected_assignments = assignments.loc[
        assignments[role_column].astype(str).str.casefold().eq("test"),
        assignment_columns,
    ].copy()
    if selected_assignments.empty:
        raise BenchmarkEvaluationError(f"Fold {fold} has no test assignments")
    conflicts = selected_assignments.groupby(join_columns, dropna=False).agg(
        clusters=(cluster_column, "nunique"),
        roles=(role_column, "nunique"),
    )
    if (conflicts[["clusters", "roles"]] != 1).any(axis=None):
        raise BenchmarkEvaluationError(f"Fold {fold} contains conflicting assignment rows")
    selected_assignments = selected_assignments.drop_duplicates(join_columns)

    eligible_points = points.loc[
        _boolean_mask(points[PRIMARY_MASK_COLUMN], PRIMARY_MASK_COLUMN)
        & _boolean_mask(points[RESPONSE_MASK_COLUMN], RESPONSE_MASK_COLUMN)
    ].copy()
    selected = eligible_points.merge(
        selected_assignments,
        on=join_columns,
        how="inner",
        validate="many_to_one",
    )
    if selected.empty:
        raise BenchmarkEvaluationError(f"Fold {fold} has no evaluable primary test points")

    expected_keys = set(
        selected_assignments[join_columns].itertuples(index=False, name=None)
    )
    observed_keys = set(selected[join_columns].drop_duplicates().itertuples(index=False, name=None))
    missing_keys = expected_keys.difference(observed_keys)
    if missing_keys:
        raise BenchmarkEvaluationError(
            f"Fold {fold} has test assignments without primary response points; "
            f"first missing key: {next(iter(missing_keys))}"
        )

    target = pd.to_numeric(selected[TARGET_COLUMN], errors="coerce").to_numpy(float)
    if not np.isfinite(target).all():
        raise BenchmarkEvaluationError("Evaluated target values must all be finite")
    source_ids = pd.to_numeric(
        selected[SOURCE_POINT_COLUMN], errors="coerce"
    ).to_numpy(float)
    dose_a = pd.to_numeric(selected[DOSE_A_COLUMN], errors="coerce").to_numpy(float)
    dose_b = pd.to_numeric(selected[DOSE_B_COLUMN], errors="coerce").to_numpy(float)
    if not np.isfinite(source_ids).all() or not np.isfinite(dose_a).all() or not np.isfinite(
        dose_b
    ).all():
        raise BenchmarkEvaluationError("Point IDs and model doses must all be finite")
    if selected[SOURCE_POINT_COLUMN].duplicated().any():
        raise BenchmarkEvaluationError(f"Fold {fold} contains duplicate source point IDs")
    prediction = _reference_prediction(selected, reference_model)
    ledger = pd.DataFrame(
        {
            "fold": np.full(len(selected), fold, dtype=np.int16),
            "reference_model": reference_model,
            SOURCE_POINT_COLUMN: source_ids.astype(np.int64),
            GROUP_COLUMN: selected[GROUP_COLUMN].to_numpy(),
            SURFACE_COLUMN: selected[SURFACE_COLUMN].to_numpy(),
            "cluster_id": selected[cluster_column].to_numpy(),
            DOSE_A_COLUMN: dose_a,
            DOSE_B_COLUMN: dose_b,
            "target": target,
            "prediction": prediction,
            "residual": prediction - target,
        }
    )
    return ledger.loc[:, PREDICTION_LEDGER_COLUMNS]


def _aggregate_fold_metrics(fold_metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for family in (
        "surface_macro",
        "prediction_group_macro",
        "point_micro",
        "cluster_macro",
    ):
        aggregate[family] = {"mean": {}, "std": {}}
        for metric in ("rmse", "mae", "pearson", "spearman"):
            values = np.asarray(
                [float(fold[family][metric]) for fold in fold_metrics],
                dtype=np.float64,
            )
            finite = values[np.isfinite(values)]
            aggregate[family]["mean"][metric] = (
                float(np.mean(finite)) if finite.size else float("nan")
            )
            aggregate[family]["std"][metric] = (
                float(np.std(finite, ddof=1)) if finite.size > 1 else float("nan")
            )
    return aggregate


def evaluate_reference_benchmark(
    points: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    reference_model: str,
    folds: int = 5,
    endpoint: str = "clipped_PERCENTGROWTHNOTZ_over_100",
    endpoint_decision_status: str = "provisional_engineering",
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate a frozen, parameter-free observed-monotherapy reference."""

    if reference_model not in REFERENCE_MODELS:
        _reference_prediction(points.iloc[:0], reference_model)
    if folds != 5:
        raise BenchmarkEvaluationError("The primary benchmark requires exactly five folds")
    if not endpoint or not endpoint_decision_status:
        raise BenchmarkEvaluationError("Endpoint and endpoint decision status must be explicit")
    _validate_fold_test_disjointness(assignments, folds)

    ledgers: list[pd.DataFrame] = []
    per_fold: list[dict[str, Any]] = []
    for fold in range(folds):
        ledger = _fold_test_ledger(points, assignments, fold, reference_model)
        metrics = summarize_reference_ledger(ledger)
        per_fold.append({"fold": fold, "metrics": metrics})
        ledgers.append(ledger)

    pooled = pd.concat(ledgers, ignore_index=True)
    pooled_metrics = summarize_reference_ledger(pooled)
    fold_metric_payload = [entry["metrics"] for entry in per_fold]
    aggregate = _aggregate_fold_metrics(fold_metric_payload)
    aggregate["pooled"] = pooled_metrics
    model = dict(REFERENCE_MODELS[reference_model])
    model["slug"] = reference_model
    model["information_setting"] = "observed_monotherapy_anchored"
    result = {
        "status": "completed_engineering_benchmark",
        "claim_scope": "engineering_only_no_scientific_or_sota_claim",
        "model": model,
        "endpoint": endpoint,
        "endpoint_decision_status": endpoint_decision_status,
        "fold_count": folds,
        "fold_standard_deviation": "sample_sd_ddof_1",
        "residual_convention": "prediction_minus_target",
        "folds": per_fold,
        "aggregate": aggregate,
    }
    return result, pooled.loc[:, PREDICTION_LEDGER_COLUMNS]


def evaluate_observed_bliss_benchmark(
    points: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    folds: int = 5,
    endpoint: str = "clipped_PERCENTGROWTHNOTZ_over_100",
    endpoint_decision_status: str = "provisional_engineering",
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate observed Bliss on pairwise-disjoint outer-fold test ledgers."""

    return evaluate_reference_benchmark(
        points,
        assignments,
        reference_model="observed_bliss",
        folds=folds,
        endpoint=endpoint,
        endpoint_decision_status=endpoint_decision_status,
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(_json_ready(payload), handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _write_parquet_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(handle)
    temporary = Path(name)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _file_record(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return {
        "path": str(path.resolve()),
        "bytes": int(path.stat().st_size),
        "sha256": digest.hexdigest(),
    }


def _read_primary_test_points_by_row_group(
    points_path: Path,
    assignments: pd.DataFrame,
    folds: int,
) -> pd.DataFrame:
    """Read only outer-test points while bounding each source-table read.

    NCI-ALMANAC's prepared point table expands to several gigabytes as one
    pandas frame. The benchmark therefore scans one Parquet row group at a time
    and retains only surfaces that are a test member in one of the frozen folds.
    """

    import pyarrow.parquet as parquet

    _validate_fold_test_disjointness(assignments, folds)
    _require_columns(assignments, (SURFACE_COLUMN,), "surface assignments")
    test_surface_ids: set[str] = set()
    for fold in range(folds):
        role_column = f"role_fold_{fold}"
        selected = assignments.loc[
            assignments[role_column].astype(str).str.casefold().eq("test"),
            SURFACE_COLUMN,
        ]
        test_surface_ids.update(selected.astype(str))
    if not test_surface_ids:
        raise BenchmarkEvaluationError("Frozen assignments contain no outer-test surfaces")

    parquet_file = parquet.ParquetFile(points_path)
    missing_columns = sorted(set(POINT_COLUMNS) - set(parquet_file.schema_arrow.names))
    if missing_columns:
        raise BenchmarkEvaluationError(
            f"Prepared point table is missing required columns: {missing_columns}"
        )
    chunks: list[pd.DataFrame] = []
    for row_group in range(parquet_file.metadata.num_row_groups):
        chunk = parquet_file.read_row_group(
            row_group,
            columns=list(POINT_COLUMNS),
        ).to_pandas()
        selected = chunk[SURFACE_COLUMN].astype(str).isin(test_surface_ids)
        if bool(selected.any()):
            chunks.append(chunk.loc[selected].copy())
    if not chunks:
        raise BenchmarkEvaluationError("No prepared points match the frozen test surfaces")
    return pd.concat(chunks, ignore_index=True)


def _artifact_path(
    manifest: Mapping[str, Any],
    base_dir: Path,
    *names: str,
) -> Path | None:
    artifacts = manifest.get("artifacts")
    for source in (artifacts, manifest):
        if not isinstance(source, Mapping):
            continue
        for name in names:
            value = source.get(name)
            if isinstance(value, Mapping):
                value = value.get("path") or value.get("filename")
            if value:
                candidate = Path(str(value))
                return candidate if candidate.is_absolute() else (base_dir / candidate).resolve()
    return None


def _load_verified_manifest(config: BenchmarkConfig) -> Mapping[str, Any]:
    # Delayed import prevents a split-preparation dependency cycle and makes the
    # in-memory evaluator independently reusable.
    from .data import chemical_splits

    loader = getattr(chemical_splits, "load_benchmark_manifest", None)
    if loader is None:
        loader = getattr(chemical_splits, "verify_benchmark_manifest", None)
    if loader is None:
        raise BenchmarkEvaluationError(
            "chemical_splits does not expose load_benchmark_manifest or "
            "verify_benchmark_manifest"
        )
    loaded = loader(config)
    if isinstance(loaded, tuple):
        loaded = loaded[0]
    if not isinstance(loaded, Mapping):
        raise BenchmarkEvaluationError("Verified benchmark manifest must be a mapping")
    return loaded


def run_observed_bliss_benchmark(
    config: BenchmarkConfig | str | Path,
    *,
    manifest: Mapping[str, Any] | None = None,
    points: pd.DataFrame | None = None,
    assignments: pd.DataFrame | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Compatibility wrapper for the product-reference benchmark."""

    return run_reference_benchmark(
        config,
        reference_model="observed_bliss",
        manifest=manifest,
        points=points,
        assignments=assignments,
        output_dir=output_dir,
    )


def run_reference_benchmark(
    config: BenchmarkConfig | str | Path,
    *,
    reference_model: str,
    manifest: Mapping[str, Any] | None = None,
    points: pd.DataFrame | None = None,
    assignments: pd.DataFrame | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run and persist one parameter-free reference benchmark.

    When in-memory frames are omitted, the split manifest is loaded through the
    chemical split module's verifier, assignments are resolved from the manifest,
    and prepared points are read from the frozen NCI processed directory.
    """

    benchmark_config = (
        config if isinstance(config, BenchmarkConfig) else load_benchmark_config(config)
    )
    loaded_manifest = manifest
    if loaded_manifest is None and (points is None or assignments is None):
        loaded_manifest = _load_verified_manifest(benchmark_config)

    if points is None:
        points_path = benchmark_config.nci_processed_dir / "points.parquet"
        if not points_path.is_file():
            raise BenchmarkEvaluationError(f"Prepared point table not found: {points_path}")
    else:
        points_path = None

    if assignments is None:
        if loaded_manifest is None:
            raise BenchmarkEvaluationError("A verified manifest is required to locate assignments")
        assignment_path = _artifact_path(
            loaded_manifest,
            benchmark_config.output_dir,
            "surface_assignments",
            "surface_assignments_path",
            "assignment_ledger",
        )
        if assignment_path is None or not assignment_path.is_file():
            raise BenchmarkEvaluationError(
                "Verified manifest does not resolve an existing surface assignment ledger"
            )
        if assignment_path.suffix.casefold() == ".csv":
            assignments = pd.read_csv(assignment_path)
        else:
            assignments = pd.read_parquet(assignment_path)

    if points is None:
        if points_path is None:
            raise AssertionError("points_path must exist when points are file-backed")
        points = _read_primary_test_points_by_row_group(
            points_path,
            assignments,
            benchmark_config.folds,
        )

    result, prediction_ledger = evaluate_reference_benchmark(
        points,
        assignments,
        reference_model=reference_model,
        folds=benchmark_config.folds,
        endpoint=benchmark_config.endpoint,
        endpoint_decision_status=benchmark_config.endpoint_decision_status,
    )
    if loaded_manifest is not None:
        result["split_manifest_hash"] = (
            loaded_manifest.get("manifest_sha256")
            or loaded_manifest.get("content_hash")
            or loaded_manifest.get("split_hash")
        )
        result["split_hash"] = loaded_manifest.get("split_hash")
        result["protocol"] = loaded_manifest.get("protocol", benchmark_config.protocol)
    else:
        result["split_manifest_hash"] = None
        result["split_hash"] = None
        result["protocol"] = benchmark_config.protocol
    if points_path is not None:
        result["evaluation_source"] = {"points": _file_record(points_path)}

    destination = (
        Path(output_dir).resolve()
        if output_dir is not None
        else benchmark_config.runs_dir / reference_model
    )
    predictions_path = destination / "predictions.parquet"
    metrics_path = destination / "metrics.json"
    run_manifest_path = destination / "run_manifest.json"
    _write_parquet_atomic(predictions_path, prediction_ledger)
    result["artifacts"] = {
        "predictions": str(predictions_path),
        "metrics": str(metrics_path),
        "run_manifest": str(run_manifest_path),
    }
    result["artifact_records"] = {
        "predictions": {
            **_file_record(predictions_path),
            "rows": int(len(prediction_ledger)),
        }
    }
    _write_json_atomic(metrics_path, result)
    run_manifest: dict[str, Any] = {
        "protocol": result["protocol"],
        "reference_model": reference_model,
        "endpoint": benchmark_config.endpoint,
        "endpoint_decision_status": benchmark_config.endpoint_decision_status,
        "split_manifest_hash": result["split_manifest_hash"],
        "split_hash": result["split_hash"],
        "evaluation_source": result.get("evaluation_source"),
        "artifacts": {
            "predictions": result["artifact_records"]["predictions"],
            "metrics": _file_record(metrics_path),
        },
    }
    run_manifest["content_hash"] = hashlib.sha256(
        json.dumps(
            run_manifest,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    _write_json_atomic(run_manifest_path, run_manifest)
    return result


__all__ = [
    "BenchmarkEvaluationError",
    "PREDICTION_LEDGER_COLUMNS",
    "REFERENCE_MODELS",
    "evaluate_reference_benchmark",
    "evaluate_observed_bliss_benchmark",
    "observed_bliss_prediction",
    "observed_hsa_prediction",
    "run_reference_benchmark",
    "run_observed_bliss_benchmark",
    "summarize_reference_ledger",
    "summarize_observed_bliss_ledger",
]
