"""Validation-only Morgan control for the frozen warm PairOut discriminator."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from .benchmark_config import load_benchmark_config
from .data.benchmark_context import load_benchmark_cell_context
from .data.cellminer import load_cellminer_config
from .learned_benchmark import (
    HALLMARK_DIMENSION,
    LEARNED_BENCHMARK_VERSION,
    MODEL_SLUG,
    LearnedMorganConfig,
    _canonical_hash,
    _feature_bank_hash,
    _file_record,
    _runtime_environment,
    _selection_code_paths,
    _verify_recorded_file,
    _write_checkpoint_atomic,
    _write_json_atomic,
    build_compact_point_table,
    build_feature_banks,
    fit_fold_seed,
    load_learned_morgan_config,
    surface_macro_rmse,
)
from .models.baselines import MorganHallmarkResidualMLP
from .warm_pairout import load_warm_pairout_config, load_warm_pairout_probe

WARM_MORGAN_VERSION = "warm-pairout-morgan-selection-v1"
SELECTION_RUN_MANIFEST = "selection_run_manifest.json"


class WarmMorganError(RuntimeError):
    """Raised when warm Morgan selection violates its frozen contract."""


def _selected(values: Sequence[int] | None, allowed: Sequence[int], label: str) -> tuple[int, ...]:
    selected = tuple(allowed) if values is None else tuple(int(value) for value in values)
    if not selected or len(set(selected)) != len(selected):
        raise WarmMorganError(f"{label} must be a non-empty unique sequence")
    unknown = sorted(set(selected) - set(allowed))
    if unknown:
        raise WarmMorganError(f"Unknown {label}: {unknown}")
    return selected


def _sample_complete_surfaces(
    frame: pd.DataFrame,
    max_train_points: int | None,
    *,
    seed: int,
) -> pd.DataFrame:
    if max_train_points is None or len(frame) <= max_train_points:
        return frame.sort_values("source_combodrugseq").reset_index(drop=True)
    if max_train_points < 1:
        raise WarmMorganError("max_train_points must be positive")
    surfaces = sorted(set(frame["surface_replicate_id"].astype(str)))
    ranks = {
        surface: hashlib.sha256(f"{seed}|{surface}".encode("utf-8")).hexdigest()
        for surface in surfaces
    }
    sizes = frame.groupby("surface_replicate_id", sort=True).size().to_dict()
    chosen: list[str] = []
    points = 0
    for surface in sorted(surfaces, key=ranks.get):
        size = int(sizes[surface])
        if points + size <= max_train_points:
            chosen.append(surface)
            points += size
    if not chosen:
        raise WarmMorganError("max_train_points is smaller than one complete surface")
    return (
        frame.loc[frame["surface_replicate_id"].astype(str).isin(chosen)]
        .sort_values("source_combodrugseq")
        .reset_index(drop=True)
    )


def _anchor_rmse(frame: pd.DataFrame) -> float:
    target = frame["response"].to_numpy(dtype=np.float64)
    reference = np.minimum(
        frame["mono_response_a"].to_numpy(dtype=np.float64),
        frame["mono_response_b"].to_numpy(dtype=np.float64),
    )
    return surface_macro_rmse(
        target,
        reference,
        frame["surface_replicate_id"].astype(str).to_numpy(object),
    )


def _parameter_count(training: LearnedMorganConfig, fingerprint_dim: int) -> int:
    model = MorganHallmarkResidualMLP(
        fingerprint_dim=fingerprint_dim,
        hallmark_dim=HALLMARK_DIMENSION,
        hidden_dim=training.hidden_dim,
        dropout=training.dropout,
        reference_mode=training.reference_mode,
    )
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _code_paths() -> dict[str, Path]:
    paths = dict(_selection_code_paths())
    paths["warm_pairout"] = Path(__file__).with_name("warm_pairout.py")
    paths["warm_morgan"] = Path(__file__)
    return paths


def _freeze_job(
    destination: Path,
    *,
    fold: int,
    seed: int,
    selection: Mapping[str, Any],
    state: Mapping[str, Tensor],
    training: LearnedMorganConfig,
    warm_manifest_hash: str,
    context_split_hash: str,
    context_state_hash: str,
    feature_bank_hash: str,
    train_shard: Mapping[str, Any],
    validation_shard: Mapping[str, Any],
) -> tuple[Path, Path, dict[str, Any]]:
    checkpoint_path = destination / "checkpoint.pt"
    selection_path = destination / "selection_manifest.json"
    semantic = {
        "version": WARM_MORGAN_VERSION,
        "base_model_version": LEARNED_BENCHMARK_VERSION,
        "model": MODEL_SLUG,
        "protocol": "WarmPairOutDevelopment-v1",
        "fold": fold,
        "seed": seed,
        "best_epoch": int(selection["best_epoch"]),
        "warm_manifest_hash": warm_manifest_hash,
        "context_split_hash": context_split_hash,
        "context_state_hash": context_state_hash,
        "feature_bank_hash": feature_bank_hash,
        "training_config": training.to_dict(),
        "runtime_environment": _runtime_environment(),
        "train_shard": dict(train_shard),
        "validation_shard": dict(validation_shard),
        "selection_hash": _canonical_hash(selection),
        "test_response_rows_read_during_selection": 0,
        "test_evaluations_during_selection": 0,
    }
    _write_checkpoint_atomic(
        checkpoint_path,
        {**semantic, "state_dict": copy.deepcopy(dict(state))},
    )
    manifest: dict[str, Any] = {
        "status": "warm_selection_frozen_no_test_shard_available",
        **semantic,
        "selection": dict(selection),
        "checkpoint": _file_record(checkpoint_path),
        "code_sources": {
            name: _file_record(path) for name, path in _code_paths().items()
        },
    }
    manifest["manifest_sha256"] = _canonical_hash(manifest)
    _write_json_atomic(selection_path, manifest)
    return checkpoint_path, selection_path, manifest


def _verify_job(checkpoint_path: Path, selection_path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(selection_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise WarmMorganError(f"Cannot read warm selection: {selection_path}") from error
    recorded = manifest.pop("manifest_sha256", None)
    if recorded != _canonical_hash(manifest):
        raise WarmMorganError("Warm selection manifest content hash changed")
    checkpoint = manifest.get("checkpoint")
    if not isinstance(checkpoint, dict) or _file_record(checkpoint_path) != checkpoint:
        raise WarmMorganError("Warm checkpoint record changed")
    code_sources = manifest.get("code_sources")
    if not isinstance(code_sources, dict):
        raise WarmMorganError("Warm selection has no code-source binding")
    for name, path in _code_paths().items():
        record = code_sources.get(name)
        if not isinstance(record, dict):
            raise WarmMorganError(f"Warm selection lacks {name} source binding")
        _verify_recorded_file(path, record, f"Warm selection {name} source")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise WarmMorganError("Warm checkpoint payload is invalid")
    semantic = (
        "version",
        "base_model_version",
        "model",
        "protocol",
        "fold",
        "seed",
        "best_epoch",
        "warm_manifest_hash",
        "context_split_hash",
        "context_state_hash",
        "feature_bank_hash",
        "training_config",
        "runtime_environment",
        "train_shard",
        "validation_shard",
        "selection_hash",
        "test_response_rows_read_during_selection",
        "test_evaluations_during_selection",
    )
    mismatched = [key for key in semantic if payload.get(key) != manifest.get(key)]
    if mismatched:
        raise WarmMorganError(f"Warm checkpoint semantic binding changed: {mismatched}")
    if manifest.get("runtime_environment") != _runtime_environment():
        raise WarmMorganError("Warm checkpoint runtime environment changed")
    selection = manifest.get("selection")
    if not isinstance(selection, dict) or _canonical_hash(selection) != manifest.get(
        "selection_hash"
    ):
        raise WarmMorganError("Warm selection payload hash changed")
    manifest["manifest_sha256"] = recorded
    return manifest


def routing_decision(
    jobs: Sequence[Mapping[str, Any]],
    *,
    complete_inventory: bool,
) -> dict[str, Any]:
    """Apply the predeclared development-only warm routing rule."""

    fold_rows = []
    for fold in sorted({int(job["fold"]) for job in jobs}):
        selected = [job for job in jobs if int(job["fold"]) == fold]
        model = float(np.mean([job["best_validation_surface_macro_rmse"] for job in selected]))
        anchor = float(np.mean([job["anchor_validation_surface_macro_rmse"] for job in selected]))
        improvement = (anchor - model) / anchor
        fold_rows.append(
            {
                "fold": fold,
                "model_seed_mean_surface_macro_rmse": model,
                "anchor_surface_macro_rmse": anchor,
                "relative_improvement": improvement,
                "clears_five_percent": improvement >= 0.05,
                "improves_directionally": improvement > 0,
            }
        )
    passing = sum(bool(row["clears_five_percent"]) for row in fold_rows)
    non_improving = sum(not bool(row["improves_directionally"]) for row in fold_rows)
    mean_model = float(np.mean([row["model_seed_mean_surface_macro_rmse"] for row in fold_rows]))
    mean_anchor = float(np.mean([row["anchor_surface_macro_rmse"] for row in fold_rows]))
    mean_improvement = (mean_anchor - mean_model) / mean_anchor
    if not complete_inventory:
        decision = "smoke_no_branch_decision"
    elif passing >= 4:
        decision = "chemical_ood_bottleneck_reopened"
    elif mean_improvement < 0.01 and non_improving >= 2:
        decision = "strengthens_endpoint_metric_limitation"
    else:
        decision = "ambiguous_run_response_stratified_validation"
    return {
        "decision": decision,
        "complete_five_fold_three_seed_inventory": complete_inventory,
        "folds": fold_rows,
        "folds_clearing_five_percent": passing,
        "non_improving_folds": non_improving,
        "fold_mean_model_surface_macro_rmse": mean_model,
        "fold_mean_anchor_surface_macro_rmse": mean_anchor,
        "fold_mean_relative_improvement": mean_improvement,
        "claim_scope": "development_routing_only_no_significance_claim",
        "test_action": "none",
    }


def select_warm_morgan_probe(
    warm_config: str | Path,
    benchmark_config: str | Path,
    *,
    cell_config: str | Path,
    folds: Sequence[int] | None = None,
    seeds: Sequence[int] | None = None,
    max_train_points: int | None = None,
    output_dir: str | Path = "runs/warm_pairout/morgan_hallmark_residual_mlp",
    device: str | torch.device | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Train and freeze warm validation selections without any test role."""

    warm_configuration = load_warm_pairout_config(warm_config)
    warm = load_warm_pairout_probe(warm_config)
    benchmark = load_benchmark_config(benchmark_config)
    cell = load_cellminer_config(cell_config)
    context = load_benchmark_cell_context(benchmark, cell)
    training_full = load_learned_morgan_config(benchmark_config)
    selected_folds = _selected(folds, tuple(warm["base_folds"]), "folds")
    selected_seeds = _selected(seeds, training_full.seeds, "seeds")
    training = replace(training_full, seeds=selected_seeds)
    destination = Path(output_dir).resolve()
    run_path = destination / SELECTION_RUN_MANIFEST
    if run_path.exists() and not overwrite:
        raise WarmMorganError(f"Warm selection already exists: {run_path}")

    warm_folds = {int(entry["fold"]): entry for entry in warm["folds"]}
    jobs: list[dict[str, Any]] = []
    parameter_count: int | None = None
    for fold in selected_folds:
        entry = warm_folds[fold]
        outputs = entry["outputs"]
        train_path = Path(outputs["train"]["path"])
        validation_path = Path(outputs["validation"]["path"])
        train_frame = pd.read_parquet(train_path)
        validation_frame = pd.read_parquet(validation_path)
        if set(train_frame["source_combodrugseq"]) & set(
            validation_frame["source_combodrugseq"]
        ):
            raise WarmMorganError(f"Fold {fold} warm source rows overlap")
        train_frame = _sample_complete_surfaces(
            train_frame,
            max_train_points,
            seed=int(warm["split_seed"]),
        )
        required_drugs = sorted(
            set(train_frame["drug_a_id"])
            | set(train_frame["drug_b_id"])
            | set(validation_frame["drug_a_id"])
            | set(validation_frame["drug_b_id"])
        )
        required_cells = sorted(
            set(train_frame["cell_id"]) | set(validation_frame["cell_id"])
        )
        banks = build_feature_banks(
            required_drugs,
            required_cells,
            context,
            fold,
            benchmark,
        )
        current_count = _parameter_count(training, banks.fingerprint_dim)
        if parameter_count is None:
            parameter_count = current_count
        elif parameter_count != current_count:
            raise WarmMorganError("Morgan parameter count changed across folds")
        train_table = build_compact_point_table(
            train_frame,
            banks,
            role="train",
            fold=fold,
            monotherapy_upper_bound=benchmark.response_upper_bound,
        )
        validation_table = build_compact_point_table(
            validation_frame,
            banks,
            role="validation",
            fold=fold,
            monotherapy_upper_bound=benchmark.response_upper_bound,
        )
        anchor = _anchor_rmse(validation_frame)
        recorded_anchor = float(entry["hsa_validation"]["rmse"])
        if not math.isclose(anchor, recorded_anchor, rel_tol=0, abs_tol=1e-12):
            raise WarmMorganError(f"Fold {fold} HSA anchor changed")
        bank_hash = _feature_bank_hash(banks)
        for seed in selected_seeds:
            selection, state = fit_fold_seed(
                fold,
                banks,
                train_table,
                validation_table,
                training,
                seed=seed,
                device=device,
            )
            selection["anchor_validation_surface_macro_rmse"] = anchor
            selection["validation_relative_improvement_vs_anchor"] = (
                anchor - float(selection["best_validation_surface_macro_rmse"])
            ) / anchor
            job_dir = destination / f"fold_{fold}" / f"seed_{seed}"
            checkpoint_path, selection_path, frozen = _freeze_job(
                job_dir,
                fold=fold,
                seed=seed,
                selection=selection,
                state=state,
                training=training,
                warm_manifest_hash=str(warm["manifest_sha256"]),
                context_split_hash=context.folds[fold].context_split_hash,
                context_state_hash=context.folds[fold].state.state_hash,
                feature_bank_hash=bank_hash,
                train_shard=outputs["train"],
                validation_shard=outputs["validation"],
            )
            verified = _verify_job(checkpoint_path, selection_path)
            if verified["manifest_sha256"] != frozen["manifest_sha256"]:
                raise WarmMorganError("Warm selection changed immediately after freezing")
            jobs.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "best_epoch": int(selection["best_epoch"]),
                    "epochs_ran": int(selection["epochs_ran"]),
                    "best_validation_surface_macro_rmse": float(
                        selection["best_validation_surface_macro_rmse"]
                    ),
                    "anchor_validation_surface_macro_rmse": anchor,
                    "validation_relative_improvement_vs_anchor": float(
                        selection["validation_relative_improvement_vs_anchor"]
                    ),
                    "point_counts": {
                        "train": len(train_table),
                        "validation": len(validation_table),
                    },
                    "surface_counts": {
                        "train": len(set(train_table.surface_ids)),
                        "validation": len(set(validation_table.surface_ids)),
                    },
                    "feature_bank_hash": bank_hash,
                    "checkpoint": _file_record(checkpoint_path),
                    "selection_manifest": _file_record(selection_path),
                }
            )
    complete = (
        selected_folds == tuple(warm["base_folds"])
        and selected_seeds == training_full.seeds
        and max_train_points is None
    )
    routing = routing_decision(jobs, complete_inventory=complete)
    manifest: dict[str, Any] = {
        "version": WARM_MORGAN_VERSION,
        "status": "selection_frozen_no_test_role_exists",
        "claim_scope": "development_routing_no_scientific_causal_or_sota_claim",
        "protocol": warm["protocol"],
        "source_protocol": warm["source_protocol"],
        "endpoint": warm["endpoint"],
        "endpoint_decision_status": warm["endpoint_decision_status"],
        "phase": "warm_validation_selection_only",
        "run_mode": "formal_5x3" if complete else "engineering_smoke",
        "selected_folds": list(selected_folds),
        "selected_seeds": list(selected_seeds),
        "max_train_points": max_train_points,
        "model": MODEL_SLUG,
        "parameter_count": parameter_count,
        "training_config": training.to_dict(),
        "warm_manifest": _file_record(
            warm_configuration.output_dir / "warm_pairout_manifest.json"
        ),
        "warm_manifest_hash": warm["manifest_sha256"],
        "base_benchmark_configuration": _file_record(Path(benchmark_config)),
        "cell_configuration": _file_record(Path(cell_config)),
        "context_manifest_hash": context.manifest["manifest_sha256"],
        "authorized_response_roles": ["warm_train", "warm_validation"],
        "test_response_rows_read": 0,
        "test_evaluations": 0,
        "phase_b_implementation": "absent_by_design",
        "jobs": jobs,
        "routing_decision": routing,
        "runtime_environment": _runtime_environment(),
    }
    manifest["configuration_hash"] = _canonical_hash(
        {
            "warm_manifest_hash": warm["manifest_sha256"],
            "benchmark_config": str(Path(benchmark_config).resolve()),
            "cell_config": str(Path(cell_config).resolve()),
            "training": training.to_dict(),
            "folds": list(selected_folds),
            "seeds": list(selected_seeds),
            "max_train_points": max_train_points,
        }
    )
    manifest["manifest_sha256"] = _canonical_hash(manifest)
    _write_json_atomic(run_path, manifest)
    return {**manifest, "selection_run_manifest": _file_record(run_path)}


__all__ = [
    "WARM_MORGAN_VERSION",
    "WarmMorganError",
    "routing_decision",
    "select_warm_morgan_probe",
]
