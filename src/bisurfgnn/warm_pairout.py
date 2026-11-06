"""Validation-only warm PairOut discriminator built from authorized shards.

The probe re-splits each One-new-scaffold development pool (train plus
validation) by complete unordered drug pair.  A validation pair is absent from
training, while both drugs, the cell, and both exact drug-cell contexts must be
present in training.  The physical test response shard is neither resolved nor
opened.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from .learned_benchmark import (
    _canonical_hash,
    _file_record,
    _runtime_environment,
    _write_json_atomic,
    _write_parquet_atomic,
    surface_macro_rmse,
)

WARM_PAIROUT_VERSION = "warm-pairout-development-v1"
READABLE_SOURCE_ROLES = ("train", "validation")
REFUSED_SOURCE_ROLES = ("test",)
REQUIRED_POINT_COLUMNS = (
    "source_combodrugseq",
    "prediction_group_id",
    "surface_replicate_id",
    "cell_id",
    "drug_a_id",
    "drug_b_id",
    "mono_response_a",
    "mono_response_b",
    "mono_mask_a",
    "mono_mask_b",
    "response",
    "response_clipped_high",
    "response_mask",
    "primary_metric_mask",
)


class WarmPairOutError(RuntimeError):
    """Raised when the warm development split violates its frozen contract."""


@dataclass(frozen=True)
class WarmPairOutConfig:
    config_path: Path
    source_shard_dir: Path
    output_dir: Path
    protocol: str = "WarmPairOutDevelopment-v1"
    source_protocol: str = "OneNewScaffoldAnchored-v1"
    endpoint: str = "clipped_PERCENTGROWTHNOTZ_over_100"
    endpoint_decision_status: str = "provisional_engineering"
    base_folds: tuple[int, ...] = (0, 1, 2, 3, 4)
    pair_bins: int = 5
    validation_bin: int = 0
    split_seed: int = 41
    require_drug_cell_seen: bool = True

    def __post_init__(self) -> None:
        for name in ("config_path", "source_shard_dir", "output_dir"):
            object.__setattr__(self, name, Path(getattr(self, name)).resolve())
        if self.protocol != "WarmPairOutDevelopment-v1":
            raise WarmPairOutError("Unsupported warm PairOut protocol")
        if self.source_protocol != "OneNewScaffoldAnchored-v1":
            raise WarmPairOutError("Warm probe source protocol is not frozen")
        if not self.require_drug_cell_seen:
            raise WarmPairOutError("The frozen warm protocol requires drug-cell overlap")
        if not self.base_folds or len(set(self.base_folds)) != len(self.base_folds):
            raise WarmPairOutError("base_folds must be a non-empty unique sequence")
        if any(fold < 0 or fold > 4 for fold in self.base_folds):
            raise WarmPairOutError("base_folds must be selected from 0 through 4")
        if self.pair_bins < 2 or not 0 <= self.validation_bin < self.pair_bins:
            raise WarmPairOutError("Invalid pair-bin configuration")
        if self.source_shard_dir == self.output_dir:
            raise WarmPairOutError("Warm output must not overwrite source shards")
        if self.source_shard_dir in self.output_dir.parents:
            raise WarmPairOutError("Warm output cannot be nested inside source shards")


def load_warm_pairout_config(path: str | Path) -> WarmPairOutConfig:
    """Load a path-resolved warm-probe configuration."""

    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise WarmPairOutError("Warm PairOut configuration must be a mapping")
    configured_root = Path(str(raw.get("project_root", "..")))
    project_root = (
        configured_root.resolve()
        if configured_root.is_absolute()
        else (config_path.parent / configured_root).resolve()
    )

    def resolve_path(key: str, default: str) -> Path:
        value = Path(str(raw.get(key, default)))
        return value.resolve() if value.is_absolute() else (project_root / value).resolve()

    folds = raw.get("base_folds", [0, 1, 2, 3, 4])
    if not isinstance(folds, list):
        raise WarmPairOutError("base_folds must be a list")
    return WarmPairOutConfig(
        config_path=config_path,
        source_shard_dir=resolve_path(
            "source_shard_dir",
            "data/processed/one_new_scaffold_benchmark/learned_point_shards",
        ),
        output_dir=resolve_path(
            "output_dir", "data/processed/warm_pairout_development"
        ),
        protocol=str(raw.get("protocol", "WarmPairOutDevelopment-v1")),
        source_protocol=str(
            raw.get("source_protocol", "OneNewScaffoldAnchored-v1")
        ),
        endpoint=str(raw.get("endpoint", "clipped_PERCENTGROWTHNOTZ_over_100")),
        endpoint_decision_status=str(
            raw.get("endpoint_decision_status", "provisional_engineering")
        ),
        base_folds=tuple(int(fold) for fold in folds),
        pair_bins=int(raw.get("pair_bins", 5)),
        validation_bin=int(raw.get("validation_bin", 0)),
        split_seed=int(raw.get("split_seed", 41)),
        require_drug_cell_seen=bool(raw.get("require_drug_cell_seen", True)),
    )


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise WarmPairOutError(f"Cannot read {label}: {path}") from error
    if not isinstance(payload, dict):
        raise WarmPairOutError(f"{label} must contain a JSON object")
    return payload


def _verify_self_hash(payload: Mapping[str, Any], label: str) -> None:
    recorded = str(payload.get("manifest_sha256", ""))
    unhashed = dict(payload)
    unhashed.pop("manifest_sha256", None)
    if recorded != _canonical_hash(unhashed):
        raise WarmPairOutError(f"{label} content hash changed")


def _verify_record(path: Path, record: Mapping[str, Any], label: str) -> None:
    if not path.is_file():
        raise WarmPairOutError(f"Missing {label}: {path}")
    actual = _file_record(path)
    if any(actual[key] != record.get(key) for key in ("bytes", "sha256")):
        raise WarmPairOutError(f"Frozen {label} bytes or hash changed")


def _source_role_path(source_dir: Path, fold: int, role: str) -> Path:
    if role in REFUSED_SOURCE_ROLES:
        raise WarmPairOutError(
            f"Role {role!r} is sealed; only {READABLE_SOURCE_ROLES} may be opened"
        )
    if role not in READABLE_SOURCE_ROLES:
        raise WarmPairOutError(f"Unknown source role {role!r}")
    return source_dir / f"fold_{fold}" / f"{role}.parquet"


def canonical_pair_key(drug_a: object, drug_b: object) -> str:
    """Return an order-invariant exact-drug-pair key."""

    left, right = str(drug_a).strip(), str(drug_b).strip()
    if not left or not right or left == right:
        raise WarmPairOutError("Warm PairOut requires two distinct non-empty drug IDs")
    low, high = sorted((left, right))
    return f"{low}|{high}"


def build_group_ledger(points: pd.DataFrame) -> pd.DataFrame:
    """Collapse point rows to repeat-atomic pair-cell prediction groups."""

    missing = sorted(set(REQUIRED_POINT_COLUMNS) - set(points.columns))
    if missing:
        raise WarmPairOutError(f"Point shards are missing required columns: {missing}")
    keys = [
        "prediction_group_id",
        "drug_a_id",
        "drug_b_id",
        "cell_id",
        "source_selection_role",
    ]
    identities = points[keys].drop_duplicates()
    duplicates = identities["prediction_group_id"].duplicated(keep=False)
    if duplicates.any():
        bad = sorted(identities.loc[duplicates, "prediction_group_id"].unique())[:10]
        raise WarmPairOutError(f"Prediction groups have inconsistent identities: {bad}")
    counts = points.groupby("prediction_group_id", sort=False).agg(
        surface_replicate_count=("surface_replicate_id", "nunique"),
        point_count=("source_combodrugseq", "size"),
    )
    ledger = identities.set_index("prediction_group_id").join(counts).reset_index()
    ledger["warm_pair_key"] = [
        canonical_pair_key(left, right)
        for left, right in zip(ledger["drug_a_id"], ledger["drug_b_id"], strict=True)
    ]
    return ledger.sort_values("prediction_group_id").reset_index(drop=True)


def assign_pair_bins(
    groups: pd.DataFrame,
    *,
    pair_bins: int,
    split_seed: int,
    base_fold: int,
    protocol: str = "WarmPairOutDevelopment-v1",
) -> dict[str, int]:
    """Greedily balance complete drug pairs over response-free bins."""

    required = {"warm_pair_key", "surface_replicate_count", "point_count"}
    if not required.issubset(groups.columns):
        raise WarmPairOutError("Group ledger cannot be binned")
    stats = (
        groups.groupby("warm_pair_key", sort=False)
        .agg(
            group_count=("prediction_group_id", "size"),
            surface_count=("surface_replicate_count", "sum"),
            point_count=("point_count", "sum"),
        )
        .reset_index()
    )
    if len(stats) < pair_bins:
        raise WarmPairOutError("Fewer exact drug pairs than requested bins")
    stats["tie_hash"] = stats["warm_pair_key"].map(
        lambda key: hashlib.sha256(
            f"{protocol}|{split_seed}|{base_fold}|{key}".encode("utf-8")
        ).hexdigest()
    )
    stats = stats.sort_values(
        ["surface_count", "group_count", "point_count", "tie_hash"],
        ascending=[False, False, False, True],
    )
    loads = [
        {"surface_count": 0, "group_count": 0, "point_count": 0}
        for _ in range(pair_bins)
    ]
    assignment: dict[str, int] = {}
    for row in stats.itertuples(index=False):
        selected = min(
            range(pair_bins),
            key=lambda index: (
                loads[index]["surface_count"],
                loads[index]["group_count"],
                loads[index]["point_count"],
                index,
            ),
        )
        assignment[str(row.warm_pair_key)] = selected
        loads[selected]["surface_count"] += int(row.surface_count)
        loads[selected]["group_count"] += int(row.group_count)
        loads[selected]["point_count"] += int(row.point_count)
    return assignment


def assign_warm_roles(
    groups: pd.DataFrame,
    pair_bins: Mapping[str, int],
    *,
    validation_bin: int,
    require_drug_cell_seen: bool = True,
) -> pd.DataFrame:
    """Assign train/validation while enforcing the warm-information contract."""

    assigned = groups.copy()
    assigned["warm_pair_bin"] = assigned["warm_pair_key"].map(pair_bins)
    if assigned["warm_pair_bin"].isna().any():
        raise WarmPairOutError("Some groups have no warm pair bin")
    assigned["warm_pair_bin"] = assigned["warm_pair_bin"].astype(int)
    train_mask = assigned["warm_pair_bin"] != validation_bin
    train = assigned[train_mask]
    train_drugs = set(train["drug_a_id"]) | set(train["drug_b_id"])
    train_cells = set(train["cell_id"])
    train_drug_cells = {
        (row.drug_a_id, row.cell_id) for row in train.itertuples(index=False)
    } | {(row.drug_b_id, row.cell_id) for row in train.itertuples(index=False)}

    roles: list[str] = []
    reasons: list[str] = []
    for row in assigned.itertuples(index=False):
        if row.warm_pair_bin != validation_bin:
            roles.append("train")
            reasons.append("")
            continue
        checks = {
            "drug_a_unseen": row.drug_a_id not in train_drugs,
            "drug_b_unseen": row.drug_b_id not in train_drugs,
            "cell_unseen": row.cell_id not in train_cells,
        }
        if require_drug_cell_seen:
            checks.update(
                {
                    "drug_a_cell_unseen": (
                        row.drug_a_id,
                        row.cell_id,
                    )
                    not in train_drug_cells,
                    "drug_b_cell_unseen": (
                        row.drug_b_id,
                        row.cell_id,
                    )
                    not in train_drug_cells,
                }
            )
        failed = [name for name, value in checks.items() if value]
        roles.append("excluded" if failed else "validation")
        reasons.append(";".join(failed))
    assigned["warm_role"] = roles
    assigned["warm_exclusion_reason"] = reasons
    return assigned


def audit_warm_roles(groups: pd.DataFrame) -> dict[str, Any]:
    """Return fail-closed PairOut and warm-context checks."""

    train = groups[groups["warm_role"] == "train"]
    validation = groups[groups["warm_role"] == "validation"]
    train_pairs = set(train["warm_pair_key"])
    validation_pairs = set(validation["warm_pair_key"])
    train_drugs = set(train["drug_a_id"]) | set(train["drug_b_id"])
    train_cells = set(train["cell_id"])
    train_drug_cells = {
        (row.drug_a_id, row.cell_id) for row in train.itertuples(index=False)
    } | {(row.drug_b_id, row.cell_id) for row in train.itertuples(index=False)}
    validation_drug_cells = {
        (row.drug_a_id, row.cell_id) for row in validation.itertuples(index=False)
    } | {
        (row.drug_b_id, row.cell_id) for row in validation.itertuples(index=False)
    }
    checks = {
        "train_nonempty": not train.empty,
        "validation_nonempty": not validation.empty,
        "exact_pairs_disjoint": train_pairs.isdisjoint(validation_pairs),
        "validation_drugs_seen": (
            set(validation["drug_a_id"]) | set(validation["drug_b_id"])
        ).issubset(train_drugs),
        "validation_cells_seen": set(validation["cell_id"]).issubset(train_cells),
        "validation_drug_cells_seen": validation_drug_cells.issubset(
            train_drug_cells
        ),
        "roles_valid": set(groups["warm_role"]).issubset(
            {"train", "validation", "excluded"}
        ),
    }
    if not all(checks.values()):
        failed = sorted(name for name, value in checks.items() if not value)
        raise WarmPairOutError(f"Warm PairOut audit failed: {failed}")
    return {
        "status": "passed",
        "checks": checks,
        "train_groups": int(len(train)),
        "validation_groups": int(len(validation)),
        "excluded_groups": int((groups["warm_role"] == "excluded").sum()),
        "train_pairs": int(len(train_pairs)),
        "validation_pairs": int(len(validation_pairs)),
    }


def _hsa_summary(frame: pd.DataFrame) -> dict[str, Any]:
    mask = (
        frame["response_mask"].to_numpy(dtype=bool)
        & frame["primary_metric_mask"].to_numpy(dtype=bool)
        & frame["mono_mask_a"].to_numpy(dtype=bool)
        & frame["mono_mask_b"].to_numpy(dtype=bool)
    )
    selected = frame.loc[mask]
    if selected.empty:
        raise WarmPairOutError("Warm validation has no eligible HSA metric point")
    target = selected["response"].to_numpy(dtype=np.float64)
    reference = np.minimum(
        selected["mono_response_a"].to_numpy(dtype=np.float64),
        selected["mono_response_b"].to_numpy(dtype=np.float64),
    )
    rmse = surface_macro_rmse(
        target,
        reference,
        selected["surface_replicate_id"].astype(str).to_numpy(object),
    )
    return {
        "metric": "surface_macro.rmse",
        "reference": "observed_hsa",
        "rmse": rmse,
        "points": int(len(selected)),
        "surfaces": int(selected["surface_replicate_id"].nunique()),
        "response_mean": float(np.mean(target)),
        "response_sd": float(np.std(target, ddof=1)),
        "clipped_high_fraction": float(
            selected["response_clipped_high"].to_numpy(dtype=bool).mean()
        ),
    }


def _source_manifest(config: WarmPairOutConfig) -> dict[str, Any]:
    path = config.source_shard_dir / "point_shard_manifest.json"
    manifest = _load_json_object(path, "source point-shard manifest")
    _verify_self_hash(manifest, "Source point-shard manifest")
    if manifest.get("protocol") != config.source_protocol:
        raise WarmPairOutError("Source point-shard protocol changed")
    if manifest.get("endpoint") != config.endpoint:
        raise WarmPairOutError("Source endpoint changed")
    if manifest.get("endpoint_decision_status") != config.endpoint_decision_status:
        raise WarmPairOutError("Source endpoint decision status changed")
    return manifest


def _fold_source_record(manifest: Mapping[str, Any], fold: int) -> Mapping[str, Any]:
    folds = manifest.get("folds")
    if not isinstance(folds, list):
        raise WarmPairOutError("Source point-shard manifest has no fold inventory")
    for entry in folds:
        if isinstance(entry, dict) and entry.get("fold") == fold:
            return entry
    raise WarmPairOutError(f"Source point-shard manifest has no fold {fold}")


def _prepare_fold(
    config: WarmPairOutConfig,
    source_manifest: Mapping[str, Any],
    fold: int,
) -> dict[str, Any]:
    source_fold = _fold_source_record(source_manifest, fold)
    role_records = source_fold.get("roles")
    if not isinstance(role_records, dict):
        raise WarmPairOutError(f"Source fold {fold} has no role records")
    frames = []
    source_records: dict[str, Any] = {}
    for role in READABLE_SOURCE_ROLES:
        path = _source_role_path(config.source_shard_dir, fold, role)
        record = role_records.get(role)
        if not isinstance(record, dict):
            raise WarmPairOutError(f"Source fold {fold} has no {role} record")
        _verify_record(path, record, f"fold {fold} {role} shard")
        frame = pd.read_parquet(path)
        frame["source_selection_role"] = role
        frames.append(frame)
        source_records[role] = _file_record(path)
    combined = pd.concat(frames, ignore_index=True)
    source_ids = combined["source_combodrugseq"]
    if source_ids.duplicated().any():
        raise WarmPairOutError(f"Fold {fold} source train/validation rows overlap")

    groups = build_group_ledger(combined)
    bins = assign_pair_bins(
        groups,
        pair_bins=config.pair_bins,
        split_seed=config.split_seed,
        base_fold=fold,
        protocol=config.protocol,
    )
    assigned = assign_warm_roles(
        groups,
        bins,
        validation_bin=config.validation_bin,
        require_drug_cell_seen=config.require_drug_cell_seen,
    )
    audit = audit_warm_roles(assigned)
    by_group = assigned.set_index("prediction_group_id")
    combined["warm_pair_key"] = combined["prediction_group_id"].map(
        by_group["warm_pair_key"]
    )
    combined["warm_pair_bin"] = combined["prediction_group_id"].map(
        by_group["warm_pair_bin"]
    )
    combined["warm_role"] = combined["prediction_group_id"].map(
        by_group["warm_role"]
    )
    combined["warm_exclusion_reason"] = combined["prediction_group_id"].map(
        by_group["warm_exclusion_reason"]
    )
    if combined["warm_role"].isna().any():
        raise WarmPairOutError(f"Fold {fold} point rows lost their group assignment")

    output_fold = config.output_dir / f"fold_{fold}"
    group_path = output_fold / "group_assignments.parquet"
    train_path = output_fold / "train.parquet"
    validation_path = output_fold / "validation.parquet"
    _write_parquet_atomic(group_path, assigned)
    train_frame = combined[combined["warm_role"] == "train"].reset_index(drop=True)
    validation_frame = combined[
        combined["warm_role"] == "validation"
    ].reset_index(drop=True)
    _write_parquet_atomic(train_path, train_frame)
    _write_parquet_atomic(validation_path, validation_frame)
    if set(train_frame["warm_pair_key"]) & set(validation_frame["warm_pair_key"]):
        raise WarmPairOutError(f"Fold {fold} physical shards overlap by exact pair")
    return {
        "fold": fold,
        "source_roles_read": list(READABLE_SOURCE_ROLES),
        "source_records": source_records,
        "source_points": int(len(combined)),
        "source_groups": int(len(groups)),
        "audit": audit,
        "hsa_validation": _hsa_summary(validation_frame),
        "outputs": {
            "group_assignments": {
                **_file_record(group_path),
                "rows": int(len(assigned)),
            },
            "train": {
                **_file_record(train_path),
                "rows": int(len(train_frame)),
                "groups": int(train_frame["prediction_group_id"].nunique()),
                "surfaces": int(train_frame["surface_replicate_id"].nunique()),
            },
            "validation": {
                **_file_record(validation_path),
                "rows": int(len(validation_frame)),
                "groups": int(validation_frame["prediction_group_id"].nunique()),
                "surfaces": int(
                    validation_frame["surface_replicate_id"].nunique()
                ),
            },
        },
    }


def prepare_warm_pairout_probe(
    config_path: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Freeze the validation-only warm PairOut shards and audit manifest."""

    config = load_warm_pairout_config(config_path)
    manifest_path = config.output_dir / "warm_pairout_manifest.json"
    if manifest_path.exists() and not overwrite:
        raise WarmPairOutError(
            f"Warm PairOut manifest already exists: {manifest_path}; use --overwrite"
        )
    source = _source_manifest(config)
    fold_results = [
        _prepare_fold(config, source, fold) for fold in config.base_folds
    ]
    hsa_values = [float(result["hsa_validation"]["rmse"]) for result in fold_results]
    manifest: dict[str, Any] = {
        "version": WARM_PAIROUT_VERSION,
        "protocol": config.protocol,
        "source_protocol": config.source_protocol,
        "scope": "development_falsifier_train_validation_only",
        "claim_scope": "split_and_reference_engineering_no_model_performance_claim",
        "endpoint": config.endpoint,
        "endpoint_decision_status": config.endpoint_decision_status,
        "assignment_unit": "complete_unordered_exact_drug_pair",
        "validation_information_contract": {
            "exact_pair_absent_from_train": True,
            "both_exact_drugs_seen_in_train": True,
            "cell_seen_in_train": True,
            "both_exact_drug_cell_contexts_seen_in_train": (
                config.require_drug_cell_seen
            ),
            "all_prediction_group_replicates_atomic": True,
        },
        "base_folds": list(config.base_folds),
        "pair_bins": config.pair_bins,
        "validation_bin": config.validation_bin,
        "split_seed": config.split_seed,
        "source_roles_read": list(READABLE_SOURCE_ROLES),
        "source_roles_refused": list(REFUSED_SOURCE_ROLES),
        "test_response_rows_read": 0,
        "test_shard_opened": False,
        "phase_b_action": "none",
        "source_point_shard_manifest": _file_record(
            config.source_shard_dir / "point_shard_manifest.json"
        ),
        "configuration": _file_record(config.config_path),
        "source_code": _file_record(Path(__file__)),
        "folds": fold_results,
        "fold_mean_hsa_validation_surface_macro_rmse": float(np.mean(hsa_values)),
        "fold_sd_hsa_validation_surface_macro_rmse": (
            float(np.std(hsa_values, ddof=1))
            if len(hsa_values) > 1
            else None
        ),
        "runtime_environment": _runtime_environment(),
        "status": "passed_validation_only_no_test_shard_opened",
    }
    manifest["manifest_sha256"] = _canonical_hash(manifest)
    _write_json_atomic(manifest_path, manifest)
    return manifest


def load_warm_pairout_probe(config_path: str | Path) -> dict[str, Any]:
    """Verify and reload every frozen warm-probe artifact."""

    config = load_warm_pairout_config(config_path)
    manifest_path = config.output_dir / "warm_pairout_manifest.json"
    manifest = _load_json_object(manifest_path, "warm PairOut manifest")
    _verify_self_hash(manifest, "Warm PairOut manifest")
    expected = {
        "version": WARM_PAIROUT_VERSION,
        "protocol": config.protocol,
        "source_protocol": config.source_protocol,
        "endpoint": config.endpoint,
        "endpoint_decision_status": config.endpoint_decision_status,
        "base_folds": list(config.base_folds),
        "pair_bins": config.pair_bins,
        "validation_bin": config.validation_bin,
        "split_seed": config.split_seed,
        "test_response_rows_read": 0,
        "test_shard_opened": False,
        "phase_b_action": "none",
        "status": "passed_validation_only_no_test_shard_opened",
    }
    mismatched = sorted(
        key for key, value in expected.items() if manifest.get(key) != value
    )
    if mismatched:
        raise WarmPairOutError(f"Warm PairOut manifest context changed: {mismatched}")
    for key, path in (
        ("configuration", config.config_path),
        (
            "source_point_shard_manifest",
            config.source_shard_dir / "point_shard_manifest.json",
        ),
        ("source_code", Path(__file__)),
    ):
        record = manifest.get(key)
        if not isinstance(record, dict):
            raise WarmPairOutError(f"Warm PairOut manifest has no {key} record")
        _verify_record(path, record, key)

    entries = manifest.get("folds")
    if not isinstance(entries, list):
        raise WarmPairOutError("Warm PairOut manifest has no fold inventory")
    by_fold = {
        int(entry["fold"]): entry
        for entry in entries
        if isinstance(entry, dict) and "fold" in entry
    }
    if set(by_fold) != set(config.base_folds):
        raise WarmPairOutError("Warm PairOut fold inventory changed")
    for fold in config.base_folds:
        entry = by_fold[fold]
        source_records = entry.get("source_records")
        outputs = entry.get("outputs")
        if not isinstance(source_records, dict) or not isinstance(outputs, dict):
            raise WarmPairOutError(f"Fold {fold} artifact inventory is invalid")
        for role in READABLE_SOURCE_ROLES:
            record = source_records.get(role)
            if not isinstance(record, dict):
                raise WarmPairOutError(f"Fold {fold} has no source {role} record")
            _verify_record(
                _source_role_path(config.source_shard_dir, fold, role),
                record,
                f"fold {fold} source {role}",
            )
        paths = {
            "group_assignments": config.output_dir
            / f"fold_{fold}"
            / "group_assignments.parquet",
            "train": config.output_dir / f"fold_{fold}" / "train.parquet",
            "validation": config.output_dir
            / f"fold_{fold}"
            / "validation.parquet",
        }
        for name, path in paths.items():
            record = outputs.get(name)
            if not isinstance(record, dict):
                raise WarmPairOutError(f"Fold {fold} has no {name} record")
            _verify_record(path, record, f"fold {fold} {name}")
        forbidden = config.output_dir / f"fold_{fold}" / "test.parquet"
        if forbidden.exists():
            raise WarmPairOutError(f"Warm output must not contain a test shard: {forbidden}")

        groups = pd.read_parquet(paths["group_assignments"])
        train = pd.read_parquet(paths["train"])
        validation = pd.read_parquet(paths["validation"])
        if len(groups) != int(outputs["group_assignments"].get("rows", -1)):
            raise WarmPairOutError(f"Fold {fold} group row count changed")
        for role, frame in (("train", train), ("validation", validation)):
            if len(frame) != int(outputs[role].get("rows", -1)):
                raise WarmPairOutError(f"Fold {fold} {role} row count changed")
            expected_groups = set(
                groups.loc[groups["warm_role"] == role, "prediction_group_id"]
            )
            if set(frame["prediction_group_id"]) != expected_groups:
                raise WarmPairOutError(f"Fold {fold} {role} group inventory changed")
        audit = audit_warm_roles(groups)
        if audit != entry.get("audit"):
            raise WarmPairOutError(f"Fold {fold} warm audit changed")
        hsa = _hsa_summary(validation)
        recorded_hsa = entry.get("hsa_validation")
        if not isinstance(recorded_hsa, dict):
            raise WarmPairOutError(f"Fold {fold} has no HSA summary")
        if any(
            not math.isclose(
                float(hsa[key]),
                float(recorded_hsa.get(key, float("nan"))),
                rel_tol=0,
                abs_tol=1e-12,
            )
            for key in ("rmse", "response_mean", "response_sd", "clipped_high_fraction")
        ):
            raise WarmPairOutError(f"Fold {fold} HSA summary changed")
    return manifest


__all__ = [
    "READABLE_SOURCE_ROLES",
    "REFUSED_SOURCE_ROLES",
    "WARM_PAIROUT_VERSION",
    "WarmPairOutConfig",
    "WarmPairOutError",
    "assign_pair_bins",
    "assign_warm_roles",
    "audit_warm_roles",
    "build_group_ledger",
    "canonical_pair_key",
    "load_warm_pairout_config",
    "load_warm_pairout_probe",
    "prepare_warm_pairout_probe",
]
