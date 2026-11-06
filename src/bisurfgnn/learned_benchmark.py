"""Learned Morgan/Hallmark benchmark for the frozen scaffold protocol.

The runner in this module is deliberately narrow.  It trains a symmetric
residual MLP around a predeclared observed-monotherapy reference, selects
checkpoints only by validation surface-macro RMSE, and evaluates the outer test
role only after every selected checkpoint has been frozen.  All emitted results
are engineering evidence on a provisional endpoint, not scientific or SOTA
claims.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import platform
import random
import shutil
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator
from torch import Tensor

from .benchmark import summarize_reference_ledger
from .benchmark_config import BenchmarkConfig, load_benchmark_config
from .data.benchmark_context import PreparedBenchmarkCellContext, load_benchmark_cell_context
from .data.cellminer import CellMinerConfig, load_cellminer_config
from .data.chemical_splits import PreparedChemicalSplit, load_one_new_scaffold_benchmark
from .models.baselines import MorganHallmarkResidualMLP


LEARNED_BENCHMARK_VERSION = "morgan-hallmark-residual-v3"
LEARNED_POINT_SHARD_VERSION = "morgan-hallmark-point-shards-v3"
MODEL_SLUG = "morgan_hallmark_residual_mlp"
POINT_SHARD_DIRECTORY = "learned_point_shards"
POINT_SHARD_MANIFEST = "point_shard_manifest.json"
SELECTION_RUN_MANIFEST = "selection_run_manifest.json"
EVALUATION_RUN_MANIFEST = "run_manifest.json"
TEST_ACCESS_SEAL = "test_access_seal.json"
SOURCE_PARTITION_LEDGER = "source_partition.parquet"
HALLMARK_NODE_COUNT = 50
HALLMARK_DIMENSION = 100


class LearnedBenchmarkError(ValueError):
    """Raised when learned-benchmark inputs violate the frozen contract."""


@dataclass(frozen=True)
class FrozenLearnedPointShards:
    """Hash-verified physical role partitions for model selection and testing."""

    root: Path
    manifest: dict[str, Any]

    def role_record(self, fold: int, role: str) -> dict[str, Any]:
        if role not in {"train", "validation", "test"}:
            raise LearnedBenchmarkError(f"Invalid learned point-shard role: {role}")
        records = self.manifest.get("folds")
        if not isinstance(records, list):
            raise LearnedBenchmarkError("Point-shard fold inventory is invalid")
        for record in records:
            if isinstance(record, dict) and record.get("fold") == fold:
                roles = record.get("roles")
                if isinstance(roles, dict) and isinstance(roles.get(role), dict):
                    return dict(roles[role])
        raise LearnedBenchmarkError(f"Point-shard record is missing fold {fold} {role}")

    def role_path(self, fold: int, role: str) -> Path:
        record = self.role_record(fold, role)
        relative = Path(str(record.get("relative_path", "")))
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise LearnedBenchmarkError("Point-shard path is unsafe")
        path = (self.root / relative).resolve()
        try:
            path.relative_to(self.root.resolve())
        except ValueError as error:
            raise LearnedBenchmarkError("Point-shard path escapes its root") from error
        return path

    def source_partition_record(self, fold: int) -> dict[str, Any]:
        records = self.manifest.get("folds")
        if not isinstance(records, list):
            raise LearnedBenchmarkError("Point-shard fold inventory is invalid")
        for record in records:
            if isinstance(record, dict) and record.get("fold") == fold:
                partition = record.get("source_partition")
                if isinstance(partition, dict):
                    return dict(partition)
        raise LearnedBenchmarkError(
            f"Point-shard source partition is missing for fold {fold}"
        )

    def source_partition_path(self, fold: int) -> Path:
        record = self.source_partition_record(fold)
        relative = Path(str(record.get("relative_path", "")))
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise LearnedBenchmarkError("Source-partition path is unsafe")
        path = (self.root / relative).resolve()
        try:
            path.relative_to(self.root.resolve())
        except ValueError as error:
            raise LearnedBenchmarkError(
                "Source-partition path escapes its root"
            ) from error
        return path


@dataclass(frozen=True)
class LearnedMorganConfig:
    max_epochs: int = 200
    patience: int = 20
    min_epochs: int = 10
    min_delta: float = 1e-5
    batch_surfaces: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    hidden_dim: int = 256
    dropout: float = 0.1
    seeds: tuple[int, ...] = (41, 42, 43)
    reference_mode: str = "observed_hsa"
    include_anchor_candidate: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.include_anchor_candidate, bool):
            raise LearnedBenchmarkError("include_anchor_candidate must be boolean")
        if self.max_epochs < 1 or self.patience < 1 or self.batch_surfaces < 1:
            raise LearnedBenchmarkError(
                "max_epochs, patience, and batch_surfaces must be positive"
            )
        if self.min_epochs < 1 or self.min_epochs > self.max_epochs:
            raise LearnedBenchmarkError("min_epochs must lie in [1,max_epochs]")
        if self.min_delta < 0:
            raise LearnedBenchmarkError("min_delta must be non-negative")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise LearnedBenchmarkError(
                "learning rate must be positive and weight decay non-negative"
            )
        if self.hidden_dim < 1:
            raise LearnedBenchmarkError("hidden dimension must be positive")
        if self.gradient_clip <= 0:
            raise LearnedBenchmarkError("gradient_clip must be positive")
        if not 0 <= self.dropout < 1:
            raise LearnedBenchmarkError("dropout must be in [0,1)")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise LearnedBenchmarkError("seeds must be a non-empty unique sequence")
        if self.reference_mode not in {"observed_hsa", "observed_product"}:
            raise LearnedBenchmarkError(
                "reference_mode must be observed_hsa or observed_product"
            )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["seeds"] = list(self.seeds)
        return payload


@dataclass(frozen=True)
class CompactFeatureBanks:
    """Unique molecular and cell features; points retain only integer indices."""

    drug_ids: tuple[str, ...]
    cell_ids: tuple[str, ...]
    fingerprints: Tensor
    hallmarks: Tensor
    fingerprint_hashes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.fingerprints.ndim != 2 or self.fingerprints.shape[0] != len(self.drug_ids):
            raise LearnedBenchmarkError("Fingerprint bank shape does not match drug IDs")
        if self.hallmarks.shape != (len(self.cell_ids), HALLMARK_DIMENSION):
            raise LearnedBenchmarkError("Hallmark bank must have shape [cells,100]")
        if self.fingerprints.dtype != torch.uint8:
            raise LearnedBenchmarkError("Fingerprint bank must use compact uint8 storage")
        if self.hallmarks.dtype != torch.float32:
            raise LearnedBenchmarkError("Hallmark bank must use float32 storage")
        if len(self.fingerprint_hashes) != len(self.drug_ids):
            raise LearnedBenchmarkError("Fingerprint hash inventory is incomplete")
        if not bool(torch.isfinite(self.hallmarks).all()):
            raise LearnedBenchmarkError("Hallmark bank contains non-finite values")

    @property
    def fingerprint_dim(self) -> int:
        return int(self.fingerprints.shape[1])


@dataclass(frozen=True)
class CompactPointTable:
    """Point-level scalars and bank indices without replicated feature vectors."""

    role: str
    fold: int
    source_rows: np.ndarray
    group_ids: np.ndarray
    surface_ids: np.ndarray
    cluster_ids: np.ndarray
    drug_a_index: Tensor
    drug_b_index: Tensor
    cell_index: Tensor
    doses: Tensor
    mono_a: Tensor
    mono_b: Tensor
    target: Tensor
    response_clipped_high: np.ndarray
    # 1.0 for the clipped endpoint; the D1 unclipped endpoint declares a wider
    # bound through BenchmarkConfig.response_upper_bound.
    monotherapy_upper_bound: float = 1.0

    def __post_init__(self) -> None:
        size = len(self.source_rows)
        if self.role not in {"train", "validation", "test"}:
            raise LearnedBenchmarkError(f"Invalid compact-table role: {self.role}")
        if any(
            len(values) != size
            for values in (
                self.group_ids,
                self.surface_ids,
                self.cluster_ids,
                self.response_clipped_high,
            )
        ):
            raise LearnedBenchmarkError("Compact metadata arrays have inconsistent lengths")
        if any(
            tensor.shape != (size,)
            for tensor in (
                self.drug_a_index,
                self.drug_b_index,
                self.cell_index,
                self.mono_a,
                self.mono_b,
                self.target,
            )
        ):
            raise LearnedBenchmarkError("Compact point tensors have inconsistent lengths")
        if self.doses.shape != (size, 2):
            raise LearnedBenchmarkError("Compact doses must have shape [points,2]")
        if size == 0:
            raise LearnedBenchmarkError(f"Fold {self.fold} {self.role} table is empty")
        if self.drug_a_index.dtype != torch.int32 or self.drug_b_index.dtype != torch.int32:
            raise LearnedBenchmarkError("Compact drug indices must use int32 storage")
        if self.cell_index.dtype != torch.int32:
            raise LearnedBenchmarkError("Compact cell indices must use int32 storage")
        numeric = (self.doses, self.mono_a, self.mono_b, self.target)
        if not all(tensor.dtype == torch.float32 for tensor in numeric):
            raise LearnedBenchmarkError("Compact numeric tensors must use float32 storage")
        if not all(bool(torch.isfinite(tensor).all()) for tensor in numeric):
            raise LearnedBenchmarkError("Compact point table contains non-finite values")
        if bool((self.doses < 0).any()):
            raise LearnedBenchmarkError("Model doses must be non-negative")
        bound = float(self.monotherapy_upper_bound)
        if bound < 1.0:
            raise LearnedBenchmarkError("monotherapy_upper_bound must be at least 1.0")
        if bool(((self.mono_a < 0) | (self.mono_a > bound)).any()) or bool(
            ((self.mono_b < 0) | (self.mono_b > bound)).any()
        ):
            raise LearnedBenchmarkError(
                f"Observed monotherapy values must lie in [0,{bound}]"
            )
        if self.response_clipped_high.dtype != np.bool_:
            raise LearnedBenchmarkError("response_clipped_high must be a strict boolean array")

    def __len__(self) -> int:
        return len(self.source_rows)


@dataclass(frozen=True)
class CompactFoldData:
    fold: int
    banks: CompactFeatureBanks
    train: CompactPointTable
    validation: CompactPointTable
    test: CompactPointTable

    def __post_init__(self) -> None:
        role_sets = {
            role: set(table.source_rows.tolist())
            for role, table in (
                ("train", self.train),
                ("validation", self.validation),
                ("test", self.test),
            )
        }
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            if not role_sets[left].isdisjoint(role_sets[right]):
                raise LearnedBenchmarkError(f"Fold {self.fold} point roles overlap: {left}/{right}")


def load_learned_morgan_config(path: str | Path) -> LearnedMorganConfig:
    """Parse the required ``learned_morgan`` YAML mapping."""

    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise LearnedBenchmarkError("Benchmark configuration must be a mapping")
    section = raw.get("learned_morgan")
    if not isinstance(section, dict):
        raise LearnedBenchmarkError("Benchmark YAML must define a learned_morgan mapping")

    def value(primary: str, alias: str | None, default: Any) -> Any:
        if primary in section:
            return section[primary]
        return section.get(alias, default) if alias else default

    seeds = value("seeds", None, (41, 42, 43))
    if not isinstance(seeds, (list, tuple)) or isinstance(seeds, (str, bytes)):
        raise LearnedBenchmarkError("learned_morgan.seeds must be a sequence")
    return LearnedMorganConfig(
        max_epochs=int(value("max_epochs", "epochs", 200)),
        patience=int(value("patience", None, 20)),
        min_epochs=int(value("min_epochs", None, 10)),
        min_delta=float(value("min_delta", None, 1e-5)),
        batch_surfaces=int(value("batch_surfaces", None, 128)),
        learning_rate=float(value("learning_rate", "lr", 3e-4)),
        weight_decay=float(value("weight_decay", None, 1e-4)),
        gradient_clip=float(value("gradient_clip", None, 1.0)),
        hidden_dim=int(value("hidden_dim", "hidden", 256)),
        dropout=float(value("dropout", None, 0.1)),
        seeds=tuple(int(seed) for seed in seeds),
        reference_mode=str(value("reference_mode", None, "observed_hsa")),
        include_anchor_candidate=value("include_anchor_candidate", None, False),
    )


def _clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _require_columns(frame: pd.DataFrame, required: Sequence[str], label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise LearnedBenchmarkError(f"{label} is missing required columns: {missing}")


def _strict_boolean(values: pd.Series, label: str) -> np.ndarray:
    if values.isna().any():
        raise LearnedBenchmarkError(f"{label} contains missing values")
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.to_numpy(dtype=bool)
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or not numeric.isin((0, 1)).all():
        raise LearnedBenchmarkError(f"{label} must contain only boolean or 0/1 values")
    return numeric.to_numpy(dtype=bool)


def _sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": int(resolved.stat().st_size),
        "sha256": _sha256_file(resolved),
    }


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _runtime_environment() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": str(torch.__version__),
        "cuda_runtime": str(torch.version.cuda) if torch.version.cuda else None,
        "cuda_available": cuda_available,
        "cuda_devices": (
            [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ]
            if cuda_available
            else []
        ),
        "rdkit": rdBase.rdkitVersion,
        "numeric_precision": "float32",
    }


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
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(_json_ready(payload), handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
        os.replace(name, path)
    except Exception:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass
        raise


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    """Create a fail-closed JSON seal that can never overwrite an existing file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            _json_ready(payload),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as error:
        raise LearnedBenchmarkError(
            f"Outer-test access is already sealed and cannot be repeated: {path}"
        ) from error
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        # Deliberately retain even a partial seal.  Once exclusive creation has
        # succeeded, the cohort is burned rather than silently becoming reusable.
        raise


def _selection_code_paths() -> dict[str, Path]:
    package = Path(__file__).resolve().parent
    return {
        "learned_benchmark": Path(__file__).resolve(),
        "benchmark_metrics": package / "benchmark.py",
        "baselines": package / "models" / "baselines.py",
        "surface_head": package / "models" / "surface_head.py",
    }


def _write_parquet_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        frame.to_parquet(name, index=False)
        os.replace(name, path)
    except Exception:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass
        raise


def _write_checkpoint_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(dict(payload), name)
        os.replace(name, path)
    except Exception:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass
        raise


def _feature_bank_hash(banks: CompactFeatureBanks) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "drug_ids": list(banks.drug_ids),
                "cell_ids": list(banks.cell_ids),
                "fingerprint_hashes": list(banks.fingerprint_hashes),
                "fingerprint_shape": list(banks.fingerprints.shape),
                "hallmark_shape": list(banks.hallmarks.shape),
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(banks.fingerprints.contiguous().numpy().tobytes())
    digest.update(banks.hallmarks.contiguous().numpy().tobytes())
    return digest.hexdigest()


def _point_table_hash(table: CompactPointTable) -> str:
    digest = hashlib.sha256()
    digest.update(table.role.encode("utf-8"))
    digest.update(str(table.fold).encode("ascii"))
    for values in (table.source_rows, table.group_ids, table.surface_ids, table.cluster_ids):
        for value in values:
            digest.update(str(value).encode("utf-8"))
            digest.update(b"\0")
    for tensor in (
        table.drug_a_index,
        table.drug_b_index,
        table.cell_index,
        table.doses,
        table.mono_a,
        table.mono_b,
        table.target,
    ):
        digest.update(tensor.contiguous().numpy().tobytes())
    digest.update(table.response_clipped_high.tobytes())
    return digest.hexdigest()


def _freeze_selection_checkpoint(
    job_dir: Path,
    *,
    fold: int,
    seed: int,
    selection: Mapping[str, Any],
    state: Mapping[str, Tensor],
    training: LearnedMorganConfig,
    split_hash: str,
    context_split_hash: str,
    context_state_hash: str,
    feature_bank_hash: str,
    train_shard: Mapping[str, Any],
    validation_shard: Mapping[str, Any],
    source_partition: Mapping[str, Any],
) -> tuple[Path, Path, dict[str, Any]]:
    """Persist validation selection before any outer-test response is read."""

    checkpoint_path = job_dir / "checkpoint.pt"
    selection_path = job_dir / "selection_manifest.json"
    semantic_binding = {
        "benchmark_version": LEARNED_BENCHMARK_VERSION,
        "model": MODEL_SLUG,
        "fold": fold,
        "seed": seed,
        "best_epoch": selection["best_epoch"],
        "split_hash": split_hash,
        "context_split_hash": context_split_hash,
        "context_state_hash": context_state_hash,
        "feature_bank_hash": feature_bank_hash,
        "training_config": training.to_dict(),
        "runtime_environment": _runtime_environment(),
        "train_shard": dict(train_shard),
        "validation_shard": dict(validation_shard),
        "source_partition": dict(source_partition),
        "selection_hash": _canonical_hash(selection),
    }
    _write_checkpoint_atomic(
        checkpoint_path,
        {
            **semantic_binding,
            "state_dict": dict(state),
        },
    )
    checkpoint_record = _file_record(checkpoint_path)
    manifest: dict[str, Any] = {
        "status": "selection_frozen_before_test_response_read",
        **semantic_binding,
        "selection": dict(selection),
        "checkpoint": checkpoint_record,
        "code_sources": {
            name: _file_record(path)
            for name, path in _selection_code_paths().items()
        },
    }
    manifest["manifest_sha256"] = _canonical_hash(manifest)
    _write_json_atomic(selection_path, manifest)
    return checkpoint_path, selection_path, manifest


def _load_frozen_selection_checkpoint(
    checkpoint_path: Path,
    selection_path: Path,
    *,
    expected: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    """Verify both hashes before returning the state accepted by test evaluation."""

    manifest = json.loads(selection_path.read_text(encoding="utf-8"))
    recorded_hash = manifest.pop("manifest_sha256", None)
    if recorded_hash != _canonical_hash(manifest):
        raise LearnedBenchmarkError("Selection manifest content hash changed")
    checkpoint_record = manifest.get("checkpoint")
    if not isinstance(checkpoint_record, dict):
        raise LearnedBenchmarkError("Selection manifest has no checkpoint record")
    actual_record = _file_record(checkpoint_path)
    if any(
        actual_record.get(key) != checkpoint_record.get(key)
        for key in ("path", "bytes", "sha256")
    ):
        raise LearnedBenchmarkError("Frozen validation-selected checkpoint hash changed")
    code_sources = manifest.get("code_sources")
    if not isinstance(code_sources, dict):
        raise LearnedBenchmarkError("Selection manifest has no code-source binding")
    for name, path in _selection_code_paths().items():
        record = code_sources.get(name)
        if not isinstance(record, dict):
            raise LearnedBenchmarkError(f"Selection manifest lacks {name} source binding")
        _verify_recorded_file(path, record, f"Selection {name} source")
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise LearnedBenchmarkError("Frozen checkpoint payload is invalid")
    semantic_keys = (
        "benchmark_version",
        "model",
        "fold",
        "seed",
        "best_epoch",
        "split_hash",
        "context_split_hash",
        "context_state_hash",
        "feature_bank_hash",
        "training_config",
        "runtime_environment",
        "train_shard",
        "validation_shard",
        "source_partition",
        "selection_hash",
    )
    mismatched_payload = [
        key for key in semantic_keys if payload.get(key) != manifest.get(key)
    ]
    if mismatched_payload:
        raise LearnedBenchmarkError(
            f"Checkpoint semantic binding changed: {mismatched_payload}"
        )
    if manifest.get("runtime_environment") != _runtime_environment():
        raise LearnedBenchmarkError(
            "Frozen selection runtime environment differs from current evaluation"
        )
    selection = manifest.get("selection")
    if not isinstance(selection, dict) or manifest.get("selection_hash") != _canonical_hash(
        selection
    ):
        raise LearnedBenchmarkError("Selection result hash changed")
    if expected is not None:
        mismatched_expected = [
            key for key, value in expected.items() if manifest.get(key) != value
        ]
        if mismatched_expected:
            raise LearnedBenchmarkError(
                f"Frozen selection differs from current run: {mismatched_expected}"
            )
    manifest["manifest_sha256"] = recorded_hash
    return manifest, payload["state_dict"]


def build_morgan_bank(
    drugs: pd.DataFrame,
    required_drug_ids: Sequence[str],
    *,
    radius: int = 2,
    bits: int = 2_048,
) -> tuple[tuple[str, ...], Tensor, tuple[str, ...]]:
    """Build one whole-molecule Morgan vector per unique drug."""

    if radius != 2 or bits != 2_048:
        raise LearnedBenchmarkError("The frozen benchmark requires Morgan radius=2/2048")
    _require_columns(drugs, ("drug_id", "canonical_smiles"), "drug table")
    normalized = drugs.loc[:, ["drug_id", "canonical_smiles"]].copy()
    normalized["drug_id"] = normalized["drug_id"].map(_clean_text)
    normalized["canonical_smiles"] = normalized["canonical_smiles"].map(_clean_text)
    if normalized["drug_id"].duplicated().any():
        raise LearnedBenchmarkError("Drug table contains duplicate drug IDs")
    by_id = normalized.set_index("drug_id")["canonical_smiles"].to_dict()
    drug_ids = tuple(sorted({_clean_text(value) for value in required_drug_ids}))
    missing = sorted(set(drug_ids) - set(by_id))
    if missing:
        raise LearnedBenchmarkError(f"Required drugs are missing from drugs.csv: {missing[:10]}")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
    vectors: list[np.ndarray] = []
    hashes: list[str] = []
    for drug_id in drug_ids:
        molecule = Chem.MolFromSmiles(by_id[drug_id])
        if molecule is None:
            raise LearnedBenchmarkError(f"Invalid canonical SMILES for {drug_id}")
        fingerprint = generator.GetFingerprint(molecule)
        vector = np.zeros(bits, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fingerprint, vector)
        vectors.append(vector)
        hashes.append(hashlib.sha256(DataStructs.BitVectToBinaryText(fingerprint)).hexdigest())
    return drug_ids, torch.from_numpy(np.stack(vectors)), tuple(hashes)


def build_hallmark_bank(
    context: PreparedBenchmarkCellContext,
    fold: int,
    required_source_cell_ids: Sequence[str],
) -> tuple[tuple[str, ...], Tensor]:
    """Flatten fixed-order activity-z and observed-fraction channels to 100 values."""

    if fold < 0 or fold >= len(context.folds):
        raise LearnedBenchmarkError(f"Context has no fold {fold}")
    mapping = context.mapping
    _require_columns(
        mapping,
        ("source_cell_id", "biological_cell_id", "status"),
        "cell mapping",
    )
    mapped = mapping.loc[mapping["status"].astype(str).eq("mapped")].copy()
    lookup = {
        _clean_text(row.source_cell_id): _clean_text(row.biological_cell_id)
        for row in mapped.itertuples(index=False)
    }
    source_ids = tuple(sorted({_clean_text(value) for value in required_source_cell_ids}))
    missing = sorted(set(source_ids) - set(lookup))
    if missing:
        raise LearnedBenchmarkError(f"Required cells are not mapped: {missing[:10]}")
    graphs = context.folds[fold].graphs
    vectors: list[Tensor] = []
    for source_id in source_ids:
        biological_id = lookup[source_id]
        if biological_id not in graphs:
            raise LearnedBenchmarkError(f"Fold {fold} graph cache lacks {biological_id}")
        node_features = graphs[biological_id].node_features
        if node_features.ndim != 2 or node_features.shape[0] != HALLMARK_NODE_COUNT:
            raise LearnedBenchmarkError(
                f"Fold {fold} graph {biological_id} does not have 50 Hallmark nodes"
            )
        if node_features.shape[1] < 2:
            raise LearnedBenchmarkError(
                f"Fold {fold} graph {biological_id} lacks activity/coverage channels"
            )
        features = (
            node_features[:, 0:2]
            .detach()
            .to(dtype=torch.float32, device="cpu")
            .contiguous()
            .reshape(-1)
        )
        if not bool(torch.isfinite(features).all()):
            raise LearnedBenchmarkError(f"Fold {fold} graph {biological_id} is non-finite")
        vectors.append(features)
    return source_ids, torch.stack(vectors)


POINT_COLUMNS = (
    "source_combodrugseq",
    "prediction_group_id",
    "surface_replicate_id",
    "cell_id",
    "drug_a_id",
    "drug_b_id",
    "dose_a_model",
    "dose_b_model",
    "mono_response_a",
    "mono_response_b",
    "mono_mask_a",
    "mono_mask_b",
    "response",
    "response_clipped_high",
    "response_mask",
    "primary_metric_mask",
)

SHARD_POINT_COLUMNS = (*POINT_COLUMNS, "heldout_cluster_id")
MODEL_ROLES = ("train", "validation", "test")


def _strict_source_ids(values: pd.Series, label: str) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any():
        raise LearnedBenchmarkError(f"{label} contains non-numeric source IDs")
    as_float = numeric.to_numpy(dtype=np.float64)
    if not bool(np.isfinite(as_float).all()) or not bool(
        np.equal(as_float, np.floor(as_float)).all()
    ):
        raise LearnedBenchmarkError(f"{label} source IDs must be finite integers")
    limits = np.iinfo(np.int64)
    if bool((as_float < limits.min).any()) or bool((as_float > limits.max).any()):
        raise LearnedBenchmarkError(f"{label} source IDs exceed int64 range")
    return as_float.astype(np.int64)


def _source_id_sequence_hash(source_ids: np.ndarray) -> str:
    canonical = np.asarray(source_ids, dtype="<i8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _source_id_inventory(source_ids: np.ndarray) -> dict[str, Any]:
    canonical = np.asarray(source_ids, dtype=np.int64)
    return {
        "count": int(len(canonical)),
        "sequence_sha256": _source_id_sequence_hash(canonical),
    }


def _verify_fold_source_partition(
    shards: FrozenLearnedPointShards,
    fold: int,
) -> None:
    record = shards.source_partition_record(fold)
    path = shards.source_partition_path(fold)
    expected_relative = str(_source_partition_relative_path(fold))
    if record.get("relative_path") != expected_relative:
        raise LearnedBenchmarkError(
            f"Fold {fold} source-partition relative path changed"
        )
    if record.get("path") != str(path):
        raise LearnedBenchmarkError(f"Fold {fold} source-partition path changed")
    if record.get("columns") != ["source_combodrugseq", "role"]:
        raise LearnedBenchmarkError(f"Fold {fold} source-partition columns changed")
    if record.get("contains_response_values") is not False:
        raise LearnedBenchmarkError(
            f"Fold {fold} source-partition ledger is not label-free"
        )
    _verify_recorded_file(path, record, f"Fold {fold} source-partition ledger")
    import pyarrow.parquet as parquet

    parquet_file = parquet.ParquetFile(path)
    if parquet_file.schema_arrow.names != ["source_combodrugseq", "role"]:
        raise LearnedBenchmarkError(
            f"Fold {fold} source-partition ledger contains forbidden columns"
        )
    frame = pd.read_parquet(path, columns=["source_combodrugseq", "role"])
    if len(frame) != int(record.get("rows", -1)):
        raise LearnedBenchmarkError(
            f"Fold {fold} source-partition ledger row count changed"
        )
    source_ids = _strict_source_ids(
        frame["source_combodrugseq"],
        f"fold {fold} source-partition ledger",
    )
    roles = frame["role"].map(_clean_text)
    if not bool(roles.isin(MODEL_ROLES).all()):
        raise LearnedBenchmarkError(
            f"Fold {fold} source-partition ledger contains an invalid role"
        )
    if len(set(source_ids.tolist())) != len(source_ids):
        raise LearnedBenchmarkError(
            f"Fold {fold} source rows overlap across train/validation/test"
        )
    role_counts: dict[str, int] = {}
    for role in MODEL_ROLES:
        role_record = shards.role_record(fold, role)
        inventory = role_record.get("source_ids")
        if not isinstance(inventory, dict):
            raise LearnedBenchmarkError(
                f"Fold {fold} {role} point shard lacks source-ID inventory"
            )
        selected_ids = source_ids[roles.to_numpy() == role]
        actual_inventory = _source_id_inventory(selected_ids)
        if actual_inventory != inventory:
            raise LearnedBenchmarkError(
                f"Fold {fold} {role} source IDs differ from the partition ledger"
            )
        count = actual_inventory["count"]
        if count != int(role_record.get("rows", -1)):
            raise LearnedBenchmarkError(
                f"Fold {fold} {role} source-ID count differs from point-shard rows"
            )
        role_counts[role] = count
    if record.get("role_counts") != role_counts or int(
        record.get("rows", -1)
    ) != sum(role_counts.values()):
        raise LearnedBenchmarkError(
            f"Fold {fold} source-partition ledger counts are inconsistent"
        )


def _assert_source_roles_disjoint(
    frames: Mapping[str, pd.DataFrame],
    fold: int,
) -> None:
    role_sets = {
        role: set(_strict_source_ids(frame["source_combodrugseq"], role).tolist())
        for role, frame in frames.items()
    }
    roles = tuple(role_sets)
    for index, left in enumerate(roles):
        for right in roles[index + 1 :]:
            if not role_sets[left].isdisjoint(role_sets[right]):
                raise LearnedBenchmarkError(
                    f"Fold {fold} source rows overlap across {left}/{right}"
                )


def _point_shard_root(config: BenchmarkConfig) -> Path:
    return (config.output_dir / POINT_SHARD_DIRECTORY).resolve()


def _point_shard_relative_path(fold: int, role: str) -> Path:
    return Path(f"fold_{fold}") / f"{role}.parquet"


def _source_partition_relative_path(fold: int) -> Path:
    return Path(f"fold_{fold}") / SOURCE_PARTITION_LEDGER


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise LearnedBenchmarkError(f"{label} is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LearnedBenchmarkError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise LearnedBenchmarkError(f"{label} must contain a JSON object")
    return payload


def _verify_self_hash(payload: Mapping[str, Any], label: str) -> None:
    recorded = payload.get("manifest_sha256")
    unhashed = dict(payload)
    unhashed.pop("manifest_sha256", None)
    if recorded != _canonical_hash(unhashed):
        raise LearnedBenchmarkError(f"{label} content hash changed")


def _verify_recorded_file(path: Path, record: Mapping[str, Any], label: str) -> None:
    if not path.is_file():
        raise LearnedBenchmarkError(f"{label} is missing: {path}")
    actual = _file_record(path)
    expected = {
        "path": str(path.resolve()),
        "bytes": int(record.get("bytes", -1)),
        "sha256": str(record.get("sha256", "")),
    }
    if actual != expected:
        raise LearnedBenchmarkError(f"{label} hash or byte size changed")


def prepare_learned_point_shards(
    config: BenchmarkConfig | str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Materialize response rows into physically separate fold/role Parquet files.

    This deterministic preparation step is the only learned-benchmark operation
    that scans the mixed-role source Parquet.  Model selection subsequently opens
    only train/validation shards, while evaluation opens test shards only after a
    global checkpoint barrier.
    """

    benchmark_config = (
        config if isinstance(config, BenchmarkConfig) else load_benchmark_config(config)
    )
    root = _point_shard_root(benchmark_config)
    manifest_path = root / POINT_SHARD_MANIFEST
    if manifest_path.is_file() and not overwrite:
        return load_learned_point_shards(benchmark_config).manifest

    split = load_one_new_scaffold_benchmark(benchmark_config)
    points_path = benchmark_config.nci_processed_dir / "points.parquet"
    if not points_path.is_file():
        raise LearnedBenchmarkError(f"Prepared point table is missing: {points_path}")

    import pyarrow as arrow
    import pyarrow.parquet as parquet

    parquet_file = parquet.ParquetFile(points_path)
    missing = sorted(set(POINT_COLUMNS) - set(parquet_file.schema_arrow.names))
    if missing:
        raise LearnedBenchmarkError(f"Prepared point Parquet is missing columns: {missing}")

    role_ledgers: dict[int, pd.DataFrame] = {}
    role_maps: dict[int, pd.Series] = {}
    cluster_maps: dict[int, pd.Series] = {}
    expected: dict[tuple[int, str], dict[str, int]] = {}
    for fold in range(benchmark_config.folds):
        ledger = _fold_surface_roles(split.surfaces, fold)
        role_ledgers[fold] = ledger
        indexed = ledger.set_index("surface_replicate_id")
        role_maps[fold] = indexed["role"]
        cluster_maps[fold] = indexed["heldout_cluster_id"]
        for role in MODEL_ROLES:
            selected = ledger.loc[ledger["role"].eq(role)]
            expected[(fold, role)] = {
                "rows": int(selected["primary_points"].sum()),
                "surfaces": int(len(selected)),
                "groups": int(selected["prediction_group_id"].nunique()),
            }

    root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{POINT_SHARD_DIRECTORY}.", dir=root.parent)
    ).resolve()
    writers: dict[tuple[int, str], Any] = {}
    observed_rows = {
        (fold, role): 0
        for fold in range(benchmark_config.folds)
        for role in MODEL_ROLES
    }
    observed_surfaces = {
        (fold, role): set()
        for fold in range(benchmark_config.folds)
        for role in MODEL_ROLES
    }
    observed_groups = {
        (fold, role): set()
        for fold in range(benchmark_config.folds)
        for role in MODEL_ROLES
    }
    observed_source_hashers = {
        (fold, role): hashlib.sha256()
        for fold in range(benchmark_config.folds)
        for role in MODEL_ROLES
    }
    eligible_source_ids: set[int] = set()
    eligible_source_hasher = hashlib.sha256()
    try:
        for row_group in range(parquet_file.metadata.num_row_groups):
            frame = parquet_file.read_row_group(
                row_group, columns=list(POINT_COLUMNS)
            ).to_pandas()
            primary = _strict_boolean(frame["primary_metric_mask"], "primary_metric_mask")
            response = _strict_boolean(frame["response_mask"], "response_mask")
            eligible = frame.loc[primary & response].copy()
            if eligible.empty:
                continue
            source_ids = _strict_source_ids(
                eligible["source_combodrugseq"],
                "eligible prepared points",
            )
            source_set = set(source_ids.tolist())
            if len(source_set) != len(source_ids) or not eligible_source_ids.isdisjoint(
                source_set
            ):
                raise LearnedBenchmarkError(
                    "Eligible prepared points contain repeated source IDs"
                )
            eligible_source_ids.update(source_set)
            eligible_source_hasher.update(np.asarray(source_ids, dtype="<i8").tobytes())
            eligible["source_combodrugseq"] = source_ids
            eligible["surface_replicate_id"] = eligible[
                "surface_replicate_id"
            ].map(_clean_text)
            for fold in range(benchmark_config.folds):
                roles = eligible["surface_replicate_id"].map(role_maps[fold])
                modelled = roles.isin(MODEL_ROLES)
                if not bool(modelled.any()):
                    continue
                candidate = eligible.loc[modelled].copy()
                candidate["_role"] = roles.loc[modelled].to_numpy()
                candidate["heldout_cluster_id"] = candidate[
                    "surface_replicate_id"
                ].map(cluster_maps[fold])
                indexed = role_ledgers[fold].set_index("surface_replicate_id")
                for column in (
                    "prediction_group_id",
                    "drug_a_id",
                    "drug_b_id",
                    "cell_id",
                ):
                    observed = candidate[column].map(_clean_text)
                    expected_values = candidate["surface_replicate_id"].map(
                        indexed[column]
                    )
                    if not bool(observed.eq(expected_values).all()):
                        raise LearnedBenchmarkError(
                            f"Fold {fold} shard metadata disagrees for {column}"
                        )
                    candidate[column] = observed
                mono_a = _strict_boolean(candidate["mono_mask_a"], "mono_mask_a")
                mono_b = _strict_boolean(candidate["mono_mask_b"], "mono_mask_b")
                if not bool((mono_a & mono_b).all()):
                    raise LearnedBenchmarkError(
                        f"Fold {fold} shard contains incomplete observed monotherapy"
                    )
                for role in MODEL_ROLES:
                    role_frame = candidate.loc[
                        candidate["_role"].eq(role), list(SHARD_POINT_COLUMNS)
                    ]
                    if role_frame.empty:
                        continue
                    key = (fold, role)
                    output_path = temporary_root / _point_shard_relative_path(
                        fold, role
                    )
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    table = arrow.Table.from_pandas(role_frame, preserve_index=False)
                    if key not in writers:
                        writers[key] = parquet.ParquetWriter(
                            output_path,
                            table.schema,
                            compression="zstd",
                        )
                    writers[key].write_table(table)
                    observed_rows[key] += len(role_frame)
                    observed_surfaces[key].update(
                        role_frame["surface_replicate_id"].astype(str)
                    )
                    observed_groups[key].update(
                        role_frame["prediction_group_id"].astype(str)
                    )
                    role_source_ids = _strict_source_ids(
                        role_frame["source_combodrugseq"],
                        f"fold {fold} {role} prepared shard",
                    )
                    observed_source_hashers[key].update(
                        np.asarray(role_source_ids, dtype="<i8").tobytes()
                    )
        for writer in writers.values():
            writer.close()
        writers.clear()

        fold_records: list[dict[str, Any]] = []
        for fold in range(benchmark_config.folds):
            roles_record: dict[str, Any] = {}
            partition_frames: dict[str, pd.DataFrame] = {}
            for role in MODEL_ROLES:
                key = (fold, role)
                expected_role = expected[key]
                actual_role = {
                    "rows": observed_rows[key],
                    "surfaces": len(observed_surfaces[key]),
                    "groups": len(observed_groups[key]),
                }
                if actual_role != expected_role:
                    raise LearnedBenchmarkError(
                        f"Fold {fold} {role} point shard differs from frozen ledger: "
                        f"expected {expected_role}, observed {actual_role}"
                    )
                temporary_path = temporary_root / _point_shard_relative_path(
                    fold, role
                )
                if not temporary_path.is_file():
                    raise LearnedBenchmarkError(
                        f"Fold {fold} {role} point shard was not written"
                    )
                source_frame = pd.read_parquet(
                    temporary_path,
                    columns=["source_combodrugseq"],
                )
                source_ids = _strict_source_ids(
                    source_frame["source_combodrugseq"],
                    f"fold {fold} {role} temporary point shard",
                )
                source_inventory = _source_id_inventory(source_ids)
                expected_source_inventory = {
                    "count": actual_role["rows"],
                    "sequence_sha256": observed_source_hashers[key].hexdigest(),
                }
                if source_inventory != expected_source_inventory:
                    raise LearnedBenchmarkError(
                        f"Fold {fold} {role} written source-ID sequence changed"
                    )
                partition_frames[role] = pd.DataFrame(
                    {
                        "source_combodrugseq": source_ids,
                        "role": np.full(len(source_ids), role, dtype=object),
                    }
                )
                final_path = root / _point_shard_relative_path(fold, role)
                final_path.parent.mkdir(parents=True, exist_ok=True)
                os.replace(temporary_path, final_path)
                roles_record[role] = {
                    "relative_path": str(_point_shard_relative_path(fold, role)),
                    **_file_record(final_path),
                    **actual_role,
                    "source_ids": source_inventory,
                }
            _assert_source_roles_disjoint(partition_frames, fold)
            partition_frame = pd.concat(
                [partition_frames[role] for role in MODEL_ROLES],
                ignore_index=True,
            )
            role_counts = {
                role: int(roles_record[role]["source_ids"]["count"])
                for role in MODEL_ROLES
            }
            temporary_partition = (
                temporary_root / _source_partition_relative_path(fold)
            )
            _write_parquet_atomic(temporary_partition, partition_frame)
            final_partition = root / _source_partition_relative_path(fold)
            final_partition.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temporary_partition, final_partition)
            partition_record = {
                "relative_path": str(_source_partition_relative_path(fold)),
                **_file_record(final_partition),
                "columns": ["source_combodrugseq", "role"],
                "rows": len(partition_frame),
                "role_counts": role_counts,
                "contains_response_values": False,
                "pairwise_disjoint_verified_during_preparation": True,
            }
            fold_records.append(
                {
                    "fold": fold,
                    "roles": roles_record,
                    "source_partition": partition_record,
                }
            )

        manifest: dict[str, Any] = {
            "status": "passed",
            "point_shard_version": LEARNED_POINT_SHARD_VERSION,
            "protocol": benchmark_config.protocol,
            "endpoint": benchmark_config.endpoint,
            "endpoint_decision_status": benchmark_config.endpoint_decision_status,
            "split_manifest_hash": split.manifest["manifest_sha256"],
            "split_hash": split.split_hash,
            "source_points": dict(split.manifest["sources"]["points"]),
            "eligible_source_ids": {
                "count": len(eligible_source_ids),
                "sequence_sha256": eligible_source_hasher.hexdigest(),
                "globally_unique": True,
            },
            "columns": list(SHARD_POINT_COLUMNS),
            "preparation_contract": (
                "deterministic_physical_role_partition_plus_label_free_"
                "source_partition_ledger_no_model_selection"
            ),
            "folds": fold_records,
        }
        manifest["manifest_sha256"] = _canonical_hash(manifest)
        _write_json_atomic(manifest_path, manifest)
    finally:
        for writer in writers.values():
            writer.close()
        shutil.rmtree(temporary_root, ignore_errors=True)
    return load_learned_point_shards(benchmark_config).manifest


def load_learned_point_shards(
    config: BenchmarkConfig | str | Path,
    *,
    verify_roles: Sequence[str] = MODEL_ROLES,
) -> FrozenLearnedPointShards:
    """Verify label-free partitions plus only explicitly authorized response files."""

    benchmark_config = (
        config if isinstance(config, BenchmarkConfig) else load_benchmark_config(config)
    )
    requested_roles = tuple(str(role) for role in verify_roles)
    if not set(requested_roles).issubset(MODEL_ROLES):
        raise LearnedBenchmarkError(f"Invalid roles requested for verification: {requested_roles}")
    root = _point_shard_root(benchmark_config)
    manifest = _load_json_object(root / POINT_SHARD_MANIFEST, "Point-shard manifest")
    _verify_self_hash(manifest, "Point-shard manifest")
    expected_header = {
        "status": "passed",
        "point_shard_version": LEARNED_POINT_SHARD_VERSION,
        "protocol": benchmark_config.protocol,
        "endpoint": benchmark_config.endpoint,
        "endpoint_decision_status": benchmark_config.endpoint_decision_status,
        "columns": list(SHARD_POINT_COLUMNS),
    }
    mismatched = [
        key for key, expected_value in expected_header.items()
        if manifest.get(key) != expected_value
    ]
    if mismatched:
        raise LearnedBenchmarkError(
            f"Point-shard manifest context changed: {sorted(mismatched)}"
        )
    frozen = FrozenLearnedPointShards(root=root, manifest=manifest)
    records = manifest.get("folds")
    if not isinstance(records, list) or len(records) != benchmark_config.folds:
        raise LearnedBenchmarkError("Point-shard fold inventory is incomplete")
    if {record.get("fold") for record in records if isinstance(record, dict)} != set(
        range(benchmark_config.folds)
    ):
        raise LearnedBenchmarkError("Point-shard fold IDs are incomplete")
    eligible_inventory = manifest.get("eligible_source_ids")
    if (
        not isinstance(eligible_inventory, dict)
        or eligible_inventory.get("globally_unique") is not True
        or int(eligible_inventory.get("count", -1)) < 1
        or len(str(eligible_inventory.get("sequence_sha256", ""))) != 64
    ):
        raise LearnedBenchmarkError("Point-shard global source-ID inventory is invalid")
    all_paths: list[Path] = []
    for fold in range(benchmark_config.folds):
        _verify_fold_source_partition(frozen, fold)
        all_paths.append(frozen.source_partition_path(fold))
        for role in MODEL_ROLES:
            record = frozen.role_record(fold, role)
            path = frozen.role_path(fold, role)
            expected_relative = str(_point_shard_relative_path(fold, role))
            if record.get("relative_path") != expected_relative:
                raise LearnedBenchmarkError(
                    f"Fold {fold} {role} point-shard relative path changed"
                )
            if record.get("path") != str(path):
                raise LearnedBenchmarkError(
                    f"Fold {fold} {role} point-shard path changed"
                )
            all_paths.append(path)
            if role in requested_roles:
                _verify_recorded_file(
                    path, record, f"Fold {fold} {role} point shard"
                )
    expected_path_count = benchmark_config.folds * (len(MODEL_ROLES) + 1)
    if len(set(all_paths)) != expected_path_count:
        raise LearnedBenchmarkError(
            "Point-shard and source-partition paths are not pairwise distinct"
        )
    return frozen


def load_learned_role_frame(
    shards: FrozenLearnedPointShards,
    surfaces: pd.DataFrame,
    fold: int,
    role: str,
    *,
    max_train_points: int | None = None,
    sample_seed: int = 0,
) -> pd.DataFrame:
    """Decode one authorized physical role file and recheck it against the ledger."""

    path = shards.role_path(fold, role)
    record = shards.role_record(fold, role)
    _verify_fold_source_partition(shards, fold)
    _verify_recorded_file(path, record, f"Fold {fold} {role} point shard")
    frame = pd.read_parquet(path, columns=list(SHARD_POINT_COLUMNS))
    if len(frame) != int(record.get("rows", -1)):
        raise LearnedBenchmarkError(
            f"Fold {fold} {role} raw shard row count changed before filtering"
        )
    source_ids = _strict_source_ids(
        frame["source_combodrugseq"],
        f"fold {fold} {role} raw shard",
    )
    if len(set(source_ids.tolist())) != len(source_ids):
        raise LearnedBenchmarkError(
            f"Fold {fold} {role} raw shard source IDs are invalid or duplicated"
        )
    inventory = record["source_ids"]
    if _source_id_inventory(source_ids) != inventory:
        raise LearnedBenchmarkError(
            f"Fold {fold} {role} raw shard source-ID inventory changed"
        )
    frame["source_combodrugseq"] = source_ids
    assignments = _fold_surface_roles(surfaces, fold).set_index(
        "surface_replicate_id"
    )
    raw_surface_ids = frame["surface_replicate_id"].map(_clean_text)
    raw_roles = raw_surface_ids.map(assignments["role"])
    if raw_roles.isna().any() or not bool(raw_roles.eq(role).all()):
        raise LearnedBenchmarkError(
            f"Fold {fold} {role} physical shard contains rows assigned to another role"
        )
    selected = select_fold_points_from_chunks(
        [frame],
        surfaces,
        fold,
        roles=(role,),
        max_train_points=max_train_points if role == "train" else None,
        sample_seed=sample_seed,
    )[role]
    if max_train_points is None and len(selected) != len(frame):
        raise LearnedBenchmarkError(f"Fold {fold} {role} shard filtering changed rows")
    return selected


def _fold_surface_roles(surfaces: pd.DataFrame, fold: int) -> pd.DataFrame:
    role_column = f"role_fold_{fold}"
    required = (
        "surface_replicate_id",
        "prediction_group_id",
        "drug_a_id",
        "drug_b_id",
        "cell_id",
        "heldout_cluster_id",
        "primary_points",
        role_column,
    )
    _require_columns(surfaces, required, "surface assignment ledger")
    selected = surfaces.loc[
        surfaces[role_column].astype(str).isin(("train", "validation", "test")),
        list(required),
    ].copy()
    if selected.empty:
        raise LearnedBenchmarkError(f"Fold {fold} contains no modelled surface roles")
    if selected["surface_replicate_id"].duplicated().any():
        raise LearnedBenchmarkError("Surface assignment ledger contains duplicate surface IDs")
    text_columns = (
        "surface_replicate_id",
        "prediction_group_id",
        "drug_a_id",
        "drug_b_id",
        "cell_id",
        "heldout_cluster_id",
    )
    for column in text_columns:
        selected[column] = selected[column].map(_clean_text)
        if (selected[column] == "").any():
            raise LearnedBenchmarkError(f"Surface assignment column {column} contains empty IDs")
    primary_points = pd.to_numeric(selected["primary_points"], errors="coerce")
    if primary_points.isna().any() or (primary_points < 1).any():
        raise LearnedBenchmarkError("Surface ledger primary_points must be positive integers")
    selected["primary_points"] = primary_points.astype(np.int64)
    selected = selected.rename(columns={role_column: "role"})
    return selected


def select_fold_points_from_chunks(
    chunks: Iterable[pd.DataFrame],
    surfaces: pd.DataFrame,
    fold: int,
    *,
    roles: Sequence[str] = ("train", "validation", "test"),
    max_train_points: int | None = None,
    sample_seed: int = 0,
) -> dict[str, pd.DataFrame]:
    """Select a fold from bounded chunks and fail closed on masks or metadata drift."""

    selected_roles = tuple(str(role) for role in roles)
    if not selected_roles or len(set(selected_roles)) != len(selected_roles):
        raise LearnedBenchmarkError("Requested roles must be non-empty and unique")
    if not set(selected_roles).issubset({"train", "validation", "test"}):
        raise LearnedBenchmarkError(f"Invalid requested roles: {selected_roles}")
    assignments = _fold_surface_roles(surfaces, fold)
    assignments = assignments.loc[assignments["role"].isin(selected_roles)].copy()
    by_surface = assignments.set_index("surface_replicate_id")
    surface_ids = set(by_surface.index)
    selected_chunks: list[pd.DataFrame] = []
    for chunk in chunks:
        _require_columns(chunk, POINT_COLUMNS, "point chunk")
        candidate = chunk.loc[
            chunk["surface_replicate_id"].astype(str).isin(surface_ids), list(POINT_COLUMNS)
        ].copy()
        if candidate.empty:
            continue
        primary = _strict_boolean(candidate["primary_metric_mask"], "primary_metric_mask")
        response = _strict_boolean(candidate["response_mask"], "response_mask")
        evaluated = primary & response
        candidate = candidate.loc[evaluated].copy()
        if candidate.empty:
            continue
        mono_a = _strict_boolean(candidate["mono_mask_a"], "mono_mask_a")
        mono_b = _strict_boolean(candidate["mono_mask_b"], "mono_mask_b")
        if not bool((mono_a & mono_b).all()):
            count = int((~(mono_a & mono_b)).sum())
            raise LearnedBenchmarkError(
                f"Fold {fold} has {count} primary response points without both monotherapies"
            )
        candidate["role"] = candidate["surface_replicate_id"].astype(str).map(
            by_surface["role"]
        )
        candidate["heldout_cluster_id"] = candidate["surface_replicate_id"].astype(str).map(
            by_surface["heldout_cluster_id"]
        )
        for column in (
            "prediction_group_id",
            "drug_a_id",
            "drug_b_id",
            "cell_id",
        ):
            observed = candidate[column].map(_clean_text)
            expected = candidate["surface_replicate_id"].astype(str).map(by_surface[column])
            mismatch = ~observed.eq(expected)
            if bool(mismatch.any()):
                source = candidate.loc[mismatch, "source_combodrugseq"].iloc[0]
                raise LearnedBenchmarkError(
                    f"Point metadata disagrees with the surface ledger for {column}; row {source}"
                )
            candidate[column] = observed
        selected_chunks.append(candidate)
    if not selected_chunks:
        raise LearnedBenchmarkError(f"No prepared points match fold {fold}")
    selected = pd.concat(selected_chunks, ignore_index=True)
    source_numeric = pd.to_numeric(selected["source_combodrugseq"], errors="coerce")
    if source_numeric.isna().any() or selected["source_combodrugseq"].duplicated().any():
        raise LearnedBenchmarkError("Selected source row IDs must be finite and unique")
    selected["source_combodrugseq"] = source_numeric.astype(np.int64)
    observed_surfaces = set(selected["surface_replicate_id"].astype(str))
    missing_surfaces = sorted(surface_ids - observed_surfaces)
    if missing_surfaces:
        raise LearnedBenchmarkError(
            f"Fold {fold} assignments lack primary evaluable points: {missing_surfaces[:10]}"
        )
    observed_counts = selected.groupby("surface_replicate_id", sort=True).size()
    expected_counts = assignments.set_index("surface_replicate_id")["primary_points"]
    count_mismatch = observed_counts.reindex(expected_counts.index).ne(expected_counts)
    if bool(count_mismatch.any()):
        surface_id = str(count_mismatch[count_mismatch].index[0])
        raise LearnedBenchmarkError(
            f"Fold {fold} surface {surface_id} primary-point count differs from frozen ledger"
        )
    roles: dict[str, pd.DataFrame] = {}
    for role in selected_roles:
        frame = selected.loc[selected["role"].eq(role)].copy()
        if frame.empty:
            raise LearnedBenchmarkError(f"Fold {fold} has no {role} points")
        roles[role] = frame.sort_values(
            "source_combodrugseq", kind="mergesort"
        ).reset_index(drop=True)
    role_sets = {role: set(frame["source_combodrugseq"]) for role, frame in roles.items()}
    for left_index, left in enumerate(selected_roles):
        for right in selected_roles[left_index + 1 :]:
            if not role_sets[left].isdisjoint(role_sets[right]):
                raise LearnedBenchmarkError(f"{left} and {right} point roles overlap")
    if max_train_points is not None and "train" in roles:
        if max_train_points < 1:
            raise LearnedBenchmarkError("max_train_points must be positive when supplied")
        train = roles["train"]
        if len(train) > max_train_points:
            surface_ranks = {
                surface: hashlib.sha256(
                    f"{sample_seed}|{surface}".encode("utf-8")
                ).hexdigest()
                for surface in sorted(set(train["surface_replicate_id"].astype(str)))
            }
            chosen_surfaces: list[str] = []
            chosen_points = 0
            surface_sizes = train.groupby("surface_replicate_id", sort=True).size().to_dict()
            for surface in sorted(surface_ranks, key=surface_ranks.get):
                size = int(surface_sizes[surface])
                if chosen_points + size <= max_train_points:
                    chosen_surfaces.append(surface)
                    chosen_points += size
            if not chosen_surfaces:
                raise LearnedBenchmarkError(
                    "max_train_points is smaller than one complete training surface"
                )
            roles["train"] = train.loc[
                train["surface_replicate_id"].astype(str).isin(chosen_surfaces)
            ].sort_values(
                "source_combodrugseq", kind="mergesort"
            ).reset_index(drop=True)
    return roles


def build_compact_point_table(
    frame: pd.DataFrame,
    banks: CompactFeatureBanks,
    *,
    role: str,
    fold: int,
    monotherapy_upper_bound: float = 1.0,
) -> CompactPointTable:
    """Map point IDs to compact feature-bank indices."""

    required = (
        "source_combodrugseq",
        "prediction_group_id",
        "surface_replicate_id",
        "heldout_cluster_id",
        "drug_a_id",
        "drug_b_id",
        "cell_id",
        "dose_a_model",
        "dose_b_model",
        "mono_response_a",
        "mono_response_b",
        "response",
        "response_clipped_high",
    )
    _require_columns(frame, required, f"fold {fold} {role} points")
    drug_lookup = {drug_id: index for index, drug_id in enumerate(banks.drug_ids)}
    cell_lookup = {cell_id: index for index, cell_id in enumerate(banks.cell_ids)}

    def indices(column: str, lookup: Mapping[str, int]) -> Tensor:
        mapped = frame[column].map(_clean_text).map(lookup)
        if mapped.isna().any():
            missing = sorted(set(frame.loc[mapped.isna(), column].map(_clean_text)))
            raise LearnedBenchmarkError(f"Feature bank lacks {column} IDs: {missing[:10]}")
        return torch.as_tensor(mapped.to_numpy(np.int32), dtype=torch.int32)

    numeric_columns = (
        "dose_a_model",
        "dose_b_model",
        "mono_response_a",
        "mono_response_b",
        "response",
    )
    numeric = frame.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any(axis=None) or not np.isfinite(numeric.to_numpy(float)).all():
        raise LearnedBenchmarkError(f"Fold {fold} {role} numeric values must be finite")
    source = pd.to_numeric(frame["source_combodrugseq"], errors="coerce")
    if source.isna().any():
        raise LearnedBenchmarkError("Source row IDs must be numeric")
    clipped_high = _strict_boolean(
        frame["response_clipped_high"], "response_clipped_high"
    )
    return CompactPointTable(
        role=role,
        fold=fold,
        source_rows=source.to_numpy(np.int64),
        group_ids=frame["prediction_group_id"].map(_clean_text).to_numpy(object),
        surface_ids=frame["surface_replicate_id"].map(_clean_text).to_numpy(object),
        cluster_ids=frame["heldout_cluster_id"].map(_clean_text).to_numpy(object),
        drug_a_index=indices("drug_a_id", drug_lookup),
        drug_b_index=indices("drug_b_id", drug_lookup),
        cell_index=indices("cell_id", cell_lookup),
        doses=torch.as_tensor(
            numeric.loc[:, ["dose_a_model", "dose_b_model"]].to_numpy(np.float32)
        ),
        mono_a=torch.as_tensor(numeric["mono_response_a"].to_numpy(np.float32)),
        mono_b=torch.as_tensor(numeric["mono_response_b"].to_numpy(np.float32)),
        target=torch.as_tensor(numeric["response"].to_numpy(np.float32)),
        response_clipped_high=clipped_high,
        monotherapy_upper_bound=float(monotherapy_upper_bound),
    )


def build_feature_banks(
    required_drugs: Sequence[str],
    required_cells: Sequence[str],
    context: PreparedBenchmarkCellContext,
    fold: int,
    config: BenchmarkConfig,
) -> CompactFeatureBanks:
    """Build banks from assignment metadata without reading any response values."""

    drugs = pd.read_csv(config.nci_processed_dir / "drugs.csv", keep_default_na=False)
    drug_ids, fingerprints, fingerprint_hashes = build_morgan_bank(
        drugs,
        required_drugs,
        radius=config.fingerprint_radius,
        bits=config.fingerprint_bits,
    )
    cell_ids, hallmarks = build_hallmark_bank(context, fold, required_cells)
    return CompactFeatureBanks(
        drug_ids=drug_ids,
        cell_ids=cell_ids,
        fingerprints=fingerprints,
        hallmarks=hallmarks,
        fingerprint_hashes=fingerprint_hashes,
    )


def build_compact_fold_data(
    frames: Mapping[str, pd.DataFrame],
    split: PreparedChemicalSplit,
    context: PreparedBenchmarkCellContext,
    fold: int,
    config: BenchmarkConfig,
) -> CompactFoldData:
    del split  # The verified split supplies frames; it is retained in this public contract.
    required_drugs = sorted(
        {
            _clean_text(drug_id)
            for role in ("train", "validation", "test")
            for column in ("drug_a_id", "drug_b_id")
            for drug_id in frames[role][column]
        }
    )
    required_cells = sorted(
        {
            _clean_text(cell_id)
            for role in ("train", "validation", "test")
            for cell_id in frames[role]["cell_id"]
        }
    )
    banks = build_feature_banks(required_drugs, required_cells, context, fold, config)
    tables = {
        role: build_compact_point_table(
            frames[role],
            banks,
            role=role,
            fold=fold,
            monotherapy_upper_bound=config.response_upper_bound,
        )
        for role in ("train", "validation", "test")
    }
    return CompactFoldData(fold=fold, banks=banks, **tables)


def surface_macro_rmse(target: np.ndarray, prediction: np.ndarray, surfaces: np.ndarray) -> float:
    """Mean within-surface RMSE with equal surface weight."""

    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    surfaces = np.asarray(surfaces)
    if target.ndim != 1 or target.shape != prediction.shape or target.shape != surfaces.shape:
        raise LearnedBenchmarkError("Metric inputs must be aligned one-dimensional arrays")
    if target.size == 0 or not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise LearnedBenchmarkError("Metric inputs must be non-empty and finite")
    frame = pd.DataFrame(
        {"surface": surfaces, "squared_error": np.square(prediction - target)}
    )
    per_surface = np.sqrt(frame.groupby("surface", sort=True)["squared_error"].mean())
    if per_surface.empty or not np.isfinite(per_surface.to_numpy()).all():
        raise LearnedBenchmarkError("Surface-macro RMSE is undefined")
    return float(per_surface.mean())


def _resolve_device(requested: str | torch.device | None) -> torch.device:
    if requested is None or str(requested) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise LearnedBenchmarkError("CUDA was requested but is unavailable")
    return device


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _batch_indices(size: int, batch_size: int, *, shuffle_seed: int | None) -> list[Tensor]:
    if shuffle_seed is None:
        order = torch.arange(size, dtype=torch.int64)
    else:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(shuffle_seed))
        order = torch.randperm(size, generator=generator)
    return list(order.split(batch_size))


def _surface_batch_indices(
    table: CompactPointTable,
    batch_surfaces: int,
    *,
    shuffle_seed: int,
) -> list[Tensor]:
    """Select surfaces and include every point belonging to each selected surface."""

    surface_to_indices: dict[str, list[int]] = {}
    for point_index, surface_id in enumerate(table.surface_ids):
        surface_to_indices.setdefault(str(surface_id), []).append(point_index)
    surface_ids = sorted(surface_to_indices)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(shuffle_seed))
    order = torch.randperm(len(surface_ids), generator=generator).tolist()
    batches: list[Tensor] = []
    for start in range(0, len(order), batch_surfaces):
        selected = [surface_ids[index] for index in order[start : start + batch_surfaces]]
        point_indices = [index for surface in selected for index in surface_to_indices[surface]]
        batch = torch.as_tensor(point_indices, dtype=torch.int64)
        observed = {
            str(table.surface_ids[index])
            for index in batch.tolist()
        }
        if observed != set(selected):
            raise LearnedBenchmarkError("A training surface batch lost a selected surface")
        for surface in selected:
            expected = surface_to_indices[surface]
            actual = [index for index in batch.tolist() if str(table.surface_ids[index]) == surface]
            if actual != expected:
                raise LearnedBenchmarkError(
                    f"Training batch does not contain every primary point for {surface}"
                )
        batches.append(batch)
    return batches


def _surface_mean_mse(
    prediction: Tensor,
    target: Tensor,
    surface_ids: np.ndarray,
) -> Tensor:
    """Mean point MSE within each surface, followed by equal surface averaging."""

    if prediction.shape != target.shape or prediction.ndim != 1:
        raise LearnedBenchmarkError("Surface loss tensors must have aligned shape [points]")
    if len(surface_ids) != prediction.shape[0]:
        raise LearnedBenchmarkError("Surface loss IDs do not align with predictions")
    groups: dict[str, list[int]] = {}
    for index, surface_id in enumerate(surface_ids):
        groups.setdefault(str(surface_id), []).append(index)
    losses = [
        torch.mean(torch.square(prediction[indices] - target[indices]))
        for indices in groups.values()
    ]
    if not losses:
        raise LearnedBenchmarkError("Cannot compute a loss over zero surfaces")
    return torch.stack(losses).mean()


def _evaluation_batch_size(training: LearnedMorganConfig) -> int:
    return max(1_024, training.batch_surfaces * 16)


def _model_forward(
    model: MorganHallmarkResidualMLP,
    table: CompactPointTable,
    banks: CompactFeatureBanks,
    indices: Tensor,
    device: torch.device,
):
    index_cpu = indices.to(dtype=torch.int64, device="cpu")
    drug_a = table.drug_a_index[index_cpu].to(dtype=torch.int64)
    drug_b = table.drug_b_index[index_cpu].to(dtype=torch.int64)
    cells = table.cell_index[index_cpu].to(dtype=torch.int64)
    return model(
        banks.fingerprints[drug_a].to(device=device, dtype=torch.float32),
        banks.fingerprints[drug_b].to(device=device, dtype=torch.float32),
        banks.hallmarks[cells].to(device=device, dtype=torch.float32),
        table.doses[index_cpu].to(device),
        table.mono_a[index_cpu].to(device),
        table.mono_b[index_cpu].to(device),
        torch.ones(len(index_cpu), dtype=torch.bool, device=device),
        torch.ones(len(index_cpu), dtype=torch.bool, device=device),
    )


def predict_compact(
    model: MorganHallmarkResidualMLP,
    table: CompactPointTable,
    banks: CompactFeatureBanks,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    prediction_chunks: list[np.ndarray] = []
    reference_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for indices in _batch_indices(len(table), batch_size, shuffle_seed=None):
            output = _model_forward(model, table, banks, indices, device)
            prediction_chunks.append(output.prediction.detach().cpu().numpy())
            reference_chunks.append(output.reference.detach().cpu().numpy())
    return np.concatenate(prediction_chunks), np.concatenate(reference_chunks)


def _prediction_ledger(
    table: CompactPointTable,
    prediction: np.ndarray,
    reference: np.ndarray,
    *,
    reference_mode: str,
) -> pd.DataFrame:
    target = table.target.numpy().astype(np.float64)
    hsa = torch.minimum(table.mono_a, table.mono_b).numpy().astype(np.float64)
    ledger = pd.DataFrame(
        {
            "fold": np.full(len(table), table.fold, dtype=np.int16),
            "reference_model": f"{MODEL_SLUG}:{reference_mode}",
            "anchor_reference_model": reference_mode,
            "source_combodrugseq": table.source_rows,
            "prediction_group_id": table.group_ids,
            "surface_replicate_id": table.surface_ids,
            "cluster_id": table.cluster_ids,
            "dose_a_model": table.doses[:, 0].numpy(),
            "dose_b_model": table.doses[:, 1].numpy(),
            "target": target,
            "response_clipped_high": table.response_clipped_high,
            "prediction": np.asarray(prediction, dtype=np.float64),
            "reference": np.asarray(reference, dtype=np.float64),
            "hsa_prediction": hsa,
            "residual": np.asarray(prediction, dtype=np.float64) - target,
        }
    )
    if ledger["source_combodrugseq"].duplicated().any():
        raise LearnedBenchmarkError("Prediction ledger has duplicate source rows")
    if not np.isfinite(
        ledger.loc[:, ["target", "prediction", "reference", "hsa_prediction", "residual"]]
        .to_numpy(float)
    ).all():
        raise LearnedBenchmarkError("Prediction ledger contains non-finite values")
    return ledger


def summarize_learned_ledger(ledger: pd.DataFrame) -> dict[str, Any]:
    """Model metrics plus a descriptive HSA comparison on identical test rows."""

    model_metrics = summarize_reference_ledger(ledger)
    hsa_ledger = ledger.copy()
    hsa_ledger["reference_model"] = "observed_hsa"
    hsa_ledger["prediction"] = hsa_ledger["hsa_prediction"]
    hsa_ledger["residual"] = hsa_ledger["prediction"] - hsa_ledger["target"]
    hsa_metrics = summarize_reference_ledger(hsa_ledger)
    model_rmse = float(model_metrics["surface_macro"]["rmse"])
    hsa_rmse = float(hsa_metrics["surface_macro"]["rmse"])
    strata: dict[str, Any] = {}
    for label, expected in (("clipped_high", True), ("not_clipped_high", False)):
        stratum = ledger.loc[ledger["response_clipped_high"].eq(expected)].copy()
        if stratum.empty:
            strata[label] = {
                "point_count": 0,
                "surface_count": 0,
                "rmse": None,
            }
        else:
            error = stratum["prediction"].to_numpy(float) - stratum["target"].to_numpy(float)
            strata[label] = {
                "point_count": len(stratum),
                "surface_count": int(stratum["surface_replicate_id"].nunique()),
                "rmse": float(np.sqrt(np.mean(np.square(error)))),
            }
    return {
        "model": model_metrics,
        "observed_hsa_reference": hsa_metrics,
        "hsa_comparison": {
            "metric": "surface_macro.rmse",
            "model_minus_hsa": model_rmse - hsa_rmse,
            "lower_is_better": True,
            "paired_significance_test": "not_run",
            "interpretation": "descriptive_direction_only_no_significance_claim",
        },
        "posthoc_response_clipped_high_strata": {
            "feature_use": "excluded_from_model_features",
            **strata,
        },
    }


def fit_fold_seed(
    fold: int,
    banks: CompactFeatureBanks,
    train: CompactPointTable,
    validation: CompactPointTable,
    training: LearnedMorganConfig,
    *,
    seed: int,
    device: str | torch.device | None = None,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    """Fit and select a checkpoint without accepting or reading test data."""

    _seed_everything(seed)
    resolved_device = _resolve_device(device)
    model = MorganHallmarkResidualMLP(
        fingerprint_dim=banks.fingerprint_dim,
        hallmark_dim=HALLMARK_DIMENSION,
        hidden_dim=training.hidden_dim,
        dropout=training.dropout,
        reference_mode=training.reference_mode,
        monotherapy_upper_bound=train.monotherapy_upper_bound,
    ).to(resolved_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )
    history: list[dict[str, Any]] = []
    best_state: dict[str, Tensor] | None = None
    best_epoch = -1
    best_validation = float("inf")
    anchor_validation: float | None = None
    if training.include_anchor_candidate:
        initial_prediction, initial_reference = predict_compact(
            model, validation, banks, batch_size=_evaluation_batch_size(training),
            device=resolved_device,
        )
        if not np.allclose(initial_prediction, initial_reference, rtol=2e-6, atol=2e-7):
            raise LearnedBenchmarkError("Zero-residual Morgan initialization is not the anchor")
        anchor_validation = surface_macro_rmse(
            validation.target.numpy(), initial_reference, validation.surface_ids
        )
        best_validation = anchor_validation
        best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        history.append({
            "epoch": -1, "checkpoint_kind": "anchor_no_learning", "train_mse": None,
            "validation_surface_macro_rmse": anchor_validation, "is_best": True,
        })
    stale_epochs = 0
    for epoch in range(training.max_epochs):
        model.train()
        squared_error_sum = 0.0
        point_count = 0
        for indices in _surface_batch_indices(
            train,
            training.batch_surfaces,
            shuffle_seed=seed * 1_000_003 + epoch,
        ):
            optimizer.zero_grad(set_to_none=True)
            output = _model_forward(model, train, banks, indices, resolved_device)
            target = train.target[indices.to(dtype=torch.int64)].to(resolved_device)
            loss = _surface_mean_mse(
                output.prediction,
                target,
                train.surface_ids[indices.to(dtype=torch.int64).numpy()],
            )
            if not bool(torch.isfinite(loss)):
                raise LearnedBenchmarkError("Training loss became non-finite")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), training.gradient_clip
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise LearnedBenchmarkError("Training gradient norm became non-finite")
            optimizer.step()
            batch_surface_count = len(
                set(train.surface_ids[indices.to(dtype=torch.int64).numpy()].tolist())
            )
            squared_error_sum += float(loss.detach().cpu()) * batch_surface_count
            point_count += batch_surface_count
        validation_prediction, _ = predict_compact(
            model,
            validation,
            banks,
            batch_size=_evaluation_batch_size(training),
            device=resolved_device,
        )
        validation_rmse = surface_macro_rmse(
            validation.target.numpy(),
            validation_prediction,
            validation.surface_ids,
        )
        improved = validation_rmse < best_validation - training.min_delta
        if improved:
            best_validation = validation_rmse
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        history.append(
            {
                "epoch": epoch,
                "train_mse": squared_error_sum / point_count,
                "validation_surface_macro_rmse": validation_rmse,
                "is_best": improved,
            }
        )
        if epoch + 1 >= training.min_epochs and stale_epochs >= training.patience:
            break
    if best_state is None or (best_epoch < 0 and not training.include_anchor_candidate):
        raise LearnedBenchmarkError("No finite validation checkpoint was selected")
    result = {
        "model": MODEL_SLUG,
        "anchor_reference_model": training.reference_mode,
        "fold": fold,
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_ran": sum(entry["epoch"] >= 0 for entry in history),
        "selection_version": "morgan_anchor_candidate_v4" if training.include_anchor_candidate else "morgan_legacy_v3",
        "selected_checkpoint_kind": "anchor_no_learning" if best_epoch == -1 else "learned",
        "anchor_validation_surface_macro_rmse": anchor_validation,
        "best_validation_surface_macro_rmse": best_validation,
        "selection_metric": "validation.surface_macro.rmse",
        "train_point_count": len(train),
        "validation_point_count": len(validation),
        "train_surface_count": len(set(train.surface_ids)),
        "validation_surface_count": len(set(validation.surface_ids)),
        "train_table_hash": _point_table_hash(train),
        "validation_table_hash": _point_table_hash(validation),
        "test_response_rows_read_during_selection": 0,
        "test_evaluations_during_selection": 0,
        "history": history,
    }
    return result, copy.deepcopy(best_state)


def evaluate_best_state(
    fold: int,
    banks: CompactFeatureBanks,
    test: CompactPointTable,
    training: LearnedMorganConfig,
    best_state: Mapping[str, Tensor],
    *,
    seed: int,
    device: str | torch.device | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Restore one validation-selected state and then access the test table once."""

    resolved_device = _resolve_device(device)
    model = MorganHallmarkResidualMLP(
        fingerprint_dim=banks.fingerprint_dim,
        hallmark_dim=HALLMARK_DIMENSION,
        hidden_dim=training.hidden_dim,
        dropout=training.dropout,
        reference_mode=training.reference_mode,
        monotherapy_upper_bound=test.monotherapy_upper_bound,
    ).to(resolved_device)
    model.load_state_dict(dict(best_state), strict=True)
    test_prediction, test_reference = predict_compact(
        model,
        test,
        banks,
        batch_size=_evaluation_batch_size(training),
        device=resolved_device,
    )
    ledger = _prediction_ledger(
        test,
        test_prediction,
        test_reference,
        reference_mode=training.reference_mode,
    )
    metrics = summarize_learned_ledger(ledger)
    return {
        "model": MODEL_SLUG,
        "anchor_reference_model": training.reference_mode,
        "fold": fold,
        "seed": seed,
        "test_response_rows_read_after_best_restore": len(test),
        "test_evaluations_after_best_restore": 1,
        "test_metrics": metrics,
    }, ledger


def _fold_selection(folds: Sequence[int] | None, fold_count: int) -> tuple[int, ...]:
    selected = tuple(range(fold_count)) if folds is None else tuple(int(fold) for fold in folds)
    if not selected or len(set(selected)) != len(selected):
        raise LearnedBenchmarkError("Fold selection must be non-empty and unique")
    if any(fold < 0 or fold >= fold_count for fold in selected):
        raise LearnedBenchmarkError(f"Fold selection must lie in [0,{fold_count})")
    return selected


def _resolve_learned_inputs(
    config: BenchmarkConfig | str | Path,
    *,
    cell_config: CellMinerConfig | str | Path,
    learned_config: LearnedMorganConfig | None = None,
) -> tuple[BenchmarkConfig, CellMinerConfig, LearnedMorganConfig]:
    if isinstance(config, BenchmarkConfig):
        benchmark_config = config
        if learned_config is None:
            raise LearnedBenchmarkError(
                "learned_config is required when no benchmark YAML path is supplied"
            )
    else:
        benchmark_config = load_benchmark_config(config)
        learned_config = learned_config or load_learned_morgan_config(config)
    if isinstance(cell_config, CellMinerConfig):
        resolved_cell_config = cell_config
    else:
        resolved_cell_config = load_cellminer_config(cell_config)
    return benchmark_config, resolved_cell_config, learned_config


def _learned_run_directory(
    benchmark_config: BenchmarkConfig,
    training: LearnedMorganConfig,
    output_dir: str | Path | None,
) -> Path:
    if output_dir is not None:
        return Path(output_dir).resolve()
    return (
        benchmark_config.runs_dir
        / f"{MODEL_SLUG}_{training.reference_mode}"
    ).resolve()


def _validate_shard_binding(
    shards: FrozenLearnedPointShards,
    split: PreparedChemicalSplit,
) -> None:
    expected = {
        "split_manifest_hash": split.manifest["manifest_sha256"],
        "split_hash": split.split_hash,
        "source_points": split.manifest["sources"]["points"],
    }
    mismatched = [
        key for key, value in expected.items() if shards.manifest.get(key) != value
    ]
    if mismatched:
        raise LearnedBenchmarkError(
            f"Point shards are bound to another benchmark: {mismatched}"
        )


def _selection_job_expected(
    *,
    fold: int,
    seed: int,
    split: PreparedChemicalSplit,
    context: PreparedBenchmarkCellContext,
    banks: CompactFeatureBanks,
    training: LearnedMorganConfig,
    shards: FrozenLearnedPointShards,
) -> dict[str, Any]:
    return {
        "benchmark_version": LEARNED_BENCHMARK_VERSION,
        "model": MODEL_SLUG,
        "fold": fold,
        "seed": seed,
        "split_hash": split.split_hash,
        "context_split_hash": context.folds[fold].context_split_hash,
        "context_state_hash": context.folds[fold].state.state_hash,
        "feature_bank_hash": _feature_bank_hash(banks),
        "training_config": training.to_dict(),
        "runtime_environment": _runtime_environment(),
        "train_shard": shards.role_record(fold, "train"),
        "validation_shard": shards.role_record(fold, "validation"),
        "source_partition": shards.source_partition_record(fold),
    }


def select_learned_morgan_benchmark(
    config: BenchmarkConfig | str | Path,
    *,
    cell_config: CellMinerConfig | str | Path,
    learned_config: LearnedMorganConfig | None = None,
    folds: Sequence[int] | None = None,
    max_train_points: int | None = None,
    output_dir: str | Path | None = None,
    device: str | torch.device | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Phase A: fit every requested fold/seed without opening any test shard."""

    benchmark_config, resolved_cell_config, training = _resolve_learned_inputs(
        config,
        cell_config=cell_config,
        learned_config=learned_config,
    )
    selected_folds = _fold_selection(folds, benchmark_config.folds)
    destination = _learned_run_directory(benchmark_config, training, output_dir)
    protected_evaluation = [
        path
        for path in (
            destination / TEST_ACCESS_SEAL,
            destination / EVALUATION_RUN_MANIFEST,
        )
        if path.exists()
    ]
    if protected_evaluation:
        raise LearnedBenchmarkError(
            "Selection artifacts are protected after outer-test access: "
            f"{protected_evaluation[0]}"
        )
    selection_run_path = destination / SELECTION_RUN_MANIFEST
    if selection_run_path.exists() and not overwrite:
        raise LearnedBenchmarkError(
            f"Selection run already exists; use overwrite explicitly: {selection_run_path}"
        )
    split = load_one_new_scaffold_benchmark(benchmark_config)
    context = load_benchmark_cell_context(benchmark_config, resolved_cell_config)
    if context.split_hash != split.split_hash:
        raise LearnedBenchmarkError("Cell context is bound to another chemical split")
    shards = load_learned_point_shards(
        benchmark_config,
        verify_roles=("train", "validation"),
    )
    _validate_shard_binding(shards, split)

    jobs: list[dict[str, Any]] = []
    for fold in selected_folds:
        fold_assignments = _fold_surface_roles(split.surfaces, fold)
        required_drugs = sorted(
            set(fold_assignments["drug_a_id"]) | set(fold_assignments["drug_b_id"])
        )
        required_cells = sorted(set(fold_assignments["cell_id"]))
        banks = build_feature_banks(
            required_drugs,
            required_cells,
            context,
            fold,
            benchmark_config,
        )
        train_frame = load_learned_role_frame(
            shards,
            split.surfaces,
            fold,
            "train",
            max_train_points=max_train_points,
            sample_seed=benchmark_config.split_seed,
        )
        validation_frame = load_learned_role_frame(
            shards,
            split.surfaces,
            fold,
            "validation",
            sample_seed=benchmark_config.split_seed,
        )
        _assert_source_roles_disjoint(
            {"train": train_frame, "validation": validation_frame},
            fold,
        )
        train_table = build_compact_point_table(
            train_frame,
            banks,
            role="train",
            fold=fold,
            monotherapy_upper_bound=benchmark_config.response_upper_bound,
        )
        validation_table = build_compact_point_table(
            validation_frame,
            banks,
            role="validation",
            fold=fold,
            monotherapy_upper_bound=benchmark_config.response_upper_bound,
        )
        bank_hash = _feature_bank_hash(banks)
        for seed in training.seeds:
            selection, best_state = fit_fold_seed(
                fold,
                banks,
                train_table,
                validation_table,
                training,
                seed=seed,
                device=device,
            )
            job_dir = destination / f"fold_{fold}" / f"seed_{seed}"
            checkpoint_path, selection_path, selection_manifest = (
                _freeze_selection_checkpoint(
                    job_dir,
                    fold=fold,
                    seed=seed,
                    selection=selection,
                    state=best_state,
                    training=training,
                    split_hash=split.split_hash,
                    context_split_hash=context.folds[fold].context_split_hash,
                    context_state_hash=context.folds[fold].state.state_hash,
                    feature_bank_hash=bank_hash,
                    train_shard=shards.role_record(fold, "train"),
                    validation_shard=shards.role_record(fold, "validation"),
                    source_partition=shards.source_partition_record(fold),
                )
            )
            verified, _ = _load_frozen_selection_checkpoint(
                checkpoint_path,
                selection_path,
                expected=_selection_job_expected(
                    fold=fold,
                    seed=seed,
                    split=split,
                    context=context,
                    banks=banks,
                    training=training,
                    shards=shards,
                ),
            )
            if verified["manifest_sha256"] != selection_manifest["manifest_sha256"]:
                raise LearnedBenchmarkError("Selection changed immediately after freezing")
            jobs.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "best_epoch": selection["best_epoch"],
                    "best_validation_surface_macro_rmse": selection[
                        "best_validation_surface_macro_rmse"
                    ],
                    "context_split_hash": context.folds[fold].context_split_hash,
                    "state_hash": context.folds[fold].state.state_hash,
                    "point_counts": {
                        "train": len(train_table),
                        "validation": len(validation_table),
                    },
                    "surface_counts": {
                        "train": len(set(train_table.surface_ids)),
                        "validation": len(set(validation_table.surface_ids)),
                    },
                    "feature_bank_shapes": {
                        "fingerprints": list(banks.fingerprints.shape),
                        "hallmarks": list(banks.hallmarks.shape),
                    },
                    "feature_bank_hash": bank_hash,
                    "source_partition": shards.source_partition_record(fold),
                    "checkpoint": _file_record(checkpoint_path),
                    "selection_manifest": _file_record(selection_path),
                }
            )
    manifest: dict[str, Any] = {
        "status": "selection_frozen_no_test_shard_opened",
        "claim_scope": "engineering_only_no_scientific_causal_or_sota_claim",
        "benchmark_version": LEARNED_BENCHMARK_VERSION,
        "model": MODEL_SLUG,
        "protocol": benchmark_config.protocol,
        "information_setting": benchmark_config.information_setting,
        "endpoint": benchmark_config.endpoint,
        "endpoint_decision_status": benchmark_config.endpoint_decision_status,
        "phase": "selection",
        "run_mode": (
            "engineering_smoke"
            if max_train_points is not None or len(selected_folds) < benchmark_config.folds
            else "full_five_fold_engineering"
        ),
        "selected_folds": list(selected_folds),
        "max_train_points": max_train_points,
        "training_config": training.to_dict(),
        "fingerprint": {
            "scope": "whole_canonical_molecule",
            "kind": "Morgan",
            "radius": benchmark_config.fingerprint_radius,
            "bits": benchmark_config.fingerprint_bits,
            "rdkit_version": rdBase.rdkitVersion,
        },
        "hallmark_features": {
            "source": "fold_train_fit_graph.node_features[:,0:2]",
            "channels": ["activity_z", "observed_fraction"],
            "layout": "fixed_pathway_order_then_channel_order_flattened",
            "excluded_from_model_features": ["response_clipped_high"],
        },
        "hallmark_dimension": HALLMARK_DIMENSION,
        "selection_contract": {
            "initial_candidate": (
                "epoch_-1_exact_observed_anchor" if training.include_anchor_candidate
                else "epoch_0_after_one_training_epoch"
            ),
            "batch_unit": "complete_surfaces",
            "loss": "mean_of_within_surface_point_mse",
            "optimizer": "AdamW",
            "learning_rate_schedule": "fixed",
            "early_stopping": {
                "metric": "validation_surface_macro_rmse",
                "min_epochs": training.min_epochs,
                "patience": training.patience,
                "min_delta": training.min_delta,
            },
            "test_access": "physical_test_shards_not_opened_in_selection_phase",
            "source_row_isolation": (
                "runtime_label_free_three_role_partition_verification_plus_"
                "train_validation_response_shard_disjoint_check"
            ),
            "monotherapy_auxiliary_weight": 0.0,
            "uncertainty_nll_weight": 0.0,
            "gradient_clip_norm": training.gradient_clip,
        },
        "split_manifest_hash": split.manifest["manifest_sha256"],
        "split_hash": split.split_hash,
        "context_manifest_hash": context.manifest["manifest_sha256"],
        "point_shard_manifest_hash": shards.manifest["manifest_sha256"],
        "source_partitions": [
            shards.source_partition_record(fold) for fold in selected_folds
        ],
        "authorized_response_roles": ["train", "validation"],
        "jobs": jobs,
    }
    manifest["configuration_hash"] = _canonical_hash(
        {
            "benchmark": benchmark_config.to_dict(),
            "learned": training.to_dict(),
            "folds": list(selected_folds),
            "max_train_points": max_train_points,
            "split_hash": split.split_hash,
            "context_manifest_hash": context.manifest["manifest_sha256"],
            "point_shard_manifest_hash": shards.manifest["manifest_sha256"],
        }
    )
    manifest["manifest_sha256"] = _canonical_hash(manifest)
    _write_json_atomic(selection_run_path, manifest)
    return {**manifest, "selection_run_manifest": _file_record(selection_run_path)}


def evaluate_learned_morgan_benchmark(
    config: BenchmarkConfig | str | Path,
    *,
    cell_config: CellMinerConfig | str | Path,
    learned_config: LearnedMorganConfig | None = None,
    output_dir: str | Path | None = None,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Phase B: verify the global selection barrier, then open test shards."""

    benchmark_config, resolved_cell_config, training = _resolve_learned_inputs(
        config,
        cell_config=cell_config,
        learned_config=learned_config,
    )
    destination = _learned_run_directory(benchmark_config, training, output_dir)
    access_seal_path = destination / TEST_ACCESS_SEAL
    manifest_path = destination / EVALUATION_RUN_MANIFEST
    if access_seal_path.exists():
        raise LearnedBenchmarkError(
            "Outer-test access was already committed; use a new independent "
            f"cohort/output directory instead of re-evaluating: {access_seal_path}"
        )
    if manifest_path.exists():
        raise LearnedBenchmarkError(
            f"Evaluation run already exists and cannot be overwritten: {manifest_path}"
        )
    selection_run_path = destination / SELECTION_RUN_MANIFEST
    selection_run = _load_json_object(selection_run_path, "Selection-run manifest")
    _verify_self_hash(selection_run, "Selection-run manifest")
    split = load_one_new_scaffold_benchmark(benchmark_config)
    context = load_benchmark_cell_context(benchmark_config, resolved_cell_config)
    shards = load_learned_point_shards(benchmark_config, verify_roles=())
    _validate_shard_binding(shards, split)
    selected_folds = _fold_selection(
        selection_run.get("selected_folds"), benchmark_config.folds
    )
    expected_run = {
        "status": "selection_frozen_no_test_shard_opened",
        "benchmark_version": LEARNED_BENCHMARK_VERSION,
        "model": MODEL_SLUG,
        "protocol": benchmark_config.protocol,
        "information_setting": benchmark_config.information_setting,
        "endpoint": benchmark_config.endpoint,
        "endpoint_decision_status": benchmark_config.endpoint_decision_status,
        "training_config": training.to_dict(),
        "split_manifest_hash": split.manifest["manifest_sha256"],
        "split_hash": split.split_hash,
        "context_manifest_hash": context.manifest["manifest_sha256"],
        "point_shard_manifest_hash": shards.manifest["manifest_sha256"],
        "source_partitions": [
            shards.source_partition_record(fold)
            for fold in selected_folds
        ],
    }
    mismatched_run = [
        key for key, value in expected_run.items() if selection_run.get(key) != value
    ]
    if mismatched_run:
        raise LearnedBenchmarkError(
            f"Selection run differs from current evaluation: {mismatched_run}"
        )
    if selected_folds != tuple(range(benchmark_config.folds)):
        raise LearnedBenchmarkError(
            "Outer-test evaluation requires all five folds to cross the global "
            "selection barrier"
        )
    if selection_run.get("max_train_points") is not None:
        raise LearnedBenchmarkError(
            "Outer-test evaluation is forbidden for a subsampled engineering selection"
        )
    raw_jobs = selection_run.get("jobs")
    if not isinstance(raw_jobs, list):
        raise LearnedBenchmarkError("Selection-run job inventory is invalid")
    expected_job_ids = {
        (fold, seed) for fold in selected_folds for seed in training.seeds
    }
    jobs_by_id: dict[tuple[int, int], dict[str, Any]] = {}
    banks_by_fold: dict[int, CompactFeatureBanks] = {}
    states_by_id: dict[tuple[int, int], dict[str, Tensor]] = {}

    # Global barrier: every requested fold/seed checkpoint is verified before a
    # test shard is hashed or decoded.
    for fold in selected_folds:
        assignments = _fold_surface_roles(split.surfaces, fold)
        required_drugs = sorted(
            set(assignments["drug_a_id"]) | set(assignments["drug_b_id"])
        )
        required_cells = sorted(set(assignments["cell_id"]))
        banks_by_fold[fold] = build_feature_banks(
            required_drugs,
            required_cells,
            context,
            fold,
            benchmark_config,
        )
    for raw_job in raw_jobs:
        if not isinstance(raw_job, dict):
            raise LearnedBenchmarkError("Selection-run job must be an object")
        job_id = (int(raw_job.get("fold", -1)), int(raw_job.get("seed", -1)))
        if job_id in jobs_by_id:
            raise LearnedBenchmarkError(f"Duplicate selection job: {job_id}")
        jobs_by_id[job_id] = raw_job
    if set(jobs_by_id) != expected_job_ids:
        raise LearnedBenchmarkError("Selection-run fold/seed inventory is incomplete")
    preexisting_outputs = [
        destination / f"fold_{fold}" / f"seed_{seed}" / filename
        for fold, seed in sorted(expected_job_ids)
        for filename in ("history.json", "metrics.json", "predictions.parquet")
        if (destination / f"fold_{fold}" / f"seed_{seed}" / filename).exists()
    ]
    if preexisting_outputs:
        raise LearnedBenchmarkError(
            "Evaluation artifacts already exist and cannot be overwritten: "
            f"{preexisting_outputs[0]}"
        )
    for fold, seed in sorted(expected_job_ids):
        job = jobs_by_id[(fold, seed)]
        checkpoint_path = destination / f"fold_{fold}" / f"seed_{seed}" / "checkpoint.pt"
        selection_path = (
            destination / f"fold_{fold}" / f"seed_{seed}" / "selection_manifest.json"
        )
        for key, path in (
            ("checkpoint", checkpoint_path),
            ("selection_manifest", selection_path),
        ):
            record = job.get(key)
            if not isinstance(record, dict):
                raise LearnedBenchmarkError(f"Selection job lacks {key} record")
            _verify_recorded_file(path, record, f"Selection job {fold}/{seed} {key}")
        _, state = _load_frozen_selection_checkpoint(
            checkpoint_path,
            selection_path,
            expected=_selection_job_expected(
                fold=fold,
                seed=seed,
                split=split,
                context=context,
                banks=banks_by_fold[fold],
                training=training,
                shards=shards,
            ),
        )
        states_by_id[(fold, seed)] = state

    access_seal: dict[str, Any] = {
        "status": "outer_test_access_committed_before_test_file_open",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_version": LEARNED_BENCHMARK_VERSION,
        "model": MODEL_SLUG,
        "protocol": benchmark_config.protocol,
        "endpoint": benchmark_config.endpoint,
        "endpoint_decision_status": benchmark_config.endpoint_decision_status,
        "selection_run_manifest": _file_record(selection_run_path),
        "split_hash": split.split_hash,
        "context_manifest_hash": context.manifest["manifest_sha256"],
        "point_shard_manifest_hash": shards.manifest["manifest_sha256"],
        "source_partitions": [
            shards.source_partition_record(fold) for fold in selected_folds
        ],
        "runtime_environment": _runtime_environment(),
        "test_response_rows_read_before_seal": 0,
        "pre_seal_source_access": (
            "mixed_source_provenance_byte_hash_allowed_no_test_shard_or_"
            "decoded_test_response_access"
        ),
        "rerun_contract": (
            "existing_seal_or_evaluation_artifact_refuses_all_repeat_evaluation"
        ),
    }
    access_seal["seal_content_sha256"] = _canonical_hash(access_seal)
    _write_json_exclusive(access_seal_path, access_seal)

    test_shards = load_learned_point_shards(
        benchmark_config,
        verify_roles=("test",),
    )
    if test_shards.manifest["manifest_sha256"] != shards.manifest["manifest_sha256"]:
        raise LearnedBenchmarkError("Point-shard manifest changed across phase barrier")

    run_entries: list[dict[str, Any]] = []
    surface_rmses: list[float] = []
    for fold in selected_folds:
        test_frame = load_learned_role_frame(
            test_shards,
            split.surfaces,
            fold,
            "test",
            sample_seed=benchmark_config.split_seed,
        )
        banks = banks_by_fold[fold]
        test_table = build_compact_point_table(
            test_frame,
            banks,
            role="test",
            fold=fold,
            monotherapy_upper_bound=benchmark_config.response_upper_bound,
        )
        for seed in training.seeds:
            evaluation, ledger = evaluate_best_state(
                fold,
                banks,
                test_table,
                training,
                states_by_id[(fold, seed)],
                seed=seed,
                device=device,
            )
            job = jobs_by_id[(fold, seed)]
            selection_path = (
                destination
                / f"fold_{fold}"
                / f"seed_{seed}"
                / "selection_manifest.json"
            )
            frozen_selection = _load_json_object(
                selection_path, "Selection manifest"
            )["selection"]
            job_dir = destination / f"fold_{fold}" / f"seed_{seed}"
            history_path = job_dir / "history.json"
            metrics_path = job_dir / "metrics.json"
            predictions_path = job_dir / "predictions.parquet"
            _write_json_atomic(
                history_path, {"history": frozen_selection["history"]}
            )
            result = {
                key: value for key, value in frozen_selection.items() if key != "history"
            }
            result.update(evaluation)
            _write_json_atomic(metrics_path, result)
            _write_parquet_atomic(predictions_path, ledger)
            surface_rmse = float(
                result["test_metrics"]["model"]["surface_macro"]["rmse"]
            )
            surface_rmses.append(surface_rmse)
            run_entries.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "best_epoch": result["best_epoch"],
                    "test_surface_macro_rmse": surface_rmse,
                    "point_count": len(test_table),
                    "surface_count": len(set(test_table.surface_ids)),
                    "test_shard": test_shards.role_record(fold, "test"),
                    "artifacts": {
                        "history": _file_record(history_path),
                        "metrics": _file_record(metrics_path),
                        "predictions": _file_record(predictions_path),
                        "checkpoint": job["checkpoint"],
                        "selection_manifest": job["selection_manifest"],
                    },
                }
            )
    aggregate = {
        "metric": "test.surface_macro.rmse",
        "job_count": len(surface_rmses),
        "mean": float(np.mean(surface_rmses)),
        "sample_sd": (
            float(np.std(surface_rmses, ddof=1)) if len(surface_rmses) > 1 else None
        ),
        "aggregation_note": "descriptive_across_selected_seed_fold_jobs",
    }
    manifest: dict[str, Any] = {
        "status": "completed_engineering_evaluation",
        "claim_scope": "engineering_only_no_scientific_causal_or_sota_claim",
        "benchmark_version": LEARNED_BENCHMARK_VERSION,
        "model": MODEL_SLUG,
        "anchor_reference_model": training.reference_mode,
        "protocol": benchmark_config.protocol,
        "information_setting": benchmark_config.information_setting,
        "endpoint": benchmark_config.endpoint,
        "endpoint_decision_status": benchmark_config.endpoint_decision_status,
        "phase": "evaluation_after_global_selection_barrier",
        "selected_folds": list(selected_folds),
        "training_config": training.to_dict(),
        "selection_run_manifest": _file_record(selection_run_path),
        "test_access_seal": _file_record(access_seal_path),
        "point_shard_manifest_hash": test_shards.manifest["manifest_sha256"],
        "source_partitions": [
            test_shards.source_partition_record(fold) for fold in selected_folds
        ],
        "authorized_response_roles": ["test"],
        "aggregate": aggregate,
        "jobs": run_entries,
    }
    manifest["manifest_sha256"] = _canonical_hash(manifest)
    _write_json_atomic(manifest_path, manifest)
    return {**manifest, "run_manifest": _file_record(manifest_path)}


def run_learned_morgan_benchmark(
    config: BenchmarkConfig | str | Path,
    *,
    cell_config: CellMinerConfig | str | Path,
    learned_config: LearnedMorganConfig | None = None,
    folds: Sequence[int] | None = None,
    max_train_points: int | None = None,
    output_dir: str | Path | None = None,
    device: str | torch.device | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run the physically separated selection and evaluation phases."""

    selection = select_learned_morgan_benchmark(
        config,
        cell_config=cell_config,
        learned_config=learned_config,
        folds=folds,
        max_train_points=max_train_points,
        output_dir=output_dir,
        device=device,
        overwrite=overwrite,
    )
    selected_folds = _fold_selection(folds, 5)
    if max_train_points is not None or selected_folds != tuple(range(5)):
        return selection
    return evaluate_learned_morgan_benchmark(
        config,
        cell_config=cell_config,
        learned_config=learned_config,
        output_dir=output_dir,
        device=device,
    )


__all__ = [
    "LEARNED_BENCHMARK_VERSION",
    "LEARNED_POINT_SHARD_VERSION",
    "CompactFeatureBanks",
    "CompactFoldData",
    "CompactPointTable",
    "FrozenLearnedPointShards",
    "LearnedBenchmarkError",
    "LearnedMorganConfig",
    "build_compact_fold_data",
    "build_compact_point_table",
    "build_hallmark_bank",
    "build_morgan_bank",
    "evaluate_learned_morgan_benchmark",
    "load_learned_point_shards",
    "load_learned_role_frame",
    "load_learned_morgan_config",
    "predict_compact",
    "prepare_learned_point_shards",
    "run_learned_morgan_benchmark",
    "select_learned_morgan_benchmark",
    "select_fold_points_from_chunks",
    "summarize_learned_ledger",
    "surface_macro_rmse",
]
