from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml


@dataclass(frozen=True)
class DataConfig:
    kind: str = "synthetic"
    num_drugs: int = 16
    num_cells: int = 6
    num_surfaces: int = 48
    dose_levels: tuple[float, ...] = (0.0, 0.35, 0.8, 1.4)
    atom_feature_dim: int = 12
    cell_feature_dim: int = 6
    min_atoms: int = 5
    max_atoms: int = 11
    pathway_nodes: int = 8
    noise_std: float = 0.01


@dataclass(frozen=True)
class SplitConfig:
    train_fraction: float = 0.70
    validation_fraction: float = 0.15


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int = 32
    graph_layers: int = 2
    attention_heads: int = 4
    dropout: float = 0.0
    dose_frequencies: int = 4


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    monotherapy_weight: float = 0.25
    variance_weight: float = 0.02
    num_workers: int = 0
    grad_clip_norm: float = 2.0


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    device: str
    output_dir: str
    data: DataConfig
    split: SplitConfig
    model: ModelConfig
    training: TrainingConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _tuple_doses(raw: Any) -> tuple[float, ...]:
    if not isinstance(raw, (list, tuple)):
        raise TypeError("data.dose_levels must be a list of non-negative numbers")
    return tuple(float(value) for value in raw)


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")

    data_raw = dict(raw.get("data", {}))
    if "dose_levels" in data_raw:
        data_raw["dose_levels"] = _tuple_doses(data_raw["dose_levels"])

    config = ExperimentConfig(
        seed=int(raw.get("seed", 17)),
        device=str(raw.get("device", "auto")),
        output_dir=str(raw.get("output_dir", "runs/default")),
        data=DataConfig(**data_raw),
        split=SplitConfig(**dict(raw.get("split", {}))),
        model=ModelConfig(**dict(raw.get("model", {}))),
        training=TrainingConfig(**dict(raw.get("training", {}))),
    )
    validate_config(config)
    return config


def validate_config(config: ExperimentConfig) -> None:
    if config.data.kind != "synthetic":
        raise ValueError("v0.1 currently supports data.kind='synthetic' only")
    if not config.data.dose_levels or min(config.data.dose_levels) < 0:
        raise ValueError("Dose levels must be non-empty and non-negative")
    if 0.0 not in config.data.dose_levels:
        raise ValueError("At least one zero-dose level is required for boundary auditing")
    if config.data.min_atoms < 1 or config.data.max_atoms < config.data.min_atoms:
        raise ValueError("Invalid atom-count range")
    if config.data.num_surfaces < 6:
        raise ValueError("At least six surfaces are required for grouped train/val/test splits")
    if config.model.hidden_dim % config.model.attention_heads != 0:
        raise ValueError("model.hidden_dim must be divisible by model.attention_heads")
    if not 0 < config.split.train_fraction < 1:
        raise ValueError("split.train_fraction must be in (0, 1)")
    if not 0 < config.split.validation_fraction < 1:
        raise ValueError("split.validation_fraction must be in (0, 1)")
    if config.split.train_fraction + config.split.validation_fraction >= 1:
        raise ValueError("Train and validation fractions must leave a non-empty test split")
    if config.training.epochs < 1 or config.training.batch_size < 1:
        raise ValueError("Training epochs and batch size must be positive")


def resolve_device(requested: str) -> torch.device:
    normalized = requested.lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(normalized)

