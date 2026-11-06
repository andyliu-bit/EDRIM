from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class BenchmarkConfig:
    nci_processed_dir: Path
    cell_processed_dir: Path
    output_dir: Path
    runs_dir: Path
    protocol: str = "OneNewScaffoldAnchored-v1"
    information_setting: str = "observed_monotherapy_anchored"
    endpoint: str = "clipped_PERCENTGROWTHNOTZ_over_100"
    endpoint_decision_status: str = "provisional_engineering"
    folds: int = 5
    validation_offset: int = 1
    split_seed: int = 41
    similarity_threshold: float = 0.70
    fingerprint_radius: int = 2
    fingerprint_bits: int = 2_048
    # Sanity bound on observed responses and monotherapy anchors. The clipped
    # endpoint truncates both to 1.0; the D1 unclipped endpoint keeps the raw
    # ratio, whose largest recorded value is 2.2638. The bound stays declared so
    # a unit error such as a forgotten division by 100 is still rejected.
    response_upper_bound: float = 1.0

    def __post_init__(self) -> None:
        for field_name in (
            "nci_processed_dir",
            "cell_processed_dir",
            "output_dir",
            "runs_dir",
        ):
            object.__setattr__(self, field_name, Path(getattr(self, field_name)).resolve())
        if self.protocol != "OneNewScaffoldAnchored-v1":
            raise ValueError("Only the frozen OneNewScaffoldAnchored-v1 protocol is supported")
        if self.information_setting != "observed_monotherapy_anchored":
            raise ValueError("This benchmark requires the observed-monotherapy anchored setting")
        if self.folds != 5:
            raise ValueError("The primary benchmark is frozen to five outer folds")
        if not 0 < self.validation_offset < self.folds:
            raise ValueError("validation_offset must select another outer cluster bin")
        if not 0 < self.similarity_threshold < 1:
            raise ValueError("similarity_threshold must be in (0, 1)")
        if self.fingerprint_radius < 1 or self.fingerprint_bits < 128:
            raise ValueError("Invalid Morgan fingerprint parameters")
        if not self.endpoint or not self.endpoint_decision_status:
            raise ValueError("Endpoint name and decision status must be explicit")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("nci_processed_dir", "cell_processed_dir", "output_dir", "runs_dir"):
            payload[key] = str(payload[key])
        return payload


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Benchmark configuration must be a mapping")
    configured_root = raw.get("project_root")
    if configured_root is None:
        project_root = config_path.parent.parent
    else:
        candidate = Path(str(configured_root))
        project_root = (
            candidate.resolve()
            if candidate.is_absolute()
            else (config_path.parent / candidate).resolve()
        )

    def resolve_path(key: str, default: str) -> Path:
        candidate = Path(str(raw.get(key, default)))
        return (
            candidate.resolve()
            if candidate.is_absolute()
            else (project_root / candidate).resolve()
        )

    return BenchmarkConfig(
        nci_processed_dir=resolve_path(
            "nci_processed_dir", "data/processed/nci_almanac"
        ),
        cell_processed_dir=resolve_path(
            "cell_processed_dir", "data/processed/cellminer"
        ),
        output_dir=resolve_path(
            "output_dir", "data/processed/one_new_scaffold_benchmark"
        ),
        runs_dir=resolve_path("runs_dir", "runs/one_new_scaffold"),
        protocol=str(raw.get("protocol", "OneNewScaffoldAnchored-v1")),
        information_setting=str(
            raw.get("information_setting", "observed_monotherapy_anchored")
        ),
        endpoint=str(raw.get("endpoint", "clipped_PERCENTGROWTHNOTZ_over_100")),
        endpoint_decision_status=str(
            raw.get("endpoint_decision_status", "provisional_engineering")
        ),
        folds=int(raw.get("folds", 5)),
        validation_offset=int(raw.get("validation_offset", 1)),
        split_seed=int(raw.get("split_seed", 41)),
        response_upper_bound=float(raw.get("response_upper_bound", 1.0)),
        similarity_threshold=float(raw.get("similarity_threshold", 0.70)),
        fingerprint_radius=int(raw.get("fingerprint_radius", 2)),
        fingerprint_bits=int(raw.get("fingerprint_bits", 2_048)),
    )


__all__ = ["BenchmarkConfig", "load_benchmark_config"]
