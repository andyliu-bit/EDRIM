from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Sequence

import torch

from .batch import SurfaceSample
from .cellminer import normalize_cell_label


@dataclass(frozen=True)
class SplitIndices:
    train: list[int]
    validation: list[int]
    test: list[int]
    train_surfaces: tuple[int, ...]
    validation_surfaces: tuple[int, ...]
    test_surfaces: tuple[int, ...]

    def assert_disjoint(self) -> None:
        train = set(self.train_surfaces)
        validation = set(self.validation_surfaces)
        test = set(self.test_surfaces)
        if train & validation or train & test or validation & test:
            raise AssertionError("Surface groups overlap across splits")


@dataclass(frozen=True)
class CellOutSplit:
    """A deterministic split over unique biological cell identifiers."""

    train_cells: tuple[str, ...]
    validation_cells: tuple[str, ...]
    test_cells: tuple[str, ...]
    train_groups: tuple[str, ...]
    validation_groups: tuple[str, ...]
    test_groups: tuple[str, ...]
    protocol: str
    seed: int
    split_hash: str

    def assert_disjoint(self) -> None:
        train = set(self.train_cells)
        validation = set(self.validation_cells)
        test = set(self.test_cells)
        if train & validation or train & test or validation & test:
            raise AssertionError("Biological cells overlap across CellOut splits")
        if not train or not validation or not test:
            raise AssertionError("Every CellOut partition must contain at least one cell")
        train_groups = set(self.train_groups)
        validation_groups = set(self.validation_groups)
        test_groups = set(self.test_groups)
        if (
            train_groups & validation_groups
            or train_groups & test_groups
            or validation_groups & test_groups
        ):
            raise AssertionError("Lineage groups overlap across CellOut splits")

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol": self.protocol,
            "seed": self.seed,
            "split_hash": self.split_hash,
            "train_cells": list(self.train_cells),
            "validation_cells": list(self.validation_cells),
            "test_cells": list(self.test_cells),
            "train_groups": list(self.train_groups),
            "validation_groups": list(self.validation_groups),
            "test_groups": list(self.test_groups),
        }


def _cell_split_hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def split_biological_cells(
    cell_ids: Sequence[str],
    *,
    validation_cells: int,
    test_cells: int,
    seed: int,
) -> CellOutSplit:
    """Split unique biological cells before fitting any expression transform.

    Input order and duplicate aliases cannot affect the split. The returned hash
    covers the exact membership, counts, seed, and protocol version so fold-level
    context caches cannot be silently reused with another partition.
    """

    unique_cells = tuple(sorted({normalize_cell_label(cell_id) for cell_id in cell_ids}))
    if validation_cells < 1 or test_cells < 1:
        raise ValueError("CellOut validation and test counts must be positive")
    if validation_cells + test_cells >= len(unique_cells):
        raise ValueError("CellOut counts leave no training cells")

    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(unique_cells), generator=generator).tolist()
    shuffled = [unique_cells[index] for index in order]
    test = tuple(sorted(shuffled[:test_cells]))
    validation = tuple(sorted(shuffled[test_cells : test_cells + validation_cells]))
    train = tuple(sorted(shuffled[test_cells + validation_cells :]))
    hash_payload: dict[str, object] = {
        "protocol": "CellOut-v2",
        "seed": int(seed),
        "validation_count": validation_cells,
        "test_count": test_cells,
        "train_cells": train,
        "validation_cells": validation,
        "test_cells": test,
    }
    split = CellOutSplit(
        train_cells=train,
        validation_cells=validation,
        test_cells=test,
        train_groups=train,
        validation_groups=validation,
        test_groups=test,
        protocol="CellOut-v2",
        seed=seed,
        split_hash=_cell_split_hash(hash_payload),
    )
    split.assert_disjoint()
    return split


def split_biological_cell_groups(
    cell_ids: Sequence[str],
    *,
    group_by_cell: dict[str, str],
    validation_groups: int,
    test_groups: int,
    seed: int,
) -> CellOutSplit:
    """CellOut split that keeps known shared-lineage cells in one partition."""

    unique_cells = tuple(sorted({normalize_cell_label(cell_id) for cell_id in cell_ids}))
    normalized_groups: dict[str, str] = {}
    for raw_cell_id, raw_group_id in group_by_cell.items():
        cell_id = normalize_cell_label(raw_cell_id)
        group_id = normalize_cell_label(raw_group_id)
        previous = normalized_groups.get(cell_id)
        if previous is not None and previous != group_id:
            raise ValueError(
                f"Conflicting lineage groups after normalization for {cell_id}: "
                f"{previous} and {group_id}"
            )
        normalized_groups[cell_id] = group_id
    unknown = sorted(set(normalized_groups) - set(unique_cells))
    if unknown:
        raise ValueError(f"Lineage map contains cells outside the cohort: {unknown}")
    resolved_groups = {
        cell_id: normalized_groups.get(cell_id, cell_id) for cell_id in unique_cells
    }
    groups = tuple(sorted(set(resolved_groups.values())))
    if validation_groups < 1 or test_groups < 1:
        raise ValueError("Validation and test group counts must be positive")
    if validation_groups + test_groups >= len(groups):
        raise ValueError("Lineage-group counts leave no training groups")

    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(groups), generator=generator).tolist()
    shuffled = [groups[index] for index in order]
    test_group_ids = tuple(sorted(shuffled[:test_groups]))
    validation_group_ids = tuple(
        sorted(shuffled[test_groups : test_groups + validation_groups])
    )
    train_group_ids = tuple(sorted(shuffled[test_groups + validation_groups :]))
    train_group_set = set(train_group_ids)
    validation_group_set = set(validation_group_ids)
    test_group_set = set(test_group_ids)
    train = tuple(
        cell for cell in unique_cells if resolved_groups[cell] in train_group_set
    )
    validation = tuple(
        cell for cell in unique_cells if resolved_groups[cell] in validation_group_set
    )
    test = tuple(cell for cell in unique_cells if resolved_groups[cell] in test_group_set)
    hash_payload: dict[str, object] = {
        "protocol": "LineageGroupedCellOut-v2",
        "seed": int(seed),
        "validation_group_count": validation_groups,
        "test_group_count": test_groups,
        "group_by_cell": sorted(resolved_groups.items()),
        "train_cells": train,
        "validation_cells": validation,
        "test_cells": test,
        "train_groups": train_group_ids,
        "validation_groups": validation_group_ids,
        "test_groups": test_group_ids,
    }
    split = CellOutSplit(
        train_cells=train,
        validation_cells=validation,
        test_cells=test,
        train_groups=train_group_ids,
        validation_groups=validation_group_ids,
        test_groups=test_group_ids,
        protocol="LineageGroupedCellOut-v2",
        seed=seed,
        split_hash=_cell_split_hash(hash_payload),
    )
    split.assert_disjoint()
    return split


def split_by_surface(
    samples: Sequence[SurfaceSample],
    train_fraction: float,
    validation_fraction: float,
    seed: int,
) -> SplitIndices:
    surface_ids = sorted({sample.surface_id for sample in samples})
    if len(surface_ids) < 3:
        raise ValueError("At least three surfaces are required")

    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(surface_ids), generator=generator).tolist()
    shuffled = [surface_ids[index] for index in order]

    train_count = max(1, int(len(shuffled) * train_fraction))
    validation_count = max(1, int(len(shuffled) * validation_fraction))
    if train_count + validation_count >= len(shuffled):
        validation_count = max(1, len(shuffled) - train_count - 1)
    test_count = len(shuffled) - train_count - validation_count
    if test_count < 1:
        raise ValueError("Split fractions leave no test surfaces")

    train_surfaces = tuple(sorted(shuffled[:train_count]))
    validation_surfaces = tuple(sorted(shuffled[train_count : train_count + validation_count]))
    test_surfaces = tuple(sorted(shuffled[train_count + validation_count :]))

    train_set = set(train_surfaces)
    validation_set = set(validation_surfaces)
    test_set = set(test_surfaces)
    split = SplitIndices(
        train=[index for index, sample in enumerate(samples) if sample.surface_id in train_set],
        validation=[
            index for index, sample in enumerate(samples) if sample.surface_id in validation_set
        ],
        test=[index for index, sample in enumerate(samples) if sample.surface_id in test_set],
        train_surfaces=train_surfaces,
        validation_surfaces=validation_surfaces,
        test_surfaces=test_surfaces,
    )
    split.assert_disjoint()
    return split
