from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from ..benchmark_config import BenchmarkConfig
from .batch import GraphItem
from .cellminer import (
    CellMinerConfig,
    ExpressionMatrix,
    GeneSetCatalog,
    PathwayTopology,
    PathwayTransformState,
    cache_split_cell_graphs,
    fit_train_only_pathway_transform,
    load_cached_pathway_topology,
    load_cached_split_cell_graphs,
    load_hallmark_gene_sets,
    load_prepared_cell_mapping,
    load_prepared_expression,
)
from .chemical_splits import PreparedChemicalSplit, load_one_new_scaffold_benchmark


BENCHMARK_CONTEXT_VERSION = "one-new-scaffold-cell-context-v1"
CONTEXT_MANIFEST_FILENAME = "benchmark_context_manifest.json"


class BenchmarkContextError(ValueError):
    """Raised when a benchmark-specific biological context is not reproducible."""


@dataclass(frozen=True)
class BenchmarkFoldCellContext:
    fold: int
    context_split_hash: str
    fit_cell_ids: tuple[str, ...]
    state: PathwayTransformState
    graphs: dict[str, GraphItem]
    graph_manifest: dict[str, Any]
    cache_directory: Path


@dataclass(frozen=True)
class PreparedBenchmarkCellContext:
    manifest: dict[str, Any]
    benchmark: PreparedChemicalSplit
    expression: ExpressionMatrix
    catalog: GeneSetCatalog
    topology: PathwayTopology
    mapping: pd.DataFrame
    folds: tuple[BenchmarkFoldCellContext, ...]

    @property
    def split_hash(self) -> str:
        return str(self.manifest["benchmark_split_hash"])


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise BenchmarkContextError(f"Required context artifact is missing: {resolved}")
    if relative_to is None:
        recorded_path = str(resolved)
    else:
        try:
            recorded_path = str(resolved.relative_to(relative_to.resolve()))
        except ValueError as error:
            raise BenchmarkContextError(
                f"Context artifact is outside its expected root: {resolved}"
            ) from error
    return {
        "path": recorded_path,
        "bytes": int(resolved.stat().st_size),
        "sha256": _sha256_file(resolved),
    }


def _verify_file_record(
    path: Path,
    record: Mapping[str, Any],
    *,
    expected_recorded_path: str,
) -> None:
    if record.get("path") != expected_recorded_path:
        raise BenchmarkContextError(f"Context artifact path changed: {path}")
    if not path.is_file():
        raise BenchmarkContextError(f"Context artifact is missing: {path}")
    if int(record.get("bytes", -1)) != path.stat().st_size:
        raise BenchmarkContextError(f"Context artifact byte size changed: {path}")
    if record.get("sha256") != _sha256_file(path):
        raise BenchmarkContextError(f"Context artifact hash changed: {path}")


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise BenchmarkContextError(f"Benchmark context manifest is missing: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise BenchmarkContextError(
            f"Benchmark context manifest is not valid JSON: {path}"
        ) from error
    if not isinstance(payload, dict):
        raise BenchmarkContextError("Benchmark context manifest must contain an object")
    return payload


def _require_matching_processed_directories(
    benchmark_config: BenchmarkConfig,
    cell_config: CellMinerConfig,
) -> None:
    if benchmark_config.cell_processed_dir != cell_config.processed_dir:
        raise BenchmarkContextError(
            "Benchmark and CellMiner configurations reference different processed directories"
        )


def _mapping_lookup(mapping: pd.DataFrame) -> dict[str, str]:
    required = {"source_cell_id", "biological_cell_id", "status"}
    missing = sorted(required - set(mapping.columns))
    if missing:
        raise BenchmarkContextError(f"Prepared cell mapping is missing: {missing}")
    mapped = mapping.loc[mapping["status"].astype(str) == "mapped"].copy()
    mapped["source_cell_id"] = mapped["source_cell_id"].astype(str).str.strip()
    mapped["biological_cell_id"] = (
        mapped["biological_cell_id"].astype(str).str.strip()
    )
    if mapped.empty:
        raise BenchmarkContextError("Prepared cell mapping contains no mapped cells")
    if (mapped[["source_cell_id", "biological_cell_id"]] == "").any(axis=None):
        raise BenchmarkContextError("Prepared cell mapping contains empty mapped IDs")
    duplicate_sources = mapped.loc[
        mapped["source_cell_id"].duplicated(keep=False), "source_cell_id"
    ].unique()
    if len(duplicate_sources):
        raise BenchmarkContextError(
            "Prepared cell mapping has duplicate mapped source IDs: "
            f"{sorted(duplicate_sources)[:10]}"
        )
    return dict(
        zip(mapped["source_cell_id"], mapped["biological_cell_id"], strict=True)
    )


def _benchmark_cell_sets(
    benchmark: PreparedChemicalSplit,
    mapping: pd.DataFrame,
    folds: int,
) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    groups = benchmark.groups
    if "cell_id" not in groups:
        raise BenchmarkContextError("Benchmark group ledger has no source cell_id column")
    lookup = _mapping_lookup(mapping)
    source_cells = tuple(sorted(set(groups["cell_id"].astype(str))))
    unknown = sorted(set(source_cells) - set(lookup))
    if unknown:
        raise BenchmarkContextError(
            f"Benchmark ledger contains cells absent from the mapped context: {unknown[:10]}"
        )
    benchmark_cells = tuple(sorted({lookup[cell_id] for cell_id in source_cells}))
    if not benchmark_cells:
        raise BenchmarkContextError("Benchmark ledger resolves to no biological cells")

    fit_sets: list[tuple[str, ...]] = []
    for fold in range(folds):
        role_column = f"role_fold_{fold}"
        if role_column not in groups:
            raise BenchmarkContextError(
                f"Benchmark group ledger is missing {role_column}"
            )
        train_source_cells = tuple(
            sorted(
                set(
                    groups.loc[
                        groups[role_column].astype(str) == "train", "cell_id"
                    ].astype(str)
                )
            )
        )
        if not train_source_cells:
            raise BenchmarkContextError(f"Fold {fold} has no training source cells")
        fit_cells = tuple(sorted({lookup[cell_id] for cell_id in train_source_cells}))
        if not fit_cells:
            raise BenchmarkContextError(f"Fold {fold} resolves to no fit cells")
        fit_sets.append(fit_cells)
    return benchmark_cells, tuple(fit_sets)


def _context_split_hash(
    benchmark_split_hash: str,
    fold: int,
    fit_cell_ids: tuple[str, ...],
) -> str:
    payload = {
        "context_version": BENCHMARK_CONTEXT_VERSION,
        "benchmark_split_hash": benchmark_split_hash,
        "fold": fold,
        "fit_cell_ids": list(fit_cell_ids),
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _load_cellminer_inputs(
    benchmark_config: BenchmarkConfig,
    cell_config: CellMinerConfig,
) -> tuple[ExpressionMatrix, GeneSetCatalog, PathwayTopology, pd.DataFrame]:
    expression = load_prepared_expression(cell_config)
    catalog = load_hallmark_gene_sets(cell_config)
    topology = load_cached_pathway_topology(
        cell_config, expected_catalog_hash=catalog.catalog_hash
    )
    mapping = load_prepared_cell_mapping(
        cell_config, benchmark_config.nci_processed_dir / "cells.csv"
    )
    if expression.content_hash == "" or catalog.catalog_hash == "":
        raise BenchmarkContextError("Prepared expression/catalog hashes must be non-empty")
    if topology.topology_hash == "":
        raise BenchmarkContextError("Prepared pathway topology hash must be non-empty")
    if topology.catalog_hash != catalog.catalog_hash:
        raise BenchmarkContextError("Prepared topology is bound to another pathway catalog")
    return expression, catalog, topology, mapping


def _cache_directory_record(
    directory: Path,
    cell_config: CellMinerConfig,
) -> str:
    resolved = directory.resolve()
    try:
        relative = resolved.relative_to(cell_config.processed_dir)
    except ValueError as error:
        raise BenchmarkContextError(
            f"Cell graph cache is outside the CellMiner processed directory: {resolved}"
        ) from error
    if not relative.parts or ".." in relative.parts:
        raise BenchmarkContextError("Cell graph cache has an unsafe relative path")
    return str(relative)


def _resolve_cache_directory(
    recorded: Any,
    cell_config: CellMinerConfig,
) -> Path:
    relative = Path(str(recorded))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise BenchmarkContextError("Recorded cell graph cache path is unsafe")
    resolved = (cell_config.processed_dir / relative).resolve()
    try:
        resolved.relative_to(cell_config.processed_dir)
    except ValueError as error:
        raise BenchmarkContextError("Recorded cell graph cache escapes its root") from error
    return resolved


def _base_manifest(
    benchmark_config: BenchmarkConfig,
    benchmark: PreparedChemicalSplit,
    expression: ExpressionMatrix,
    catalog: GeneSetCatalog,
    topology: PathwayTopology,
    benchmark_cells: tuple[str, ...],
) -> dict[str, Any]:
    mapping_path = benchmark_config.cell_processed_dir / "cell_mapping.csv"
    return {
        "context_version": BENCHMARK_CONTEXT_VERSION,
        "protocol": benchmark_config.protocol,
        "benchmark_manifest_hash": benchmark.manifest["manifest_sha256"],
        "benchmark_split_hash": benchmark.split_hash,
        "fold_count": benchmark_config.folds,
        "expression_hash": expression.content_hash,
        "catalog_hash": catalog.catalog_hash,
        "topology_hash": topology.topology_hash,
        "cell_mapping_source": _file_record(mapping_path),
        "benchmark_biological_cell_ids": list(benchmark_cells),
        "benchmark_biological_cell_count": len(benchmark_cells),
    }


def prepare_benchmark_cell_context(
    benchmark_config: BenchmarkConfig,
    cell_config: CellMinerConfig,
) -> PreparedBenchmarkCellContext:
    """Fit and cache five train-only CellMiner contexts for the frozen benchmark."""

    _require_matching_processed_directories(benchmark_config, cell_config)
    benchmark = load_one_new_scaffold_benchmark(benchmark_config)
    expression, catalog, topology, mapping = _load_cellminer_inputs(
        benchmark_config, cell_config
    )
    benchmark_cells, fit_sets = _benchmark_cell_sets(
        benchmark, mapping, benchmark_config.folds
    )
    fold_records: list[dict[str, Any]] = []
    fold_contexts: list[BenchmarkFoldCellContext] = []
    for fold, fit_cells in enumerate(fit_sets):
        context_hash = _context_split_hash(benchmark.split_hash, fold, fit_cells)
        state = fit_train_only_pathway_transform(
            expression,
            catalog,
            topology,
            fit_cells,
            context_hash,
            min_pathway_genes=cell_config.min_pathway_genes,
            clip_value=cell_config.clip_value,
            std_epsilon=cell_config.std_epsilon,
        )
        cached = cache_split_cell_graphs(
            cell_config,
            expression,
            topology,
            state,
            benchmark_cells,
            expected_split_hash=context_hash,
        )
        cache_directory = Path(cached["directory"]).resolve()
        loaded_state, graphs, graph_manifest = load_cached_split_cell_graphs(
            cell_config,
            context_hash,
            expression=expression,
            topology=topology,
            expected_fit_cell_ids=fit_cells,
        )
        if loaded_state.state_hash != state.state_hash:
            raise BenchmarkContextError(
                f"Fold {fold} CellMiner transform changed during cache roundtrip"
            )
        if tuple(sorted(graphs)) != benchmark_cells:
            raise BenchmarkContextError(
                f"Fold {fold} cache does not contain every benchmark biological cell"
            )
        graph_manifest_path = cache_directory / "cell_graph_manifest.json"
        fold_record = {
            "fold": fold,
            "role_column": f"role_fold_{fold}",
            "context_split_hash": context_hash,
            "fit_cell_ids": list(fit_cells),
            "fit_cell_count": len(fit_cells),
            "state_hash": state.state_hash,
            "cache_directory": _cache_directory_record(cache_directory, cell_config),
            "graph_manifest": _file_record(
                graph_manifest_path, relative_to=cell_config.processed_dir
            ),
            "graph_cell_ids": list(benchmark_cells),
            "graph_cell_count": len(graphs),
        }
        fold_records.append(fold_record)
        fold_contexts.append(
            BenchmarkFoldCellContext(
                fold=fold,
                context_split_hash=context_hash,
                fit_cell_ids=fit_cells,
                state=loaded_state,
                graphs=graphs,
                graph_manifest=graph_manifest,
                cache_directory=cache_directory,
            )
        )

    manifest = {
        **_base_manifest(
            benchmark_config,
            benchmark,
            expression,
            catalog,
            topology,
            benchmark_cells,
        ),
        "folds": fold_records,
        "all_folds_fit_all_benchmark_cells": all(
            fit_cells == benchmark_cells for fit_cells in fit_sets
        ),
        "status": "passed",
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        _canonical_json_bytes(manifest)
    ).hexdigest()
    _write_json_atomic(
        benchmark_config.output_dir / CONTEXT_MANIFEST_FILENAME, manifest
    )
    return PreparedBenchmarkCellContext(
        manifest=manifest,
        benchmark=benchmark,
        expression=expression,
        catalog=catalog,
        topology=topology,
        mapping=mapping,
        folds=tuple(fold_contexts),
    )


def _verify_manifest_content_hash(manifest: Mapping[str, Any]) -> None:
    recorded = manifest.get("manifest_sha256")
    unhashed = dict(manifest)
    unhashed.pop("manifest_sha256", None)
    actual = hashlib.sha256(_canonical_json_bytes(unhashed)).hexdigest()
    if recorded != actual:
        raise BenchmarkContextError("Benchmark context manifest content hash changed")


def load_benchmark_cell_context(
    benchmark_config: BenchmarkConfig,
    cell_config: CellMinerConfig,
) -> PreparedBenchmarkCellContext:
    """Load the context only after re-deriving fold fit cells and checking all hashes."""

    _require_matching_processed_directories(benchmark_config, cell_config)
    benchmark = load_one_new_scaffold_benchmark(benchmark_config)
    expression, catalog, topology, mapping = _load_cellminer_inputs(
        benchmark_config, cell_config
    )
    benchmark_cells, fit_sets = _benchmark_cell_sets(
        benchmark, mapping, benchmark_config.folds
    )
    manifest = _load_json_object(
        benchmark_config.output_dir / CONTEXT_MANIFEST_FILENAME
    )
    _verify_manifest_content_hash(manifest)
    expected_base = _base_manifest(
        benchmark_config,
        benchmark,
        expression,
        catalog,
        topology,
        benchmark_cells,
    )
    mismatched = sorted(
        key for key, expected in expected_base.items() if manifest.get(key) != expected
    )
    expected_status = {
        "context_version": BENCHMARK_CONTEXT_VERSION,
        "protocol": benchmark_config.protocol,
        "fold_count": benchmark_config.folds,
        "status": "passed",
        "all_folds_fit_all_benchmark_cells": all(
            fit_cells == benchmark_cells for fit_cells in fit_sets
        ),
    }
    mismatched.extend(
        key
        for key, expected in expected_status.items()
        if manifest.get(key) != expected and key not in mismatched
    )
    if mismatched:
        raise BenchmarkContextError(
            f"Benchmark context does not match current inputs: {sorted(mismatched)}"
        )

    mapping_path = benchmark_config.cell_processed_dir / "cell_mapping.csv"
    _verify_file_record(
        mapping_path,
        manifest["cell_mapping_source"],
        expected_recorded_path=str(mapping_path.resolve()),
    )
    fold_records = manifest.get("folds")
    if not isinstance(fold_records, list) or len(fold_records) != benchmark_config.folds:
        raise BenchmarkContextError("Benchmark context has the wrong fold inventory")
    by_fold: dict[int, Mapping[str, Any]] = {}
    for record in fold_records:
        if not isinstance(record, dict):
            raise BenchmarkContextError("Benchmark context fold record must be an object")
        try:
            fold = int(record["fold"])
        except (KeyError, TypeError, ValueError) as error:
            raise BenchmarkContextError("Benchmark context fold ID is invalid") from error
        if fold in by_fold:
            raise BenchmarkContextError(f"Duplicate benchmark context fold: {fold}")
        by_fold[fold] = record
    if set(by_fold) != set(range(benchmark_config.folds)):
        raise BenchmarkContextError("Benchmark context fold IDs are incomplete")

    contexts: list[BenchmarkFoldCellContext] = []
    for fold, fit_cells in enumerate(fit_sets):
        record = by_fold[fold]
        context_hash = _context_split_hash(benchmark.split_hash, fold, fit_cells)
        expected_record = {
            "role_column": f"role_fold_{fold}",
            "context_split_hash": context_hash,
            "fit_cell_ids": list(fit_cells),
            "fit_cell_count": len(fit_cells),
            "graph_cell_ids": list(benchmark_cells),
            "graph_cell_count": len(benchmark_cells),
        }
        fold_mismatches = sorted(
            key for key, expected in expected_record.items() if record.get(key) != expected
        )
        if fold_mismatches:
            raise BenchmarkContextError(
                f"Fold {fold} context binding changed: {fold_mismatches}"
            )
        cache_directory = _resolve_cache_directory(
            record.get("cache_directory"), cell_config
        )
        graph_manifest_path = cache_directory / "cell_graph_manifest.json"
        expected_graph_path = str(
            graph_manifest_path.resolve().relative_to(cell_config.processed_dir)
        )
        graph_record = record.get("graph_manifest")
        if not isinstance(graph_record, dict):
            raise BenchmarkContextError(f"Fold {fold} graph manifest record is invalid")
        _verify_file_record(
            graph_manifest_path,
            graph_record,
            expected_recorded_path=expected_graph_path,
        )
        state, graphs, graph_manifest = load_cached_split_cell_graphs(
            cell_config,
            context_hash,
            expression=expression,
            topology=topology,
            expected_fit_cell_ids=fit_cells,
        )
        if cache_directory != (
            cell_config.processed_dir / "folds" / graph_manifest["split_cache_key"]
        ).resolve():
            raise BenchmarkContextError(f"Fold {fold} cache directory binding changed")
        if state.state_hash != record.get("state_hash"):
            raise BenchmarkContextError(f"Fold {fold} transform state hash changed")
        if tuple(state.fit_cell_ids) != fit_cells:
            raise BenchmarkContextError(f"Fold {fold} transform fit-cell set changed")
        if tuple(sorted(graphs)) != benchmark_cells:
            raise BenchmarkContextError(f"Fold {fold} cached graph cell set changed")
        contexts.append(
            BenchmarkFoldCellContext(
                fold=fold,
                context_split_hash=context_hash,
                fit_cell_ids=fit_cells,
                state=state,
                graphs=graphs,
                graph_manifest=graph_manifest,
                cache_directory=cache_directory,
            )
        )
    return PreparedBenchmarkCellContext(
        manifest=dict(manifest),
        benchmark=benchmark,
        expression=expression,
        catalog=catalog,
        topology=topology,
        mapping=mapping,
        folds=tuple(contexts),
    )


__all__ = [
    "BENCHMARK_CONTEXT_VERSION",
    "CONTEXT_MANIFEST_FILENAME",
    "BenchmarkContextError",
    "BenchmarkFoldCellContext",
    "PreparedBenchmarkCellContext",
    "load_benchmark_cell_context",
    "prepare_benchmark_cell_context",
]
