"""Validation-only IndependentPooledGNN runner for the frozen scaffold protocol.

This module deliberately has no outer-test evaluation entrypoint.  It reuses the
physically separated train/validation shards and label-free source-partition
ledger from :mod:`bisurfgnn.learned_benchmark`, freezes validation-selected graph
checkpoints, and leaves the test shard sealed until all candidate architectures
have been selected together.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch import Tensor

from .benchmark_config import BenchmarkConfig, load_benchmark_config
from .data.batch import GraphBatch, GraphItem, pad_graph_items
from .data.benchmark_context import (
    PreparedBenchmarkCellContext,
    load_benchmark_cell_context,
)
from .data.cellminer import CellMinerConfig, load_cellminer_config
from .data.chemical_splits import PreparedChemicalSplit, load_one_new_scaffold_benchmark
from .learned_benchmark import (
    EVALUATION_RUN_MANIFEST,
    HALLMARK_DIMENSION,
    SELECTION_RUN_MANIFEST,
    TEST_ACCESS_SEAL,
    CompactPointTable,
    FrozenLearnedPointShards,
    LearnedBenchmarkError,
    _assert_source_roles_disjoint,
    _batch_indices,
    _canonical_hash,
    _clean_text,
    _file_record,
    _fold_selection,
    _fold_surface_roles,
    _load_json_object,
    _load_frozen_selection_checkpoint,
    _point_table_hash,
    _resolve_device,
    _runtime_environment,
    _seed_everything,
    _surface_batch_indices,
    _surface_mean_mse,
    _validate_shard_binding,
    _verify_recorded_file,
    _write_checkpoint_atomic,
    _write_json_atomic,
    build_compact_point_table,
    load_learned_point_shards,
    load_learned_role_frame,
    surface_macro_rmse,
)
from .models.compact_bisurf import CompactBiSurfGNN
from .models.independent_graph import CompactIndependentPooledGNN
from .models.baselines import MorganHallmarkResidualMLP


INDEPENDENT_GRAPH_BENCHMARK_VERSION = "graph-selection-v2-complete-source-binding"
MODEL_SLUG = "independent_pooled_gnn"
BISURF_MODEL_SLUG = "compact_bisurf_gnn"

# Both architectures share this runner because they share the frozen Phase A
# contract: same shards, same anchor, same capacity band, same selection rule.
# Only the model factory and the recorded family differ.
MODEL_FAMILIES = {
    MODEL_SLUG: "independent_shared_drug_gnn_symmetric_pooling_no_cross_attention",
    BISURF_MODEL_SLUG: "shared_drug_gnn_symmetric_bixt_context_gated_continuous_dose",
}
MORGAN_CONTROL_SLUG = "morgan_hallmark_residual_mlp"
PARAMETER_MATCH_TARGET = 2_031_839
GRAPH_CONFIG_SECTION = "independent_graph"


@dataclass(frozen=True)
class IndependentGraphConfig:
    max_epochs: int = 200
    patience: int = 20
    min_epochs: int = 10
    min_delta: float = 1e-5
    batch_surfaces: int = 128
    evaluation_batch_points: int = 8_192
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    residual_shrinkage: float = 0.0
    hidden_dim: int = 171
    graph_layers: int = 2
    dose_frequencies: int = 4
    dropout: float = 0.1
    seeds: tuple[int, ...] = (41, 42, 43)
    reference_mode: str = "observed_hsa"
    architecture: str = MODEL_SLUG
    attention_heads: int = 4
    residual_monotherapy_input: bool = False
    parameter_match_target: int = PARAMETER_MATCH_TARGET
    max_parameter_relative_gap: float = 0.02
    go_relative_improvement: float = 0.05
    go_required_folds: int = 4

    def __post_init__(self) -> None:
        positive = {
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "min_epochs": self.min_epochs,
            "batch_surfaces": self.batch_surfaces,
            "evaluation_batch_points": self.evaluation_batch_points,
            "hidden_dim": self.hidden_dim,
            "graph_layers": self.graph_layers,
            "dose_frequencies": self.dose_frequencies,
            "parameter_match_target": self.parameter_match_target,
            "go_required_folds": self.go_required_folds,
        }
        invalid = sorted(name for name, value in positive.items() if value < 1)
        if invalid:
            raise LearnedBenchmarkError(f"Positive graph config values required: {invalid}")
        if self.min_epochs > self.max_epochs:
            raise LearnedBenchmarkError("min_epochs must not exceed max_epochs")
        if self.architecture not in MODEL_FAMILIES:
            supported = ", ".join(sorted(MODEL_FAMILIES))
            raise LearnedBenchmarkError(
                f"architecture must be one of: {supported}"
            )
        if not isinstance(self.residual_monotherapy_input, bool):
            raise LearnedBenchmarkError("residual_monotherapy_input must be boolean")
        if self.architecture != BISURF_MODEL_SLUG and self.residual_monotherapy_input:
            raise LearnedBenchmarkError("residual_monotherapy_input is a BiSurf version flag; pooled GNN already uses it")
        if self.attention_heads < 1:
            raise LearnedBenchmarkError("attention_heads must be positive")
        if self.architecture == BISURF_MODEL_SLUG and self.hidden_dim % self.attention_heads:
            raise LearnedBenchmarkError(
                "hidden_dim must be divisible by attention_heads for symmetric BiXT"
            )
        finite_values = {
            "min_delta": self.min_delta,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "gradient_clip": self.gradient_clip,
            "residual_shrinkage": self.residual_shrinkage,
            "dropout": self.dropout,
            "max_parameter_relative_gap": self.max_parameter_relative_gap,
            "go_relative_improvement": self.go_relative_improvement,
        }
        non_finite = sorted(
            name for name, value in finite_values.items() if not math.isfinite(value)
        )
        if non_finite:
            raise LearnedBenchmarkError(f"Graph config values must be finite: {non_finite}")
        if self.min_delta < 0 or self.weight_decay < 0 or self.residual_shrinkage < 0:
            raise LearnedBenchmarkError("Graph regularization values must be non-negative")
        if self.learning_rate <= 0 or self.gradient_clip <= 0:
            raise LearnedBenchmarkError("Learning rate and gradient clip must be positive")
        if not 0 <= self.dropout < 1:
            raise LearnedBenchmarkError("dropout must lie in [0,1)")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise LearnedBenchmarkError("seeds must be a non-empty unique sequence")
        if self.reference_mode not in {"observed_hsa", "observed_product"}:
            raise LearnedBenchmarkError("reference_mode must be observed_hsa or observed_product")
        if not 0 <= self.max_parameter_relative_gap < 1:
            raise LearnedBenchmarkError("max_parameter_relative_gap must lie in [0,1)")
        if not 0 < self.go_relative_improvement < 1:
            raise LearnedBenchmarkError("go_relative_improvement must lie in (0,1)")
        if self.go_required_folds > 5:
            raise LearnedBenchmarkError("go_required_folds cannot exceed five")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["seeds"] = list(self.seeds)
        return payload


@dataclass(frozen=True)
class GraphFeatureBanks:
    """Unique fold-specific molecular and cell graphs with source-ID indexing."""

    fold: int
    drug_ids: tuple[str, ...]
    drug_structure_keys: tuple[str, ...]
    cell_ids: tuple[str, ...]
    biological_cell_ids: tuple[str, ...]
    drug_graphs: tuple[GraphItem, ...]
    cell_graphs: tuple[GraphItem, ...]
    drug_graph_hashes: tuple[str, ...]
    cell_graph_hashes: tuple[str, ...]
    context_state_hash: str
    source_assets: dict[str, dict[str, Any]]

    def __post_init__(self) -> None:
        if self.fold < 0 or not self.context_state_hash:
            raise LearnedBenchmarkError("Graph bank fold/state binding is invalid")
        if len(self.drug_structure_keys) != len(self.drug_ids):
            raise LearnedBenchmarkError("Drug structure-key inventory is incomplete")
        if len(self.biological_cell_ids) != len(self.cell_ids):
            raise LearnedBenchmarkError("Biological cell-ID inventory is incomplete")
        inventories = (
            ("drug", self.drug_ids, self.drug_graphs, self.drug_graph_hashes),
            ("cell", self.cell_ids, self.cell_graphs, self.cell_graph_hashes),
        )
        for label, identifiers, graphs, hashes in inventories:
            if not identifiers or len(set(identifiers)) != len(identifiers):
                raise LearnedBenchmarkError(f"{label} graph IDs must be non-empty and unique")
            if len(graphs) != len(identifiers) or len(hashes) != len(identifiers):
                raise LearnedBenchmarkError(f"{label} graph inventory is incomplete")
            feature_dims = {int(graph.node_features.shape[1]) for graph in graphs}
            if len(feature_dims) != 1:
                raise LearnedBenchmarkError(f"{label} graphs have inconsistent feature dims")
            for graph in graphs:
                if not bool(torch.isfinite(graph.node_features).all()) or not bool(
                    torch.isfinite(graph.adjacency).all()
                ):
                    raise LearnedBenchmarkError(f"{label} graph contains non-finite values")

    @property
    def atom_feature_dim(self) -> int:
        return int(self.drug_graphs[0].node_features.shape[1])

    @property
    def cell_feature_dim(self) -> int:
        return int(self.cell_graphs[0].node_features.shape[1])


def load_independent_graph_config(path: str | Path) -> IndependentGraphConfig:
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise LearnedBenchmarkError("Benchmark configuration must be a mapping")
    section = raw.get(GRAPH_CONFIG_SECTION)
    if not isinstance(section, dict):
        raise LearnedBenchmarkError(
            f"Benchmark YAML must define an {GRAPH_CONFIG_SECTION} mapping"
        )
    allowed_keys = {
        "max_epochs",
        "patience",
        "min_epochs",
        "min_delta",
        "batch_surfaces",
        "evaluation_batch_points",
        "learning_rate",
        "weight_decay",
        "gradient_clip",
        "residual_shrinkage",
        "hidden_dim",
        "graph_layers",
        "dose_frequencies",
        "dropout",
        "seeds",
        "reference_mode",
        "architecture",
        "attention_heads",
        "residual_monotherapy_input",
        "parameter_match_target",
        "max_parameter_relative_gap",
        "go_relative_improvement",
        "go_required_folds",
    }
    unknown_keys = sorted(set(section) - allowed_keys)
    if unknown_keys:
        raise LearnedBenchmarkError(
            f"Unknown {GRAPH_CONFIG_SECTION} configuration keys: {unknown_keys}"
        )
    seeds = section.get("seeds", (41, 42, 43))
    if not isinstance(seeds, (list, tuple)) or isinstance(seeds, (str, bytes)):
        raise LearnedBenchmarkError(f"{GRAPH_CONFIG_SECTION}.seeds must be a sequence")
    return IndependentGraphConfig(
        max_epochs=int(section.get("max_epochs", 200)),
        patience=int(section.get("patience", 20)),
        min_epochs=int(section.get("min_epochs", 10)),
        min_delta=float(section.get("min_delta", 1e-5)),
        batch_surfaces=int(section.get("batch_surfaces", 128)),
        evaluation_batch_points=int(section.get("evaluation_batch_points", 8_192)),
        learning_rate=float(section.get("learning_rate", 3e-4)),
        weight_decay=float(section.get("weight_decay", 1e-4)),
        gradient_clip=float(section.get("gradient_clip", 1.0)),
        residual_shrinkage=float(section.get("residual_shrinkage", 0.0)),
        hidden_dim=int(section.get("hidden_dim", 171)),
        graph_layers=int(section.get("graph_layers", 2)),
        dose_frequencies=int(section.get("dose_frequencies", 4)),
        dropout=float(section.get("dropout", 0.1)),
        seeds=tuple(int(seed) for seed in seeds),
        reference_mode=str(section.get("reference_mode", "observed_hsa")),
        architecture=str(section.get("architecture", MODEL_SLUG)),
        attention_heads=int(section.get("attention_heads", 4)),
        residual_monotherapy_input=section.get("residual_monotherapy_input", False),
        parameter_match_target=int(section.get("parameter_match_target", PARAMETER_MATCH_TARGET)),
        max_parameter_relative_gap=float(section.get("max_parameter_relative_gap", 0.02)),
        go_relative_improvement=float(section.get("go_relative_improvement", 0.05)),
        go_required_folds=int(section.get("go_required_folds", 4)),
    )


def _graph_hash(graph: GraphItem) -> str:
    digest = hashlib.sha256()
    for tensor in (graph.node_features, graph.adjacency):
        value = tensor.detach().to(device="cpu").contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _graph_bank_hash(banks: GraphFeatureBanks) -> str:
    return _canonical_hash(
        {
            "fold": banks.fold,
            "drug_ids": list(banks.drug_ids),
            "drug_structure_keys": list(banks.drug_structure_keys),
            "cell_ids": list(banks.cell_ids),
            "biological_cell_ids": list(banks.biological_cell_ids),
            "drug_graph_hashes": list(banks.drug_graph_hashes),
            "cell_graph_hashes": list(banks.cell_graph_hashes),
            "atom_feature_dim": banks.atom_feature_dim,
            "cell_feature_dim": banks.cell_feature_dim,
            "context_state_hash": banks.context_state_hash,
            "source_assets": banks.source_assets,
        }
    )


def _as_graph_item(payload: Mapping[str, Any], label: str) -> GraphItem:
    node_features = payload.get("node_features")
    adjacency = payload.get("adjacency")
    if not isinstance(node_features, Tensor) or not isinstance(adjacency, Tensor):
        raise LearnedBenchmarkError(f"{label} lacks tensor node_features/adjacency")
    return GraphItem(
        node_features=node_features.detach().to(dtype=torch.float32, device="cpu").contiguous(),
        adjacency=adjacency.detach().to(dtype=torch.float32, device="cpu").contiguous(),
    )


def build_graph_feature_banks(
    required_drug_ids: Sequence[str],
    required_source_cell_ids: Sequence[str],
    context: PreparedBenchmarkCellContext,
    fold: int,
    config: BenchmarkConfig,
) -> GraphFeatureBanks:
    """Build graph banks from response-free assignment and cached graph metadata."""

    mapping_path = config.nci_processed_dir / "drug_id_to_structure_key.json"
    structures_path = config.nci_processed_dir / "structures.pt"
    structure_manifest_path = config.nci_processed_dir / "structure_manifest.json"
    for path in (mapping_path, structures_path, structure_manifest_path):
        if not path.is_file():
            raise LearnedBenchmarkError(f"Required graph asset is missing: {path}")
    raw_mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    if not isinstance(raw_mapping, dict):
        raise LearnedBenchmarkError("Drug structure mapping must contain an object")
    structure_mapping = {_clean_text(key): _clean_text(value) for key, value in raw_mapping.items()}
    if any(not key or not value for key, value in structure_mapping.items()):
        raise LearnedBenchmarkError("Drug structure mapping contains empty IDs")
    graph_cache = torch.load(structures_path, map_location="cpu", weights_only=True)
    if not isinstance(graph_cache, dict):
        raise LearnedBenchmarkError("structures.pt must contain a graph mapping")
    drug_ids = tuple(sorted({_clean_text(value) for value in required_drug_ids}))
    missing_drugs = sorted(set(drug_ids) - set(structure_mapping))
    if missing_drugs:
        raise LearnedBenchmarkError(f"Required drug graphs are unmapped: {missing_drugs[:10]}")
    drug_graphs: list[GraphItem] = []
    drug_structure_keys: list[str] = []
    for drug_id in drug_ids:
        structure_key = structure_mapping[drug_id]
        payload = graph_cache.get(structure_key)
        if not isinstance(payload, Mapping):
            raise LearnedBenchmarkError(f"Graph cache lacks structure {structure_key}")
        drug_structure_keys.append(structure_key)
        drug_graphs.append(_as_graph_item(payload, f"drug {drug_id}"))

    if fold < 0 or fold >= len(context.folds):
        raise LearnedBenchmarkError(f"Context has no fold {fold}")
    mapping = context.mapping
    required_columns = {"source_cell_id", "biological_cell_id", "status"}
    missing_columns = sorted(required_columns - set(mapping.columns))
    if missing_columns:
        raise LearnedBenchmarkError(f"Cell mapping is missing columns: {missing_columns}")
    mapped = mapping.loc[mapping["status"].astype(str).eq("mapped")]
    cell_lookup = {
        _clean_text(row.source_cell_id): _clean_text(row.biological_cell_id)
        for row in mapped.itertuples(index=False)
    }
    cell_ids = tuple(sorted({_clean_text(value) for value in required_source_cell_ids}))
    missing_cells = sorted(set(cell_ids) - set(cell_lookup))
    if missing_cells:
        raise LearnedBenchmarkError(f"Required cells are not mapped: {missing_cells[:10]}")
    fold_graphs = context.folds[fold].graphs
    cell_graphs: list[GraphItem] = []
    biological_cell_ids: list[str] = []
    for source_id in cell_ids:
        biological_id = cell_lookup[source_id]
        graph = fold_graphs.get(biological_id)
        if graph is None:
            raise LearnedBenchmarkError(f"Fold {fold} graph cache lacks {biological_id}")
        biological_cell_ids.append(biological_id)
        cell_graphs.append(
            GraphItem(
                node_features=graph.node_features.detach()
                .to(dtype=torch.float32, device="cpu")
                .contiguous(),
                adjacency=graph.adjacency.detach()
                .to(dtype=torch.float32, device="cpu")
                .contiguous(),
            )
        )

    assets = {
        "drug_structure_mapping": _file_record(mapping_path),
        "structures": _file_record(structures_path),
        "structure_manifest": _file_record(structure_manifest_path),
        "fold_cell_graph_manifest": _file_record(
            context.folds[fold].cache_directory / "cell_graph_manifest.json"
        ),
        "fold_cell_graphs": _file_record(
            context.folds[fold].cache_directory
            / str(context.folds[fold].graph_manifest["graph_file"]["filename"])
        ),
    }
    banks = GraphFeatureBanks(
        fold=fold,
        drug_ids=drug_ids,
        drug_structure_keys=tuple(drug_structure_keys),
        cell_ids=cell_ids,
        biological_cell_ids=tuple(biological_cell_ids),
        drug_graphs=tuple(drug_graphs),
        cell_graphs=tuple(cell_graphs),
        drug_graph_hashes=tuple(_graph_hash(graph) for graph in drug_graphs),
        cell_graph_hashes=tuple(_graph_hash(graph) for graph in cell_graphs),
        context_state_hash=context.folds[fold].state.state_hash,
        source_assets=assets,
    )
    structure_manifest = json.loads(structure_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(structure_manifest, dict) or int(
        structure_manifest.get("atom_feature_dim", -1)
    ) != banks.atom_feature_dim:
        raise LearnedBenchmarkError("Structure manifest atom feature dimension changed")
    return banks


def _parameter_count(model: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _build_model(
    banks: GraphFeatureBanks,
    training: IndependentGraphConfig,
    monotherapy_upper_bound: float = 1.0,
) -> CompactIndependentPooledGNN | CompactBiSurfGNN:
    shared = dict(
        atom_feature_dim=banks.atom_feature_dim,
        cell_feature_dim=banks.cell_feature_dim,
        hidden_dim=training.hidden_dim,
        graph_layers=training.graph_layers,
        dose_frequencies=training.dose_frequencies,
        dropout=training.dropout,
        reference_mode=training.reference_mode,
        monotherapy_upper_bound=monotherapy_upper_bound,
    )
    if training.architecture == BISURF_MODEL_SLUG:
        model = CompactBiSurfGNN(
            attention_heads=training.attention_heads,
            residual_monotherapy_input=training.residual_monotherapy_input,
            **shared,
        )
    else:
        model = CompactIndependentPooledGNN(**shared)
    count = _parameter_count(model)
    relative_gap = abs(count - training.parameter_match_target) / training.parameter_match_target
    if relative_gap > training.max_parameter_relative_gap:
        raise LearnedBenchmarkError(
            "Independent graph parameter count violates the frozen matched-capacity band: "
            f"count={count}, target={training.parameter_match_target}, gap={relative_gap:.4%}"
        )
    return model


def _unique_graph_batch(
    global_indices: Tensor,
    graphs: tuple[GraphItem, ...],
    device: torch.device,
) -> tuple[GraphBatch, Tensor]:
    values = global_indices.to(dtype=torch.int64, device="cpu")
    if values.ndim != 1 or len(values) < 1:
        raise LearnedBenchmarkError("Graph lookup indices must be a non-empty vector")
    if int(values.min()) < 0 or int(values.max()) >= len(graphs):
        raise LearnedBenchmarkError("Graph lookup index is out of range")
    unique, inverse = torch.unique(values, sorted=True, return_inverse=True)
    batch = pad_graph_items([graphs[int(index)] for index in unique.tolist()]).to(device)
    return batch, inverse.to(device=device, dtype=torch.int64)


def _model_forward(
    model: CompactIndependentPooledGNN,
    table: CompactPointTable,
    banks: GraphFeatureBanks,
    indices: Tensor,
    device: torch.device,
):
    index_cpu = indices.to(dtype=torch.int64, device="cpu")
    drug_a = table.drug_a_index[index_cpu].to(dtype=torch.int64)
    drug_b = table.drug_b_index[index_cpu].to(dtype=torch.int64)
    drug_graphs, drug_inverse = _unique_graph_batch(
        torch.cat([drug_a, drug_b]), banks.drug_graphs, device
    )
    point_count = len(index_cpu)
    cell_graphs, cell_inverse = _unique_graph_batch(
        table.cell_index[index_cpu].to(dtype=torch.int64), banks.cell_graphs, device
    )
    masks = torch.ones(point_count, dtype=torch.bool, device=device)
    return model(
        drug_graphs,
        cell_graphs,
        drug_inverse[:point_count],
        drug_inverse[point_count:],
        cell_inverse,
        table.doses[index_cpu].to(device),
        table.mono_a[index_cpu].to(device),
        table.mono_b[index_cpu].to(device),
        masks,
        masks,
    )


def predict_graph_compact(
    model: CompactIndependentPooledGNN,
    table: CompactPointTable,
    banks: GraphFeatureBanks,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions: list[np.ndarray] = []
    references: list[np.ndarray] = []
    with torch.no_grad():
        drug_embeddings = model.encode_drugs(
            pad_graph_items(banks.drug_graphs).to(device)
        )
        context_embeddings = model.encode_context(
            pad_graph_items(banks.cell_graphs).to(device)
        )
        for indices in _batch_indices(len(table), batch_size, shuffle_seed=None):
            index_cpu = indices.to(dtype=torch.int64, device="cpu")
            point_count = len(index_cpu)
            masks = torch.ones(point_count, dtype=torch.bool, device=device)
            output = model.forward_pooled(
                drug_embeddings,
                context_embeddings,
                table.drug_a_index[index_cpu].to(device=device, dtype=torch.int64),
                table.drug_b_index[index_cpu].to(device=device, dtype=torch.int64),
                table.cell_index[index_cpu].to(device=device, dtype=torch.int64),
                table.doses[index_cpu].to(device),
                table.mono_a[index_cpu].to(device),
                table.mono_b[index_cpu].to(device),
                masks,
                masks,
            )
            predictions.append(output.prediction.detach().cpu().numpy())
            references.append(output.reference.detach().cpu().numpy())
    return np.concatenate(predictions), np.concatenate(references)


def fit_independent_fold_seed(
    fold: int,
    banks: GraphFeatureBanks,
    train: CompactPointTable,
    validation: CompactPointTable,
    training: IndependentGraphConfig,
    *,
    seed: int,
    device: str | torch.device | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    """Fit one graph model while allowing the exact HSA anchor to win selection."""

    _seed_everything(seed)
    resolved_device = _resolve_device(device)
    model = _build_model(
        banks, training, monotherapy_upper_bound=train.monotherapy_upper_bound
    ).to(resolved_device)
    parameter_count = _parameter_count(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay
    )
    initial_prediction, initial_reference = predict_graph_compact(
        model,
        validation,
        banks,
        batch_size=training.evaluation_batch_points,
        device=resolved_device,
    )
    if not np.allclose(initial_prediction, initial_reference, rtol=2e-6, atol=2e-7):
        raise LearnedBenchmarkError("Zero-residual graph initialization is not the exact anchor")
    anchor_validation = surface_macro_rmse(
        validation.target.numpy(), initial_reference, validation.surface_ids
    )
    best_validation = anchor_validation
    best_epoch = -1
    best_state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    history: list[dict[str, Any]] = [
        {
            "epoch": -1,
            "checkpoint_kind": "anchor_no_learning",
            "train_mse": None,
            "train_residual_penalty": None,
            "validation_surface_macro_rmse": anchor_validation,
            "is_best": True,
        }
    ]
    stale_epochs = 0
    epochs_ran = 0
    for epoch in range(training.max_epochs):
        model.train()
        fit_sum = 0.0
        penalty_sum = 0.0
        surface_count = 0
        batches = _surface_batch_indices(
            train,
            training.batch_surfaces,
            shuffle_seed=seed * 1_000_003 + epoch,
        )
        if progress_callback is not None:
            progress_callback({"phase": "epoch_started", "epoch": epoch + 1, "epochs": training.max_epochs, "batches": len(batches)})
        for batch_index, indices in enumerate(batches, start=1):
            optimizer.zero_grad(set_to_none=True)
            output = _model_forward(model, train, banks, indices, resolved_device)
            index_cpu = indices.to(dtype=torch.int64, device="cpu")
            surfaces = train.surface_ids[index_cpu.numpy()]
            target = train.target[index_cpu].to(resolved_device)
            fit_loss = _surface_mean_mse(output.prediction, target, surfaces)
            residual_penalty = _surface_mean_mse(
                output.prediction, output.reference, surfaces
            )
            loss = fit_loss + training.residual_shrinkage * residual_penalty
            if not bool(torch.isfinite(loss)):
                raise LearnedBenchmarkError("Independent graph training loss became non-finite")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), training.gradient_clip
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise LearnedBenchmarkError("Independent graph gradient became non-finite")
            optimizer.step()
            if progress_callback is not None:
                progress_callback({"phase": "batch_complete", "epoch": epoch + 1, "epochs": training.max_epochs, "batch": batch_index, "batches": len(batches), "loss": float(loss.detach().cpu())})
            batch_surfaces = len(set(surfaces.tolist()))
            fit_sum += float(fit_loss.detach().cpu()) * batch_surfaces
            penalty_sum += float(residual_penalty.detach().cpu()) * batch_surfaces
            surface_count += batch_surfaces
        validation_prediction, _ = predict_graph_compact(
            model,
            validation,
            banks,
            batch_size=training.evaluation_batch_points,
            device=resolved_device,
        )
        validation_rmse = surface_macro_rmse(
            validation.target.numpy(), validation_prediction, validation.surface_ids
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
                "checkpoint_kind": "learned",
                "train_mse": fit_sum / surface_count,
                "train_residual_penalty": penalty_sum / surface_count,
                "validation_surface_macro_rmse": validation_rmse,
                "is_best": improved,
            }
        )
        epochs_ran += 1
        if progress_callback is not None:
            progress_callback({"phase": "epoch_complete", "epoch": epoch + 1, "epochs": training.max_epochs, "train_mse": fit_sum / surface_count, "validation_surface_macro_rmse": validation_rmse, "best_epoch": best_epoch})
        if epochs_ran >= training.min_epochs and stale_epochs >= training.patience:
            break
    relative_improvement = (
        (anchor_validation - best_validation) / anchor_validation
        if anchor_validation > 0
        else 0.0
    )
    result = {
        "model": training.architecture,
        "anchor_reference_model": training.reference_mode,
        "fold": fold,
        "seed": seed,
        "best_epoch": best_epoch,
        "selected_checkpoint_kind": (
            "anchor_no_learning" if best_epoch == -1 else "learned"
        ),
        "epochs_ran": epochs_ran,
        "best_validation_surface_macro_rmse": best_validation,
        "anchor_validation_surface_macro_rmse": anchor_validation,
        "validation_relative_improvement_vs_anchor": relative_improvement,
        "selection_metric": "validation.surface_macro.rmse",
        "parameter_count": parameter_count,
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


def _selection_code_paths() -> dict[str, Path]:
    package = Path(__file__).resolve().parent
    return {
        "independent_graph_benchmark": Path(__file__).resolve(),
        "independent_graph_model": package / "models" / "independent_graph.py",
        "compact_bisurf_model": package / "models" / "compact_bisurf.py",
        "symmetric_bixt": package / "models" / "symmetric_bixt.py",
        "context_gate": package / "models" / "context_gate.py",
        "learned_shard_contract": package / "learned_benchmark.py",
        "benchmark_metrics": package / "benchmark.py",
        "benchmark_config": package / "benchmark_config.py",
        "baseline_anchor": package / "models" / "baselines.py",
        "graph_encoder": package / "models" / "graph_encoder.py",
        "surface_head": package / "models" / "surface_head.py",
        "graph_batch": package / "data" / "batch.py",
        "benchmark_context": package / "data" / "benchmark_context.py",
        "cellminer": package / "data" / "cellminer.py",
        "chemical_splits": package / "data" / "chemical_splits.py",
    }


def _current_code_records() -> dict[str, dict[str, Any]]:
    return {name: _file_record(path) for name, path in _selection_code_paths().items()}


def _verify_code_records(records: Mapping[str, Any]) -> None:
    expected_names = set(_selection_code_paths())
    if set(records) != expected_names:
        raise LearnedBenchmarkError("Frozen graph code-source inventory changed")
    for name, path in _selection_code_paths().items():
        record = records.get(name)
        if not isinstance(record, Mapping):
            raise LearnedBenchmarkError(f"Frozen graph code record is invalid: {name}")
        _verify_recorded_file(path, record, f"Frozen graph {name} source")


def _snapshot_code_sources(
    job_dir: Path, records: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Archive verified source bytes for new jobs; never repair historical records."""
    _verify_code_records(records)
    snapshots = {}
    for name, source in _selection_code_paths().items():
        destination = job_dir / "source_snapshot" / name / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copyfile(source, destination)
        record = _file_record(destination)
        if record["sha256"] != records[name]["sha256"]:
            raise LearnedBenchmarkError(f"Source snapshot differs from frozen code: {name}")
        snapshots[name] = record
    _verify_code_records(records)
    return snapshots


def _graph_feature_metadata(training: IndependentGraphConfig) -> dict[str, Any]:
    return {
        "molecular": "prepared_atom_graph_node_features_and_dense_adjacency",
        "cell": "fold_train_fit_50_hallmark_pathway_graph",
        "drug_encoder_shared_between_arms": True,
        "cross_drug_attention": training.architecture == BISURF_MODEL_SLUG,
        "residual_reads_both_monotherapies": (
            training.architecture == MODEL_SLUG or training.residual_monotherapy_input
        ),
        "decoder_version": (
            "bisurf_observed_mono_v2" if training.residual_monotherapy_input
            else "bisurf_legacy_v1" if training.architecture == BISURF_MODEL_SLUG
            else "pooled_observed_mono_v1"
        ),
        "point_batching": "encode_unique_graphs_then_gather_point_indices",
    }


def _load_morgan_capacity_control(
    benchmark: BenchmarkConfig,
    training: IndependentGraphConfig,
    *,
    expected_split_hash: str,
) -> dict[str, Any]:
    """Verify and bind the actual full Morgan Phase A control artifacts."""

    control_dir = (
        benchmark.runs_dir / f"{MORGAN_CONTROL_SLUG}_{training.reference_mode}"
    ).resolve()
    run_path = control_dir / SELECTION_RUN_MANIFEST
    run = _load_json_object(run_path, "Morgan capacity-control selection run")
    recorded_hash = run.get("manifest_sha256")
    unhashed = dict(run)
    unhashed.pop("manifest_sha256", None)
    if recorded_hash != _canonical_hash(unhashed):
        raise LearnedBenchmarkError("Morgan capacity-control manifest hash changed")
    expected_run = {
        "status": "selection_frozen_no_test_shard_opened",
        "model": MORGAN_CONTROL_SLUG,
        "run_mode": "full_five_fold_engineering",
        "selected_folds": list(range(benchmark.folds)),
        "max_train_points": None,
        "split_hash": expected_split_hash,
        "protocol": benchmark.protocol,
        "endpoint": benchmark.endpoint,
    }
    mismatched = [
        key for key, value in expected_run.items() if run.get(key) != value
    ]
    if mismatched:
        raise LearnedBenchmarkError(
            f"Morgan capacity-control run is not a full frozen selection: {mismatched}"
        )
    control_training = run.get("training_config")
    if not isinstance(control_training, dict) or control_training.get(
        "reference_mode"
    ) != training.reference_mode:
        raise LearnedBenchmarkError("Morgan capacity control uses another anchor")
    jobs = run.get("jobs")
    if not isinstance(jobs, list):
        raise LearnedBenchmarkError("Morgan capacity-control job inventory is invalid")
    expected_jobs = {
        (fold, int(seed))
        for fold in range(benchmark.folds)
        for seed in control_training.get("seeds", [])
    }
    jobs_by_id: dict[tuple[int, int], Mapping[str, Any]] = {}
    observed_state_counts: set[int] = set()
    for job in jobs:
        if not isinstance(job, Mapping):
            raise LearnedBenchmarkError("Morgan capacity-control job must be an object")
        job_id = (int(job.get("fold", -1)), int(job.get("seed", -1)))
        if job_id in jobs_by_id:
            raise LearnedBenchmarkError(f"Duplicate Morgan capacity-control job: {job_id}")
        jobs_by_id[job_id] = job
        fold, seed = job_id
        checkpoint_path = control_dir / f"fold_{fold}" / f"seed_{seed}" / "checkpoint.pt"
        selection_path = (
            control_dir / f"fold_{fold}" / f"seed_{seed}" / "selection_manifest.json"
        )
        for key, path in (
            ("checkpoint", checkpoint_path),
            ("selection_manifest", selection_path),
        ):
            record = job.get(key)
            if not isinstance(record, Mapping):
                raise LearnedBenchmarkError(f"Morgan capacity-control job lacks {key}")
            _verify_recorded_file(path, record, f"Morgan capacity-control {fold}/{seed} {key}")
        _, state = _load_frozen_selection_checkpoint(checkpoint_path, selection_path)
        observed_state_counts.add(
            sum(
                value.numel()
                for name, value in state.items()
                if name != "dose_features.frequencies"
            )
        )
    if set(jobs_by_id) != expected_jobs or len(expected_jobs) != benchmark.folds * 3:
        raise LearnedBenchmarkError("Morgan capacity-control fold/seed inventory is incomplete")
    hidden_dim = int(control_training.get("hidden_dim", -1))
    dropout = float(control_training.get("dropout", float("nan")))
    control_model = MorganHallmarkResidualMLP(
        fingerprint_dim=benchmark.fingerprint_bits,
        hallmark_dim=HALLMARK_DIMENSION,
        hidden_dim=hidden_dim,
        dropout=dropout,
        reference_mode=training.reference_mode,
    )
    current_parameter_count = _parameter_count(control_model)
    if observed_state_counts != {current_parameter_count}:
        raise LearnedBenchmarkError(
            "Morgan capacity-control checkpoint structure differs from the current model"
        )
    if current_parameter_count != training.parameter_match_target:
        raise LearnedBenchmarkError(
            "Graph parameter target differs from the verified Morgan control: "
            f"target={training.parameter_match_target}, control={current_parameter_count}"
        )
    return {
        "model": MORGAN_CONTROL_SLUG,
        "benchmark_version": run.get("benchmark_version"),
        "selection_run_manifest": _file_record(run_path),
        "selection_run_content_sha256": recorded_hash,
        "job_count": len(jobs),
        "trainable_parameter_count": current_parameter_count,
        "state_count_rule": "exclude_persistent_dose_frequency_buffer",
    }


def _freeze_graph_selection_checkpoint(
    job_dir: Path,
    *,
    fold: int,
    seed: int,
    selection: Mapping[str, Any],
    state: Mapping[str, Tensor],
    training: IndependentGraphConfig,
    configuration_source: Mapping[str, Any],
    code_sources: Mapping[str, Any],
    capacity_control: Mapping[str, Any],
    split_hash: str,
    context_split_hash: str,
    context_state_hash: str,
    graph_bank_hash: str,
    graph_assets: Mapping[str, Any],
    train_shard: Mapping[str, Any],
    validation_shard: Mapping[str, Any],
    source_partition: Mapping[str, Any],
) -> tuple[Path, Path, dict[str, Any]]:
    _verify_code_records(code_sources)
    source_snapshot = _snapshot_code_sources(job_dir, code_sources)
    checkpoint_path = job_dir / "checkpoint.pt"
    selection_path = job_dir / "selection_manifest.json"
    semantic_binding = {
        "benchmark_version": INDEPENDENT_GRAPH_BENCHMARK_VERSION,
        "model": training.architecture,
        "fold": fold,
        "seed": seed,
        "best_epoch": selection["best_epoch"],
        "selected_checkpoint_kind": selection["selected_checkpoint_kind"],
        "parameter_count": selection["parameter_count"],
        "split_hash": split_hash,
        "context_split_hash": context_split_hash,
        "context_state_hash": context_state_hash,
        "graph_bank_hash": graph_bank_hash,
        "graph_assets": dict(graph_assets),
        "configuration_source": dict(configuration_source),
        "capacity_control": dict(capacity_control),
        "training_config": training.to_dict(),
        "runtime_environment": _runtime_environment(),
        "train_shard": dict(train_shard),
        "validation_shard": dict(validation_shard),
        "source_partition": dict(source_partition),
        "selection_hash": _canonical_hash(selection),
    }
    _write_checkpoint_atomic(checkpoint_path, {**semantic_binding, "state_dict": dict(state)})
    manifest: dict[str, Any] = {
        "status": "selection_frozen_before_test_response_read",
        **semantic_binding,
        "selection": dict(selection),
        "checkpoint": _file_record(checkpoint_path),
        "code_sources": dict(code_sources),
        "source_snapshot": source_snapshot,
    }
    manifest["manifest_sha256"] = _canonical_hash(manifest)
    _write_json_atomic(selection_path, manifest)
    return checkpoint_path, selection_path, manifest


def _load_frozen_graph_selection(
    checkpoint_path: Path,
    selection_path: Path,
    *,
    expected: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    manifest = json.loads(selection_path.read_text(encoding="utf-8"))
    recorded_hash = manifest.pop("manifest_sha256", None)
    if recorded_hash != _canonical_hash(manifest):
        raise LearnedBenchmarkError("Graph selection manifest content hash changed")
    checkpoint_record = manifest.get("checkpoint")
    if not isinstance(checkpoint_record, dict):
        raise LearnedBenchmarkError("Graph selection manifest has no checkpoint record")
    actual_record = _file_record(checkpoint_path)
    if any(
        actual_record.get(key) != checkpoint_record.get(key)
        for key in ("path", "bytes", "sha256")
    ):
        raise LearnedBenchmarkError("Frozen graph checkpoint hash changed")
    code_sources = manifest.get("code_sources")
    if not isinstance(code_sources, dict):
        raise LearnedBenchmarkError("Graph selection has no code-source binding")
    for name, path in _selection_code_paths().items():
        record = code_sources.get(name)
        if not isinstance(record, dict):
            raise LearnedBenchmarkError(f"Graph selection lacks {name} source binding")
        _verify_recorded_file(path, record, f"Graph selection {name} source")
    snapshots = manifest.get("source_snapshot")
    if not isinstance(snapshots, dict) or set(snapshots) != set(code_sources):
        raise LearnedBenchmarkError("Graph selection has incomplete source snapshots")
    for name, record in snapshots.items():
        _verify_recorded_file(Path(record["path"]), record, f"Snapshot {name}")
        if record["sha256"] != code_sources[name]["sha256"]:
            raise LearnedBenchmarkError(f"Source snapshot binding differs: {name}")
    graph_assets = manifest.get("graph_assets")
    if not isinstance(graph_assets, dict) or not graph_assets:
        raise LearnedBenchmarkError("Graph selection has no source-asset binding")
    for name, record in graph_assets.items():
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise LearnedBenchmarkError(f"Graph asset record is invalid: {name}")
        _verify_recorded_file(
            Path(record["path"]), record, f"Graph selection {name} asset"
        )
    configuration_source = manifest.get("configuration_source")
    if not isinstance(configuration_source, dict):
        raise LearnedBenchmarkError("Graph selection has no configuration binding")
    if configuration_source.get("kind") == "yaml_file":
        if not isinstance(configuration_source.get("path"), str):
            raise LearnedBenchmarkError("Graph YAML configuration record is invalid")
        _verify_recorded_file(
            Path(configuration_source["path"]),
            configuration_source,
            "Graph selection YAML configuration",
        )
    elif configuration_source.get("kind") != "explicit_in_memory_configuration":
        raise LearnedBenchmarkError("Graph configuration binding kind is invalid")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise LearnedBenchmarkError("Frozen graph checkpoint payload is invalid")
    state = payload["state_dict"]
    if not state or any(
        not isinstance(value, Tensor) or not bool(torch.isfinite(value).all())
        for value in state.values()
    ):
        raise LearnedBenchmarkError("Frozen graph checkpoint state is empty or non-finite")
    semantic_keys = (
        "benchmark_version",
        "model",
        "fold",
        "seed",
        "best_epoch",
        "selected_checkpoint_kind",
        "parameter_count",
        "split_hash",
        "context_split_hash",
        "context_state_hash",
        "graph_bank_hash",
        "graph_assets",
        "configuration_source",
        "capacity_control",
        "training_config",
        "runtime_environment",
        "train_shard",
        "validation_shard",
        "source_partition",
        "selection_hash",
    )
    mismatched = [key for key in semantic_keys if payload.get(key) != manifest.get(key)]
    if mismatched:
        raise LearnedBenchmarkError(f"Graph checkpoint semantic binding changed: {mismatched}")
    if manifest.get("runtime_environment") != _runtime_environment():
        raise LearnedBenchmarkError("Frozen graph runtime differs from the current runtime")
    selection = manifest.get("selection")
    if not isinstance(selection, dict) or manifest.get(
        "selection_hash"
    ) != _canonical_hash(selection):
        raise LearnedBenchmarkError("Graph selection result hash changed")
    required_expected = {
        "benchmark_version",
        "model",
        "fold",
        "seed",
        "parameter_count",
        "split_hash",
        "context_split_hash",
        "context_state_hash",
        "graph_bank_hash",
        "graph_assets",
        "configuration_source",
        "capacity_control",
        "training_config",
        "runtime_environment",
        "train_shard",
        "validation_shard",
        "source_partition",
    }
    missing_expected = sorted(required_expected - set(expected))
    if missing_expected:
        raise LearnedBenchmarkError(
            "Graph selection verification requires a rebuilt current contract: "
            f"{missing_expected}"
        )
    expected_mismatch = [
        key for key, value in expected.items() if manifest.get(key) != value
    ]
    if expected_mismatch:
        raise LearnedBenchmarkError(
            f"Frozen graph selection differs from current run: {expected_mismatch}"
        )
    manifest["manifest_sha256"] = recorded_hash
    return manifest, state


def _resolve_inputs(
    config: BenchmarkConfig | str | Path,
    *,
    cell_config: CellMinerConfig | str | Path,
    graph_config: IndependentGraphConfig | None,
) -> tuple[BenchmarkConfig, CellMinerConfig, IndependentGraphConfig]:
    if isinstance(config, BenchmarkConfig):
        if graph_config is None:
            raise LearnedBenchmarkError("graph_config is required with a BenchmarkConfig object")
        benchmark = config
    else:
        benchmark = load_benchmark_config(config)
        graph_config = graph_config or load_independent_graph_config(config)
    resolved_cell = (
        cell_config
        if isinstance(cell_config, CellMinerConfig)
        else load_cellminer_config(cell_config)
    )
    return benchmark, resolved_cell, graph_config


def _configuration_binding(
    config: BenchmarkConfig | str | Path,
    benchmark: BenchmarkConfig,
    training: IndependentGraphConfig,
) -> dict[str, Any]:
    if isinstance(config, BenchmarkConfig):
        return {
            "kind": "explicit_in_memory_configuration",
            "benchmark_hash": _canonical_hash(benchmark.to_dict()),
            "independent_graph_hash": _canonical_hash(training.to_dict()),
        }
    path = Path(config).resolve()
    return {"kind": "yaml_file", **_file_record(path)}


def _run_directory(
    benchmark: BenchmarkConfig,
    training: IndependentGraphConfig,
    output_dir: str | Path | None,
) -> Path:
    if output_dir is not None:
        return Path(output_dir).resolve()
    return (
        benchmark.runs_dir / f"{training.architecture}_{training.reference_mode}"
    ).resolve()


def _selected_seeds(
    seeds: Sequence[int] | None, training: IndependentGraphConfig
) -> tuple[int, ...]:
    selected = training.seeds if seeds is None else tuple(int(seed) for seed in seeds)
    if not selected or len(set(selected)) != len(selected):
        raise LearnedBenchmarkError("Seed selection must be non-empty and unique")
    unknown = sorted(set(selected) - set(training.seeds))
    if unknown:
        raise LearnedBenchmarkError(f"Selected seeds are outside the frozen config: {unknown}")
    return selected


def _protected_test_artifacts(benchmark: BenchmarkConfig, destination: Path) -> list[Path]:
    candidates = [
        benchmark.runs_dir / TEST_ACCESS_SEAL,
        destination / TEST_ACCESS_SEAL,
        destination / EVALUATION_RUN_MANIFEST,
    ]
    candidates.extend(benchmark.runs_dir.glob(f"*/{TEST_ACCESS_SEAL}"))
    return sorted({path.resolve() for path in candidates if path.exists()})


def _clear_known_selection_artifacts(destination: Path, expected_model: str) -> None:
    if not destination.exists():
        return
    selection_path = destination / SELECTION_RUN_MANIFEST
    if not selection_path.is_file():
        raise LearnedBenchmarkError(
            "Overwrite requires a complete owned graph selection manifest; "
            f"use a new output directory for partial artifacts: {destination}"
        )
    owner = _load_json_object(selection_path, "Graph selection-run manifest")
    recorded_hash = owner.get("manifest_sha256")
    unhashed = dict(owner)
    unhashed.pop("manifest_sha256", None)
    if recorded_hash != _canonical_hash(unhashed):
        raise LearnedBenchmarkError("Graph overwrite owner manifest hash changed")
    if owner.get("model") != expected_model or owner.get(
        "benchmark_version"
    ) != INDEPENDENT_GRAPH_BENCHMARK_VERSION:
        raise LearnedBenchmarkError(
            f"Refusing to overwrite a directory owned by another run: {destination}"
        )
    children = list(destination.iterdir())
    unknown = [
        child
        for child in children
        if not (
            (child == selection_path and child.is_file())
            or (child.is_dir() and re.fullmatch(r"fold_\d+", child.name))
        )
    ]
    if unknown:
        raise LearnedBenchmarkError(
            f"Refusing to overwrite unknown graph-run artifact: {unknown[0]}"
        )
    # Deletion starts only after ownership and the complete child inventory pass.
    selection_path.unlink()
    for child in children:
        if child.is_dir():
            shutil.rmtree(child)


def _selection_expected(
    *,
    fold: int,
    seed: int,
    split: PreparedChemicalSplit,
    context: PreparedBenchmarkCellContext,
    banks: GraphFeatureBanks,
    training: IndependentGraphConfig,
    configuration_source: Mapping[str, Any],
    capacity_control: Mapping[str, Any],
    shards: FrozenLearnedPointShards,
    parameter_count: int,
) -> dict[str, Any]:
    return {
        "benchmark_version": INDEPENDENT_GRAPH_BENCHMARK_VERSION,
        "model": training.architecture,
        "fold": fold,
        "seed": seed,
        "parameter_count": parameter_count,
        "split_hash": split.split_hash,
        "context_split_hash": context.folds[fold].context_split_hash,
        "context_state_hash": context.folds[fold].state.state_hash,
        "graph_bank_hash": _graph_bank_hash(banks),
        "graph_assets": banks.source_assets,
        "configuration_source": dict(configuration_source),
        "capacity_control": dict(capacity_control),
        "training_config": training.to_dict(),
        "runtime_environment": _runtime_environment(),
        "train_shard": shards.role_record(fold, "train"),
        "validation_shard": shards.role_record(fold, "validation"),
        "source_partition": shards.source_partition_record(fold),
    }


def _validation_gate(
    jobs: Sequence[Mapping[str, Any]],
    selected_folds: Sequence[int],
    selected_seeds: Sequence[int],
    training: IndependentGraphConfig,
    fold_count: int,
) -> dict[str, Any]:
    full_inventory = tuple(selected_folds) == tuple(range(fold_count)) and tuple(
        selected_seeds
    ) == training.seeds
    folds: list[dict[str, Any]] = []
    passing = 0
    for fold in selected_folds:
        fold_jobs = [job for job in jobs if int(job["fold"]) == int(fold)]
        learned = float(np.mean([job["best_validation_surface_macro_rmse"] for job in fold_jobs]))
        anchor = float(np.mean([job["anchor_validation_surface_macro_rmse"] for job in fold_jobs]))
        relative = (anchor - learned) / anchor if anchor > 0 else 0.0
        meets = relative >= training.go_relative_improvement
        passing += int(meets)
        folds.append(
            {
                "fold": int(fold),
                "learned_seed_mean_surface_macro_rmse": learned,
                "anchor_surface_macro_rmse": anchor,
                "relative_improvement": relative,
                "meets_fold_threshold": meets,
            }
        )
    if not full_inventory:
        decision = "not_applicable_engineering_smoke"
    elif passing >= training.go_required_folds:
        decision = "go_candidate_requires_human_signoff_no_test_entrypoint"
    else:
        decision = "no_go_keep_test_sealed"
    return {
        "decision": decision,
        "full_five_fold_three_seed_inventory": full_inventory,
        "required_relative_improvement_per_fold": training.go_relative_improvement,
        "required_passing_folds": training.go_required_folds,
        "observed_passing_folds": passing,
        "folds": folds,
        "test_action": "none_validation_only",
    }


def select_independent_graph_benchmark(
    config: BenchmarkConfig | str | Path,
    *,
    cell_config: CellMinerConfig | str | Path,
    graph_config: IndependentGraphConfig | None = None,
    folds: Sequence[int] | None = None,
    seeds: Sequence[int] | None = None,
    max_train_points: int | None = None,
    output_dir: str | Path | None = None,
    device: str | torch.device | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run graph-model Phase A without hashing or decoding a test response shard."""

    benchmark, resolved_cell, training = _resolve_inputs(
        config, cell_config=cell_config, graph_config=graph_config
    )
    configuration_source = _configuration_binding(config, benchmark, training)
    frozen_code_sources = _current_code_records()
    selected_folds = _fold_selection(folds, benchmark.folds)
    selected_seeds = _selected_seeds(seeds, training)
    destination = _run_directory(benchmark, training, output_dir)
    protected = _protected_test_artifacts(benchmark, destination)
    if protected:
        raise LearnedBenchmarkError(
            f"Graph selection is forbidden after any outer-test seal: {protected[0]}"
        )
    selection_run_path = destination / SELECTION_RUN_MANIFEST
    if destination.exists() and any(destination.iterdir()):
        if not overwrite:
            raise LearnedBenchmarkError(
                f"Graph selection output already exists; use overwrite explicitly: {destination}"
            )
        _clear_known_selection_artifacts(destination, training.architecture)

    split = load_one_new_scaffold_benchmark(benchmark)
    context = load_benchmark_cell_context(benchmark, resolved_cell)
    if context.split_hash != split.split_hash:
        raise LearnedBenchmarkError("Cell context is bound to another chemical split")
    shards = load_learned_point_shards(benchmark, verify_roles=("train", "validation"))
    _validate_shard_binding(shards, split)
    capacity_control = _load_morgan_capacity_control(
        benchmark, training, expected_split_hash=split.split_hash
    )

    jobs: list[dict[str, Any]] = []
    parameter_count: int | None = None
    for fold in selected_folds:
        assignments = _fold_surface_roles(split.surfaces, fold)
        required_drugs = sorted(set(assignments["drug_a_id"]) | set(assignments["drug_b_id"]))
        required_cells = sorted(set(assignments["cell_id"]))
        banks = build_graph_feature_banks(
            required_drugs, required_cells, context, fold, benchmark
        )
        fold_parameter_count = _parameter_count(_build_model(banks, training))
        if parameter_count is None:
            parameter_count = fold_parameter_count
        elif parameter_count != fold_parameter_count:
            raise LearnedBenchmarkError("Graph model parameter count changed across folds")
        train_frame = load_learned_role_frame(
            shards,
            split.surfaces,
            fold,
            "train",
            max_train_points=max_train_points,
            sample_seed=benchmark.split_seed,
        )
        validation_frame = load_learned_role_frame(
            shards,
            split.surfaces,
            fold,
            "validation",
            sample_seed=benchmark.split_seed,
        )
        _assert_source_roles_disjoint({"train": train_frame, "validation": validation_frame}, fold)
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
        bank_hash = _graph_bank_hash(banks)
        for seed in selected_seeds:
            selection, best_state = fit_independent_fold_seed(
                fold,
                banks,
                train_table,
                validation_table,
                training,
                seed=seed,
                device=device,
            )
            job_dir = destination / f"fold_{fold}" / f"seed_{seed}"
            checkpoint_path, selection_path, frozen = _freeze_graph_selection_checkpoint(
                job_dir,
                fold=fold,
                seed=seed,
                selection=selection,
                state=best_state,
                training=training,
                configuration_source=configuration_source,
                code_sources=frozen_code_sources,
                capacity_control=capacity_control,
                split_hash=split.split_hash,
                context_split_hash=context.folds[fold].context_split_hash,
                context_state_hash=context.folds[fold].state.state_hash,
                graph_bank_hash=bank_hash,
                graph_assets=banks.source_assets,
                train_shard=shards.role_record(fold, "train"),
                validation_shard=shards.role_record(fold, "validation"),
                source_partition=shards.source_partition_record(fold),
            )
            verified, _ = _load_frozen_graph_selection(
                checkpoint_path,
                selection_path,
                expected=_selection_expected(
                    fold=fold,
                    seed=seed,
                    split=split,
                    context=context,
                    banks=banks,
                    training=training,
                    configuration_source=configuration_source,
                    capacity_control=capacity_control,
                    shards=shards,
                    parameter_count=fold_parameter_count,
                ),
            )
            if verified["manifest_sha256"] != frozen["manifest_sha256"]:
                raise LearnedBenchmarkError("Graph selection changed immediately after freezing")
            jobs.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "best_epoch": selection["best_epoch"],
                    "selected_checkpoint_kind": selection["selected_checkpoint_kind"],
                    "best_validation_surface_macro_rmse": selection[
                        "best_validation_surface_macro_rmse"
                    ],
                    "anchor_validation_surface_macro_rmse": selection[
                        "anchor_validation_surface_macro_rmse"
                    ],
                    "validation_relative_improvement_vs_anchor": selection[
                        "validation_relative_improvement_vs_anchor"
                    ],
                    "parameter_count": fold_parameter_count,
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
                    "graph_bank_shapes": {
                        "drugs": len(banks.drug_graphs),
                        "cells": len(banks.cell_graphs),
                        "atom_feature_dim": banks.atom_feature_dim,
                        "cell_feature_dim": banks.cell_feature_dim,
                    },
                    "graph_bank_hash": bank_hash,
                    "graph_assets": banks.source_assets,
                    "source_partition": shards.source_partition_record(fold),
                    "checkpoint": _file_record(checkpoint_path),
                    "selection_manifest": _file_record(selection_path),
                }
            )
    assert parameter_count is not None
    _verify_code_records(frozen_code_sources)
    validation_gate = _validation_gate(
        jobs, selected_folds, selected_seeds, training, benchmark.folds
    )
    manifest: dict[str, Any] = {
        "status": "selection_frozen_no_test_shard_opened",
        "claim_scope": "engineering_only_no_scientific_causal_or_sota_claim",
        "benchmark_version": INDEPENDENT_GRAPH_BENCHMARK_VERSION,
        "model": training.architecture,
        "model_family": MODEL_FAMILIES[training.architecture],
        "anchor_reference_model": training.reference_mode,
        "protocol": benchmark.protocol,
        "information_setting": benchmark.information_setting,
        "endpoint": benchmark.endpoint,
        "endpoint_decision_status": benchmark.endpoint_decision_status,
        "phase": "selection_only",
        "phase_b_implementation": "absent_by_design_until_joint_model_freeze",
        "run_mode": (
            "engineering_smoke"
            if max_train_points is not None
            or len(selected_folds) < benchmark.folds
            or selected_seeds != training.seeds
            else "full_five_fold_engineering"
        ),
        "selected_folds": list(selected_folds),
        "selected_seeds": list(selected_seeds),
        "max_train_points": max_train_points,
        "training_config": training.to_dict(),
        "configuration_source": configuration_source,
        "capacity_control": capacity_control,
        "parameter_count": parameter_count,
        "parameter_match": {
            "target_morgan_parameter_count": training.parameter_match_target,
            "absolute_difference": parameter_count - training.parameter_match_target,
            "relative_gap": abs(parameter_count - training.parameter_match_target)
            / training.parameter_match_target,
            "within_frozen_band": True,
        },
        "graph_features": _graph_feature_metadata(training),
        "selection_contract": {
            "initial_candidate": "epoch_-1_exact_observed_anchor",
            "batch_unit": "complete_surfaces",
            "loss": "mean_of_within_surface_point_mse",
            "residual_shrinkage": training.residual_shrinkage,
            "optimizer": "AdamW",
            "learning_rate_schedule": "fixed",
            "early_stopping_metric": "validation_surface_macro_rmse",
            "test_access": "no_test_evaluation_entrypoint_and_test_shards_not_opened",
            "source_row_isolation": (
                "runtime_label_free_three_role_partition_verification_plus_"
                "train_validation_response_shard_disjoint_check"
            ),
        },
        "validation_gate": validation_gate,
        "split_manifest_hash": split.manifest["manifest_sha256"],
        "split_hash": split.split_hash,
        "context_manifest_hash": context.manifest["manifest_sha256"],
        "point_shard_manifest_hash": shards.manifest["manifest_sha256"],
        "source_partitions": [shards.source_partition_record(fold) for fold in selected_folds],
        "authorized_response_roles": ["train", "validation"],
        "test_response_rows_read": 0,
        "test_evaluations": 0,
        "jobs": jobs,
    }
    manifest["configuration_hash"] = _canonical_hash(
        {
            "benchmark": benchmark.to_dict(),
            "independent_graph": training.to_dict(),
            "capacity_control": capacity_control,
            "folds": list(selected_folds),
            "seeds": list(selected_seeds),
            "max_train_points": max_train_points,
            "split_hash": split.split_hash,
            "context_manifest_hash": context.manifest["manifest_sha256"],
            "point_shard_manifest_hash": shards.manifest["manifest_sha256"],
        }
    )
    manifest["manifest_sha256"] = _canonical_hash(manifest)
    _write_json_atomic(selection_run_path, manifest)
    return {**manifest, "selection_run_manifest": _file_record(selection_run_path)}


__all__ = [
    "BISURF_MODEL_SLUG",
    "GRAPH_CONFIG_SECTION",
    "MODEL_FAMILIES",
    "INDEPENDENT_GRAPH_BENCHMARK_VERSION",
    "MODEL_SLUG",
    "GraphFeatureBanks",
    "IndependentGraphConfig",
    "build_graph_feature_banks",
    "fit_independent_fold_seed",
    "load_independent_graph_config",
    "predict_graph_compact",
    "select_independent_graph_benchmark",
]
