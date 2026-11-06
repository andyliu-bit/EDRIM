from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Collection, Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ..config import DataConfig, ModelConfig
from ..losses import BiSurfLoss
from ..models import BiSurfGNN
from .batch import GraphItem, SurfaceSample, collate_surface_samples, swap_drug_order
from .cellminer import (
    CellMinerConfig,
    audit_train_only_invariance,
    cache_split_cell_graphs,
    fit_train_only_pathway_transform,
    load_cached_pathway_topology,
    load_cached_split_cell_graphs,
    load_cell_alias_rules,
    load_hallmark_gene_sets,
    load_prepared_cell_mapping,
    load_prepared_expression,
    normalize_cell_label,
)
from .nci_almanac import NCIAlmanacConfig
from .splits import split_biological_cell_groups


class PreparedNCISurfaceDataset(Dataset[SurfaceSample]):
    """In-memory selection of complete prepared NCI-ALMANAC surfaces.

    This v0.1 adapter intentionally requires an explicit surface limit. Full-scale
    training will use a sharded/streaming sampler in a later milestone.
    """

    def __init__(
        self,
        processed_dir: str | Path,
        *,
        max_surfaces: int,
        seed: int,
        cell_graphs: Mapping[str, GraphItem],
        cell_mapping: str | Path | pd.DataFrame,
        allowed_biological_cells: Collection[str],
        require_complete_monotherapy: bool = True,
        ensure_allowed_cell_coverage: bool = False,
    ) -> None:
        if max_surfaces < 1:
            raise ValueError("max_surfaces must be positive")
        self.processed_dir = Path(processed_dir)
        mapping = json.loads(
            (self.processed_dir / "drug_id_to_structure_key.json").read_text(
                encoding="utf-8"
            )
        )
        graph_cache = torch.load(
            self.processed_dir / "structures.pt", map_location="cpu", weights_only=True
        )
        self.drug_graphs = {
            drug_id: GraphItem(
                node_features=graph_cache[structure_key]["node_features"].float(),
                adjacency=graph_cache[structure_key]["adjacency"].float(),
            )
            for drug_id, structure_key in mapping.items()
        }

        if not cell_graphs:
            raise ValueError("A real CellMiner pathway graph cache is required")
        self.cell_graphs: dict[str, GraphItem] = {}
        for raw_cell_id, graph in cell_graphs.items():
            cell_id = normalize_cell_label(raw_cell_id)
            if cell_id in self.cell_graphs:
                raise ValueError(f"Duplicate cell graph after normalization: {cell_id}")
            self.cell_graphs[cell_id] = graph
        first_cell_graph = next(iter(self.cell_graphs.values()))
        cell_shape = (
            first_cell_graph.node_features.shape[0],
            first_cell_graph.node_features.shape[1],
        )
        if any(
            tuple(graph.node_features.shape) != cell_shape
            for graph in self.cell_graphs.values()
        ):
            raise ValueError("All cached cell pathway graphs must share one shape")
        if any(
            not bool(torch.isfinite(graph.node_features).all())
            or not bool(torch.isfinite(graph.adjacency).all())
            for graph in self.cell_graphs.values()
        ):
            raise ValueError("Cached cell pathway graphs must be finite")
        if any(
            not torch.equal(graph.adjacency, first_cell_graph.adjacency)
            for graph in self.cell_graphs.values()
        ):
            raise ValueError("All cached cell pathway graphs must share one topology")
        if isinstance(cell_mapping, pd.DataFrame):
            mapping_frame = cell_mapping.copy()
        else:
            mapping_frame = pd.read_csv(cell_mapping, dtype=str, keep_default_na=False)
        required_mapping_columns = {"source_cell_id", "biological_cell_id", "status"}
        if not required_mapping_columns.issubset(mapping_frame.columns):
            missing = sorted(required_mapping_columns - set(mapping_frame.columns))
            raise ValueError(f"Cell mapping is missing columns: {missing}")
        mapping_frame["status"] = mapping_frame["status"].astype(str).str.strip()
        valid_statuses = {"mapped", "excluded", "expression_only"}
        unexpected_statuses = sorted(set(mapping_frame["status"]) - valid_statuses)
        if unexpected_statuses:
            raise ValueError(f"Cell mapping contains invalid statuses: {unexpected_statuses}")
        mapping_frame["biological_cell_id"] = mapping_frame[
            "biological_cell_id"
        ].map(normalize_cell_label)
        source_rows = mapping_frame[
            mapping_frame["status"].isin({"mapped", "excluded"})
        ].copy()
        if source_rows["source_cell_id"].astype(str).str.strip().eq("").any():
            raise ValueError("Mapped/excluded cell rows require ALMANAC source IDs")
        if source_rows["source_cell_id"].duplicated().any():
            raise ValueError("Cell mapping contains duplicate ALMANAC source IDs")
        mapped = mapping_frame[mapping_frame["status"] == "mapped"].copy()
        self.source_to_biological = dict(
            zip(mapped["source_cell_id"], mapped["biological_cell_id"], strict=True)
        )
        missing_graphs = sorted(
            set(self.source_to_biological.values()) - set(self.cell_graphs)
        )
        if missing_graphs:
            raise ValueError(f"Mapped biological cells have no pathway graph: {missing_graphs}")
        allowed = {normalize_cell_label(cell_id) for cell_id in allowed_biological_cells}
        mapped_biological = set(self.source_to_biological.values())
        unknown_allowed = sorted(allowed - mapped_biological)
        if unknown_allowed:
            raise ValueError(f"Allowed cells are not mapped ALMANAC cells: {unknown_allowed}")
        missing_allowed_graphs = sorted(allowed - set(self.cell_graphs))
        if missing_allowed_graphs:
            raise ValueError(
                f"Allowed cells have no pathway graph: {missing_allowed_graphs}"
            )
        if not allowed:
            raise ValueError("At least one biological cell must be allowed")

        surface_index = pd.read_parquet(self.processed_dir / "surface_index.parquet")
        surface_index["cell_id"] = surface_index["cell_id"].astype(str)
        represented_source_ids = set(source_rows["source_cell_id"].astype(str))
        unknown_surface_cells = sorted(
            set(surface_index["cell_id"]) - represented_source_ids
        )
        if unknown_surface_cells:
            raise ValueError(
                "Prepared surface index contains cells absent from the frozen mapping: "
                f"{unknown_surface_cells}"
            )
        eligible = surface_index[
            surface_index["drug_a_id"].isin(self.drug_graphs)
            & surface_index["drug_b_id"].isin(self.drug_graphs)
        ].copy()
        eligible["biological_cell_id"] = eligible["cell_id"].map(
            self.source_to_biological
        )
        eligible = eligible[eligible["biological_cell_id"].isin(allowed)]
        if require_complete_monotherapy:
            eligible = eligible[eligible["mono_complete_fraction"] == 1.0]
        eligible = eligible[eligible["primary_points"] > 0]
        if eligible.empty:
            raise ValueError("No eligible prepared surfaces remain after structure/mono filters")
        eligible["surface_replicate_id"] = eligible["surface_replicate_id"].astype(str)
        if eligible["surface_replicate_id"].duplicated().any():
            raise ValueError("Eligible surface index contains duplicate replicate IDs")
        eligible.sort_values("surface_replicate_id", inplace=True)
        generator = np.random.default_rng(seed)
        selected_count = min(max_surfaces, len(eligible))
        if ensure_allowed_cell_coverage:
            available_cells = set(eligible["biological_cell_id"].astype(str))
            missing_eligible_cells = sorted(allowed - available_cells)
            if missing_eligible_cells:
                raise ValueError(
                    "Allowed cells have no eligible complete surfaces: "
                    f"{missing_eligible_cells}"
                )
            if max_surfaces < len(allowed):
                raise ValueError(
                    "max_surfaces must be at least the number of allowed cells when "
                    "coverage is required"
                )
            selected_ids = []
            for cell_id in sorted(allowed):
                candidates = eligible.loc[
                    eligible["biological_cell_id"] == cell_id,
                    "surface_replicate_id",
                ].tolist()
                selected_ids.append(candidates[int(generator.integers(len(candidates)))])
            remaining = sorted(
                set(eligible["surface_replicate_id"].tolist()) - set(selected_ids)
            )
            fill_count = selected_count - len(selected_ids)
            if fill_count:
                selected_ids.extend(
                    generator.choice(remaining, size=fill_count, replace=False).tolist()
                )
        else:
            selected_ids = generator.choice(
                eligible["surface_replicate_id"].to_numpy(),
                size=selected_count,
                replace=False,
            ).tolist()
        points = pd.read_parquet(
            self.processed_dir / "points.parquet",
            filters=[("surface_replicate_id", "in", selected_ids)],
        )
        points = points[points["surface_replicate_id"].isin(selected_ids)].copy()
        points = points[
            points["primary_metric_mask"].astype(bool)
            & points["response_mask"].astype(bool)
            & points["response"].notna()
        ].copy()
        points.sort_values(
            ["surface_replicate_id", "dose_a_M", "dose_b_M", "source_combodrugseq"],
            inplace=True,
        )
        if points.empty:
            raise RuntimeError("Parquet filter returned no selected NCI points")

        points["cell_id"] = points["cell_id"].astype(str)
        points["biological_cell_id"] = points["cell_id"].map(self.source_to_biological)
        if points["biological_cell_id"].isna().any():
            missing_cells = sorted(
                points.loc[points["biological_cell_id"].isna(), "cell_id"]
                .astype(str)
                .unique()
            )
            raise ValueError(f"Prepared point table contains unmapped cells: {missing_cells}")
        if not set(points["biological_cell_id"]).issubset(allowed):
            raise AssertionError("Selected points crossed the allowed CellOut partition")
        biological_ids = sorted(allowed)
        self.cell_to_index = {
            cell_id: index for index, cell_id in enumerate(biological_ids)
        }
        surface_ids = sorted(points["surface_replicate_id"].astype(str).unique().tolist())
        surface_to_index = {surface_id: index for index, surface_id in enumerate(surface_ids)}
        self.selected_surface_ids = tuple(surface_ids)
        self.samples: list[SurfaceSample] = []
        for row in points.itertuples(index=False):
            mono_a_mask = bool(row.mono_mask_a) and pd.notna(row.mono_response_a)
            mono_b_mask = bool(row.mono_mask_b) and pd.notna(row.mono_response_b)
            self.samples.append(
                SurfaceSample(
                    drug_a=self.drug_graphs[str(row.drug_a_id)],
                    drug_b=self.drug_graphs[str(row.drug_b_id)],
                    cell=self.cell_graphs[str(row.biological_cell_id)],
                    doses=torch.tensor(
                        [row.dose_a_model, row.dose_b_model], dtype=torch.float32
                    ),
                    mono_response_a=torch.tensor(
                        float(row.mono_response_a) if mono_a_mask else 0.0,
                        dtype=torch.float32,
                    ),
                    mono_response_b=torch.tensor(
                        float(row.mono_response_b) if mono_b_mask else 0.0,
                        dtype=torch.float32,
                    ),
                    mono_mask_a=torch.tensor(mono_a_mask),
                    mono_mask_b=torch.tensor(mono_b_mask),
                    response=torch.tensor(float(row.response), dtype=torch.float32),
                    surface_id=surface_to_index[str(row.surface_replicate_id)],
                    drug_a_id=int(str(row.drug_a_id).removeprefix("S")),
                    drug_b_id=int(str(row.drug_b_id).removeprefix("S")),
                    cell_id=self.cell_to_index[str(row.biological_cell_id)],
                )
            )
        first_graph = next(iter(self.drug_graphs.values()))
        self.atom_feature_dim = int(first_graph.node_features.shape[-1])
        self.cell_feature_dim = int(first_cell_graph.node_features.shape[-1])
        self.pathway_nodes = int(first_cell_graph.node_features.shape[0])
        self.num_surfaces = len(surface_ids)
        self.num_points = len(self.samples)
        self.num_cells = len(allowed)
        self.selected_biological_cells = tuple(
            sorted(points["biological_cell_id"].astype(str).unique())
        )
        if ensure_allowed_cell_coverage and set(self.selected_biological_cells) != allowed:
            raise AssertionError("Selected surfaces do not cover every allowed biological cell")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> SurfaceSample:
        return self.samples[index]

    def model_data_config(self) -> DataConfig:
        atom_counts = [graph.node_features.shape[0] for graph in self.drug_graphs.values()]
        return DataConfig(
            kind="nci_almanac_prepared",
            num_drugs=len(self.drug_graphs),
            num_cells=self.num_cells,
            num_surfaces=self.num_surfaces,
            dose_levels=(0.0, 1.0),
            atom_feature_dim=self.atom_feature_dim,
            cell_feature_dim=self.cell_feature_dim,
            min_atoms=min(atom_counts),
            max_atoms=max(atom_counts),
            pathway_nodes=self.pathway_nodes,
            noise_std=0.0,
        )


def _mapping_attrition(
    nci_processed_dir: Path,
    mapping: pd.DataFrame,
) -> dict[str, Any]:
    mapped_source_ids = set(
        mapping.loc[mapping["status"] == "mapped", "source_cell_id"].astype(str)
    )
    excluded_source_ids = set(
        mapping.loc[mapping["status"] == "excluded", "source_cell_id"].astype(str)
    )
    surface_index = pd.read_parquet(nci_processed_dir / "surface_index.parquet")
    surface_index["cell_id"] = surface_index["cell_id"].astype(str)
    unknown = sorted(
        set(surface_index["cell_id"]) - mapped_source_ids - excluded_source_ids
    )
    if unknown:
        raise ValueError(f"Attrition cannot classify unmapped ALMANAC cells: {unknown}")
    excluded = surface_index[surface_index["cell_id"].isin(excluded_source_ids)]
    total_points = int(surface_index["point_count"].sum())
    total_primary = int(surface_index["primary_points"].sum())
    return {
        "excluded_combination_points": int(excluded["point_count"].sum()),
        "excluded_primary_points": int(excluded["primary_points"].sum()),
        "excluded_surface_replicates": len(excluded),
        "excluded_prediction_groups": int(excluded["prediction_group_id"].nunique()),
        "excluded_point_fraction": float(excluded["point_count"].sum() / total_points),
        "excluded_primary_fraction": float(excluded["primary_points"].sum() / total_primary),
    }


def run_real_data_smoke(
    nci_config: NCIAlmanacConfig,
    cell_config: CellMinerConfig,
    smoke: Mapping[str, Any],
) -> dict[str, Any]:
    seed = int(smoke.get("seed", 23))
    torch.manual_seed(seed)

    expression = load_prepared_expression(cell_config)
    catalog = load_hallmark_gene_sets(cell_config)
    topology = load_cached_pathway_topology(
        cell_config, expected_catalog_hash=catalog.catalog_hash
    )
    almanac_cells = nci_config.processed_dir / "cells.csv"
    mapping = load_prepared_cell_mapping(cell_config, almanac_cells)
    biological_cells = tuple(
        sorted(
            mapping.loc[mapping["status"] == "mapped", "biological_cell_id"].unique()
        )
    )
    alias_rules = load_cell_alias_rules(cell_config.cell_aliases)
    lineage_groups = {
        str(cell_id): str(group_id)
        for cell_id, group_id in alias_rules.get("lineage_groups", {}).items()
    }
    split = split_biological_cell_groups(
        biological_cells,
        group_by_cell=lineage_groups,
        validation_groups=int(smoke.get("validation_groups", 6)),
        test_groups=int(smoke.get("test_groups", 6)),
        seed=seed,
    )
    state = fit_train_only_pathway_transform(
        expression,
        catalog,
        topology,
        split.train_cells,
        split.split_hash,
        min_pathway_genes=cell_config.min_pathway_genes,
        clip_value=cell_config.clip_value,
        std_epsilon=cell_config.std_epsilon,
    )
    leakage_audit = audit_train_only_invariance(
        expression,
        catalog,
        topology,
        split.train_cells,
        split.validation_cells + split.test_cells,
        split.split_hash,
        min_pathway_genes=cell_config.min_pathway_genes,
        clip_value=cell_config.clip_value,
        std_epsilon=cell_config.std_epsilon,
    )
    graph_cache = cache_split_cell_graphs(
        cell_config,
        expression,
        topology,
        state,
        biological_cells,
        expected_split_hash=split.split_hash,
    )
    split_path = Path(graph_cache["directory"]) / "split_manifest.json"
    split_path.write_text(
        json.dumps(split.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    loaded_state, cell_graphs, loaded_graph_manifest = load_cached_split_cell_graphs(
        cell_config,
        split.split_hash,
        expression=expression,
        topology=topology,
        expected_fit_cell_ids=split.train_cells,
    )
    if loaded_state.state_hash != state.state_hash:
        raise AssertionError("Serialized pathway transform differs from fitted state")

    train_dataset = PreparedNCISurfaceDataset(
        nci_config.processed_dir,
        max_surfaces=int(smoke.get("max_train_surfaces", 24)),
        seed=seed,
        cell_graphs=cell_graphs,
        cell_mapping=mapping,
        allowed_biological_cells=split.train_cells,
        require_complete_monotherapy=True,
    )
    test_dataset = PreparedNCISurfaceDataset(
        nci_config.processed_dir,
        max_surfaces=int(smoke.get("max_test_surfaces", 12)),
        seed=seed + 1,
        cell_graphs=cell_graphs,
        cell_mapping=mapping,
        allowed_biological_cells=split.test_cells,
        require_complete_monotherapy=True,
        ensure_allowed_cell_coverage=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(32, len(train_dataset)),
        shuffle=False,
        collate_fn=collate_surface_samples,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=min(32, len(test_dataset)),
        shuffle=False,
        collate_fn=collate_surface_samples,
    )
    train_batch = next(iter(train_loader))
    model_config = ModelConfig(
        hidden_dim=int(smoke.get("hidden_dim", 32)),
        graph_layers=int(smoke.get("graph_layers", 2)),
        attention_heads=int(smoke.get("attention_heads", 4)),
        dropout=0.0,
        dose_frequencies=int(smoke.get("dose_frequencies", 4)),
    )
    model = BiSurfGNN(train_dataset.model_data_config(), model_config)
    criterion = BiSurfLoss(monotherapy_weight=0.25, variance_weight=0.02)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    index_to_train_cell = {
        index: cell_id for cell_id, index in train_dataset.cell_to_index.items()
    }
    optimizer_cells = tuple(
        sorted({index_to_train_cell[int(index)] for index in train_batch.cell_id.tolist()})
    )
    optimizer_test_overlap = tuple(sorted(set(optimizer_cells) & set(split.test_cells)))
    if optimizer_test_overlap:
        raise AssertionError("Held-out cells entered the optimizer batch")

    model.eval()
    with torch.no_grad():
        train_eval = model(train_batch)
        train_swapped = model(swap_drug_order(train_batch))
        train_symmetry_error = float(
            (train_eval.prediction - train_swapped.prediction).abs().max()
        )
        heldout_predictions: list[torch.Tensor] = []
        test_symmetry_error = 0.0
        test_batches_evaluated = 0
        test_points_evaluated = 0
        for test_batch in test_loader:
            test_eval = model(test_batch)
            test_swapped = model(swap_drug_order(test_batch))
            heldout_predictions.append(test_eval.prediction.detach())
            test_symmetry_error = max(
                test_symmetry_error,
                float((test_eval.prediction - test_swapped.prediction).abs().max()),
            )
            test_batches_evaluated += 1
            test_points_evaluated += int(test_batch.response.shape[0])
        all_heldout_predictions = torch.cat(heldout_predictions)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(train_batch)
    loss = criterion(output, train_batch)
    loss.total.backward()
    gradients = [
        parameter.grad.detach()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    gradients_finite = bool(gradients) and all(
        bool(torch.isfinite(gradient).all()) for gradient in gradients
    )
    gradient_l1 = float(sum(gradient.abs().sum() for gradient in gradients))
    optimizer_step_completed = False
    optimizer.step()
    optimizer_step_completed = True

    heldout_cells_evaluated = tuple(test_dataset.selected_biological_cells)
    all_test_cells_evaluated = set(heldout_cells_evaluated) == set(split.test_cells)
    if not all_test_cells_evaluated:
        raise AssertionError("Not every held-out CellOut cell was evaluated")
    lineage_grouped = len(set(lineage_groups.values())) < len(lineage_groups)
    report = {
        "status": "completed",
        "cell_features": "cellminer_log2_fpkm_hallmark_pathway_graph",
        "split": split.to_dict(),
        "lineage_grouped": lineage_grouped,
        "train_fit_cells": len(state.fit_cell_ids),
        "mapped_biological_cells": len(biological_cells),
        "train_selected_surfaces": train_dataset.num_surfaces,
        "train_selected_points": train_dataset.num_points,
        "test_selected_surfaces": test_dataset.num_surfaces,
        "test_selected_points": test_dataset.num_points,
        "train_batch_points": int(train_batch.response.shape[0]),
        "test_batches_evaluated": test_batches_evaluated,
        "test_points_evaluated": test_points_evaluated,
        "heldout_cells_evaluated": list(heldout_cells_evaluated),
        "heldout_cell_count_evaluated": len(heldout_cells_evaluated),
        "all_test_cells_evaluated": all_test_cells_evaluated,
        "atom_feature_dim": train_dataset.atom_feature_dim,
        "pathway_nodes": train_dataset.pathway_nodes,
        "cell_feature_dim": train_dataset.cell_feature_dim,
        "loss": loss.detached(),
        "train_symmetry_max_abs_error": train_symmetry_error,
        "test_symmetry_max_abs_error": test_symmetry_error,
        "heldout_predictions_finite": bool(torch.isfinite(all_heldout_predictions).all()),
        "heldout_prediction_min": float(all_heldout_predictions.min()),
        "heldout_prediction_max": float(all_heldout_predictions.max()),
        "train_target_min": float(train_batch.response.min()),
        "train_target_max": float(train_batch.response.max()),
        "train_mono_mask_a_fraction": float(train_batch.mono_mask_a.float().mean()),
        "train_mono_mask_b_fraction": float(train_batch.mono_mask_b.float().mean()),
        "gradients_finite": gradients_finite,
        "gradient_l1": gradient_l1,
        "optimizer_cells": list(optimizer_cells),
        "optimizer_test_cell_overlap": list(optimizer_test_overlap),
        "optimizer_step_completed_on_train_cells_only": optimizer_step_completed
        and not optimizer_test_overlap
        and set(optimizer_cells) <= set(split.train_cells),
        "heldout_cells_used_for_optimizer": bool(optimizer_test_overlap),
        "leakage_counterfactual_audit": leakage_audit,
        "graph_cache_audit": graph_cache["audit"],
        "graph_cache_roundtrip_verified": loaded_graph_manifest["split_hash"]
        == split.split_hash,
        "mapping_attrition": _mapping_attrition(nci_config.processed_dir, mapping),
    }
    output_path = cell_config.processed_dir / "cellout_smoke.json"
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
