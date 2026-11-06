from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class GraphItem:
    node_features: Tensor
    adjacency: Tensor

    def __post_init__(self) -> None:
        if self.node_features.ndim != 2:
            raise ValueError("node_features must have shape [nodes, features]")
        nodes = self.node_features.shape[0]
        if self.adjacency.shape != (nodes, nodes):
            raise ValueError("adjacency must have shape [nodes, nodes]")
        if nodes < 1:
            raise ValueError("A graph must contain at least one node")


@dataclass(frozen=True)
class GraphBatch:
    node_features: Tensor
    adjacency: Tensor
    node_mask: Tensor

    def to(self, device: torch.device | str) -> GraphBatch:
        return GraphBatch(
            node_features=self.node_features.to(device),
            adjacency=self.adjacency.to(device),
            node_mask=self.node_mask.to(device),
        )


@dataclass(frozen=True)
class SurfaceSample:
    drug_a: GraphItem
    drug_b: GraphItem
    cell: GraphItem
    doses: Tensor
    mono_response_a: Tensor
    mono_response_b: Tensor
    mono_mask_a: Tensor
    mono_mask_b: Tensor
    response: Tensor
    surface_id: int
    drug_a_id: int
    drug_b_id: int
    cell_id: int


@dataclass(frozen=True)
class SurfaceBatch:
    drug_a: GraphBatch
    drug_b: GraphBatch
    cell: GraphBatch
    doses: Tensor
    mono_response_a: Tensor
    mono_response_b: Tensor
    mono_mask_a: Tensor
    mono_mask_b: Tensor
    response: Tensor
    surface_id: Tensor
    drug_a_id: Tensor
    drug_b_id: Tensor
    cell_id: Tensor

    def to(self, device: torch.device | str) -> SurfaceBatch:
        return SurfaceBatch(
            drug_a=self.drug_a.to(device),
            drug_b=self.drug_b.to(device),
            cell=self.cell.to(device),
            doses=self.doses.to(device),
            mono_response_a=self.mono_response_a.to(device),
            mono_response_b=self.mono_response_b.to(device),
            mono_mask_a=self.mono_mask_a.to(device),
            mono_mask_b=self.mono_mask_b.to(device),
            response=self.response.to(device),
            surface_id=self.surface_id.to(device),
            drug_a_id=self.drug_a_id.to(device),
            drug_b_id=self.drug_b_id.to(device),
            cell_id=self.cell_id.to(device),
        )


def pad_graph_items(items: Sequence[GraphItem]) -> GraphBatch:
    if not items:
        raise ValueError("Cannot collate an empty graph collection")
    feature_dim = items[0].node_features.shape[-1]
    if any(item.node_features.shape[-1] != feature_dim for item in items):
        raise ValueError("All graphs in one modality must share feature dimension")

    batch_size = len(items)
    max_nodes = max(item.node_features.shape[0] for item in items)
    dtype = items[0].node_features.dtype
    node_features = torch.zeros(batch_size, max_nodes, feature_dim, dtype=dtype)
    adjacency = torch.zeros(batch_size, max_nodes, max_nodes, dtype=dtype)
    node_mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool)

    for index, item in enumerate(items):
        nodes = item.node_features.shape[0]
        node_features[index, :nodes] = item.node_features
        adjacency[index, :nodes, :nodes] = item.adjacency.to(dtype=dtype)
        node_mask[index, :nodes] = True
    return GraphBatch(node_features=node_features, adjacency=adjacency, node_mask=node_mask)


def collate_surface_samples(samples: Sequence[SurfaceSample]) -> SurfaceBatch:
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    return SurfaceBatch(
        drug_a=pad_graph_items([sample.drug_a for sample in samples]),
        drug_b=pad_graph_items([sample.drug_b for sample in samples]),
        cell=pad_graph_items([sample.cell for sample in samples]),
        doses=torch.stack([sample.doses for sample in samples]),
        mono_response_a=torch.stack([sample.mono_response_a for sample in samples]),
        mono_response_b=torch.stack([sample.mono_response_b for sample in samples]),
        mono_mask_a=torch.stack([sample.mono_mask_a for sample in samples]).bool(),
        mono_mask_b=torch.stack([sample.mono_mask_b for sample in samples]).bool(),
        response=torch.stack([sample.response for sample in samples]),
        surface_id=torch.tensor([sample.surface_id for sample in samples], dtype=torch.long),
        drug_a_id=torch.tensor([sample.drug_a_id for sample in samples], dtype=torch.long),
        drug_b_id=torch.tensor([sample.drug_b_id for sample in samples], dtype=torch.long),
        cell_id=torch.tensor([sample.cell_id for sample in samples], dtype=torch.long),
    )


def swap_drug_order(batch: SurfaceBatch) -> SurfaceBatch:
    """Swap both drugs and their aligned dose/monotherapy metadata."""

    return SurfaceBatch(
        drug_a=batch.drug_b,
        drug_b=batch.drug_a,
        cell=batch.cell,
        doses=batch.doses.flip(dims=(-1,)),
        mono_response_a=batch.mono_response_b,
        mono_response_b=batch.mono_response_a,
        mono_mask_a=batch.mono_mask_b,
        mono_mask_b=batch.mono_mask_a,
        response=batch.response,
        surface_id=batch.surface_id,
        drug_a_id=batch.drug_b_id,
        drug_b_id=batch.drug_a_id,
        cell_id=batch.cell_id,
    )
