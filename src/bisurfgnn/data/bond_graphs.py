"""Molecular-only edge interface; existing pathway/graph interfaces stay intact."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from .batch import pad_graph_items, GraphItem
from .chemistry import BOND_FEATURE_DIM


@dataclass(frozen=True)
class BondGraphItem:
    node_features: Tensor
    adjacency: Tensor
    edge_index: Tensor
    edge_features: Tensor

    def __post_init__(self):
        n = len(self.node_features)
        if self.node_features.ndim != 2 or n < 1 or self.adjacency.shape != (n, n):
            raise ValueError("Invalid molecular graph shapes")
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2 or self.edge_index.dtype != torch.long:
            raise ValueError("Edge indices must be int64 [2,E]")
        if self.edge_features.shape != (self.edge_index.shape[1], BOND_FEATURE_DIM):
            raise ValueError("Frozen bond schema mismatch")
        if any(not bool(torch.isfinite(t).all()) for t in (self.node_features, self.adjacency, self.edge_features)):
            raise ValueError("Non-finite molecular features")
        if not torch.equal(self.adjacency, self.adjacency.T) or not bool((self.adjacency.diag() == 1).all()):
            raise ValueError("Molecular adjacency must be symmetric with self loops")
        observed = torch.zeros_like(self.adjacency)
        edge_lookup = {}
        for index, (a, b) in enumerate(self.edge_index.T.tolist()):
            if not (0 <= a < n and 0 <= b < n) or a == b or (a, b) in edge_lookup:
                raise ValueError("Invalid or duplicated chemical edge")
            observed[a, b] = 1
            edge_lookup[(a, b)] = self.edge_features[index]
        if not torch.equal(observed + torch.eye(n, dtype=observed.dtype), self.adjacency):
            raise ValueError("Edge records do not match molecular adjacency")
        for (a, b), features in edge_lookup.items():
            if (b, a) not in edge_lookup or not torch.equal(features, edge_lookup[(b, a)]):
                raise ValueError("Reverse bond records must agree")


@dataclass(frozen=True)
class BondGraphBatch:
    node_features: Tensor
    adjacency: Tensor
    node_mask: Tensor
    bond_features: Tensor

    def to(self, device):
        return BondGraphBatch(*(v.to(device) for v in (
            self.node_features, self.adjacency, self.node_mask, self.bond_features)))


def pad_bond_graphs(items):
    base = pad_graph_items([GraphItem(g.node_features, g.adjacency) for g in items])
    b, n = base.node_mask.shape
    features = torch.zeros(b, n, n, BOND_FEATURE_DIM+1, dtype=base.node_features.dtype)
    for index, graph in enumerate(items):
        a, c = graph.edge_index
        features[index, a, c, :BOND_FEATURE_DIM] = graph.edge_features
        ids = torch.arange(len(graph.node_features))
        features[index, ids, ids, -1] = 1  # Explicit self-loop indicator, not a chemical bond.
    return BondGraphBatch(base.node_features, base.adjacency, base.node_mask, features)


def load_bond_graph_bank(banks, structures_path: Path):
    cache = torch.load(structures_path, weights_only=True, map_location="cpu")
    result = []
    for key, old in zip(banks.drug_structure_keys, banks.drug_graphs, strict=True):
        payload = cache[key]
        if "edge_index" not in payload or "edge_features" not in payload:
            raise ValueError(f"Frozen cache lacks bond fields: {key}")
        graph = BondGraphItem(payload["node_features"].float().cpu(), payload["adjacency"].float().cpu(),
            payload["edge_index"].long().cpu(), payload["edge_features"].float().cpu())
        if not torch.equal(graph.node_features, old.node_features) or not torch.equal(graph.adjacency, old.adjacency):
            raise ValueError("Bond interface changed existing atoms or adjacency")
        result.append(graph)
    return tuple(result)
