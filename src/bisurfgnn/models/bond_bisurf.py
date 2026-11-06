"""Minimal bond-weighted messages over the existing BiSurf molecular encoder."""
from __future__ import annotations

import torch
from torch import nn

from ..data.bond_graphs import BondGraphBatch
from ..data.chemistry import BOND_FEATURE_DIM
from .graph_encoder import GraphEncoder, GraphEncoding, masked_mean, masked_max

MODES = ("full_bonds", "uniform_bonds", "node_control")


class BondGraphEncoder(GraphEncoder):
    """Same base blocks plus a scalar gate per molecular edge (34 parameters).

    Original degree normalisation is retained; gates modulate messages rather
    than change topology. The node-only control uses latent neighbour channels
    instead of chemical edge information, with exactly the same gate capacity.
    """

    def __init__(self, input_dim, hidden_dim, layers, dropout=0.0, mode="full_bonds"):
        if mode not in MODES or hidden_dim < BOND_FEATURE_DIM+1:
            raise ValueError("Invalid bond mode or insufficient latent width")
        super().__init__(input_dim, hidden_dim, layers, dropout)
        self.mode = mode
        self.bond_gates = nn.ModuleList(nn.Linear(BOND_FEATURE_DIM+1, 1) for _ in range(layers))
        for gate in self.bond_gates:
            nn.init.zeros_(gate.weight); nn.init.zeros_(gate.bias)

    def forward(self, graph: BondGraphBatch):
        features, adjacency, mask = graph.node_features, graph.adjacency, graph.node_mask
        if features.shape[:2] != mask.shape or adjacency.shape != (*mask.shape, mask.shape[1]):
            raise ValueError("Invalid bond batch shape")
        if graph.bond_features.shape != (*adjacency.shape, BOND_FEATURE_DIM+1):
            raise ValueError("Invalid bond feature batch shape")
        if not bool(mask.any(dim=1).all()):
            raise ValueError("Empty molecular graph")
        valid = mask.unsqueeze(1) & mask.unsqueeze(2)
        adjacency = adjacency*valid.to(adjacency.dtype)
        tokens = self.input_projection(features)*mask.unsqueeze(-1)
        edge_inputs = graph.bond_features
        if self.mode == "uniform_bonds":
            edge_inputs = torch.zeros_like(edge_inputs)
            edge_inputs[..., 0] = adjacency*(1-graph.bond_features[..., -1])
            edge_inputs[..., -1] = graph.bond_features[..., -1]
        for block, gate in zip(self.blocks, self.bond_gates, strict=True):
            normalized = block.pre_norm(tokens)
            degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1)
            if self.mode == "node_control":
                weights = 1+0.5*torch.tanh(gate(normalized[..., :BOND_FEATURE_DIM+1]))
                neighborhood = torch.bmm(adjacency, normalized*weights)/degree
            else:
                weights = 1+0.5*torch.tanh(gate(edge_inputs).squeeze(-1))
                neighborhood = torch.bmm(adjacency*weights, normalized)/degree
            tokens = tokens+block.message_dropout(block.message(neighborhood))
            tokens = tokens+block.ffn(block.ffn_norm(tokens))
            tokens = tokens*mask.unsqueeze(-1)
        tokens = self.output_norm(tokens)*mask.unsqueeze(-1)
        pooled = self.pool_projection(torch.cat([masked_mean(tokens, mask), masked_max(tokens, mask)], -1))
        return GraphEncoding(tokens=tokens, pooled=pooled)
