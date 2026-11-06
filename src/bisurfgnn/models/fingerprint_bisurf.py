"""Molecular-fingerprint input experiment; no response-derived features.

The sigmoid two-input fusion follows the equation in Plus/6_1_Gated_Fusion.py.
The implementation is local and adds mask handling. This is an engineering
component, not a new attention mechanism or a claimed performance improvement.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from ..data.batch import GraphBatch, pad_graph_items
from .compact_bisurf import CompactBiSurfGNN
from .graph_encoder import masked_mean


@dataclass(frozen=True)
class FingerprintGraphBatch(GraphBatch):
    fingerprints: Tensor

    def to(self, device):
        return FingerprintGraphBatch(self.node_features.to(device), self.adjacency.to(device),
                                     self.node_mask.to(device), self.fingerprints.to(device))


def pad_fingerprint_graphs(items, fingerprints):
    plain = pad_graph_items(items)
    if fingerprints.ndim != 2 or fingerprints.shape[0] != len(items):
        raise ValueError("Fingerprint rows must align with the supplied graph order")
    if not bool(torch.isfinite(fingerprints).all()):
        raise ValueError("Fingerprints must be finite")
    return FingerprintGraphBatch(plain.node_features, plain.adjacency, plain.node_mask, fingerprints)


class SharedInputGate(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.graph_projection = nn.Linear(width, width)
        self.feature_projection = nn.Linear(width, width)

    def forward(self, tokens, features, mask):
        features = features[:, None, :]
        gate = torch.sigmoid(self.graph_projection(tokens) + self.feature_projection(features))
        return (gate * tokens + (1 - gate) * features) * mask[..., None]


class FingerprintBiSurfGNN(CompactBiSurfGNN):
    MODES = ("fingerprint_only", "fused", "graph_summary_control")

    def __init__(self, atom_feature_dim, cell_feature_dim, hidden_dim, *, mode,
                 fingerprint_dim=512, graph_layers=2, attention_heads=4, dropout=.1,
                 monotherapy_upper_bound=3., residual_monotherapy_input=True):
        if mode not in self.MODES or fingerprint_dim < hidden_dim:
            raise ValueError("Unknown fingerprint mode or insufficient fingerprint width")
        # Construct the identical base first to align all common initial weights.
        super().__init__(atom_feature_dim, cell_feature_dim, hidden_dim,
                         graph_layers=graph_layers, attention_heads=attention_heads,
                         dropout=dropout, monotherapy_upper_bound=monotherapy_upper_bound,
                         residual_monotherapy_input=residual_monotherapy_input)
        self.fingerprint_mode, self.fingerprint_dim = mode, fingerprint_dim
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(torch.initial_seed() + 15401)
            self.feature_encoder = nn.Sequential(nn.Linear(fingerprint_dim, hidden_dim),
                nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim))
            if mode != "fingerprint_only":
                self.input_gate = SharedInputGate(hidden_dim)
        if mode == "fingerprint_only":
            self.drug_encoder.requires_grad_(False)
        else:
            self.drug_encoder.pool_projection.requires_grad_(False)
        self.interaction_decoder.log_variance_head.requires_grad_(False)

    def encode_drugs(self, graphs):
        if not isinstance(graphs, FingerprintGraphBatch):
            raise ValueError("Explicit aligned fingerprints are required")
        if graphs.fingerprints.shape != (graphs.node_mask.shape[0], self.fingerprint_dim):
            raise ValueError("Fingerprint width or graph-row alignment differs")
        if not bool(graphs.node_mask.any(1).all()) or not bool(torch.isfinite(graphs.fingerprints).all()):
            raise ValueError("Graphs must be nonempty and fingerprints finite")
        if self.fingerprint_mode == "fingerprint_only":
            # One token per molecule: atom count/connectivity are not input here.
            features = self.feature_encoder(graphs.fingerprints)
            return features[:, None, :], torch.ones((len(features), 1), dtype=torch.bool, device=features.device)
        tokens = self.drug_encoder(graphs).tokens
        if self.fingerprint_mode == "fused":
            source = graphs.fingerprints
        else:
            # Same learned branch/gate, only information already in the GNN.
            # Repeating the summary to 512 dims adds no structural information.
            summary = masked_mean(tokens, graphs.node_mask)
            source = summary.repeat(1, math.ceil(self.fingerprint_dim / self.hidden_dim))[:, :self.fingerprint_dim]
        features = self.feature_encoder(source)
        return self.input_gate(tokens, features, graphs.node_mask), graphs.node_mask
