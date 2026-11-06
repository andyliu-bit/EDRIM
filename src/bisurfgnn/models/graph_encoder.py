from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..data.batch import GraphBatch


def masked_mean(tokens: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(dtype=tokens.dtype).unsqueeze(-1)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    return (tokens * weights).sum(dim=1) / denominator


def masked_max(tokens: Tensor, mask: Tensor) -> Tensor:
    masked = tokens.masked_fill(~mask.unsqueeze(-1), torch.finfo(tokens.dtype).min)
    values = masked.max(dim=1).values
    return torch.where(torch.isfinite(values), values, torch.zeros_like(values))


@dataclass(frozen=True)
class GraphEncoding:
    tokens: Tensor
    pooled: Tensor


class GraphMessageBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.pre_norm = nn.LayerNorm(hidden_dim)
        self.message = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.message_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: Tensor, adjacency: Tensor, mask: Tensor) -> Tensor:
        normalized = self.pre_norm(tokens)
        degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
        neighborhood = torch.bmm(adjacency, normalized) / degree
        tokens = tokens + self.message_dropout(self.message(neighborhood))
        tokens = tokens + self.ffn(self.ffn_norm(tokens))
        return tokens * mask.unsqueeze(-1).to(dtype=tokens.dtype)


class GraphEncoder(nn.Module):
    """Pure-PyTorch encoder for padded dense molecular/pathway graphs."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("GraphEncoder requires at least one message-passing layer")
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            GraphMessageBlock(hidden_dim, dropout) for _ in range(layers)
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.pool_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, graph: GraphBatch) -> GraphEncoding:
        features = graph.node_features
        adjacency = graph.adjacency
        mask = graph.node_mask
        if features.ndim != 3 or adjacency.ndim != 3 or mask.ndim != 2:
            raise ValueError("GraphBatch shapes must be [B,N,F], [B,N,N], and [B,N]")
        if features.shape[:2] != mask.shape:
            raise ValueError("Node feature and mask shapes do not agree")
        if adjacency.shape != (features.shape[0], features.shape[1], features.shape[1]):
            raise ValueError("Adjacency shape does not agree with node features")
        if not bool(mask.any(dim=1).all()):
            raise ValueError("Every graph must contain at least one valid node")

        valid_pairs = mask.unsqueeze(1) & mask.unsqueeze(2)
        adjacency = adjacency * valid_pairs.to(dtype=adjacency.dtype)
        tokens = self.input_projection(features)
        tokens = tokens * mask.unsqueeze(-1).to(dtype=tokens.dtype)
        for block in self.blocks:
            tokens = block(tokens, adjacency, mask)
        tokens = self.output_norm(tokens)
        tokens = tokens * mask.unsqueeze(-1).to(dtype=tokens.dtype)
        pooled = self.pool_projection(
            torch.cat([masked_mean(tokens, mask), masked_max(tokens, mask)], dim=-1)
        )
        return GraphEncoding(tokens=tokens, pooled=pooled)

