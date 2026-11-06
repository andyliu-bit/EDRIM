from __future__ import annotations

import torch
from torch import Tensor, nn


class ContextGatedFusion(nn.Module):
    """Condition an order-invariant drug-pair representation on cell context."""

    def __init__(self, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.pair_norm = nn.LayerNorm(hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.pair_projection = nn.Linear(hidden_dim, hidden_dim)
        self.context_projection = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        nn.init.constant_(self.gate[-1].bias, -1.0)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_dropout = nn.Dropout(dropout)

    def forward(
        self,
        pair: Tensor,
        context: Tensor,
        return_gate: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        if pair.ndim != 2 or context.ndim != 2 or pair.shape != context.shape:
            raise ValueError("pair and context must have the same [B,D] shape")
        pair_projected = self.pair_projection(self.pair_norm(pair))
        context_projected = self.context_projection(self.context_norm(context))
        gate_input = torch.cat(
            [
                pair_projected,
                context_projected,
                pair_projected * context_projected,
                torch.abs(pair_projected - context_projected),
            ],
            dim=-1,
        )
        gate = torch.sigmoid(self.gate(gate_input))
        fused = self.output_norm(
            pair + self.output_dropout(pair_projected + gate * context_projected)
        )
        if return_gate:
            return fused, gate
        return fused

