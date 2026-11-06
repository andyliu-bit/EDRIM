from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .graph_encoder import masked_mean


@dataclass(frozen=True)
class BiXTOutput:
    tokens_a: Tensor
    tokens_b: Tensor
    pooled_a: Tensor
    pooled_b: Tensor
    pair: Tensor
    attention_ab: Tensor | None = None
    attention_ba: Tensor | None = None


def _masked_softmax(logits: Tensor, key_mask: Tensor) -> Tensor:
    mask = key_mask[:, None, None, :]
    masked = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    probabilities = torch.softmax(masked, dim=-1)
    return probabilities * mask.to(dtype=probabilities.dtype)


class SymmetricBiXT(nn.Module):
    """Bidirectional cross-token exchange with exact A/B equivariance in eval mode.

    The Plus BiXT idea of using one relation matrix and its transpose is retained.
    Both drug branches share every projection, normalization, and feed-forward weight.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.token_norm = nn.LayerNorm(hidden_dim)
        self.relation_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.pair_readout = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def _heads(self, values: Tensor) -> Tensor:
        batch, tokens, _ = values.shape
        return values.reshape(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, values: Tensor) -> Tensor:
        batch, _, tokens, _ = values.shape
        return values.transpose(1, 2).reshape(batch, tokens, self.hidden_dim)

    def forward(
        self,
        tokens_a: Tensor,
        tokens_b: Tensor,
        mask_a: Tensor,
        mask_b: Tensor,
        return_attention: bool = False,
    ) -> BiXTOutput:
        if tokens_a.ndim != 3 or tokens_b.ndim != 3:
            raise ValueError("Drug tokens must have shape [B,N,D]")
        if tokens_a.shape[0] != tokens_b.shape[0]:
            raise ValueError("Drug A and Drug B batch sizes must match")
        if tokens_a.shape[-1] != self.hidden_dim or tokens_b.shape[-1] != self.hidden_dim:
            raise ValueError("Drug token dimensions do not match hidden_dim")
        if mask_a.shape != tokens_a.shape[:2] or mask_b.shape != tokens_b.shape[:2]:
            raise ValueError("Drug masks must match their token dimensions")
        if not bool(mask_a.any(dim=1).all()) or not bool(mask_b.any(dim=1).all()):
            raise ValueError("Every drug must contain at least one valid atom")

        normalized_a = self.token_norm(tokens_a)
        normalized_b = self.token_norm(tokens_b)
        relation_a = self._heads(self.relation_projection(normalized_a))
        relation_b = self._heads(self.relation_projection(normalized_b))
        value_a = self._heads(self.value_projection(normalized_a))
        value_b = self._heads(self.value_projection(normalized_b))

        logits_ab = torch.matmul(relation_a, relation_b.transpose(-2, -1)) * self.scale
        attention_ab = _masked_softmax(logits_ab, mask_b)
        attention_ba = _masked_softmax(logits_ab.transpose(-2, -1), mask_a)
        attention_ab = attention_ab * mask_a[:, None, :, None].to(attention_ab.dtype)
        attention_ba = attention_ba * mask_b[:, None, :, None].to(attention_ba.dtype)

        update_a = torch.matmul(self.attention_dropout(attention_ab), value_b)
        update_b = torch.matmul(self.attention_dropout(attention_ba), value_a)
        update_a = self.output_projection(self._merge_heads(update_a))
        update_b = self.output_projection(self._merge_heads(update_b))

        tokens_a = tokens_a + self.output_dropout(update_a)
        tokens_b = tokens_b + self.output_dropout(update_b)
        tokens_a = tokens_a + self.ffn(self.ffn_norm(tokens_a))
        tokens_b = tokens_b + self.ffn(self.ffn_norm(tokens_b))
        tokens_a = tokens_a * mask_a.unsqueeze(-1).to(tokens_a.dtype)
        tokens_b = tokens_b * mask_b.unsqueeze(-1).to(tokens_b.dtype)

        pooled_a = masked_mean(tokens_a, mask_a)
        pooled_b = masked_mean(tokens_b, mask_b)
        invariant = torch.cat(
            [
                pooled_a + pooled_b,
                torch.abs(pooled_a - pooled_b),
                pooled_a * pooled_b,
            ],
            dim=-1,
        )
        pair = self.pair_readout(invariant)
        return BiXTOutput(
            tokens_a=tokens_a,
            tokens_b=tokens_b,
            pooled_a=pooled_a,
            pooled_b=pooled_b,
            pair=pair,
            attention_ab=attention_ab if return_attention else None,
            attention_ba=attention_ba if return_attention else None,
        )

