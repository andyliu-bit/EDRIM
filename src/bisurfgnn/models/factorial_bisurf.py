"""Constant-width exchange x context-fusion controls with one shared decoder.

These are development ablations, not the legacy parameter-matched benchmark.
Construct every variant with the same RNG seed to align common initial weights;
reset the training RNG after construction (variant construction consumes RNG).
"""

from __future__ import annotations

from copy import deepcopy
import hashlib

import torch
from torch import nn

from .compact_bisurf import CompactBiSurfGNN
from .context_gate import ContextGatedFusion
from .graph_encoder import masked_mean
from .symmetric_bixt import BiXTOutput, SymmetricBiXT, _masked_softmax

EXCHANGES = ("pooled", "bixt", "cross_attention")
FUSIONS = ("concat", "gate")
VARIANTS = tuple(f"{exchange}__{fusion}" for exchange in EXCHANGES for fusion in FUSIONS)


class ControlledExchange(nn.Module):
    """No cross messages, or ordinary shared-Q/K bidirectional cross attention.

Both controls retain the BiXT post-message token FFN and invariant pair readout.
The pooled arm therefore isolates the cross-message block, not the token FFN.
Q and K are shared between drugs but distinct from one another; reverse attention
is computed independently, unlike the tied-relation transpose used by BiXT.
"""

    def __init__(self, source: SymmetricBiXT, mode: str) -> None:
        super().__init__()
        if mode not in ("pooled", "cross_attention"):
            raise ValueError("ControlledExchange expects pooled or cross_attention")
        self.mode = mode
        self.hidden_dim, self.num_heads = source.hidden_dim, source.num_heads
        self.head_dim, self.scale = source.head_dim, source.scale
        self.ffn_norm = deepcopy(source.ffn_norm)
        self.ffn = deepcopy(source.ffn)
        self.pair_readout = deepcopy(source.pair_readout)
        if mode == "cross_attention":
            self.token_norm = deepcopy(source.token_norm)
            self.query_projection = deepcopy(source.relation_projection)
            self.key_projection = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
            self.value_projection = deepcopy(source.value_projection)
            self.output_projection = deepcopy(source.output_projection)
            self.attention_dropout = deepcopy(source.attention_dropout)
            self.output_dropout = deepcopy(source.output_dropout)

    def _heads(self, values):
        batch, tokens, _ = values.shape
        return values.reshape(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge(self, values):
        batch, _, tokens, _ = values.shape
        return values.transpose(1, 2).reshape(batch, tokens, self.hidden_dim)

    def forward(self, tokens_a, tokens_b, mask_a, mask_b, return_attention=False):
        for tokens, mask in ((tokens_a, mask_a), (tokens_b, mask_b)):
            if tokens.ndim != 3 or tokens.shape[-1] != self.hidden_dim:
                raise ValueError("Drug tokens must have shape [B,N,hidden_dim]")
            if mask.dtype != torch.bool or mask.shape != tokens.shape[:2]:
                raise ValueError("Drug masks must be aligned boolean tensors")
            if not bool(mask.any(dim=1).all()):
                raise ValueError("Every drug must contain at least one valid atom")
        if tokens_a.shape[0] != tokens_b.shape[0]:
            raise ValueError("Drug batch sizes must match")
        attention_ab = attention_ba = None
        if self.mode == "cross_attention":
            norm_a, norm_b = self.token_norm(tokens_a), self.token_norm(tokens_b)
            qa, qb = self._heads(self.query_projection(norm_a)), self._heads(self.query_projection(norm_b))
            ka, kb = self._heads(self.key_projection(norm_a)), self._heads(self.key_projection(norm_b))
            va, vb = self._heads(self.value_projection(norm_a)), self._heads(self.value_projection(norm_b))
            attention_ab = _masked_softmax((qa @ kb.transpose(-2, -1)) * self.scale, mask_b)
            attention_ba = _masked_softmax((qb @ ka.transpose(-2, -1)) * self.scale, mask_a)
            attention_ab = attention_ab * mask_a[:, None, :, None]
            attention_ba = attention_ba * mask_b[:, None, :, None]
            update_a = self.output_projection(self._merge(self.attention_dropout(attention_ab) @ vb))
            update_b = self.output_projection(self._merge(self.attention_dropout(attention_ba) @ va))
            tokens_a = tokens_a + self.output_dropout(update_a)
            tokens_b = tokens_b + self.output_dropout(update_b)
        tokens_a = (tokens_a + self.ffn(self.ffn_norm(tokens_a))) * mask_a.unsqueeze(-1)
        tokens_b = (tokens_b + self.ffn(self.ffn_norm(tokens_b))) * mask_b.unsqueeze(-1)
        pooled_a, pooled_b = masked_mean(tokens_a, mask_a), masked_mean(tokens_b, mask_b)
        pair = self.pair_readout(torch.cat([
            pooled_a + pooled_b, (pooled_a - pooled_b).abs(), pooled_a * pooled_b
        ], dim=-1))
        return BiXTOutput(tokens_a, tokens_b, pooled_a, pooled_b, pair,
                          attention_ab if return_attention else None,
                          attention_ba if return_attention else None)


class ConcatContextFusion(nn.Module):
    """Additive concatenation MLP control; no sigmoid or multiplicative gate.

Common projections, output normalization and residual path match the gate arm.
The changed block is MLP([p,c]) instead of sigmoid(MLP([p,c,p*c,|p-c|]))*c.
This is a fusion-block contrast, not a single-operator causal intervention.
"""

    def __init__(self, source: ContextGatedFusion, hidden_dim: int, dropout: float):
        super().__init__()
        for name in ("pair_norm", "context_norm", "pair_projection", "context_projection",
                     "output_norm", "output_dropout"):
            setattr(self, name, deepcopy(getattr(source, name)))
        self.concat = nn.Sequential(nn.Linear(2 * hidden_dim, 2 * hidden_dim), nn.GELU(),
                                    nn.Dropout(dropout), nn.Linear(2 * hidden_dim, hidden_dim))

    def forward(self, pair, context):
        p = self.pair_projection(self.pair_norm(pair))
        c = self.context_projection(self.context_norm(context))
        return self.output_norm(pair + self.output_dropout(p + self.concat(torch.cat([p, c], -1))))


class FactorialBiSurfGNN(CompactBiSurfGNN):
    def __init__(self, atom_feature_dim: int, cell_feature_dim: int, hidden_dim: int,
                 *, exchange: str, fusion: str, graph_layers: int = 2,
                 attention_heads: int = 4, dose_frequencies: int = 4,
                 dropout: float = 0.1, monotherapy_upper_bound: float = 3.0):
        if exchange not in EXCHANGES or fusion not in FUSIONS:
            raise ValueError("Unknown factorial exchange/fusion")
        # Always construct the identical base first: common decoder/encoder
        # initialization must not depend on how many random draws an arm uses.
        super().__init__(atom_feature_dim, cell_feature_dim, hidden_dim,
                         graph_layers=graph_layers, attention_heads=attention_heads,
                         dose_frequencies=dose_frequencies, dropout=dropout,
                         reference_mode="observed_hsa", monotherapy_upper_bound=monotherapy_upper_bound,
                         residual_monotherapy_input=True)
        self.exchange_mode, self.fusion_mode = exchange, fusion
        if exchange != "bixt":
            self.bixt = ControlledExchange(self.bixt, exchange)
        if fusion == "concat":
            # Keep concat initial weights aligned across all exchange arms too.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(torch.initial_seed() + 7919)
                self.context_fusion = ConcatContextFusion(self.context_fusion, hidden_dim, dropout)
        # The common legacy decoder returns uncertainty, unused by this MSE
        # experiment. Keep its state for compatibility but exclude it from training.
        self.interaction_decoder.log_variance_head.requires_grad_(False)
        # CompactBiSurf consumes drug tokens, not GraphEncoder.pooled. Its
        # legacy pool projection is therefore outside the prediction path too.
        self.drug_encoder.pool_projection.requires_grad_(False)


def common_initial_state_hash(model: FactorialBiSurfGNN) -> str:
    """Hash shared encoders, decoder, FFN/readout and fusion projections by name."""
    excluded = ("bixt.token_norm.", "bixt.relation_projection.", "bixt.query_projection.",
                "bixt.key_projection.", "bixt.value_projection.", "bixt.output_projection.",
                "context_fusion.gate.", "context_fusion.concat.")
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name.startswith(excluded):
            continue
        digest.update(f"{name}|{value.dtype}|{tuple(value.shape)}|".encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
