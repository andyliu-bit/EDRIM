"""Development variants for the position of cell-conditioned interaction."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .baselines import _anchored_output, _validate_observed_monotherapy
from .compact_bisurf import CompactBiSurfGNN
from .graph_encoder import masked_mean
from .symmetric_bixt import BiXTOutput, SymmetricBiXT, _masked_softmax


class CellSymmetricBiXT(SymmetricBiXT):
    """One cell-weighted relation matrix, shared with its transpose."""

    def forward(self, tokens_a: Tensor, tokens_b: Tensor, mask_a: Tensor,
                mask_b: Tensor, weights: Tensor, return_attention: bool = False) -> BiXTOutput:
        if tokens_a.ndim != 3 or tokens_b.ndim != 3:
            raise ValueError("Drug tokens must have shape [B,N,D]")
        if tokens_a.shape[0] != tokens_b.shape[0]:
            raise ValueError("Drug batch sizes must match")
        if tokens_a.shape[-1] != self.hidden_dim or tokens_b.shape[-1] != self.hidden_dim:
            raise ValueError("Drug token width must match hidden_dim")
        if mask_a.shape != tokens_a.shape[:2] or mask_b.shape != tokens_b.shape[:2]:
            raise ValueError("Masks must match drug tokens")
        if weights.shape != (tokens_a.shape[0], self.hidden_dim):
            raise ValueError("Cell relation weights must have shape [B,D]")
        if not bool(mask_a.any(dim=1).all()) or not bool(mask_b.any(dim=1).all()):
            raise ValueError("Every drug must have a valid atom")
        normalized_a, normalized_b = self.token_norm(tokens_a), self.token_norm(tokens_b)
        relation_a = self._heads(self.relation_projection(normalized_a))
        relation_b = self._heads(self.relation_projection(normalized_b))
        value_a = self._heads(self.value_projection(normalized_a))
        value_b = self._heads(self.value_projection(normalized_b))
        # Applying sqrt(w) to both branches retains A/B equivariance even when
        # the caller swaps the branches. w=1 exactly restores the old relation.
        channel_scale = weights.sqrt().reshape(-1, self.num_heads, 1, self.head_dim)
        logits_ab = torch.matmul(relation_a * channel_scale,
                                 (relation_b * channel_scale).transpose(-2, -1)) * self.scale
        attention_ab = _masked_softmax(logits_ab, mask_b) * mask_a[:, None, :, None]
        attention_ba = _masked_softmax(logits_ab.transpose(-2, -1), mask_a) * mask_b[:, None, :, None]
        update_a = self.output_projection(self._merge_heads(torch.matmul(self.attention_dropout(attention_ab), value_b)))
        update_b = self.output_projection(self._merge_heads(torch.matmul(self.attention_dropout(attention_ba), value_a)))
        tokens_a = tokens_a + self.output_dropout(update_a)
        tokens_b = tokens_b + self.output_dropout(update_b)
        tokens_a = (tokens_a + self.ffn(self.ffn_norm(tokens_a))) * mask_a.unsqueeze(-1)
        tokens_b = (tokens_b + self.ffn(self.ffn_norm(tokens_b))) * mask_b.unsqueeze(-1)
        pooled_a, pooled_b = masked_mean(tokens_a, mask_a), masked_mean(tokens_b, mask_b)
        pair = self.pair_readout(torch.cat([pooled_a + pooled_b, (pooled_a - pooled_b).abs(),
                                          pooled_a * pooled_b], dim=-1))
        return BiXTOutput(tokens_a, tokens_b, pooled_a, pooled_b, pair,
                          attention_ab if return_attention else None,
                          attention_ba if return_attention else None)


class CellConditionedBiSurfGNN(CompactBiSurfGNN):
    """Early relation modulation or parameter-matched late pair modulation.

    Both variants deduplicate by (ordered drug A, ordered drug B, cell). This
    matches their attention workload; the existing model still caches pairs.
    No cache persists between optimizer steps or prediction batches.
    """

    def __init__(self, *args, mode: str, **kwargs):
        super().__init__(*args, **kwargs)
        if mode not in {"early", "late"}:
            raise ValueError("mode must be early or late")
        self.mode = mode
        self.cell_modulation = nn.Linear(self.hidden_dim, self.hidden_dim)
        nn.init.zeros_(self.cell_modulation.weight)
        nn.init.zeros_(self.cell_modulation.bias)
        if mode == "early":
            exchange = CellSymmetricBiXT(self.hidden_dim, self.bixt.num_heads,
                                         self.bixt.attention_dropout.p)
            exchange.load_state_dict(self.bixt.state_dict())
            self.bixt = exchange

    def modulation_weights(self, context: Tensor) -> Tensor:
        return 1.0 + 0.5 * torch.tanh(self.cell_modulation(context))

    def exchange_unique_contexts(self, drug_bank, context_bank, a, b, cell):
        tokens, mask = drug_bank
        n_drugs, n_cells = len(tokens), len(context_bank)
        keys = (a.long() * n_drugs + b.long()) * n_cells + cell.long()
        unique, inverse = torch.unique(keys, return_inverse=True)
        cells = unique % n_cells
        pairs = torch.div(unique, n_cells, rounding_mode="floor")
        ua, ub = torch.div(pairs, n_drugs, rounding_mode="floor"), pairs % n_drugs
        context = context_bank[cells]
        if self.mode == "early":
            exchange = self.bixt(tokens[ua], tokens[ub], mask[ua], mask[ub],
                                 self.modulation_weights(context))
            pair = exchange.pair
        else:
            exchange = self.bixt(tokens[ua], tokens[ub], mask[ua], mask[ub])
            pair = exchange.pair * self.modulation_weights(context)
        return pair[inverse], exchange.pooled_a[inverse], exchange.pooled_b[inverse]

    def forward_pooled(self, drug_bank, context_bank, drug_a_index, drug_b_index,
                       cell_index, doses, mono_a, mono_b, mono_mask_a, mono_mask_b):
        _validate_observed_monotherapy(doses, mono_a, mono_b, mono_mask_a, mono_mask_b,
                                      self.monotherapy_upper_bound)
        tokens, mask = drug_bank
        if tokens.ndim != 3 or tokens.shape[-1] != self.hidden_dim or mask.shape != tokens.shape[:2]:
            raise ValueError("Drug token bank and masks must match hidden_dim")
        if context_bank.ndim != 2 or context_bank.shape[-1] != self.hidden_dim:
            raise ValueError("Context bank must have shape [cells,hidden_dim]")
        pair, exchanged_a, exchanged_b = self.exchange_unique_contexts(
            drug_bank, context_bank, drug_a_index, drug_b_index, cell_index)
        context = context_bank[cell_index.long()]
        conditioned_pair = self.context_fusion(pair, context)
        logit, _ = self.interaction_decoder(exchanged_a, exchanged_b, context,
            conditioned_pair, doses, mono_a=mono_a if self.residual_monotherapy_input else None,
            mono_b=mono_b if self.residual_monotherapy_input else None)
        return _anchored_output(doses, mono_a, mono_b, mono_mask_a, mono_mask_b,
                               logit, self.reference_mode, self.monotherapy_upper_bound)
