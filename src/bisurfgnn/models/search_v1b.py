"""AlgorithmSearch-v1 batch-2 candidates: monotherapy–dose branch with drug/cell bottlenecks.

Evidence behind the design (validation-only comparator diagnostics, seeds 41/42):
an XGBoost residual model on 7 monotherapy/dose scalars already reaches 0.06644
(HSA 0.06995, v2 0.06858); adding 16-d drug PCA gives 0.06568 and 16-d cell PCA
0.06540. The explicit branch here therefore conditions the symmetric
monotherapy–dose encoder on low-dimensional (bottlenecked) drug and cell codes,
instead of feeding the full graph representation. Zero-initialised head: the
untrained model equals the v2 prediction, which equals HSA when the v2 head is zero.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .baselines import _anchored_output, _validate_observed_monotherapy
from .graph_encoder import masked_mean
from .pathway_readout_bisurf import MonotherapyDoseBranch, SearchBiSurfGNN


class ContextualMonotherapyDoseBranch(MonotherapyDoseBranch):
    def __init__(self, hidden_dim: int, dose_frequencies: int, dropout: float,
                 upper_bound: float, bottleneck: int = 16) -> None:
        super().__init__(hidden_dim, dose_frequencies, dropout, upper_bound)
        arm_width = 2 + self.dose_features.output_dim + 2 * bottleneck
        self.arm = nn.Sequential(nn.Linear(arm_width, hidden_dim), nn.GELU(),
                                 nn.Linear(hidden_dim, hidden_dim))
        self.drug_code = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, bottleneck),
                                       nn.Dropout(dropout))
        self.cell_code = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, bottleneck),
                                       nn.Dropout(dropout))

    def forward(self, mono_a: Tensor, mono_b: Tensor, doses: Tensor, drug_a: Tensor,
                drug_b: Tensor, cell: Tensor) -> Tensor:
        cell_code = self.cell_code(cell)
        def arm(mono, dose, drug):
            return self.arm(torch.cat([mono[:, None], self._logit(mono)[:, None],
                                       self.dose_features(dose), self.drug_code(drug), cell_code], dim=-1))
        ua, ub = arm(mono_a, doses[:, 0], drug_a), arm(mono_b, doses[:, 1], drug_b)
        pair_scalars = torch.stack([torch.minimum(mono_a, mono_b), mono_a * mono_b,
                                    torch.abs(mono_a - mono_b)], dim=-1)
        hidden = self.pair(torch.cat([ua + ub, torch.abs(ua - ub), ua * ub, pair_scalars], dim=-1))
        return self.head(hidden).squeeze(-1)


class SearchV1bBiSurfGNN(SearchBiSurfGNN):
    """v2 + monotherapy–dose branch; optionally conditioned on drug/cell bottleneck codes."""

    def __init__(self, atom_feature_dim: int, cell_feature_dim: int, hidden_dim: int, *,
                 contextual_branch: bool = False, **kwargs) -> None:
        kwargs["monotherapy_dose_branch"] = True
        super().__init__(atom_feature_dim, cell_feature_dim, hidden_dim, **kwargs)
        self.contextual_branch = bool(contextual_branch)
        if self.contextual_branch:
            self.monotherapy_dose_branch = ContextualMonotherapyDoseBranch(
                hidden_dim, self.interaction_decoder.dose_features.frequencies.numel(),
                self.bixt.attention_dropout.p, self.monotherapy_upper_bound)

    def branch_parameters(self):
        return list(self.monotherapy_dose_branch.parameters())

    def _forward_parts(self, drug_bank, context_bank, drug_a_index, drug_b_index, cell_index,
                       doses, mono_a, mono_b, mono_mask_a, mono_mask_b):
        if self.use_pathway_readout or self.use_drug_cell_drug:
            raise NotImplementedError("Batch-2 candidates do not combine readout or triple exchange")
        _validate_observed_monotherapy(doses, mono_a, mono_b, mono_mask_a, mono_mask_b,
                                       self.monotherapy_upper_bound)
        drug_tokens, drug_mask = drug_bank
        pooled = context_bank[0]
        pair, ex_a, ex_b = self.exchange_unique_pairs(drug_tokens, drug_mask, drug_a_index, drug_b_index)
        context = pooled[cell_index.to(torch.int64)]
        conditioned_pair = self.context_fusion(pair, context)
        decoder = self.interaction_decoder
        arm_a = decoder._arm(ex_a, context, doses[:, 0], mono_a)
        arm_b = decoder._arm(ex_b, context, doses[:, 1], mono_b)
        hidden = decoder.interaction_network(torch.cat(
            [arm_a + arm_b, torch.abs(arm_a - arm_b), arm_a * arm_b, conditioned_pair, context], dim=-1))
        logit = decoder.residual_head(hidden).squeeze(-1)
        if self.contextual_branch:
            a, b = drug_a_index.to(torch.int64), drug_b_index.to(torch.int64)
            drug_a = masked_mean(drug_tokens[a], drug_mask[a])  # drug's own, partner-free
            drug_b = masked_mean(drug_tokens[b], drug_mask[b])
            logit = logit + self.monotherapy_dose_branch(mono_a, mono_b, doses, drug_a, drug_b, context)
        else:
            logit = logit + self.monotherapy_dose_branch(mono_a, mono_b, doses)
        output = _anchored_output(doses, mono_a, mono_b, mono_mask_a, mono_mask_b, logit,
                                  self.reference_mode, self.monotherapy_upper_bound)
        return output, None


__all__ = ["SearchV1bBiSurfGNN"]
