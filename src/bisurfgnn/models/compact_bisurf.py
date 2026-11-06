"""BiSurfGNN over compact, unique graph banks with an observed-monotherapy anchor.

``BiSurfGNN`` in ``models.bisurf`` consumes one padded ``SurfaceBatch`` per
sample, which is the right shape for the synthetic benchmark but not for the
real One-new-scaffold shards, where the same 105 drugs and 59 cells recur across
hundreds of thousands of points. This module keeps the four architectural
assumptions of BiSurfGNN and re-expresses them over unique banks, matching the
calling convention of ``CompactIndependentPooledGNN`` so the frozen Phase A
runner can drive either model.

The reason a separate compact class is needed rather than a thin wrapper is that
symmetric BiXT is a *pair*-level operation on atom tokens, not a per-drug pooled
one. Encoding drugs once is not enough: the exchange has to run per drug pair.
Running it per point would repeat the same pair thousands of times and make the
attention cost quadratic in atoms for every row, so the pair axis is deduplicated
inside the forward pass and the result is gathered back to points. Cell context
and dose stay at point level because they vary within a pair.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..data.batch import GraphBatch
from .baselines import _anchored_output, _validate_observed_monotherapy
from .context_gate import ContextGatedFusion
from .graph_encoder import GraphEncoder
from .surface_head import ContinuousInteractionDecoder
from .symmetric_bixt import SymmetricBiXT

_REFERENCE_MODES = ("observed_product", "observed_hsa")


class CompactBiSurfGNN(nn.Module):
    """Symmetric BiXT + context gate + continuous dose decoder on unique banks.

    Exchange symmetry is preserved end to end: ``SymmetricBiXT`` shares every
    projection across the two branches, and the decoder combines the two arms
    only through ``a + b``, ``|a - b|`` and ``a * b``. Swapping the aligned drug,
    dose and monotherapy inputs therefore leaves the prediction unchanged.
    """

    def __init__(
        self,
        atom_feature_dim: int,
        cell_feature_dim: int,
        hidden_dim: int,
        graph_layers: int = 2,
        attention_heads: int = 4,
        dose_frequencies: int = 4,
        dropout: float = 0.0,
        reference_mode: str = "observed_hsa",
        monotherapy_upper_bound: float = 1.0,
        residual_monotherapy_input: bool = False,
    ) -> None:
        super().__init__()
        if reference_mode not in _REFERENCE_MODES:
            raise ValueError(
                "reference_mode must be 'observed_product' or 'observed_hsa'"
            )
        if atom_feature_dim < 1 or cell_feature_dim < 1:
            raise ValueError("Graph feature dimensions must be positive")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0,1)")
        if hidden_dim % attention_heads != 0:
            raise ValueError("hidden_dim must be divisible by attention_heads")

        self.hidden_dim = int(hidden_dim)
        self.reference_mode = reference_mode
        self.monotherapy_upper_bound = float(monotherapy_upper_bound)
        if not isinstance(residual_monotherapy_input, bool):
            raise TypeError("residual_monotherapy_input must be boolean")
        self.residual_monotherapy_input = residual_monotherapy_input

        self.drug_encoder = GraphEncoder(
            input_dim=atom_feature_dim,
            hidden_dim=hidden_dim,
            layers=graph_layers,
            dropout=dropout,
        )
        self.context_encoder = GraphEncoder(
            input_dim=cell_feature_dim,
            hidden_dim=hidden_dim,
            layers=graph_layers,
            dropout=dropout,
        )
        self.bixt = SymmetricBiXT(
            hidden_dim=hidden_dim,
            num_heads=attention_heads,
            dropout=dropout,
        )
        self.context_fusion = ContextGatedFusion(hidden_dim, dropout)
        self.interaction_decoder = ContinuousInteractionDecoder(
            hidden_dim, dose_frequencies, dropout,
            monotherapy_conditioned=residual_monotherapy_input,
        )
        # The frozen Phase A contract requires the untrained model to reproduce
        # the observed anchor exactly, so that the epoch -1 candidate really is
        # the zero-parameter reference and a fold can decline to learn. The
        # pooled control gets this by zeroing its residual head; the shared
        # continuous decoder is zeroed here for the same reason.
        nn.init.zeros_(self.interaction_decoder.residual_head.weight)
        nn.init.zeros_(self.interaction_decoder.residual_head.bias)

    def encode_drugs(self, drug_graphs: GraphBatch) -> tuple[Tensor, Tensor]:
        """Atom tokens and their padding mask for every unique drug graph.

        The pooled graph control returns a ``[drugs, hidden]`` embedding bank
        here. Symmetric BiXT exchanges *atom* tokens, so this bank is
        ``[drugs, atoms, hidden]`` plus its mask. The frozen evaluation path
        passes whatever this returns straight into ``forward_pooled`` without
        inspecting it, so both architectures share that path unchanged.
        """
        encoded = self.drug_encoder(drug_graphs)
        return encoded.tokens, drug_graphs.node_mask

    def encode_context(self, cell_graphs: GraphBatch) -> Tensor:
        """Pooled pathway-graph embedding for every unique cell."""
        return self.context_encoder(cell_graphs).pooled

    def exchange_unique_pairs(
        self,
        drug_tokens: Tensor,
        drug_mask: Tensor,
        drug_a_index: Tensor,
        drug_b_index: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run symmetric BiXT once per distinct ordered pair, then gather to points.

        Returns the per-point pair embedding and the two per-point exchanged drug
        embeddings.
        """
        bank_size = drug_tokens.shape[0]
        pair_key = drug_a_index.to(torch.int64) * bank_size + drug_b_index.to(torch.int64)
        unique_keys, inverse = torch.unique(pair_key, return_inverse=True)
        unique_a = torch.div(unique_keys, bank_size, rounding_mode="floor")
        unique_b = unique_keys % bank_size

        exchange = self.bixt(
            drug_tokens[unique_a],
            drug_tokens[unique_b],
            drug_mask[unique_a],
            drug_mask[unique_b],
        )
        return (
            exchange.pair[inverse],
            exchange.pooled_a[inverse],
            exchange.pooled_b[inverse],
        )

    def forward_pooled(
        self,
        drug_bank: tuple[Tensor, Tensor],
        context_bank: Tensor,
        drug_a_index: Tensor,
        drug_b_index: Tensor,
        cell_index: Tensor,
        doses: Tensor,
        mono_a: Tensor,
        mono_b: Tensor,
        mono_mask_a: Tensor,
        mono_mask_b: Tensor,
    ):
        """Score points against already-encoded drug and context banks."""
        _validate_observed_monotherapy(
            doses,
            mono_a,
            mono_b,
            mono_mask_a,
            mono_mask_b,
            self.monotherapy_upper_bound,
        )
        drug_tokens, drug_mask = drug_bank
        if drug_tokens.ndim != 3:
            raise ValueError("Drug token bank must have shape [drugs, atoms, hidden]")
        if drug_tokens.shape[-1] != self.hidden_dim:
            raise ValueError("Drug token bank width does not match hidden_dim")
        if drug_mask.shape != drug_tokens.shape[:2]:
            raise ValueError("Drug token mask must match the token bank")
        if context_bank.ndim != 2 or context_bank.shape[-1] != self.hidden_dim:
            raise ValueError("Context bank must have shape [cells, hidden]")

        pair, exchanged_a, exchanged_b = self.exchange_unique_pairs(
            drug_tokens, drug_mask, drug_a_index, drug_b_index
        )
        context = context_bank[cell_index.to(torch.int64)]
        conditioned_pair = self.context_fusion(pair, context)
        interaction_logit, _ = self.interaction_decoder(
            exchanged_a,
            exchanged_b,
            context,
            conditioned_pair,
            doses,
            mono_a=mono_a if self.residual_monotherapy_input else None,
            mono_b=mono_b if self.residual_monotherapy_input else None,
        )
        return _anchored_output(
            doses,
            mono_a,
            mono_b,
            mono_mask_a,
            mono_mask_b,
            interaction_logit,
            self.reference_mode,
            self.monotherapy_upper_bound,
        )

    def forward(
        self,
        drug_graphs: GraphBatch,
        cell_graphs: GraphBatch,
        drug_a_index: Tensor,
        drug_b_index: Tensor,
        cell_index: Tensor,
        doses: Tensor,
        mono_a: Tensor,
        mono_b: Tensor,
        mono_mask_a: Tensor,
        mono_mask_b: Tensor,
    ):
        """Encode the supplied unique banks, then score the points against them."""
        return self.forward_pooled(
            self.encode_drugs(drug_graphs),
            self.encode_context(cell_graphs),
            drug_a_index,
            drug_b_index,
            cell_index,
            doses,
            mono_a,
            mono_b,
            mono_mask_a,
            mono_mask_b,
        )


__all__ = ["CompactBiSurfGNN"]
