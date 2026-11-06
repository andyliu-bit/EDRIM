"""AdditiveCompactBiSurfGNN with optional external drug and cell vectors (BiSurfGNN v2, phase 1: input parity).

Protocol: ``docs/BISURFGNN_V2_DEV_PROTOCOL.md`` section 3; design: ``docs/INPUT_PARITY_FEASIBILITY_20261013.md`` 4.2.

* drug: ``phi_d(MoLFormer)`` is added to every valid atom token after the graph encoder, so symmetric BiXT, the
  pair embedding and both decoder arms see it;
* cell: ``phi_c(976 genes)`` is added to the pooled pathway context, which feeds the context gate, both decoder
  arms and the interaction network.

``phi = LayerNorm -> Linear(in, hidden) -> GELU -> Dropout -> Linear(hidden, hidden)``; the last layer is
zero-initialised, so the injected terms are exactly zero at initialisation and the untrained model still predicts
the observed HSA reference. The new modules are created after the parent's, so the parent's parameters and the RNG
draws that initialise them are those of the gate model for the same seed; with both inputs disabled the module is
parameter- and output-identical to ``AdditiveCompactBiSurfGNN``. Both drugs share ``phi_d`` and the cell vector is
common to both arms, so exchange symmetry is unchanged.

External rows are inputs, never stored in the model: callers pass them aligned to the graph batch they encode (the
unique, sorted bank rows of a training batch, or the full bank at prediction time). An enabled projection without
its rows raises, so the gate's feature-free evaluation path cannot silently score this model.
"""

from __future__ import annotations

from torch import Tensor, nn

from ..data.batch import GraphBatch
from .compact_bisurf_additive import AdditiveCompactBiSurfGNN


def external_projection(input_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
    """LayerNorm -> Linear -> GELU -> Dropout -> Linear, with a zero-initialised output layer."""
    projection = nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, hidden_dim),
    )
    nn.init.zeros_(projection[-1].weight)
    nn.init.zeros_(projection[-1].bias)
    return projection


class ParityAdditiveCompactBiSurfGNN(AdditiveCompactBiSurfGNN):
    """``AdditiveCompactBiSurfGNN`` plus optional projected per-drug and per-cell external vectors."""

    def __init__(
        self,
        atom_feature_dim: int,
        cell_feature_dim: int,
        hidden_dim: int,
        *,
        drug_ext_dim: int | None = None,
        cell_ext_dim: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(atom_feature_dim, cell_feature_dim, hidden_dim, **kwargs)
        for name, value in (("drug_ext_dim", drug_ext_dim), ("cell_ext_dim", cell_ext_dim)):
            if value is not None and (isinstance(value, bool) or int(value) < 1):
                raise ValueError(f"{name} must be a positive integer or None")
        dropout = float(kwargs.get("dropout", 0.0))
        self.drug_ext_dim = int(drug_ext_dim) if drug_ext_dim is not None else None
        self.cell_ext_dim = int(cell_ext_dim) if cell_ext_dim is not None else None
        # plain None attributes are not registered, so disabled inputs leave state dict and repr unchanged
        self.drug_ext_proj = (external_projection(self.drug_ext_dim, hidden_dim, dropout)
                              if self.drug_ext_dim is not None else None)
        self.cell_ext_proj = (external_projection(self.cell_ext_dim, hidden_dim, dropout)
                              if self.cell_ext_dim is not None else None)

    def encode_drugs(self, drug_graphs: GraphBatch, drug_ext: Tensor | None = None) -> tuple[Tensor, Tensor]:
        tokens, mask = super().encode_drugs(drug_graphs)
        if self.drug_ext_proj is None:
            if drug_ext is not None:
                raise ValueError("drug_ext was given but the drug projection is disabled")
            return tokens, mask
        if drug_ext is None or tuple(drug_ext.shape) != (tokens.shape[0], self.drug_ext_dim):
            raise ValueError("drug_ext rows must align with the encoded drug graphs")
        injected = self.drug_ext_proj(drug_ext).unsqueeze(1) * mask.unsqueeze(-1).to(tokens.dtype)
        return tokens + injected, mask

    def encode_context(self, cell_graphs: GraphBatch, cell_ext: Tensor | None = None) -> Tensor:
        context = super().encode_context(cell_graphs)
        if self.cell_ext_proj is None:
            if cell_ext is not None:
                raise ValueError("cell_ext was given but the cell projection is disabled")
            return context
        if cell_ext is None or tuple(cell_ext.shape) != (context.shape[0], self.cell_ext_dim):
            raise ValueError("cell_ext rows must align with the encoded cell graphs")
        return context + self.cell_ext_proj(cell_ext)

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
        drug_ext: Tensor | None = None,
        cell_ext: Tensor | None = None,
    ):
        return self.forward_pooled(
            self.encode_drugs(drug_graphs, drug_ext),
            self.encode_context(cell_graphs, cell_ext),
            drug_a_index,
            drug_b_index,
            cell_index,
            doses,
            mono_a,
            mono_b,
            mono_mask_a,
            mono_mask_b,
        )


__all__ = ["ParityAdditiveCompactBiSurfGNN", "external_projection"]
