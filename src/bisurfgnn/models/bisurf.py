from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from ..config import DataConfig, ModelConfig
from ..data.batch import SurfaceBatch
from .context_gate import ContextGatedFusion
from .graph_encoder import GraphEncoder
from .surface_head import ContinuousInteractionDecoder, SharedMonotherapyHead
from .symmetric_bixt import SymmetricBiXT


ReferenceMode = Literal["predicted_monotherapy", "observed_monotherapy"]
_REFERENCE_MODES: tuple[ReferenceMode, ...] = (
    "predicted_monotherapy",
    "observed_monotherapy",
)


def _safe_logit(value: Tensor, upper_bound: float = 1.0) -> Tensor:
    """Logit of ``value`` rescaled onto the declared response range ``[0,upper_bound]``.

    Mirrors ``models.baselines._safe_logit``. At ``upper_bound = 1`` this is the
    original unit-interval logit; the D1 unclipped endpoint declares a wider
    bound so references and predictions above one survive. See
    ``docs/ENDPOINT_DECISION.md``.
    """

    if float(upper_bound) <= 0.0:
        raise ValueError("upper_bound must be positive")
    scaled = (value / float(upper_bound)).clamp(1e-6, 1.0 - 1e-6)
    return torch.log(scaled) - torch.log1p(-scaled)


@dataclass(frozen=True)
class BiSurfOutput:
    prediction: Tensor
    reference: Tensor
    mono_a: Tensor
    mono_b: Tensor
    interaction_logit: Tensor
    boundary_gate: Tensor
    log_variance: Tensor
    context_gate: Tensor
    attention_ab: Tensor | None = None
    attention_ba: Tensor | None = None


class BiSurfGNN(nn.Module):
    def __init__(
        self,
        data_config: DataConfig,
        model_config: ModelConfig,
        reference_mode: ReferenceMode = "predicted_monotherapy",
        response_upper_bound: float = 1.0,
    ) -> None:
        super().__init__()
        if reference_mode not in _REFERENCE_MODES:
            supported = ", ".join(_REFERENCE_MODES)
            raise ValueError(
                f"Unsupported reference_mode {reference_mode!r}; expected one of: {supported}"
            )
        self.reference_mode: ReferenceMode = reference_mode
        self.response_upper_bound = float(response_upper_bound)
        hidden_dim = model_config.hidden_dim
        self.drug_encoder = GraphEncoder(
            input_dim=data_config.atom_feature_dim,
            hidden_dim=hidden_dim,
            layers=model_config.graph_layers,
            dropout=model_config.dropout,
        )
        self.context_encoder = GraphEncoder(
            input_dim=data_config.cell_feature_dim,
            hidden_dim=hidden_dim,
            layers=model_config.graph_layers,
            dropout=model_config.dropout,
        )
        self.bixt = SymmetricBiXT(
            hidden_dim=hidden_dim,
            num_heads=model_config.attention_heads,
            dropout=model_config.dropout,
        )
        self.context_fusion = ContextGatedFusion(hidden_dim, model_config.dropout)
        self.monotherapy_head = SharedMonotherapyHead(
            hidden_dim, model_config.dose_frequencies, model_config.dropout
        )
        self.interaction_decoder = ContinuousInteractionDecoder(
            hidden_dim, model_config.dose_frequencies, model_config.dropout
        )

    def forward(
        self,
        batch: SurfaceBatch,
        return_attention: bool = False,
    ) -> BiSurfOutput:
        if bool((batch.doses < 0).any()):
            raise ValueError("Doses must be non-negative")
        if self.reference_mode == "observed_monotherapy" and not bool(
            (batch.mono_mask_a & batch.mono_mask_b).all()
        ):
            raise ValueError(
                "reference_mode='observed_monotherapy' requires both monotherapy "
                "masks to be true for every sample"
            )

        encoded_a = self.drug_encoder(batch.drug_a)
        encoded_b = self.drug_encoder(batch.drug_b)
        encoded_context = self.context_encoder(batch.cell)
        exchange = self.bixt(
            encoded_a.tokens,
            encoded_b.tokens,
            batch.drug_a.node_mask,
            batch.drug_b.node_mask,
            return_attention=return_attention,
        )
        conditioned_pair, context_gate = self.context_fusion(
            exchange.pair, encoded_context.pooled, return_gate=True
        )

        mono_a = self.monotherapy_head(
            exchange.pooled_a, encoded_context.pooled, batch.doses[:, 0]
        )
        mono_b = self.monotherapy_head(
            exchange.pooled_b, encoded_context.pooled, batch.doses[:, 1]
        )
        if self.reference_mode == "observed_monotherapy":
            reference = batch.mono_response_a * batch.mono_response_b
        else:
            reference = mono_a * mono_b
        interaction_logit, log_variance = self.interaction_decoder(
            exchange.pooled_a,
            exchange.pooled_b,
            encoded_context.pooled,
            conditioned_pair,
            batch.doses,
        )
        boundary_gate = (1.0 - torch.exp(-2.0 * batch.doses[:, 0])) * (
            1.0 - torch.exp(-2.0 * batch.doses[:, 1])
        )
        shifted = self.response_upper_bound * torch.sigmoid(
            _safe_logit(reference, self.response_upper_bound)
            + boundary_gate * interaction_logit
        )
        prediction = torch.where(boundary_gate == 0, reference, shifted)

        return BiSurfOutput(
            prediction=prediction,
            reference=reference,
            mono_a=mono_a,
            mono_b=mono_b,
            interaction_logit=interaction_logit,
            boundary_gate=boundary_gate,
            log_variance=log_variance,
            context_gate=context_gate,
            attention_ab=exchange.attention_ab,
            attention_ba=exchange.attention_ba,
        )
