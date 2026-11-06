from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..config import DataConfig, ModelConfig
from ..data.batch import SurfaceBatch
from .graph_encoder import GraphEncoder
from .surface_head import DoseFeatures


@dataclass(frozen=True)
class BaselineOutput:
    """Common output contract for reference and learned surface baselines."""

    prediction: Tensor
    reference: Tensor
    mono_a: Tensor
    mono_b: Tensor
    interaction_logit: Tensor
    boundary_gate: Tensor
    log_variance: Tensor


def _validate_observed_monotherapy(
    doses: Tensor,
    mono_a: Tensor,
    mono_b: Tensor,
    mask_a: Tensor,
    mask_b: Tensor,
    upper_bound: float = 1.0,
) -> None:
    if doses.ndim != 2 or doses.shape[-1] != 2:
        raise ValueError("doses must have shape [B,2]")
    batch_size = doses.shape[0]
    expected = (batch_size,)
    if mono_a.shape != expected or mono_b.shape != expected:
        raise ValueError("Observed monotherapy responses must have shape [B]")
    if mask_a.shape != expected or mask_b.shape != expected:
        raise ValueError("Observed monotherapy masks must have shape [B]")
    if not bool(torch.isfinite(doses).all()) or bool((doses < 0).any()):
        raise ValueError("Doses must be finite and non-negative")
    if not bool(torch.isfinite(mono_a).all()) or not bool(torch.isfinite(mono_b).all()):
        raise ValueError("Observed monotherapy responses must be finite")
    if float(upper_bound) < 1.0:
        raise ValueError("Monotherapy upper bound must be at least 1.0")
    if bool(((mono_a < 0) | (mono_a > upper_bound)).any()) or bool(
        ((mono_b < 0) | (mono_b > upper_bound)).any()
    ):
        raise ValueError(
            f"Observed monotherapy responses must lie in [0,{float(upper_bound)}]"
        )
    if not bool(mask_a.bool().all()) or not bool(mask_b.bool().all()):
        raise ValueError(
            "Observed-monotherapy baselines require both monotherapy anchors for every point"
        )


def _observed_reference(
    doses: Tensor,
    mono_a: Tensor,
    mono_b: Tensor,
    mask_a: Tensor,
    mask_b: Tensor,
    reference_mode: str = "observed_product",
    upper_bound: float = 1.0,
) -> tuple[Tensor, Tensor]:
    _validate_observed_monotherapy(doses, mono_a, mono_b, mask_a, mask_b, upper_bound)
    if reference_mode == "observed_product":
        reference = mono_a * mono_b
    elif reference_mode == "observed_hsa":
        reference = torch.minimum(mono_a, mono_b)
    else:
        raise ValueError(
            "reference_mode must be 'observed_product' or 'observed_hsa'"
        )
    boundary_gate = (1.0 - torch.exp(-2.0 * doses[:, 0])) * (
        1.0 - torch.exp(-2.0 * doses[:, 1])
    )
    return reference, boundary_gate


def _safe_logit(value: Tensor, upper_bound: float = 1.0) -> Tensor:
    """Logit of ``value`` rescaled onto the declared response range ``[0,upper_bound]``.

    The clipped endpoint bounds responses at 1.0 and this reduces exactly to the
    original unit-interval logit. The D1 unclipped endpoint keeps ratios above 1,
    so the anchored parameterisation is generalised to the declared bound instead
    of truncating both the reference and the prediction at one. See
    ``docs/ENDPOINT_DECISION.md``.
    """

    if float(upper_bound) <= 0.0:
        raise ValueError("upper_bound must be positive")
    scaled = (value / float(upper_bound)).clamp(1e-6, 1.0 - 1e-6)
    return torch.log(scaled) - torch.log1p(-scaled)


def _anchored_output(
    doses: Tensor,
    mono_a: Tensor,
    mono_b: Tensor,
    mask_a: Tensor,
    mask_b: Tensor,
    interaction_logit: Tensor,
    reference_mode: str = "observed_product",
    upper_bound: float = 1.0,
) -> BaselineOutput:
    reference, boundary_gate = _observed_reference(
        doses,
        mono_a,
        mono_b,
        mask_a,
        mask_b,
        reference_mode,
        upper_bound,
    )
    if interaction_logit.shape != reference.shape:
        raise ValueError("interaction_logit must have shape [B]")
    if not bool(torch.isfinite(interaction_logit).all()):
        raise ValueError("interaction_logit must be finite")
    shifted = float(upper_bound) * torch.sigmoid(
        _safe_logit(reference, upper_bound) + boundary_gate * interaction_logit
    )
    prediction = torch.where(boundary_gate == 0, reference, shifted)
    return BaselineOutput(
        prediction=prediction,
        reference=reference,
        mono_a=mono_a,
        mono_b=mono_b,
        interaction_logit=interaction_logit,
        boundary_gate=boundary_gate,
        log_variance=torch.zeros_like(reference),
    )


class ObservedBlissReference(nn.Module):
    """Compatibility name for the observed product reference."""

    def __init__(self, monotherapy_upper_bound: float = 1.0) -> None:
        super().__init__()
        self.monotherapy_upper_bound = float(monotherapy_upper_bound)

    def forward(self, batch: SurfaceBatch) -> BaselineOutput:
        reference, boundary_gate = _observed_reference(
            batch.doses,
            batch.mono_response_a,
            batch.mono_response_b,
            batch.mono_mask_a,
            batch.mono_mask_b,
            upper_bound=self.monotherapy_upper_bound,
        )
        zeros = torch.zeros_like(reference)
        return BaselineOutput(
            prediction=reference,
            reference=reference,
            mono_a=batch.mono_response_a,
            mono_b=batch.mono_response_b,
            interaction_logit=zeros,
            boundary_gate=boundary_gate,
            log_variance=zeros,
        )


class ObservedProductReference(ObservedBlissReference):
    """Observed product reference without assuming a finalized endpoint meaning."""


class ObservedHSAReference(nn.Module):
    """Parameter-free minimum-single-agent reference on the prepared endpoint."""

    def __init__(self, monotherapy_upper_bound: float = 1.0) -> None:
        super().__init__()
        self.monotherapy_upper_bound = float(monotherapy_upper_bound)

    def forward(self, batch: SurfaceBatch) -> BaselineOutput:
        _, boundary_gate = _observed_reference(
            batch.doses,
            batch.mono_response_a,
            batch.mono_response_b,
            batch.mono_mask_a,
            batch.mono_mask_b,
            upper_bound=self.monotherapy_upper_bound,
        )
        reference = torch.minimum(batch.mono_response_a, batch.mono_response_b)
        zeros = torch.zeros_like(reference)
        return BaselineOutput(
            prediction=reference,
            reference=reference,
            mono_a=batch.mono_response_a,
            mono_b=batch.mono_response_b,
            interaction_logit=zeros,
            boundary_gate=boundary_gate,
            log_variance=zeros,
        )


class MorganHallmarkResidualMLP(nn.Module):
    """Symmetric Morgan/Hallmark MLP around an observed Bliss reference.

    Fingerprints are supplied explicitly because ``SurfaceBatch`` intentionally
    contains molecular graphs rather than precomputed Morgan features.
    """

    def __init__(
        self,
        fingerprint_dim: int,
        hallmark_dim: int,
        hidden_dim: int = 256,
        dose_frequencies: int = 4,
        dropout: float = 0.0,
        reference_mode: str = "observed_product",
        monotherapy_upper_bound: float = 1.0,
    ) -> None:
        super().__init__()
        if fingerprint_dim < 1 or hallmark_dim < 1 or hidden_dim < 1:
            raise ValueError("Input and hidden dimensions must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")
        if reference_mode not in {"observed_product", "observed_hsa"}:
            raise ValueError(
                "reference_mode must be 'observed_product' or 'observed_hsa'"
            )
        self.fingerprint_dim = int(fingerprint_dim)
        self.hallmark_dim = int(hallmark_dim)
        self.reference_mode = reference_mode
        self.monotherapy_upper_bound = float(monotherapy_upper_bound)
        self.dose_features = DoseFeatures(dose_frequencies)
        # The observed single-agent value is an inference-time covariate in this
        # benchmark, not an auxiliary target.  Putting it in the shared arm lets
        # the residual depend on the local mono curve while preserving A/B
        # exchange symmetry.
        arm_dim = self.fingerprint_dim + self.dose_features.output_dim + 1
        self.shared_arm = nn.Sequential(
            nn.LayerNorm(arm_dim),
            nn.Linear(arm_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.context_encoder = nn.Sequential(
            nn.LayerNorm(self.hallmark_dim),
            nn.Linear(self.hallmark_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.residual_network = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.residual_network[-1].weight)
        nn.init.zeros_(self.residual_network[-1].bias)

    @staticmethod
    def _validate_features(
        fingerprint_a: Tensor,
        fingerprint_b: Tensor,
        hallmark: Tensor,
        fingerprint_dim: int,
        hallmark_dim: int,
    ) -> None:
        if fingerprint_a.ndim != 2 or fingerprint_b.ndim != 2:
            raise ValueError("Morgan fingerprints must have shape [B,F]")
        if fingerprint_a.shape != fingerprint_b.shape:
            raise ValueError("Both Morgan fingerprint tensors must have the same shape")
        if fingerprint_a.shape[-1] != fingerprint_dim:
            raise ValueError(f"Expected Morgan fingerprint dimension {fingerprint_dim}")
        if hallmark.ndim != 2 or hallmark.shape != (
            fingerprint_a.shape[0],
            hallmark_dim,
        ):
            raise ValueError(f"Hallmark features must have shape [B,{hallmark_dim}]")
        features = (fingerprint_a, fingerprint_b, hallmark)
        if not all(tensor.is_floating_point() for tensor in features):
            raise TypeError("Morgan and Hallmark features must be floating-point tensors")
        if not all(bool(torch.isfinite(tensor).all()) for tensor in features):
            raise ValueError("Morgan and Hallmark features must be finite")

    def _arm(self, fingerprint: Tensor, dose: Tensor, mono_response: Tensor) -> Tensor:
        return self.shared_arm(
            torch.cat(
                [fingerprint, self.dose_features(dose), mono_response.unsqueeze(-1)],
                dim=-1,
            )
        )

    def forward(
        self,
        fingerprint_a: Tensor,
        fingerprint_b: Tensor,
        hallmark: Tensor,
        doses: Tensor,
        mono_response_a: Tensor,
        mono_response_b: Tensor,
        mono_mask_a: Tensor,
        mono_mask_b: Tensor,
    ) -> BaselineOutput:
        self._validate_features(
            fingerprint_a,
            fingerprint_b,
            hallmark,
            self.fingerprint_dim,
            self.hallmark_dim,
        )
        _validate_observed_monotherapy(
            doses,
            mono_response_a,
            mono_response_b,
            mono_mask_a,
            mono_mask_b,
            self.monotherapy_upper_bound,
        )
        if fingerprint_a.shape[0] != doses.shape[0]:
            raise ValueError("Feature and response batch sizes must agree")

        arm_a = self._arm(fingerprint_a, doses[:, 0], mono_response_a)
        arm_b = self._arm(fingerprint_b, doses[:, 1], mono_response_b)
        context = self.context_encoder(hallmark)
        invariant = torch.cat(
            [arm_a + arm_b, torch.abs(arm_a - arm_b), arm_a * arm_b, context],
            dim=-1,
        )
        interaction_logit = self.residual_network(invariant).squeeze(-1)
        return _anchored_output(
            doses,
            mono_response_a,
            mono_response_b,
            mono_mask_a,
            mono_mask_b,
            interaction_logit,
            self.reference_mode,
            self.monotherapy_upper_bound,
        )


class IndependentPooledGNN(nn.Module):
    """Independent shared drug GNNs with symmetric pooling and no cross-attention."""

    def __init__(
        self,
        data_config: DataConfig,
        model_config: ModelConfig,
        reference_mode: str = "observed_product",
        monotherapy_upper_bound: float = 1.0,
    ) -> None:
        super().__init__()
        if reference_mode not in {"observed_product", "observed_hsa"}:
            raise ValueError(
                "reference_mode must be 'observed_product' or 'observed_hsa'"
            )
        self.reference_mode = reference_mode
        self.monotherapy_upper_bound = float(monotherapy_upper_bound)
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
        self.dose_features = DoseFeatures(model_config.dose_frequencies)
        arm_dim = hidden_dim + self.dose_features.output_dim + 1
        self.shared_arm = nn.Sequential(
            nn.LayerNorm(arm_dim),
            nn.Linear(arm_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.residual_network = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(model_config.dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.residual_network[-1].weight)
        nn.init.zeros_(self.residual_network[-1].bias)

    def _arm(self, drug: Tensor, dose: Tensor, mono_response: Tensor) -> Tensor:
        return self.shared_arm(
            torch.cat(
                [drug, self.dose_features(dose), mono_response.unsqueeze(-1)],
                dim=-1,
            )
        )

    def forward(self, batch: SurfaceBatch) -> BaselineOutput:
        _validate_observed_monotherapy(
            batch.doses,
            batch.mono_response_a,
            batch.mono_response_b,
            batch.mono_mask_a,
            batch.mono_mask_b,
            self.monotherapy_upper_bound,
        )
        drug_a = self.drug_encoder(batch.drug_a).pooled
        drug_b = self.drug_encoder(batch.drug_b).pooled
        context = self.context_encoder(batch.cell).pooled
        arm_a = self._arm(drug_a, batch.doses[:, 0], batch.mono_response_a)
        arm_b = self._arm(drug_b, batch.doses[:, 1], batch.mono_response_b)
        invariant = torch.cat(
            [arm_a + arm_b, torch.abs(arm_a - arm_b), arm_a * arm_b, context],
            dim=-1,
        )
        interaction_logit = self.residual_network(invariant).squeeze(-1)
        return _anchored_output(
            batch.doses,
            batch.mono_response_a,
            batch.mono_response_b,
            batch.mono_mask_a,
            batch.mono_mask_b,
            interaction_logit,
            self.reference_mode,
            self.monotherapy_upper_bound,
        )
