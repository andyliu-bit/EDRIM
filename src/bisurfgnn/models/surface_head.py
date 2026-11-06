from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class DoseFeatures(nn.Module):
    def __init__(self, frequencies: int) -> None:
        super().__init__()
        if frequencies < 1:
            raise ValueError("At least one dose frequency is required")
        values = math.pi * (2.0 ** torch.arange(frequencies, dtype=torch.float32))
        self.register_buffer("frequencies", values, persistent=True)
        self.output_dim = 2 + 2 * frequencies

    def forward(self, dose: Tensor) -> Tensor:
        if dose.ndim != 1:
            raise ValueError("DoseFeatures expects a [B] tensor")
        non_negative = dose.clamp_min(0.0)
        angles = non_negative.unsqueeze(-1) * self.frequencies
        return torch.cat(
            [
                non_negative.unsqueeze(-1),
                torch.log1p(non_negative).unsqueeze(-1),
                torch.sin(angles),
                torch.cos(angles),
            ],
            dim=-1,
        )


class SharedMonotherapyHead(nn.Module):
    def __init__(self, hidden_dim: int, dose_frequencies: int, dropout: float) -> None:
        super().__init__()
        self.dose_features = DoseFeatures(dose_frequencies)
        input_dim = hidden_dim * 2 + self.dose_features.output_dim
        self.rate_network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 1),
        )

    def forward(self, drug: Tensor, context: Tensor, dose: Tensor) -> Tensor:
        features = torch.cat([drug, context, self.dose_features(dose)], dim=-1)
        rate = F.softplus(self.rate_network(features).squeeze(-1)) + 1e-4
        return torch.exp(-dose.clamp_min(0.0) * rate)


class ContinuousInteractionDecoder(nn.Module):
    def __init__(
        self, hidden_dim: int, dose_frequencies: int, dropout: float,
        monotherapy_conditioned: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(monotherapy_conditioned, bool):
            raise TypeError("monotherapy_conditioned must be boolean")
        self.monotherapy_conditioned = monotherapy_conditioned
        self.dose_features = DoseFeatures(dose_frequencies)
        arm_input_dim = hidden_dim * 2 + self.dose_features.output_dim + int(monotherapy_conditioned)
        self.shared_dose_arm = nn.Sequential(
            nn.LayerNorm(arm_input_dim),
            nn.Linear(arm_input_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        interaction_dim = hidden_dim * 5
        self.interaction_network = nn.Sequential(
            nn.LayerNorm(interaction_dim),
            nn.Linear(interaction_dim, hidden_dim * 3),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
        )
        self.residual_head = nn.Linear(hidden_dim, 1)
        self.log_variance_head = nn.Linear(hidden_dim, 1)

    def _arm(
        self, drug: Tensor, context: Tensor, dose: Tensor, mono: Tensor | None = None
    ) -> Tensor:
        features = [drug, context, self.dose_features(dose)]
        if self.monotherapy_conditioned:
            if mono is None or mono.shape != dose.shape:
                raise ValueError("Conditioned decoder requires aligned monotherapy responses")
            if not bool(torch.isfinite(mono).all()):
                raise ValueError("Monotherapy responses must be finite")
            features.append(mono.unsqueeze(-1))
        elif mono is not None:
            raise ValueError("Legacy decoder does not accept monotherapy covariates")
        return self.shared_dose_arm(torch.cat(features, dim=-1))

    def forward(
        self,
        drug_a: Tensor,
        drug_b: Tensor,
        context: Tensor,
        conditioned_pair: Tensor,
        doses: Tensor,
        mono_a: Tensor | None = None,
        mono_b: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if doses.ndim != 2 or doses.shape[-1] != 2:
            raise ValueError("doses must have shape [B,2]")
        arm_a = self._arm(drug_a, context, doses[:, 0], mono_a)
        arm_b = self._arm(drug_b, context, doses[:, 1], mono_b)
        invariant = torch.cat(
            [
                arm_a + arm_b,
                torch.abs(arm_a - arm_b),
                arm_a * arm_b,
                conditioned_pair,
                context,
            ],
            dim=-1,
        )
        hidden = self.interaction_network(invariant)
        residual = self.residual_head(hidden).squeeze(-1)
        log_variance = self.log_variance_head(hidden).squeeze(-1).clamp(-8.0, 2.0)
        return residual, log_variance
