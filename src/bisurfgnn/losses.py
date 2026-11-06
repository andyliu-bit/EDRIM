from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .data.batch import SurfaceBatch
from .models.bisurf import BiSurfOutput


@dataclass(frozen=True)
class LossBreakdown:
    total: Tensor
    response_mse: Tensor
    monotherapy_mse: Tensor
    uncertainty_nll: Tensor

    def detached(self) -> dict[str, float]:
        return {
            "loss": float(self.total.detach().cpu()),
            "response_mse": float(self.response_mse.detach().cpu()),
            "monotherapy_mse": float(self.monotherapy_mse.detach().cpu()),
            "uncertainty_nll": float(self.uncertainty_nll.detach().cpu()),
        }


class BiSurfLoss(nn.Module):
    def __init__(self, monotherapy_weight: float, variance_weight: float) -> None:
        super().__init__()
        self.monotherapy_weight = float(monotherapy_weight)
        self.variance_weight = float(variance_weight)

    def forward(self, output: BiSurfOutput, batch: SurfaceBatch) -> LossBreakdown:
        squared_error = (output.prediction - batch.response).square()
        response_mse = squared_error.mean()
        def masked_mse(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
            weights = mask.to(dtype=prediction.dtype)
            if not bool(mask.any()):
                return prediction.sum() * 0.0
            return ((prediction - target).square() * weights).sum() / weights.sum()

        monotherapy_mse = 0.5 * (
            masked_mse(output.mono_a, batch.mono_response_a, batch.mono_mask_a)
            + masked_mse(output.mono_b, batch.mono_response_b, batch.mono_mask_b)
        )
        uncertainty_nll = 0.5 * (
            torch.exp(-output.log_variance) * squared_error + output.log_variance
        ).mean()
        total = (
            response_mse
            + self.monotherapy_weight * monotherapy_mse
            + self.variance_weight * uncertainty_nll
        )
        return LossBreakdown(
            total=total,
            response_mse=response_mse,
            monotherapy_mse=monotherapy_mse,
            uncertainty_nll=uncertainty_nll,
        )
