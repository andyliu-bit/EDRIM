"""Same-network controls for anchored residual output geometry."""
from dataclasses import replace

import torch

from .baselines import _anchored_output
from .compact_bisurf import CompactBiSurfGNN


class ResidualParameterizationBiSurf(CompactBiSurfGNN):
    def __init__(self, *args, output_mode: str, **kwargs):
        super().__init__(*args, **kwargs)
        if output_mode not in {"additive", "logit_gain4"}:
            raise ValueError("Unknown residual output mode")
        self.output_mode = output_mode

    def forward_pooled(self, drug_bank, context_bank, drug_a_index, drug_b_index,
                       cell_index, doses, mono_a, mono_b, mono_mask_a, mono_mask_b):
        output = super().forward_pooled(drug_bank, context_bank, drug_a_index,
            drug_b_index, cell_index, doses, mono_a, mono_b, mono_mask_a, mono_mask_b)
        if self.output_mode == "logit_gain4":
            return _anchored_output(doses, mono_a, mono_b, mono_mask_a, mono_mask_b,
                4.0 * output.interaction_logit, self.reference_mode, self.monotherapy_upper_bound)
        candidate = torch.clamp(output.reference + output.boundary_gate * output.interaction_logit,
                                min=0., max=self.monotherapy_upper_bound)
        prediction = torch.where(output.boundary_gate == 0, output.reference, candidate)
        return replace(output, prediction=prediction)
