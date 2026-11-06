"""CompactBiSurfGNN with an additive residual output (CrossScreenAdditive-v1).

Encoders, symmetric BiXT, context gate, decoder, parameters and initialisation are
those of ``CompactBiSurfGNN``; only the output map changes, from the anchored logit
form ``U * sigmoid(logit(r / U) + gate * z)`` to the additive form ``r + gate * z``
in response units. Protocol: ``docs/CROSS_SCREEN_ADDITIVE_V1_PROTOCOL.md``.
"""

from __future__ import annotations

from dataclasses import replace

from .baselines import BaselineOutput
from .compact_bisurf import CompactBiSurfGNN


class AdditiveCompactBiSurfGNN(CompactBiSurfGNN):
    """``CompactBiSurfGNN`` whose correction is added to the observed reference.

    The parent's forward pass is reused unchanged, so the reference ``r`` (observed
    HSA), the boundary gate and the decoder output ``z`` are exactly the parent's.
    The anchored prediction computed there is discarded, so gradients reach the
    network only through ``z``. With the zero-initialised residual head the untrained
    prediction equals ``r`` exactly, as in the parent. The output is not clipped
    because the D1 endpoint is unclipped.
    """

    def forward_pooled(self, *args, **kwargs) -> BaselineOutput:
        output = super().forward_pooled(*args, **kwargs)
        prediction = output.reference + output.boundary_gate * output.interaction_logit
        return replace(output, prediction=prediction)


__all__ = ["AdditiveCompactBiSurfGNN"]
