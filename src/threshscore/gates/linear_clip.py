"""Linear-clip gate function."""

from __future__ import annotations

from typing import Any

from threshscore.gates.base import BaseGate


class LinearClipGate(BaseGate):
    """Linear-clip gate function: proportional partial credit below threshold.

    gate(value, threshold) = 0.5 * value / threshold               if value < threshold
                           = 0.5 + 0.5*(value - threshold)/(1 - threshold)  otherwise

    Returns exactly 0.5 at the threshold and 1.0 at value = 1.0,
    matching the [0, 1] convention of smooth gates.  Unlike relu_clip,
    values below the threshold receive proportional partial credit in [0, 0.5).
    """

    @property
    def default_params(self) -> dict[str, Any]:
        return {}

    def __call__(self, value: float, threshold: float, **params: Any) -> float:
        if threshold <= 0.0:
            return 1.0
        if value >= threshold:
            denom = 1.0 - threshold
            if denom <= 0.0:
                return 1.0
            return float(min(1.0, 0.5 + 0.5 * (value - threshold) / denom))
        return float(max(0.0, 0.5 * value / threshold))
