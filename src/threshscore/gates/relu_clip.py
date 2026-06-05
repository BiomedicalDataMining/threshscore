"""ReLU-clip gate function."""

from __future__ import annotations

from typing import Any

from threshscore.gates.base import BaseGate


class ReluClipGate(BaseGate):
    """ReLU-clip gate function: hard zero below threshold, linear above.

    gate(value, threshold) = 0.0                                       if value < threshold
                           = 0.5 + 0.5*(value - threshold)/(1 - threshold)  otherwise

    Returns exactly 0.5 at the threshold and 1.0 at value = 1.0,
    matching the [0, 1] convention of smooth gates.  Values below
    the threshold receive a hard zero (no partial credit).
    """

    @property
    def default_params(self) -> dict[str, Any]:
        return {}

    def __call__(self, value: float, threshold: float, **params: Any) -> float:
        if value < threshold:
            return 0.0
        if threshold <= 0.0:
            return 1.0
        denom = 1.0 - threshold
        if denom <= 0.0:
            return 1.0
        return float(min(1.0, 0.5 + 0.5 * (value - threshold) / denom))
