"""Sigmoid gate function."""

from __future__ import annotations

import math
from typing import Any

from threshscore.gates.base import BaseGate


class SigmoidGate(BaseGate):
    """Logistic sigmoid gate function.

    gate(value, threshold) = 1 / (1 + exp(-k * (value - threshold)))

    clamped to [0, 1].  At the threshold the gate returns exactly 0.5.

    Default k = 15.0.  Increase k for a steeper penalty cliff.
    """

    @property
    def default_params(self) -> dict[str, Any]:
        return {"k": 15.0}

    def __call__(self, value: float, threshold: float, **params: Any) -> float:
        p = self._merged_params(**params)
        k: float = float(p["k"])
        exponent = -k * (value - threshold)
        # Guard against overflow for very large exponents
        if exponent > 500:
            return 0.0
        if exponent < -500:
            return 1.0
        raw = 1.0 / (1.0 + math.exp(exponent))
        return float(max(0.0, min(1.0, raw)))
