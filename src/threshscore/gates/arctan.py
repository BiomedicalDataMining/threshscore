"""Arctan gate function."""

from __future__ import annotations

import math
from typing import Any

from threshscore.gates.base import BaseGate


class ArctanGate(BaseGate):
    """Smooth arctan-based gate function.

    gate(value, threshold) = arctan(k * (value - threshold)) / π + 0.5

    clamped to [0, 1].  At the threshold the gate returns exactly 0.5;
    above the threshold it approaches 1, below it approaches 0.

    Default k = 25.0 (sharp transition around the threshold).

    The centering-based score formula that combines sensitivity and specificity
    gates is implemented in the scoring engine (``core/scoring.py``), not here.
    """

    @property
    def default_params(self) -> dict[str, Any]:
        return {"k": 25.0}

    def __call__(self, value: float, threshold: float, **params: Any) -> float:
        p = self._merged_params(**params)
        k: float = float(p["k"])
        raw = math.atan(k * (value - threshold)) / math.pi + 0.5
        return float(max(0.0, min(1.0, raw)))
