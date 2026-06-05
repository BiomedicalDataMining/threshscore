"""Base class for gate functions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseGate(ABC):
    """Abstract base class for gate functions.

    A gate function maps a single metric value and its target threshold
    to a gate value in [0, 1], always returning 0.5 when value equals threshold.
    The centering-based score formula is applied externally by the scoring engine.

    Subclasses must implement:
    - ``__call__``: the gate function (value, threshold) → [0, 1].
    - ``default_params``: default keyword parameters.
    """

    @property
    @abstractmethod
    def default_params(self) -> dict[str, Any]:
        """Default keyword parameters for this gate function."""

    @abstractmethod
    def __call__(self, value: float, threshold: float, **params: Any) -> float:
        """Apply the gate function.

        Maps a single metric value and its target threshold to a gate value
        in [0, 1].  Used by the scoring engine to compute g_s and g_p.

        Parameters
        ----------
        value:
            The achieved metric value (e.g., sensitivity = 0.75).
        threshold:
            The target threshold (e.g., minimum sensitivity = 0.80).
        **params:
            Override parameters.

        Returns
        -------
        float
            Gate value in [0, 1].
        """

    def _merged_params(self, **overrides: Any) -> dict[str, Any]:
        """Return default_params with overrides applied."""
        merged = dict(self.default_params)
        merged.update(overrides)
        return merged
