"""Threshold sweep and analysis utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from threshscore.utils.types import ArrayLike, FloatArray


@dataclass(frozen=True)
class ThresholdPoint:
    """Metrics at a single operating threshold."""

    threshold: float
    sensitivity: float
    specificity: float
    precision: float
    f1: float
    accuracy: float
    n_predicted_positive: int


@dataclass(frozen=True)
class ThresholdSweep:
    """Result of sweeping a range of classification thresholds."""

    points: list[ThresholdPoint]
    thresholds: FloatArray
    sensitivities: FloatArray
    specificities: FloatArray

    def best_by_f1(self) -> ThresholdPoint:
        """Return the threshold point with the highest F1 score."""
        return max(self.points, key=lambda p: p.f1)

    def best_by_sensitivity_at_specificity(
        self, min_specificity: float
    ) -> ThresholdPoint | None:
        """Return highest-sensitivity point meeting a minimum specificity."""
        candidates = [p for p in self.points if p.specificity >= min_specificity]
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.sensitivity)

    def best_by_specificity_at_sensitivity(
        self, min_sensitivity: float
    ) -> ThresholdPoint | None:
        """Return highest-specificity point meeting a minimum sensitivity."""
        candidates = [p for p in self.points if p.sensitivity >= min_sensitivity]
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.specificity)


def sweep_thresholds(
    y_true: ArrayLike,
    y_score: ArrayLike,
    n_thresholds: int = 101,
) -> ThresholdSweep:
    """Sweep classification thresholds and compute metrics at each.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels (0 or 1).
    y_score:
        Predicted probability scores in [0, 1].
    n_thresholds:
        Number of threshold values to evaluate (default: 101 → 0.00 to 1.00).

    Returns
    -------
    ThresholdSweep
    """
    yt = np.asarray(y_true, dtype=np.int_)
    ys = np.asarray(y_score, dtype=np.float64)

    if yt.ndim != 1 or ys.ndim != 1:
        raise ValueError("y_true and y_score must be 1-D")
    if len(yt) != len(ys):
        raise ValueError("y_true and y_score must have the same length")
    if len(np.unique(yt)) < 2:
        raise ValueError("y_true must contain both classes")

    thresholds: FloatArray = np.linspace(0.0, 1.0, n_thresholds)  # type: ignore[assignment]
    points: list[ThresholdPoint] = []
    sensitivities: list[float] = []
    specificities: list[float] = []

    for thresh in thresholds:
        yp = (ys >= thresh).astype(np.int_)
        tp = float(np.sum((yt == 1) & (yp == 1)))
        tn = float(np.sum((yt == 0) & (yp == 0)))
        fp = float(np.sum((yt == 0) & (yp == 1)))
        fn = float(np.sum((yt == 1) & (yp == 0)))

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
        acc = (tp + tn) / len(yt)

        sensitivities.append(sens)
        specificities.append(spec)
        points.append(
            ThresholdPoint(
                threshold=float(thresh),
                sensitivity=sens,
                specificity=spec,
                precision=prec,
                f1=f1,
                accuracy=acc,
                n_predicted_positive=int(np.sum(yp)),
            )
        )

    return ThresholdSweep(
        points=points,
        thresholds=thresholds,
        sensitivities=np.array(sensitivities, dtype=np.float64),
        specificities=np.array(specificities, dtype=np.float64),
    )
