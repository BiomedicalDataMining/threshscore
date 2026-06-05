"""threshscore: threshold-aware binary classification scoring."""

from __future__ import annotations

import threshscore.gates as gates
import threshscore.plotting as plot
from threshscore.core.metrics import ClassificationMetrics, compute_metrics
from threshscore.core.scoring import (
    ScoreResult,
    score_from_metrics,
    score_from_predictions,
    score_from_scores,
)
from threshscore.core.thresholds import ThresholdSweep, sweep_thresholds
from threshscore.gates.base import BaseGate
from threshscore.utils.types import ArrayLike

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "score_from_metrics",
    "score_from_predictions",
    "score_from_scores",
    "compute_metrics",
    "sweep",
    "plot",
    "gates",
    "ScoreResult",
    "ClassificationMetrics",
    "ThresholdSweep",
    "BaseGate",
]


def sweep(
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
        Number of thresholds to evaluate (default: 101).

    Returns
    -------
    ThresholdSweep
    """
    return sweep_thresholds(y_true, y_score, n_thresholds)
