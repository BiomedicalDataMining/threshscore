"""threshscore.plotting: publication-quality plot functions.

All functions return ``(Figure, Axes)`` and accept an optional ``ax`` kwarg
so callers can embed plots into larger figures.
"""

from __future__ import annotations

from threshscore.plotting.confusion import plot_confusion
from threshscore.plotting.gate import plot_gate
from threshscore.plotting.heatmap import plot_heatmap
from threshscore.plotting.pr import plot_pr
from threshscore.plotting.roc import plot_roc
from threshscore.plotting.threshold import plot_threshold

__all__ = [
    "plot_gate",
    "plot_confusion",
    "plot_heatmap",
    "plot_pr",
    "plot_roc",
    "plot_threshold",
]
