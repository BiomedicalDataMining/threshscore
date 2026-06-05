"""Threshold analysis plot: metrics across the decision threshold sweep."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from threshscore.core.thresholds import ThresholdSweep, sweep_thresholds
from threshscore.plotting._style import PALETTE, threshscore_style
from threshscore.utils.types import ArrayLike


def plot_threshold(
    sweep: ThresholdSweep | None = None,
    *,
    y_true: ArrayLike | None = None,
    y_score: ArrayLike | None = None,
    n_thresholds: int = 101,
    sensitivity_threshold: float | None = None,
    specificity_threshold: float | None = None,
    title: str = "Threshold Analysis",
    figsize: tuple[float, float] = (7.0, 4.5),
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot sensitivity, specificity, F1 and accuracy vs decision threshold.

    Supply either a pre-computed *sweep* **or** raw ``y_true`` / ``y_score``
    arrays (the sweep is computed internally).

    Parameters
    ----------
    sweep:
        Pre-computed :class:`~threshscore.core.thresholds.ThresholdSweep`.
    y_true:
        Ground-truth binary labels (0 or 1). Used when *sweep* is *None*.
    y_score:
        Predicted probability scores in [0, 1]. Used when *sweep* is *None*.
    n_thresholds:
        Number of thresholds to evaluate when computing from raw arrays.
    sensitivity_threshold:
        If given, draw a horizontal reference line at this sensitivity level.
    specificity_threshold:
        If given, draw a horizontal reference line at this specificity level.
    title:
        Plot title.
    figsize:
        Figure size in inches (used only when *ax* is *None*).
    ax:
        Pre-existing :class:`~matplotlib.axes.Axes` to draw on. If *None* a new
        figure is created.

    Returns
    -------
    (Figure, Axes)
    """
    if sweep is None:
        if y_true is None or y_score is None:
            raise ValueError("Provide either 'sweep' or both 'y_true' and 'y_score'.")
        sweep = sweep_thresholds(y_true, y_score, n_thresholds)

    thresholds = sweep.thresholds
    sensitivities = sweep.sensitivities
    specificities = sweep.specificities
    f1s = np.array([p.f1 for p in sweep.points])
    accuracies = np.array([p.accuracy for p in sweep.points])

    with threshscore_style():
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()  # type: ignore[assignment]

        ax.plot(thresholds, sensitivities, color=PALETTE["primary"], label="Sensitivity")
        ax.plot(thresholds, specificities, color=PALETTE["secondary"], label="Specificity")
        ax.plot(thresholds, f1s, color=PALETTE["warning"], label="F1")
        ax.plot(thresholds, accuracies, color=PALETTE["neutral"],
                linewidth=1.4, linestyle=":", label="Accuracy")

        if sensitivity_threshold is not None:
            ax.axhline(
                sensitivity_threshold,
                color=PALETTE["primary"], linestyle="--", linewidth=1.2, alpha=0.6,
                label=f"Min sensitivity = {sensitivity_threshold:.2f}",
            )
        if specificity_threshold is not None:
            ax.axhline(
                specificity_threshold,
                color=PALETTE["secondary"], linestyle="--", linewidth=1.2, alpha=0.6,
                label=f"Min specificity = {specificity_threshold:.2f}",
            )

        ax.set_xlabel("Decision threshold", fontsize=11)
        ax.set_ylabel("Metric value", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-0.02, 1.05)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
        fig.tight_layout()

    return fig, ax
