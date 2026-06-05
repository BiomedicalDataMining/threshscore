"""Precision-Recall curve visualisation."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.metrics import average_precision_score, precision_recall_curve

from threshscore.plotting._style import PALETTE, threshscore_style
from threshscore.utils.types import ArrayLike


def plot_pr(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    title: str = "Precision-Recall Curve",
    label: str | None = None,
    figsize: tuple[float, float] = (5.5, 5.0),
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot the Precision-Recall (PR) curve.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels (0 or 1).
    y_score:
        Predicted probability scores in [0, 1].
    title:
        Plot title.
    label:
        Legend label for the curve (default: ``"PR curve (AP = X.XXX)"``).
    figsize:
        Figure size in inches (used only when *ax* is *None*).
    ax:
        Pre-existing :class:`~matplotlib.axes.Axes` to draw on. If *None* a new
        figure is created.

    Returns
    -------
    (Figure, Axes)
    """
    yt = np.asarray(y_true, dtype=np.int_)
    ys = np.asarray(y_score, dtype=np.float64)

    precision, recall, _ = precision_recall_curve(yt, ys)
    ap = average_precision_score(yt, ys)
    baseline = float(yt.sum()) / len(yt)

    if label is None:
        label = f"PR curve (AP = {ap:.3f})"

    with threshscore_style():
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()  # type: ignore[assignment]

        ax.plot(recall, precision, color=PALETTE["secondary"], linewidth=2.0, label=label)
        ax.axhline(
            baseline, color=PALETTE["neutral"], linewidth=1.2,
            linestyle="--", label=f"Baseline (prevalence = {baseline:.2f})"
        )

        ax.fill_between(recall, precision, alpha=0.10, color=PALETTE["secondary"])

        ax.set_xlabel("Recall (Sensitivity)", fontsize=11)
        ax.set_ylabel("Precision", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(loc="upper right")
        fig.tight_layout()

    return fig, ax
