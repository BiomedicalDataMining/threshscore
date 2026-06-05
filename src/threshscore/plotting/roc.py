"""ROC curve visualisation."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.metrics import roc_auc_score, roc_curve

from threshscore.plotting._style import PALETTE, threshscore_style
from threshscore.utils.types import ArrayLike


def plot_roc(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    title: str = "ROC Curve",
    label: str | None = None,
    figsize: tuple[float, float] = (5.5, 5.0),
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot the Receiver Operating Characteristic (ROC) curve.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels (0 or 1).
    y_score:
        Predicted probability scores in [0, 1].
    title:
        Plot title.
    label:
        Legend label for the curve (default: ``"ROC (AUC = X.XXX)"``).
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

    fpr, tpr, _ = roc_curve(yt, ys)
    auc = roc_auc_score(yt, ys)

    if label is None:
        label = f"ROC (AUC = {auc:.3f})"

    with threshscore_style():
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()  # type: ignore[assignment]

        ax.plot(fpr, tpr, color=PALETTE["primary"], linewidth=2.0, label=label)
        ax.plot(
            [0, 1], [0, 1],
            color=PALETTE["neutral"], linewidth=1.2, linestyle="--", label="Random"
        )

        ax.fill_between(fpr, tpr, alpha=0.10, color=PALETTE["primary"])

        ax.set_xlabel("False Positive Rate (1 – Specificity)", fontsize=11)
        ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(loc="lower right")
        fig.tight_layout()

    return fig, ax
