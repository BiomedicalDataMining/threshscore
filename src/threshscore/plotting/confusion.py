"""Confusion matrix visualisation."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from threshscore.plotting._style import PALETTE, threshscore_style


def plot_confusion(
    confusion_matrix: np.ndarray,  # type: ignore[type-arg]
    *,
    labels: tuple[str, str] | None = None,
    title: str = "Confusion Matrix",
    normalise: bool = False,
    figsize: tuple[float, float] = (5.0, 4.5),
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot a 2×2 binary confusion matrix.

    Parameters
    ----------
    confusion_matrix:
        2×2 numpy array ``[[TN, FP], [FN, TP]]`` as returned by
        :func:`threshscore.core.metrics.compute_metrics`.
    labels:
        ``(negative_label, positive_label)`` class names (default: ``("Negative", "Positive")``).
    title:
        Plot title.
    normalise:
        If *True*, normalise each row so cells show fractions instead of counts.
    figsize:
        Figure size in inches (used only when *ax* is *None*).
    ax:
        Pre-existing :class:`~matplotlib.axes.Axes` to draw on. If *None* a new
        figure is created.

    Returns
    -------
    (Figure, Axes)
    """
    if labels is None:
        labels = ("Negative", "Positive")

    cm = np.asarray(confusion_matrix, dtype=np.float64)
    if cm.shape != (2, 2):
        raise ValueError(f"confusion_matrix must be 2×2, got shape {cm.shape}")

    if normalise:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.where(row_sums > 0, cm / row_sums, 0.0)
        fmt = ".2f"
        vmax = 1.0
    else:
        fmt = ".0f"
        vmax = float(cm.max())

    with threshscore_style():
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()  # type: ignore[assignment]

        ax.grid(False)
        # fig.colorbar() renders via pcolormesh internally and reads rcParams["axes.grid"] directly.
        with mpl.rc_context({"axes.grid": False}):
            im = ax.imshow(cm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=vmax)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        tick_marks = [0, 1]
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted label", fontsize=11)
        ax.set_ylabel("True label", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")

        thresh = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                cell_label = f"{cm[i, j]:{fmt}}"
                ax.text(
                    j, i, cell_label,
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else PALETTE["text"],
                    fontsize=12, fontweight="bold",
                )

        # Annotate corner cells with TP/TN/FP/FN
        corner_map = {(0, 0): "TN", (0, 1): "FP", (1, 0): "FN", (1, 1): "TP"}
        for (row, col), abbrev in corner_map.items():
            ax.text(
                col + 0.35, row - 0.35, abbrev,
                ha="right", va="top",
                color="white" if cm[row, col] > thresh else PALETTE["neutral"],
                fontsize=7,
            )

        ax.grid(False)
        fig.tight_layout()

    return fig, ax
