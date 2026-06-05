"""Score heatmap on the sensitivity-specificity plane."""

from __future__ import annotations

from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from threshscore.core.scoring import _centering_score
from threshscore.gates.base import BaseGate
from threshscore.gates.registry import get as get_gate
from threshscore.plotting._style import PALETTE, SCORE_CMAP, threshscore_style


def plot_heatmap(
    *,
    sensitivity_threshold: float = 0.8,
    specificity_threshold: float = 0.6,
    gate: str | BaseGate = "arctan",
    gate_params: dict[str, Any] | None = None,
    target_sensitivity: float | None = None,
    target_specificity: float | None = None,
    n_grid: int = 200,
    show_contours: bool = True,
    contour_levels: list[float] | None = None,
    title: str = "Score Surface: Sensitivity–Specificity Plane",
    figsize: tuple[float, float] = (7.0, 5.5),
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot the threshscore score surface as a 2-D heatmap.

    The colour encodes the centering-based score computed by the unified
    scoring engine:

        score = _centering_score(gate, sens, spec, s_thr, p_thr)

    The score equals 0.5 at the operating point (sensitivity_threshold,
    specificity_threshold), > 0.5 when both metrics exceed their thresholds,
    and < 0.5 below.

    Parameters
    ----------
    sensitivity_threshold:
        Minimum required sensitivity (drawn as a horizontal dashed line).
    specificity_threshold:
        Minimum required specificity (drawn as a vertical dashed line).
    gate:
        Gate function name or instance.
    gate_params:
        Extra keyword parameters forwarded to the gate function.
    target_sensitivity:
        If given, mark this operating point on the plot (y-coordinate).
    target_specificity:
        If given, mark this operating point on the plot (x-coordinate).
    n_grid:
        Grid resolution (default: 200 × 200 cells).
    show_contours:
        Overlay contour lines at fixed score levels (default: True).
    contour_levels:
        Explicit contour levels to draw (default: 0.2, 0.4, 0.5, 0.6, 0.8, 0.9).
    title:
        Plot title.
    figsize:
        Figure size in inches (used only when *ax* is *None*).
    ax:
        Pre-existing :class:`~matplotlib.axes.Axes` to draw on. If *None* a
        new figure is created.

    Returns
    -------
    (Figure, Axes)
    """
    params = gate_params or {}

    if isinstance(gate, str):
        gate_fn: BaseGate = get_gate(gate)
        gate_name = gate
    else:
        gate_fn = gate
        gate_name = type(gate).__name__

    s_thr = sensitivity_threshold
    p_thr = specificity_threshold

    # Build grid: x = specificity, y = sensitivity
    spec_vals = np.linspace(0.0, 1.0, n_grid)
    sens_vals = np.linspace(0.0, 1.0, n_grid)
    spec_grid, sens_grid = np.meshgrid(spec_vals, sens_vals)

    # Vectorised score computation via the unified scoring engine
    def _score(sens: float, spec: float) -> float:
        return _centering_score(gate_fn, sens, spec, s_thr, p_thr, params)

    score_grid = np.vectorize(_score)(sens_grid, spec_grid)

    if contour_levels is None:
        contour_levels = [0.2, 0.4, 0.5, 0.6, 0.8, 0.9]

    with threshscore_style():
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()  # type: ignore[assignment]

        ax.grid(False)
        # pcolormesh reads rcParams["axes.grid"] directly, not the axes state.
        with mpl.rc_context({"axes.grid": False}):
            im = ax.pcolormesh(
                spec_grid, sens_grid, score_grid,
                cmap=SCORE_CMAP, vmin=0.0, vmax=1.0, shading="auto",
            )
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Score", fontsize=10)
        cbar.locator = mticker.MultipleLocator(0.1)
        cbar.update_ticks()

        # Contour lines
        if show_contours and contour_levels:
            cs = ax.contour(
                spec_grid, sens_grid, score_grid,
                levels=sorted(contour_levels),
                colors="white",
                linewidths=0.8,
                alpha=0.7,
            )
            ax.clabel(cs, fmt="%.2f", fontsize=7, inline=True)

        # Constraint threshold lines
        ax.axhline(
            s_thr,
            color=PALETTE["threshold_line"],
            linestyle="--",
            linewidth=1.6,
            label=f"Min sensitivity = {s_thr:.2f}",
        )
        ax.axvline(
            p_thr,
            color=PALETTE["threshold_line"],
            linestyle="-.",
            linewidth=1.6,
            label=f"Min specificity = {p_thr:.2f}",
        )

        # Target operating point
        if target_sensitivity is not None and target_specificity is not None:
            ax.scatter(
                [target_specificity], [target_sensitivity],
                color=PALETTE["target_point"],
                s=120,
                zorder=5,
                edgecolors="white",
                linewidths=1.2,
                label=(
                    f"Target ({target_specificity:.2f}, {target_sensitivity:.2f})"
                    f" → score={_score(target_sensitivity, target_specificity):.3f}"
                ),
            )

        ax.set_xlabel("Specificity", fontsize=11)
        ax.set_ylabel("Sensitivity", fontsize=11)
        ax.set_title(
            f"{title}\n(gate: {gate_name})", fontsize=12, fontweight="bold"
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()

    return fig, ax
