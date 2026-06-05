"""Gate function visualisation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from threshscore.gates.base import BaseGate
from threshscore.gates.registry import get as get_gate
from threshscore.gates.registry import list_gates
from threshscore.plotting._style import PALETTE, threshscore_style

_CURVE_COLOURS = [
    PALETTE["primary"],
    PALETTE["secondary"],
    PALETTE["warning"],
    PALETTE["accent"],
    PALETTE["neutral"],
]


def plot_gate(
    gates: str | BaseGate | Sequence[str | BaseGate] | None = None,
    *,
    threshold: float = 0.8,
    gate_params: dict[str, Any] | list[dict[str, Any] | None] | None = None,
    n_points: int = 500,
    title: str = "Gate Functions",
    figsize: tuple[float, float] = (7.0, 4.5),
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot one or more gate function curves over the metric value range [0, 1].

    Parameters
    ----------
    gates:
        Gate function name(s) or instance(s) to plot. Pass a single string/instance
        or a list of them. If *None*, all registered gate functions are plotted.
    threshold:
        The target threshold at which the gate is anchored (shown as a
        vertical dashed line, default: 0.8).
    gate_params:
        Extra keyword parameters forwarded to the gate function(s). A single dict
        is shared across all curves; a list of dicts applies per-curve (use
        *None* entries to use defaults for that curve).
    n_points:
        Number of sample points across [0, 1] (default: 500).
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
    if gates is None:
        gate_list: list[tuple[str, BaseGate]] = [
            (name, get_gate(name)) for name in list_gates()
        ]
    else:
        raw = gates if isinstance(gates, (list, tuple)) else [gates]
        gate_list = []
        for item in raw:
            if isinstance(item, str):
                gate_list.append((item, get_gate(item)))
            else:
                gate_list.append((type(item).__name__, item))  # type: ignore[arg-type]

    if gate_params is None:
        params_list: list[dict[str, Any]] = [{} for _ in gate_list]
    elif isinstance(gate_params, dict):
        params_list = [gate_params] * len(gate_list)
    else:
        padded = list(gate_params) + [None] * (len(gate_list) - len(gate_params))
        params_list = [p if p is not None else {} for p in padded]

    x = np.linspace(0.0, 1.0, n_points)

    with threshscore_style():
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()  # type: ignore[assignment]

        for idx, ((label, gate_fn), params) in enumerate(zip(gate_list, params_list)):
            colour = _CURVE_COLOURS[idx % len(_CURVE_COLOURS)]
            y = np.array([gate_fn(float(v), threshold, **params) for v in x])
            ax.plot(x, y, color=colour, linewidth=2.0, label=label)

        ax.axvline(
            threshold,
            color=PALETTE["threshold_line"],
            linestyle="--",
            linewidth=1.4,
            alpha=0.8,
            label=f"Threshold = {threshold:.2f}",
        )
        ax.axhline(1.0, color=PALETTE["neutral"], linestyle=":", linewidth=1.0, alpha=0.5)
        ax.axhline(0.0, color=PALETTE["neutral"], linestyle=":", linewidth=1.0, alpha=0.5)

        ax.set_xlabel("Metric value", fontsize=11)
        ax.set_ylabel("Gate output", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-0.05, 1.08)
        ax.legend(loc="upper left")
        fig.tight_layout()

    return fig, ax
