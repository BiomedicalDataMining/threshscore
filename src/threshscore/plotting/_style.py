"""Shared matplotlib style and colour palette for all threshscore plots."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

import matplotlib as mpl

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

PALETTE = {
    "primary": "#2563EB",       # blue
    "secondary": "#16A34A",     # green
    "accent": "#DC2626",        # red
    "warning": "#D97706",       # amber
    "neutral": "#6B7280",       # grey
    "background": "#F9FAFB",    # very light grey
    "grid": "#E5E7EB",          # light grey
    "text": "#111827",          # near black
    "threshold_line": "#DC2626",  # red: used for threshold indicators
    "target_point": "#7C3AED",  # violet: used for target operating point
}

# Heatmap colourmap: perceptually uniform, appropriate for [0, 1] scores.
SCORE_CMAP = "RdYlGn"
DENSITY_CMAP = "Blues"

# ---------------------------------------------------------------------------
# rcParams overrides
# ---------------------------------------------------------------------------

_RC = {
    "figure.facecolor": "white",
    "axes.facecolor": PALETTE["background"],
    "axes.edgecolor": PALETTE["neutral"],
    "axes.labelcolor": PALETTE["text"],
    "axes.titlecolor": PALETTE["text"],
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.grid": True,
    "grid.color": PALETTE["grid"],
    "grid.linewidth": 0.8,
    "xtick.color": PALETTE["text"],
    "ytick.color": PALETTE["text"],
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.framealpha": 0.9,
    "lines.linewidth": 2.0,
    "lines.antialiased": True,
    "font.family": "sans-serif",
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
}


@contextmanager
def threshscore_style() -> Generator[None, None, None]:
    """Context manager applying the shared threshscore matplotlib style."""
    with mpl.rc_context(_RC):
        yield


def apply_style() -> None:
    """Apply the shared threshscore style globally (mutates rcParams in place)."""
    mpl.rcParams.update(_RC)


def restore_style() -> None:
    """Restore matplotlib defaults."""
    mpl.rcdefaults()
