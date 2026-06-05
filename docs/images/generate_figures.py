"""Generate README figures for threshscore.

Run from the repository root:
    python docs/images/generate_figures.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import threshscore
from threshscore.gates.arctan import ArctanGate
from threshscore.plotting._style import PALETTE, threshscore_style

OUT = Path(__file__).parent


# ---------------------------------------------------------------------------
# Figure 1: all four gate functions at threshold = 0.80
# ---------------------------------------------------------------------------

fig, ax = threshscore.plot.plot_gate(
    ["arctan", "sigmoid", "relu_clip", "linear_clip"],
    threshold=0.80,
    title="Gate Functions  (threshold = 0.80)",
    figsize=(6.5, 4.0),
)
fig.savefig(OUT / "gates.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved gates.png")


# ---------------------------------------------------------------------------
# Helpers for individual gate component figures
# ---------------------------------------------------------------------------

def _gate_figure(threshold, xlabel, title, outfile):
    gate = ArctanGate()
    x = np.linspace(0.0, 1.0, 500)
    y = np.array([gate(float(v), threshold) for v in x])

    with threshscore_style():
        fig, ax = plt.subplots(figsize=(5.5, 3.8))

        ax.plot(x, y, color=PALETTE["primary"], linewidth=2.5, label="arctan gate")

        ax.axvline(
            threshold,
            color=PALETTE["threshold_line"],
            linestyle="--",
            linewidth=1.5,
            alpha=0.9,
            label=f"threshold = {threshold:.2f}",
        )
        ax.axhline(
            0.5,
            color=PALETTE["neutral"],
            linestyle=":",
            linewidth=1.2,
            alpha=0.8,
            label="output = 0.5",
        )
        ax.axhline(1.0, color=PALETTE["neutral"], linestyle=":", linewidth=0.8, alpha=0.35)
        ax.axhline(0.0, color=PALETTE["neutral"], linestyle=":", linewidth=0.8, alpha=0.35)

        ax.scatter(
            [threshold], [0.5],
            color=PALETTE["target_point"],
            s=90,
            zorder=5,
            label=f"g({threshold:.2f}, {threshold:.2f}) = 0.5",
        )

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Gate output", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-0.05, 1.12)
        ax.legend(loc="upper left", fontsize=9)
        fig.tight_layout()

    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: sensitivity gate  g_s  (arctan, s_thr = 0.80)
# ---------------------------------------------------------------------------

_gate_figure(
    threshold=0.80,
    xlabel="Sensitivity",
    title="Sensitivity gate  g_s  (arctan, s_thr = 0.80)",
    outfile=OUT / "gate_sensitivity.png",
)
print("Saved gate_sensitivity.png")


# ---------------------------------------------------------------------------
# Figure 3: specificity gate  g_p  (arctan, p_thr = 0.60)
# ---------------------------------------------------------------------------

_gate_figure(
    threshold=0.60,
    xlabel="Specificity",
    title="Specificity gate  g_p  (arctan, p_thr = 0.60)",
    outfile=OUT / "gate_specificity.png",
)
print("Saved gate_specificity.png")


# ---------------------------------------------------------------------------
# Figure 4: score heatmap  (arctan, s_thr = 0.80, p_thr = 0.60)
# ---------------------------------------------------------------------------

fig, ax = threshscore.plot.plot_heatmap(
    sensitivity_threshold=0.80,
    specificity_threshold=0.60,
    title="ThreshScore Surface",
    figsize=(6.5, 5.0),
)
fig.savefig(OUT / "heatmap.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved heatmap.png")
