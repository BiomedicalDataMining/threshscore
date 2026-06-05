"""Smoke tests for the threshscore plotting suite.

Each test verifies:
- The function runs without error.
- The return type is (Figure, Axes).
- Key axes properties (labels, title) are set.

A non-interactive Agg backend is forced so tests run headlessly.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # non-interactive backend: must be set before pyplot import

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import threshscore
import threshscore.plotting as plot
from threshscore.plotting.confusion import plot_confusion
from threshscore.plotting.gate import plot_gate
from threshscore.plotting.heatmap import plot_heatmap
from threshscore.plotting.pr import plot_pr
from threshscore.plotting.roc import plot_roc
from threshscore.plotting.threshold import plot_threshold

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
N = 200

@pytest.fixture()
def binary_data():
    y_true = RNG.integers(0, 2, size=N)
    # Scores correlated with labels so both classes are represented
    y_score = np.clip(y_true * 0.6 + RNG.uniform(0, 0.4, N), 0.0, 1.0)
    y_pred = (y_score >= 0.5).astype(int)
    return y_true, y_pred, y_score


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test to free memory."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

class TestPlotConfusion:
    def test_returns_figure_axes(self):
        cm = np.array([[50, 10], [5, 35]])
        fig, ax = plot_confusion(cm)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_normalised(self):
        cm = np.array([[50, 10], [5, 35]])
        fig, ax = plot_confusion(cm, normalise=True)
        assert isinstance(fig, Figure)

    def test_custom_labels_and_title(self):
        cm = np.array([[40, 5], [3, 52]])
        fig, ax = plot_confusion(cm, labels=("Healthy", "Sick"), title="My CM")
        assert ax.get_title() == "My CM"

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError):
            plot_confusion(np.ones((3, 3)))

    def test_ax_injection(self):
        _, existing_ax = plt.subplots()
        cm = np.array([[30, 5], [2, 63]])
        fig, ax = plot_confusion(cm, ax=existing_ax)
        assert ax is existing_ax

    def test_public_api_accessible(self):
        cm = np.array([[10, 2], [1, 7]])
        fig, ax = plot.plot_confusion(cm)
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# ROC curve
# ---------------------------------------------------------------------------

class TestPlotRoc:
    def test_returns_figure_axes(self, binary_data):
        y_true, _, y_score = binary_data
        fig, ax = plot_roc(y_true, y_score)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_xlabel_ylabel(self, binary_data):
        y_true, _, y_score = binary_data
        _, ax = plot_roc(y_true, y_score)
        assert "False Positive Rate" in ax.get_xlabel()
        assert "True Positive Rate" in ax.get_ylabel()

    def test_custom_label(self, binary_data):
        y_true, _, y_score = binary_data
        _, ax = plot_roc(y_true, y_score, label="Custom ROC")
        labels = [line.get_label() for line in ax.get_lines()]
        assert "Custom ROC" in labels

    def test_ax_injection(self, binary_data):
        y_true, _, y_score = binary_data
        _, existing_ax = plt.subplots()
        fig, ax = plot_roc(y_true, y_score, ax=existing_ax)
        assert ax is existing_ax

    def test_public_api_accessible(self, binary_data):
        y_true, _, y_score = binary_data
        fig, ax = plot.plot_roc(y_true, y_score)
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# Precision-Recall curve
# ---------------------------------------------------------------------------

class TestPlotPr:
    def test_returns_figure_axes(self, binary_data):
        y_true, _, y_score = binary_data
        fig, ax = plot_pr(y_true, y_score)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_xlabel_ylabel(self, binary_data):
        y_true, _, y_score = binary_data
        _, ax = plot_pr(y_true, y_score)
        assert "Recall" in ax.get_xlabel()
        assert "Precision" in ax.get_ylabel()

    def test_custom_label(self, binary_data):
        y_true, _, y_score = binary_data
        _, ax = plot_pr(y_true, y_score, label="Custom PR")
        labels = [line.get_label() for line in ax.get_lines()]
        assert "Custom PR" in labels

    def test_ax_injection(self, binary_data):
        y_true, _, y_score = binary_data
        _, existing_ax = plt.subplots()
        fig, ax = plot_pr(y_true, y_score, ax=existing_ax)
        assert ax is existing_ax

    def test_public_api_accessible(self, binary_data):
        y_true, _, y_score = binary_data
        fig, ax = plot.plot_pr(y_true, y_score)
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# Threshold analysis
# ---------------------------------------------------------------------------

class TestPlotThreshold:
    def test_from_raw_arrays(self, binary_data):
        y_true, _, y_score = binary_data
        fig, ax = plot_threshold(y_true=y_true, y_score=y_score)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_from_sweep(self, binary_data):
        y_true, _, y_score = binary_data
        sw = threshscore.sweep(y_true, y_score)
        fig, ax = plot_threshold(sw)
        assert isinstance(fig, Figure)

    def test_with_reference_lines(self, binary_data):
        y_true, _, y_score = binary_data
        fig, ax = plot_threshold(
            y_true=y_true, y_score=y_score,
            sensitivity_threshold=0.7,
            specificity_threshold=0.6,
        )
        assert isinstance(fig, Figure)

    def test_missing_inputs_raises(self):
        with pytest.raises(ValueError):
            plot_threshold()

    def test_ax_injection(self, binary_data):
        y_true, _, y_score = binary_data
        _, existing_ax = plt.subplots()
        fig, ax = plot_threshold(y_true=y_true, y_score=y_score, ax=existing_ax)
        assert ax is existing_ax

    def test_public_api_accessible(self, binary_data):
        y_true, _, y_score = binary_data
        fig, ax = plot.plot_threshold(y_true=y_true, y_score=y_score)
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# Gate curves
# ---------------------------------------------------------------------------

class TestPlotGate:
    def test_default_all_gates(self):
        fig, ax = plot_gate()
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_single_gate_by_name(self):
        fig, ax = plot_gate("arctan")
        assert isinstance(fig, Figure)

    def test_multiple_gates(self):
        fig, ax = plot_gate(["arctan", "sigmoid", "relu_clip"])
        assert isinstance(fig, Figure)
        # Should have at least 3 labelled lines (plus threshold reference)
        labelled = [line for line in ax.get_lines() if not line.get_label().startswith("_")]
        assert len(labelled) >= 3

    def test_custom_threshold(self):
        fig, ax = plot_gate("sigmoid", threshold=0.5)
        assert isinstance(fig, Figure)

    def test_gate_params_forwarded(self):
        fig, ax = plot_gate("arctan", gate_params={"k": 50.0})
        assert isinstance(fig, Figure)

    def test_ax_injection(self):
        _, existing_ax = plt.subplots()
        fig, ax = plot_gate("sigmoid", ax=existing_ax)
        assert ax is existing_ax

    def test_xlabel_ylabel(self):
        _, ax = plot_gate("sigmoid")
        assert "Metric value" in ax.get_xlabel()
        assert "Gate" in ax.get_ylabel()

    def test_public_api_accessible(self):
        fig, ax = plot.plot_gate()
        assert isinstance(fig, Figure)

    def test_top_level_plot_namespace(self):
        """threshscore.plot.plot_gate is accessible from the package root."""
        fig, ax = threshscore.plot.plot_gate("arctan")
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# Score heatmap
# ---------------------------------------------------------------------------

class TestPlotHeatmap:
    def test_default(self):
        fig, ax = plot_heatmap()
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_xlabel_ylabel(self):
        _, ax = plot_heatmap()
        assert "Specificity" in ax.get_xlabel()
        assert "Sensitivity" in ax.get_ylabel()

    def test_custom_thresholds(self):
        fig, ax = plot_heatmap(
            sensitivity_threshold=0.7,
            specificity_threshold=0.5,
        )
        assert isinstance(fig, Figure)

    def test_with_target_point(self):
        fig, ax = plot_heatmap(
            target_sensitivity=0.85,
            target_specificity=0.70,
        )
        assert isinstance(fig, Figure)

    def test_no_contours(self):
        fig, ax = plot_heatmap(show_contours=False)
        assert isinstance(fig, Figure)

    def test_different_gate(self):
        fig, ax = plot_heatmap(gate="sigmoid")
        assert isinstance(fig, Figure)

    def test_gate_params(self):
        fig, ax = plot_heatmap(gate="arctan", gate_params={"k": 10.0})
        assert isinstance(fig, Figure)

    def test_ax_injection(self):
        _, existing_ax = plt.subplots()
        fig, ax = plot_heatmap(ax=existing_ax)
        assert ax is existing_ax

    def test_public_api_accessible(self):
        fig, ax = plot.plot_heatmap()
        assert isinstance(fig, Figure)

    def test_top_level_plot_namespace(self):
        """threshscore.plot.plot_heatmap is accessible from the package root."""
        fig, ax = threshscore.plot.plot_heatmap()
        assert isinstance(fig, Figure)
