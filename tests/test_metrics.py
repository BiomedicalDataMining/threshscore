"""Tests for core metrics module."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

import threshscore
from threshscore.core.metrics import (
    ClassificationMetrics,
    compute_metrics,
    sensitivity_score,
    specificity_score,
)

# Simple fixture data
Y_TRUE = np.array([0, 0, 1, 1, 1, 0, 1, 0])
Y_PRED = np.array([0, 1, 1, 1, 0, 0, 1, 0])
Y_SCORE = np.array([0.1, 0.6, 0.8, 0.9, 0.4, 0.2, 0.7, 0.3])


class TestComputeMetrics:
    def test_returns_dataclass(self) -> None:
        result = compute_metrics(Y_TRUE, Y_PRED)
        assert isinstance(result, ClassificationMetrics)

    def test_sensitivity_range(self) -> None:
        result = compute_metrics(Y_TRUE, Y_PRED)
        assert 0.0 <= result.sensitivity <= 1.0

    def test_specificity_range(self) -> None:
        result = compute_metrics(Y_TRUE, Y_PRED)
        assert 0.0 <= result.specificity <= 1.0

    def test_precision_range(self) -> None:
        result = compute_metrics(Y_TRUE, Y_PRED)
        assert 0.0 <= result.precision <= 1.0

    def test_f1_range(self) -> None:
        result = compute_metrics(Y_TRUE, Y_PRED)
        assert 0.0 <= result.f1 <= 1.0

    def test_accuracy_range(self) -> None:
        result = compute_metrics(Y_TRUE, Y_PRED)
        assert 0.0 <= result.accuracy <= 1.0

    def test_roc_auc_none_without_score(self) -> None:
        result = compute_metrics(Y_TRUE, Y_PRED)
        assert result.roc_auc is None
        assert result.pr_auc is None

    def test_roc_auc_with_score(self) -> None:
        result = compute_metrics(Y_TRUE, Y_PRED, Y_SCORE)
        assert result.roc_auc is not None
        assert 0.0 <= result.roc_auc <= 1.0

    def test_pr_auc_with_score(self) -> None:
        result = compute_metrics(Y_TRUE, Y_PRED, Y_SCORE)
        assert result.pr_auc is not None
        assert 0.0 <= result.pr_auc <= 1.0

    def test_n_samples(self) -> None:
        result = compute_metrics(Y_TRUE, Y_PRED)
        assert result.n_samples == 8

    def test_n_positive_negative(self) -> None:
        result = compute_metrics(Y_TRUE, Y_PRED)
        assert result.n_positive == 4
        assert result.n_negative == 4

    def test_confusion_matrix_shape(self) -> None:
        result = compute_metrics(Y_TRUE, Y_PRED)
        assert result.confusion_matrix.shape == (2, 2)

    def test_perfect_classifier(self) -> None:
        y = np.array([0, 0, 1, 1])
        result = compute_metrics(y, y)
        assert result.sensitivity == 1.0
        assert result.specificity == 1.0
        assert result.accuracy == 1.0
        assert result.f1 == 1.0

    def test_all_wrong_classifier(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        result = compute_metrics(y_true, y_pred)
        assert result.sensitivity == 0.0
        assert result.specificity == 0.0

    def test_sensitivity_equals_recall(self) -> None:
        result = compute_metrics(Y_TRUE, Y_PRED)
        assert abs(result.sensitivity - result.recall) < 1e-10

    def test_f1_formula(self) -> None:
        result = compute_metrics(Y_TRUE, Y_PRED)
        if result.precision + result.recall > 0:
            expected_f1 = 2 * result.precision * result.recall / (result.precision + result.recall)
            assert abs(result.f1 - expected_f1) < 1e-10

    def test_repr(self) -> None:
        result = compute_metrics(Y_TRUE, Y_PRED)
        r = repr(result)
        assert "sens=" in r
        assert "spec=" in r

    def test_public_api_metrics(self) -> None:
        result = threshscore.compute_metrics(Y_TRUE, Y_PRED, Y_SCORE)
        assert isinstance(result, ClassificationMetrics)


class TestSensitivityScore:
    def test_perfect(self) -> None:
        y = np.array([0, 1, 1])
        assert sensitivity_score(y, y) == 1.0

    def test_zero(self) -> None:
        y_true = np.array([0, 1, 1])
        y_pred = np.array([1, 0, 0])
        assert sensitivity_score(y_true, y_pred) == 0.0

    def test_partial(self) -> None:
        y_true = np.array([0, 1, 1, 1])
        y_pred = np.array([0, 1, 0, 0])
        val = sensitivity_score(y_true, y_pred)
        assert abs(val - 1 / 3) < 1e-9


class TestSpecificityScore:
    def test_perfect(self) -> None:
        y = np.array([0, 0, 1])
        assert specificity_score(y, y) == 1.0

    def test_zero(self) -> None:
        y_true = np.array([0, 0, 1])
        y_pred = np.array([1, 1, 0])
        assert specificity_score(y_true, y_pred) == 0.0


# ---- Property-based tests ----

@given(
    n=st.integers(min_value=4, max_value=50),
    seed=st.integers(min_value=0, max_value=10000),
)
@settings(max_examples=100)
def test_metrics_ranges(n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    # Ensure at least one of each class
    y_true = np.concatenate([np.zeros(n // 2, dtype=np.int_), np.ones(n - n // 2, dtype=np.int_)])
    y_pred = rng.integers(0, 2, size=n)
    # Fix y_pred to have valid binary values
    y_pred = y_pred.astype(np.int_)

    result = compute_metrics(y_true, y_pred)
    assert 0.0 <= result.sensitivity <= 1.0
    assert 0.0 <= result.specificity <= 1.0
    assert 0.0 <= result.precision <= 1.0
    assert 0.0 <= result.f1 <= 1.0
    assert 0.0 <= result.accuracy <= 1.0
    assert result.n_samples == n
