"""Tests for threshold sweep utilities."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import threshscore
from threshscore.core.thresholds import ThresholdPoint, ThresholdSweep, sweep_thresholds

Y_TRUE = np.array([0, 0, 0, 0, 1, 1, 1, 1])
Y_SCORE = np.array([0.1, 0.2, 0.4, 0.6, 0.5, 0.7, 0.8, 0.9])


class TestSweepThresholds:
    def test_returns_sweep(self) -> None:
        result = sweep_thresholds(Y_TRUE, Y_SCORE)
        assert isinstance(result, ThresholdSweep)

    def test_n_points(self) -> None:
        result = sweep_thresholds(Y_TRUE, Y_SCORE, n_thresholds=51)
        assert len(result.points) == 51

    def test_default_n_thresholds(self) -> None:
        result = sweep_thresholds(Y_TRUE, Y_SCORE)
        assert len(result.points) == 101

    def test_threshold_range(self) -> None:
        result = sweep_thresholds(Y_TRUE, Y_SCORE)
        assert result.thresholds[0] == 0.0
        assert result.thresholds[-1] == 1.0

    def test_sensitivities_shape(self) -> None:
        result = sweep_thresholds(Y_TRUE, Y_SCORE)
        assert result.sensitivities.shape == (101,)

    def test_specificities_shape(self) -> None:
        result = sweep_thresholds(Y_TRUE, Y_SCORE)
        assert result.specificities.shape == (101,)

    def test_sensitivity_in_range(self) -> None:
        result = sweep_thresholds(Y_TRUE, Y_SCORE)
        assert np.all(result.sensitivities >= 0.0)
        assert np.all(result.sensitivities <= 1.0)

    def test_specificity_in_range(self) -> None:
        result = sweep_thresholds(Y_TRUE, Y_SCORE)
        assert np.all(result.specificities >= 0.0)
        assert np.all(result.specificities <= 1.0)

    def test_at_zero_threshold_all_positive(self) -> None:
        result = sweep_thresholds(Y_TRUE, Y_SCORE)
        # All predicted positive at threshold=0 → sensitivity=1, specificity=0
        assert result.points[0].sensitivity == 1.0
        assert result.points[0].specificity == 0.0

    def test_at_one_threshold_all_negative(self) -> None:
        result = sweep_thresholds(Y_TRUE, Y_SCORE)
        # All predicted negative at threshold=1 → sensitivity=0, specificity=1
        assert result.points[-1].sensitivity == 0.0
        assert result.points[-1].specificity == 1.0

    def test_best_by_f1(self) -> None:
        result = sweep_thresholds(Y_TRUE, Y_SCORE)
        best = result.best_by_f1()
        assert isinstance(best, ThresholdPoint)
        assert best.f1 == max(p.f1 for p in result.points)

    def test_best_by_sensitivity_at_specificity(self) -> None:
        result = sweep_thresholds(Y_TRUE, Y_SCORE)
        best = result.best_by_sensitivity_at_specificity(min_specificity=0.5)
        if best is not None:
            assert best.specificity >= 0.5

    def test_best_by_sensitivity_at_specificity_none(self) -> None:
        result = sweep_thresholds(Y_TRUE, Y_SCORE)
        best = result.best_by_sensitivity_at_specificity(min_specificity=1.01)
        assert best is None

    def test_best_by_specificity_at_sensitivity(self) -> None:
        result = sweep_thresholds(Y_TRUE, Y_SCORE)
        best = result.best_by_specificity_at_sensitivity(min_sensitivity=0.5)
        if best is not None:
            assert best.sensitivity >= 0.5

    def test_best_by_specificity_at_sensitivity_none(self) -> None:
        result = sweep_thresholds(Y_TRUE, Y_SCORE)
        best = result.best_by_specificity_at_sensitivity(min_sensitivity=1.01)
        assert best is None

    def test_invalid_2d_y_true(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            sweep_thresholds([[0, 1], [1, 0]], Y_SCORE)

    def test_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            sweep_thresholds(Y_TRUE, [0.1, 0.9])

    def test_single_class_raises(self) -> None:
        with pytest.raises(ValueError, match="both classes"):
            sweep_thresholds([1, 1, 1, 1], [0.1, 0.5, 0.7, 0.9])

    def test_public_api_sweep(self) -> None:
        result = threshscore.sweep(Y_TRUE, Y_SCORE)
        assert isinstance(result, ThresholdSweep)
        assert len(result.points) == 101


# ---- Property-based tests ----

@given(
    n_pos=st.integers(min_value=2, max_value=15),
    n_neg=st.integers(min_value=2, max_value=15),
    seed=st.integers(min_value=0, max_value=9999),
)
@settings(max_examples=80)
def test_sweep_invariants(n_pos: int, n_neg: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    y_true = np.concatenate([np.ones(n_pos), np.zeros(n_neg)]).astype(np.int_)
    y_score = rng.uniform(0.0, 1.0, size=len(y_true))

    result = sweep_thresholds(y_true, y_score)

    # At threshold=0: predict all positive → sens=1, spec=0
    assert result.points[0].sensitivity == 1.0
    assert result.points[0].specificity == 0.0

    # At threshold=1: predict all negative → sens=0, spec=1
    assert result.points[-1].sensitivity == 0.0
    assert result.points[-1].specificity == 1.0

    # All values in range
    assert np.all(result.sensitivities >= 0.0)
    assert np.all(result.sensitivities <= 1.0)
    assert np.all(result.specificities >= 0.0)
    assert np.all(result.specificities <= 1.0)
