"""Tests for the unified centering-based scoring engine."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import threshscore
from threshscore.core.scoring import (
    ScoreResult,
    _centering_score,
    score_from_metrics,
    score_from_predictions,
    score_from_scores,
)
from threshscore.gates.arctan import ArctanGate
from threshscore.gates.registry import get as get_gate

Y_TRUE = np.array([0, 0, 1, 1, 1, 0, 1, 0])
Y_PRED = np.array([0, 1, 1, 1, 0, 0, 1, 0])
Y_SCORE = np.array([0.1, 0.6, 0.8, 0.9, 0.4, 0.2, 0.7, 0.3])


# ---------------------------------------------------------------------------
# Centering formula unit tests
# ---------------------------------------------------------------------------

class TestCenteringScore:
    """Test the core _centering_score engine directly."""

    def test_centering_at_operating_point_arctan(self) -> None:
        gate_fn = get_gate("arctan")
        s = _centering_score(gate_fn, 0.8, 0.6, 0.8, 0.6, {})
        assert abs(s - 0.5) < 1e-9

    def test_centering_at_operating_point_sigmoid(self) -> None:
        gate_fn = get_gate("sigmoid")
        s = _centering_score(gate_fn, 0.8, 0.6, 0.8, 0.6, {})
        assert abs(s - 0.5) < 1e-6  # sigmoid(0) = 0.5 exactly

    def test_centering_at_operating_point_relu_clip(self) -> None:
        gate_fn = get_gate("relu_clip")
        s = _centering_score(gate_fn, 0.8, 0.6, 0.8, 0.6, {})
        assert abs(s - 0.5) < 1e-9

    def test_centering_at_operating_point_linear_clip(self) -> None:
        gate_fn = get_gate("linear_clip")
        s = _centering_score(gate_fn, 0.8, 0.6, 0.8, 0.6, {})
        assert abs(s - 0.5) < 1e-9

    def test_above_operating_point(self) -> None:
        gate_fn = get_gate("arctan")
        s = _centering_score(gate_fn, 0.9, 0.75, 0.8, 0.6, {})
        assert s > 0.5

    def test_below_operating_point(self) -> None:
        gate_fn = get_gate("arctan")
        s = _centering_score(gate_fn, 0.5, 0.4, 0.8, 0.6, {})
        assert s < 0.5

    def test_score_in_range(self) -> None:
        gate_fn = get_gate("arctan")
        for sens in [0.0, 0.5, 0.8, 1.0]:
            for spec in [0.0, 0.5, 0.6, 1.0]:
                s = _centering_score(gate_fn, sens, spec, 0.8, 0.6, {})
                assert 0.0 <= s <= 1.0

    def test_gate_thr_uniform_all_gates(self) -> None:
        """All built-in gates return 0.5 at the threshold, so gate_thr = 0.25."""
        for name in ["arctan", "sigmoid", "relu_clip", "linear_clip"]:
            gate_fn = get_gate(name)
            gate_at_thr = gate_fn(0.8, 0.8)
            assert abs(gate_at_thr - 0.5) < 0.01, (
                f"{name}: expected gate(thr, thr) ≈ 0.5, got {gate_at_thr}"
            )

    def test_all_gates_approach_one_at_best_case(self) -> None:
        """All built-in gates should reach score > 0.9 at (1.0, 1.0)."""
        for name in ["arctan", "sigmoid", "relu_clip", "linear_clip"]:
            gate_fn = get_gate(name)
            s = _centering_score(gate_fn, 1.0, 1.0, 0.8, 0.6, {})
            assert s > 0.9, (
                f"{name}: expected score > 0.9 at best case, got {s:.4f}"
            )

    def test_all_gates_approach_zero_at_worst_case(self) -> None:
        """All built-in gates should reach score < 0.1 at (0.0, 0.0)."""
        for name in ["arctan", "sigmoid", "relu_clip", "linear_clip"]:
            gate_fn = get_gate(name)
            s = _centering_score(gate_fn, 0.0, 0.0, 0.8, 0.6, {})
            assert s < 0.1, (
                f"{name}: expected score < 0.1 at worst case, got {s:.4f}"
            )

    def test_custom_gate_params(self) -> None:
        gate_fn = get_gate("arctan")
        s1 = _centering_score(gate_fn, 0.9, 0.75, 0.8, 0.6, {"k": 1.0})
        s2 = _centering_score(gate_fn, 0.9, 0.75, 0.8, 0.6, {"k": 100.0})
        assert s1 != s2


# ---------------------------------------------------------------------------
# score_from_metrics
# ---------------------------------------------------------------------------

class TestScoreFromMetrics:
    def test_returns_float(self) -> None:
        s = score_from_metrics(0.85, 0.70)
        assert isinstance(s, float)

    def test_score_in_range(self) -> None:
        s = score_from_metrics(0.85, 0.70)
        assert 0.0 <= s <= 1.0

    def test_centering_property_at_operating_point(self) -> None:
        s = score_from_metrics(0.8, 0.6, s_thr=0.8, p_thr=0.6)
        assert abs(s - 0.5) < 1e-9

    def test_centering_custom_thresholds(self) -> None:
        s = score_from_metrics(0.7, 0.5, s_thr=0.7, p_thr=0.5)
        assert abs(s - 0.5) < 1e-9

    def test_above_operating_point(self) -> None:
        s = score_from_metrics(0.9, 0.8, s_thr=0.8, p_thr=0.6)
        assert s > 0.5

    def test_below_operating_point(self) -> None:
        s = score_from_metrics(0.5, 0.4, s_thr=0.8, p_thr=0.6)
        assert s < 0.5

    def test_gate_params_forwarded(self) -> None:
        s1 = score_from_metrics(0.9, 0.75, gate_params={"k": 1.0})
        s2 = score_from_metrics(0.9, 0.75, gate_params={"k": 50.0})
        assert s1 != s2

    def test_gate_instance(self) -> None:
        gate_fn = ArctanGate()
        s = score_from_metrics(0.8, 0.6, s_thr=0.8, p_thr=0.6, gate=gate_fn)
        assert abs(s - 0.5) < 1e-9

    def test_default_gate_params_is_none(self) -> None:
        import inspect
        sig = inspect.signature(score_from_metrics)
        assert sig.parameters["gate_params"].default is None


# ---------------------------------------------------------------------------
# score_from_predictions
# ---------------------------------------------------------------------------

class TestScoreFromPredictions:
    def test_returns_score_result(self) -> None:
        result = score_from_predictions(Y_TRUE, Y_PRED)
        assert isinstance(result, ScoreResult)

    def test_score_in_range(self) -> None:
        result = score_from_predictions(Y_TRUE, Y_PRED)
        assert 0.0 <= result.score <= 1.0

    def test_centering_property(self) -> None:
        # tp=4, fn=1 → sens=0.8; tn=3, fp=2 → spec=0.6
        y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 1])
        result = score_from_predictions(y_true, y_pred, s_thr=0.8, p_thr=0.6)
        assert abs(result.score - 0.5) < 1e-9

    def test_sensitivity_threshold_stored(self) -> None:
        result = score_from_predictions(Y_TRUE, Y_PRED, s_thr=0.9)
        assert result.sensitivity_threshold == 0.9

    def test_specificity_threshold_stored(self) -> None:
        result = score_from_predictions(Y_TRUE, Y_PRED, p_thr=0.7)
        assert result.specificity_threshold == 0.7

    def test_gate_name_stored(self) -> None:
        result = score_from_predictions(Y_TRUE, Y_PRED, gate="sigmoid")
        assert result.gate_name == "sigmoid"

    def test_gate_instance_name(self) -> None:
        gate_fn = ArctanGate()
        result = score_from_predictions(Y_TRUE, Y_PRED, gate=gate_fn)
        assert result.gate_name == "ArctanGate"

    def test_gate_values_in_range(self) -> None:
        result = score_from_predictions(Y_TRUE, Y_PRED)
        assert 0.0 <= result.sensitivity_gate <= 1.0
        assert 0.0 <= result.specificity_gate <= 1.0

    def test_metrics_populated(self) -> None:
        result = score_from_predictions(Y_TRUE, Y_PRED)
        assert result.metrics.sensitivity == pytest.approx(result.sensitivity)
        assert result.metrics.specificity == pytest.approx(result.specificity)

    def test_repr_contains_score(self) -> None:
        result = score_from_predictions(Y_TRUE, Y_PRED)
        r = repr(result)
        assert "score=" in r
        assert "sensitivity=" in r
        assert "specificity=" in r

    def test_perfect_classifier_high_score(self) -> None:
        y = np.array([0, 0, 0, 1, 1, 1])
        result = score_from_predictions(y, y, s_thr=0.5, p_thr=0.5)
        assert result.score > 0.9

    def test_inverted_classifier_low_score(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        result = score_from_predictions(y_true, y_pred)
        assert result.score < 0.1

    def test_default_gate_params_is_none(self) -> None:
        import inspect
        sig = inspect.signature(score_from_predictions)
        assert sig.parameters["gate_params"].default is None


# ---------------------------------------------------------------------------
# score_from_scores
# ---------------------------------------------------------------------------

class TestScoreFromScores:
    def test_returns_score_result(self) -> None:
        result = score_from_scores(Y_TRUE, Y_SCORE)
        assert isinstance(result, ScoreResult)

    def test_score_in_range(self) -> None:
        result = score_from_scores(Y_TRUE, Y_SCORE)
        assert 0.0 <= result.score <= 1.0

    def test_roc_auc_populated(self) -> None:
        result = score_from_scores(Y_TRUE, Y_SCORE)
        assert result.metrics.roc_auc is not None

    def test_pr_auc_populated(self) -> None:
        result = score_from_scores(Y_TRUE, Y_SCORE)
        assert result.metrics.pr_auc is not None

    def test_custom_threshold_binarisation(self) -> None:
        r05 = score_from_scores(Y_TRUE, Y_SCORE, threshold=0.5)
        r07 = score_from_scores(Y_TRUE, Y_SCORE, threshold=0.7)
        assert 0.0 <= r05.score <= 1.0
        assert 0.0 <= r07.score <= 1.0

    def test_gate_params_forwarded(self) -> None:
        r1 = score_from_scores(Y_TRUE, Y_SCORE, gate_params={"k": 1.0})
        r2 = score_from_scores(Y_TRUE, Y_SCORE, gate_params={"k": 100.0})
        assert r1.score != r2.score

    def test_default_gate_params_is_none(self) -> None:
        import inspect
        sig = inspect.signature(score_from_scores)
        assert sig.parameters["gate_params"].default is None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class TestPublicAPI:
    def test_score_from_metrics_accessible(self) -> None:
        s = threshscore.score_from_metrics(0.85, 0.70)
        assert 0.0 <= s <= 1.0

    def test_score_from_predictions_accessible(self) -> None:
        result = threshscore.score_from_predictions(Y_TRUE, Y_PRED)
        assert isinstance(result, ScoreResult)

    def test_score_from_scores_accessible(self) -> None:
        result = threshscore.score_from_scores(Y_TRUE, Y_SCORE)
        assert isinstance(result, ScoreResult)

    def test_compute_metrics_accessible(self) -> None:
        m = threshscore.compute_metrics(Y_TRUE, Y_PRED)
        assert m.sensitivity >= 0.0

    def test_centering_at_threshold(self) -> None:
        s = threshscore.score_from_metrics(0.8, 0.6, s_thr=0.8, p_thr=0.6)
        assert abs(s - 0.5) < 1e-9

    def test_centering_holds_for_all_built_in_gates(self) -> None:
        for name in ["arctan", "sigmoid", "relu_clip", "linear_clip"]:
            s = threshscore.score_from_metrics(
                0.8, 0.6, s_thr=0.8, p_thr=0.6, gate=name
            )
            assert abs(s - 0.5) < 1e-9, f"{name}: expected 0.5, got {s}"


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------

@given(
    n_pos=st.integers(min_value=2, max_value=20),
    n_neg=st.integers(min_value=2, max_value=20),
    seed=st.integers(min_value=0, max_value=9999),
    s_thr=st.floats(min_value=0.1, max_value=0.95, allow_nan=False),
    p_thr=st.floats(min_value=0.1, max_value=0.95, allow_nan=False),
)
@settings(max_examples=100)
def test_score_from_predictions_always_in_range(
    n_pos: int, n_neg: int, seed: int, s_thr: float, p_thr: float
) -> None:
    rng = np.random.default_rng(seed)
    y_true = np.concatenate([np.ones(n_pos, dtype=np.int_), np.zeros(n_neg, dtype=np.int_)])
    y_pred = rng.integers(0, 2, size=len(y_true)).astype(np.int_)
    result = score_from_predictions(y_true, y_pred, s_thr=s_thr, p_thr=p_thr)
    assert 0.0 <= result.score <= 1.0


@given(
    sens=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    spec=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    s_thr=st.floats(min_value=0.05, max_value=0.95, allow_nan=False),
    p_thr=st.floats(min_value=0.05, max_value=0.95, allow_nan=False),
)
@settings(max_examples=200)
def test_score_from_metrics_always_in_range(
    sens: float, spec: float, s_thr: float, p_thr: float
) -> None:
    s = score_from_metrics(sens, spec, s_thr=s_thr, p_thr=p_thr)
    assert 0.0 <= s <= 1.0


@given(
    s_thr=st.floats(min_value=0.05, max_value=0.95, allow_nan=False),
    p_thr=st.floats(min_value=0.05, max_value=0.95, allow_nan=False),
)
@settings(max_examples=200)
def test_centering_property_all_gates_all_thresholds(
    s_thr: float, p_thr: float,
) -> None:
    """score = 0.5 at the operating point for all gate functions and threshold pairs."""
    for gate_name in ["arctan", "sigmoid", "relu_clip", "linear_clip"]:
        s = score_from_metrics(s_thr, p_thr, s_thr=s_thr, p_thr=p_thr, gate=gate_name)
        assert abs(s - 0.5) < 1e-9, (
            f"{gate_name}: expected 0.5 at operating point "
            f"(s_thr={s_thr}, p_thr={p_thr}), got {s}"
        )
