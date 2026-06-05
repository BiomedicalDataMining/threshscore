"""Threshold-aware scoring engine: unified centering-based formula."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from threshscore.core.metrics import ClassificationMetrics, compute_metrics
from threshscore.gates.base import BaseGate
from threshscore.gates.registry import get as get_gate
from threshscore.utils.types import ArrayLike


@dataclass(frozen=True)
class ScoreResult:
    """Result of threshold-aware centering-based scoring."""

    score: float
    sensitivity: float
    specificity: float
    sensitivity_gate: float
    specificity_gate: float
    sensitivity_threshold: float
    specificity_threshold: float
    gate_name: str
    metrics: ClassificationMetrics

    def __repr__(self) -> str:
        return (
            f"ScoreResult("
            f"score={self.score:.4f}, "
            f"sensitivity={self.sensitivity:.4f} "
            f"(threshold={self.sensitivity_threshold}, gate={self.sensitivity_gate:.4f}), "
            f"specificity={self.specificity:.4f} "
            f"(threshold={self.specificity_threshold}, gate={self.specificity_gate:.4f}), "
            f"gate={self.gate_name})"
        )


def _resolve_gate(
    gate: str | BaseGate,
) -> tuple[BaseGate, str]:
    if isinstance(gate, str):
        return get_gate(gate), gate
    return gate, type(gate).__name__


def _centering_score(
    gate_fn: BaseGate,
    sens: float,
    spec: float,
    s_thr: float,
    p_thr: float,
    params: dict[str, Any],
) -> float:
    """Compute centering-based score using ``gate_fn`` as the gate function.

    Formula
    -------
        q        = 0.5 * (sens + spec)
        g_s      = gate_fn(sens, s_thr, **params)
        g_p      = gate_fn(spec, p_thr, **params)
        gate     = g_s * g_p
        gate_thr = gate_fn(s_thr, s_thr, **params) * gate_fn(p_thr, p_thr, **params)
        q_thr    = 0.5 * (s_thr + p_thr)
        score    = clip(0.5 + 0.5*(gate - gate_thr) + 0.5*(q - q_thr), 0, 1)

    ``gate_thr`` is computed dynamically so the formula centres correctly for
    any gate function.  All built-in gate functions return 0.5 at the threshold,
    so gate_thr = 0.25 uniformly across arctan, sigmoid, relu_clip, and
    linear_clip.  Custom gate functions may return different values at the
    threshold; the formula handles them correctly via the dynamic gate_thr.

    Properties
    ----------
    - score = 0.5 exactly when sens = s_thr AND spec = p_thr
    - score > 0.5 when both metrics exceed their thresholds
    - score < 0.5 when either metric falls short
    - score in [0, 1] for all inputs
    - score approaches 1.0 at the best-case point (sens=1, spec=1)
    - score approaches 0.0 at the worst-case point
    """
    sens = max(0.0, min(1.0, sens))
    spec = max(0.0, min(1.0, spec))
    q = 0.5 * (sens + spec)
    g_s = gate_fn(sens, s_thr, **params)
    g_p = gate_fn(spec, p_thr, **params)
    gate = g_s * g_p
    # gate_thr: gate output at the operating point, determined by the gate shape
    gate_thr = gate_fn(s_thr, s_thr, **params) * gate_fn(p_thr, p_thr, **params)
    q_thr = 0.5 * (s_thr + p_thr)
    raw = 0.5 + 0.5 * (gate - gate_thr) + 0.5 * (q - q_thr)
    return float(max(0.0, min(1.0, raw)))


def score_from_metrics(
    sens: float,
    spec: float,
    *,
    s_thr: float = 0.8,
    p_thr: float = 0.6,
    gate: str | BaseGate = "arctan",
    gate_params: dict[str, Any] | None = None,
) -> float:
    """Compute centering-based score directly from sensitivity and specificity.

    Returns 0.5 when (sens, spec) equals the operating point (s_thr, p_thr),
    > 0.5 when both metrics exceed their thresholds, < 0.5 when either falls short.

    Parameters
    ----------
    sens:
        Achieved sensitivity in [0, 1].
    spec:
        Achieved specificity in [0, 1].
    s_thr:
        Minimum required sensitivity (default: 0.8).
    p_thr:
        Minimum required specificity (default: 0.6).
    gate:
        Gate function name or instance (default: ``"arctan"``).
    gate_params:
        Parameters to override the gate's defaults.

    Returns
    -------
    float
        Score in [0, 1].
    """
    gate_fn, _ = _resolve_gate(gate)
    params = gate_params or {}
    return _centering_score(gate_fn, sens, spec, s_thr, p_thr, params)


def score_from_predictions(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    s_thr: float = 0.8,
    p_thr: float = 0.6,
    gate: str | BaseGate = "arctan",
    gate_params: dict[str, Any] | None = None,
) -> ScoreResult:
    """Compute centering-based score from binary predictions.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels (0 or 1).
    y_pred:
        Predicted binary labels (0 or 1).
    s_thr:
        Minimum required sensitivity (default: 0.8).
    p_thr:
        Minimum required specificity (default: 0.6).
    gate:
        Gate function name or instance (default: ``"arctan"``).
    gate_params:
        Parameters to override the gate's defaults.

    Returns
    -------
    ScoreResult
    """
    m = compute_metrics(y_true, y_pred)
    return _build_score_result(
        m, s_thr=s_thr, p_thr=p_thr,
        gate=gate, gate_params=gate_params,
    )


def score_from_scores(
    y_true: ArrayLike,
    y_score: ArrayLike,
    threshold: float = 0.5,
    *,
    s_thr: float = 0.8,
    p_thr: float = 0.6,
    gate: str | BaseGate = "arctan",
    gate_params: dict[str, Any] | None = None,
) -> ScoreResult:
    """Compute centering-based score from probability scores.

    Binarises ``y_score`` at ``threshold`` to obtain binary predictions, then
    computes the centering-based score.  The full metrics object (including ROC
    AUC and PR AUC) is available on the result.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels (0 or 1).
    y_score:
        Predicted probability scores in [0, 1].
    threshold:
        Decision threshold for binarising ``y_score`` (default: 0.5).
    s_thr:
        Minimum required sensitivity (default: 0.8).
    p_thr:
        Minimum required specificity (default: 0.6).
    gate:
        Gate function name or instance (default: ``"arctan"``).
    gate_params:
        Parameters to override the gate's defaults.

    Returns
    -------
    ScoreResult
    """
    ys = np.asarray(y_score, dtype=np.float64)
    y_pred = (ys >= threshold).astype(np.int_)
    m = compute_metrics(y_true, y_pred, y_score)
    return _build_score_result(
        m, s_thr=s_thr, p_thr=p_thr,
        gate=gate, gate_params=gate_params,
    )


def _build_score_result(
    m: ClassificationMetrics,
    *,
    s_thr: float,
    p_thr: float,
    gate: str | BaseGate,
    gate_params: dict[str, Any] | None,
) -> ScoreResult:
    gate_fn, gate_fn_name = _resolve_gate(gate)
    params = gate_params or {}
    g_s = gate_fn(m.sensitivity, s_thr, **params)
    g_p = gate_fn(m.specificity, p_thr, **params)
    s = _centering_score(gate_fn, m.sensitivity, m.specificity, s_thr, p_thr, params)
    return ScoreResult(
        score=s,
        sensitivity=m.sensitivity,
        specificity=m.specificity,
        sensitivity_gate=g_s,
        specificity_gate=g_p,
        sensitivity_threshold=s_thr,
        specificity_threshold=p_thr,
        gate_name=gate_fn_name,
        metrics=m,
    )
