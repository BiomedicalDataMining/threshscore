"""Standard classification metrics for binary classification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)

from threshscore.utils.types import ArrayLike, FloatArray
from threshscore.utils.validation import validate_binary_inputs


@dataclass(frozen=True)
class ClassificationMetrics:
    """Container for binary classification metrics."""

    sensitivity: float
    specificity: float
    precision: float
    recall: float
    f1: float
    accuracy: float
    roc_auc: float | None
    pr_auc: float | None
    confusion_matrix: FloatArray
    n_samples: int
    n_positive: int
    n_negative: int

    def __repr__(self) -> str:
        roc = f"{self.roc_auc:.4f}" if self.roc_auc is not None else "N/A"
        pr = f"{self.pr_auc:.4f}" if self.pr_auc is not None else "N/A"
        return (
            f"ClassificationMetrics("
            f"sens={self.sensitivity:.4f}, "
            f"spec={self.specificity:.4f}, "
            f"prec={self.precision:.4f}, "
            f"recall={self.recall:.4f}, "
            f"f1={self.f1:.4f}, "
            f"acc={self.accuracy:.4f}, "
            f"roc_auc={roc}, "
            f"pr_auc={pr})"
        )


def compute_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    y_score: ArrayLike | None = None,
) -> ClassificationMetrics:
    """Compute standard binary classification metrics.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels (0 or 1).
    y_pred:
        Predicted binary labels (0 or 1).
    y_score:
        Predicted probability scores in [0, 1]. Required for ROC AUC and PR AUC.

    Returns
    -------
    ClassificationMetrics
    """
    yt, yp, ys = validate_binary_inputs(y_true, y_pred, y_score)

    cm: FloatArray = confusion_matrix(yt, yp).astype(np.float64)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = sensitivity
    f1 = (
        float(2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )
    accuracy = float((tp + tn) / len(yt))

    roc_auc: float | None = None
    pr_auc: float | None = None
    if ys is not None:
        roc_auc = float(roc_auc_score(yt, ys))
        pr_auc = float(average_precision_score(yt, ys))

    return ClassificationMetrics(
        sensitivity=sensitivity,
        specificity=specificity,
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        confusion_matrix=cm,
        n_samples=len(yt),
        n_positive=int(np.sum(yt)),
        n_negative=int(np.sum(1 - yt)),
    )


def sensitivity_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Compute sensitivity (true positive rate / recall)."""
    yt, yp, _ = validate_binary_inputs(y_true, y_pred)
    tp = float(np.sum((yt == 1) & (yp == 1)))
    fn = float(np.sum((yt == 1) & (yp == 0)))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def specificity_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Compute specificity (true negative rate)."""
    yt, yp, _ = validate_binary_inputs(y_true, y_pred)
    tn = float(np.sum((yt == 0) & (yp == 0)))
    fp = float(np.sum((yt == 0) & (yp == 1)))
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0
