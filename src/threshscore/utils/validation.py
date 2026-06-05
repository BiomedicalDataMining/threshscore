"""Input validation for threshscore."""

from __future__ import annotations

import numpy as np

from threshscore.utils.types import ArrayLike, FloatArray, IntArray


def validate_binary_inputs(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    y_score: ArrayLike | None = None,
) -> tuple[IntArray, IntArray, FloatArray | None]:
    """Validate and coerce inputs for binary classification.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels (0 or 1).
    y_pred:
        Predicted binary labels (0 or 1).
    y_score:
        Predicted probability scores in [0, 1]. Required for ROC/PR AUC.

    Returns
    -------
    Tuple of (y_true, y_pred, y_score) as numpy arrays.

    Raises
    ------
    ValueError
        If inputs are invalid (wrong shape, values out of range, etc.).
    """
    yt = np.asarray(y_true, dtype=np.int_)
    yp = np.asarray(y_pred, dtype=np.int_)

    if yt.ndim != 1:
        raise ValueError(f"y_true must be 1-D, got shape {yt.shape}")
    if yp.ndim != 1:
        raise ValueError(f"y_pred must be 1-D, got shape {yp.shape}")
    if len(yt) != len(yp):
        raise ValueError(
            f"y_true and y_pred must have the same length, got {len(yt)} vs {len(yp)}"
        )
    if len(yt) == 0:
        raise ValueError("Inputs must not be empty")
    if not np.all(np.isin(yt, [0, 1])):
        raise ValueError("y_true must contain only 0 and 1")
    if not np.all(np.isin(yp, [0, 1])):
        raise ValueError("y_pred must contain only 0 and 1")
    if len(np.unique(yt)) < 2:
        raise ValueError("y_true must contain both classes (0 and 1)")

    ys: FloatArray | None = None
    if y_score is not None:
        ys = np.asarray(y_score, dtype=np.float64)
        if ys.ndim != 1:
            raise ValueError(f"y_score must be 1-D, got shape {ys.shape}")
        if len(ys) != len(yt):
            raise ValueError(
                f"y_score must have the same length as y_true, got {len(ys)} vs {len(yt)}"
            )
        if np.any(ys < 0.0) or np.any(ys > 1.0):
            raise ValueError("y_score values must be in [0, 1]")

    return yt, yp, ys
