"""Tests for input validation."""

from __future__ import annotations

import numpy as np
import pytest

from threshscore.utils.validation import validate_binary_inputs


def test_valid_inputs_basic() -> None:
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 1, 0]
    yt, yp, ys = validate_binary_inputs(y_true, y_pred)
    assert len(yt) == 4
    assert ys is None


def test_valid_inputs_with_score() -> None:
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 1, 0]
    y_score = [0.1, 0.9, 0.6, 0.4]
    yt, yp, ys = validate_binary_inputs(y_true, y_pred, y_score)
    assert ys is not None
    assert len(ys) == 4


def test_wrong_shape_y_true() -> None:
    with pytest.raises(ValueError, match="1-D"):
        validate_binary_inputs([[0, 1], [1, 0]], [0, 1, 1, 0])


def test_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        validate_binary_inputs([0, 1], [0, 1, 0])


def test_empty_inputs() -> None:
    with pytest.raises(ValueError, match="empty"):
        validate_binary_inputs([], [])


def test_invalid_y_true_values() -> None:
    with pytest.raises(ValueError, match="0 and 1"):
        validate_binary_inputs([0, 2, 1], [0, 1, 1])


def test_invalid_y_pred_values() -> None:
    with pytest.raises(ValueError, match="0 and 1"):
        validate_binary_inputs([0, 1, 1], [0, 1, -1])


def test_single_class_y_true() -> None:
    with pytest.raises(ValueError, match="both classes"):
        validate_binary_inputs([1, 1, 1], [0, 1, 1])


def test_score_out_of_range() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate_binary_inputs([0, 1], [0, 1], [0.5, 1.5])


def test_score_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        validate_binary_inputs([0, 1], [0, 1], [0.5])


def test_numpy_array_inputs() -> None:
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0])
    yt, yp, _ = validate_binary_inputs(y_true, y_pred)
    assert yt.dtype == np.int_
