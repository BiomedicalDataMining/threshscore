"""Tests for gate functions."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import threshscore.gates as gates
from threshscore.gates.arctan import ArctanGate
from threshscore.gates.base import BaseGate
from threshscore.gates.linear_clip import LinearClipGate
from threshscore.gates.relu_clip import ReluClipGate
from threshscore.gates.sigmoid import SigmoidGate

# ---------------------------------------------------------------------------
# Unit tests: gate (__call__) behaviour
# ---------------------------------------------------------------------------

class TestArctanGate:
    def setup_method(self) -> None:
        self.gate = ArctanGate()

    def test_default_params(self) -> None:
        assert self.gate.default_params == {"k": 25.0}

    def test_at_threshold(self) -> None:
        val = self.gate(0.8, 0.8)
        assert 0.49 < val < 0.51  # arctan(0) / pi + 0.5 = 0.5

    def test_above_threshold(self) -> None:
        assert self.gate(0.9, 0.8) > 0.5

    def test_below_threshold(self) -> None:
        assert self.gate(0.5, 0.8) < 0.5

    def test_well_above_threshold(self) -> None:
        assert self.gate(1.0, 0.5) > 0.9

    def test_well_below_threshold(self) -> None:
        assert self.gate(0.0, 0.8) < 0.1

    def test_output_range(self) -> None:
        for v in [0.0, 0.25, 0.5, 0.75, 1.0]:
            for t in [0.3, 0.5, 0.7, 0.9]:
                out = self.gate(v, t)
                assert 0.0 <= out <= 1.0

    def test_custom_k(self) -> None:
        steep = self.gate(0.81, 0.80, k=100.0)
        shallow = self.gate(0.81, 0.80, k=1.0)
        assert steep > shallow


class TestSigmoidGate:
    def setup_method(self) -> None:
        self.gate = SigmoidGate()

    def test_default_params(self) -> None:
        assert self.gate.default_params == {"k": 15.0}

    def test_at_threshold(self) -> None:
        assert abs(self.gate(0.8, 0.8) - 0.5) < 0.001

    def test_above_threshold(self) -> None:
        assert self.gate(0.9, 0.8) > 0.5

    def test_below_threshold(self) -> None:
        assert self.gate(0.7, 0.8) < 0.5

    def test_output_range(self) -> None:
        for v in [0.0, 0.5, 1.0]:
            for t in [0.3, 0.7]:
                out = self.gate(v, t)
                assert 0.0 <= out <= 1.0

    def test_overflow_guards(self) -> None:
        assert self.gate(-100.0, 0.0) == 0.0
        assert self.gate(100.0, 0.0) == 1.0


class TestReluClipGate:
    def setup_method(self) -> None:
        self.gate = ReluClipGate()

    def test_default_params(self) -> None:
        assert self.gate.default_params == {}

    def test_below_threshold(self) -> None:
        assert self.gate(0.5, 0.8) == 0.0

    def test_at_threshold(self) -> None:
        assert abs(self.gate(0.8, 0.8) - 0.5) < 1e-9

    def test_above_threshold(self) -> None:
        # 0.5 + 0.5*(0.1/0.2) = 0.75
        assert abs(self.gate(0.9, 0.8) - 0.75) < 1e-9

    def test_at_value_one(self) -> None:
        assert abs(self.gate(1.0, 0.8) - 1.0) < 1e-9

    def test_zero_threshold(self) -> None:
        assert self.gate(0.5, 0.0) == 1.0

    def test_output_range(self) -> None:
        for v in [0.0, 0.4, 0.8, 1.0]:
            for t in [0.5, 0.8]:
                out = self.gate(v, t)
                assert 0.0 <= out <= 1.0


class TestLinearClipGate:
    def setup_method(self) -> None:
        self.gate = LinearClipGate()

    def test_default_params(self) -> None:
        assert self.gate.default_params == {}

    def test_at_threshold(self) -> None:
        assert abs(self.gate(0.8, 0.8) - 0.5) < 1e-9

    def test_above_threshold(self) -> None:
        # 0.5 + 0.5*(0.1/0.2) = 0.75
        assert abs(self.gate(0.9, 0.8) - 0.75) < 1e-9

    def test_at_value_one(self) -> None:
        assert abs(self.gate(1.0, 0.8) - 1.0) < 1e-9

    def test_below_threshold_partial_credit(self) -> None:
        # Proportional partial credit in [0, 0.5): 0.5*(0.4/0.8) = 0.25
        assert abs(self.gate(0.4, 0.8) - 0.25) < 1e-9

    def test_zero_value(self) -> None:
        assert self.gate(0.0, 0.8) == 0.0

    def test_zero_threshold(self) -> None:
        assert self.gate(0.5, 0.0) == 1.0

    def test_output_range(self) -> None:
        for v in [0.0, 0.3, 0.6, 1.0]:
            for t in [0.5, 0.9]:
                out = self.gate(v, t)
                assert 0.0 <= out <= 1.0


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_get_arctan(self) -> None:
        gate = gates.get("arctan")
        assert isinstance(gate, ArctanGate)

    def test_get_sigmoid(self) -> None:
        gate = gates.get("sigmoid")
        assert isinstance(gate, SigmoidGate)

    def test_get_relu_clip(self) -> None:
        gate = gates.get("relu_clip")
        assert isinstance(gate, ReluClipGate)

    def test_get_linear_clip(self) -> None:
        gate = gates.get("linear_clip")
        assert isinstance(gate, LinearClipGate)

    def test_get_unknown(self) -> None:
        with pytest.raises(KeyError, match="unknown_gate"):
            gates.get("unknown_gate")

    def test_list_gates(self) -> None:
        names = gates.list_gates()
        assert "arctan" in names
        assert "sigmoid" in names
        assert "relu_clip" in names
        assert "linear_clip" in names

    def test_register_custom(self) -> None:
        class MyGate(BaseGate):
            @property
            def default_params(self) -> dict:
                return {}

            def __call__(self, value: float, threshold: float, **params: object) -> float:
                return 1.0 if value >= threshold else 0.0

        gates.register("my_custom_v3", MyGate())
        retrieved = gates.get("my_custom_v3")
        assert retrieved(0.9, 0.8) == 1.0
        assert retrieved(0.5, 0.8) == 0.0

    def test_register_non_gate_raises(self) -> None:
        with pytest.raises(TypeError):
            gates.register("bad", "not_a_gate")  # type: ignore[arg-type]

    def test_register_class(self) -> None:
        gates.register_class("arctan_via_class", ArctanGate)
        gate = gates.get("arctan_via_class")
        assert isinstance(gate, ArctanGate)


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------

@given(
    value=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    threshold=st.floats(min_value=0.01, max_value=0.99, allow_nan=False),
)
@settings(max_examples=200)
def test_arctan_gate_output_in_range(value: float, threshold: float) -> None:
    gate = ArctanGate()
    out = gate(value, threshold)
    assert 0.0 <= out <= 1.0


@given(
    value=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    threshold=st.floats(min_value=0.01, max_value=0.99, allow_nan=False),
)
@settings(max_examples=200)
def test_sigmoid_gate_output_in_range(value: float, threshold: float) -> None:
    gate = SigmoidGate()
    out = gate(value, threshold)
    assert 0.0 <= out <= 1.0


@given(
    value=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    threshold=st.floats(min_value=0.01, max_value=0.99, allow_nan=False),
)
@settings(max_examples=200)
def test_relu_clip_gate_output_in_range(value: float, threshold: float) -> None:
    gate = ReluClipGate()
    out = gate(value, threshold)
    assert 0.0 <= out <= 1.0


@given(
    value=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    threshold=st.floats(min_value=0.01, max_value=0.99, allow_nan=False),
)
@settings(max_examples=200)
def test_linear_clip_gate_output_in_range(value: float, threshold: float) -> None:
    gate = LinearClipGate()
    out = gate(value, threshold)
    assert 0.0 <= out <= 1.0


@given(
    threshold=st.floats(min_value=0.01, max_value=0.99, allow_nan=False),
)
@settings(max_examples=100)
def test_all_gates_monotone(threshold: float) -> None:
    """All gate functions should be monotone: higher value → higher or equal gate output."""
    for name in ["arctan", "sigmoid", "relu_clip", "linear_clip"]:
        gate = gates.get(name)
        scores = [gate(v / 100.0, threshold) for v in range(0, 101)]
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1] + 1e-10, (
                f"{name} not monotone at threshold={threshold}: "
                f"scores[{i}]={scores[i]}, scores[{i+1}]={scores[i+1]}"
            )
