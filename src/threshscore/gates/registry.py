"""Gate function registry."""

from __future__ import annotations

from threshscore.gates.arctan import ArctanGate
from threshscore.gates.base import BaseGate
from threshscore.gates.linear_clip import LinearClipGate
from threshscore.gates.relu_clip import ReluClipGate
from threshscore.gates.sigmoid import SigmoidGate

_REGISTRY: dict[str, BaseGate] = {
    "arctan": ArctanGate(),
    "sigmoid": SigmoidGate(),
    "relu_clip": ReluClipGate(),
    "linear_clip": LinearClipGate(),
}


def get(name: str) -> BaseGate:
    """Retrieve a registered gate by name.

    Parameters
    ----------
    name:
        Registered gate name (e.g., ``"arctan"``, ``"sigmoid"``).

    Returns
    -------
    BaseGate

    Raises
    ------
    KeyError
        If no gate is registered under ``name``.
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"Unknown gate {name!r}. Available: {available}")
    return _REGISTRY[name]


def register(name: str, gate: BaseGate) -> None:
    """Register a custom gate function.

    Parameters
    ----------
    name:
        Unique name for the gate.
    gate:
        An instance of a :class:`BaseGate` subclass.

    Raises
    ------
    TypeError
        If ``gate`` is not a :class:`BaseGate` instance.
    """
    if not isinstance(gate, BaseGate):
        raise TypeError(
            f"gate must be a BaseGate instance, got {type(gate)}"
        )
    _REGISTRY[name] = gate


def list_gates() -> list[str]:
    """Return sorted list of all registered gate names."""
    return sorted(_REGISTRY)


def register_class(name: str, cls: type[BaseGate]) -> None:
    """Convenience: instantiate ``cls`` and register it under ``name``."""
    register(name, cls())
