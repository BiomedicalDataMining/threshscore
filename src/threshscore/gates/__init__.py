"""Gate functions for threshold-aware scoring."""

from threshscore.gates.arctan import ArctanGate
from threshscore.gates.base import BaseGate
from threshscore.gates.linear_clip import LinearClipGate
from threshscore.gates.registry import get, list_gates, register, register_class
from threshscore.gates.relu_clip import ReluClipGate
from threshscore.gates.sigmoid import SigmoidGate

__all__ = [
    "BaseGate",
    "ArctanGate",
    "SigmoidGate",
    "ReluClipGate",
    "LinearClipGate",
    "get",
    "register",
    "register_class",
    "list_gates",
]
