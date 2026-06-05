"""Type aliases used throughout threshscore."""

from __future__ import annotations

from typing import Union

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]
ArrayLike = Union[FloatArray, IntArray, list[float], list[int]]
