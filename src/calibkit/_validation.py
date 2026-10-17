"""Input validation shared by all metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def check_binary_inputs(
    probs: ArrayLike, outcomes: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate and coerce predicted probabilities and binary outcomes.

    ``probs`` are the predicted probability of the positive event (for a
    confidence-vs-correctness setup, the stated confidence). ``outcomes`` are
    0/1 indicators that the event occurred (or the answer was correct).
    """
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(outcomes, dtype=np.float64)
    if p.ndim != 1 or y.ndim != 1:
        raise ValueError("probs and outcomes must be 1-dimensional")
    if p.shape != y.shape:
        raise ValueError(f"length mismatch: probs has {p.size}, outcomes has {y.size}")
    if p.size == 0:
        raise ValueError("inputs must be non-empty")
    if not (np.all(np.isfinite(p)) and np.all(np.isfinite(y))):
        raise ValueError("inputs must not contain NaN or infinity")
    if p.min() < 0.0 or p.max() > 1.0:
        raise ValueError("probs must lie in [0, 1]")
    if not np.all((y == 0.0) | (y == 1.0)):
        raise ValueError("outcomes must be binary (0 or 1)")
    return p, y
