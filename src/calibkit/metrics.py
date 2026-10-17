"""Point-estimate calibration metrics for binary outcomes."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import check_binary_inputs

BinStrategy = Literal["uniform", "quantile"]


def brier_score(probs: ArrayLike, outcomes: ArrayLike) -> float:
    """Mean squared error between predicted probability and the 0/1 outcome.

    Lower is better; range [0, 1]. A strictly proper scoring rule, so it is
    minimised in expectation by the true conditional probability.
    """
    p, y = check_binary_inputs(probs, outcomes)
    return _brier(p, y)


def expected_calibration_error(
    probs: ArrayLike,
    outcomes: ArrayLike,
    n_bins: int = 15,
    strategy: BinStrategy = "uniform",
    norm: Literal["l1", "l2"] = "l1",
) -> float:
    """Binned expected calibration error (plug-in estimator).

    ``l1``: ``sum_b (n_b / N) * |acc_b - conf_b|`` (the usual ECE).
    ``l2``: ``sqrt(sum_b (n_b / N) * (acc_b - conf_b)**2)``.

    ``strategy="uniform"`` uses equal-width bins on [0, 1]; ``"quantile"``
    uses equal-mass bins. Note this estimator is biased upward in finite
    samples and depends on the binning; see docs/assumptions.md.
    """
    p, y = check_binary_inputs(probs, outcomes)
    return _ece(p, y, n_bins, strategy, norm)


def _brier(p: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    return float(np.mean((p - y) ** 2))


def _bin_edges(p: NDArray[np.float64], n_bins: int, strategy: BinStrategy) -> NDArray[np.float64]:
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    if strategy == "uniform":
        return np.linspace(0.0, 1.0, n_bins + 1)
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1)))
        if edges.size < 2:  # all probabilities identical
            edges = np.array([edges[0], edges[0] + 1.0])
        return edges
    raise ValueError(f"unknown strategy {strategy!r}")


def _ece(
    p: NDArray[np.float64],
    y: NDArray[np.float64],
    n_bins: int,
    strategy: BinStrategy,
    norm: Literal["l1", "l2"],
) -> float:
    if norm not in ("l1", "l2"):
        raise ValueError(f"unknown norm {norm!r}")
    edges = _bin_edges(p, n_bins, strategy)
    k = edges.size - 1
    # Interior edges only: bins are [e_i, e_{i+1}) with the last bin closed on the right.
    idx = np.searchsorted(edges[1:-1], p, side="right")
    counts = np.bincount(idx, minlength=k).astype(np.float64)
    nonempty = counts > 0
    conf = np.bincount(idx, weights=p, minlength=k)[nonempty] / counts[nonempty]
    acc = np.bincount(idx, weights=y, minlength=k)[nonempty] / counts[nonempty]
    w = counts[nonempty] / p.size
    gap = acc - conf
    if norm == "l1":
        return float(np.sum(w * np.abs(gap)))
    return float(np.sqrt(np.sum(w * gap**2)))
