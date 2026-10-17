"""Point-estimate calibration metrics for binary outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import check_binary_inputs

BinStrategy = Literal["uniform", "quantile"]


@dataclass(frozen=True)
class ReliabilityBins:
    """Per-bin summary behind a reliability diagram.

    Empty bins have ``count == 0`` and NaN ``mean_confidence`` / ``accuracy``.
    """

    edges: NDArray[np.float64]
    counts: NDArray[np.int64]
    mean_confidence: NDArray[np.float64]
    accuracy: NDArray[np.float64]


def brier_score(probs: ArrayLike, outcomes: ArrayLike) -> float:
    """Mean squared error between predicted probability and the 0/1 outcome.

    Lower is better; range [0, 1]. A strictly proper scoring rule, so it is
    minimised in expectation by the true conditional probability.
    """
    p, y = check_binary_inputs(probs, outcomes)
    return _brier(p, y)


def log_loss(probs: ArrayLike, outcomes: ArrayLike, eps: float = 1e-15) -> float:
    """Mean negative log-likelihood (natural log). Probabilities are clipped to ``[eps, 1-eps]``.

    Strictly proper like Brier but punishes confident errors much harder;
    unbounded above without clipping.
    """
    p, y = check_binary_inputs(probs, outcomes)
    return _log_loss(p, y, eps)


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


def maximum_calibration_error(
    probs: ArrayLike,
    outcomes: ArrayLike,
    n_bins: int = 15,
    strategy: BinStrategy = "uniform",
) -> float:
    """Largest ``|acc_b - conf_b|`` over non-empty bins (worst-case gap).

    Very noisy for sparsely populated bins; it is a maximum over noisy gaps, so
    it is biased upward more strongly than ECE.
    """
    p, y = check_binary_inputs(probs, outcomes)
    bins = _reliability_bins(p, y, n_bins, strategy)
    ok = bins.counts > 0
    return float(np.max(np.abs(bins.accuracy[ok] - bins.mean_confidence[ok])))


def adaptive_calibration_error(probs: ArrayLike, outcomes: ArrayLike, n_bins: int = 15) -> float:
    """Unweighted mean ``|acc_b - conf_b|`` over equal-mass (quantile) bins.

    Binary-outcome version of ACE (Nixon et al., 2019). Equal-mass bins avoid the
    empty/sparse-bin problem of equal-width binning when predictions cluster.
    """
    p, y = check_binary_inputs(probs, outcomes)
    bins = _reliability_bins(p, y, n_bins, "quantile")
    ok = bins.counts > 0
    return float(np.mean(np.abs(bins.accuracy[ok] - bins.mean_confidence[ok])))


def debiased_ece_l2(
    probs: ArrayLike,
    outcomes: ArrayLike,
    n_bins: int = 15,
    strategy: BinStrategy = "quantile",
) -> float:
    """Bias-corrected L2 calibration error (after Kumar et al., 2019).

    Within each bin, estimates the squared gap ``(E[y - p])**2`` without the
    noise-induced upward bias of the plug-in: ``mean(d)**2 - var(d) / n_b`` with
    ``d = y - p`` and the unbiased sample variance. Per-bin estimates are
    weighted by bin size, summed, floored at 0, and square-rooted.

    Conditions on the bin assignment (treats it as fixed) and skips bins with
    fewer than 2 points, renormalising weights over the remaining bins. The
    floor at 0 and the square root reintroduce a small bias, so this is
    *less* biased than plug-in L2, not unbiased.
    """
    p, y = check_binary_inputs(probs, outcomes)
    edges = _bin_edges(p, n_bins, strategy)
    idx = _assign(p, edges)
    total, weight = 0.0, 0
    for b in range(edges.size - 1):
        d = (y - p)[idx == b]
        if d.size < 2:
            continue
        total += d.size * (np.mean(d) ** 2 - np.var(d, ddof=1) / d.size)
        weight += d.size
    if weight == 0:
        raise ValueError("no bin has at least 2 observations; use fewer bins or more data")
    return float(np.sqrt(max(total / weight, 0.0)))


def reliability_bins(
    probs: ArrayLike,
    outcomes: ArrayLike,
    n_bins: int = 15,
    strategy: BinStrategy = "uniform",
) -> ReliabilityBins:
    """Per-bin counts, mean confidence and empirical accuracy (reliability-diagram data)."""
    p, y = check_binary_inputs(probs, outcomes)
    return _reliability_bins(p, y, n_bins, strategy)


def _brier(p: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    return float(np.mean((p - y) ** 2))


def _log_loss(p: NDArray[np.float64], y: NDArray[np.float64], eps: float = 1e-15) -> float:
    if not 0.0 < eps < 0.5:
        raise ValueError("eps must be in (0, 0.5)")
    q = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(q) + (1.0 - y) * np.log1p(-q)))


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


def _assign(p: NDArray[np.float64], edges: NDArray[np.float64]) -> NDArray[np.intp]:
    # Interior edges only: bins are [e_i, e_{i+1}) with the last bin closed on the right.
    return np.searchsorted(edges[1:-1], p, side="right")


def _reliability_bins(
    p: NDArray[np.float64], y: NDArray[np.float64], n_bins: int, strategy: BinStrategy
) -> ReliabilityBins:
    edges = _bin_edges(p, n_bins, strategy)
    k = edges.size - 1
    idx = _assign(p, edges)
    counts = np.bincount(idx, minlength=k)
    safe = np.maximum(counts, 1)
    empty = counts == 0
    conf = np.bincount(idx, weights=p, minlength=k) / safe
    acc = np.bincount(idx, weights=y, minlength=k) / safe
    conf[empty] = np.nan
    acc[empty] = np.nan
    return ReliabilityBins(edges, counts.astype(np.int64), conf, acc)


def _ece(
    p: NDArray[np.float64],
    y: NDArray[np.float64],
    n_bins: int,
    strategy: BinStrategy,
    norm: Literal["l1", "l2"],
) -> float:
    if norm not in ("l1", "l2"):
        raise ValueError(f"unknown norm {norm!r}")
    bins = _reliability_bins(p, y, n_bins, strategy)
    ok = bins.counts > 0
    w = bins.counts[ok] / p.size
    gap = bins.accuracy[ok] - bins.mean_confidence[ok]
    if norm == "l1":
        return float(np.sum(w * np.abs(gap)))
    return float(np.sqrt(np.sum(w * gap**2)))
