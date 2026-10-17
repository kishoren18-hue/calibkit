"""Bootstrap confidence intervals for calibration metrics."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import check_binary_inputs
from .metrics import BinStrategy, _brier, _ece

MetricFn = Callable[[NDArray[np.float64], NDArray[np.float64]], float]
CIMethod = Literal["percentile", "basic"]


@dataclass(frozen=True)
class BootstrapResult:
    """Point estimate with a bootstrap confidence interval."""

    estimate: float
    low: float
    high: float
    confidence_level: float
    n_resamples: int
    method: str

    def __iter__(self) -> Iterator[float]:  # ``est, lo, hi = result``
        yield from (self.estimate, self.low, self.high)


def bootstrap_ci(
    metric: MetricFn,
    probs: ArrayLike,
    outcomes: ArrayLike,
    *,
    n_resamples: int = 2000,
    confidence_level: float = 0.95,
    method: CIMethod = "percentile",
    seed: int | np.random.Generator | None = None,
) -> BootstrapResult:
    """Nonparametric (i.i.d. case-resampling) bootstrap CI for any metric.

    Pairs ``(prob, outcome)`` are resampled with replacement, so this assumes
    exchangeable observations. It is *not* valid for clustered or
    time-dependent data (e.g. many questions per document) without resampling
    at the cluster level. ``method="percentile"`` uses the bootstrap
    quantiles directly; ``"basic"`` reflects them around the estimate.
    """
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    if n_resamples < 2:
        raise ValueError("n_resamples must be >= 2")
    if method not in ("percentile", "basic"):
        raise ValueError(f"unknown method {method!r}")
    p, y = check_binary_inputs(probs, outcomes)
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)

    n = p.size
    estimate = float(metric(p, y))
    stats = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        stats[i] = metric(p[idx], y[idx])

    alpha = 1.0 - confidence_level
    lo_q, hi_q = np.quantile(stats, [alpha / 2.0, 1.0 - alpha / 2.0])
    if method == "percentile":
        low, high = float(lo_q), float(hi_q)
    else:
        low, high = float(2.0 * estimate - hi_q), float(2.0 * estimate - lo_q)
    return BootstrapResult(estimate, low, high, confidence_level, n_resamples, method)


def brier_score_ci(probs: ArrayLike, outcomes: ArrayLike, **kwargs: object) -> BootstrapResult:
    """Brier score with a bootstrap CI. Keyword args go to :func:`bootstrap_ci`."""
    return bootstrap_ci(_brier, probs, outcomes, **kwargs)  # type: ignore[arg-type]


def ece_ci(
    probs: ArrayLike,
    outcomes: ArrayLike,
    *,
    n_bins: int = 15,
    strategy: BinStrategy = "uniform",
    norm: Literal["l1", "l2"] = "l1",
    **kwargs: object,
) -> BootstrapResult:
    """ECE with a bootstrap CI. Remaining keyword args go to :func:`bootstrap_ci`.

    The bins are fixed by ``n_bins``/``strategy`` and recomputed per resample
    for ``quantile``. Because plug-in ECE is biased upward, the point estimate
    can sit near or outside the lower end of a percentile interval.
    """

    def metric(p: NDArray[np.float64], y: NDArray[np.float64]) -> float:
        return _ece(p, y, n_bins, strategy, norm)

    return bootstrap_ci(metric, probs, outcomes, **kwargs)  # type: ignore[arg-type]
