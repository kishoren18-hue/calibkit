"""Reliability diagram (requires the optional ``matplotlib`` dependency)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

from .metrics import BinStrategy, reliability_bins

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def plot_reliability_diagram(
    probs: ArrayLike,
    outcomes: ArrayLike,
    n_bins: int = 10,
    strategy: BinStrategy = "uniform",
    ax: Axes | None = None,
) -> Any:
    """Plot per-bin accuracy against mean confidence with the diagonal for reference.

    Marker area is proportional to bin count. Returns the matplotlib ``Axes``.
    Install with ``pip install calibkit[plot]``.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "plot_reliability_diagram needs matplotlib: pip install calibkit[plot]"
        ) from exc

    bins = reliability_bins(probs, outcomes, n_bins=n_bins, strategy=strategy)
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4.5))
    ok = bins.counts > 0
    sizes = 20 + 380 * bins.counts[ok] / bins.counts.sum()
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    ax.plot(bins.mean_confidence[ok], bins.accuracy[ok], color="C0", alpha=0.5)
    ax.scatter(bins.mean_confidence[ok], bins.accuracy[ok], s=np.asarray(sizes), color="C0")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("mean predicted confidence")
    ax.set_ylabel("empirical accuracy")
    ax.legend(loc="upper left")
    return ax
