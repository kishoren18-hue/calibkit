"""Post-hoc recalibration maps for binary predictions (NumPy only).

Each recalibrator is fit on held-out ``(prob, outcome)`` pairs and maps raw
probabilities to recalibrated ones. Fit on data *disjoint* from the data you
evaluate on; otherwise reported calibration is optimistic.
"""

from __future__ import annotations

from typing import TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import check_binary_inputs

_EPS = 1e-12
R = TypeVar("R", bound="Recalibrator")


def _logit(p: NDArray[np.float64]) -> NDArray[np.float64]:
    q = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(q) - np.log1p(-q)


def _sigmoid(z: NDArray[np.float64]) -> NDArray[np.float64]:
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _nll(z: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    # log(1 + e^z) - y z, computed stably
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def _check_probs(probs: ArrayLike) -> NDArray[np.float64]:
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError("probs must be 1-dimensional")
    if not np.all(np.isfinite(p)) or p.min(initial=0.0) < 0.0 or p.max(initial=1.0) > 1.0:
        raise ValueError("probs must be finite and lie in [0, 1]")
    return p


class Recalibrator:
    """Base class: ``fit(probs, outcomes)`` returns ``self``; ``predict(probs)`` maps probs."""

    _fitted = False

    def fit(self: R, probs: ArrayLike, outcomes: ArrayLike) -> R:
        p, y = check_binary_inputs(probs, outcomes)
        self._fit(p, y)
        self._fitted = True
        return self

    def predict(self, probs: ArrayLike) -> NDArray[np.float64]:
        if not self._fitted:
            raise RuntimeError("call fit() before predict()")
        return self._predict(_check_probs(probs))

    def _fit(self, p: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        raise NotImplementedError

    def _predict(self, p: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError


class TemperatureScaling(Recalibrator):
    """``p' = sigmoid(logit(p) / T)``, one parameter ``T > 0`` fit by minimising log-loss.

    ``T > 1`` softens overconfident predictions, ``T < 1`` sharpens underconfident
    ones. Preserves ranking (and so AUC/accuracy at any fixed threshold of the
    ordering). The NLL is convex in ``1/T``; it is solved by golden-section search.
    If the fit data are perfectly separated, the MLE is ``T -> 0`` and the search
    stops at ``T_min``.
    """

    def __init__(self, t_min: float = 1e-2, t_max: float = 1e2) -> None:
        if not 0.0 < t_min < t_max:
            raise ValueError("require 0 < t_min < t_max")
        self.t_min, self.t_max = t_min, t_max
        self.temperature_: float = 1.0

    def _fit(self, p: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        z = _logit(p)
        lo, hi = np.log(1.0 / self.t_max), np.log(1.0 / self.t_min)  # search log(beta), beta=1/T

        def f(log_beta: float) -> float:
            return _nll(np.exp(log_beta) * z, y)

        phi = (np.sqrt(5.0) - 1.0) / 2.0
        a, b = lo, hi
        c, d = b - phi * (b - a), a + phi * (b - a)
        fc, fd = f(c), f(d)
        for _ in range(100):
            if b - a < 1e-10:
                break
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - phi * (b - a)
                fc = f(c)
            else:
                a, c, fc = c, d, fd
                d = a + phi * (b - a)
                fd = f(d)
        self.temperature_ = float(np.exp(-(a + b) / 2.0))

    def _predict(self, p: NDArray[np.float64]) -> NDArray[np.float64]:
        return _sigmoid(_logit(p) / self.temperature_)


class PlattScaling(Recalibrator):
    """``p' = sigmoid(a * logit(p) + b)``, fit by Newton's method on log-loss.

    Two parameters (slope and intercept in logit space), so it can fix a shifted
    base rate as well as over/under-confidence. A small L2 penalty on ``a - 1``
    and ``b`` (``l2``) keeps the fit finite when the data are separable.
    """

    def __init__(self, l2: float = 1e-6, max_iter: int = 100, tol: float = 1e-10) -> None:
        self.l2, self.max_iter, self.tol = l2, max_iter, tol
        self.a_: float = 1.0
        self.b_: float = 0.0

    def _fit(self, p: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        z = _logit(p)
        n = z.size
        theta = np.array([1.0, 0.0])
        prior = np.array([1.0, 0.0])
        X = np.column_stack([z, np.ones(n)])

        def objective(t: NDArray[np.float64]) -> float:
            return _nll(X @ t, y) + 0.5 * self.l2 * float(np.sum((t - prior) ** 2))

        cur = objective(theta)
        for _ in range(self.max_iter):
            q = _sigmoid(X @ theta)
            grad = X.T @ (q - y) / n + self.l2 * (theta - prior)
            w = q * (1.0 - q)
            hess = (X * w[:, None]).T @ X / n + self.l2 * np.eye(2)
            step = np.linalg.solve(hess, grad)
            t = 1.0
            while t > 1e-8:  # backtracking line search
                new = objective(theta - t * step)
                if new <= cur:
                    break
                t *= 0.5
            theta = theta - t * step
            if abs(cur - new) < self.tol and np.max(np.abs(t * step)) < 1e-8:
                cur = new
                break
            cur = new
        self.a_, self.b_ = float(theta[0]), float(theta[1])

    def _predict(self, p: NDArray[np.float64]) -> NDArray[np.float64]:
        return _sigmoid(self.a_ * _logit(p) + self.b_)


class IsotonicRegression(Recalibrator):
    """Monotone non-decreasing map fit by pool-adjacent-violators, linearly interpolated.

    Nonparametric: flexible, but needs a few hundred or more points and can
    overfit small samples. Many raw scores collapse to the same output (ties),
    so ranking information is partially lost. Outside the fit range, predictions
    are clamped to the end values.
    """

    def __init__(self) -> None:
        self.x_: NDArray[np.float64] = np.empty(0)
        self.y_: NDArray[np.float64] = np.empty(0)

    def _fit(self, p: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        # Collapse tied scores first so the fitted knots have strictly increasing x.
        xs, inv = np.unique(p, return_inverse=True)
        counts = np.bincount(inv).astype(np.float64)
        ysum = np.bincount(inv, weights=y)
        # Each block: [sum_y, sum_x, weight]; merge while the monotone constraint is violated.
        sy: list[float] = []
        sx: list[float] = []
        wt: list[float] = []
        for xi, yi, ci in zip(xs, ysum, counts):
            sy.append(float(yi))
            sx.append(float(xi) * float(ci))
            wt.append(float(ci))
            while len(sy) > 1 and sy[-2] / wt[-2] >= sy[-1] / wt[-1]:
                y_last, x_last, w_last = sy.pop(), sx.pop(), wt.pop()
                sy[-1] += y_last
                sx[-1] += x_last
                wt[-1] += w_last
        w = np.array(wt)
        self.x_ = np.array(sx) / w
        self.y_ = np.array(sy) / w

    def _predict(self, p: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.clip(np.interp(p, self.x_, self.y_), 0.0, 1.0)
