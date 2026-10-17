import numpy as np
import pytest

from calibkit import bootstrap_ci, brier_score, brier_score_ci, ece_ci, expected_calibration_error


def _data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(size=n)
    y = (rng.uniform(size=n) < p).astype(int)
    return p, y


def test_estimate_matches_point_metric():
    p, y = _data()
    r = brier_score_ci(p, y, n_resamples=200, seed=1)
    assert r.estimate == pytest.approx(brier_score(p, y))
    r = ece_ci(p, y, n_resamples=200, seed=1, n_bins=10)
    assert r.estimate == pytest.approx(expected_calibration_error(p, y, n_bins=10))


def test_seed_reproducible_and_seeds_differ():
    p, y = _data()
    a = brier_score_ci(p, y, n_resamples=300, seed=7)
    b = brier_score_ci(p, y, n_resamples=300, seed=7)
    c = brier_score_ci(p, y, n_resamples=300, seed=8)
    assert (a.low, a.high) == (b.low, b.high)
    assert (a.low, a.high) != (c.low, c.high)


def test_interval_brackets_estimate_and_widens_with_confidence():
    p, y = _data()
    r90 = brier_score_ci(p, y, confidence_level=0.90, seed=3)
    r99 = brier_score_ci(p, y, confidence_level=0.99, seed=3)
    assert r90.low < r90.estimate < r90.high
    assert r99.high - r99.low > r90.high - r90.low


def test_interval_narrows_with_more_data():
    small = brier_score_ci(*_data(100), seed=0)
    large = brier_score_ci(*_data(2000), seed=0)
    assert (large.high - large.low) < (small.high - small.low)


def test_basic_method_runs_and_is_reflection():
    p, y = _data()
    pct = brier_score_ci(p, y, method="percentile", seed=5)
    basic = brier_score_ci(p, y, method="basic", seed=5)
    assert basic.low == pytest.approx(2 * pct.estimate - pct.high)
    assert basic.high == pytest.approx(2 * pct.estimate - pct.low)


def test_ece_ci_bounds_and_unpacking():
    p, y = _data()
    est, lo, hi = ece_ci(p, y, seed=2, n_resamples=200)
    assert 0.0 <= lo <= hi <= 1.0
    assert est >= 0.0


def test_generator_seed_accepted():
    p, y = _data()
    r = brier_score_ci(p, y, n_resamples=50, seed=np.random.default_rng(0))
    assert r.n_resamples == 50


def test_custom_metric():
    p, y = _data()
    r = bootstrap_ci(lambda a, b: float(np.mean(a)), p, y, n_resamples=100, seed=0)
    assert r.low < r.estimate < r.high


@pytest.mark.parametrize(
    "kwargs",
    [
        {"confidence_level": 1.0},
        {"confidence_level": 0.0},
        {"n_resamples": 1},
        {"method": "bca"},
    ],
)
def test_bad_arguments(kwargs):
    p, y = _data(50)
    with pytest.raises(ValueError):
        brier_score_ci(p, y, **kwargs)


@pytest.mark.slow
def test_brier_percentile_ci_coverage():
    """Nominal 95% percentile interval should cover the true Brier score ~95% of the time."""
    rng = np.random.default_rng(123)
    # True model: y ~ Bernoulli(p) with p ~ U(0,1); E[Brier] = E[p(1-p)] = 1/6.
    truth = 1.0 / 6.0
    trials, covered = 200, 0
    for t in range(trials):
        p = rng.uniform(size=200)
        y = (rng.uniform(size=200) < p).astype(int)
        r = brier_score_ci(p, y, n_resamples=300, seed=t)
        covered += r.low <= truth <= r.high
    assert 0.90 <= covered / trials <= 0.99
