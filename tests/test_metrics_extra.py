import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from calibkit import (
    adaptive_calibration_error,
    debiased_ece_l2,
    expected_calibration_error,
    log_loss,
    maximum_calibration_error,
    plot_reliability_diagram,
    reliability_bins,
)


def test_log_loss_known_value():
    assert log_loss([0.8, 0.3], [1, 0]) == pytest.approx(-(np.log(0.8) + np.log(0.7)) / 2)


def test_log_loss_clips_confident_errors():
    v = log_loss([0.0, 1.0], [1, 0])
    assert np.isfinite(v) and v == pytest.approx(-np.log(1e-15), rel=1e-3)


def test_log_loss_bad_eps():
    with pytest.raises(ValueError):
        log_loss([0.5], [1], eps=0.6)


def test_reliability_bins_values():
    b = reliability_bins([0.1, 0.2, 0.9, 0.95], [0, 1, 1, 1], n_bins=2)
    assert list(b.counts) == [2, 2]
    assert b.mean_confidence == pytest.approx([0.15, 0.925])
    assert b.accuracy == pytest.approx([0.5, 1.0])


def test_reliability_bins_empty_bin_is_nan():
    b = reliability_bins([0.05, 0.06], [0, 1], n_bins=4)
    assert list(b.counts) == [2, 0, 0, 0]
    assert np.isnan(b.accuracy[1:]).all()


def test_ece_consistent_with_reliability_bins():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=400)
    y = (rng.uniform(size=400) < p**1.5).astype(int)
    b = reliability_bins(p, y, n_bins=10)
    ok = b.counts > 0
    manual = np.sum(b.counts[ok] / 400 * np.abs(b.accuracy[ok] - b.mean_confidence[ok]))
    assert expected_calibration_error(p, y, n_bins=10) == pytest.approx(manual)


def test_mce_at_least_ece():
    rng = np.random.default_rng(1)
    p = rng.uniform(size=300)
    y = (rng.uniform(size=300) < p**2).astype(int)
    assert maximum_calibration_error(p, y) >= expected_calibration_error(p, y)


def test_mce_hand_computed():
    # bins at n_bins=2: gaps are 0.2 (bin0: conf .3, acc .5) and 0.0 (bin1: conf .85, acc .85)
    p = [0.2, 0.4, 0.8, 0.9]
    y = [0, 1, 1, 1]
    assert maximum_calibration_error(p, y, n_bins=2) == pytest.approx(0.2)


def test_ace_equals_mean_gap_on_equal_mass_bins():
    p = np.array([0.1, 0.2, 0.8, 0.9])
    y = np.array([0, 0, 1, 1])
    # two quantile bins: gaps |0-.15| and |1-.85| -> mean .15
    assert adaptive_calibration_error(p, y, n_bins=2) == pytest.approx(0.15)


def test_debiased_less_biased_than_plugin_when_calibrated():
    rng = np.random.default_rng(42)
    plug, deb = [], []
    for _ in range(200):
        p = rng.uniform(size=300)
        y = (rng.uniform(size=300) < p).astype(int)  # perfectly calibrated
        plug.append(expected_calibration_error(p, y, n_bins=15, strategy="quantile", norm="l2"))
        deb.append(debiased_ece_l2(p, y, n_bins=15, strategy="quantile"))
    assert np.mean(deb) < 0.5 * np.mean(plug)


def test_debiased_detects_miscalibration():
    rng = np.random.default_rng(7)
    p = rng.uniform(0.7, 1.0, size=5000)
    y = (rng.uniform(size=5000) < p - 0.2).astype(int)  # 0.2 overconfident
    assert debiased_ece_l2(p, y) == pytest.approx(0.2, abs=0.03)


def test_debiased_needs_two_per_bin():
    with pytest.raises(ValueError, match="at least 2"):
        debiased_ece_l2([0.1, 0.9], [0, 1], n_bins=2, strategy="uniform")


def test_plot_returns_axes():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=100)
    y = (rng.uniform(size=100) < p).astype(int)
    ax = plot_reliability_diagram(p, y, n_bins=5)
    assert ax.get_xlabel() == "mean predicted confidence"
    assert len(ax.lines) == 2
