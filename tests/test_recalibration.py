import numpy as np
import pytest

from calibkit import (
    IsotonicRegression,
    PlattScaling,
    TemperatureScaling,
    expected_calibration_error,
    log_loss,
)


def _overconfident(n, seed, temp=3.0, bias=0.0):
    """True logit z; the model reports probabilities from the inflated logit temp*z + bias."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 1.5, size=n)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-z))).astype(int)
    p = 1 / (1 + np.exp(-(temp * z + bias)))
    return p, y


def test_temperature_recovers_known_temperature():
    p, y = _overconfident(20000, 0, temp=3.0)
    t = TemperatureScaling().fit(p, y).temperature_
    assert t == pytest.approx(3.0, rel=0.1)


def test_temperature_is_one_when_already_calibrated():
    p, y = _overconfident(20000, 1, temp=1.0)
    assert TemperatureScaling().fit(p, y).temperature_ == pytest.approx(1.0, abs=0.08)


def test_platt_recovers_slope_and_intercept():
    p, y = _overconfident(30000, 2, temp=2.0, bias=1.0)
    m = PlattScaling().fit(p, y)
    # logit(p) = 2 z + 1  ->  z = (logit - 1)/2, so a = 0.5, b = -0.5
    assert m.a_ == pytest.approx(0.5, abs=0.05)
    assert m.b_ == pytest.approx(-0.5, abs=0.07)


@pytest.mark.parametrize("model", [TemperatureScaling, PlattScaling, IsotonicRegression])
def test_recalibration_improves_heldout_calibration(model):
    p_fit, y_fit = _overconfident(4000, 3)
    p_test, y_test = _overconfident(4000, 4)
    before = expected_calibration_error(p_test, y_test)
    after = expected_calibration_error(model().fit(p_fit, y_fit).predict(p_test), y_test)
    assert after < 0.5 * before


@pytest.mark.parametrize("model", [TemperatureScaling, PlattScaling])
def test_parametric_models_reduce_log_loss(model):
    p_fit, y_fit = _overconfident(4000, 5)
    p_test, y_test = _overconfident(4000, 6)
    assert log_loss(model().fit(p_fit, y_fit).predict(p_test), y_test) < log_loss(p_test, y_test)


@pytest.mark.parametrize("model", [TemperatureScaling, PlattScaling, IsotonicRegression])
def test_outputs_in_unit_interval_and_monotone(model):
    p, y = _overconfident(1000, 7)
    m = model().fit(p, y)
    grid = np.linspace(0, 1, 201)
    out = m.predict(grid)
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert np.all(np.diff(out) >= -1e-12)


def test_isotonic_pav_known_values():
    # scores 0.1..0.5 with outcomes 0,1,0,1,1 -> pools (1,0) into .5
    m = IsotonicRegression().fit([0.1, 0.2, 0.3, 0.4, 0.5], [0, 1, 0, 1, 1])
    assert m.predict([0.1, 0.2, 0.3, 0.4, 0.5]) == pytest.approx([0.0, 0.5, 0.5, 1.0, 1.0], abs=0.3)
    assert list(m.y_) == [0.0, 0.5, 1.0]


def test_isotonic_handles_ties():
    m = IsotonicRegression().fit([0.5, 0.5, 0.5, 0.9], [0, 1, 1, 1])
    assert np.all(np.diff(m.x_) > 0)
    assert m.predict([0.5])[0] == pytest.approx(2 / 3)


def test_platt_finite_on_separable_data():
    p = np.array([0.1, 0.2, 0.8, 0.9])
    y = np.array([0, 0, 1, 1])
    m = PlattScaling().fit(p, y)
    assert np.isfinite(m.a_) and np.isfinite(m.b_)


def test_temperature_extreme_probabilities_do_not_break():
    m = TemperatureScaling().fit([0.0, 1.0, 0.0, 1.0, 0.5], [0, 1, 1, 0, 1])
    assert np.isfinite(m.temperature_)


def test_errors():
    with pytest.raises(RuntimeError):
        TemperatureScaling().predict([0.5])
    m = PlattScaling().fit([0.2, 0.8, 0.4, 0.6], [0, 1, 0, 1])
    with pytest.raises(ValueError):
        m.predict([1.5])
    with pytest.raises(ValueError):
        TemperatureScaling(t_min=2, t_max=1)
    with pytest.raises(ValueError):
        IsotonicRegression().fit([0.5, 1.5], [0, 1])
