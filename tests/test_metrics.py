import numpy as np
import pytest

from calibkit import brier_score, expected_calibration_error


def test_brier_known_value():
    # errors: 0.1, 0.2, 0.3, 0.4 -> squares .01 .04 .09 .16 -> mean .075
    assert brier_score([0.9, 0.8, 0.3, 0.4], [1, 1, 0, 0]) == pytest.approx(0.075)


def test_brier_perfect_and_worst():
    assert brier_score([1, 0, 1], [1, 0, 1]) == 0.0
    assert brier_score([0, 1], [1, 0]) == 1.0


def test_ece_hand_computed():
    # two bins at n_bins=2. bin0 [0,.5): p=.2,.4 conf=.3, y=0,1 acc=.5 gap=.2
    # bin1 [.5,1]: p=.6,.8,1.0 conf=.8, y=1,1,0 acc=2/3 gap=2/15
    p = [0.2, 0.4, 0.6, 0.8, 1.0]
    y = [0, 1, 1, 1, 0]
    expected = (2 / 5) * 0.2 + (3 / 5) * (0.8 - 2 / 3)
    assert expected_calibration_error(p, y, n_bins=2) == pytest.approx(expected)


def test_ece_l2_hand_computed():
    p = [0.2, 0.4, 0.6, 0.8, 1.0]
    y = [0, 1, 1, 1, 0]
    expected = np.sqrt((2 / 5) * 0.2**2 + (3 / 5) * (0.8 - 2 / 3) ** 2)
    assert expected_calibration_error(p, y, n_bins=2, norm="l2") == pytest.approx(expected)


def test_ece_probability_one_lands_in_last_bin():
    assert expected_calibration_error([1.0, 1.0], [1, 1], n_bins=10) == 0.0


def test_ece_perfectly_calibrated_bins_is_zero():
    p = [0.25] * 4 + [0.75] * 4
    y = [1, 0, 0, 0] + [1, 1, 1, 0]
    assert expected_calibration_error(p, y, n_bins=4) == pytest.approx(0.0)


def test_ece_overconfident_constant():
    # always says 0.9 but right half the time
    p = [0.9] * 10
    y = [1] * 5 + [0] * 5
    assert expected_calibration_error(p, y) == pytest.approx(0.4)


def test_ece_quantile_strategy_handles_ties_and_constants():
    assert expected_calibration_error([0.7] * 6, [1, 0, 1, 0, 1, 0], strategy="quantile") == (
        pytest.approx(0.2)
    )


def test_ece_bounds_random():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=500)
    y = rng.integers(0, 2, size=500)
    for strategy in ("uniform", "quantile"):
        v = expected_calibration_error(p, y, strategy=strategy)
        assert 0.0 <= v <= 1.0


def test_ece_invariant_to_permutation():
    rng = np.random.default_rng(1)
    p = rng.uniform(size=200)
    y = (rng.uniform(size=200) < p).astype(int)
    perm = rng.permutation(200)
    assert expected_calibration_error(p, y) == pytest.approx(
        expected_calibration_error(p[perm], y[perm])
    )


@pytest.mark.parametrize(
    "p, y, msg",
    [
        ([0.5, 1.2], [1, 0], "lie in"),
        ([0.5, -0.1], [1, 0], "lie in"),
        ([0.5, 0.5], [1, 2], "binary"),
        ([0.5], [1, 0], "mismatch"),
        ([], [], "non-empty"),
        ([0.5, float("nan")], [1, 0], "NaN"),
        ([[0.5], [0.5]], [1, 0], "1-dimensional"),
    ],
)
def test_validation_errors(p, y, msg):
    with pytest.raises(ValueError, match=msg):
        brier_score(p, y)
    with pytest.raises(ValueError, match=msg):
        expected_calibration_error(p, y)


def test_bad_options():
    with pytest.raises(ValueError):
        expected_calibration_error([0.5], [1], n_bins=0)
    with pytest.raises(ValueError):
        expected_calibration_error([0.5], [1], strategy="nope")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        expected_calibration_error([0.5], [1], norm="l3")  # type: ignore[arg-type]
