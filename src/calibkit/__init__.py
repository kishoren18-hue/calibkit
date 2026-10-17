"""calibkit: calibration metrics with uncertainty quantification."""

from .intervals import BootstrapResult, bootstrap_ci, brier_score_ci, ece_ci
from .metrics import (
    ReliabilityBins,
    adaptive_calibration_error,
    brier_score,
    debiased_ece_l2,
    expected_calibration_error,
    log_loss,
    maximum_calibration_error,
    reliability_bins,
)
from .plotting import plot_reliability_diagram

__version__ = "0.2.0"

__all__ = [
    "BootstrapResult",
    "ReliabilityBins",
    "adaptive_calibration_error",
    "bootstrap_ci",
    "brier_score",
    "brier_score_ci",
    "debiased_ece_l2",
    "ece_ci",
    "expected_calibration_error",
    "log_loss",
    "maximum_calibration_error",
    "plot_reliability_diagram",
    "reliability_bins",
]
