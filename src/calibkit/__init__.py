"""calibkit: calibration metrics with uncertainty quantification."""

from .intervals import BootstrapResult, bootstrap_ci, brier_score_ci, ece_ci
from .metrics import brier_score, expected_calibration_error

__version__ = "0.1.0"

__all__ = [
    "BootstrapResult",
    "bootstrap_ci",
    "brier_score",
    "brier_score_ci",
    "ece_ci",
    "expected_calibration_error",
]
