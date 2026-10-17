# calibkit

Calibration metrics with honest uncertainty for probabilistic classifiers and LLM confidence scores.
Only depends on NumPy.

```python
import calibkit as ck

est, lo, hi = ck.ece_ci(confidences, correct, n_bins=15, strategy="quantile", seed=0)
brier = ck.brier_score_ci(confidences, correct, confidence_level=0.95, seed=0)
```

## Status

Milestone 1 (v0.1): ECE (L1/L2, uniform/quantile bins), Brier score, bootstrap CIs
(percentile/basic), tests, CI.

Planned: reliability diagrams, more metrics (debiased ECE, ACE, log-loss), recalibration methods
(temperature, isotonic, Platt), shift detection, model-provider adapters, BCa intervals.

## Correct use

Read [docs/assumptions.md](docs/assumptions.md). In short: ECE depends on binning and is biased
upward; bootstrap CIs assume i.i.d. data.

## Development

```bash
pip install -e ".[dev]"
pytest                 # all tests
pytest -m "not slow"   # skip coverage simulations
ruff check . && mypy
```

## License

MIT
