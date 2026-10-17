# Statistical assumptions

Every estimator in calibkit states the assumptions under which its output is valid.
If your data violates them, the numbers still compute, but they do not mean what they claim.

## Inputs

- Binary outcomes `y ∈ {0, 1}` and predicted probabilities `p ∈ [0, 1]` of the positive event.
  For LLMs, `p` is the stated or derived confidence and `y` is answer correctness.
- Multiclass top-label calibration reduces to this by using top-label confidence and correctness.

## Brier score

- Mean of `(p - y)^2`. Unbiased for the expected squared error under i.i.d. sampling.
- Mixes calibration and discrimination (refinement). A low Brier score is not proof of calibration.

## ECE (binned, plug-in)

- `ECE = Σ_b (n_b/N) · |acc_b − conf_b|`.
- **Binning dependence.** The value changes with `n_bins` and `strategy`. Report both.
- **Upward bias.** Within-bin noise inflates `|acc_b − conf_b|`, so ECE is biased upward,
  most strongly with small `N` or many bins. Even a perfectly calibrated model has ECE > 0 in finite samples.
- Equal-width bins are unstable where predictions concentrate (common for LLM confidence);
  prefer `strategy="quantile"` or compare both.

## Bootstrap confidence intervals

- **Case resampling** of `(p, y)` pairs with replacement; assumes observations are i.i.d. or exchangeable.
- **Not valid** for clustered data (many prompts per document/user) or time series
  unless you resample at the cluster/block level. Naive intervals will be too narrow.
- `percentile` is the default. It can undercover for small `N` and for skewed or boundary-limited
  statistics. `basic` reflects the quantiles around the estimate. BCa is planned.
- The ECE interval is the interval for the *binned plug-in statistic*, including its bias.
  It is **not** an interval for true calibration error.
- Coverage of the Brier interval is checked by simulation in `tests/test_intervals.py`
  (`pytest -m slow`). ECE coverage is not claimed.
