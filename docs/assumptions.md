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

## Other metrics (v0.2)

- **Log-loss:** probabilities are clipped to `[eps, 1-eps]`; the value depends on `eps` for confident errors.
- **MCE:** maximum bin gap. A maximum over noisy gaps, so more upward-biased than ECE. Sparse bins dominate it.
- **ACE:** unweighted mean gap over equal-mass bins (binary-outcome version of Nixon et al., 2019).
- **`debiased_ece_l2`:** subtracts the within-bin sampling-noise term from the squared gap (after Kumar et al., 2019).
  Treats bin assignment as fixed, skips bins with fewer than 2 points, and floors at 0, so it is
  *less* biased than plug-in L2, not unbiased. Its bias reduction is checked by simulation in the tests.

## Recalibration (v0.3)

- Fit on data **disjoint** from the data you evaluate on. Reusing the fit data makes calibration look better than it is.
- Assumes the miscalibration is stable between fit and use. It does not survive distribution shift.
- **Temperature:** one parameter, preserves ranking. Cannot fix a shifted base rate.
- **Platt:** slope and intercept in logit space. Assumes a sigmoid-shaped distortion; a small L2 penalty keeps it finite on separable data.
- **Isotonic:** monotone, nonparametric. Needs hundreds of points; overfits small samples and creates ties.

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
