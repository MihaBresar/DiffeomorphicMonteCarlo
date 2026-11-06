# Functional effective sample size

ESS is calculated only in the main DCS–SCS comparison and its Numba extension.
Other benchmark experiments skip chain diagnostics. ESS describes Monte Carlo
precision of a specific estimated mean; the main reports calculate it separately for the radial-tail **indicator**
and the **norm**. This is not coordinate ESS, rank-normalized bulk ESS, or the
quantile diagnostic that ArviZ calls tail ESS. The event's ESS is the raw mean
ESS of its zero/one time series.

The implementation uses the pinned ArviZ `ess(method='mean')`: each chain is
split into two halves, autocovariances are computed by FFT, and Geyer's
initial-positive and initial-monotone sequence rules truncate autocorrelation.
There is no rank transformation, thinning multiplier, deduplication or clipping
of the observable. An affine rescaling prevents numerical covariance overflow
without changing the observable's ESS. Anticorrelation can produce ESS larger
than the recorded draw count. See the [ArviZ API](https://python.arviz.org/en/v0.21.0/api/generated/arviz.ess.html)
and [Stan's analysis reference](https://mc-stan.org/docs/reference-manual/analysis.html).

## Which iterations are used

The current sampler comparisons capture every scalar radius during
production, including repeated states after rejection, then discard exactly
the first third. ESS uses that **unthinned post-burn-in** history. The configured
`thin` still controls the small saved trace, but does not change ESS, estimates
or MSE. Unthinned temporary histories are removed after diagnostics; they are
not duplicated into the saved chain archive.

The first/last half split drops at most one draw when the history length is odd;
`draws` and `estimator_draws` record this. A saved archive cannot reconstruct
transitions that were not retained.

## Time, memory and aggregation

Scalar capture and the final buffer flush are charged inside production time
for every comparison method. ESS computation starts after every chain in its
sampling wave finishes. Only one FFT workspace is active at a time, keeping
memory bounded when many CPU workers run. Diagnostic computation seconds are
recorded separately; waiting, tuning and compilation do not enter ESS/s.

Each chain has its own ESS. Tables show the **sum of per-chain ESS**, mean and
median ESS per chain, and the descriptive rate

`sum(per-chain ESS) / sum(full production seconds)`.

The denominator includes production burn-in, so rejected and discarded
iterations still cost time. The sum is not asserted to be the ESS of a
concatenated or unequal-length pooled estimator. It is not a cross-chain
convergence diagnostic. Raw counts, exact times and draw strides remain
available beside it.

## Unavailable values and limits

Fewer than eight post-burn-in draws, a constant trace, nonfinite values or a
nonfinite/nonpositive ESS estimate produce an unavailable value with a reason.
In particular, a chain that never observes a tail event does not receive an
artificial ESS equal to its sample count. If any chain lacks ESS, its method's
sum/mean/median and ESS/s are unavailable; valid-chain counts and reasons stay
visible. Successful subsets are not silently reported as complete totals.

For a Student-t norm with ν≤2, target variance is infinite and raw mean-norm
ESS would be undefined. The separate super-heavy-tail experiment reports
log norm and tail errors without calculating ESS. A finite estimated ESS never establishes convergence, lack of bias,
or a Markov-chain CLT. Interpret it with MSE, chain variability and the target's
moment assumptions. Current t(3) norm has finite variance but no fourth moment,
so empirical MSE uncertainty can still be unstable.
