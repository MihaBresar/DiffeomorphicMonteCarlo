# Original figures and exact values

The original figure families are restored. The corrected saved chains and
references determine the numbers; old paper values are never pasted into new
results. Both fixed-time and fixed-iteration designs remain in the repository.

| Main paper file | Panels, in their original order |
|---|---|
| `NUTS_PBW_vs` | Radial tail and norm boxplots with numerical reference lines |
| `VsRWM_NUTS` | Linear MSE bars for tail probability and mean norm |
| `VsSCS` | Linear MSE bars for tail probability, mean norm, then average runtime |
| `SuperInfiniteFirstTest` | Linear MSE bars for tail probability, mean log norm, then average iterations |
| `QQ-regresssion` | Two-by-two marginal Q–Q plots for β₁, with ν=4 and ν=6 |

Every skew-t experiment also saves the original four-coordinate marginal Q–Q
plots (`*_marginal_qq`) and radial boxplots (`*_ergodic_averages`, or the main
`NUTS_PBW_vs` file). The DCS/NUTS run restores `*_mse_and_ess`: two MSE panels
and aggregate ESS. ESS is the sum over chains of each chain's mean bulk ESS in
its first ten coordinates. An unavailable ESS is marked unavailable, never zero.

Each skew-t run additionally saves `*_sample_counts`, showing mean generated
iterations, mean retained samples, and mean production runtime. The fixed-step
experiment has the same iterations per method by design; its runtime and MSE
compare efficiency. Timed experiments compare the iterations and retained draws
generated within the common wall-time budget, along with MSE.

The original purple/orange/pink method colors, linear MSE axes, boxplots with
mean lines, and Q–Q median lines with empirical 5–95% chain bands are retained.
The logistic horizontal axis uses the independent reference, not the old pooled
output of the competing samplers. Bands describe variation across chains; they
are not confidence bands for a CLT. Unsaved third/fourth skew-t reference
coordinates with zero skew coefficient use their analytic Student-t marginal.
If a custom target gives an unsaved coordinate a nonzero skew coefficient, only
the independent IID reference is replayed using its recorded seed and batch size.
No MCMC is run to rebuild figures.

Directional checks and the newer logarithmic MSE plots with bootstrap intervals
are in `figures/supplementary`. The launcher puts them in a collapsed
**Supplementary diagnostics** section. They supplement all the original plots.

## Exact numerical values

MSE, iteration/sample counts, and runtime bars print the shortest decimal that
recovers the stored floating-point value without loss. These are numerical
estimates, not exact population quantities. References may have Monte Carlo
uncertainty; that uncertainty remains in the summary and exported data.

Each experiment saves:

- **`paper_values.html`**: a readable table opened directly in the browser.
- **`paper_values.csv`**: all MSE/RMSE values, reference values and MCSE, bootstrap
  intervals, chain counts, generated iterations, retained samples and timings.
- **`sample_counts.csv`**: integer totals and per-chain means; exact fractions
  accompany means so non-integer means are never silently rounded down.
- **`chain_metrics.csv`**: individual-chain estimates, squared errors and work
  counts, sufficient to reconstruct the reported MSE independently.
- **`*_marginal_qq.csv`** and **`QQ-regresssion.csv`**: the plotted quantiles and bands.

Generated iterations include repeated states after rejected proposals. Retained
samples exclude warmup and reflect thinning. Neither is an effective sample size.
The separate posterior-table CSV retains full numerical precision as well.

## Rebuild with clicks

Open **Start.bat** on Windows or **Start.command** on macOS. Select a completed
run under **Saved launcher runs**, click **Rebuild figures**, then
**Show figures and tables**. Under **Exact numbers and tables**, open
`paper_values.html` or download the CSV. Existing completed runs can be replotted
with the updated code; changing the kernel or worker count needs new sampling
to measure its performance. See [the rerun guide](RERUN.md).
