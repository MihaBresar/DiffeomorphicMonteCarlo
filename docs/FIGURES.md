# Original figures and exact values

The original figure families are restored. The corrected saved chains and
references determine the numbers; old paper values are never pasted into new
results. Both fixed-time and fixed-iteration designs remain in the repository.

New DCS labels identify the selected map as **Möbius VI** or **vanilla**. Vanilla
means R=1, centre=0 and Möbius=0, with the experiment's β unchanged. Exact-value
tables record the policy and its parameter meaning. The two PosteriorDB targets
keep their identity maps. Old fixed-R=√d results retain their historical labels.

| Main paper file | Panels, in their original order |
|---|---|
| `NUTS_PBW_vs` | Radial tail and norm boxplots with numerical reference lines |
| `VsRWM_NUTS` | Linear MSE bars for tail probability and mean norm |
| `VsSCS` | Linear MSE bars for tail probability, mean norm, then average runtime |
| `SuperInfiniteFirstTest` | Linear MSE bars for tail probability, mean log norm, then average runtime at equal iterations |
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
generated within the common wall-time budget, along with MSE. The super-heavy-tail
50D target (ν=0.5) now uses equal production iterations: 500,000 per chain and
50 chains per method under Full paper settings, retaining 50,000 draws per chain
at thinning 10. Its original image basenames and plot families remain, with
runtime replacing unequal iteration counts in the main figure's third panel.
For this target the mean norm does not exist; its second MSE panel continues to
estimate the finite mean of log₁₀(1+‖X‖), and the marginal quantile plots remain.
Older timed results keep their recorded design when replotted.

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

## Open the generated figures

The launcher automatically shows figure and table links below the progress log
when a run finishes. Under **Exact numbers and tables**, open `paper_values.html`
or download the CSV. The files remain in `results/click-runs/<dated folder>`.
Advanced command-line users can rebuild completed results with `--plot-only`;
changing the kernel or worker count needs new sampling to measure its
performance. See [the rerun guide](RERUN.md).
