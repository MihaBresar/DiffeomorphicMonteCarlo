# Figures and exact values

The [reported zero-centre study](REPORTED_STUDY.md) preserves all five radial
figure families, including ESS, and adds explicit frozen-map provenance. Its
reports use the study-specific steps, not the previous version-11 h=0.07
settings described below. The original reported batches are bundled in
`studies/zero_centre/results`.

Figures and tables are generated from each run's saved chains and references.
Both fixed-time and fixed-iteration designs are available. Filenames preserve
the corresponding manuscript figure families; numerical values describe the
recorded current experiment.

The two current sampler-comparison reports identify their selected target,
**VI-fitted maps for both families**, and JAX or optimized Numba execution.
Both support 100D and 500D skewed Student-t targets with ν=3 and DCS β=2.
See [the current comparison protocol](COMPARISONS.md). Both current reports
include DCS-RWM, DCS-BW, SCS, DCS-BW-reflect and DCS-Billiard. The Numba option runs the four
DCS methods natively and keeps author SCS in JAX. Their figures describe the
selected run; they do not imply a general sampler ranking.

New synthetic and logistic runs use the **Möbius VI** DCS map. Historical
**vanilla** labels retain their recorded meaning: R=1, centre=0 and Möbius=0,
with the experiment's β unchanged. Exact-value tables record the policy and
its parameter meaning. The two PosteriorDB targets keep their identity maps.

| Main paper file | Panels, in their original order |
|---|---|
| `NUTS_PBW_vs` | Radial tail and norm boxplots with numerical reference lines |
| `VsRWM_NUTS` | Linear MSE bars for tail probability and mean norm |
| `VsSCS` | Linear MSE bars for tail probability and mean norm, then average iteration count at equal time |
| `SuperInfiniteFirstTest` | Linear MSE bars for tail probability, mean log norm, then average runtime at equal iterations |
| `QQ-regresssion` | Two-by-two marginal Q–Q plots for β₁, with ν=4 and ν=6 |

Current reports identify `timed-sampler-comparison-v11` or
`timed-numba-comparison-v11`. They retain `VsSCS`, `scs_100d_estimates`,
`scs_100d_runtime` and `scs_100d_bias_variance`, and add `scs_100d_ess` for
mean functional ESS per chain and ESS/s. The filenames remain the same in
500D; titles identify the target. New comparisons report only radial-tail
probability and mean norm, with analytic references. Coordinate plots are no
longer generated. CSV/JSON preserve exact counts and full precision.

All methods have equal production-time and discarded pilot transition budgets.
All four DCS variants share the same fixed move parameter.
The single SCS baseline uses the authors' fixed h=0.1; new comparisons do not
offer alternative SCS step tuning. BW, reflected BW and billiard use fixed
increment radius h=0.07; RWM uses coordinate standard deviation h/√(d+2).
Each uses its own kernel during discarded warmup without step adaptation.
Public entry points select these preset steps automatically; historical
alternative-step reports retain their original recorded policy.
Every step freezes before production and the first third is discarded. Effective
maps and fixed steps are recorded. Adaptive historical controls additionally
export their billiard and reflected-BW pilot histories. See the
[current protocol](COMPARISONS.md) and [SCS step-policy study](AUTHOR_SCS_CALIBRATION.md).
Version 10 reports retain their recorded method-specific calibration rules;
replotting does not turn them into v11 fixed-move results. The fitted maps are
unchanged by the v11 step-policy update, but sampling must be rerun.
Functional ESS uses unthinned post-burn-in scalar histories, including
rejections. Any unavailable chain makes that method aggregate unavailable.
See [ESS definitions and limits](ESS.md). Replotting uses recorded measurements;
it cannot reconstruct unrecorded transitions.

The All seven sequence uses the current six-method JAX comparison on the selected
100D/500D skewed t, loading its bundled maps without refitting. The other six
experiments retain their specified fitted or identity-map policies.

The compact **200D random-walk baselines** report keeps just `VsRWM_NUTS`
(two MSE panels), `VsRWM_NUTS_ergodic_averages` (tail and norm estimates), and
`VsRWM_NUTS_sample_counts` (generated samples, retained samples and runtime).
It also reports functional ESS; it does not compute marginal quantiles, directional checks or R-hat.
Its HTML/CSV tables also include the average estimate across chains.

The other generic skew-t experiments also save the original four-coordinate marginal Q–Q
plots (`*_marginal_qq`) and radial boxplots (`*_ergodic_averages`, or the main
`NUTS_PBW_vs` file). The DCS/NUTS run restores `*_mse_and_ess`: two MSE panels
and aggregate ESS. ESS is the sum over chains of each chain's mean bulk ESS in
its first ten coordinates. An unavailable ESS is marked unavailable, never zero.

The generic skew-t runs additionally save `*_sample_counts`, showing mean generated
iterations, mean retained samples, and mean production runtime. The fixed-step
experiment has the same iterations per method by design; its runtime and MSE
compare efficiency. Timed experiments compare the iterations and retained draws
generated within the common wall-time budget, along with MSE. The super-heavy-tail
50D target (ν=0.5) now uses equal production iterations: 500,000 per chain and
50 chains per method under Full paper settings. It discards 166,667 iterations
and uses the remaining 333,333 for estimation, storing 33,334 post-burn-in draws
per chain at thinning 10. Its original image basenames and plot families remain, with
runtime replacing unequal iteration counts in the main figure's third panel.
For this target the mean norm does not exist; its second MSE panel continues to
estimate the finite mean of log₁₀(1+‖X‖), and the marginal quantile plots remain.

The original purple/orange/pink method colors, linear MSE axes, boxplots with
mean lines, and Q–Q median lines with empirical 5–95% chain bands are retained.
The logistic horizontal axis uses the independent reference, not the old pooled
output of the competing samplers. Bands describe variation across chains; they
are not confidence bands for a CLT. Unsaved third/fourth skew-t reference
coordinates with zero skew coefficient use their analytic Student-t marginal.
If a custom target gives an unsaved coordinate a nonzero skew coefficient, only
the independent IID reference is replayed using its recorded seed and batch size.
No MCMC is run to rebuild figures.

Where enabled in the generic experiments, directional checks and newer
logarithmic MSE plots with bootstrap intervals are in `figures/supplementary`.
The launcher puts them in a collapsed **Supplementary diagnostics** section.
The compact 200D baseline report skips these outputs. The two new comparison
suites report radial estimates, MSE and functional ESS with descriptive between-chain error
summaries. Coordinate statistics are omitted.

## Exact numerical values

Plot annotations and HTML tables use two decimals. MSE labels use scientific
notation, for example `2.10e-07`; runtimes use `33.20`. Small nonzero values use
scientific notation so they are never displayed as zero. Actual iteration and
sample counts remain exact integers. CSV and JSON retain every stored numerical
digit. These are numerical estimates, not exact population quantities; reference
uncertainty remains in the summary and exported data.

Each experiment saves:

- **`paper_values.html`**: a readable table opened directly in the browser.
- **`paper_values.csv`**: estimates, error summaries, references, chain counts,
  generated iterations, retained samples and timings. Additional uncertainty
  columns depend on the experiment's report profile.
- **`sample_counts.csv`**: exact integer totals and per-chain means; use the
  totals and chain count to recover a non-integer mean without display rounding.
- **`chain_metrics.csv`**, or **`scs_100d_chains.csv`** in the current comparisons:
  individual-chain estimates, squared errors and work counts, sufficient to
  reconstruct the reported MSE independently.
- **`*_marginal_qq.csv`** and **`QQ-regresssion.csv`**: the plotted quantiles and bands, where that experiment includes them.

The current comparison suites also save `tuned_parameters.csv`, `setup_times.csv`,
`radial_tuning.csv` and `reflected_tuning.csv`, recording map parameters,
excluded setup costs and pilot candidates. `reference_values.csv` records the
two analytic radial references. New comparisons record the actual fitting costs
and training counts for both families. Historical VI-off reports retain their
recorded zero fitting costs and separate pilot costs.

Generated iterations include repeated states after rejected proposals. Retained
samples exclude the first-third sampling burn-in and reflect thinning; separate
step tuning is excluded from all production counts. Every post-burn-in iteration
enters the new comparisons' estimates, including states omitted from the saved
thinned traces. Neither generated nor retained counts are effective sample sizes.
The separate posterior-table CSV retains full numerical precision as well.

## Open the generated figures

The launcher automatically shows figure and table links below the progress log
when a run finishes. Under **Exact numbers and tables**, open `paper_values.html`
or download the CSV. The files remain in `results/click-runs/<dated folder>`.
Advanced command-line users can rebuild completed results with `--plot-only`;
changing the kernel or worker count needs new sampling to measure its
performance. See [the rerun guide](RERUN.md).
