# Figures and exact values

Open `paper_values.html` after a run. It links the generated PNG/PDF figures and full-precision CSV exports. The report belongs to its saved configuration and source revision.

## Main comparison

Both JAX and Numba modes produce the same six-method report layout:

- `VsSCS`: radial-tail and mean-norm MSE comparison.
- `scs_100d_estimates`: chain estimates against references.
- `scs_100d_runtime`: counts and runtime.
- `scs_100d_bias_variance`: error decomposition.
- `scs_100d_ess`: functional ESS and ESS/s.

These filenames also apply to 500D; the saved target and report labels identify dimension. No coordinate-marginal statistics enter this comparison. See [the protocol](REPORTED_STUDY.md).

## Exact values

`paper_values.csv` and `summary.json` retain full precision. `sample_counts.csv` records generated, discarded and estimation counts. `scs_100d_chains.csv` records individual-chain estimates and settings. `functional_ess.csv` records each functional's ESS basis and availability. Map, preparation and tuning tables describe the actual supplied parameters.

Displayed real values use two decimals or scientific notation. Integer sample counts are exact. Every post-burn-in observation, including rejection repeats, contributes to main-comparison estimates and ESS; saved-trace thinning does not change them.

## Additional experiments

The 50D super-heavy-tail report uses tail probability and finite mean log norm. The 200D baseline report contains radial estimates, MSE, counts, time and ESS. Logistic reports compare quantiles with an independent reference. Eight Schools and GARCH report physical-parameter moments and posterior diagnostics.

The applicable references, estimators and retained-draw stride are saved with each result. Comparing errors across different targets or timing policies requires their configurations.

Recreate figures from a completed run with:

```sh
python run_experiment.py --plot-only results/example-run
```

This uses saved results without new sampling. A changed scientific configuration requires a new complete comparison.
