# Experiment catalogue

Current experiments share a mathematical target within each comparison, discard the first third of production after separate warmup, and record resolved settings.

| Experiment | Command identifier | Comparison |
|---|---|---|
| DCS vs original SCS | `reported-zero-centre` | Six methods, equal time, 100D/500D |
| Numba extension | `reported-zero-centre-numba` | Same parameters; Numba DCS versus author JAX SCS |
| 200D skew-t versus NUTS | `skew-t-200d-nuts` | DCS-BW and NUTS, equal time |
| 200D random-walk baselines | `skew-t-200d-baselines` | DCS-BW, Gaussian/stable RWM and NUTS, equal time |
| 50D super-heavy tails | `skew-t-50d-nu-0.5` | DCS-RWM, DCS-BW and SCS, equal iterations |
| Eight Schools | `eight-schools` | HnR matched to mean NUTS production time |
| GARCH(1,1) | `garch11` | HnR matched to mean NUTS production time |
| Logistic regression | `logistic-regression` | DCS-BW and SCS, equal time, two priors |

The main comparisons follow the [frozen-map protocol](REPORTED_STUDY.md). Presets are 2 × 0.05 seconds, 20 × 3 seconds and 50 × 20 seconds per method, always with 1,500 separate warmup transitions.

Other experiments use named `configs/*.json` files and `configs/presets.json`. Quick uses two chains, 100 warmup transitions and small fitting/reference budgets. Smaller uses eight chains, 1,000 warmup transitions and reduced budgets. Full settings come from each named file. Inspect resolved settings with:

```sh
python run_experiment.py skew-t-200d-baselines --preset paper --dry-run
```

These six experiments use Numba DCS and random-walk production kernels. Logistic and super-heavy-tail SCS also use checked Numba transitions; NUTS retains its compiled backend. Every VI-enabled SCS fit uses the original author routine. Paired DCS/SCS fits match update counts and fixed-cloud sizes, retaining distinct parameterizations and reference distributions.

Eight Schools and GARCH use specified identity maps. Logistic references are independent. The t(0.5) mean norm does not exist; the finite mean log norm and tail probability are used instead.

**All seven experiments + posterior table** runs the JAX main comparison on the chosen dimension and the six additional experiments, not every dimension/backend combination. See [running instructions](RERUN.md), [targets](TARGETS.md) and [kernels](NATIVE_KERNELS.md).
