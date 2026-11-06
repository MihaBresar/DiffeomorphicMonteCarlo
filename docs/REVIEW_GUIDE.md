# Reviewer guide

Read the [comparison protocol](REPORTED_STUDY.md), [targets](TARGETS.md) and [SCS provenance](../third_party/scs_paper/README.md). The repository provides current executable tests without an assumed ranking.

## Independent check

Install Python 3.12 and pinned dependencies, or use the launcher.

```sh
python run_experiment.py reported-zero-centre --preset quick --workers 1 --output-dir results/reviewer-check
python run_experiment.py --plot-only results/reviewer-check
```

Quick verifies execution. Use Smaller or Full settings for assessment. Eight seed choices change sampling; they do not measure refitting variability.

| Check | Evidence |
|---|---|
| Target/reference | Resolved configuration, target definitions and analytic radial constants |
| Author source | Unmodified SCS projection, VI and transition with revision/checksums |
| Effective maps | Source fits and explicit per-method production overrides |
| Budgets | Chain count, requested time, warmup, worker allocation |
| Observations | Exact first-third discard, rejected repeats included, declared estimator basis |
| Timing | Actual seconds, overshoot, separate preparation/diagnostic costs |
| Diagnostics | Functional ESS, valid-chain counts and unavailable reasons |
| Reproducibility | Source hashes, environment, seeds and full-precision exports |

## Scope

Original SCS keeps author VI and h=0.1 as the author baseline. DCS and calibrated zero-SCS select R/h on their final zero-centre/deformation maps using the same number of candidates, common exact physical starts, transition budgets and score. The DCS-BW selection is shared across the four DCS variants.

The [public preparation driver](PREPARATION.md) reproduces both full-budget source fits and calibration. It records seeds, candidate outcomes, selection freeze, source hashes and preparation costs. Native VI routines and proposal units remain different. Equal counts do not imply equal elapsed cost or globally optimal parameters. The pilot uses exact target starts, analytic radial information and a lag-eight movement proxy rather than production MSE or ESS.

The common target is normal-CDF-skewed t(3), not the SCS paper's skew family. JAX provides a common execution framework; the Numba extension changes DCS implementation while retaining author JAX SCS.

A common framework does not establish equal engineering effort or equally
optimized kernels. Production timing also excludes preparation. Order rotates
between waves only; a single-wave run has no counterbalancing.

Radial measurements do not establish coordinate or general superiority. The t(3) norm has no fourth moment, complicating empirical MSE uncertainty. ESS is not proof of convergence.

Additional experiments follow their own documented fitting/reference policies. Their Numba SCS transitions are not represented as unmodified author performance implementations. See [the catalogue](EXPERIMENTS.md).

## Reporting

Share commit, configuration, manifest, summary, worker and seed selections. Keep all methods and chains. Investigate failures before interpreting incomplete comparisons. See [contributing](../CONTRIBUTING.md) and [validation](VALIDATION.md).
