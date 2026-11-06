# Validation

Automated tests check mathematical kernels, experiment settings, saved outputs and public interfaces. Passing tests and Quick checks do not establish a performance ranking.

Local validation on 6 November 2026 passed the **397-test suite**, followed by
23 focused processing/policy tests and nine report tests after removing unused
posterior quantile sorting. All six standard Quick experiments completed
(**34 production chains**) without benchmark-chain diagnostics. Eight Schools
and GARCH were reprocessed without sampling; estimates, errors, counts, timing,
CSV tables and saved archives remained unchanged.

The main comparison's four Quick checks (**48 production chains**) and matched
preparation replay also passed. Both full-budget VI fits reproduced exactly;
all 64 pilot and six diagnostic scores were recomputed independently. These
main-comparison settings and ESS calculations are unchanged by the standard
experiment processing update. See the [current validation receipt](../validation/current_checks.json).

```sh
python -m pip check
python -m unittest discover -s tests -v
python -m examples.basic_sampler
python run_experiment.py reported-zero-centre --preset quick --workers 1 --output-dir results/check-jax
python run_experiment.py reported-zero-centre-numba --target skew-t-500d --preset quick --workers 1 --output-dir results/check-numba
```

Coverage includes target densities, Jacobians, inverse maps, boundary behavior, author-source integrity, SCS transition parity, reflected/billiard proposals and native/JAX agreement. Other checks cover frozen parameters, first-third burn-in, storage, counts, timing, ESS, worker allocation, exports and launcher routing.

CI uses Linux and Windows with Python 3.12. Consult [Actions](https://github.com/MihaBresar/DiffeomorphicMonteCarlo/actions) for actual results at a commit; configuration alone does not mean full experiments ran on every platform.

Numerical tests and execution checks are mandatory. Uploading compact reports
is an optional final step with seven-day retention; an upload/storage failure
does not hide an earlier test failure. Raw chain/reference archives are not
uploaded by CI. Check the workflow log if report artifacts are unavailable.

For scientific changes, check the affected complete Quick comparison and report, then choose independent production runs if performance assessment is needed. Documentation-only changes require link and configuration checks, not new simulations.
