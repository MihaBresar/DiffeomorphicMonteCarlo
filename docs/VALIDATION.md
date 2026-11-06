# Validation

Automated tests check mathematical kernels, experiment settings, saved outputs and public interfaces. Passing tests and Quick checks do not establish a performance ranking.

Local validation on 6 November 2026 passed **391 tests** and all four current
DCS/SCS Quick comparisons: 100D/500D in JAX and the Numba extension (**48 production chains**).
Both full-budget VI fits were replayed exactly, including every loss value.
All 64 pilot and six diagnostic scores were recomputed independently, and
fresh-fit and bundled-fit preparation produced identical arrays and selections.
The checks verify the actual backends, frozen steps, first-third discard and complete reports.
See the [current validation receipt](../validation/current_checks.json).

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
