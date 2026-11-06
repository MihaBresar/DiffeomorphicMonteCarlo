# Validation

Automated tests check mathematical kernels, experiment settings, saved outputs and public interfaces. Passing tests and Quick checks do not establish a performance ranking.

Local validation on 6 November 2026 passed **382 tests**, all six standard
Quick experiments (**34 production chains**) and **72 launcher seed-control
checks**. The records verify the required Numba/native NUTS backends and
unchanged published DCS/SCS settings. See the
[current validation receipt](../validation/current_checks.json).

```sh
python -m pip check
python -m unittest discover -s tests -v
python -m examples.basic_sampler
python run_experiment.py reported-zero-centre --preset quick --workers 1 --output-dir results/check-jax
python run_experiment.py reported-zero-centre-numba --target skew-t-500d --preset quick --workers 1 --output-dir results/check-numba
```

Coverage includes target densities, Jacobians, inverse maps, boundary behavior, author-source integrity, SCS transition parity, reflected/billiard proposals and native/JAX agreement. Other checks cover frozen parameters, first-third burn-in, storage, counts, timing, ESS, worker allocation, exports and launcher routing.

CI uses Linux and Windows with Python 3.12. Consult [Actions](https://github.com/MihaBresar/DiffeomorphicMonteCarlo/actions) for actual results at a commit; configuration alone does not mean full experiments ran on every platform.

For scientific changes, check the affected complete Quick comparison and report, then choose independent production runs if performance assessment is needed. Documentation-only changes require link and configuration checks, not new simulations.
