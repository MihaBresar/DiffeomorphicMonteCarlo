# Contributing

Thank you for helping make the experiments more transparent and reproducible.

## Development workflow

1. Create a clean environment with `requirements-cpu.txt`.
2. Run `python -m unittest discover -s tests -v` and `python -m compileall -q .`.
3. Use a small `quick` run for the experiment you changed.
4. Include the resulting `configuration.json`, `manifest.json`, and a concise
   description of the hardware/backend when reporting a numerical change.

## Changes requiring explicit documentation

Treat these as scientific changes, not routine refactors:

- target log densities and parameterizations;
- synthetic-reference generators or their random seeds;
- sampler kernels, warmup/adaptation behavior, optimizer objectives, or
  numerical stabilizations;
- chain count, budget, thinning, burn-in, or reported performance metrics;
- external data or fixed posterior reference summaries.

For any such change, update the relevant document under `docs/`, retain the
old configuration where feasible, and do not overwrite archival figures.

## Code style

Keep experiment modules executable via `python -m experiments.<module>`, keep
the public identifier in `experiment_registry.py` stable, and route new output
through `reproducibility.finalize_experiment`.  Generated `results/` content
does not belong in routine commits unless it is an intentionally versioned
release artifact.
