# Reproducing and reporting an experiment

## Recommended workflow

1. Start from a fresh clone and a clean Python 3.11 virtual environment.
2. Install `requirements-cpu.txt` and record `python -m pip freeze` in the run
   archive for every final figure.
3. Run the static checks:

   ```bash
   python -m unittest discover -s tests -v
   python -m compileall -q .
   ```

   The GitHub Actions workflow also installs and imports the documented CPU
   stack, but intentionally does not run a costly scientific benchmark.

4. Run `--preset quick` first.  Inspect `run.log`, `summary.json`, and the
   saved figures before spending resources on a paper-scale run.
5. Run `--preset paper` with an explicit `--output-dir`, retaining the entire
   result directory and the terminal command used.

## Determinism and machine dependence

The synthetic reference generators use fixed JAX seeds.  Chain workers use
deterministic, script-defined seed sequences.  Bitwise identity is not
promised across JAX/JAXLIB versions, CPU architectures, accelerator backends,
or parallel execution strategies.  `manifest.json` captures the JAX backend,
devices, selected scientific-package versions, Python version, platform,
outer CLI invocation, child module command, and `HEAVY_TAILS_*` overrides so a
comparison can be interpreted in context.

For timing experiments, report all of the following:

- CPU/GPU model, operating system, and number of physical/logical cores;
- Python, JAX/JAXLIB, NumPyro, BlackJAX, NumPy, SciPy, and ArviZ versions;
- selected backend and worker count;
- whether timings include warmup, adaptation, optimization, reference
  generation, compilation, and data loading;
- wall-clock elapsed time as well as any aggregate ESS-derived throughput.

The synthetic time-budget modules intentionally exclude most setup work.  In
contrast, Eight Schools and GARCH measure NumPyro MCMC construction, warmup,
and draws, then use the average NUTS elapsed time as the projected
slice/HnR per-chain budget.  An ESS/sec value that aggregates over chains is a
throughput diagnostic, not a substitute for a full end-to-end wall-clock
benchmark.

## Quick versus paper profiles

`quick` deliberately shrinks values in `configs/presets.json`; it may change
optimization quality and estimator variability.  Never substitute a quick run
for a paper result.  `paper` leaves the source defaults intact, including
large synthetic reference sizes and lengthy chains.  Estimate resource use on
your hardware before starting it.

## Final release bundle

For each published figure/table, preserve:

- the Git commit SHA and a clean `git status` output;
- the exact CLI command and full `results/<experiment>/<timestamp>/` directory;
- `python -m pip freeze` and relevant hardware/backend details;
- data checksums and any externally acquired posterior reference draws;
- the completed target/reference validation record from
  [`VALIDATION.md`](VALIDATION.md).
