# Heavy-Tailed Sampling Benchmarks for a Projection Sampler

This repository contains the simulation code supporting a paper on a novel
projection sampler for challenging heavy-tailed targets.  It preserves the
research implementations while adding stable experiment names, portable data,
small smoke configurations, saved figures, and per-run provenance files.

All benchmark model specifications and ground-truth/reference quantities are
sourced from PosteriorDB. Some experiment modules reproduce reference
quantities locally with fixed-seed Monte Carlo calculations; those local
calculations should be checked against the corresponding pinned PosteriorDB
records before the final paper release.

The repository deliberately does **not** make a performance claim in this
README.  Before using any generated number in a manuscript, read the
[validation notes](docs/VALIDATION.md); they identify checks that are still
needed to establish the intended target/reference correspondence.

## Quick start

Use a clean Python 3.11 environment.  This is the documented CPU baseline;
record the exact installed package set for a final paper run.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-cpu.txt
python -m unittest discover -s tests -v

python run_experiment.py --list
python run_experiment.py skew-t-50d-nu-0.5 --preset quick
```

Continuous integration installs and imports this dependency stack and runs the
repository-contract checks, but it does not certify the numerical conclusions
of a full scientific experiment.

`quick` is a reduced smoke profile.  It verifies the execution path and writes
artifacts; it is not intended to reproduce paper-quality estimates.  To use a
module's original paper-scale constants, use `--preset paper`:

```bash
python run_experiment.py skew-t-100d-baselines --preset paper \
  --output-dir results/paper/skew-t-100d-baselines
```

Use `--dry-run` to inspect the command and output location without starting an
experiment.  Command-line overrides such as `--workers`, `--runs`,
`--time-budget`, `--iterations`, `--reference-samples`,
`--reference-batch-size`, and `--optimizer-steps` make an invocation explicit
and are stored in its manifest.  The runner rejects a flag that is not used by
the chosen experiment and automatically chooses a compatible reference batch
size when only `--reference-samples` is supplied.

## What a run saves

Each successful CLI run writes to `results/<experiment>/<UTC timestamp>/` by
default:

```text
results/<experiment>/<timestamp>/
├── configuration.json  # model and sampler settings reported by the module
├── manifest.json       # outer CLI, child command, package/JAX/platform metadata
├── run.log             # runner header plus combined child stdout/stderr
├── summary.json        # aggregate numerical diagnostics
└── figures/            # saved PNG and PDF figures (validate before publication)
```

`results/` is intentionally ignored by Git.  Keep the directory for each
manuscript figure, or archive it alongside the paper's data/code release.

## Experiment catalogue

| Public identifier | Target and comparison | Paper-scale profile |
| --- | --- | --- |
| `eight-schools` | Centered Eight Schools funnel; NumPyro NUTS vs. projected slice/HnR | 50 chains; 150,000 post-warmup NUTS draws per chain |
| `garch11` | PosteriorDB GARCH(1,1); NumPyro NUTS vs. projected slice/HnR | 50 chains; 150,000 post-warmup NUTS draws per chain |
| `logistic-regression` | Student-t-prior logistic regression at ν=4, 6; DCS-BW/PBW vs. SCS | 20 chains per sampler/ν; 20 seconds per chain |
| `skew-t-50d-nu-0.5` | 50-dimensional skew-t at ν=0.5; DCS-RWM, DCS-BW/PBW, SCS | 50 chains; 10 seconds per chain; 10M-draw reference estimate |
| `skew-t-100d-fixed` | 100-dimensional skew-t at ν=2; DCS-RWM, DCS-BW/PBW, SCS | 20 chains; 500,000 iterations per chain; 10M-draw reference estimate |
| `skew-t-200d-nuts` | 200-dimensional skew-t at ν=3; DCS-BW/PBW vs. BlackJAX NUTS | 50 chains; 20 seconds per chain; 10M-draw reference estimate |
| `skew-t-100d-baselines` | 100-dimensional skew-t at ν=3; DCS-BW/PBW, Gaussian RWM, symmetric α-stable RWM (α=0.5), BlackJAX NUTS | 50 chains; 10 seconds per chain; 10M-draw reference estimate |

Further details, including the exact module and output diagnostics, are in
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

## Repository layout

```text
experiments/            Descriptive experiment modules; run through the CLI
configs/presets.json    `quick` overrides and the empty `paper` profile
data/raw/garch11.json   Bundled PosteriorDB GARCH(1,1) data
figures/archival/       Original supplied figures, retained as archival assets
docs/                   Reproduction, data provenance, and validation notes
run_experiment.py       Public entry point
run_*.py                Descriptive one-experiment convenience entry points
reproducibility.py      Shared artifact/provenance helpers
```

Descriptive `run_*.py` convenience entry points are provided for direct
execution. For reproducible work, prefer `run_experiment.py` and the stable
identifiers above because that path records the selected preset and run
metadata.

## Terminology and comparability

The original working scripts used several overlapping labels (for example,
`DCS-BW`, `PBW`, `DCS-RWM`, `PHR`, and `HnR`).  This cleanup preserves the
algorithms and uses cautious combined labels such as **DCS-BW/PBW** and
**projected slice/HnR** where the source did not establish a single canonical
expansion.  Before publication, replace these combined labels with the exact
notation and definition adopted by the paper.

Timing policy is target-specific.  The synthetic time-budget experiments
exclude reference generation, optimizer fitting, adaptation/warmup, and JIT
compilation; the Eight Schools and GARCH modules instead time NumPyro MCMC
construction, warmup, and draws, then use that average as the projected
sampler's per-chain budget.  Every module records its policy in
`configuration.json`.  Hardware, library versions, CPU/GPU backend, and
parallel-worker count can materially affect wall-clock comparisons.  See
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for a reporting checklist.

## Data and archival figures

The GARCH data are bundled at `data/raw/garch11.json`, with source and
integrity information in [data/README.md](data/README.md).  The supplied PNG
files were moved to `figures/archival/` and renamed descriptively; they are not
claimed to be regenerable from the current repository until the validation
checks are completed.

## Before making the repository public

- Resolve the required items in [docs/VALIDATION.md](docs/VALIDATION.md).
- Record the exact PosteriorDB model, data, and reference-posterior identifiers
  and repository revision for every experiment.
- Replace [CITATION.cff.template](CITATION.cff.template) with manuscript
  authors, title, version, repository URL, and DOI when available.
- Choose and add a license.  [LICENSE-CHOICE.md](LICENSE-CHOICE.md) explains
  why this has intentionally not been guessed.
- Record the exact environment used for final figures and archive the matching
  `results/` directories.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md).  In particular, do not alter a
target density, reference generator, benchmark budget, or reported metric
without documenting the change and rerunning the affected configuration.
