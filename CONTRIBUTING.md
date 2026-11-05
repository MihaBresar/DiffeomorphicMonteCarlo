# Contributing

Contributions that improve correctness, reproducibility, documentation or implementation efficiency are welcome. Describe the problem and the resulting behavior, with the smallest reproducible example or saved configuration that demonstrates it.

## Development environment and checks

Use Python 3.12 and the pinned `requirements-cpu.txt` in a local virtual environment. Follow the [installation instructions](README.md), then run checks appropriate to the change:

```sh
python -m pip check
python -m unittest discover -s tests -v
python -m examples.basic_sampler
```

For a sampler or experiment change, also run the affected quick experiment and its `--plot-only` replay in a fresh result directory. Include test outcomes and material platform limitations in the pull request. Quick runs establish execution, not a performance ranking. Documentation-only changes need link and configuration-consistency checks; they do not require new simulations.

## Scientific changes

Changes to targets, transforms, proposals, acceptance rules, fitting, adaptation, seeds, budgets, retained samples, timing boundaries, estimators or diagnostics can change the scientific experiment. Identify them explicitly. Update protocol/configuration documentation and add a focused regression or numerical check. Preserve historical results and their original settings; never relabel them as evidence for a new protocol.

Keep shared implementation in `dcs_bench/`; `experiments/` and compatibility entry points should call that implementation. Preserve pinned third-party source and licenses. A proposed vendor update must identify its upstream revision and explain any resulting experiment changes.

For performance claims, compare the complete affected method set under the same target, environment, worker allocation, clock scope and diagnostic policy. Report preparation costs separately when they are outside production time. Retain unfavorable outcomes and disclose development-based choices. Freeze comparison settings before assessing a new independent sampling batch.

## Reporting reproduction problems

Include the source commit, operating system, Python version, exact command or launcher selections, preset, target, VI setting, SCS step rule, seed batch and worker count. Attach the relevant error and small configuration/manifest/summary files. Remove private paths, credentials or unrelated data before posting. Large chain archives can be shared separately when needed; do not paste them into an issue.

Keep environments, temporary results and profiling files out of normal commits. Intentional publication archives should include the exact source snapshot, resolved inputs and complete successful runs, with execution checks and development evidence clearly distinguished. See the [reviewer guide](docs/REVIEW_GUIDE.md) and [security policy](SECURITY.md).
