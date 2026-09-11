# Contributing

Use Python 3.12 and `requirements-cpu.txt`. Run `python -m unittest discover -s tests -v`, the affected quick experiment, and plot-only replay before proposing changes.

Target formulas, coordinate transforms, kernel transitions, optimization, adaptation, seeds, budgets and metrics are scientific changes. Update the corresponding JSON configuration and documentation and produce new results; do not overwrite the archival figures. Shared implementation belongs in `dcs_bench/`, while `experiments/` is only a compatibility layer.

Keep results, environments and local profiling files out of normal commits. A publication archive should intentionally include the exact source snapshot, inputs and complete successful run directories. Never mark short smoke outputs as paper results.
