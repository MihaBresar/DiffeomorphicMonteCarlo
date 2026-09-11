#!/usr/bin/env python3
"""Run, resume, or replot a named benchmark with a recorded configuration."""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_name, '1')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('JAX_ENABLE_X64', 'true')
import argparse
from datetime import datetime, timezone
from pathlib import Path
import json
import sys
from dcs_bench.engine import ALIASES, experiment_names, load_config, run


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('experiment', nargs='?', choices=experiment_names()+list(ALIASES))
    p.add_argument('--list', action='store_true')
    p.add_argument('--preset', choices=['quick', 'reproduce', 'paper'], default='quick')
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--reference-dir', type=Path, default=Path('results/references/logistic-paper'))
    p.add_argument('--resume', action='store_true')
    p.add_argument('--plot-only', type=Path, metavar='SAVED_RUN')
    p.add_argument('--dry-run', action='store_true')
    for name in ('runs', 'iterations', 'warmup', 'thin', 'nuts-samples', 'nuts-max-depth', 'nuts-batch-size', 'optimizer-steps', 'optimizer-samples', 'reference-samples', 'reference-batch-size', 'seed', 'workers'):
        p.add_argument('--'+name, type=int)
    p.add_argument('--time-budget', type=float)
    p.add_argument('--nuts-target-accept', type=float)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.list:
        print('\n'.join(experiment_names()))
        return 0
    if args.plot_only:
        from dcs_bench.plotting import rebuild
        print(rebuild(args.plot_only))
        return 0
    if args.experiment is None:
        p.error('choose an experiment or --plot-only SAVED_RUN')
    if args.experiment in ALIASES:
        print(f'Legacy name {args.experiment} now resolves to {ALIASES[args.experiment]} (d=200).', file=sys.stderr)
    overrides = {k: v for k, v in vars(args).items() if k not in ('experiment', 'list', 'preset', 'output_dir', 'reference_dir', 'resume', 'plot_only', 'dry_run')}
    config = load_config(args.experiment, args.preset, overrides)
    output = args.output_dir or Path('results')/config['experiment']/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    if args.dry_run:
        print(json.dumps({'output_dir': str(output), 'configuration': config}, indent=2))
        return 0
    result = run(config, output, resume=args.resume, reference_dir=args.reference_dir)
    print(f'Results: {result.resolve()}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
