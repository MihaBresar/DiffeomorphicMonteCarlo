#!/usr/bin/env python3
"""Run, resume, or replot a named benchmark with a recorded configuration."""
import os
import signal
import sys
if os.name == 'nt':
    # Greek parameter names must also work in redirected Windows console logs.
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, 'reconfigure'):
            _stream.reconfigure(encoding='utf-8', errors='backslashreplace')
if hasattr(signal, 'SIGBREAK'):
    signal.signal(signal.SIGBREAK, signal.default_int_handler)
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
    p.add_argument('--ball-map-policy', choices=['mobius', 'vanilla'],
                   help='Fit R, centre and Möbius parameters by VI, or use the untuned map R=1, centre=0, Möbius=0')
    p.add_argument('--ball-scale-policy', help=argparse.SUPPRESS)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.list:
        print('\n'.join(experiment_names()))
        return 0
    if args.plot_only:
        summary_path = args.plot_only/'summary.json'
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding='utf-8'))
            if summary.get('config', {}).get('protocol') == 'original-dcs-author-scs-v1':
                from scripts.scs_100d_figures import render_report
                render_report(args.plot_only, summary)
                print(args.plot_only.resolve())
                return 0
        from dcs_bench.plotting import rebuild
        print(rebuild(args.plot_only))
        return 0
    if args.experiment is None:
        p.error('choose an experiment or --plot-only SAVED_RUN')
    if args.experiment == 'skew-t-100d-fixed':
        if args.resume:
            p.error('The single SCS comparison starts from scratch; choose a fresh output folder.')
        from scripts.reproduce_scs_100d import main as single_main
        command = ['--preset', args.preset, '--workers', str(args.workers or 1)]
        if args.output_dir:
            command += ['--output-dir', str(args.output_dir)]
        if args.dry_run:
            command.append('--dry-run')
        for name in ('runs', 'iterations', 'warmup', 'thin', 'optimizer_steps', 'optimizer_samples', 'seed'):
            value = getattr(args, name)
            if value is not None:
                command += ['--'+name.replace('_', '-'), str(value)]
        if args.ball_map_policy == 'vanilla' or args.ball_scale_policy is not None or args.time_budget is not None:
            p.error('This comparison uses the original fitted DCS map and fixed iterations; map variants and time budgets are not supported.')
        unused = [name for name in ('nuts_samples', 'nuts_max_depth', 'nuts_batch_size', 'nuts_target_accept',
                                     'reference_samples', 'reference_batch_size') if getattr(args, name) is not None]
        if unused:
            p.error('These options are not used by the single SCS comparison: '+', '.join(unused))
        return single_main(command)
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
