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
from dcs_bench.cpu import worker_environment
os.environ.update(worker_environment())  # Before coordinator NumPy/JAX imports.
import argparse
from datetime import datetime, timezone
from pathlib import Path
import json
import sys
from dcs_bench.engine import ALIASES, experiment_names, load_config, run

BOUNDARY_EXPERIMENT = 'skew-t-100d-boundary'
COMPARISONS = ('sampler-comparison', 'sampler-comparison-numba')


def named_experiments():
    return sorted(set(experiment_names()) | {BOUNDARY_EXPERIMENT} | set(COMPARISONS))


def seed_batch_number(value):
    try:
        batch = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError('Seed batch must be an integer from 0 to 999')
    if not 0 <= batch <= 999:
        raise argparse.ArgumentTypeError('Seed batch must be an integer from 0 to 999')
    return batch


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('experiment', nargs='?', choices=named_experiments()+list(ALIASES))
    p.add_argument('--list', action='store_true')
    p.add_argument('--preset', choices=['quick', 'reproduce', 'paper'], default='quick')
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--reference-dir', type=Path, default=Path('results/references/logistic-paper'))
    p.add_argument('--resume', action='store_true')
    p.add_argument('--plot-only', type=Path, metavar='SAVED_RUN')
    p.add_argument('--recompute-diagnostics', action='store_true',
                   help='With --plot-only, recalculate statistics from saved chains instead of using the completed summary')
    p.add_argument('--dry-run', action='store_true')
    for name in ('runs', 'iterations', 'warmup', 'thin', 'nuts-samples', 'nuts-max-depth', 'nuts-batch-size', 'optimizer-steps', 'optimizer-samples', 'reference-samples', 'reference-batch-size', 'workers'):
        p.add_argument('--'+name, type=int)
    seeds = p.add_mutually_exclusive_group()
    seeds.add_argument('--seed', type=int)
    seeds.add_argument('--seed-batch', type=seed_batch_number,
                       help='For sampler comparisons: use sampling seed 901001 + 1000000 × batch (0–999); map-fitting seeds stay unchanged')
    p.add_argument('--target', choices=('skew-t-100d', 'skew-t-500d'),
                   help='Target for either sampler comparison')
    p.add_argument('--vi', choices=('on', 'off'), help='Enable or disable VI fitting for both comparison map families')
    p.add_argument('--time-budget', type=float)
    p.add_argument('--nuts-target-accept', type=float)
    p.add_argument('--ball-map-policy', choices=['mobius', 'vanilla'],
                   help='Fit R, centre and Möbius parameters by VI, or use the untuned map R=1, centre=0, Möbius=0')
    p.add_argument('--ball-scale-policy', help=argparse.SUPPRESS)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.seed_batch is not None and (args.plot_only or args.experiment not in ('skew-t-100d-fixed', BOUNDARY_EXPERIMENT) + COMPARISONS):
        p.error('--seed-batch is available only when running a sampler comparison')
    if (args.target is not None or args.vi is not None) and (args.plot_only or args.experiment not in COMPARISONS):
        p.error('--target and --vi apply only to the new sampler comparisons')
    if args.recompute_diagnostics and not args.plot_only:
        p.error('--recompute-diagnostics requires --plot-only SAVED_RUN')
    if args.list:
        print('\n'.join(named_experiments()))
        return 0
    if args.plot_only:
        summary_path = args.plot_only/'summary.json'
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding='utf-8'))
            if summary.get('config', {}).get('protocol') in ('original-dcs-author-scs-v1', 'timed-dcs-author-scs-v2', 'timed-matched-dcs-scs-v3', 'timed-boundary-dcs-scs-v1', 'timed-boundary-dcs-scs-v2', 'timed-boundary-dcs-scs-v3', 'timed-sampler-comparison-v1', 'timed-numba-comparison-v1', 'timed-sampler-comparison-v2', 'timed-numba-comparison-v2', 'timed-sampler-comparison-v3', 'timed-numba-comparison-v3', 'timed-sampler-comparison-v4', 'timed-numba-comparison-v4', 'timed-sampler-comparison-v5', 'timed-numba-comparison-v5'):
                from scripts.scs_100d_figures import render_report
                render_report(args.plot_only, summary)
                print(args.plot_only.resolve())
                return 0
            if 'scenarios' in summary and not args.recompute_diagnostics:
                from dcs_bench.plotting import render_saved_summary
                print('Rendering figures from saved statistics (no diagnostic recomputation).', flush=True)
                print(render_saved_summary(args.plot_only))
                return 0
        from dcs_bench.plotting import rebuild
        print(rebuild(args.plot_only))
        return 0
    if args.experiment is None:
        p.error('choose an experiment or --plot-only SAVED_RUN')
    if args.experiment in COMPARISONS:
        if args.resume:
            p.error('Sampler comparisons start from scratch; choose a fresh output folder.')
        unused = [name for name in ('iterations', 'ball_map_policy', 'ball_scale_policy', 'nuts_samples',
                    'nuts_max_depth', 'nuts_batch_size', 'nuts_target_accept', 'reference_samples',
                    'reference_batch_size') if getattr(args, name) is not None]
        if unused:
            p.error('These options are not used by sampler comparisons: '+', '.join(unused))
        from scripts.reproduce_comparison import main as comparison_main
        command = ['--comparison', 'numba' if args.experiment.endswith('-numba') else 'jax',
                   '--target', args.target or 'skew-t-100d', '--vi', args.vi or 'on',
                   '--preset', args.preset, '--workers', str(args.workers or 1)]
        if args.output_dir:
            command += ['--output-dir', str(args.output_dir)]
        if args.dry_run:
            command.append('--dry-run')
        for name in ('runs', 'time_budget', 'warmup', 'thin', 'optimizer_steps', 'optimizer_samples', 'seed', 'seed_batch'):
            if getattr(args, name) is not None:
                command += ['--'+name.replace('_', '-'), str(getattr(args, name))]
        return comparison_main(command)
    if args.experiment in ('skew-t-100d-fixed', BOUNDARY_EXPERIMENT):
        if args.resume:
            p.error('The single SCS comparison starts from scratch; choose a fresh output folder.')
        if args.experiment == BOUNDARY_EXPERIMENT:
            from scripts.reproduce_boundary_100d import main as single_main
        else:
            from scripts.reproduce_scs_100d import main as single_main
        command = ['--preset', args.preset, '--workers', str(args.workers or 1)]
        if args.output_dir:
            command += ['--output-dir', str(args.output_dir)]
        if args.dry_run:
            command.append('--dry-run')
        for name in ('runs', 'time_budget', 'warmup', 'thin', 'optimizer_steps', 'optimizer_samples', 'seed', 'seed_batch'):
            value = getattr(args, name)
            if value is not None:
                command += ['--'+name.replace('_', '-'), str(value)]
        if args.ball_map_policy == 'vanilla' or args.ball_scale_policy is not None or args.iterations is not None:
            p.error('This comparison uses matched map fitting and equal sampling time; map variants and fixed iteration counts are not supported.')
        unused = [name for name in ('nuts_samples', 'nuts_max_depth', 'nuts_batch_size', 'nuts_target_accept',
                                     'reference_samples', 'reference_batch_size') if getattr(args, name) is not None]
        if unused:
            p.error('These options are not used by the single SCS comparison: '+', '.join(unused))
        return single_main(command)
    if args.experiment in ALIASES:
        print(f'Legacy name {args.experiment} now resolves to {ALIASES[args.experiment]} (d=200).', file=sys.stderr)
    overrides = {k: v for k, v in vars(args).items() if k not in ('experiment', 'list', 'preset', 'output_dir', 'reference_dir', 'resume', 'plot_only', 'recompute_diagnostics', 'dry_run', 'seed_batch', 'target', 'vi')}
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
