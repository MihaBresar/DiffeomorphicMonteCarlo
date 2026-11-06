"""One command for all seven experiments, independent references, and the table."""
import argparse
from pathlib import Path
import subprocess
import sys
import signal

if hasattr(signal, 'SIGBREAK'):
    signal.signal(signal.SIGBREAK, signal.default_int_handler)

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ['skew-t-200d-nuts', 'skew-t-200d-baselines', 'reported-zero-centre',
               'skew-t-50d-nu-0.5', 'eight-schools', 'garch11', 'logistic-regression']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preset', choices=['quick', 'reproduce', 'paper'], default='quick')
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--target', choices=['skew-t-100d', 'skew-t-500d'], default='skew-t-100d',
                        help='Target for the DCS versus original SCS comparison')
    parser.add_argument('--seed-batch', type=int, choices=range(1000), default=22,
                        metavar='0..999', help='Sampling batch for the DCS versus original SCS comparison')
    parser.add_argument('--generic-seed-batch', type=int, choices=range(1000), default=0,
                        metavar='0..999', help='Sampling batch for the other six experiments; 0 preserves their original sampling seeds, 1..7 give Seed 2..8. Fitting and reference seeds stay fixed.')
    parser.add_argument('--ball-map-policy', choices=['mobius'], default='mobius', help=argparse.SUPPRESS)
    parser.add_argument('--vi', choices=['on'], default='on', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.resume and not args.plot_only:
        parser.error('The comparison starts from scratch; choose a fresh output folder.')
    output = (args.output_root or ROOT/'results'/args.preset).resolve()
    reference = output/'logistic-reference'
    if not args.plot_only:
        command = [sys.executable, str(ROOT/'prepare_reference.py'), '--preset', args.preset,
                   '--output-dir', str(reference), '--workers', str(args.workers)]
        if args.resume and reference.exists():
            command.append('--resume')
        subprocess.run(command, cwd=ROOT, check=True)
    for name in EXPERIMENTS:
        directory = output/name
        if args.plot_only and name == 'reported-zero-centre' and not directory.exists():
            # Preserve report regeneration for earlier seven-experiment archives.
            for historical_name in ('sampler-comparison', 'skew-t-100d-fixed'):
                historical = output/historical_name
                if historical.exists():
                    directory = historical
                    break
        command = [sys.executable, str(ROOT/'run_experiment.py')]
        if args.plot_only:
            command += ['--plot-only', str(directory)]
        else:
            command += [name, '--preset', args.preset, '--output-dir', str(directory),
                        '--workers', str(args.workers)]
            if name != 'reported-zero-centre':
                command += ['--reference-dir', str(reference), '--ball-map-policy', args.ball_map_policy]
                if args.generic_seed_batch:
                    command += ['--seed-batch', str(args.generic_seed_batch)]
            else:
                command += ['--target', args.target, '--seed-batch', str(args.seed_batch)]
            if args.resume and (directory/'manifest.json').exists():
                command.append('--resume')
        print(' '.join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT/'scripts/collect_posterior_table.py'), str(output)], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
