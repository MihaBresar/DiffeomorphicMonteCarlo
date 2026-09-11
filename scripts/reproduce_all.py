"""One command for all seven experiments, independent references, and the table."""
import argparse
from pathlib import Path
import subprocess
import sys
import signal

if hasattr(signal, 'SIGBREAK'):
    signal.signal(signal.SIGBREAK, signal.default_int_handler)

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ['skew-t-200d-nuts', 'skew-t-200d-baselines', 'skew-t-100d-fixed',
               'skew-t-50d-nu-0.5', 'eight-schools', 'garch11', 'logistic-regression']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preset', choices=['quick', 'reproduce', 'paper'], default='quick')
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--ball-scale-policy', choices=['fit', 'sqrt-d'], default='fit')
    args = parser.parse_args()
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
        command = [sys.executable, str(ROOT/'run_experiment.py')]
        if args.plot_only:
            command += ['--plot-only', str(directory)]
        else:
            command += [name, '--preset', args.preset, '--output-dir', str(directory), '--reference-dir', str(reference),
                        '--workers', str(args.workers), '--ball-scale-policy', args.ball_scale_policy]
            if args.resume and (directory/'manifest.json').exists():
                command.append('--resume')
        print(' '.join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT/'scripts/collect_posterior_table.py'), str(output)], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
