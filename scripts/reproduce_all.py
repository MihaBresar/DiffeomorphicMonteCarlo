"""One command for all seven experiments, independent references, and the table."""
import argparse
from pathlib import Path
import subprocess
import sys
import signal

if hasattr(signal, 'SIGBREAK'):
    signal.signal(signal.SIGBREAK, signal.default_int_handler)

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ['skew-t-200d-nuts', 'skew-t-200d-baselines', 'sampler-comparison',
               'skew-t-50d-nu-0.5', 'eight-schools', 'garch11', 'logistic-regression']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preset', choices=['quick', 'reproduce', 'paper'], default='quick')
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--ball-map-policy', choices=['mobius', 'vanilla'], default='mobius')
    parser.add_argument('--vi', choices=['on', 'off'], default='on',
                        help='VI on/off for both map families in the five-method JAX comparison')
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
        if args.plot_only and name == 'sampler-comparison' and not directory.exists():
            # Preserve report regeneration for an earlier seven-experiment archive.
            historical = output/'skew-t-100d-fixed'
            if historical.exists():
                directory = historical
        command = [sys.executable, str(ROOT/'run_experiment.py')]
        if args.plot_only:
            command += ['--plot-only', str(directory)]
        else:
            command += [name, '--preset', args.preset, '--output-dir', str(directory), '--reference-dir', str(reference),
                        '--workers', str(args.workers)]
            if name != 'sampler-comparison':
                command += ['--ball-map-policy', args.ball_map_policy]
            else:
                command += ['--target', 'skew-t-100d', '--vi', args.vi]
            if args.resume and (directory/'manifest.json').exists():
                if name == 'sampler-comparison':
                    import json
                    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
                    if manifest.get('protocol') == 'timed-sampler-comparison-v8' and manifest.get('status') == 'complete' and manifest.get('config', {}).get('vi') == args.vi:
                        continue
                    raise ValueError('The sampler comparison requires a fresh output folder after an interrupted run or a VI change.')
                command.append('--resume')
        print(' '.join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT/'scripts/collect_posterior_table.py'), str(output)], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
