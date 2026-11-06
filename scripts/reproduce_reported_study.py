"""Run the frozen zero-centre study against the original author SCS baseline."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dcs_bench.reported_study import PRESETS, TARGETS, configuration, run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog='Maps and steps are bundled and frozen. '
        'JAX reproduces the reported implementation; Numba is a separately labelled extension. '
        'Timed counts and results depend on hardware. No manual fitting or step-size choice is required.')
    parser.add_argument('--preset', choices=PRESETS, default='quick')
    parser.add_argument('--comparison', choices=('jax', 'numba'), default='jax')
    parser.add_argument('--target', choices=TARGETS, default='skew-t-100d')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--seed-batch', type=int, default=22,
                        help='22 is a fresh batch; 20/21 reuse the reported random-stream inputs')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    config = configuration(args.preset, args.workers, comparison=args.comparison,
                           target=args.target, seed_batch=args.seed_batch)
    output = args.output_dir or ROOT / 'results' / config['experiment'] / (
        args.target + '-batch' + str(args.seed_batch) + '-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ'))
    if args.dry_run:
        print(json.dumps(dict(output_dir=str(output), configuration=config), indent=2))
        return 0
    print(f'Results: {run(config, output)}', flush=True)
    return 0


if __name__ == '__main__':
    if os.name == 'nt':
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, 'reconfigure'):
                stream.reconfigure(encoding='utf-8', errors='backslashreplace')
    raise SystemExit(main())
