"""Profile DCS/SCS throughput using one saved fit, fixed seeds and step counts.

Run this script with the Python environment of each code version to compare
implementation overhead without changing the fitted maps or experiment design.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_experiment  # CPU/thread policy before numerical imports.
import numpy as np
from dcs_bench.engine import source_hashes, environment
from dcs_bench.samplers import run_metropolis
from dcs_bench.targets import Target
from dcs_bench.transforms import BallMap, SphereMap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('saved_run', type=Path)
    parser.add_argument('--iterations', type=int, default=100000)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.saved_run/'configuration.json').read_text())
    config.update(iterations=args.iterations, thin=10, warmup=1500)
    results = []
    for scenario, spec in enumerate(config['targets']):
        path = args.saved_run/f'scenario-{scenario}'/'setup.json'
        setup = json.loads(path.read_text())
        target = Target(spec)
        methods = [a for a in config['algorithms'] if a.startswith('DCS') or a == 'SCS']
        # Alternate order to reduce systematic ordering effects on timing.
        for repeat in range(args.repeats):
            for method in methods[::1 if repeat % 2 == 0 else -1]:
                family, cls = ('sphere', SphereMap) if method == 'SCS' else ('ball', BallMap)
                draws, record = run_metropolis(target, method, cls(**setup[family]['parameters']), config, 62001+repeat)
                results.append(dict(record, scenario=scenario, repeat=repeat,
                    iterations_per_second=record['iterations']/record['sampling_seconds'],
                    draws_sha256=hashlib.sha256(draws.tobytes()).hexdigest(),
                    fitted_setup_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'source_hashes': source_hashes(), 'environment': environment(),
        'config': config, 'results': results, 'interpretation':
        'Kernel throughput audit with frozen saved maps. Not a new mixing comparison or publication result.'}, indent=2)+'\n')
    for method in methods:
        rates = [r['iterations_per_second'] for r in results if r['algorithm'] == method]
        print(method, 'median iterations/second:', float(np.median(rates)), flush=True)


if __name__ == '__main__':
    main()
