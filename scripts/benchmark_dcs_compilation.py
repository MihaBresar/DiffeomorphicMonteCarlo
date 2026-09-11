"""Paired implementation timings; statistical protocol held constant."""
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'
import numpy as np
from dcs_bench.fast_dcs import run_chain
from dcs_bench.samplers import MetropolisKernel
from dcs_bench.targets import Target
from dcs_bench.transforms import BallMap
from dcs_bench.tuning import DualAveraging


def reference(mapping, config, seed):
    target = Target(config['target'])
    kernel = MetropolisKernel(target, config['algorithm'], mapping, seed)
    adapter, scale = DualAveraging(.1), .1
    for _ in range(config['warmup']):
        probability, _ = kernel.step(scale)
        adapter.update(probability)
        scale = np.exp(adapter.logbar)
    radius = np.linalg.norm(kernel.draw())
    norm_sum, tail_count = 0., 0
    traces = np.empty((config['iterations']//config['thin'], 2))
    means = np.empty_like(traces)
    started = time.perf_counter()
    for i in range(1, config['iterations']+1):
        probability, accepted = kernel.step(scale)
        adapter.update(probability)
        scale = np.exp(adapter.logbar)
        if accepted:
            radius = np.linalg.norm(kernel.draw())
        tail = radius >= config['threshold']
        norm_sum += radius
        tail_count += tail
        if i % config['thin'] == 0:
            traces[i//config['thin']-1] = tail, radius
            means[i//config['thin']-1] = tail_count/i, norm_sum/i
    seconds = time.perf_counter()-started
    return {'sampling_seconds': seconds, 'mean_all_iterations': [tail_count/config['iterations'], norm_sum/config['iterations']]}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/dcs-compilation-benchmark.json'))
    args = parser.parse_args()
    d = 100
    mapping = BallMap(d, scale=1.2, mu=np.r_[.3, -.3, np.zeros(d-2)],
                      delta=np.r_[.1, -.1, np.zeros(d-2)])
    target = {'kind': 'skew_t', 'dimension': d, 'nu': 3., 'alpha': {'0': 100., '1': -100.}}
    config = {'target': target, 'warmup': 1000, 'iterations': 25000, 'thin': 50, 'threshold': 50.}
    rows = []
    for algorithm in ('DCS-RWM', 'DCS-BW'):
        c = dict(config, algorithm=algorithm)
        run_chain(mapping.parameters(), dict(c, iterations=1000), 2)
        for repeat in range(3):
            seed = 52371+repeat
            # Alternate order so one implementation does not always follow
            # the same cache/thermal state. No independent chains overlap.
            functions = [('numpy', lambda: reference(mapping, c, seed)),
                         ('numba', lambda: run_chain(mapping.parameters(), c, seed)[1])]
            if repeat % 2:
                functions.reverse()
            measurements = {name: function() for name, function in functions}
            row = {'algorithm': algorithm, 'repeat': repeat, 'seed': seed,
                   'numpy_seconds': measurements['numpy']['sampling_seconds'],
                   'numba_seconds': measurements['numba']['sampling_seconds']}
            row['speedup'] = row['numpy_seconds']/row['numba_seconds']
            rows.append(row)
            print(row, flush=True)
    result = {'platform': platform.platform(), 'processor': platform.processor(),
              'python': platform.python_version(), 'config': config, 'mapping': mapping.parameters(),
              'notes': ['Single process, one native thread; compilation and warmup excluded.',
                        'Same normal-CDF skew-t target, map, proposals and continuous smoothed adaptation.',
                        'Both include all-iteration radial summaries and cached rejected-state radii.',
                        'Floating-point summation can produce different long adaptive trajectories; these are timing measurements, not an MSE comparison.'],
              'measurements': rows}
    destination = args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
