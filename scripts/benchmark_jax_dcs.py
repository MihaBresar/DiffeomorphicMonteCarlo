"""Measure matched JAX drivers with fixed maps, initial states and proposal steps.

Use --baseline-root with a checkout of 289f3bf to measure the earlier driver.
This is a fixed-work implementation benchmark, not an MSE comparison. Proposal
RNG implementations may differ while preserving the same transition law.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dcs_bench.cpu import CPU_THREAD_POLICY, THREAD_VARIABLES, cpu_info, worker_environment
os.environ.update(worker_environment())

import numpy as np
from dcs_bench import timed_radial as current

METHODS = ('DCS-RWM', 'DCS-BW', 'SCS')
SEED = 901001


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_baseline(root):
    """Import only the old driver after verifying its shared dependencies."""
    root = root.resolve()
    driver = root / 'dcs_bench/timed_radial.py'
    if not driver.is_file():
        raise ValueError('--baseline-root must contain dcs_bench/timed_radial.py')
    dependencies = ['dcs_bench/scs_paper.py', 'dcs_bench/targets.py',
                    'dcs_bench/transforms.py', 'third_party/scs_paper/manifest.json']
    vendor = json.loads((ROOT / 'third_party/scs_paper/manifest.json').read_text())
    dependencies += ['third_party/scs_paper/' + name for name in vendor['files']]
    hashes = {}
    for name in dependencies:
        local, old = ROOT / name, root / name
        if not old.is_file() or sha256(local) != sha256(old):
            raise ValueError('Baseline shared dependency differs or is missing: ' + name)
        hashes[name] = sha256(local)
    name = 'dcs_bench._jax_benchmark_baseline'
    spec = importlib.util.spec_from_file_location(name, driver)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if module.BATCH_SIZE != current.BATCH_SIZE or module.PREFIX_STRIDE != current.PREFIX_STRIDE:
        raise ValueError('Baseline and current timing batches must match')
    return module, hashes


def reset(backend, snapshot):
    # JAX arrays are immutable; assignment restores the exact saved state and
    # key without consuming new draws. Both scale representations must agree.
    backend.state = snapshot['state']
    backend.key = snapshot['key']
    backend.scale = snapshot['scale']
    backend._scale = snapshot['scale_array']


def measure(backend, snapshot, batches, thin, threshold):
    """Use run_timed's batch driver and compact recording for fixed work."""
    reset(backend, snapshot)
    capacity = 1024
    prefix_buffer = np.empty((capacity * 3, 3), dtype=float)
    norm_buffer = np.empty(capacity * current.BATCH_SIZE // thin, dtype=float)
    totals = np.zeros(3, dtype=float)
    started, cpu_started = time.perf_counter(), time.process_time()
    for batch_index in range(batches):
        prefixes, norms = backend.batch()  # Synchronizes all completed JAX work.
        if batch_index == capacity:
            capacity *= 2
            larger = np.empty((capacity * 3, 3), dtype=float)
            larger[:batch_index * 3] = prefix_buffer
            prefix_buffer = larger
            larger = np.empty(capacity * current.BATCH_SIZE // thin, dtype=float)
            larger[:batch_index * current.BATCH_SIZE // thin] = norm_buffer
            norm_buffer = larger
        prefix_buffer[batch_index * 3:(batch_index + 1) * 3] = prefixes + totals
        totals = prefix_buffer[(batch_index + 1) * 3 - 1].copy()
        start = batch_index * current.BATCH_SIZE // thin
        norm_buffer[start:start + current.BATCH_SIZE // thin] = norms
        elapsed = time.perf_counter() - started
    cpu_seconds = time.process_time() - cpu_started
    prefixes = prefix_buffer[:batches * 3]
    norms = norm_buffer[:batches * current.BATCH_SIZE // thin]
    if not np.all(np.isfinite(prefixes)) or not np.all(np.isfinite(norms)):
        raise FloatingPointError('Nonfinite benchmark states/statistics')
    _, count, burn, used, totals, sums = current._postburn(prefixes, norms, thin, threshold)
    return dict(iterations=count, seconds=elapsed, cpu_seconds=cpu_seconds,
        iterations_per_second=count / elapsed, accepted=int(totals[2]),
        acceptance=float(totals[2] / count), discarded_iterations=burn,
        estimation_iterations=used, tail_count=int(sums[0]), norm_sum=float(sums[1]),
        mean_post_burnin=(sums[:2] / used).tolist(), final_scale=backend.scale)


def run(saved_summary, baseline_root=None, repeats=5, batches=150):
    if repeats < 1 or batches < 1:
        raise ValueError('repeats and batches must be positive')
    payload = json.loads(saved_summary.read_text())
    saved = payload['runs']['matched_v3'] if 'runs' in payload else payload
    config, maps = dict(saved['config']), saved['maps']
    config['seed'], config['workers'] = SEED, 1
    if current.PREFIX_STRIDE % config['thin']:
        raise ValueError('Saved thinning must divide the prefix stride')
    dependencies = {}
    baseline = None
    if baseline_root is not None:
        baseline, dependencies = load_baseline(baseline_root)
    import jax
    modules = [('before', baseline), ('after', current)] if baseline else [('current', current)]
    drivers, snapshots = {}, {}
    for method in METHODS:
        mapping = maps['sphere' if method == 'SCS' else 'ball']
        settings = {**config, 'algorithm': method}
        source = baseline if baseline else current
        reference = source._backend(mapping, settings, SEED)
        snapshot = dict(state=reference.state, key=reference.key, scale=reference.scale,
                        scale_array=reference._scale)
        jax.block_until_ready((snapshot['state'], snapshot['key'], snapshot['scale_array']))
        snapshots[method] = snapshot
        for variant, module in modules:
            backend = reference if module is source else module._backend(mapping, settings, SEED)
            label = variant + '/' + method
            drivers[label] = backend
            reset(backend, snapshot)
            backend.batch()  # Compile/dispatch/cache priming is wholly untimed.
            jax.block_until_ready(backend.state)
            reset(backend, snapshot)
    labels = list(drivers)
    records = []
    for repeat in range(repeats):
        shift = repeat % len(labels)
        order = labels[shift:] + labels[:shift]
        if repeat % 2:
            order.reverse()
        for position, label in enumerate(order):
            method = label.split('/', 1)[1]
            row = measure(drivers[label], snapshots[method], batches, config['thin'],
                          float(config.get('threshold', 50.)))
            row.update(label=label, algorithm=method, repeat=repeat,
                       order_position=position)
            records.append(row)
            print(f'{label}, repeat {repeat + 1}: {row["iterations_per_second"]:,.2f} iterations/s', flush=True)
    cases = {}
    for label in labels:
        rows = [row for row in records if row['label'] == label]
        median_seconds = float(np.median([row['seconds'] for row in rows]))
        cases[label] = dict(median_seconds=median_seconds,
            median_iterations_per_second=float(np.median([row['iterations_per_second'] for row in rows])),
            min_seconds=min(row['seconds'] for row in rows),
            max_seconds=max(row['seconds'] for row in rows))
    speedups = {}
    if baseline:
        speedups = {method: cases['before/' + method]['median_seconds'] /
                    cases['after/' + method]['median_seconds'] for method in METHODS}
    return dict(protocol='jax-fixed-work-implementation-benchmark-v1',
        current_implementation_revision=current.IMPLEMENTATION_REVISION,
        config=config, maps=maps, repeats=repeats, batches=batches,
        iterations_per_repeat=current.BATCH_SIZE * batches, seed=SEED,
        saved_summary_sha256=sha256(saved_summary),
        baseline_directory_name=baseline_root.resolve().name if baseline_root else None,
        source_sha256={'current_driver': sha256(Path(current.__file__)),
                      'baseline_driver': sha256(Path(baseline.__file__)) if baseline else None,
                      'benchmark': sha256(Path(__file__))},
        verified_shared_dependency_sha256=dependencies,
        environment=dict(python=sys.version, platform=platform.platform(),
            processor=platform.processor(), cpu_allocation=cpu_info(),
            cpu_thread_policy=CPU_THREAD_POLICY,
            jax_devices=[str(device) for device in jax.devices()],
            packages={name: importlib.metadata.version(name) for name in ('jax', 'jaxlib', 'numpy', 'scipy')},
            thread_settings={name: os.environ.get(name) for name in
                (*THREAD_VARIABLES, 'JAX_PLATFORMS', 'JAX_ENABLE_X64', 'XLA_FLAGS')}),
        frozen_starts={method: dict(state=np.asarray(s['state']).tolist(),
                                   key_data=np.asarray(jax.random.key_data(s['key'])).tolist(),
                                   scale=s['scale']) for method, s in snapshots.items()},
        cases=cases, speedup_after_over_before=speedups, records=records,
        notes=['Single process; cases run sequentially in rotated/reversed order.',
               'Maps and baseline warmup endpoints/steps are shared by before and after variants.',
               'Compilation, fitting, warmup and one priming batch are excluded from every clock.',
               'All methods include complete 300-state scans, physical projection, radial reduction, '
               'synchronization and the same host recording work.',
               'The SCS transition source is unchanged; before/after SCS rows check driver stability.',
               'This benchmark measures fixed-work speed, not MSE. Changed random-number generation '
               'can change seeded paths while preserving the transition law.',
               'Wall times depend on hardware and competing load; keep the computer otherwise idle.'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--saved-summary', type=Path,
        required=True, help='Saved comparison summary.json containing fitted maps and calibration records')
    parser.add_argument('--baseline-root', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/jax-dcs-benchmark.json')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--batches', type=int, default=150)
    args = parser.parse_args(argv)
    result = run(args.saved_summary, args.baseline_root, args.repeats, args.batches)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f'Results: {args.output.resolve()}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
