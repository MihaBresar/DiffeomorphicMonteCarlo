"""Fixed-work implementation benchmark; no production MSE ranking.

Run only when other computational checks are idle. All cases use the saved
full-budget 100D ball fit, the same method-specific JAX pretuning, and 900,000
production transitions. Native production uses a different PRNG stream.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dcs_bench.cpu import cpu_info, pin_current_process, worker_environment
os.environ.update(worker_environment())
hardware = cpu_info()
allocation = pin_current_process(hardware['cpu_ids'][0])

import numpy as np
from dcs_bench import timed_radial
from dcs_bench.numba_radial import NumbaRadial
from dcs_bench.engine import environment


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measure(backend, batches, config):
    # Same compact buffering as the common timer, including every state and
    # identical300-transition stopping/recording granularity.
    capacity = max(1024, batches)
    prefix_buffer = np.empty((capacity * 3, 3), dtype=float)
    norm_buffer = np.empty(capacity * 300 // config['thin'], dtype=float)
    totals = np.zeros(3, dtype=float)
    started, cpu_started = time.perf_counter(), time.process_time()
    for index in range(batches):
        prefixes, norms = backend.batch()
        prefix_buffer[index * 3:(index + 1) * 3] = prefixes + totals
        totals = prefix_buffer[(index + 1) * 3 - 1].copy()
        start = index * 300 // config['thin']
        norm_buffer[start:start + 300 // config['thin']] = norms
        elapsed = time.perf_counter() - started
    cpu_seconds = time.process_time() - cpu_started
    prefixes, norms = prefix_buffer[:batches * 3], norm_buffer[:batches * 300 // config['thin']]
    if not np.all(np.isfinite(prefixes)) or not np.all(np.isfinite(norms)):
        raise FloatingPointError('Nonfinite benchmark statistics')
    _, count, burn, used, totals, sums = timed_radial._postburn(prefixes, norms, config['thin'], config['threshold'])
    return dict(iterations=count, seconds=elapsed, cpu_seconds=cpu_seconds,
                iterations_per_second=count / elapsed, acceptance=float(totals[2] / count),
                first_third_removed=burn, estimation_iterations=used,
                frozen_scale=backend.scale,
                interpretation='Throughput check only; no estimates or MSE ranking are reported.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/numba-throughput.json')
    parser.add_argument('--record', type=Path, default=ROOT/'validation/radial_numba_throughput.json',
                        help='Saved map/config from a fixed-work benchmark; no fitting is done')
    parser.add_argument('--batches', type=int, default=3000)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.batches < 1 or args.repeats < 1:
        parser.error('Positive batch and repeat counts required')
    summary = args.record.resolve()
    saved = json.loads(summary.read_text())
    config = dict(saved['config'], warmup=1500, thin=50, threshold=50., workers=1,
                  seed=901001, production_backend='jax')
    config['step_tuning_policies'] = {'DCS-BW-reflect': timed_radial.BALL_TUNING}
    mapping = saved['map']
    paths = [ROOT / 'dcs_bench' / name for name in
             ('timed_radial.py', 'numba_radial.py', 'fast_dcs.py', 'ball_boundary.py',
              'ball_tuning.py', 'targets.py', 'transforms.py')]
    hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}
    cases = [('DCS-BW', 'JAX'), ('DCS-BW', 'Numba'),
             ('DCS-BW-reflect', 'JAX'), ('DCS-BW-reflect', 'Numba')]
    rows, starts = [], {}
    started_utc = datetime.now(timezone.utc).isoformat()
    for repeat in range(args.repeats):
        order = cases[repeat % len(cases):] + cases[:repeat % len(cases)]
        for position, (algorithm, implementation) in enumerate(order):
            local = dict(config, algorithm=algorithm)
            # Construct and compile outside measurement, restoring identical
            # calibration state and frozen scale at the start of every case.
            calibration = timed_radial._Matched(mapping, local, 901001)
            state = np.asarray(calibration.state)
            initial = {'state_sha256': hashlib.sha256(state.tobytes()).hexdigest(),
                       'scale': calibration.scale}
            if algorithm in starts:
                assert initial == starts[algorithm]
            else:
                starts[algorithm] = initial
            backend = calibration if implementation == 'JAX' else NumbaRadial(mapping, local, 901001, calibration)
            np.testing.assert_array_equal(backend.state, state)
            row = measure(backend, args.batches, local)
            row.update(method=algorithm, implementation=implementation, repeat=repeat,
                       order_position=position, compile_seconds=backend.compile_seconds,
                       warmup_seconds=backend.warmup_seconds)
            rows.append(row)
            print(f'{repeat + 1}: {algorithm} {implementation} {row["seconds"]:.3f}s', flush=True)
    assert all(digest(ROOT / name) == value for name, value in hashes.items())
    summaries = []
    for method in ('DCS-BW', 'DCS-BW-reflect'):
        times = {implementation: [row['seconds'] for row in rows
                 if row['method'] == method and row['implementation'] == implementation]
                 for implementation in ('JAX', 'Numba')}
        summaries.append(dict(method=method, jax_median_seconds=float(np.median(times['JAX'])),
            numba_median_seconds=float(np.median(times['Numba'])),
            median_paired_speedup=float(np.median(np.asarray(times['JAX']) / np.asarray(times['Numba'])))))
    payload = dict(started_utc=started_utc, completed_utc=datetime.now(timezone.utc).isoformat(),
        saved_fit=str(summary), saved_fit_sha256=digest(summary),
        map=mapping, config=config, seed=901001, repeats=args.repeats, batches=args.batches,
        iterations_per_case=args.batches * 300, initial_states=starts, measurements=rows,
        summary=summaries, environment=environment(), hardware=hardware, allocation=allocation,
        source_sha256=hashes, source_files_unchanged=True,
        interpretation='Fixed-work implementation throughput on this computer. Calibration/map/step/state '
        'match within each method. Different JAX/PCG64 production randomness and floating-point paths '
        'are intentional; this is not a paired trajectory or MSE comparison. All fitting, compilation '
        'and calibration are outside measured sampling time. No universal speedup is claimed.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
