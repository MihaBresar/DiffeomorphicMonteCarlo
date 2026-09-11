#!/usr/bin/env python3
"""Generate logistic reference chains with independent seeds before benchmarking."""
# Import the runner first to establish the documented CPU/thread settings.
import run_experiment
import argparse
from pathlib import Path
import hashlib
import json
import numpy as np
from dcs_bench.engine import load_config, write_json, environment, save_chain, read_chain, source_hashes
from dcs_bench.targets import Target
from dcs_bench.samplers import NumPyroRunner
from dcs_bench.references import mcmc_diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preset', choices=['quick', 'reproduce', 'paper'], default='quick')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--draws', type=int)
    parser.add_argument('--warmup', type=int)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    root = args.output_dir or Path(f'results/references/logistic-{args.preset}')
    config = load_config('logistic-regression', args.preset)
    config.update(warmup=args.warmup or {'paper': 4000, 'reproduce': 1000, 'quick': 100}[args.preset],
                  nuts_samples=args.draws or {'paper': 100000, 'reproduce': 10000, 'quick': 128}[args.preset],
                  nuts_target_accept=.95, nuts_max_depth=12, thin=1)
    root.mkdir(parents=True, exist_ok=True)
    failures = []
    for spec in config['targets']:
        target = Target(spec)
        seed = 918274  # Separate stream from all production and optimization seeds.
        hashes = source_hashes()
        fingerprint = hashlib.sha256(json.dumps([spec, config, seed, hashes], sort_keys=True).encode()).hexdigest()
        checkpoint_dir = root/f'nu-{target.nu:g}-chains'
        records, draws = [], []
        runner = None
        for i in range(4):
            checkpoint = checkpoint_dir/f'{i}.npz'
            if checkpoint.exists():
                if not args.resume:
                    raise FileExistsError(f'{checkpoint} exists; pass --resume')
                x, record = read_chain(checkpoint, fingerprint)
            else:
                print(f'Independent logistic reference ν={target.nu:g}, chain {i+1}/4', flush=True)
                if runner is None:
                    runner = NumPyroRunner(target, config)
                x, record = runner.run(seed+i)
                record['fingerprint'] = fingerprint
                save_chain(checkpoint, x, record)
            draws.append(x)
            records.append(record)
        draws = np.stack(draws)
        diag = mcmc_diagnostics(draws, target.names)
        quality = (args.preset != 'quick'
                   and all(v is not None and v <= 1.01 for v in diag['rhat'].values())
                   and all(v is not None and v >= 2000 for v in diag['ess_bulk'].values())
                   and all(v is not None and v >= 1000 for v in diag['ess_tail'].values())
                   and sum(r['divergences'] for r in records) == 0
                   and sum(r['max_depth_hits'] for r in records) == 0)
        path = root/f'nu-{target.nu:g}.npz'
        np.savez_compressed(path, draws=draws)
        write_json(path.with_suffix('.json'), {'source': 'Independent NumPyro NUTS reference',
            'exact': False, 'target': spec, 'config': config, 'records': records,
            'diagnostics': diag, 'quality_passed': quality, 'environment': environment(), 'source_hashes': hashes,
            'quality_thresholds': {'max_rhat': 1.01, 'min_bulk_ess': 2000, 'min_tail_ess': 1000,
                                   'max_divergences': 0, 'max_depth_hits': 0}})
        if args.preset == 'paper' and not quality:
            failures.append(target.nu)
    if failures:
        raise RuntimeError(f'Reference diagnostic thresholds failed for ν={failures}. Inspect JSON diagnostics and generate longer/better-tuned references in a new directory; do not use these as truth.')
    print(root.resolve())


if __name__ == '__main__':
    main()
