#!/usr/bin/env python3
"""Generate logistic reference chains with independent seeds before benchmarking."""
# Import the runner first to establish the documented CPU/thread settings.
import run_experiment
import argparse
from pathlib import Path
import hashlib
import json
import numpy as np
from dcs_bench.engine import load_config, write_json, environment, read_chain, source_hashes
from dcs_bench.targets import Target
from dcs_bench.references import mcmc_diagnostics
from dcs_bench.cpu import execution_plan
from dcs_bench.parallel import launch_chains, compilation_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preset', choices=['quick', 'reproduce', 'paper'], default='quick')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--draws', type=int)
    parser.add_argument('--warmup', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    root = (args.output_dir or Path(f'results/references/logistic-{args.preset}')).resolve()
    config = load_config('logistic-regression', args.preset, {'workers': args.workers})
    config.update(warmup=args.warmup or {'paper': 4000, 'reproduce': 1000, 'quick': 100}[args.preset],
                  nuts_samples=args.draws or {'paper': 100000, 'reproduce': 10000, 'quick': 128}[args.preset],
                  nuts_target_accept=.95, nuts_max_depth=12, thin=1, reference_diagnostics=True)
    root.mkdir(parents=True, exist_ok=True)
    plan = execution_plan(config['workers'], 4)
    context = environment()
    # Git metadata can change without altering executable sources; execution
    # conditions must remain identical across resumed reference chains.
    context_key = {key: value for key, value in context.items() if not key.startswith('git_')}
    failures = []
    for scenario, spec in enumerate(config['targets']):
        target = Target(spec)
        seed = 918274  # Separate stream from all production and optimization seeds.
        hashes = source_hashes()
        scientific_hashes = {k: v for k, v in hashes.items() if k != 'dcs_bench/plotting.py'}
        fingerprint = hashlib.sha256(json.dumps([spec, config, seed, scientific_hashes, context_key, plan], sort_keys=True).encode()).hexdigest()
        checkpoint_dir = root/f'nu-{target.nu:g}-chains'
        records, draws = [], []
        pending = []
        for i in range(4):
            checkpoint = checkpoint_dir/f'{i}.npz'
            if checkpoint.exists():
                if not args.resume:
                    raise FileExistsError(f'{checkpoint} exists; pass --resume')
                read_chain(checkpoint, fingerprint)
            else:
                pending.append({'index': i, 'seed': seed+i, 'path': str(checkpoint)})
        print(f'Independent logistic reference ν={target.nu:g}: {len(pending)} pending chains, up to {plan["max_concurrent_chains"]} workers', flush=True)
        launch_chains({'target': spec, 'config': config, 'algorithm': 'NUTS', 'reference': True,
                       'fingerprint': fingerprint, 'scenario': scenario},
                      pending, checkpoint_dir/'execution', plan)
        for i in range(4):
            x, record = read_chain(checkpoint_dir/f'{i}.npz', fingerprint)
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
            'diagnostics': diag, 'quality_passed': quality, 'environment': context, 'source_hashes': hashes,
            'execution_plan': plan, 'compilation': compilation_summary(checkpoint_dir/'execution'),
            'quality_thresholds': {'max_rhat': 1.01, 'min_bulk_ess': 2000, 'min_tail_ess': 1000,
                                   'max_divergences': 0, 'max_depth_hits': 0}})
        if args.preset == 'paper' and not quality:
            failures.append(target.nu)
    if failures:
        raise RuntimeError(f'Reference diagnostic thresholds failed for ν={failures}. Inspect JSON diagnostics and generate longer/better-tuned references in a new directory; do not use these as truth.')
    print(root.resolve())


if __name__ == '__main__':
    main()
