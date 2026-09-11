"""Compare bounded adapter with one unmodified-author scan on the saved map."""
from pathlib import Path
import sys
import os
import json
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dcs_bench.cpu import worker_environment
os.environ.update(worker_environment())
import numpy as np
from dcs_bench import scs_paper
from dcs_bench.targets import Target

core = scs_paper._author()
import jax
import jax.numpy as jnp

import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--saved-summary', type=Path, default=ROOT/'validation/scs_100d_comparison.json')
parser.add_argument('--output', type=Path, default=ROOT/'results/scs-adapter-timing.json')
args = parser.parse_args()
mapping_path = args.saved_summary
mapping = json.loads(mapping_path.read_text(encoding='utf-8'))['maps']['sphere']
config = dict(dimension=100, nu=3., alpha={0: 100., 1: -100.},
              iterations=50000, warmup=100, thin=50)
target = Target(scs_paper.target_spec(config))
model = core.SCP(100, 1.1)
params = model.inverse_transform_params(jnp.asarray(mapping['observer']),
    jnp.asarray(mapping['mu']), jnp.asarray(mapping['scale']))
density = model.transform_target(lambda x: target.log_prob(x, 'jax'), params)
initial = model.inverse_projection(params, jnp.ones(100, dtype=jnp.float64))
checkpoint_indices = jnp.asarray([max(1, round(config['iterations'] * f)) - 1
                                  for f in scs_paper.CHECKPOINT_FRACTIONS])


def burn(x, key):
    states, accepted = core.rwm_bright_side_stepout(density, x, 1.1, key,
        nsample=config['warmup'], stepsize=.1)
    return states[-1], accepted.sum()


def direct(x, key):
    # One uninterrupted call to the original author routine. Projection of
    # every production state and the same radial summaries are inside timing.
    states, accepted = core.rwm_bright_side_stepout(density, x, 1.1, key,
        nsample=config['iterations'], stepsize=.1)
    draws = model.projection(params, states)
    norm = jnp.linalg.norm(draws, axis=1)
    radial = jnp.stack((norm >= 50., norm), axis=1)
    cumulative = jnp.cumsum(radial, axis=0)[checkpoint_indices]
    return radial[config['thin']-1::config['thin']], radial.sum(axis=0), accepted.sum(), cumulative


start = time.perf_counter()
burn_program = jax.jit(burn).lower(initial, jax.random.key(0)).compile()
direct_program = jax.jit(direct).lower(initial, jax.random.key(0)).compile()
direct_compile_seconds = time.perf_counter() - start


def run_direct(seed):
    burn_key, sample_key = jax.random.split(jax.random.key(seed))
    state, accepted = burn_program(initial, burn_key)
    jax.block_until_ready((state, accepted))
    start, cpu_start = time.perf_counter(), time.process_time()
    result = direct_program(state, sample_key)
    jax.block_until_ready(result)
    seconds, cpu_seconds = time.perf_counter() - start, time.process_time() - cpu_start
    values = [np.asarray(item) for item in result]
    if not all(np.all(np.isfinite(value)) for value in values):
        raise ValueError('Nonfinite direct-scan result')
    return dict(seconds=seconds, cpu_seconds=cpu_seconds,
                mean_all_iterations=(values[1]/config['iterations']).tolist(),
                acceptance=float(values[2])/config['iterations'])


def run_adapter(seed):
    _, record = scs_paper.run_chain(mapping, config, seed)
    return {key: record[key] for key in ('seconds', 'cpu_seconds', 'mean_all_iterations', 'acceptance')}


# Prime both executables and memory before alternating recorded calls.
run_direct(950001)
run_adapter(950001)
records = []
for repeat in range(3):
    order = ('adapter', 'direct') if repeat % 2 == 0 else ('direct', 'adapter')
    for method in order:
        record = (run_adapter if method == 'adapter' else run_direct)(950100 + repeat)
        records.append(dict(method=method, repeat=repeat, **record))

medians = {method: float(np.median([r['seconds'] for r in records if r['method'] == method]))
           for method in ('adapter', 'direct')}
result = dict(config=config, source_commit=scs_paper.UPSTREAM_COMMIT,
    mapping_file=str(mapping_path), direct_compile_seconds=direct_compile_seconds,
    time_scope='Synchronized compiled production; all states contribute tail/norm sums and checkpoints; excludes compilation and discarded warmup',
    direct_description='One unmodified author SCS scan, followed by original projection of every state and radial summaries inside the same compiled executable',
    adapter_description='Current block=thin adapter calling the unchanged author scan and accumulating radial summaries online',
    median_seconds=medians, adapter_over_direct=medians['adapter']/medians['direct'],
    records=records)
path = args.output
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
