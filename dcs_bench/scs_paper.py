"""Pinned author SCS fitting for all targets and skew-t comparison adapter.

The vendored SCS projection, VI optimizer and transitions are unchanged. This
adapter removes progress callbacks, bounds memory with short scan blocks, and
accumulates the same two radial functions used by the DCS benchmark. Independent
PRNG splits between blocks preserve the transition law, not the upstream random
number sequence. JAX is imported lazily after worker CPU limits are configured.
"""
import builtins
from functools import lru_cache
import hashlib
import json
from numbers import Real
from pathlib import Path
import time
import types

import numpy as np

from .targets import Target
from .transforms import SphereMap


UPSTREAM_COMMIT = 'ea6f27a6cb1d0ca90d8c1cd92c32fc3dcf8ea565'
VENDOR = Path(__file__).resolve().parents[1] / 'third_party' / 'scs_paper'
STEP_SIZE = .1
LATITUDE = 1.1
CHECKPOINT_FRACTIONS = (.05, .2, .5, 1.)


def verify_vendor():
    manifest = json.loads((VENDOR / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['commit'] != UPSTREAM_COMMIT:
        raise ValueError('Unexpected SCS source revision')
    for filename, expected in manifest['files'].items():
        data = (VENDOR / filename).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected['sha256']:
            raise ValueError(f'SCS source checksum mismatch: {filename}')
    return manifest


@lru_cache(maxsize=1)
def _author():
    """Load unmodified source with private imports for its two local helpers."""
    verify_vendor()
    import jax
    jax.config.update('jax_enable_x64', True)
    progress = types.ModuleType('jax_tqdm')
    progress.scan_tqdm = lambda count: lambda function: function
    train = types.ModuleType('_scs_paper_train')
    original_import = builtins.__import__

    def private_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == 'jax_tqdm':
            return progress
        if name == 'src.train':
            return train
        return original_import(name, globals, locals, fromlist, level)

    private_builtins = dict(vars(builtins), __import__=private_import)
    train.__dict__['__builtins__'] = private_builtins
    train.__file__ = str(VENDOR / 'src' / 'train.py')
    exec(compile(Path(train.__file__).read_bytes(), train.__file__, 'exec'), train.__dict__)
    core = types.ModuleType('_scs_paper_core')
    core.__dict__['__builtins__'] = private_builtins
    core.__file__ = str(VENDOR / 'src' / 'scp_core.py')
    exec(compile(Path(core.__file__).read_bytes(), core.__file__, 'exec'), core.__dict__)
    return core


def target_spec(config):
    """Use the original DCS target, explicitly not the paper's AC skew-t law."""
    spec = dict(config.get('target', {
        'kind': 'skew_t', 'dimension': config.get('dimension', 100),
        'nu': config.get('nu', 3.), 'alpha': config.get('alpha', {0: 100., 1: -100.}),
    }))
    if spec['kind'] != 'skew_t':
        raise ValueError('The comparison adapter requires a skew-t benchmark target')
    return spec


def author_vi_budget(config):
    """Resolve the independent SCS budget, never inheriting DCS fit settings."""
    budget = {'optimizer_steps': 1000, 'optimizer_samples': 2000}
    supplied = config.get('scs_vi', {})
    if not isinstance(supplied, dict) or set(supplied) - set(budget):
        raise ValueError('scs_vi accepts only optimizer_steps and optimizer_samples')
    budget.update(supplied)
    for name, value in budget.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'scs_vi.{name} must be a positive integer')
    return budget


def fit(config):
    """Return ``(mapping_parameters, metadata)`` using the author VI routine.

    Full defaults are 2,000 fixed reference draws, 1,000 Adam iterations,
    learning rate .01 and fit seed 0. Small presets may override only the first
    two quantities through ``optimizer_samples`` and ``optimizer_steps``.
    """
    budget = author_vi_budget({'scs_vi': {
        name: config[name] for name in ('optimizer_steps', 'optimizer_samples') if name in config}})
    latitude = config.get('latitude', LATITUDE)
    if (isinstance(latitude, bool) or not isinstance(latitude, Real)
            or not np.isfinite(latitude) or not 1 <= latitude < 2):
        raise ValueError('SCS VI latitude must be finite and in [1, 2)')
    started = time.perf_counter()
    import jax
    core = _author()
    # The author's objective accepts any log density. The production adapter's
    # skew-t-only contract must not restrict fitting for the generic experiments.
    target = Target(config['target'] if 'target' in config else target_spec(config))
    model = core.SCP(target.d, float(latitude))
    ntrain, steps = budget['optimizer_samples'], budget['optimizer_steps']
    params, losses = model.minimize_reverse_kl(
        lambda x: target.log_prob(x, 'jax'), seed=0, ntrain=ntrain,
        learning_rate=.01, max_iter=steps, clip_value=1000.)
    jax.block_until_ready(params)
    observer, centre, radius = model.transform_params(params)
    loss = np.asarray(losses, dtype=float)
    if not all(np.all(np.isfinite(value)) for value in (loss, observer, centre, radius)):
        raise FloatingPointError('The original SCS VI optimizer produced nonfinite output')
    mapping = SphereMap(target.d, mu=np.asarray(centre), scale=float(radius),
                        observer=np.asarray(observer), latitude=float(latitude))
    elapsed = time.perf_counter() - started
    metadata = {
        'seconds': elapsed, 'steps': steps, 'training_draws': ntrain,
        'total_training_draws': ntrain, 'seed': 0, 'learning_rate': .01,
        'gradient_updates': steps, 'objective_point_evaluations': steps*ntrain,
        'unique_reference_draws': ntrain,
        'batch_policy': 'fixed', 'initial_scale': 1., 'loss': loss.tolist(),
        'initial_centre': 'zero', 'initial_observer': 'zero',
        'objective': 'author reverse KL; target and Jacobian evaluated at clipped physical coordinates',
        'vi_coordinate_clip': 1000., 'gradient_clipping': False,
        'source_commit': UPSTREAM_COMMIT,
        'source': 'third_party/scs_paper/src/scp_core.py',
        'source_sha256': {path: details['sha256'] for path, details in verify_vendor()['files'].items()},
        'target_kind': target.kind,
        'target_note': 'Author VI applied to the experiment target without changing its density',
        'scale_policy': 'VI-fitted', 'map_policy': 'SCS-paper-VI',
        'scale_initialization': 'unit', 'symmetric_constraints': False,
        'vi_enabled': True, 'coordinate_clipping': True, 'cdf_clipping': False,
        'fitting_policy': ('Original SCS author reverse KL, fixed base draws with seed 0, '
                          'Adam 0.01, physical-coordinate clip 1000; no gradient clipping'),
        'full_author_vi_budget': ntrain == 2000 and steps == 1000,
    }
    if target.kind == 'skew_t':
        metadata.update(target_skew_cdf='normal',
            target_note='Original DCS target; differs from the SCS paper Student-t-CDF skew target')
    return mapping.parameters(), metadata


def _parameters(model, mapping):
    import jax.numpy as jnp
    if float(mapping['latitude']) != LATITUDE:
        raise ValueError('The SCS paper comparison fixes latitude at 1.1')
    return model.inverse_transform_params(jnp.asarray(mapping['observer']),
                                          jnp.asarray(mapping['mu']),
                                          jnp.asarray(mapping['scale']))


@lru_cache(maxsize=8)
def _programs(mapping_json, spec_json, iterations, warmup, thin):
    """Compile all numerical work before either measured chain phase."""
    import jax
    import jax.numpy as jnp
    core = _author()
    mapping = json.loads(mapping_json)
    target = Target(json.loads(spec_json))
    model = core.SCP(target.d, LATITUDE)
    params = _parameters(model, mapping)
    density = model.transform_target(lambda x: target.log_prob(x, 'jax'), params)
    initial = model.inverse_projection(params, jnp.ones(target.d, dtype=jnp.float64))

    def author_block(x, key, count):
        # Calls the entire original function, including its original inner
        # transition. No proposal or acceptance expression is reimplemented.
        return core.rwm_bright_side_stepout(
            density, x, LATITUDE, key, nsample=count, stepsize=STEP_SIZE)

    def radial(states):
        physical = model.projection(params, states)
        norms = jnp.linalg.norm(physical, axis=1)
        return jnp.stack((norms >= 50., norms), axis=1)

    def burn(x, key):
        if warmup:
            states, accepts = author_block(x, key, warmup)
            return states[-1], jnp.sum(accepts)
        return x, jnp.asarray(0., dtype=jnp.float64)

    blocks, remainder = divmod(iterations, thin)
    checkpoint_steps = jnp.asarray([max(1, round(iterations * fraction))
                                   for fraction in CHECKPOINT_FRACTIONS])

    def save_checkpoints(values, totals, offset, saved):
        indices = checkpoint_steps - 1 - offset
        valid = (indices >= 0) & (indices < len(values))
        def update(_):
            cumulative = totals + jnp.cumsum(values, axis=0)
            candidate = cumulative[jnp.clip(indices, 0, len(values) - 1)]
            return jnp.where(valid[:, None], candidate, saved)
        return jax.lax.cond(jnp.any(valid), update, lambda _: saved, None)

    def production(x, key):
        def block(carry, index):
            position, rng, totals, accepted, checkpoints = carry
            rng, subkey = jax.random.split(rng)
            states, accepts = author_block(position, subkey, thin)
            values = radial(states)
            checkpoints = save_checkpoints(values, totals, index * thin, checkpoints)
            carry = (states[-1], rng, totals + jnp.sum(values, axis=0),
                     accepted + jnp.sum(accepts), checkpoints)
            return carry, values[-1]

        carry = (x, key, jnp.zeros(2, dtype=jnp.float64), jnp.asarray(0., dtype=jnp.float64),
                 jnp.zeros((len(CHECKPOINT_FRACTIONS), 2), dtype=jnp.float64))
        (x, key, totals, accepted, checkpoints), traces = jax.lax.scan(
            block, carry, xs=jnp.arange(blocks))
        if remainder:
            key, subkey = jax.random.split(key)
            states, accepts = author_block(x, subkey, remainder)
            values = radial(states)
            checkpoints = save_checkpoints(values, totals, blocks * thin, checkpoints)
            totals = totals + jnp.sum(values, axis=0)
            accepted = accepted + jnp.sum(accepts)
            x = states[-1]
        return traces, totals, accepted, x, checkpoints

    key = jax.random.key(0)
    start = time.perf_counter()
    burn_program = jax.jit(burn).lower(initial, key).compile()
    production_program = jax.jit(production).lower(initial, key).compile()
    seconds = time.perf_counter() - start
    return burn_program, production_program, initial, seconds


def run_chain(mapping, config, seed, before_sampling=None):
    """Return radial traces and exact all-iteration estimates with phase clocks.

    ``config`` supplies iterations, warmup and thin, and optionally target.
    The step is fixed at .1, as in the paper's skew-t comparison. Compilation
    and discarded warmup are excluded from sampling_seconds. Every timed JAX
    result is explicitly synchronized before the sampling clock stops.
    """
    core = _author()
    del core
    import jax
    iterations, warmup, thin = (int(config[k]) for k in ('iterations', 'warmup', 'thin'))
    if iterations < 1 or warmup < 0 or thin < 1 or thin > iterations:
        raise ValueError('Invalid SCS sampling, warmup or thinning counts')
    spec = target_spec(config)
    if int(mapping['dimension']) != int(spec['dimension']):
        raise ValueError('SCS map and target dimensions differ')
    mapping_json, spec_json = json.dumps(mapping, sort_keys=True), json.dumps(spec, sort_keys=True)
    cache_before = _programs.cache_info().misses
    burn, production, initial, compilation = _programs(mapping_json, spec_json, iterations, warmup, thin)
    compiled_now = _programs.cache_info().misses != cache_before
    burn_key, sample_key = jax.random.split(jax.random.key(int(seed)))
    started = time.perf_counter()
    state, warmup_accepts = burn(initial, burn_key)
    jax.block_until_ready((state, warmup_accepts))
    warmup_seconds = time.perf_counter() - started
    if before_sampling is not None:
        before_sampling()
    wall_started, cpu_started = time.perf_counter(), time.process_time()
    result = production(state, sample_key)
    jax.block_until_ready(result)
    sampling_seconds = time.perf_counter() - wall_started
    cpu_seconds = time.process_time() - cpu_started
    traces, totals, accepted, final_state, cumulative = (np.asarray(item) for item in result)
    if not all(np.all(np.isfinite(item)) for item in (traces, totals, final_state, cumulative)):
        raise FloatingPointError('The original SCS transition produced a nonfinite result')
    estimates = totals / iterations
    checkpoints = []
    for fraction, sums in zip(CHECKPOINT_FRACTIONS, cumulative):
        count = max(1, round(iterations * fraction))
        checkpoints.append({'fraction': fraction, 'iterations': count,
                            'mean_all_iterations': (sums / count).tolist(),
                            'tail_count': int(sums[0]), 'norm_sum': float(sums[1])})
    return traces, {
        'algorithm': 'SCS', 'iterations': iterations, 'retained': len(traces),
        'sampling_seconds': sampling_seconds, 'seconds': sampling_seconds, 'cpu_seconds': cpu_seconds,
        'warmup_seconds': warmup_seconds,
        'compile_seconds': compilation if compiled_now else 0.,
        'compile_cache_hit': not compiled_now,
        'acceptance': float(accepted) / iterations,
        'warmup_acceptance': float(warmup_accepts) / warmup if warmup else None,
        'accepted': int(accepted), 'scale': STEP_SIZE, 'final_scale': STEP_SIZE,
        'warmup_iterations': warmup, 'adaptation_during_sampling': False,
        'density_evaluations': 2 * iterations,
        'estimates': {'tail_probability': float(estimates[0]), 'mean_norm': float(estimates[1])},
        'function_sums': {'tail_probability': float(totals[0]), 'mean_norm': float(totals[1])},
        'mean_all_iterations': estimates.tolist(),
        'tail_count': int(totals[0]), 'norm_sum': float(totals[1]),
        'checkpoints': checkpoints,
        'source_commit': UPSTREAM_COMMIT, 'implementation': 'original-author-JAX-transition',
        'target_skew_cdf': 'normal', 'rng_block_size': thin,
        'rng_note': 'Independent keys per block; original transition law, not upstream bitwise RNG sequence',
        'estimator': 'all production iterations including repeated states after rejection',
    }
