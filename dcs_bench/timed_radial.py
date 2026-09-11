"""Equal-time, frozen-kernel radial comparison with exact first-third burn-in.

All methods finish batches of 300 transitions. Prefix sums every 100 steps
therefore locate one third of the final iteration count exactly, without
retaining every state. Compilation and separate DCS step tuning are untimed.
"""
from functools import lru_cache
import json
import time

import numpy as np
from numba import njit, typeof

from . import fast_dcs

BATCH_SIZE = 300
PREFIX_STRIDE = 100


@njit(nogil=True)
def _frozen_dcs(rng, z, lp, radius, scale, thin, threshold, ball_walk,
                mu, delta, alpha, constants):
    """Same compiled DCS transition, with no adaptation inside production."""
    prefixes = np.empty((3, 3))
    norms = np.empty(BATCH_SIZE // thin)
    tail_count, norm_sum, accepted = 0, 0., 0
    for index in range(BATCH_SIZE):
        z, lp, radius, probability, did_accept = fast_dcs._transition(
            rng, z, lp, radius, scale, ball_walk, mu, delta, alpha, constants)
        tail_count += int(radius >= threshold)
        norm_sum += radius
        accepted += int(did_accept)
        if (index + 1) % thin == 0:
            norms[(index + 1) // thin - 1] = radius
        if (index + 1) % PREFIX_STRIDE == 0:
            row = (index + 1) // PREFIX_STRIDE - 1
            prefixes[row, 0] = tail_count
            prefixes[row, 1] = norm_sum
            prefixes[row, 2] = accepted
    return z, lp, radius, prefixes, norms


class _DCS:
    def __init__(self, mapping_dict, config, seed):
        mapping, mu, delta, alpha, constants = fast_dcs._inputs(mapping_dict, config)
        self.extra = (config['thin'], float(config.get('threshold', 50.)),
                      config['algorithm'] == 'DCS-BW', mu, delta, alpha, constants)
        z = np.ascontiguousarray(mapping.inverse(np.ones(mapping.d)))
        started = time.perf_counter()
        lp, radius = fast_dcs._evaluate(z, mu, delta, alpha, constants)
        fast_dcs._run_block(np.random.default_rng(0), z.copy(), lp, radius,
            .1, 0., 0., 0, 0, 1, self.extra[1], self.extra[2], mu, delta, alpha, constants)
        # Compile without executing production transitions or consuming the
        # actual chain RNG. Each process resolves its own SciPy CDF address.
        signature_values = (np.random.default_rng(0), z.copy(), lp, radius, .1, *self.extra)
        _frozen_dcs.compile(tuple(typeof(value) for value in signature_values))
        self.compile_seconds = time.perf_counter() - started
        if not np.isfinite(lp) or not np.isfinite(radius):
            raise ValueError('Nonfinite initial DCS density or radius')
        self.rng = np.random.default_rng(seed)
        warmup = config['warmup']
        started = time.perf_counter()
        warm = fast_dcs._run_block(self.rng, z, lp, radius, .1, 0., 0., 0, warmup,
            max(1, warmup + 1), self.extra[1], self.extra[2], mu, delta, alpha, constants)
        self.warmup_seconds = time.perf_counter() - started
        self.state, self.lp, self.radius, self.scale = warm[:4]
        self.warmup_acceptance = warm[8] / warmup if warmup else None
        self.implementation = 'Numba DCS transition; step tuned separately then frozen'

    def batch(self):
        self.state, self.lp, self.radius, prefixes, norms = _frozen_dcs(
            self.rng, self.state, self.lp, self.radius, self.scale, *self.extra)
        return prefixes, norms


@lru_cache(maxsize=8)
def _scs_programs(mapping_json, spec_json, warmup, thin, threshold):
    from . import scs_paper
    core = scs_paper._author()
    import jax
    import jax.numpy as jnp
    from .targets import Target
    mapping = json.loads(mapping_json)
    target = Target(json.loads(spec_json))
    model = core.SCP(target.d, scs_paper.LATITUDE)
    params = scs_paper._parameters(model, mapping)
    density = model.transform_target(lambda x: target.log_prob(x, 'jax'), params)
    initial = model.inverse_projection(params, jnp.ones(target.d, dtype=jnp.float64))

    def author(x, key, count):
        return core.rwm_bright_side_stepout(density, x, scs_paper.LATITUDE,
            key, nsample=count, stepsize=scs_paper.STEP_SIZE)

    def burn(x, key):
        if not warmup:
            return x, jnp.asarray(0., dtype=jnp.float64)
        states, accepted = author(x, key, warmup)
        return states[-1], accepted.sum()

    def batch(x, key):
        # The original author's complete scan is invoked unchanged.
        states, accepted = author(x, key, BATCH_SIZE)
        norm = jnp.linalg.norm(model.projection(params, states), axis=1)
        values = jnp.stack((norm >= threshold, norm, accepted), axis=1)
        # A 3x100 reduction gives exactly the same checkpoints used by DCS.
        prefixes = jnp.cumsum(values.reshape(3, PREFIX_STRIDE, 3).sum(axis=1), axis=0)
        return states[-1], prefixes, norm[thin-1::thin]

    key = jax.random.key(0)
    started = time.perf_counter()
    burn_program = jax.jit(burn).lower(initial, key).compile()
    batch_program = jax.jit(batch).lower(initial, key).compile()
    return burn_program, batch_program, initial, time.perf_counter() - started


class _SCS:
    def __init__(self, mapping, config, seed):
        from . import scs_paper
        scs_paper._author()
        import jax
        spec = scs_paper.target_spec(config)
        if int(mapping['dimension']) != int(spec['dimension']):
            raise ValueError('SCS map and target dimensions differ')
        previous = _scs_programs.cache_info().misses
        burn, self.program, initial, compile_seconds = _scs_programs(
            json.dumps(mapping, sort_keys=True), json.dumps(spec, sort_keys=True),
            config['warmup'], config['thin'], float(config.get('threshold', 50.)))
        self.compile_seconds = compile_seconds if _scs_programs.cache_info().misses > previous else 0.
        burn_key, self.key = jax.random.split(jax.random.key(seed))
        started = time.perf_counter()
        self.state, accepts = jax.block_until_ready(burn(initial, burn_key))
        self.warmup_seconds = time.perf_counter() - started
        self.warmup_acceptance = float(accepts) / config['warmup'] if config['warmup'] else None
        self.scale = scs_paper.STEP_SIZE
        self.implementation = 'Original author JAX SCS transition, fixed paper step 0.1'

    def batch(self):
        import jax
        self.key, subkey = jax.random.split(self.key)
        self.state, prefixes, norms = jax.block_until_ready(self.program(self.state, subkey))
        return np.asarray(prefixes), np.asarray(norms)


def _backend(mapping, config, seed):
    return (_SCS if config['algorithm'] == 'SCS' else _DCS)(mapping, config, seed)


def _validate(config):
    if config['algorithm'] not in ('DCS-RWM', 'DCS-BW', 'SCS'):
        raise ValueError('Timed radial comparison supports DCS-RWM, DCS-BW and SCS')
    if not np.isfinite(config['time_budget']) or config['time_budget'] <= 0:
        raise ValueError('time_budget must be positive and finite')
    for name in ('thin', 'warmup'):
        if isinstance(config[name], bool) or not isinstance(config[name], int):
            raise ValueError(f'{name} must be an integer')
    if config['warmup'] < 0 or config['thin'] < 1 or PREFIX_STRIDE % config['thin']:
        raise ValueError('Require nonnegative warmup and thin to be a positive divisor of 100')


def _postburn(prefixes, norms, thin, threshold):
    """Remove exactly N/3 transitions using stored every-100-step prefixes."""
    iterations = len(prefixes) * PREFIX_STRIDE
    if iterations == 0 or iterations % BATCH_SIZE:
        raise ValueError('Timed histories must contain complete 300-step batches')
    burn = iterations // 3
    totals = prefixes[-1]
    sums = totals - prefixes[burn // PREFIX_STRIDE - 1]
    count = iterations - burn
    retained_norms = norms[burn // thin:]
    traces = np.column_stack((retained_norms >= threshold, retained_norms))
    return traces, iterations, burn, count, totals, sums


def run_timed(mapping_dict, config, seed, before_sampling=None, after_sampling=None):
    """Return post-burn radial traces and exact post-burn iteration summaries.

    ``time_budget`` is measured only after separate warmup/step calibration.
    Production kernels are frozen. The first third of actual timed iterations
    is discarded for every method. Completion is rounded up to a 300-step
    batch; the measured overshoot is always reported.
    """
    _validate(config)
    backend = _backend(mapping_dict, config, int(seed))
    thin, threshold = config['thin'], float(config.get('threshold', 50.))
    capacity, batches = 1024, 0
    prefix_buffer = np.empty((capacity * 3, 3), dtype=float)
    norm_buffer = np.empty(capacity * BATCH_SIZE // thin, dtype=float)
    totals = np.zeros(3, dtype=float)
    if before_sampling is not None:
        before_sampling()
    started, cpu_started = time.perf_counter(), time.process_time()
    while True:
        prefixes, norms = backend.batch()
        if batches == capacity:
            capacity *= 2
            larger = np.empty((capacity * 3, 3), dtype=float)
            larger[:batches * 3] = prefix_buffer
            prefix_buffer = larger
            larger = np.empty(capacity * BATCH_SIZE // thin, dtype=float)
            larger[:batches * BATCH_SIZE // thin] = norm_buffer
            norm_buffer = larger
        prefix_buffer[batches * 3:(batches + 1) * 3] = prefixes + totals
        totals = prefix_buffer[(batches + 1) * 3 - 1].copy()
        start = batches * BATCH_SIZE // thin
        norm_buffer[start:start + BATCH_SIZE // thin] = norms
        batches += 1
        elapsed = time.perf_counter() - started
        if elapsed >= config['time_budget']:
            break
    cpu_seconds = time.process_time() - cpu_started
    if after_sampling is not None:
        after_sampling()
    prefixes = prefix_buffer[:batches * 3]
    norms = norm_buffer[:batches * BATCH_SIZE // thin]
    if not np.all(np.isfinite(prefixes)) or not np.all(np.isfinite(norms)):
        raise FloatingPointError('Nonfinite timed radial summaries')
    traces, iterations, burn, count, totals, sums = _postburn(prefixes, norms, thin, threshold)
    estimate = sums[:2] / count
    record = {
        'algorithm': config['algorithm'], 'seed': int(seed),
        'iterations': iterations, 'discarded_iterations': burn, 'estimation_iterations': count,
        'burnin_iterations': burn, 'discarded': burn,
        'retained': len(traces), 'retained_draws': len(traces), 'thin': thin,
        'seconds': elapsed, 'sampling_seconds': elapsed, 'cpu_seconds': cpu_seconds,
        'requested_seconds': float(config['time_budget']),
        'budget_overshoot_seconds': max(0., elapsed - config['time_budget']),
        'batch_size': BATCH_SIZE, 'prefix_stride': PREFIX_STRIDE, 'batches': batches,
        'compile_seconds': backend.compile_seconds, 'warmup_seconds': backend.warmup_seconds,
        'warmup_iterations': config['warmup'], 'warmup_acceptance': backend.warmup_acceptance,
        'initial_scale': .1, 'scale': float(backend.scale), 'final_scale': float(backend.scale),
        'frozen_scale': float(backend.scale), 'adaptation_during_sampling': False,
        'step_policy': ('fixed SCS paper step' if config['algorithm'] == 'SCS'
                        else 'original smoothed dual averaging to 0.234 during separate warmup only'),
        'accepted': int(totals[2]), 'acceptance': float(totals[2] / iterations),
        'postburn_accepted': int(sums[2]), 'postburn_acceptance': float(sums[2] / count),
        'tail_count': int(sums[0]), 'norm_sum': float(sums[1]),
        'mean_all_iterations': estimate.tolist(), 'mean_post_burnin': estimate.tolist(),
        'estimates': {'tail_probability': float(estimate[0]), 'mean_norm': float(estimate[1])},
        'all_timed_tail_count': int(totals[0]), 'all_timed_norm_sum': float(totals[1]),
        'burn_fraction': '1/3 of actual timed iterations after separate warmup',
        'estimator': 'all post-burn timed iterations, including rejection repeats; thinning affects saved traces only',
        'implementation': backend.implementation,
        'history_storage_bytes': int(prefix_buffer.nbytes + norm_buffer.nbytes),
        'target_skew_cdf': 'normal', 'checkpoints': [],
    }
    if config['algorithm'] == 'SCS':
        from .scs_paper import UPSTREAM_COMMIT
        record['source_commit'] = UPSTREAM_COMMIT
        record['rng_note'] = 'Independent keys per 300-step block; unchanged original author transition'
    return traces, record
