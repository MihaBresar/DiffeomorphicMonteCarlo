"""Equal-time comparison using the author SCS and matched JAX DCS execution.

Both methods materialize the same 300-state scan and run the same vectorized
radial bookkeeping. DCS evaluates both densities for admissible proposals;
standard BW rejects outside-ball proposals and its separate reflected variant
relocates them along the proposal chord. Compilation and
shared separate step calibration are untimed; all production steps are frozen.
"""
from functools import lru_cache
import json
from numbers import Real
import time

import numpy as np

from . import scs_paper
from .targets import Target
from .transforms import BallMap, mobius

BATCH_SIZE = 300
PREFIX_STRIDE = 100
STEP_POLICY = 'common raw dual averaging of observed acceptance to 0.234 during separate warmup; smoothed final scale frozen'
IMPLEMENTATION_REVISION = 'jax-dcs-exact-ball-v1'


def resolve_step_scale_ceiling(config):
    """Resolve an optional method limit while preserving the original log-15 default.

    Per-method limits take precedence over the historical shared scalar limit.
    Returning None for an unspecified limit keeps the original calibration and
    compiled-program cache key unchanged, rather than round-tripping exp/log.
    """
    initial_scale = config.get('initial_scale', .1)
    if isinstance(initial_scale, bool) or not isinstance(initial_scale, Real) or \
            not np.isfinite(initial_scale) or initial_scale <= 0.:
        raise ValueError('initial_scale must be positive and finite')
    limits = config.get('step_scale_ceilings', {})
    if not isinstance(limits, dict):
        raise ValueError('step_scale_ceilings must map method names to finite numeric limits')
    methods = ('DCS-RWM', 'DCS-BW', 'DCS-BW-reflect', 'SCS')
    for name, value in limits.items():
        if name not in methods:
            raise ValueError(f'Unknown step-scale ceiling method: {name}')
        if isinstance(value, bool) or not isinstance(value, Real) or \
                not np.isfinite(value) or value < initial_scale:
            raise ValueError(f'step_scale_ceilings[{name!r}] must be finite and at least initial_scale')
    shared = config.get('step_scale_ceiling')
    if shared is not None and (isinstance(shared, bool) or not isinstance(shared, Real) or
            not np.isfinite(shared) or shared < initial_scale):
        raise ValueError('step_scale_ceiling must be finite and at least initial_scale')
    selected = limits.get(config['algorithm'], shared)
    return None if selected is None else float(selected)


def _uniform_ball(key, dimension):
    """Uniform d-ball draw with one Gaussian-vector call and no radial RNG.

    For independent standard normals G in R^(d+2), G[:d]/||G|| has an
    independent uniform direction and squared radius Beta(d/2, 1). Thus its
    radius has CDF r^d, exactly as direction * U**(1/d). The random stream
    differs from that older construction; the proposal distribution does not.
    """
    import jax
    import jax.numpy as jnp
    gaussian = jax.random.normal(key, shape=(dimension + 2,))
    return gaussian[:dimension] / jnp.linalg.norm(gaussian)


def _dcs_components(mapping_dict, target, algorithm):
    """DCS scan written in the same JAX style as the unmodified SCS scan."""
    import jax
    import jax.numpy as jnp
    mapping = BallMap(**mapping_dict)
    mu, delta = jnp.asarray(mapping.mu), jnp.asarray(mapping.delta)
    beta, scale, dimension = mapping.beta, mapping.scale, mapping.d

    def transform(z):
        if beta == 2.:
            r2 = jnp.sum(z * z, axis=-1)
            delta2 = jnp.sum(delta * delta)
            dot = z @ delta
            denominator = 1. + 2. * dot + delta2 * r2
            w = ((1. - delta2) * z +
                 (1. + 2. * dot + r2)[..., None] * delta) / denominator[..., None]
            logmobius = dimension * (jnp.log1p(-delta2) - jnp.log(denominator))
            # Exact identity for 1-||Mobius_delta(z)||^2. This avoids both
            # general powers and subtracting the rounded norm of w from one.
            eps = (1. - delta2) * (1. - r2) / denominator
            logeps = jnp.log(eps)
            radial_denominator = jnp.sqrt(eps)
        else:
            w, logmobius = mobius(z, delta, jnp)
            radius = jnp.linalg.norm(w, axis=-1)
            logeps = jnp.log(-jnp.expm1(beta * jnp.log(radius)))
            radial_denominator = jnp.exp(logeps / beta)
        physical = mu + scale * w / radial_denominator[..., None]
        logjac = dimension * jnp.log(scale) - (1 + dimension / beta) * logeps + logmobius
        return physical, logjac

    def interior(z):
        x, logjac = transform(z)
        value = target.log_prob(x, 'jax') + logjac
        return jnp.where(jnp.isfinite(value), value, -jnp.inf)

    def density(z):
        return jax.lax.cond(jnp.sum(z * z) < 1., interior,
                            lambda _: jnp.asarray(-jnp.inf), z)

    def block(x0, key, count, stepsize):
        def step(carry, index):
            x, key = carry
            key, subkey = jax.random.split(key)
            if algorithm in ('DCS-BW', 'DCS-BW-reflect'):
                eps = stepsize * _uniform_ball(subkey, dimension)
            else:
                eps = stepsize * jax.random.normal(subkey, shape=x.shape)
            proposed = x + eps
            valid = jnp.asarray(True)
            if algorithm == 'DCS-BW-reflect':
                from .ball_boundary import reflect_ball_proposal
                proposed, valid = reflect_ball_proposal(x, eps)
            key, subkey = jax.random.split(key)
            logu = jnp.log(jax.random.uniform(subkey))
            # No current-state cache: both densities are recomputed for every
            # admissible proposal. Outside-ball proposals reject by definition,
            # so neither evaluation is necessary. Keep key advancement common
            # to both branches. The initial and all accepted states are valid.
            accept = jax.lax.cond(valid & (jnp.sum(proposed * proposed) < 1.),
                lambda _: (logu < interior(proposed) - interior(x)) * 1.,
                lambda _: jnp.asarray(0.), operand=None)
            x = x * (1. - accept) + proposed * accept
            return (x, key), (x, accept)
        return jax.lax.scan(step, (x0, key), jnp.arange(count))[1]

    initial = jnp.asarray(mapping.inverse(np.ones(dimension)))
    return block, lambda states: transform(states)[0], initial, density


def _components(mapping, spec, algorithm):
    """Return a full-state scan, physical projection, initial point and density."""
    core = scs_paper._author()  # Also enables the same float64 policy for DCS.
    import jax.numpy as jnp
    target = Target(spec)
    if algorithm != 'SCS':
        return _dcs_components(mapping, target, algorithm)
    model = core.SCP(target.d, scs_paper.LATITUDE)
    params = scs_paper._parameters(model, mapping)
    density = model.transform_target(lambda x: target.log_prob(x, 'jax'), params)
    initial = model.inverse_projection(params, jnp.ones(target.d, dtype=jnp.float64))

    def author_block(x, key, count, stepsize):
        # Invoke the original function unchanged, including its complete scan,
        # proposal, bright-cap stepout and Metropolis decision.
        return core.rwm_bright_side_stepout(density, x, scs_paper.LATITUDE,
                                           key, nsample=count, stepsize=stepsize)
    return author_block, lambda states: model.projection(params, states), initial, density


def _calibrate(block, initial, key, warmup, initial_scale, scale_ceiling=None):
    """Identical raw dual averaging of observed 0/1 acceptance for all methods."""
    import jax
    import jax.numpy as jnp
    log_ceiling = 15. if scale_ceiling is None else jnp.log(scale_ceiling)
    def step(carry, iteration):
        state, key, scale, hbar, logbar, accepts = carry
        key, subkey = jax.random.split(key)
        states, accepted = block(state, subkey, 1, scale)
        accept = accepted[0]
        eta = 1. / (iteration + 10.)
        hbar = (1. - eta) * hbar + eta * (.234 - accept)
        logscale = jnp.log(10. * initial_scale) - jnp.sqrt(iteration) / .05 * hbar
        weight = iteration ** (-.75)
        logbar = weight * logscale + (1. - weight) * logbar
        scale = jnp.exp(jnp.clip(logscale, -30., log_ceiling))
        return (states[-1], key, scale, hbar, logbar, accepts + accept), None
    carry = (initial, key, initial_scale, jnp.asarray(0.),
             jnp.log(initial_scale), jnp.asarray(0.))
    final, _ = jax.lax.scan(step, carry, jnp.arange(1, warmup + 1, dtype=jnp.float64))
    return final[0], jnp.exp(jnp.clip(final[4], -30., log_ceiling)), final[5]


def _radial_batch(block, projection, state, key, scale, thin, threshold):
    """The same full-state projection and statistic work, including rejections."""
    import jax.numpy as jnp
    states, accepted = block(state, key, BATCH_SIZE, scale)
    norms = jnp.linalg.norm(projection(states), axis=1)
    values = jnp.stack((norms >= threshold, norms, accepted), axis=1)
    prefixes = jnp.cumsum(values.reshape(3, PREFIX_STRIDE, 3).sum(axis=1), axis=0)
    return states[-1], prefixes, norms[thin - 1::thin]


@lru_cache(maxsize=8)
def _programs(mapping_json, spec_json, algorithm, warmup, thin, threshold, initial_scale, scale_ceiling=None):
    """Compile both phases without consuming a chain's RNG or timing budget."""
    import jax
    import jax.numpy as jnp
    block, projection, initial, density = _components(
        json.loads(mapping_json), json.loads(spec_json), algorithm)
    if not np.isfinite(float(density(initial))) or not np.all(np.isfinite(np.asarray(projection(initial)))):
        raise ValueError('Nonfinite initial density or physical point')
    burn = lambda state, key: _calibrate(block, state, key, warmup, initial_scale, scale_ceiling)
    batch = lambda state, key, scale: _radial_batch(block, projection, state, key, scale, thin, threshold)
    device = jax.local_devices(backend='cpu')[0]
    initial, key, scale = jax.device_put((initial, jax.random.key(0), jnp.asarray(initial_scale)), device)
    started = time.perf_counter()
    burn_program = jax.jit(burn).lower(initial, key).compile()
    batch_program = jax.jit(batch).lower(initial, key, scale).compile()
    return burn_program, batch_program, initial, time.perf_counter() - started


class _Matched:
    """Common JAX driver, calibration, recording and clocks for DCS and SCS."""
    def __init__(self, mapping, config, seed):
        scs_paper._author()
        import jax
        import jax.numpy as jnp
        spec = scs_paper.target_spec(config)
        if int(mapping['dimension']) != int(spec['dimension']):
            raise ValueError('Map and target dimensions differ')
        scale_ceiling = resolve_step_scale_ceiling(config)
        self.initial_scale = float(config.get('initial_scale', .1))
        previous = _programs.cache_info().misses
        burn, self.program, initial, compile_seconds = _programs(
            json.dumps(mapping, sort_keys=True), json.dumps(spec, sort_keys=True),
            config['algorithm'], config['warmup'], config['thin'],
            float(config.get('threshold', 50.)), self.initial_scale, scale_ceiling)
        self.compile_seconds = compile_seconds if _programs.cache_info().misses > previous else 0.
        key = jax.device_put(jax.random.key(seed), jax.local_devices(backend='cpu')[0])
        burn_key, self.key = jax.random.split(key)
        started = time.perf_counter()
        self.state, scale, accepts = jax.block_until_ready(burn(initial, burn_key))
        self.warmup_seconds = time.perf_counter() - started
        self.scale = float(scale)
        self._scale = scale
        self.warmup_acceptance = float(accepts) / config['warmup'] if config['warmup'] else None
        self.implementation = ('Original author JAX SCS transition' if config['algorithm'] == 'SCS'
            else 'DCS JAX transition') + '; shared calibration, full-state scans and radial recording'

    def batch(self):
        import jax
        self.key, subkey = jax.random.split(self.key)
        self.state, prefixes, norms = jax.block_until_ready(self.program(self.state, subkey, self._scale))
        return np.asarray(prefixes), np.asarray(norms)


def _backend(mapping, config, seed):
    return _Matched(mapping, config, seed)


def _validate(config):
    if config['algorithm'] not in ('DCS-RWM', 'DCS-BW', 'DCS-BW-reflect', 'SCS'):
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
    scale_ceiling = resolve_step_scale_ceiling(config)
    recorded_ceiling = float(np.exp(15.) if scale_ceiling is None else scale_ceiling)
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
        'initial_scale': float(config.get('initial_scale', .1)), 'scale': float(backend.scale), 'final_scale': float(backend.scale),
        'frozen_scale': float(backend.scale), 'adaptation_during_sampling': False,
        'step_policy': STEP_POLICY,
        'target_acceptance': .234,
        'common_cpu_backend': 'JAX',
        'implementation_revision': config.get('implementation_revision', IMPLEMENTATION_REVISION),
        'boundary_policy': ('reflect_along_proposal_chord' if config['algorithm'] == 'DCS-BW-reflect'
                            else 'author_cap_stepout' if config['algorithm'] == 'SCS' else 'direct_rejection'),
        'step_scale_ceiling': recorded_ceiling,
        'scale_ceiling_reached': bool(backend.scale >= recorded_ceiling*(1-1e-10)),
        'proposal_generator': ('normal_d_plus_2_projection' if config['algorithm'] in ('DCS-BW', 'DCS-BW-reflect')
                               else 'author_tangent_normal' if config['algorithm'] == 'SCS'
                               else 'isotropic_normal'),
        'common_statistic_policy': 'Full 300-state scans; physical projection and norm for every state, including rejection repeats',
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
        record['rng_note'] = 'JAX PRNG for all methods; independent keys between warmup steps and production blocks preserve the transition law'
        record['scs_implementation'] = 'Unmodified author rwm_bright_side_stepout; shared external step calibration and radial recording'
    return traces, record
