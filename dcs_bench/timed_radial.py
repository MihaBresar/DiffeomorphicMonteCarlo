"""Equal-time comparison using the author SCS and matched JAX DCS execution.

Both methods materialize the same 300-state scan and run the same vectorized
radial bookkeeping. DCS evaluates both densities for admissible proposals;
standard BW rejects outside-ball proposals and its separate reflected variant
relocates them along the proposal chord. Compilation and
separate step calibration are untimed; all production steps are frozen.
"""
from functools import lru_cache
from copy import deepcopy
from contextlib import ExitStack, nullcontext
import hashlib
import json
from numbers import Real
from pathlib import Path
import tempfile
import time

import numpy as np

from . import scs_paper
from .targets import Target
from .transforms import BallMap, mobius

BATCH_SIZE = 300
PREFIX_STRIDE = 100
STEP_POLICY = 'common raw dual averaging of observed acceptance to 0.234 during separate warmup; smoothed final scale frozen'
ACCEPTANCE_TUNING = 'indicator_dual_averaging_0234'
SCS_FIXED_TUNING = 'author_scs_fixed_01'
FIXED_SCS_TUNING = 'fixed_scs_move_v1'
FIXED_DCS_TUNING = 'fixed_dcs_move_v1'
BALL_TUNING = 'ball_esjd_pilot_v1'
RADIAL_TUNING = 'own_kernel_radial_pilot_v1'
ROTATION_TUNING = 'split_ball_angular_esjd_v1'
ROTATION_METHOD = 'DCS-BW-rotate'
RWM_ROTATION_METHOD = 'DCS-RWM-rotate'
ROTATION_METHODS = (ROTATION_METHOD, RWM_ROTATION_METHOD)
BILLIARD_METHOD = 'DCS-Billiard'
FIXED_DCS_METHODS = ('DCS-RWM', 'DCS-BW', 'DCS-BW-reflect', BILLIARD_METHOD)
BALL_STEP_POLICY = ('Discarded-pilot squared jump distance in ball coordinates; '
                    'up to eight rotating h/2, h, 2h triplet rounds within the existing warmup budget, then frozen')
ROTATION_STEP_POLICY = ('Split the existing discarded warmup budget between reflected-BW ball movement '
                        'and fixed-radius angular movement; freeze both steps, then select one component '
                        'with probability 1/2 per production iteration')
IMPLEMENTATION_REVISION = 'jax-dcs-exact-ball-v1'


def effective_mapping(mapping, algorithm):
    """Return the map actually used, without mutating the shared VI fit.

    Billiards move in w = Mobius_delta(z), so the deformation and its Jacobian
    cancel from their production coordinates. The jointly fitted centre, R
    and beta remain; this is not a refit with delta constrained to zero.
    """
    result = deepcopy(mapping)
    if algorithm == BILLIARD_METHOD:
        result['delta'] = [0.] * int(result['dimension'])
    return result


def resolve_step_tuning(config):
    """Keep historical calibration unless a method explicitly opts into ball ESJD."""
    policies = config.get('step_tuning_policies', {})
    if not isinstance(policies, dict):
        raise ValueError('step_tuning_policies must map methods to tuning policy identifiers')
    for method, policy in policies.items():
        if method not in ('DCS-RWM', 'DCS-BW', 'DCS-BW-reflect', ROTATION_METHOD, RWM_ROTATION_METHOD, BILLIARD_METHOD, 'SCS'):
            raise ValueError(f'Unknown tuning method: {method}')
        if policy not in (ACCEPTANCE_TUNING, SCS_FIXED_TUNING, FIXED_SCS_TUNING, FIXED_DCS_TUNING, BALL_TUNING, RADIAL_TUNING, ROTATION_TUNING):
            raise ValueError(f'Unknown step tuning policy: {policy}')
        if policy == FIXED_DCS_TUNING and method not in FIXED_DCS_METHODS:
            raise ValueError('Fixed DCS moves are defined for RWM, BW, reflected BW and billiards only')
        if policy == SCS_FIXED_TUNING and method != 'SCS':
            raise ValueError('The author fixed h=0.1 control is defined for SCS only')
        if policy == FIXED_SCS_TUNING and method != 'SCS':
            raise ValueError('Supplied fixed SCS moves are defined for SCS only')
        if policy == BALL_TUNING and method not in ('DCS-BW-reflect', BILLIARD_METHOD):
            raise ValueError('Ball ESJD calibration is defined for reflected BW and billiards only')
        if policy == RADIAL_TUNING and method not in (BILLIARD_METHOD, 'SCS'):
            raise ValueError('Own-kernel radial calibration is defined for billiards and SCS only')
        if method == BILLIARD_METHOD and policy not in (BALL_TUNING, RADIAL_TUNING, FIXED_DCS_TUNING):
            raise ValueError('Billiards require a fixed own-kernel move, the historical reflected-BW pilot or own-kernel radial calibration')
        if (policy == ROTATION_TUNING) != (method in ROTATION_METHODS):
            raise ValueError('Rotation mixtures require split reflected-step/angular calibration')
    selected = policies.get(config['algorithm'],
                            ROTATION_TUNING if config['algorithm'] in ROTATION_METHODS else
                            BALL_TUNING if config['algorithm'] == BILLIARD_METHOD else ACCEPTANCE_TUNING)
    if selected == SCS_FIXED_TUNING and config.get('initial_scale', .1) != .1:
        raise ValueError('The author fixed-step control requires initial_scale=0.1')
    if selected == FIXED_SCS_TUNING:
        _fixed_scs_move(config)
        if isinstance(config['warmup'], bool) or not isinstance(config['warmup'], int) or config['warmup'] < 1:
            raise ValueError('Fixed SCS moves require a positive integer discarded warmup')
    if selected == FIXED_DCS_TUNING:
        _fixed_dcs_move(config)
        if isinstance(config['warmup'], bool) or not isinstance(config['warmup'], int) or config['warmup'] < 1:
            raise ValueError('Fixed DCS moves require a positive integer discarded warmup')
    if selected == BALL_TUNING and config['warmup'] < 3:
        raise ValueError('Ball ESJD calibration requires at least three warmup transitions')
    if selected == RADIAL_TUNING and config['warmup'] < 8:
        raise ValueError('Own-kernel radial calibration requires at least eight warmup transitions')
    if selected == ROTATION_TUNING:
        if config['warmup'] < 6:
            raise ValueError('Split ball/angular calibration requires at least six warmup transitions')
        initial = config.get('initial_angular_scale', 1.)
        if isinstance(initial, bool) or not isinstance(initial, Real) or not np.isfinite(initial) or initial <= 0.:
            raise ValueError('initial_angular_scale must be positive and finite')
        probability = config.get('mixture_probability', .5)
        if isinstance(probability, bool) or not isinstance(probability, Real) or probability != .5:
            raise ValueError('Rotation mixtures use a fixed mixture_probability of 0.5')
    return selected


def _fixed_scs_move(config):
    """Validate an explicitly supplied author-kernel tangent proposal scale."""
    if config['algorithm'] != 'SCS':
        raise ValueError('Supplied fixed SCS moves are defined for SCS only')
    step = config.get('scs_fixed_move')
    if isinstance(step, bool) or not isinstance(step, Real) or not np.isfinite(step) or step <= 0.:
        raise ValueError('scs_fixed_move must be positive and finite')
    return float(step)


def _fixed_dcs_move(config):
    """Validate the prespecified BW radius used to normalize fixed DCS moves."""
    if config['algorithm'] not in FIXED_DCS_METHODS:
        raise ValueError('Fixed DCS moves are defined for RWM, BW, reflected BW and billiards only')
    radius = config.get('dcs_fixed_move')
    if isinstance(radius, bool) or not isinstance(radius, Real) or not np.isfinite(radius) or radius <= 0.:
        raise ValueError('dcs_fixed_move must be positive and finite')
    return float(radius)


def _initial_proposal_scale(config):
    """Resolve the actual BW radius or Gaussian coordinate standard deviation."""
    policies = config.get('step_tuning_policies', {})
    if isinstance(policies, dict) and policies.get(config['algorithm']) == FIXED_SCS_TUNING:
        return _fixed_scs_move(config)
    if isinstance(policies, dict) and policies.get(config['algorithm']) == FIXED_DCS_TUNING:
        radius = _fixed_dcs_move(config)
        if config['algorithm'] == 'DCS-RWM':
            dimension = int(scs_paper.target_spec(config)['dimension'])
            return radius / np.sqrt(dimension + 2.)
        return radius
    radius = float(config.get('initial_scale', .1))
    if config['algorithm'] == RWM_ROTATION_METHOD:
        from .rwm_rotation import initial_sigma
        return initial_sigma(int(scs_paper.target_spec(config)['dimension']), radius)
    return radius


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
    methods = ('DCS-RWM', 'DCS-BW', 'DCS-BW-reflect', ROTATION_METHOD, RWM_ROTATION_METHOD, BILLIARD_METHOD, 'SCS')
    for name, value in limits.items():
        if name not in methods:
            raise ValueError(f'Unknown step-scale ceiling method: {name}')
        policies = config.get('step_tuning_policies', {})
        minimum = (_initial_proposal_scale(dict(config, algorithm=name))
                   if name == RWM_ROTATION_METHOD or
                   (isinstance(policies, dict) and policies.get(name) in (FIXED_DCS_TUNING, FIXED_SCS_TUNING))
                   else initial_scale)
        if isinstance(value, bool) or not isinstance(value, Real) or \
                not np.isfinite(value) or value < minimum:
            raise ValueError(f'step_scale_ceilings[{name!r}] must be finite and at least initial_scale')
    shared = config.get('step_scale_ceiling')
    if shared is not None and (isinstance(shared, bool) or not isinstance(shared, Real) or
            not np.isfinite(shared) or shared < _initial_proposal_scale(config)):
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
    # Exact parameter test: never approximate a small nonzero deformation.
    identity_mobius = not np.any(mapping.delta)

    def transform(z):
        if identity_mobius:
            # M_0(z)=z and log|J_M_0|=0. Keep the existing radial arithmetic
            # and domain checks; remove only the identity Mobius algebra.
            if beta == 2.:
                eps = 1. - jnp.sum(z * z, axis=-1)
                logeps = jnp.log(eps)
                radial_denominator = jnp.sqrt(eps)
            else:
                radius = jnp.linalg.norm(z, axis=-1)
                logeps = jnp.log(-jnp.expm1(beta * jnp.log(radius)))
                radial_denominator = jnp.exp(logeps / beta)
            physical = mu + scale * z / radial_denominator[..., None]
            logjac = dimension * jnp.log(scale) - (1 + dimension / beta) * logeps
            return physical, logjac
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
            if algorithm in ('DCS-BW', 'DCS-BW-reflect', BILLIARD_METHOD):
                eps = stepsize * _uniform_ball(subkey, dimension)
            else:
                eps = stepsize * jax.random.normal(subkey, shape=x.shape)
            proposed = x + eps
            valid = jnp.asarray(True)
            if algorithm == 'DCS-BW-reflect':
                from .ball_boundary import reflect_ball_proposal
                proposed, valid = reflect_ball_proposal(x, eps)
            elif algorithm == BILLIARD_METHOD:
                from .billiard import reflect_ball_billiard
                proposed, valid = reflect_ball_billiard(x, eps)
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
    if algorithm == BILLIARD_METHOD:
        mapping = effective_mapping(mapping, algorithm)
    if algorithm == RWM_ROTATION_METHOD:
        from .rwm_rotation import build_components
        return build_components(mapping, spec)
    if algorithm == ROTATION_METHOD:
        from .rotation import build_components
        parts = build_components(mapping, target, _dcs_components(mapping, target, 'DCS-BW-reflect'))
        return parts['block'], parts['projection'], parts['initial'], parts['density']
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


def _fixed_scs_warmup(block, initial, key, warmup):
    """Same discarded transition budget, with the paper's step fixed at 0.1."""
    import jax.numpy as jnp
    states, accepts = block(initial, key, warmup, jnp.asarray(.1))
    return states[-1], jnp.asarray(.1), jnp.sum(accepts)


def _fixed_dcs_warmup(block, initial, key, warmup, scale):
    """Discard the usual warmup budget without adapting a prespecified move."""
    import jax.numpy as jnp
    states, accepts = block(initial, key, warmup, jnp.asarray(scale))
    return states[-1], jnp.asarray(scale), jnp.sum(accepts)


def _fixed_scs_move_warmup(block, initial, key, warmup, scale):
    """Run the unchanged author kernel at the supplied fixed scale."""
    import jax.numpy as jnp
    states, accepts = block(initial, key, warmup, jnp.asarray(scale))
    return states[-1], jnp.asarray(scale), jnp.sum(accepts)


def _radial_batch(block, projection, state, key, scale, thin, threshold, coordinate_threshold=None):
    """The same full-state projection and statistic work, including rejections."""
    import jax.numpy as jnp
    states, accepted = block(state, key, BATCH_SIZE, scale)
    physical = projection(states)
    norms = jnp.linalg.norm(physical, axis=1)
    values = jnp.stack((norms >= threshold, norms, accepted), axis=1)
    saved = norms[thin - 1::thin]
    if coordinate_threshold is not None:
        x1, x2, x3 = physical[:, 0], physical[:, 1], physical[:, 2]
        values = jnp.column_stack((values, x1, x1 > coordinate_threshold,
                                   x2, x2 < -coordinate_threshold,
                                   x3, x3 > coordinate_threshold))
        saved = jnp.column_stack((saved, physical[thin - 1::thin, :3]))
    prefixes = jnp.cumsum(values.reshape(3, PREFIX_STRIDE, values.shape[1]).sum(axis=1), axis=0)
    return states[-1], prefixes, saved


@lru_cache(maxsize=8)
def _programs(mapping_json, spec_json, algorithm, warmup, thin, threshold, initial_scale, scale_ceiling=None,
              tuning_policy=ACCEPTANCE_TUNING, coordinate_threshold=None):
    """Compile both phases without consuming a chain's RNG or timing budget."""
    import jax
    import jax.numpy as jnp
    mapping, spec = json.loads(mapping_json), json.loads(spec_json)
    block, projection, initial, density = _components(mapping, spec, algorithm)
    # Historical v6/v7 billiards retain the reviewed reflected-BW pilot.
    # The explicit own-kernel policy uses the actual block for both phases.
    calibration_block = block
    if algorithm == BILLIARD_METHOD and tuning_policy == BALL_TUNING:
        calibration_block, _, _, _ = _components(
            effective_mapping(mapping, algorithm), spec, 'DCS-BW-reflect')
    elif algorithm == BILLIARD_METHOD and tuning_policy not in (RADIAL_TUNING, FIXED_DCS_TUNING):
        raise ValueError('Billiards require fixed own-kernel warmup, reflected-BW or own-kernel radial calibration')
    if not np.isfinite(float(density(initial))) or not np.all(np.isfinite(np.asarray(projection(initial)))):
        raise ValueError('Nonfinite initial density or physical point')
    if tuning_policy == RADIAL_TUNING:
        from .radial_tuning import calibrate_radial
        burn = lambda state, key: calibrate_radial(
            block, projection, state, key, warmup, initial_scale, scale_ceiling)
    elif tuning_policy == BALL_TUNING:
        from .ball_tuning import calibrate_ball_esjd
        burn = lambda state, key: calibrate_ball_esjd(calibration_block, state, key, warmup, initial_scale, scale_ceiling)
    elif tuning_policy == ACCEPTANCE_TUNING:
        burn = lambda state, key: _calibrate(block, state, key, warmup, initial_scale, scale_ceiling)
    elif tuning_policy == SCS_FIXED_TUNING:
        if algorithm != 'SCS' or initial_scale != .1 or warmup < 1:
            raise ValueError('Fixed h=0.1 requires SCS and a positive discarded warmup')
        burn = lambda state, key: _fixed_scs_warmup(block, state, key, warmup)
    elif tuning_policy == FIXED_SCS_TUNING:
        if algorithm != 'SCS' or warmup < 1 or not np.isfinite(initial_scale) or initial_scale <= 0.:
            raise ValueError('Supplied fixed SCS moves require SCS, a positive step and discarded warmup')
        burn = lambda state, key: _fixed_scs_move_warmup(block, state, key, warmup, initial_scale)
    elif tuning_policy == FIXED_DCS_TUNING:
        if algorithm not in FIXED_DCS_METHODS or warmup < 1 or \
                not np.isfinite(initial_scale) or initial_scale <= 0.:
            raise ValueError('Fixed DCS moves require RWM/BW/reflected BW/billiards, a positive step and discarded warmup')
        burn = lambda state, key: _fixed_dcs_warmup(block, state, key, warmup, initial_scale)
    else:
        raise ValueError(f'Unknown step tuning policy: {tuning_policy}')
    batch = lambda state, key, scale: _radial_batch(
        block, projection, state, key, scale, thin, threshold, coordinate_threshold)
    device = jax.local_devices(backend='cpu')[0]
    initial, key, scale = jax.device_put((initial, jax.random.key(0), jnp.asarray(initial_scale)), device)
    started = time.perf_counter()
    burn_program = jax.jit(burn).lower(initial, key).compile()
    batch_program = jax.jit(batch).lower(initial, key, scale).compile()
    return burn_program, batch_program, initial, time.perf_counter() - started


@lru_cache(maxsize=8)
def _rotation_programs(mapping_json, spec_json, warmup, thin, threshold, initial_scale,
                       initial_angular_scale, scale_ceiling=None, coordinate_threshold=None):
    """Compile the two-scale sampler without changing legacy compilation keys."""
    import jax
    import jax.numpy as jnp
    from .rotation import build_components, calibrate
    mapping, spec = json.loads(mapping_json), json.loads(spec_json)
    target = Target(spec)
    parts = build_components(mapping, target, _dcs_components(mapping, target, 'DCS-BW-reflect'))
    initial = parts['initial']
    if not np.isfinite(float(parts['density'](initial))) or not np.all(np.isfinite(np.asarray(parts['projection'](initial)))):
        raise ValueError('Nonfinite initial density or physical point')
    burn = lambda state, key: calibrate(parts, state, key, warmup, initial_scale,
                                        initial_angular_scale, scale_ceiling)
    batch = lambda state, key, scales: _radial_batch(parts['block'], parts['projection'],
        state, key, scales, thin, threshold, coordinate_threshold)
    device = jax.local_devices(backend='cpu')[0]
    initial, key, scales = jax.device_put((initial, jax.random.key(0),
        jnp.asarray([initial_scale, initial_angular_scale])), device)
    started = time.perf_counter()
    burn_program = jax.jit(burn).lower(initial, key).compile()
    batch_program = jax.jit(batch).lower(initial, key, scales).compile()
    return burn_program, batch_program, initial, time.perf_counter()-started


@lru_cache(maxsize=8)
def _rwm_rotation_programs(mapping_json, spec_json, warmup, thin, threshold, initial_scale,
                       initial_angular_scale, scale_ceiling=None, coordinate_threshold=None):
    """Compile the Gaussian mixture separately, preserving the BW program cache."""
    import jax
    import jax.numpy as jnp
    from .rotation import calibrate
    from .rwm_rotation import build_parts
    mapping, spec = json.loads(mapping_json), json.loads(spec_json)
    parts = build_parts(mapping, spec)
    initial = parts['initial']
    if not np.isfinite(float(parts['density'](initial))) or not np.all(np.isfinite(np.asarray(parts['projection'](initial)))):
        raise ValueError('Nonfinite initial density or physical point')
    burn = lambda state, key: calibrate(parts, state, key, warmup, initial_scale,
                                        initial_angular_scale, scale_ceiling)
    batch = lambda state, key, scales: _radial_batch(parts['block'], parts['projection'],
        state, key, scales, thin, threshold, coordinate_threshold)
    device = jax.local_devices(backend='cpu')[0]
    initial, key, scales = jax.device_put((initial, jax.random.key(0),
        jnp.asarray([initial_scale, initial_angular_scale])), device)
    started = time.perf_counter()
    burn_program = jax.jit(burn).lower(initial, key).compile()
    batch_program = jax.jit(batch).lower(initial, key, scales).compile()
    return burn_program, batch_program, initial, time.perf_counter()-started


class _Matched:
    """Common JAX driver, calibration, recording and clocks for DCS and SCS."""
    def __init__(self, mapping, config, seed):
        if config['algorithm'] == BILLIARD_METHOD:
            mapping = effective_mapping(mapping, config['algorithm'])
        scs_paper._author()
        import jax
        import jax.numpy as jnp
        spec = scs_paper.target_spec(config)
        if int(mapping['dimension']) != int(spec['dimension']):
            raise ValueError('Map and target dimensions differ')
        scale_ceiling = resolve_step_scale_ceiling(config)
        self.tuning_policy = resolve_step_tuning(config)
        self.initial_scale = _initial_proposal_scale(config)
        arguments = (
            json.dumps(mapping, sort_keys=True), json.dumps(spec, sort_keys=True),
            config['algorithm'], config['warmup'], config['thin'],
            float(config.get('threshold', 50.)), self.initial_scale, scale_ceiling)
        # Preserve the original compiled-program key and RNG path for controls.
        if self.tuning_policy in (BALL_TUNING, RADIAL_TUNING, SCS_FIXED_TUNING, FIXED_SCS_TUNING, FIXED_DCS_TUNING):
            arguments += (self.tuning_policy,)
        extra = ({'coordinate_threshold': float(config['coordinate_threshold'])}
                 if config.get('coordinate_metrics', False) else {})
        if self.tuning_policy == ROTATION_TUNING:
            compiler = (_rwm_rotation_programs if config['algorithm'] == RWM_ROTATION_METHOD
                        else _rotation_programs)
            previous = compiler.cache_info().misses
            burn, self.program, initial, compile_seconds = compiler(
                arguments[0], arguments[1], config['warmup'], config['thin'],
                float(config.get('threshold', 50.)), self.initial_scale,
                float(config.get('initial_angular_scale', 1.)), scale_ceiling, **extra)
            cache_missed = compiler.cache_info().misses > previous
        else:
            previous = _programs.cache_info().misses
            burn, self.program, initial, compile_seconds = _programs(*arguments, **extra)
            cache_missed = _programs.cache_info().misses > previous
        self.compile_seconds = compile_seconds if cache_missed else 0.
        key = jax.device_put(jax.random.key(seed), jax.local_devices(backend='cpu')[0])
        burn_key, self.key = jax.random.split(key)
        started = time.perf_counter()
        calibrated = jax.block_until_ready(burn(initial, burn_key))
        self.warmup_seconds = time.perf_counter() - started
        self.tuning_metadata = {}
        if self.tuning_policy == ROTATION_TUNING:
            self.state, scales, accepts, histories = calibrated
            if not np.all(np.isfinite(np.asarray(self.state))) or not np.all(np.isfinite(np.asarray(scales))):
                raise FloatingPointError('Nonfinite reflected-step/angular calibration')
            self.scale, self.angular_scale = map(float, scales)
            self._scale = scales
            self.tuning_metadata = _rotation_tuning_metadata(histories, self.scale, self.angular_scale, config)
        elif self.tuning_policy == RADIAL_TUNING:
            self.state, scale, accepts, history = calibrated
            history = np.asarray(history)
            if not np.all(np.isfinite(history)) or not np.all(np.isfinite(np.asarray(self.state))):
                raise FloatingPointError('Nonfinite own-kernel radial calibration')
            self.tuning_metadata = _radial_tuning_metadata(
                history, float(scale), config['warmup'], config['algorithm'])
        elif self.tuning_policy == BALL_TUNING:
            self.state, scale, accepts, history = calibrated
            history = np.asarray(history)
            if not np.all(np.isfinite(history)) or not np.all(np.isfinite(np.asarray(self.state))):
                raise FloatingPointError('Nonfinite reflected-BW calibration')
            self.tuning_metadata = _ball_tuning_metadata(history, float(scale), config['warmup'])
        else:
            self.state, scale, accepts = calibrated
        if self.tuning_policy != ROTATION_TUNING:
            self.scale = float(scale)
            self._scale = scale
        self.tuning_metadata['calibrated_state_sha256'] = hashlib.sha256(np.asarray(self.state).tobytes()).hexdigest()
        if self.tuning_policy == SCS_FIXED_TUNING:
            self.tuning_metadata.update(calibration_algorithm='SCS',
                tuning_note='Author skew-t fixed h=0.1; same discarded warmup transition budget, no step adaptation',
                step_size_adapted=False, fixed_step=.1)
        elif self.tuning_policy == FIXED_SCS_TUNING:
            self.tuning_metadata.update(calibration_algorithm='SCS',
                tuning_note='Supplied fixed SCS proposal scale; the unchanged author kernel runs '
                    'for the full discarded warmup and production without adaptation. '
                    'This policy does not claim the author paper fixed h=0.1 setting.',
                step_size_adapted=False, fixed_step=self.scale, scs_fixed_move=self.scale,
                scale_parameter='Gaussian ambient standard deviation before tangent projection',
                tuning_pilot_iterations=config['warmup'])
        elif self.tuning_policy == FIXED_DCS_TUNING:
            radius = _fixed_dcs_move(config)
            self.tuning_metadata.update(calibration_algorithm=config['algorithm'],
                tuning_note='Prespecified DCS move; the actual production kernel runs for the '
                    'full discarded warmup budget without step adaptation. BW, reflected BW and '
                    'billiards use initial uniform-ball increment radius h; '
                    'RWM uses Gaussian coordinate standard deviation h/sqrt(d+2), matching '
                    'the initial uniform-ball increment coordinate variance.',
                step_size_adapted=False, fixed_move_radius=radius, fixed_step=self.scale,
                fixed_move_normalization='BW/reflected BW/billiard increment radius h; RWM coordinate sigma=h/sqrt(d+2)',
                proposal_coordinate_variance=radius**2 / (int(spec['dimension'])+2.),
                scale_parameter='Gaussian coordinate standard deviation' if config['algorithm'] == 'DCS-RWM'
                    else 'initial uniform-ball increment radius',
                tuning_pilot_iterations=config['warmup'])
        self.warmup_acceptance = float(accepts) / config['warmup'] if config['warmup'] else None
        self.implementation = ('Original author JAX SCS transition' if config['algorithm'] == 'SCS'
            else 'DCS JAX transition') + '; separate calibration, full-state scans and radial recording'
        if config['algorithm'] == BILLIARD_METHOD:
            fixed_step = self.tuning_policy == FIXED_DCS_TUNING
            own_kernel = self.tuning_policy in (RADIAL_TUNING, FIXED_DCS_TUNING)
            self.implementation = ('Optimized endpoint-only JAX specular billiard with invariant extra chords; '
                'post-Mobius coordinates, ' + ('fixed-step own-kernel warmup' if fixed_step else
                    'own-kernel radial pilot' if own_kernel else 'reflected-BW pilot') +
                ', shared full-state statistics')
            self.tuning_metadata.update(
                effective_map=deepcopy(mapping),
                proposal_coordinates='Post-Mobius w; retained supplied centre/R/beta, zero direct deformation',
                calibration_coordinates=('Fixed-step own billiard warmup' if fixed_step else
                    'Own billiard radial-movement pilot' if own_kernel else 'Reflected-BW ball-movement pilot') +
                    ' in post-Mobius w, from the common physical all-ones start',
                calibration_algorithm=BILLIARD_METHOD if own_kernel else 'DCS-BW-reflect',
                calibrated_state_sha256=hashlib.sha256(np.asarray(self.state).tobytes()).hexdigest(),
                extra_chord_rule='floor(max(0, atan2(L,2a)/(2atan2(a,b)) - L/(2a)))',
                extra_density_evaluations=0, extra_gaussian_vectors=0, bounce_loop=False,
                map_note='The shared input map is retained for provenance; its centre/R/beta are used and its Mobius deformation cancels in this method. With VI these parameters come from the joint fit; without VI the input map is untuned. No delta=0 refit.',
                tuning_note=(self.tuning_metadata['tuning_note'] if own_kernel else
                    'Reflected-BW ball-movement pilot in post-Mobius coordinates; the selected scale and final pilot state are reused by billiard production. Same separate transition budget; no production adaptation.'))

    def batch(self):
        import jax
        self.key, subkey = jax.random.split(self.key)
        self.state, prefixes, norms = jax.block_until_ready(self.program(self.state, subkey, self._scale))
        return np.asarray(prefixes), np.asarray(norms)


def _backend(mapping, config, seed):
    calibration = _Matched(mapping, config, seed)
    if config.get('dcs_backend', 'jax') == 'numba' and config['algorithm'] != 'SCS':
        if config['algorithm'] == BILLIARD_METHOD:
            from .numba_billiard import NumbaBilliard
            return NumbaBilliard(mapping, config, seed, calibration)
        if config['algorithm'] == 'DCS-RWM':
            from .numba_rwm import NumbaRWM
            return NumbaRWM(mapping, config, seed, calibration)
        if config['algorithm'] == RWM_ROTATION_METHOD:
            from .numba_rwm_rotation import NumbaRWMRotation
            return NumbaRWMRotation(mapping, config, seed, calibration)
        if config['algorithm'] == ROTATION_METHOD:
            from .numba_rotation import NumbaRotation
            return NumbaRotation(mapping, config, seed, calibration)
        from .numba_radial import NumbaRadial
        return NumbaRadial(mapping, config, seed, calibration)
    return calibration


def _ball_tuning_metadata(history, scale, warmup):
    """Small, serializable pilot evidence; no retained samples enter selection."""
    if history.ndim != 3 or history.shape[1:] != (3, 5) or len(history) == 0:
        raise ValueError('Malformed ball tuning history')
    if not np.all(np.isfinite(history)) or int(history[:, :, 1].sum()) != warmup:
        raise ValueError('Ball tuning history does not account for its warmup budget')
    if not np.all(history[:, :, 4].sum(axis=1) == 1.):
        raise ValueError('Each ball tuning round must select exactly one candidate')
    chosen = int(np.argmax(history[-1, :, 4]))
    if not np.isclose(history[-1, chosen, 0], scale, rtol=1e-12, atol=0.):
        raise ValueError('Frozen step differs from the selected ball pilot step')
    names = ('lower_candidate_won_final_round', 'middle_candidate_won_final_round',
             'upper_candidate_won_final_round')
    status = names[chosen] if np.any(history[:, :, 2] > 0.) else 'no_positive_jump_observed'
    rows = [dict(round=r+1, candidate=c, scale=float(values[0]), iterations=int(values[1]),
                 ball_esjd=float(values[2]), acceptance=float(values[3]), selected=bool(values[4]))
            for r, trials in enumerate(history) for c, values in enumerate(trials)]
    return dict(tuning_history=rows, tuning_pilot_iterations=warmup,
                tuning_rounds=len(history), tuning_selected_esjd=float(history[-1, chosen, 2]),
                tuning_search_status=status,
                tuning_note='Local finite-budget pilot search of squared ball movement per transition; '
                            'not an optimum certificate or a radial-MSE criterion')


def _radial_tuning_metadata(history, scale, warmup, algorithm):
    """Record actual pilot chronology and distinguish it from historical ESJD."""
    from .radial_tuning import COARSE_FACTORS, REFINE_FACTORS, WINDOWS, SCORE_DEFINITION

    if history.shape != (WINDOWS, 7) or not np.all(np.isfinite(history)):
        raise ValueError('Malformed own-kernel radial tuning history')
    if int(history[:, 1].sum()) != warmup or not np.array_equal(np.sort(history[:, 5]), np.arange(WINDOWS)):
        raise ValueError('Radial tuning history does not account for its warmup budget and chronology')
    coarse, refined = history[:len(COARSE_FACTORS)], history[len(COARSE_FACTORS):]
    if not np.all(coarse[:, 6] == 1.) or not np.all(refined[:, 6] == 2.) or \
            coarse[:, 4].sum() != 1. or refined[:, 4].sum() != 1.:
        raise ValueError('Each radial tuning stage must select exactly one candidate')
    chosen = int(np.argmax(refined[:, 4]))
    if not np.isclose(refined[chosen, 0], scale, rtol=1e-12, atol=0.):
        raise ValueError('Frozen step differs from the selected radial pilot step')
    names = ('lower_candidate_won_refinement', 'middle_candidate_won_refinement',
             'upper_candidate_won_refinement')
    status = names[chosen] if np.any(history[:, 2] > 0.) else 'no_positive_radial_jump_observed'
    rows = [dict(stage='coarse' if stage_index == 1 else 'refinement', round=stage_index,
        candidate=c, trial_order=int(values[5])+1, scale=float(values[0]), iterations=int(values[1]),
        radial_score=float(values[2]), acceptance=float(values[3]), selected=bool(values[4]))
        for stage_index, trials in ((1, coarse), (2, refined)) for c, values in enumerate(trials)]
    return dict(radial_tuning_history=rows, tuning_pilot_iterations=warmup,
        radial_tuning_pilot_iterations=warmup, radial_tuning_rounds=2, radial_tuning_windows=WINDOWS,
        radial_tuning_selected_score=float(refined[chosen, 2]), radial_tuning_search_status=status,
        radial_tuning_score_definition=SCORE_DEFINITION,
        radial_tuning_coarse_factors=list(COARSE_FACTORS),
        radial_tuning_refinement_factors=list(REFINE_FACTORS), calibration_algorithm=algorithm,
        tuning_note='Each candidate uses this sampler\'s actual production transition and physical norm. '
            'The prespecified coarse/refinement search uses the full common warmup budget, with seeded '
            'randomized order within stages; all trials are discarded and the last pilot endpoint and '
            'selected step start production. Bounded log-radius movement is a proxy, not ESS, MSE, '
            'or an optimum certificate; no production adaptation.')


def _rotation_tuning_metadata(histories, scale, angular_scale, config):
    """Record both discarded pilot searches and their shared total budget."""
    warmup = config['warmup']
    bw_count, angle_count = warmup//2, warmup-warmup//2
    bw = _ball_tuning_metadata(np.asarray(histories[0]), scale, bw_count)
    angular = _ball_tuning_metadata(np.asarray(histories[1]), angular_scale, angle_count)
    rows = []
    for row in angular['tuning_history']:
        row = dict(row)
        row['angular_esjd'] = row.pop('ball_esjd')
        row['angle'] = float(2.*np.arctan(row['scale']))
        rows.append(row)
    bw.update(tuning_pilot_iterations=warmup, bw_tuning_pilot_iterations=bw_count,
        bw_warmup_iterations=bw_count, angular_warmup_iterations=angle_count,
        initial_angular_scale=float(config.get('initial_angular_scale', 1.)),
        angular_scale=angular_scale, angular_angle=float(2.*np.arctan(angular_scale)),
        angular_scale_ceiling=float(np.exp(15.)),
        angular_scale_ceiling_reached=bool(angular_scale >= np.exp(15.)*(1.-1e-10)),
        mixture_probability=.5, angular_tuning_history=rows,
        angular_tuning_pilot_iterations=angle_count,
        angular_tuning_rounds=angular['tuning_rounds'],
        angular_tuning_selected_esjd=angular['tuning_selected_esjd'],
        angular_tuning_search_status=angular['tuning_search_status'],
        tuning_note='Reflected-BW pilots use the first half of the common warmup budget; '
                    'angular pilots use the remainder and score movement of the unit direction after Mobius. '
                    'Both local-search steps freeze; the production mixture probability is fixed at 1/2.',
        angular_radius_policy='Preserve distance from the fixed map centre mu by rotating after Mobius',
        angular_acceptance_policy='Physical target-density ratio; radial Jacobian cancels on the fixed-radius sphere',
        iteration_policy='One independently selected reflected-BW or angular update per recorded iteration')
    if config.get('algorithm') == RWM_ROTATION_METHOD:
        dimension = int(scs_paper.target_spec(config)['dimension'])
        sigma = _initial_proposal_scale(config)
        bw.update(scale_parameter='Gaussian coordinate standard deviation',
            initial_scale_reference_bw_radius=float(config.get('initial_scale', .1)),
            initial_gaussian_sigma=sigma,
            gaussian_equivalent_bw_radius=float(scale*np.sqrt(dimension+2.)),
            reflected_warmup_iterations=bw_count, gaussian_warmup_iterations=bw_count,
            first_component='reflected_gaussian_rwm',
            tuning_note=bw['tuning_note'].replace('Reflected-BW', 'Reflected Gaussian-RWM'),
            iteration_policy='One independently selected reflected-Gaussian-RWM or angular update per recorded iteration')
    return bw


def _validate(config):
    if config['algorithm'] not in ('DCS-RWM', 'DCS-BW', 'DCS-BW-reflect', ROTATION_METHOD, RWM_ROTATION_METHOD, BILLIARD_METHOD, 'SCS'):
        raise ValueError('Unknown timed comparison method')
    if not np.isfinite(config['time_budget']) or config['time_budget'] <= 0:
        raise ValueError('time_budget must be positive and finite')
    for name in ('thin', 'warmup'):
        if isinstance(config[name], bool) or not isinstance(config[name], int):
            raise ValueError(f'{name} must be an integer')
    if config['warmup'] < 0 or config['thin'] < 1 or PREFIX_STRIDE % config['thin']:
        raise ValueError('Require nonnegative warmup and thin to be a positive divisor of 100')
    if config.get('dcs_backend', 'jax') not in ('jax', 'numba'):
        raise ValueError('dcs_backend must be jax or numba')
    if not isinstance(config.get('ess_enabled', False), bool):
        raise ValueError('ess_enabled must be a boolean')
    if not isinstance(config.get('coordinate_metrics', False), bool):
        raise ValueError('coordinate_metrics must be a boolean')
    if config.get('coordinate_metrics', False):
        value = config.get('coordinate_threshold')
        if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
            raise ValueError('coordinate_threshold must be finite when coordinate metrics are enabled')
        if int(scs_paper.target_spec(config)['dimension']) < 3:
            raise ValueError('Coordinate comparisons require at least three dimensions')


def _postburn(prefixes, norms, thin, threshold, coordinate_threshold=None):
    """Remove exactly N/3 transitions using stored every-100-step prefixes."""
    iterations = len(prefixes) * PREFIX_STRIDE
    if iterations == 0 or iterations % BATCH_SIZE:
        raise ValueError('Timed histories must contain complete 300-step batches')
    burn = iterations // 3
    totals = prefixes[-1]
    sums = totals - prefixes[burn // PREFIX_STRIDE - 1]
    count = iterations - burn
    retained_norms = norms[burn // thin:]
    if coordinate_threshold is None:
        traces = np.column_stack((retained_norms >= threshold, retained_norms))
    else:
        radii, x1, x2, x3 = retained_norms.T
        traces = np.column_stack((radii >= threshold, radii, x1, x1 > coordinate_threshold,
                                 x2, x2 < -coordinate_threshold, x3, x3 > coordinate_threshold))
    return traces, iterations, burn, count, totals, sums


def run_timed(mapping_dict, config, seed, before_sampling=None, after_sampling=None, diagnostic_lock=None):
    """Run with temporary scalar storage whose handles close before Windows cleanup."""
    with ExitStack() as resources:
        directory = (resources.enter_context(tempfile.TemporaryDirectory(prefix='dcs-ess-'))
                     if config.get('ess_enabled', False) else None)
        return _run_timed(mapping_dict, config, seed, before_sampling, after_sampling,
            diagnostic_lock, resources, Path(directory)/'norms.bin' if directory else None)


def _run_timed(mapping_dict, config, seed, before_sampling, after_sampling,
               diagnostic_lock, resources, ess_path):
    """Return post-burn radial traces and exact post-burn iteration summaries.

    ``time_budget`` is measured only after separate warmup/step calibration.
    Production kernels are frozen. The first third of actual timed iterations
    is discarded for every method. Completion is rounded up to a 300-step
    batch; the measured overshoot is always reported.
    """
    _validate(config)
    tuning_policy = resolve_step_tuning(config)
    if tuning_policy == RADIAL_TUNING:
        from .radial_tuning import STEP_POLICY as radial_step_policy
    scale_ceiling = resolve_step_scale_ceiling(config)
    recorded_ceiling = float(np.exp(15.) if scale_ceiling is None else scale_ceiling)
    ess_enabled = bool(config.get('ess_enabled', False))
    ess_setup_seconds = 0.
    if ess_enabled:
        from .functional_ess import prepare_ess
        setup_started = time.perf_counter()
        prepare_ess()
        ess_setup_seconds = time.perf_counter()-setup_started
    # Every iteration contributes to the point estimate, hence its ESS also
    # needs every iteration. Saved trace thinning remains a separate choice.
    backend_config = dict(config, thin=1) if ess_enabled else config
    backend = _backend(mapping_dict, backend_config, int(seed))
    native = config.get('dcs_backend') == 'numba' and config['algorithm'] != 'SCS'
    thin, threshold = config['thin'], float(config.get('threshold', 50.))
    coordinate_threshold = float(config['coordinate_threshold']) if config.get('coordinate_metrics', False) else None
    prefix_width = 9 if coordinate_threshold is not None else 3
    functionals = ['tail_probability', 'mean_norm']
    if coordinate_threshold is not None:
        functionals += ['mean_x1', 'x1_tail_probability', 'mean_x2', 'x2_left_tail_probability',
                        'mean_x3', 'x3_tail_probability']
    def trace_shape(count):
        return (count, 4) if coordinate_threshold is not None else (count,)
    capacity, batches = 1024, 0
    prefix_buffer = np.empty((capacity * 3, prefix_width), dtype=float)
    norm_buffer = None if ess_enabled else np.empty(trace_shape(capacity * BATCH_SIZE // thin), dtype=float)
    norm_output = resources.enter_context(ess_path.open('wb', buffering=1024*1024)) if ess_enabled else None
    totals = np.zeros(prefix_width, dtype=float)
    if before_sampling is not None:
        before_sampling()
    started, cpu_started = time.perf_counter(), time.process_time()
    while True:
        prefixes, norms = backend.batch()
        if batches == capacity:
            capacity *= 2
            larger = np.empty((capacity * 3, prefix_width), dtype=float)
            larger[:batches * 3] = prefix_buffer
            prefix_buffer = larger
            if not ess_enabled:
                larger = np.empty(trace_shape(capacity * BATCH_SIZE // thin), dtype=float)
                larger[:batches * BATCH_SIZE // thin] = norm_buffer
                norm_buffer = larger
        prefix_buffer[batches * 3:(batches + 1) * 3] = prefixes + totals
        totals = prefix_buffer[(batches + 1) * 3 - 1].copy()
        if ess_enabled:
            norm_output.write(np.ascontiguousarray(norms, dtype=np.float64).tobytes())
        else:
            start = batches * BATCH_SIZE // thin
            norm_buffer[start:start + BATCH_SIZE // thin] = norms
        batches += 1
        elapsed = time.perf_counter() - started
        if elapsed >= config['time_budget']:
            break
    if ess_enabled:
        # Capture and the final buffered write are charged identically to every
        # method. The diagnostic FFT itself starts after the sampling barrier.
        norm_output.flush()
        elapsed = time.perf_counter() - started
    cpu_seconds = time.process_time() - cpu_started
    if after_sampling is not None:
        after_sampling()
    prefixes = prefix_buffer[:batches * 3]
    if ess_enabled:
        norm_output.close()
        full_norms = np.memmap(ess_path, mode='r', dtype=np.float64, shape=trace_shape(batches*BATCH_SIZE))
        resources.callback(full_norms._mmap.close)
        norms = full_norms[thin-1::thin]
    else:
        norms = norm_buffer[:batches * BATCH_SIZE // thin]
    if not np.all(np.isfinite(prefixes)) or not np.all(np.isfinite(norms)):
        raise FloatingPointError('Nonfinite timed radial summaries')
    traces, iterations, burn, count, totals, sums = _postburn(
        prefixes, norms, thin, threshold, coordinate_threshold)
    estimate = sums[[0, 1, 3, 4, 5, 6, 7, 8] if coordinate_threshold is not None else [0, 1]] / count
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
        'initial_scale': _initial_proposal_scale(config), 'scale': float(backend.scale), 'final_scale': float(backend.scale),
        'frozen_scale': float(backend.scale), 'adaptation_during_sampling': False,
        'step_policy': (ROTATION_STEP_POLICY.replace('reflected-BW', 'reflected-Gaussian-RWM')
                        if config['algorithm'] == RWM_ROTATION_METHOD else
                        ROTATION_STEP_POLICY if tuning_policy == ROTATION_TUNING else
                        radial_step_policy if tuning_policy == RADIAL_TUNING else
                        BALL_STEP_POLICY if tuning_policy == BALL_TUNING else
                        'Prespecified DCS move: BW/reflected BW/billiard increment radius h, RWM coordinate sigma=h/sqrt(d+2); '
                        'same fixed step throughout discarded warmup and sampling' if tuning_policy == FIXED_DCS_TUNING else
                        'Supplied fixed SCS scale; unchanged author transition throughout discarded warmup and sampling' if tuning_policy == FIXED_SCS_TUNING else
                        'Author SCS fixed h=0.1; discarded warmup without adaptation' if tuning_policy == SCS_FIXED_TUNING else STEP_POLICY),
        'tuning_policy': tuning_policy,
        'target_acceptance': None if tuning_policy in (BALL_TUNING, RADIAL_TUNING, ROTATION_TUNING, SCS_FIXED_TUNING, FIXED_SCS_TUNING, FIXED_DCS_TUNING) else .234,
        'common_cpu_backend': 'mixed: optimized Numba DCS, author JAX SCS' if config.get('dcs_backend') == 'numba' else 'JAX',
        'backend': 'Numba' if native else 'JAX',
        'implementation_revision': config.get('implementation_revision', IMPLEMENTATION_REVISION),
        'boundary_policy': ('specular_ball_billiard_invariant_extra_chords' if config['algorithm'] == BILLIARD_METHOD else
                            'chord_reflection_or_fixed_radius_rotation' if config['algorithm'] in ROTATION_METHODS else
                            'reflect_along_proposal_chord' if config['algorithm'] == 'DCS-BW-reflect'
                            else 'author_cap_stepout' if config['algorithm'] == 'SCS' else 'direct_rejection'),
        'step_scale_ceiling': recorded_ceiling,
        'scale_ceiling_reached': bool(backend.scale >= recorded_ceiling*(1-1e-10)),
        'proposal_generator': ('50/50_reflected_isotropic_gaussian_or_tangent_rotation' if config['algorithm'] == RWM_ROTATION_METHOD else
                               '50/50_reflected_uniform_ball_or_tangent_rotation' if config['algorithm'] == ROTATION_METHOD else
                               'normal_d_plus_2_projection' if config['algorithm'] in ('DCS-BW', 'DCS-BW-reflect', BILLIARD_METHOD)
                               else 'author_tangent_normal' if config['algorithm'] == 'SCS'
                               else 'isotropic_normal'),
        'common_statistic_policy': ('All 300 post-MH states enter statistics; optimized DCS caches unchanged radii and accumulates without full-state output'
                                    if native else 'Full 300-state scans; physical projection and norm for every state, including rejection repeats'),
        'accepted': int(totals[2]), 'acceptance': float(totals[2] / iterations),
        'postburn_accepted': int(sums[2]), 'postburn_acceptance': float(sums[2] / count),
        'tail_count': int(sums[0]), 'norm_sum': float(sums[1]),
        'mean_all_iterations': estimate.tolist(), 'mean_post_burnin': estimate.tolist(),
        'estimates': dict(zip(functionals, estimate.tolist())), 'functionals': functionals,
        'all_timed_tail_count': int(totals[0]), 'all_timed_norm_sum': float(totals[1]),
        'burn_fraction': '1/3 of actual timed iterations after separate warmup',
        'estimator': 'all post-burn timed iterations, including rejection repeats; thinning affects saved traces only',
        'implementation': backend.implementation,
        'history_storage_bytes': int(prefix_buffer.nbytes + (0 if ess_enabled else norm_buffer.nbytes)),
        'target_skew_cdf': 'normal' if config.get('target', {}).get('kind', 'skew_t') == 'skew_t' else None,
        'checkpoints': [],
    }
    if coordinate_threshold is not None:
        record.update(coordinate_threshold=coordinate_threshold,
            x1_sum=float(sums[3]), x1_tail_count=int(sums[4]),
            x2_sum=float(sums[5]), x2_left_tail_count=int(sums[6]),
            x3_sum=float(sums[7]), x3_tail_count=int(sums[8]),
            all_timed_x1_sum=float(totals[3]), all_timed_x1_tail_count=int(totals[4]),
            all_timed_x2_sum=float(totals[5]), all_timed_x2_left_tail_count=int(totals[6]),
            all_timed_x3_sum=float(totals[7]), all_timed_x3_tail_count=int(totals[8]))
        record['common_statistic_policy'] += '; x1/x2/x3 means and right/left/right tails use every same post-MH state'
    record.update(getattr(backend, 'tuning_metadata', {}))
    if ess_enabled:
        from .functional_ess import radial_ess, DEFINITION
        print(f'{config["algorithm"]}, seed {seed}: sampling complete; calculating ESS', flush=True)
        wait_started = time.perf_counter()
        with diagnostic_lock if diagnostic_lock is not None else nullcontext():
            wait_seconds = time.perf_counter()-wait_started
            radii = full_norms[burn:] if coordinate_threshold is None else full_norms[burn:, 0]
            values, diagnostic_seconds = radial_ess(radii, threshold, elapsed,
                draw_stride=1, nu=float(scs_paper.target_spec(config).get('nu', 3.)))
        record.update(functional_ess=values, functional_ess_seconds=diagnostic_seconds,
            functional_ess_setup_seconds=ess_setup_seconds,
            functional_ess_wait_seconds=wait_seconds, functional_ess_definition=DEFINITION,
            functional_ess_series_policy='All post-burn-in iterations, including rejection repeats; stride 1. Saved traces may be thinned separately. Scalar capture and flush are inside sampling time; serialized FFT diagnostics are outside.',
            functional_ess_capture_bytes=int(full_norms.nbytes),
            functional_ess_draw_stride=1, functional_ess_saved_trace_stride=thin,
            functional_ess_serialized=diagnostic_lock is not None)
    if native:
        record['rng_note'] = 'Shared JAX calibration; independent domain-separated NumPy PCG64 production stream. Different backends preserve proposal laws, not bitwise production trajectories.'
        record['calibration_backend'] = 'JAX, identical to the shared-framework comparison'
        record['compilation_note'] = 'Recorded setup includes shared JAX calibration/scan compilation and native Numba compilation; all precede sampling.'
    if config['algorithm'] == 'SCS':
        from .scs_paper import UPSTREAM_COMMIT
        record['source_commit'] = UPSTREAM_COMMIT
        record['rng_note'] = 'SCS uses JAX PRNG; independent keys between warmup steps and production blocks preserve the transition law. Native DCS, when selected, uses a separate PCG64 production stream.'
        record['scs_implementation'] = 'Unmodified author rwm_bright_side_stepout; recorded external step calibration and radial recording'
    return traces, record
