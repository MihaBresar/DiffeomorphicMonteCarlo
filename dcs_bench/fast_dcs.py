"""Compiled original-style DCS-RWM/BW for the single skew-t comparison.

The proposal law and continuously smoothed dual averaging are unchanged.
Numba removes Python's per-transition overhead; the normal log CDF is SciPy's
actual double-precision routine, called through its exported Cython capsule.
Compilation is separate from warmup and production timing. No fast-math or
interior-ball truncation is used. The physical radius is cached on rejection.
"""
import ctypes
import time

import numpy as np
from numba import njit
from numba.extending import get_cython_function_address
from scipy.special import gammaln

from .transforms import BallMap


# SciPy 1.14.1's public Cython API has fused real/complex overloads. The second
# argument is Cython's dispatch flag, not another mathematical argument. This
# address is resolved in each process (including Windows spawn); no JIT cache
# serializes a process-local pointer.
_log_ndtr = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_int)(
    get_cython_function_address('scipy.special.cython_special', '__pyx_fuse_1log_ndtr'))


@njit(nogil=True)
def _evaluate(z, mu, delta, alpha, constants):
    """Return transformed log density and physical radius in one traversal."""
    d = len(z)
    beta, scale, nu, delta2, logscale, logdelta, normalizer = constants
    r2, dot = 0., 0.
    for j in range(d):
        r2 += z[j]*z[j]
        dot += z[j]*delta[j]
    if r2 >= 1. or not np.isfinite(r2):
        return -np.inf, np.nan
    denominator = 1.+2.*dot+delta2*r2
    w = np.empty(d)
    rho2 = 0.
    if delta2 > 0.:
        for j in range(d):
            w[j] = ((1.-delta2)*z[j]+(1.+2.*dot+r2)*delta[j])/denominator
            rho2 += w[j]*w[j]
        logmo = logdelta-d*np.log(denominator)
    else:
        for j in range(d):
            w[j] = z[j]
        rho2, logmo = r2, 0.
    if rho2 >= 1. or not np.isfinite(rho2):
        return -np.inf, np.nan
    logeps = 0. if rho2 == 0. else np.log(-np.expm1(beta*np.log(np.sqrt(rho2))))
    e = np.exp(logeps/beta)
    q, skew = 0., 0.
    for j in range(d):
        value = mu[j]*e+scale*w[j]
        q += value*value
        skew += alpha[j]*value
    s = nu*e*e+q
    lp = (normalizer+(nu/beta-1.)*logeps+logscale
          -(nu+d)/2.*np.log(s)+logmo
          +np.log(2.)+_log_ndtr(skew*np.sqrt((nu+d)/s), 0))
    if not np.isfinite(lp):
        return -np.inf, np.nan
    return lp, np.sqrt(q)/e


@njit(nogil=True)
def _transition(rng, z, lp, radius, scale, ball_walk, mu, delta, alpha, constants):
    """The same RNG calls and proposal law as MetropolisKernel.step."""
    d = len(z)
    noise = rng.normal(0., 1., d)
    proposal = np.empty(d)
    factor = scale
    if ball_walk:
        radius_draw = rng.random()
        length2 = 0.
        for j in range(d):
            length2 += noise[j]*noise[j]
        factor = scale*radius_draw**(1./d)/np.sqrt(length2)
    for j in range(d):
        proposal[j] = z[j]+factor*noise[j]
    new_lp, new_radius = _evaluate(proposal, mu, delta, alpha, constants)
    logratio = new_lp-lp
    probability = np.exp(min(0., logratio))
    accepted = -rng.exponential() < logratio
    if accepted:
        return proposal, new_lp, new_radius, probability, True
    return z, lp, radius, probability, False


@njit(nogil=True)
def _adapt(probability, iteration, hbar, logbar):
    eta = 1./(iteration+10.)
    hbar = (1.-eta)*hbar+eta*(.234-probability)
    logh = -np.sqrt(iteration)/.05*hbar
    weight = iteration**(-.75)
    logbar = weight*logh+(1.-weight)*logbar
    return np.exp(logbar), hbar, logbar


@njit(nogil=True)
def _run_block(rng, z, lp, radius, scale, hbar, logbar, offset, count, thin,
               threshold, ball_walk, mu, delta, alpha, constants):
    """Continue the original smoothed dual average across burn/production."""
    traces = np.empty((count//thin, 2))
    means = np.empty((count//thin, 2))
    accepted, tail_count, norm_sum, saved = 0, 0, 0., 0
    # Initial h=.1 implies log(10*h)=0. The original script updates the
    # smoothed scale after every transition, including retained iterations.
    for index in range(count):
        z, lp, radius, probability, did_accept = _transition(
            rng, z, lp, radius, scale, ball_walk, mu, delta, alpha, constants)
        accepted += int(did_accept)
        iteration = offset+index+1
        scale, hbar, logbar = _adapt(probability, iteration, hbar, logbar)
        tail = int(radius >= threshold)
        tail_count += tail
        norm_sum += radius
        if (index+1) % thin == 0:
            traces[saved, 0], traces[saved, 1] = tail, radius
            means[saved, 0], means[saved, 1] = tail_count/(index+1), norm_sum/(index+1)
            saved += 1
    return (z, lp, radius, scale, hbar, logbar, traces, means, accepted,
            tail_count, norm_sum)


def _inputs(mapping, config):
    mapping = BallMap(**mapping) if isinstance(mapping, dict) else mapping
    target = config.get('target', config)
    d = int(target.get('dimension', mapping.d))
    nu = float(target.get('nu', 3.))
    alpha_spec = target.get('alpha', {'0': 100., '1': -100.})
    if d != mapping.d or target.get('kind', 'skew_t') != 'skew_t':
        raise ValueError('Compiled DCS requires a matching skew-t target and BallMap')
    alpha = np.zeros(d)
    if isinstance(alpha_spec, dict):
        for index, value in alpha_spec.items():
            alpha[int(index)] = value
    else:
        alpha[:] = alpha_spec
    delta2 = float(mapping.delta @ mapping.delta)
    normalizer = (gammaln((nu+d)/2)-gammaln(nu/2)
                  +nu/2*np.log(nu)-d/2*np.log(np.pi))
    constants = np.array([mapping.beta, mapping.scale, nu, delta2,
                          d*np.log(mapping.scale), d*np.log1p(-delta2), normalizer])
    return mapping, np.ascontiguousarray(mapping.mu), np.ascontiguousarray(mapping.delta), alpha, constants


def run_chain(mapping_dict, config, seed, before_sampling=None):
    """Return thinned [tail, norm] traces and exact all-iteration estimates.

    `iterations` counts production transitions, in addition to `warmup`.
    `checkpoints` averages *all* transitions up to selected saved indices;
    thinning therefore never changes reported MSE or sample counts.
    """
    algorithm = config['algorithm']
    if algorithm not in ('DCS-RWM', 'DCS-BW'):
        raise ValueError(f'Compiled DCS does not implement {algorithm}')
    mapping, mu, delta, alpha, constants = _inputs(mapping_dict, config)
    warmup, iterations, thin = (int(config[k]) for k in ('warmup', 'iterations', 'thin'))
    if warmup < 0 or iterations < 1 or thin < 1 or thin > iterations:
        raise ValueError('Require nonnegative warmup and 1 <= thin <= iterations')
    threshold = float(config.get('threshold', 50.))
    z = np.ascontiguousarray(mapping.inverse(np.ones(mapping.d)))
    ball_walk = algorithm == 'DCS-BW'
    started = time.perf_counter()
    lp, radius = _evaluate(z, mu, delta, alpha, constants)
    # Zero transitions compile the exact signature without consuming the
    # actual chain's RNG or including compiled work in either sampling clock.
    _run_block(np.random.default_rng(0), z.copy(), lp, radius, .1, 0., 0.,
               0, 0, thin, threshold, ball_walk, mu, delta, alpha, constants)
    compile_seconds = time.perf_counter()-started
    if not np.isfinite(lp) or not np.isfinite(radius):
        raise ValueError('Nonfinite initial DCS density or radius')
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    warm = _run_block(rng, z, lp, radius, .1, 0., 0., 0, warmup,
                      max(1, warmup+1), threshold, ball_walk, mu, delta, alpha, constants)
    warmup_seconds = time.perf_counter()-started
    if before_sampling is not None:
        before_sampling()
    started = time.perf_counter()
    result = _run_block(rng, *warm[:6], warmup, iterations, thin,
                        threshold, ball_walk, mu, delta, alpha, constants)
    sampling_seconds = time.perf_counter()-started
    state, lp, radius, scale, hbar, logbar, traces, means, accepted, tail_count, norm_sum = result
    if not np.isfinite(norm_sum) or not np.all(np.isfinite(traces)):
        raise FloatingPointError('Nonfinite DCS radial summaries')
    record = {
        'algorithm': algorithm, 'seed': int(seed), 'iterations': iterations,
        'retained_draws': len(traces), 'thin': thin, 'warmup_iterations': warmup,
        'implementation': 'Numba CPU loop, SciPy log_ndtr, fastmath=False',
        'compile_seconds': compile_seconds, 'warmup_seconds': warmup_seconds,
        'sampling_seconds': sampling_seconds, 'seconds': sampling_seconds,
        'acceptance_rate': accepted/iterations, 'acceptance': accepted/iterations,
        'warmup_acceptance_rate': warm[8]/warmup if warmup else None,
        'final_scale': float(scale), 'initial_scale': .1,
        'adaptation_during_sampling': True, 'target_acceptance': .234,
        'adaptation': 'original continuous smoothed dual averaging',
        'tail_count': int(tail_count), 'norm_sum': float(norm_sum),
        'mean_all_iterations': [tail_count/iterations, norm_sum/iterations],
        'checkpoints': [
            {'iterations': (index+1)*thin, 'mean_all_iterations': means[index].tolist()}
            for index in sorted({min(len(means)-1, max(0, int(len(means)*fraction)-1))
                                 for fraction in (.05, .2, .5, 1.)})],
    }
    if record['checkpoints'][-1]['iterations'] != iterations:
        record['checkpoints'].append({'iterations': iterations,
                                      'mean_all_iterations': record['mean_all_iterations']})
    return traces, record
