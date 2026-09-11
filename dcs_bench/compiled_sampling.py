"""Native CPU DCS kernels for the canonical experiment targets.

This backend preserves the generic benchmark protocol: raw dual averaging
during warmup, a frozen production scale, and full-chord hit-and-run slice
sampling. It is separate from fast_dcs's continuously adapted legacy study.
"""
import time

import numpy as np
from numba import njit
from scipy.special import gammaln

from .fast_dcs import _evaluate as _skew_ball_density, _log_ndtr
from .transforms import BallMap, SphereMap


KINDS = {'skew_t': 0, 'logistic': 1, 'eight_schools': 2, 'garch': 3}
ALGORITHMS = {'DCS-RWM': 0, 'DCS-BW': 1, 'DCS-HnR': 2, 'RWM-Gaussian': 0, 'RWM-stable': 3, 'SCS': 4}


@njit(nogil=True, error_model='numpy')
def _softplus(value):
    return max(value, 0.)+np.log1p(np.exp(-abs(value)))


@njit(nogil=True, error_model='numpy')
def _expit(value):
    if value >= 0.:
        return 1./(1.+np.exp(-value))
    e = np.exp(value)
    return e/(1.+e)


@njit(nogil=True, error_model='numpy')
def _forward(z, mu, delta, constants):
    d = len(z)
    beta, scale, nu, delta2, logscale, logdelta, normalizer = constants[:7]
    r2, dot = 0., 0.
    for j in range(d):
        r2 += z[j]*z[j]
        dot += z[j]*delta[j]
    x = np.empty(d)
    if r2 >= 1. or not np.isfinite(r2):
        x[:] = np.nan
        return x, -np.inf
    denominator = 1.+2.*dot+delta2*r2
    rho2 = 0.
    for j in range(d):
        x[j] = ((1.-delta2)*z[j]+(1.+2.*dot+r2)*delta[j])/denominator
        rho2 += x[j]*x[j]
    if rho2 >= 1. or not np.isfinite(rho2):
        x[:] = np.nan
        return x, -np.inf
    logeps = 0. if rho2 == 0. else np.log(-np.expm1(beta*np.log(np.sqrt(rho2))))
    e = np.exp(logeps/beta)
    for j in range(d):
        x[j] = mu[j]+scale*x[j]/e
    return x, logscale-(1.+d/beta)*logeps+logdelta-d*np.log(denominator)


@njit(nogil=True, error_model='numpy')
def _sphere_forward(z, mu, observer, constants):
    d, scale, latitude = len(mu), constants[1], constants[7]
    ell = z[-1]+1.
    if ell >= latitude:
        return np.full(d, np.nan), -np.inf
    m = (latitude-ell)/latitude
    yhat = (z[:-1]-(ell/latitude)*observer)/m
    v = yhat-observer
    numerator = m*np.dot(v, v)+np.dot(v, observer)+latitude-latitude*latitude*(1.-m)
    if numerator <= 0.:
        return np.full(d, np.nan), -np.inf
    return (mu+scale*yhat,
            d*np.log(scale)+np.log(numerator)-d*np.log(m)-np.log(latitude))


@njit(nogil=True, error_model='numpy')
def _cap_stepout(z, proposed, latitude):
    if proposed[-1] < latitude-1.:
        return proposed
    dot = min(1., max(-1., np.dot(z, proposed)))
    angle = np.arccos(dot)
    tangent = proposed-dot*z
    tangent /= np.linalg.norm(tangent)
    radius = np.hypot(z[-1], tangent[-1])
    gamma = np.arccos(min(1., max(-1., (latitude-1.)/radius)))
    phi = np.arccos(min(1., max(-1., z[-1]/radius)))
    k = np.floor((phi+gamma)/angle)+1.
    result = np.cos(k*angle)*z+np.sin(k*angle)*tangent
    return result/np.linalg.norm(result)


@njit(nogil=True, error_model='numpy')
def _target_density(x, kind, alpha, parameters, data):
    d = len(x)
    nu, prior_scale, normalizer, sigma1, noncentered = parameters
    if kind == 0:
        r2, dot = 0., 0.
        for j in range(d):
            r2 += x[j]*x[j]
            dot += x[j]*alpha[j]
        return (normalizer-(nu+d)/2.*np.log1p(r2/nu)+np.log(2.)
                +_log_ndtr(dot*np.sqrt((nu+d)/(nu+r2)), 0))
    if kind == 1:
        lp = normalizer
        for j in range(d):
            lp -= (nu+1.)/2.*np.log1p((x[j]/prior_scale)**2/nu)
        # The last column stores the observed outcome sign, avoiding a
        # separate target-dependent argument or Python callback per proposal.
        for i in range(len(data)):
            eta = 0.
            for j in range(d):
                eta += data[i, j]*x[j]
            lp -= _softplus(-data[i, d]*eta)
        return lp
    if kind == 2:
        tau, mu = np.exp(x[9]), x[8]
        lp = -.5*(mu/5.)**2-_softplus(2.*(x[9]-np.log(5.)))+x[9]
        for i in range(8):
            theta = mu+tau*x[i] if noncentered else x[i]
            lp -= .5*((data[i, 0]-theta)/data[i, 1])**2
            if noncentered:
                lp -= .5*x[i]*x[i]
            else:
                lp -= x[9]+.5*((theta-mu)/tau)**2
        return lp
    # GARCH's physical alpha1,beta1 live in the triangular stationarity
    # region; retain both logistic-transform factors in the Jacobian.
    mu, a0 = x[0], np.exp(x[1])
    a1, b1 = _expit(x[2]), _expit(-x[2])*_expit(x[3])
    lp = x[1]-_softplus(-x[2])-2.*_softplus(x[2])-_softplus(-x[3])-_softplus(x[3])
    h = sigma1*sigma1
    for i in range(len(data)):
        if i:
            h = a0+a1*(data[i-1, 0]-mu)**2+b1*h
        lp -= .5*(np.log(h)+(data[i, 0]-mu)**2/h)
    return lp


@njit(nogil=True, error_model='numpy')
def _density(z, kind, mapped, mu, delta, alpha, constants, parameters, data):
    if mapped == 1 and kind == 0:
        # Preserve algebraic cancellation for super-heavy tails and beta=.2.
        return _skew_ball_density(z, mu, delta, alpha, constants[:7])[0]
    if mapped == 2:
        x, logjac = _sphere_forward(z, mu, delta, constants)
        if not np.isfinite(logjac):
            return -np.inf
    elif mapped == 1:
        x, logjac = _forward(z, mu, delta, constants)
        if not np.isfinite(logjac):
            return -np.inf
    else:
        x, logjac = z, 0.
    lp = _target_density(x, kind, alpha, parameters, data)+logjac
    return lp if np.isfinite(lp) else -np.inf


@njit(nogil=True, error_model='numpy')
def _physical(z, kind, mapped, mu, delta, constants, parameters):
    if mapped == 2:
        x = _sphere_forward(z, mu, delta, constants)[0]
    elif mapped == 1:
        x = _forward(z, mu, delta, constants)[0]
    else:
        x = z.copy()
    if kind == 2:
        tau = np.exp(x[9])
        if parameters[4]:
            for i in range(8):
                x[i] = x[8]+tau*x[i]
        x[9] = tau
    elif kind == 3:
        x[1] = np.exp(x[1])
        x[3] = _expit(-x[2])*_expit(x[3])
        x[2] = _expit(x[2])
    return x


@njit(nogil=True, error_model='numpy')
def _transition(rng, z, lp, scale, algorithm, kind, mapped, mu, delta, alpha,
                constants, parameters, data):
    d = len(z)
    if algorithm != 3:
        noise = rng.normal(0., 1., d)
    if algorithm == 2:
        # Exactly the generic kernel's full-chord slice and shrinkage rule.
        noise /= np.linalg.norm(noise)
        dot = np.dot(z, noise)
        disc = np.sqrt(dot*dot+1.-np.dot(z, z))
        left, right = -dot-disc, -dot+disc
        level = lp-rng.exponential()
        for evaluation in range(1, 10001):
            distance = rng.uniform(left, right)
            proposal = z+distance*noise
            new_lp = _density(proposal, kind, mapped, mu, delta, alpha, constants, parameters, data)
            if new_lp >= level:
                return proposal, new_lp, 1., True, evaluation
            if distance < 0.:
                left = distance
            else:
                right = distance
        raise RuntimeError('Slice bracket failed to accept after 10000 evaluations')
    if algorithm == 3:
        u = rng.uniform(-np.pi/2., np.pi/2., d)
        w = rng.exponential(1., d)
        # Same independent alpha=.5 CMS stable variates as the Python baseline.
        stable = np.sin(.5*u)/np.cos(u)**2*(np.cos(.5*u)/w)
        proposal = z+scale*stable
    elif algorithm == 4:
        noise *= scale
        noise -= z*np.dot(z, noise)
        proposal = z+noise
        proposal /= np.linalg.norm(proposal)
        proposal = _cap_stepout(z, proposal, constants[7])
    elif algorithm == 1:
        proposal = z+scale*rng.random()**(1./d)*noise/np.linalg.norm(noise)
    else:
        proposal = z+scale*noise
    new_lp = _density(proposal, kind, mapped, mu, delta, alpha, constants, parameters, data)
    logratio = new_lp-lp
    probability = np.exp(min(0., logratio))
    accepted = -rng.exponential() < logratio
    if accepted:
        return proposal, new_lp, probability, True, 1
    return z, lp, probability, False, 1


@njit(nogil=True, error_model='numpy')
def _block(rng, z, lp, scale, hbar, logbar, count, offset, thin, adapt,
           keep, initial_scale, algorithm, kind, mapped, mu, delta, alpha,
           constants, parameters, data):
    saved_count = (offset+count)//thin-offset//thin if keep else 0
    draws = np.empty((saved_count, len(mu)))
    accepted, evaluations, saved = 0, 0, 0
    for index in range(count):
        z, lp, probability, did_accept, used = _transition(
            rng, z, lp, scale, algorithm, kind, mapped, mu, delta, alpha, constants, parameters, data)
        accepted += int(did_accept)
        evaluations += used
        iteration = offset+index+1
        if adapt and algorithm != 2:
            eta = 1./(iteration+10.)
            hbar = (1.-eta)*hbar+eta*(.234-probability)
            logh = np.log(10.*initial_scale)-np.sqrt(iteration)/.05*hbar
            weight = iteration**(-.75)
            logbar = weight*logh+(1.-weight)*logbar
            scale = np.exp(min(15., max(-30., logh)))
        if keep and iteration % thin == 0:
            draws[saved] = _physical(z, kind, mapped, mu, delta, constants, parameters)
            saved += 1
    return z, lp, scale, hbar, logbar, accepted, evaluations, draws


def _prepare(target, mapping):
    if target.kind not in KINDS:
        raise ValueError(f'No canonical compiled density for target {target.kind!r}')
    if mapping is not None and not isinstance(mapping, (BallMap, SphereMap)):
        raise TypeError('Compiled sampling requires BallMap, SphereMap or no projection')
    d, kind = target.d, KINDS[target.kind]
    mapped = 2 if isinstance(mapping, SphereMap) else 1 if mapping is not None else 0
    mapping = mapping if mapped else BallMap(d)
    if mapping.d != d:
        raise ValueError('Target and map dimensions differ')
    mu, delta = np.ascontiguousarray(mapping.mu), np.ascontiguousarray(mapping.observer if mapped == 2 else mapping.delta)
    alpha, parameters = np.zeros(d), np.zeros(5)
    data = np.empty((0, 0))
    if kind == 0:
        nu = target.nu
        alpha[:] = target.alpha
        parameters[:3] = nu, 1., gammaln((nu+d)/2.)-gammaln(nu/2.)-d/2.*np.log(nu*np.pi)
    elif kind == 1:
        nu, prior_scale = target.nu, target.spec['prior_scale']
        parameters[:3] = (nu, prior_scale,
            d*(gammaln((nu+1.)/2.)-gammaln(nu/2.)-np.log(prior_scale*np.sqrt(nu*np.pi))))
        data = np.column_stack((target.design, 2.*target.y-1.))
    elif kind == 2:
        data = np.column_stack((target.y, target.sigma))
        parameters[4] = target.spec.get('parameterization') == 'noncentered'
    else:
        data = target.y[:, None].copy()
        parameters[3] = target.sigma1
    delta2 = float(delta@delta)
    nu = target.nu if kind == 0 else 1.
    normalizer = gammaln((nu+d)/2.)-gammaln(nu/2.)+nu/2.*np.log(nu)-d/2.*np.log(np.pi)
    constants = np.array([getattr(mapping, 'beta', 2.), mapping.scale, nu, delta2,
                          d*np.log(mapping.scale), d*np.log1p(-delta2), normalizer,
                          getattr(mapping, 'latitude', 1.1)])
    initial = target.initial()
    z = mapping.inverse(initial) if mapped else initial
    context = (kind, mapped, mu, delta, alpha, constants, parameters, np.ascontiguousarray(data))
    return np.ascontiguousarray(z, dtype=float), context


class CompiledRunner:
    """Compile before a worker joins the sampling start barrier."""
    def __init__(self, target, algorithm, mapping, config):
        if algorithm not in ALGORITHMS:
            raise ValueError(f'No compiled transition for {algorithm}')
        if algorithm.startswith('DCS') and not isinstance(mapping, BallMap):
            raise ValueError('A DCS proposal requires a BallMap')
        if algorithm == 'SCS' and not isinstance(mapping, SphereMap):
            raise ValueError('SCS requires a SphereMap')
        if algorithm.startswith('RWM-') and mapping is not None:
            raise ValueError('Ordinary random walks require no projection')
        self.target, self.algorithm, self.config = target, algorithm, dict(config)
        self.code = ALGORITHMS[algorithm]
        self.initial_scale = float(config.get('initial_scale', .1))
        self.warmup, self.thin = int(config['warmup']), int(config['thin'])
        self.iterations = int(config.get('iterations', 0))
        self.batch_size = int(config.get('compiled_batch_size', 256))
        if self.warmup < 0 or self.thin < 1 or self.batch_size < 1 or self.batch_size > 4096:
            raise ValueError('Require nonnegative warmup, positive thin, and compiled_batch_size in [1,4096]')
        if not np.isfinite(self.initial_scale) or self.initial_scale <= 0.:
            raise ValueError('Initial proposal scale must be positive and finite')
        start = time.perf_counter()
        self.z, self.context = _prepare(target, mapping)
        self.lp = _density(self.z, *self.context)
        if not np.isfinite(self.lp):
            raise ValueError(f'Nonfinite initial density for {algorithm}')
        self._block(np.random.default_rng(0), self.z.copy(), self.lp,
                    self.initial_scale, 0., np.log(self.initial_scale), 0, 0, False, False)
        self.compile_seconds = time.perf_counter()-start

    def _block(self, rng, z, lp, scale, hbar, logbar, count, offset, adapt, keep):
        return _block(rng, z, lp, scale, hbar, logbar, count, offset, self.thin,
                      adapt, keep, self.initial_scale, self.code, *self.context)

    def run(self, seed, seconds=None):
        if seconds is not None and (not np.isfinite(seconds) or seconds <= 0.):
            raise ValueError('Sampling time must be positive and finite')
        if seconds is None and self.iterations//self.thin < 4:
            raise ValueError('Need at least four retained draws; increase iterations or reduce thin')
        rng = np.random.default_rng(seed)
        start = time.perf_counter()
        result = self._block(rng, self.z.copy(), self.lp, self.initial_scale, 0.,
                             np.log(self.initial_scale), self.warmup, 0, True, False)
        z, lp, scale, hbar, logbar, warm_accept, warm_evaluations, _ = result
        if self.code != 2:
            scale = float(np.exp(np.clip(logbar, -30., 15.)))
        warmup_seconds = time.perf_counter()-start
        draws, iterations, accepted, evaluations = [], 0, 0, 0
        start = time.perf_counter()
        while True:
            if seconds is None:
                count = min(self.batch_size, self.iterations-iterations)
                if count <= 0:
                    break
            else:
                elapsed = time.perf_counter()-start
                if elapsed >= seconds and iterations >= 4*self.thin:
                    break
                count = self.batch_size
            result = self._block(rng, z, lp, scale, hbar, logbar, count, iterations, False, True)
            z, lp, _, hbar, logbar, batch_accept, batch_evaluations, batch_draws = result
            iterations += count
            accepted += batch_accept
            evaluations += batch_evaluations
            draws.append(batch_draws)
        elapsed = time.perf_counter()-start
        draws = np.concatenate(draws)
        if len(draws) < 4 or not np.all(np.isfinite(draws)):
            raise FloatingPointError('Insufficient or nonfinite retained compiled draws')
        return draws, {
            'algorithm': self.algorithm, 'seed': int(seed), 'iterations': iterations,
            'implementation': 'Numba CPU; canonical density; fastmath=False',
            'retained_draws': len(draws), 'thin': self.thin,
            'warmup_iterations': self.warmup, 'warmup_seconds': warmup_seconds,
            'sampling_seconds': elapsed, 'compile_seconds': self.compile_seconds,
            'requested_seconds': seconds,
            'budget_overshoot_seconds': max(0., elapsed-seconds) if seconds is not None else 0.,
            'timed_batch_size': self.batch_size if seconds is not None else None,
            'minimum_retained_draws': 4, 'acceptance_rate': accepted/iterations,
            'frozen_scale': scale, 'log_density_evaluations': evaluations,
            'warmup_log_density_evaluations': warm_evaluations+1,
            'warmup_acceptance_rate': warm_accept/self.warmup if self.warmup else None,
            'adaptation_during_sampling': False,
        }


def run_compiled(target, algorithm, mapping, config, seed, seconds=None):
    return CompiledRunner(target, algorithm, mapping, config).run(seed, seconds)
