"""Pilot-only map fitting and proposal-scale adaptation.

Map fitting minimizes a Monte Carlo reverse KL objective on independent base
draws. No fitted parameter changes after retained sampling starts.
"""
import time
import numpy as np
from scipy.special import betainc, betaincinv
from scipy.stats import f as f_distribution
from .transforms import BallMap, SphereMap, mobius


def uniform_ball(rng, n, d):
    v = rng.normal(size=(n, d))
    return v/np.linalg.norm(v, axis=1, keepdims=True)*rng.random((n, 1))**(1/d)


def uniform_cap(rng, n, d, latitude):
    height = 2*betaincinv(d/2, d/2, rng.random(n)*betainc(d/2, d/2, latitude/2))
    v = rng.normal(size=(n, d))
    v *= np.sqrt(height*(2-height))[:, None]/np.linalg.norm(v, axis=1, keepdims=True)
    return np.column_stack([v, height-1])


def fit_map(target, config, family, seed):
    start = time.perf_counter()
    d, beta = target.d, config['beta']
    latitude = config.get('latitude', 1.1)
    # Posterior experiments use the paper's identity affine map, in the
    # canonical unconstrained coordinates documented in TARGETS.md.
    if target.kind in ('garch', 'eight_schools'):
        mapping = BallMap(d, beta)
        return mapping, {'seconds': time.perf_counter()-start, 'steps': 0, 'loss': []}
    import jax
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    rng = np.random.default_rng(seed)
    n = config['optimizer_samples']
    base = uniform_ball(rng, n, d) if family == 'ball' else uniform_cap(rng, n, d, latitude)
    initial_scale = 1.
    if config.get('map_scale_initialization') == 'radial_median':
        if target.kind != 'skew_t':
            raise ValueError('Radial-median initialization requires the synthetic t target')
        # Both projection families get the same known-target scale information.
        # This avoids spending all Adam iterations traversing tens of log-scale
        # units when beta is very small (e.g. beta=.2 in d=50).
        radial_median = np.sqrt(d*f_distribution.ppf(.5, d, target.nu))
        if family == 'ball':
            radius = np.linalg.norm(base, axis=1)
            mapped_radius = radius/(-np.expm1(beta*np.log(radius)))**(1/beta)
        else:
            mapped_radius = np.linalg.norm(base[:, :-1], axis=1)*latitude/(latitude-1-base[:, -1])
        initial_scale = radial_median/np.median(mapped_radius)
    base = jnp.asarray(base)
    symmetric = target.kind == 'skew_t' and not np.any(target.alpha)
    limit = .999 if family == 'ball' else .999*np.sqrt(1-(latitude-1)**2)

    def unpack(p):
        mu = jnp.zeros(d) if symmetric else p[:d]
        raw = jnp.zeros(d) if symmetric else p[d:2*d]
        # Smooth at zero; maps R^d strictly inside the admissible ball.
        shift = limit*raw/jnp.sqrt(1+jnp.sum(raw*raw))
        return mu, shift, jnp.exp(p[-1])

    def objective(p):
        mu, shift, scale = unpack(p)
        if family == 'ball':
            w, lj = mobius(base, shift, jnp)
            logeps = jnp.log(-jnp.expm1(beta*jnp.log(jnp.linalg.norm(w, axis=1))))
            x = mu+scale*w/jnp.exp(logeps/beta)[:, None]
            lj = lj+d*p[-1]-(1+d/beta)*logeps
        else:
            ell = base[:, -1]+1
            m = (latitude-ell)/latitude
            yhat = (base[:, :-1]-(ell/latitude)[:, None]*shift)/m[:, None]
            v = yhat-shift
            num = m*jnp.sum(v*v, axis=1)+v@shift+latitude-latitude**2*(1-m)
            lj = d*p[-1]+jnp.log(num)-d*jnp.log(m)-jnp.log(latitude)
            x = mu+scale*yhat
        return -jnp.mean(target.log_prob(x, 'jax')+lj)

    value_grad = jax.jit(jax.value_and_grad(objective))
    p = jnp.zeros(2*d+1).at[-1].set(np.log(initial_scale))
    m, v = jnp.zeros_like(p), jnp.zeros_like(p)
    history = []
    for i in range(1, config['optimizer_steps']+1):
        loss, grad = value_grad(p)
        loss = float(loss)
        if not np.isfinite(loss) or not np.all(np.isfinite(grad)):
            raise FloatingPointError(f'{family} map fit became nonfinite at step {i}')
        # Gradient clipping concerns optimization only, never target evaluation.
        grad = grad*jnp.minimum(1., 10./jnp.maximum(jnp.linalg.norm(grad), 1e-12))
        m, v = .9*m+.1*grad, .999*v+.001*grad**2
        p = p-.01*(m/(1-.9**i))/(jnp.sqrt(v/(1-.999**i))+1e-8)
        history.append(loss)
    mu, shift, scale = unpack(p)
    cls = BallMap if family == 'ball' else SphereMap
    kwargs = {'beta': beta, 'delta': np.asarray(shift)} if family == 'ball' else {'observer': np.asarray(shift), 'latitude': latitude}
    mapping = cls(d, mu=np.asarray(mu), scale=float(scale), **kwargs)
    return mapping, {'seconds': time.perf_counter()-start, 'steps': config['optimizer_steps'],
                     'training_draws': n, 'seed': seed, 'loss': history,
                     'initial_scale': initial_scale, 'scale_initialization': config.get('map_scale_initialization', 'unit'),
                     'objective': 'reverse KL on fixed independent base draws', 'symmetric_constraints': symmetric}


class DualAveraging:
    def __init__(self, initial, target=.234):
        self.mu, self.target = np.log(10*initial), target
        self.hbar, self.logbar, self.i = 0., np.log(initial), 0

    def update(self, acceptance):
        self.i += 1
        eta = 1/(self.i+10.)
        self.hbar = (1-eta)*self.hbar+eta*(self.target-acceptance)
        logh = self.mu-np.sqrt(self.i)/.05*self.hbar
        weight = self.i**(-.75)
        self.logbar = weight*logh+(1-weight)*self.logbar
        return float(np.exp(np.clip(logh, -30, 15)))

    def final(self):
        return float(np.exp(np.clip(self.logbar, -30, 15)))
