"""Pilot-only map fitting and proposal-scale adaptation.

DCS map fitting uses its reverse KL objective on base draws. Current SCS
comparisons reuse a fixed cloud for each family and match cloud sizes and
update counts. SCS calls the pinned author routine. No fitted parameter
changes after sampling starts.
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


def validate_matched_vi(config):
    """Check the recorded work budget and batch policy of current comparisons."""
    from .scs_paper import author_vi_budget
    budget = author_vi_budget(config)
    if config.get('vi_batch_policy') != 'fixed':
        raise ValueError('Matched VI requires a fixed reference cloud for both methods')
    for field, value in budget.items():
        if config.get(field) != value:
            raise ValueError('Matched VI requires identical optimizer_steps and optimizer_samples for DCS and SCS')
    return budget


def fit_map(target, config, family, seed):
    start = time.perf_counter()
    if family not in ('ball', 'sphere'):
        raise ValueError('Map family must be ball or sphere')
    if family == 'sphere':
        if config.get('vi', 'on') == 'off' or config.get('sphere_map_policy') == 'vanilla':
            mapping = SphereMap(target.d, latitude=config.get('latitude', 1.1))
            return mapping, {'seconds': time.perf_counter()-start, 'steps': 0,
                'training_draws': 0, 'total_training_draws': 0, 'loss': [], 'seed': None,
                'vi_enabled': False, 'initial_scale': 1., 'scale_policy': 'unit',
                'map_policy': 'untuned', 'coordinate_clipping': False, 'cdf_clipping': False,
                'fitting_policy': 'No VI: R=1, centre=0, deformation=0'}
        from .scs_paper import author_vi_budget, fit as author_fit
        parameters, metadata = author_fit({'target': target.spec,
            'latitude': config.get('latitude', 1.1), **author_vi_budget(config)})
        return SphereMap(**parameters), metadata
    d, beta = target.d, config['beta']
    map_policy = validate_ball_map_policy(config)
    batch_policy = config.get('vi_batch_policy', 'fresh')
    if batch_policy not in ('fresh', 'fixed'):
        raise ValueError('vi_batch_policy must be fresh or fixed')
    # Posterior experiments use the paper's identity affine map, in the
    # canonical unconstrained coordinates documented in TARGETS.md.
    if target.kind in ('garch', 'eight_schools'):
        mapping = BallMap(d, beta)
        return mapping, {'seconds': time.perf_counter()-start, 'steps': 0, 'loss': [], 'map_policy': 'posterior-identity'}
    if map_policy == 'vanilla':
        # This ablation bypasses VI and radial-median initialization entirely.
        # Proposal-scale warmup is unchanged and is handled by the sampler.
        mapping = BallMap(d, beta)
        return mapping, {'seconds': time.perf_counter()-start, 'steps': 0, 'loss': [],
                         'map_policy': 'vanilla', 'scale_policy': 'unit',
                         'training_draws': 0, 'total_training_draws': 0,
                         'objective': None, 'seed': seed}
    import jax
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    rng = np.random.default_rng(seed)
    n = config['optimizer_samples']
    base = uniform_ball(rng, n, d)
    initial_scale = 1.
    if config.get('map_scale_initialization') == 'radial_median':
        if target.kind != 'skew_t':
            raise ValueError('Radial-median initialization requires the synthetic t target')
        # DCS avoids spending all Adam iterations traversing tens of log-scale
        # units when beta is very small (e.g. beta=.2 in d=50). SCS always uses
        # its author's unit initialization, in the separate branch above.
        radial_median = np.sqrt(d*f_distribution.ppf(.5, d, target.nu))
        radius = np.linalg.norm(base, axis=1)
        mapped_radius = radius/(-np.expm1(beta*np.log(radius)))**(1/beta)
        initial_scale = radial_median/np.median(mapped_radius)
    base = jnp.asarray(base)
    symmetric = target.kind == 'skew_t' and not np.any(target.alpha)
    limit = .999

    def unpack(p):
        mu = jnp.zeros(d) if symmetric else p[:d]
        raw = jnp.zeros(d) if symmetric else p[d:2*d]
        # Smooth at zero; maps R^d strictly inside the admissible ball.
        shift = limit*raw/jnp.sqrt(1+jnp.sum(raw*raw))
        return mu, shift, jnp.exp(p[-1])

    def objective(p, base):
        mu, shift, scale = unpack(p)
        w, lj = mobius(base, shift, jnp)
        logeps = jnp.log(-jnp.expm1(beta*jnp.log(jnp.linalg.norm(w, axis=1))))
        x = mu+scale*w/jnp.exp(logeps/beta)[:, None]
        lj = lj+d*jnp.log(scale)-(1+d/beta)*logeps
        return -jnp.mean(target.log_prob(x, 'jax')+lj)

    value_grad = jax.jit(jax.value_and_grad(objective))
    p = jnp.zeros(2*d+1).at[-1].set(np.log(initial_scale))
    m, v = jnp.zeros_like(p), jnp.zeros_like(p)
    history = []
    for i in range(1, config['optimizer_steps']+1):
        if i > 1 and batch_policy == 'fresh':
            base = jnp.asarray(uniform_ball(rng, n, d))
        loss, grad = value_grad(p, base)
        loss = float(loss)
        if not np.isfinite(loss) or not np.all(np.isfinite(grad)):
            raise FloatingPointError(f'{family} map fit became nonfinite at step {i}')
        # Gradient clipping concerns optimization only, never target evaluation.
        grad = grad*jnp.minimum(1., 10./jnp.maximum(jnp.linalg.norm(grad), 1e-12))
        m, v = .9*m+.1*grad, .999*v+.001*grad**2
        p = p-.01*(m/(1-.9**i))/(jnp.sqrt(v/(1-.999**i))+1e-8)
        history.append(loss)
    mu, shift, scale = unpack(p)
    mapping = BallMap(d, beta=beta, mu=np.asarray(mu), scale=float(scale), delta=np.asarray(shift))
    return mapping, {'seconds': time.perf_counter()-start, 'steps': config['optimizer_steps'],
                     'training_draws': n, 'seed': seed, 'loss': history,
                     'gradient_updates': config['optimizer_steps'],
                     'objective_point_evaluations': n*config['optimizer_steps'],
                     'unique_reference_draws': n*max(1, config['optimizer_steps']) if batch_policy == 'fresh' else n,
                     'initial_scale': initial_scale, 'scale_initialization': config.get('map_scale_initialization', 'unit'),
                     'scale_policy': 'VI-fitted', 'map_policy': map_policy,
                     'objective': 'reverse KL on independent base draws', 'batch_policy': batch_policy,
                     'total_training_draws': n*max(1, config['optimizer_steps']) if batch_policy == 'fresh' else n,
                     'symmetric_constraints': symmetric}


def validate_ball_map_policy(config):
    if config.get('ball_scale_policy', 'fit') != 'fit':
        raise ValueError('The fixed R=sqrt(d) option has been removed for new runs. '
                         'Choose ball_map_policy="mobius" or "vanilla"; saved results still support plot-only.')
    policy = config.get('ball_map_policy', 'mobius')
    if policy not in ('mobius', 'vanilla'):
        raise ValueError('ball_map_policy must be mobius or vanilla')
    return policy


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
