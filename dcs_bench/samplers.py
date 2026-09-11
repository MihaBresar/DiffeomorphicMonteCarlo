"""Frozen production kernels; adaptation and compilation have separate clocks."""
import time
import numpy as np
from .tuning import DualAveraging


def stable_noise(rng, d, alpha=.5):
    """Independent symmetric alpha-stable coordinates (CMS construction)."""
    u = rng.uniform(-np.pi/2, np.pi/2, d)
    w = rng.exponential(size=d)
    return np.sin(alpha*u)/np.cos(u)**(1/alpha)*(np.cos((1-alpha)*u)/w)**((1-alpha)/alpha)


def cap_stepout(z, proposed, latitude):
    """SCS great-circle step-out, in coordinates centered at the sphere origin.

    Matches stepout_dark_side in liusf15/sub-cauchy-sampler. Clipping here
    only protects inverse trigonometric arguments against roundoff.
    """
    if proposed[-1] < latitude-1:
        return proposed
    dot = np.clip(np.dot(z, proposed), -1., 1.)
    angle = np.arccos(dot)
    tangent = proposed-dot*z
    tangent /= np.linalg.norm(tangent)
    radius = np.hypot(z[-1], tangent[-1])
    gamma = np.arccos(np.clip((latitude-1)/radius, -1., 1.))
    phi = np.arccos(np.clip(z[-1]/radius, -1., 1.))
    k = np.floor((phi+gamma)/angle)+1
    result = np.cos(k*angle)*z+np.sin(k*angle)*tangent
    return result/np.linalg.norm(result)


class MetropolisKernel:
    def __init__(self, target, algorithm, mapping=None, seed=0):
        self.target, self.algorithm, self.mapping = target, algorithm, mapping
        self.rng = np.random.default_rng(seed)
        self.state = target.initial() if mapping is None else mapping.inverse(target.initial())
        self.evaluations = 0
        self.lp = self.evaluate(self.state)
        if not np.isfinite(self.lp):
            raise ValueError(f'Nonfinite initial density for {algorithm}')

    def evaluate(self, x):
        self.evaluations += 1
        with np.errstate(over='ignore', invalid='ignore', divide='ignore', under='ignore'):
            value = self.target.log_prob(x) if self.mapping is None else self.mapping.log_density(self.target, x)
        return float(value) if np.isfinite(value) else -np.inf

    def draw(self):
        x = self.state if self.mapping is None else self.mapping.forward(self.state)[0]
        return np.asarray(self.target.physical(x))

    def step(self, scale):
        rng, x, d = self.rng, self.state, self.target.d
        if self.algorithm == 'DCS-HnR':
            direction = rng.normal(size=d)
            direction /= np.linalg.norm(direction)
            dot = x@direction
            disc = np.sqrt(dot*dot+1-x@x)
            left, right = -dot-disc, -dot+disc
            level = self.lp-rng.exponential()
            for _ in range(10000):
                distance = rng.uniform(left, right)
                proposal = x+distance*direction
                lp = self.evaluate(proposal)
                if lp >= level:
                    self.state, self.lp = proposal, lp
                    return 1., True
                if distance < 0:
                    left = distance
                else:
                    right = distance
            raise RuntimeError('Slice bracket failed to accept after 10000 evaluations')
        if self.algorithm == 'DCS-BW':
            noise = rng.normal(size=d)
            proposal = x+scale*rng.random()**(1/d)*noise/np.linalg.norm(noise)
        elif self.algorithm in ('DCS-RWM', 'RWM-Gaussian'):
            proposal = x+scale*rng.normal(size=d)
        elif self.algorithm == 'RWM-stable':
            proposal = x+scale*stable_noise(rng, d)
        elif self.algorithm == 'SCS':
            noise = rng.normal(size=d+1)*scale
            noise -= x*np.dot(x, noise)
            proposal = x+noise
            proposal /= np.linalg.norm(proposal)
            proposal = cap_stepout(x, proposal, self.mapping.latitude)
        else:
            raise ValueError(self.algorithm)
        lp = self.evaluate(proposal)
        logratio = lp-self.lp
        probability = np.exp(min(0., logratio))
        accepted = -rng.exponential() < logratio
        if accepted:
            self.state, self.lp = proposal, lp
        return probability, accepted


def run_metropolis(target, algorithm, mapping, config, seed, seconds=None):
    kernel = MetropolisKernel(target, algorithm, mapping, seed)
    scale = config.get('initial_scale', .1)
    adapter = DualAveraging(scale)
    start = time.perf_counter()
    for _ in range(config['warmup']):
        probability, _ = kernel.step(scale)
        if algorithm != 'DCS-HnR':
            scale = adapter.update(probability)
    if algorithm != 'DCS-HnR':
        scale = adapter.final()
    warmup_seconds = time.perf_counter()-start
    warmup_evaluations = kernel.evaluations
    draws, accepted, iterations = [], 0, 0
    start = time.perf_counter()
    while True:
        if seconds is None:
            if iterations >= config['iterations']:
                break
        elif time.perf_counter()-start >= seconds:
            break
        _, did_accept = kernel.step(scale)
        accepted += did_accept
        iterations += 1
        if iterations % config['thin'] == 0:
            draws.append(kernel.draw().copy())
    elapsed = time.perf_counter()-start
    if len(draws) < 4:
        raise RuntimeError(f'{algorithm} retained only {len(draws)} draws; increase budget or reduce thinning')
    draws = np.asarray(draws)
    if not np.all(np.isfinite(draws)):
        raise FloatingPointError('Nonfinite retained draws')
    return draws, {'algorithm': algorithm, 'seed': seed, 'iterations': iterations,
                   'retained_draws': len(draws), 'thin': config['thin'], 'warmup_iterations': config['warmup'],
                   'warmup_seconds': warmup_seconds, 'sampling_seconds': elapsed,
                   'requested_seconds': seconds, 'budget_overshoot_seconds': max(0., elapsed-seconds) if seconds else 0.,
                   'acceptance_rate': accepted/iterations, 'frozen_scale': scale,
                   'log_density_evaluations': kernel.evaluations-warmup_evaluations,
                   'warmup_log_density_evaluations': warmup_evaluations, 'adaptation_during_sampling': False}


class BlackJAXRunner:
    """JIT scan batches amortize Python overhead; synchronize every timed batch."""
    def __init__(self, target, config):
        import jax
        import jax.numpy as jnp
        import blackjax
        jax.config.update('jax_enable_x64', True)
        self.target, self.config = target, config
        logdensity = lambda x: target.log_prob(x, 'jax')
        adaptation = blackjax.window_adaptation(blackjax.nuts, logdensity,
            target_acceptance_rate=config['nuts_target_accept'],
            adaptation_info_fn=lambda state, info, adapt: (info.is_divergent, info.num_integration_steps),
            max_num_doublings=config['nuts_max_depth'])
        warm = jax.jit(lambda key, x: adaptation.run(key, x, num_steps=config['warmup']))
        kernel = blackjax.nuts.build_kernel()
        batch_size = config['nuts_batch_size']

        def batch(key, state, step_size, inverse_mass_matrix):
            def step(carry, key):
                state, info = kernel(key, carry, logdensity, step_size, inverse_mass_matrix,
                                     max_num_doublings=config['nuts_max_depth'])
                return state, (state.position, info.acceptance_rate, info.is_divergent,
                               info.num_integration_steps, info.num_trajectory_expansions)
            return jax.lax.scan(step, state, jax.random.split(key, batch_size))

        start = time.perf_counter()
        key = jax.random.PRNGKey(0)
        x = jnp.asarray(target.initial())
        self.warm = warm.lower(key, x).compile()
        dummy = blackjax.nuts.init(x, logdensity)
        self.batch = jax.jit(batch).lower(key, dummy, jnp.asarray(.1), jnp.ones(target.d)).compile()
        self.compile_seconds = time.perf_counter()-start

    def run(self, seed, seconds=None, iterations=None):
        import jax
        import jax.numpy as jnp
        c = self.config
        key, pilot = jax.random.split(jax.random.PRNGKey(seed))
        start = time.perf_counter()
        (state, parameters), winfo = jax.block_until_ready(self.warm(pilot, jnp.asarray(self.target.initial())))
        warmup_seconds = time.perf_counter()-start
        arrays = [[] for _ in range(5)]
        count = 0
        start = time.perf_counter()
        while (count < iterations if seconds is None else time.perf_counter()-start < seconds):
            key, drawkey = jax.random.split(key)
            state, info = jax.block_until_ready(self.batch(drawkey, state, parameters['step_size'], parameters['inverse_mass_matrix']))
            for buf, value in zip(arrays, info):
                buf.append(np.asarray(value))
            count += c['nuts_batch_size']
        elapsed = time.perf_counter()-start
        raw, acc, div, work, depth = [np.concatenate(a) for a in arrays]
        if iterations is not None:
            raw, acc, div, work, depth = [a[:iterations] for a in (raw, acc, div, work, depth)]
        draws = np.asarray(self.target.physical(raw[c['thin']-1::c['thin']]))
        if len(draws) < 4 or not np.all(np.isfinite(draws)):
            raise RuntimeError('Insufficient or nonfinite NUTS draws')
        return draws, {'algorithm': 'NUTS', 'implementation': 'BlackJAX', 'seed': seed,
            'iterations': len(raw), 'computed_iterations': count, 'retained_draws': len(draws), 'thin': c['thin'],
            'warmup_iterations': c['warmup'], 'warmup_seconds': warmup_seconds,
            'sampling_seconds': elapsed, 'requested_seconds': seconds,
            'budget_overshoot_seconds': max(0., elapsed-seconds) if seconds else 0.,
            'acceptance_rate': float(acc.mean()), 'divergences': int(div.sum()),
            'max_depth_hits': int((depth >= c['nuts_max_depth']).sum()),
            'gradient_evaluations': int(work.sum()), 'warmup_divergences': int(np.asarray(winfo[0]).sum()),
            'frozen_scale': float(parameters['step_size']), 'inverse_mass_matrix': np.asarray(parameters['inverse_mass_matrix']).tolist(),
            'adaptation_during_sampling': False}


class NumPyroRunner:
    """NumPyro NUTS with separately compiled warmup and production loops.

    Uses the public kernel init/sample methods. The same canonical potential
    is used by DCS. Lowering/compilation occurs once per scenario, not per chain.
    """
    def __init__(self, target, config):
        import jax
        import jax.numpy as jnp
        from numpyro.infer import NUTS
        jax.config.update('jax_enable_x64', True)
        self.target, self.config = target, config
        kernel = NUTS(potential_fn=lambda x: -target.log_prob(x, 'jax'),
                      target_accept_prob=config['nuts_target_accept'], max_tree_depth=config['nuts_max_depth'])
        start = time.perf_counter()
        initialize = jax.jit(lambda key: kernel.init(key, config['warmup'], init_params=jnp.asarray(target.initial())))
        self.initialize = initialize.lower(jax.random.PRNGKey(0)).compile()
        dummy = self.initialize(jax.random.PRNGKey(0))
        def one_step(state):
            return kernel.sample(state, (), {})
        warm = jax.jit(lambda state: jax.lax.fori_loop(0, config['warmup'], lambda i, s: one_step(s), state))
        self.warm = warm.lower(dummy).compile()
        def production(state):
            def step(state, unused):
                state = one_step(state)
                return state, (state.z, state.accept_prob, state.diverging, state.num_steps)
            return jax.lax.scan(step, state, None, length=config['nuts_samples'])
        self.production = jax.jit(production).lower(dummy).compile()
        self.compile_seconds = time.perf_counter()-start

    def run(self, seed):
        import jax
        c = self.config
        start = time.perf_counter()
        state = jax.block_until_ready(self.warm(self.initialize(jax.random.PRNGKey(seed))))
        warmup_seconds = time.perf_counter()-start
        frozen_scale = float(state.adapt_state.step_size)
        start = time.perf_counter()
        final_state, result = jax.block_until_ready(self.production(state))
        raw, acc, div, work = [np.asarray(a) for a in result]
        elapsed = time.perf_counter()-start
        if float(final_state.adapt_state.step_size) != frozen_scale:
            raise RuntimeError('NUTS adapted during production')
        draws = np.asarray(self.target.physical(raw[c['thin']-1::c['thin']]))
        if len(draws) < 4 or not np.all(np.isfinite(draws)):
            raise RuntimeError('Insufficient or nonfinite NumPyro draws')
        return draws, {'algorithm': 'NUTS', 'implementation': 'NumPyro', 'seed': seed,
            'iterations': len(raw), 'retained_draws': len(draws), 'thin': c['thin'],
            'warmup_iterations': c['warmup'], 'warmup_seconds': warmup_seconds,
            'sampling_seconds': elapsed, 'acceptance_rate': float(acc.mean()),
            'divergences': int(div.sum()),
            'max_depth_hits': int((work >= 2**c['nuts_max_depth']-1).sum()),
            'max_depth_detection': 'num_steps >= 2**max_tree_depth - 1 (full tree saturation)',
            'gradient_evaluations': int(work.sum()), 'frozen_scale': frozen_scale,
            'inverse_mass_matrix': np.asarray(final_state.adapt_state.inverse_mass_matrix).tolist(),
            'adaptation_during_sampling': False}
