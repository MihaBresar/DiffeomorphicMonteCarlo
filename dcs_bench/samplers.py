"""Frozen production kernels; adaptation and compilation have separate clocks."""
from contextlib import ExitStack
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
        if mapping is not None:
            self.log_density = (mapping.bind_log_density(target) if hasattr(mapping, 'bind_log_density')
                                else lambda x: mapping.log_density(target, x))
        else:
            self.log_density = target.bind_log_prob() if hasattr(target, 'bind_log_prob') else target.log_prob
        self.rng = np.random.default_rng(seed)
        self.state = target.initial() if mapping is None else mapping.inverse(target.initial())
        self.evaluations = 0
        self.lp = self.evaluate(self.state)
        if not np.isfinite(self.lp):
            raise ValueError(f'Nonfinite initial density for {algorithm}')

    def evaluate(self, x):
        self.evaluations += 1
        with np.errstate(over='ignore', invalid='ignore', divide='ignore', under='ignore'):
            value = self.log_density(x)
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


def run_numpy_metropolis(target, algorithm, mapping, config, seed, seconds=None, *, collector=None):
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
            value = kernel.draw()
            if collector is None:
                draws.append(value.copy())
            else:
                collector.append(value[None, :])
    if collector is not None:
        collector.flush()
    elapsed = time.perf_counter()-start
    if collector is not None:
        draws = collector.finalize()
    if len(draws) < 4:
        raise RuntimeError(f'{algorithm} retained only {len(draws)} draws; increase budget or reduce thinning')
    draws = np.asarray(draws)
    if collector is None and not np.all(np.isfinite(draws)):
        raise FloatingPointError('Nonfinite retained draws')
    return draws, {'algorithm': algorithm, 'seed': seed, 'iterations': iterations,
                   'implementation': 'NumPy with frozen target and map constants',
                   'retained_draws': len(draws), 'thin': config['thin'], 'warmup_iterations': config['warmup'],
                   'warmup_seconds': warmup_seconds, 'sampling_seconds': elapsed,
                   'requested_seconds': seconds, 'budget_overshoot_seconds': max(0., elapsed-seconds) if seconds else 0.,
                   'acceptance_rate': accepted/iterations, 'frozen_scale': scale,
                   'log_density_evaluations': kernel.evaluations-warmup_evaluations,
                   'warmup_log_density_evaluations': warmup_evaluations, 'adaptation_during_sampling': False,
                   'draw_collection': (collector.metadata() if collector is not None else
                                       {'collection_strategy': 'in-memory standalone API'})}


def make_metropolis_runner(target, algorithm, mapping, config):
    """Select the native backend before a worker enters the start barrier."""
    requested = config.get('metropolis_backend', 'auto')
    if requested not in ('auto', 'numba'):
        raise ValueError('metropolis_backend must be auto or numba')
    from .compiled_sampling import ALGORITHMS, KINDS, CompiledRunner
    if target.kind in KINDS and algorithm in ALGORITHMS:
        return CompiledRunner(target, algorithm, mapping, config)
    if requested == 'numba':
        raise ValueError(f'Required Numba backend has no implementation for {target.kind}/{algorithm}')
    return None


def run_metropolis(target, algorithm, mapping, config, seed, seconds=None, *, collector=None):
    from .burnin import discard_burnin
    runner = make_metropolis_runner(target, algorithm, mapping, config)
    if runner is None:
        draws, record = run_numpy_metropolis(target, algorithm, mapping, config, seed, seconds, collector=collector)
        record['backend_note'] = 'Python fallback: target or transition has no compiled implementation'
    else:
        draws, record = runner.run(seed, seconds, collector=collector)
    return discard_burnin(draws, record, config)


def _count_at_least(values, threshold):
    return sum(int(np.count_nonzero(values[start:start+65536] >= threshold))
               for start in range(0, len(values), 65536))


class _ScalarTraces:
    """Keep NUTS counters bounded while preserving exact NumPy reductions."""
    def __init__(self, stack, directory):
        self.stack, self.directory = stack, directory
        self.collectors = []

    def append(self, values, length):
        from .draw_storage import DiskDrawCollector
        for index, value in enumerate(values):
            array = np.asarray(value)[:length]
            if index == len(self.collectors):
                self.collectors.append(self.stack.enter_context(DiskDrawCollector(
                    1, directory=self.directory, dtype=array.dtype, buffer_bytes=256*1024)))
            self.collectors[index].append(array[:, None])

    def flush(self):
        for collector in self.collectors:
            collector.flush()

    def metadata(self):
        return {'collection_strategy': 'One buffered temporary file per scalar statistic; files closed before returning draws',
                'collection_buffer_bytes_per_statistic': 256*1024,
                'collection_statistics': len(self.collectors),
                'collection_timing_policy': 'Scalar appends and final buffer flush included in sampling_seconds'}

    def finalize(self):
        # Each column is contiguous, so mean/sum retain the original reduction
        # order used after concatenating each separate scalar trace.
        return [collector.finalize()[:, 0] for collector in self.collectors]


def _append_physical(chunks, collector, target, raw, offset, thin):
    first = (thin-1-offset) % thin
    if first >= len(raw):
        return
    values = np.asarray(target.physical(np.asarray(raw)[first::thin]))
    if collector is None:
        if not np.all(np.isfinite(values)):
            raise FloatingPointError('Nonfinite retained NUTS draws')
        chunks.append(values.copy())
    else:
        collector.append(values)


def _finish_physical(chunks, collector, dimension):
    draws = (collector.finalize() if collector is not None else
             np.concatenate(chunks) if chunks else np.empty((0, dimension)))
    if len(draws) < 4:
        raise RuntimeError('Insufficient NUTS draws')
    return draws


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

    def run(self, seed, seconds=None, iterations=None, *, collector=None):
        import jax
        import jax.numpy as jnp
        c = self.config
        key, pilot = jax.random.split(jax.random.PRNGKey(seed))
        start = time.perf_counter()
        (state, parameters), winfo = jax.block_until_ready(self.warm(pilot, jnp.asarray(self.target.initial())))
        warmup_seconds = time.perf_counter()-start
        chunks, count, used_count = [], 0, 0
        with ExitStack() as stack:
            metrics = _ScalarTraces(stack, collector.directory if collector is not None else None)
            start = time.perf_counter()
            while (count < iterations if seconds is None else time.perf_counter()-start < seconds):
                key, drawkey = jax.random.split(key)
                state, info = jax.block_until_ready(self.batch(drawkey, state, parameters['step_size'], parameters['inverse_mass_matrix']))
                used = len(info[0]) if iterations is None else min(len(info[0]), max(0, iterations-count))
                if used:
                    _append_physical(chunks, collector, self.target, info[0][:used], count, c['thin'])
                    metrics.append(info[1:], used)
                    used_count += used
                count += c['nuts_batch_size']
            metrics.flush()
            if collector is not None:
                collector.flush()
            elapsed = time.perf_counter()-start
            draws = _finish_physical(chunks, collector, self.target.d)
            acc, div, work, depth = metrics.finalize()
            record = {'algorithm': 'NUTS', 'implementation': 'BlackJAX', 'sampling_backend': 'jax', 'seed': seed,
                'iterations': used_count, 'computed_iterations': count, 'retained_draws': len(draws), 'thin': c['thin'],
                'warmup_iterations': c['warmup'], 'warmup_seconds': warmup_seconds,
                'sampling_seconds': elapsed, 'requested_seconds': seconds,
                'budget_overshoot_seconds': max(0., elapsed-seconds) if seconds else 0.,
                'acceptance_rate': float(acc.mean()), 'divergences': int(div.sum()),
                'max_depth_hits': _count_at_least(depth, c['nuts_max_depth']),
                'gradient_evaluations': int(work.sum()), 'warmup_divergences': int(np.asarray(winfo[0]).sum()),
                'frozen_scale': float(parameters['step_size']), 'inverse_mass_matrix': np.asarray(parameters['inverse_mass_matrix']).tolist(),
                'adaptation_during_sampling': False,
                'sampling_output_timing': 'Thinning, physical conversion, scalar recording, draw appends and final buffer flush included',
                'diagnostic_collection': metrics.metadata(),
                'draw_collection': (collector.metadata() if collector is not None else
                                       {'collection_strategy': 'in-memory standalone API'})}
            from .burnin import discard_burnin
            return discard_burnin(draws, record, c)


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
        self.production_batch_size = min(1024, config['nuts_samples'])
        self.production_remainder = config['nuts_samples'] % self.production_batch_size
        def production(length):
            def run_batch(state):
                def step(state, unused):
                    state = one_step(state)
                    return state, (state.z, state.accept_prob, state.diverging, state.num_steps)
                return jax.lax.scan(step, state, None, length=length)
            return jax.jit(run_batch).lower(dummy).compile()
        self.production = production(self.production_batch_size)
        self.production_tail = production(self.production_remainder) if self.production_remainder else None
        self.compile_seconds = time.perf_counter()-start

    def run(self, seed, *, collector=None):
        import jax
        c = self.config
        start = time.perf_counter()
        state = jax.block_until_ready(self.warm(self.initialize(jax.random.PRNGKey(seed))))
        warmup_seconds = time.perf_counter()-start
        frozen_scale = float(state.adapt_state.step_size)
        chunks, count = [], 0
        with ExitStack() as stack:
            metrics = _ScalarTraces(stack, collector.directory if collector is not None else None)
            start = time.perf_counter()
            while count < c['nuts_samples']:
                batch = (self.production if c['nuts_samples']-count >= self.production_batch_size
                         else self.production_tail)
                state, result = jax.block_until_ready(batch(state))
                raw = np.asarray(result[0])
                _append_physical(chunks, collector, self.target, raw, count, c['thin'])
                metrics.append(result[1:], len(raw))
                count += len(raw)
            metrics.flush()
            if collector is not None:
                collector.flush()
            elapsed = time.perf_counter()-start
            if float(state.adapt_state.step_size) != frozen_scale:
                raise RuntimeError('NUTS adapted during production')
            draws = _finish_physical(chunks, collector, self.target.d)
            acc, div, work = metrics.finalize()
            record = {'algorithm': 'NUTS', 'implementation': 'NumPyro', 'sampling_backend': 'jax', 'seed': seed,
                'iterations': count, 'retained_draws': len(draws), 'thin': c['thin'],
                'warmup_iterations': c['warmup'], 'warmup_seconds': warmup_seconds,
                'sampling_seconds': elapsed, 'acceptance_rate': float(acc.mean()),
                'divergences': int(div.sum()),
                'max_depth_hits': _count_at_least(work, 2**c['nuts_max_depth']-1),
                'max_depth_detection': 'num_steps >= 2**max_tree_depth - 1 (full tree saturation)',
                'gradient_evaluations': int(work.sum()), 'frozen_scale': frozen_scale,
                'inverse_mass_matrix': np.asarray(state.adapt_state.inverse_mass_matrix).tolist(),
                'adaptation_during_sampling': False,
                'production_batch_size': self.production_batch_size,
                'sampling_output_timing': 'Thinning, physical conversion, scalar recording, draw appends and final buffer flush included',
                'diagnostic_collection': metrics.metadata(),
                'draw_collection': (collector.metadata() if collector is not None else
                                       {'collection_strategy': 'in-memory standalone API'})}
            from .burnin import discard_burnin
            return discard_burnin(draws, record, c)
