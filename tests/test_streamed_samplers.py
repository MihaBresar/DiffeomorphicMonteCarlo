"""Preserve sampler trajectories while keeping worker output bounded."""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_name, '1')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('JAX_ENABLE_X64', 'true')

from pathlib import Path
import tempfile
import unittest

import numpy as np

from dcs_bench.burnin import discard_burnin
from dcs_bench.draw_storage import DiskDrawCollector
from dcs_bench.samplers import BlackJAXRunner, NumPyroRunner, run_numpy_metropolis
from dcs_bench.targets import Target
from dcs_bench.transforms import BallMap


class StreamedSamplerTests(unittest.TestCase):
    def target(self):
        return Target(dict(kind='skew_t', dimension=2, nu=3., alpha={}))

    def config(self):
        return dict(warmup=50, thin=7, nuts_samples=1031, nuts_batch_size=16,
                    nuts_target_accept=.8, nuts_max_depth=4, burnin_fraction=1/3)

    def assert_counters_equal(self, actual, expected):
        for key in ('iterations', 'retained_draws', 'acceptance_rate', 'divergences',
                    'max_depth_hits', 'gradient_evaluations', 'frozen_scale',
                    'inverse_mass_matrix', 'burnin_iterations', 'first_retained_iteration'):
            self.assertEqual(actual[key], expected[key], key)

    def test_python_oracle_stream_keeps_same_draws_and_counters(self):
        target = self.target()
        config = dict(warmup=17, iterations=113, thin=3)
        expected, old = run_numpy_metropolis(target, 'DCS-BW', BallMap(2), config, 48)
        with tempfile.TemporaryDirectory() as tmp:
            with DiskDrawCollector(2, directory=tmp, buffer_bytes=256) as collector:
                actual, new = run_numpy_metropolis(target, 'DCS-BW', BallMap(2), config, 48,
                                                   collector=collector)
                np.testing.assert_array_equal(actual, expected)
                self.assertTrue(np.shares_memory(actual, collector.finalize()))
                for key in ('iterations', 'retained_draws', 'acceptance_rate', 'frozen_scale',
                            'log_density_evaluations', 'warmup_log_density_evaluations'):
                    self.assertEqual(new[key], old[key], key)
                self.assertEqual(new['draw_collection']['collection_rows'], len(expected))
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_numpyro_batches_match_original_full_scan_exactly(self):
        import jax
        from numpyro.infer import NUTS

        target, config = self.target(), self.config()
        runner = NumPyroRunner(target, config)
        self.assertEqual(runner.production_batch_size, 1024)
        self.assertEqual(runner.production_remainder, 7)
        seed = 917
        state = jax.block_until_ready(runner.warm(runner.initialize(jax.random.PRNGKey(seed))))
        kernel = NUTS(potential_fn=lambda x: -target.log_prob(x, 'jax'),
                      target_accept_prob=config['nuts_target_accept'], max_tree_depth=config['nuts_max_depth'])
        kernel.init(jax.random.PRNGKey(0), config['warmup'],
                    init_params=jax.numpy.asarray(target.initial()))

        def original_production(state):
            def step(state, unused):
                state = kernel.sample(state, (), {})
                return state, (state.z, state.accept_prob, state.diverging, state.num_steps)
            return jax.lax.scan(step, state, None, length=config['nuts_samples'])

        final, raw_result = jax.block_until_ready(jax.jit(original_production)(state))
        raw, acc, div, work = [np.asarray(value) for value in raw_result]
        original = np.asarray(target.physical(raw[config['thin']-1::config['thin']]))
        expected, expected_record = discard_burnin(original, dict(
            iterations=len(raw), thin=config['thin'], retained_draws=len(original),
            acceptance_rate=float(acc.mean()), divergences=int(div.sum()),
            max_depth_hits=int((work >= 2**config['nuts_max_depth']-1).sum()),
            gradient_evaluations=int(work.sum()), frozen_scale=float(state.adapt_state.step_size),
            inverse_mass_matrix=np.asarray(final.adapt_state.inverse_mass_matrix).tolist()), config)
        with tempfile.TemporaryDirectory() as tmp:
            with DiskDrawCollector(2, directory=tmp, buffer_bytes=256) as collector:
                actual, record = runner.run(seed, collector=collector)
                np.testing.assert_array_equal(actual, expected)
                self.assert_counters_equal(record, expected_record)
                self.assertTrue(np.shares_memory(actual, collector.finalize()))
                self.assertEqual(len(list(collector.directory.glob('.draw-spool-*'))), 0)
            self.assertEqual(list(Path(tmp).iterdir()), [])
        standalone, record = runner.run(seed)
        self.assertIs(type(standalone), np.ndarray)
        np.testing.assert_array_equal(standalone, expected)
        self.assert_counters_equal(record, expected_record)

    def test_blackjax_stream_preserves_global_thinning_and_partial_batch(self):
        import jax
        import jax.numpy as jnp

        target, config = self.target(), self.config()
        runner = BlackJAXRunner(target, config)
        seed, iterations = 723, 53
        key, pilot = jax.random.split(jax.random.PRNGKey(seed))
        (state, parameters), warm_info = jax.block_until_ready(
            runner.warm(pilot, jnp.asarray(target.initial())))
        arrays = [[] for _ in range(5)]
        count = 0
        while count < iterations:
            key, drawkey = jax.random.split(key)
            state, info = jax.block_until_ready(runner.batch(drawkey, state,
                parameters['step_size'], parameters['inverse_mass_matrix']))
            for chunks, value in zip(arrays, info):
                chunks.append(np.asarray(value))
            count += config['nuts_batch_size']
        raw, acc, div, work, depth = [np.concatenate(parts)[:iterations] for parts in arrays]
        original = np.asarray(target.physical(raw[config['thin']-1::config['thin']]))
        expected, expected_record = discard_burnin(original, dict(
            iterations=len(raw), thin=config['thin'], retained_draws=len(original),
            acceptance_rate=float(acc.mean()), divergences=int(div.sum()),
            max_depth_hits=int((depth >= config['nuts_max_depth']).sum()),
            gradient_evaluations=int(work.sum()), frozen_scale=float(parameters['step_size']),
            inverse_mass_matrix=np.asarray(parameters['inverse_mass_matrix']).tolist()), config)
        with tempfile.TemporaryDirectory() as tmp:
            with DiskDrawCollector(2, directory=tmp, buffer_bytes=256) as collector:
                actual, record = runner.run(seed, iterations=iterations, collector=collector)
                np.testing.assert_array_equal(actual, expected)
                self.assert_counters_equal(record, expected_record)
                self.assertEqual(record['computed_iterations'], count)
                self.assertEqual(record['warmup_divergences'], int(np.asarray(warm_info[0]).sum()))
                self.assertTrue(np.shares_memory(actual, collector.finalize()))
            self.assertEqual(list(Path(tmp).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
