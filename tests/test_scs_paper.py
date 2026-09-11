"""Check the attribution pin, unchanged author routine and online estimators."""
import hashlib
import unittest

import numpy as np

from dcs_bench import scs_paper
from dcs_bench.samplers import cap_stepout
from dcs_bench.targets import Target
from dcs_bench.transforms import SphereMap


class SCSPaperTests(unittest.TestCase):
    def test_vendored_bytes_match_original_git_blobs(self):
        expected = {'src/scp_core.py': 'c611aad0b209df92b76f77fe9be905a52e91d13d',
                    'src/train.py': 'd4f26c54ce21c6129eff1bdbcec6378b4bc24990',
                    'LICENSE': '36f82dd66179f18dcde82a2ad56efb49e227b26e'}
        manifest = scs_paper.verify_vendor()
        for filename, git_sha in expected.items():
            data = (scs_paper.VENDOR / filename).read_bytes()
            blob = b'blob ' + str(len(data)).encode() + b'\0' + data
            self.assertEqual(hashlib.sha1(blob).hexdigest(), git_sha)
            self.assertEqual(manifest['files'][filename]['git_blob_sha'], git_sha)

    def test_author_stepout_matches_canonical_geometry(self):
        import jax.numpy as jnp
        core = scs_paper._author()
        rng = np.random.default_rng(19)
        checked = 0
        for _ in range(300):
            start = rng.normal(size=6)
            start[-1] = -abs(start[-1])
            start /= np.linalg.norm(start)
            noise = rng.normal(size=6)
            noise -= start * np.dot(start, noise)
            proposed = start + noise
            proposed /= np.linalg.norm(proposed)
            if proposed[-1] < .1:
                continue
            actual = core.stepout_dark_side(jnp.asarray(start), jnp.asarray(proposed), .1)
            expected = cap_stepout(start, proposed, 1.1)
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
            self.assertLess(actual[-1], .1 + 1e-12)
            checked += 1
        self.assertGreater(checked, 20)

    def test_fit_uses_exact_author_optimizer_on_our_original_target(self):
        import jax.numpy as jnp
        config = {'dimension': 5, 'nu': 3., 'alpha': {0: 3., 1: -3.},
                  'optimizer_samples': 32, 'optimizer_steps': 6}
        actual, metadata = scs_paper.fit(config)
        model = scs_paper._author().SCP(5, 1.1)
        target = Target(scs_paper.target_spec(config))
        params, losses = model.minimize_reverse_kl(
            lambda x: target.log_prob(x, 'jax'), seed=0, ntrain=32,
            max_iter=6, learning_rate=.01, clip_value=1000.)
        observer, centre, radius = model.transform_params(params)
        np.testing.assert_allclose(actual['observer'], observer, atol=1e-14)
        np.testing.assert_allclose(actual['mu'], centre, atol=1e-14)
        self.assertEqual(actual['scale'], float(radius))
        np.testing.assert_allclose(metadata['loss'], losses, atol=1e-14)
        self.assertEqual(metadata['total_training_draws'], 32)
        self.assertFalse(metadata['gradient_clipping'])
        self.assertEqual(metadata['target_skew_cdf'], 'normal')
        # The adapter uses the original DCS density, including its normal CDF.
        x = jnp.asarray([.03, -.02, 1., 2., 3.])
        self.assertAlmostEqual(float(target.log_prob(x, 'jax')), target.log_prob(np.asarray(x)), places=12)

    def test_online_sums_include_rejections_and_unthinned_remainder(self):
        import jax
        import jax.numpy as jnp
        mapping = SphereMap(5).parameters()
        config = {'dimension': 5, 'nu': 3., 'alpha': {0: 100., 1: -100.},
                  'iterations': 37, 'warmup': 9, 'thin': 7}
        traces, record = scs_paper.run_chain(mapping, config, seed=331)
        core = scs_paper._author()
        model = core.SCP(5, 1.1)
        params = model.inverse_transform_params(jnp.zeros(5), jnp.zeros(5), jnp.asarray(1.))
        target = Target(scs_paper.target_spec(config))
        density = model.transform_target(lambda x: target.log_prob(x, 'jax'), params)
        burn_key, key = jax.random.split(jax.random.key(331))
        initial = model.inverse_projection(params, jnp.ones(5))
        states, _ = core.rwm_bright_side_stepout(density, initial, 1.1, burn_key, 9, stepsize=.1)
        state = states[-1]
        all_states, all_accepts = [], []
        for length in [7, 7, 7, 7, 7, 2]:
            key, subkey = jax.random.split(key)
            states, accepts = core.rwm_bright_side_stepout(density, state, 1.1, subkey, length, stepsize=.1)
            state = states[-1]
            all_states.append(np.asarray(states))
            all_accepts.append(np.asarray(accepts))
        physical = model.projection(params, jnp.asarray(np.concatenate(all_states)))
        norm = np.linalg.norm(np.asarray(physical), axis=1)
        radial = np.column_stack([norm >= 50, norm])
        np.testing.assert_allclose(traces, radial[6::7], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose([record['function_sums']['tail_probability'],
                                    record['function_sums']['mean_norm']], radial.sum(axis=0), rtol=1e-12)
        self.assertEqual(record['accepted'], int(np.concatenate(all_accepts).sum()))
        self.assertLess(record['accepted'], 37)
        self.assertEqual(record['iterations'], 37)
        self.assertEqual(record['retained'], 5)
        self.assertEqual(record['scale'], .1)
        self.assertEqual(record['final_scale'], .1)
        self.assertGreater(record['compile_seconds'], 0.)
        self.assertGreater(record['sampling_seconds'], 0.)
        self.assertEqual(record['seconds'], record['sampling_seconds'])
        for checkpoint in record['checkpoints']:
            end = checkpoint['iterations']
            np.testing.assert_allclose(checkpoint['mean_all_iterations'], radial[:end].mean(axis=0), rtol=1e-12)
            self.assertEqual(checkpoint['tail_count'], int(radial[:end, 0].sum()))
        again, cached = scs_paper.run_chain(mapping, config, seed=331)
        np.testing.assert_array_equal(again, traces)
        self.assertTrue(cached['compile_cache_hit'])
        self.assertEqual(cached['compile_seconds'], 0.)


if __name__ == '__main__':
    unittest.main()
