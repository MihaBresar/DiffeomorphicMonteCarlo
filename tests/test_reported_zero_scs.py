"""Zero-centre/observer SCS intervention and immutable historical inputs."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from dcs_bench import reported_study as study, scs_paper, timed_radial
from dcs_bench.targets import Target
from scripts import reproduce_scs_100d as runner


class ReportedZeroSCSTests(unittest.TestCase):
    def test_legacy_prepared_inputs_match_published_v1_and_v2_fingerprints(self):
        # Digests were captured from the published pre-intervention loader,
        # before v3 changed production mu/observer. They cover maps, fit histories
        # and every prepared-provenance field, not only selected parameters.
        digests = {
            (1, 100): '7fe7e064f2482508c87513575bc5433d2a170cd086085a39939cecd953227763',
            (1, 500): '6bffd13a3be836be8d5c0ce3d49dd45b38a85308ec1459139c38b530cb269d31',
            (2, 100): '9b093167dbf3f8457c0f8992dbde326a7d65c60d2c9899a28950e2064bd019e6',
            (2, 500): 'ffefb253a212c59e1ffe4aa52013e06d43bc37c1b224de34d27db5ecf8bc3094',
        }
        for version, builder in ((1, study.legacy_configuration), (2, study.calibrated_configuration)):
            for backend in ('jax', 'numba'):
                for dimension in (100, 500):
                    for preset in study.PRESETS:
                        with self.subTest(version=version, backend=backend, dimension=dimension, preset=preset):
                            config = builder(preset, comparison=backend, target=f'skew-t-{dimension}d')
                            prepared = study.load_prepared(config)
                            digest = hashlib.sha256(json.dumps(prepared, sort_keys=True,
                                separators=(',', ':')).encode()).hexdigest()
                            self.assertEqual(digest, digests[version, dimension])
                            self.assertEqual(config['protocol'].rsplit('-', 1)[-1], f'v{version}')

    def test_zero_map_preserves_radius_step_and_other_methods_in_every_preset(self):
        for backend in ('jax', 'numba'):
            for dimension in (100, 500):
                for preset in study.PRESETS:
                    with self.subTest(backend=backend, dimension=dimension, preset=preset):
                        config = study.configuration(preset, comparison=backend, target=f'skew-t-{dimension}d')
                        legacy = study.calibrated_configuration(preset, comparison=backend, target=f'skew-t-{dimension}d')
                        prepared, before = study.load_prepared(config), study.load_prepared(legacy)
                        self.assertEqual(config['protocol'], study.PROTOCOLS[backend])
                        self.assertEqual(config['scs_calibrated_move'], legacy['scs_calibrated_move'])
                        self.assertEqual(config['calibration_budget_per_family'], legacy['calibration_budget_per_family'])
                        self.assertEqual(prepared['maps']['ball'], before['maps']['ball'])
                        self.assertEqual(prepared['maps']['sphere'], before['maps']['sphere'])
                        self.assertEqual(prepared['fitting'], before['fitting'])
                        mapping = prepared['maps']['sphere_calibrated']
                        self.assertEqual(mapping['mu'], [0.] * dimension)
                        self.assertEqual(mapping['observer'], [0.] * dimension)
                        self.assertEqual(mapping['latitude'], 1.1)
                        self.assertEqual(mapping['scale'], before['maps']['sphere_calibrated']['scale'])
                        self.assertNotIn('delta', mapping)
                        intervention = prepared['provenance']['calibrated_scs_map_intervention']
                        self.assertFalse(intervention['radius_step_recalibrated'])
                        self.assertFalse(intervention['constrained_vi_refit'])
                        self.assertEqual(intervention['radius_step_selection_map'], 'original author VI centre and observer')
                        self.assertEqual(prepared['provenance']['schema'], 'reported-zero-centre-prepared-inputs-v3')

    def test_author_projection_is_finite_radial_and_invertible_at_zero_observer(self):
        import jax.numpy as jnp
        core = scs_paper._author()
        for dimension in (100, 500):
            config = study.configuration(target=f'skew-t-{dimension}d')
            mapping = study.load_prepared(config)['maps']['sphere_calibrated']
            model = core.SCP(dimension, 1.1)
            params = scs_paper._parameters(model, mapping)
            observer, centre, scale = model.transform_params(params)
            np.testing.assert_array_equal(observer, np.zeros(dimension))
            np.testing.assert_array_equal(centre, np.zeros(dimension))
            self.assertAlmostEqual(float(scale), mapping['scale'], places=14)
            for radius in (.01, 1., 50., 1000.):
                physical = jnp.ones(dimension) * radius / np.sqrt(dimension)
                sphere = model.inverse_projection(params, physical)
                projected = model.projection(params, sphere)
                expected = 1.1 * scale * sphere[:-1] / (1.1 - sphere[-1])
                np.testing.assert_allclose(projected, physical, rtol=2e-10, atol=2e-10)
                np.testing.assert_allclose(projected, expected, rtol=2e-14, atol=2e-14)
                self.assertTrue(np.isfinite(float(model.log_jacobian(params, physical))))
                self.assertLess(float(sphere[-1]), 1.1)

    def test_modified_map_follows_direct_author_warmup_and_production(self):
        import jax
        import jax.numpy as jnp
        core = scs_paper._author()
        for dimension in (100, 500):
            config = study.configuration(target=f'skew-t-{dimension}d')
            mapping = study.load_prepared(config)['maps']['sphere_calibrated']
            execution = runner.method_execution_config(config, 'SCS-calibrated')
            driver = timed_radial._Matched(mapping, execution, 617)
            model = core.SCP(dimension, 1.1)
            params = model.inverse_transform_params(jnp.zeros(dimension), jnp.zeros(dimension),
                                                    jnp.asarray(mapping['scale']))
            target = Target(config['target'])
            density = model.transform_target(lambda x: target.log_prob(x, 'jax'), params)
            initial = model.inverse_projection(params, jnp.ones(dimension))
            burn_key, key = jax.random.split(jax.random.key(617))
            step = study.CALIBRATED_SCS[dimension]['h']
            direct_burn = jax.jit(lambda state, rng: core.rwm_bright_side_stepout(
                density, state, 1.1, rng, config['warmup'], stepsize=step))
            burn_states, _ = direct_burn(initial, burn_key)
            np.testing.assert_array_equal(driver.state, burn_states[-1])
            key, subkey = jax.random.split(key)
            direct_batch = jax.jit(lambda state, rng: core.rwm_bright_side_stepout(
                density, state, 1.1, rng, timed_radial.BATCH_SIZE, stepsize=step))
            states, accepted = direct_batch(burn_states[-1], subkey)
            prefixes, saved = driver.batch()
            np.testing.assert_array_equal(driver.state, states[-1])
            norms = np.asarray(jnp.linalg.norm(model.projection(params, states), axis=1))
            values = np.column_stack((norms >= config['threshold'], norms, np.asarray(accepted)))
            expected = np.cumsum(values.reshape(3, 100, 3).sum(axis=1), axis=0)
            np.testing.assert_allclose(prefixes, expected, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(saved, norms[config['thin'] - 1::config['thin']], rtol=1e-12)

    def test_zero_override_cannot_be_silently_restored_or_mislabelled(self):
        config = study.configuration()
        for change in ({'calibrated_scs_map_intervention': {}},
                       {'production_calibrated_scs_map_policy': 'original author VI'},
                       {'reported_calibrated_scs_parameters': study.calibrated_configuration()['reported_calibrated_scs_parameters']}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                study.validate_configuration(dict(config, **change))
        for vector in ('mu', 'observer'):
            prepared = study.load_prepared(config)
            prepared['maps']['sphere_calibrated'][vector][0] = .01
            with tempfile.TemporaryDirectory() as temp, self.assertRaisesRegex(ValueError, 'exact bundled'):
                runner.run(config, Path(temp) / 'must-not-exist', prepared_inputs=prepared)
        prepared = study.load_prepared(config)
        summary = dict(config=config, maps=prepared['maps'], fitting=prepared['fitting'],
            prepared_input_provenance=prepared['provenance'],
            method_maps=runner.effective_method_maps(config, prepared['maps']))
        study.validate_reported_summary(summary)
        damaged = copy.deepcopy(summary)
        damaged['prepared_input_provenance']['calibrated_scs_map_intervention']['radius_step_recalibrated'] = True
        with self.assertRaisesRegex(ValueError, 'provenance changed'):
            study.validate_reported_summary(damaged)


if __name__ == '__main__':
    unittest.main()
