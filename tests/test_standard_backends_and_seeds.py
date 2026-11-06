"""Standard experiments keep native kernels and independent production repeats."""
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dcs_bench import engine, samplers
from dcs_bench.backend_policy import validate_backend_configuration
from dcs_bench.targets import Target
from dcs_bench.transforms import BallMap, SphereMap
import run_experiment

STANDARD = ('skew-t-200d-nuts', 'skew-t-200d-baselines', 'skew-t-50d-nu-0.5',
            'eight-schools', 'garch11', 'logistic-regression')


class StandardBackendAndSeedTests(unittest.TestCase):
    def test_all_standard_presets_require_numba_and_preserve_native_nuts(self):
        for name in STANDARD:
            for preset in ('quick', 'reproduce', 'paper'):
                with self.subTest(experiment=name, preset=preset):
                    config = engine.load_config(name, preset)
                    self.assertEqual(config['metropolis_backend'], 'numba')
                    self.assertEqual(config['sampling_backends'], validate_backend_configuration(config))
                    for method, backend in config['sampling_backends'].items():
                        if method == 'NUTS':
                            expected = 'NumPyro' if config['mode'] == 'nuts_matched_time' else 'BlackJAX'
                            self.assertEqual(backend, f'{expected} NUTS (JAX)')
                        else:
                            self.assertEqual(backend, 'Numba CPU')

    def test_eight_repeats_preserve_fit_reference_and_nonseed_settings(self):
        for name in STANDARD:
            original = json.loads((engine.ROOT / 'configs' / (name + '.json')).read_text())
            baseline = engine.load_config(name, 'paper')
            prior_seeds = set()
            for batch in range(8):
                with self.subTest(experiment=name, batch=batch):
                    config = engine.load_config(name, 'paper', {'seed_batch': batch})
                    self.assertEqual(config['seed'], original['seed'] + 1_000_000 * batch)
                    self.assertEqual(config['fit_seed'], original['seed'] + 811)
                    self.assertEqual(config['reference_seed'], original['reference_seed'])
                    self.assertEqual({k: v for k, v in config.items() if k not in ('seed', 'seed_batch')},
                                     {k: v for k, v in baseline.items() if k not in ('seed', 'seed_batch')})
                    seeds = {engine.chain_seed(config, scenario, method, chain)
                             for scenario in range(len(config['targets']))
                             for method in range(len(config['algorithms']))
                             for chain in range(config['runs'])}
                    self.assertFalse(prior_seeds & seeds)
                    self.assertEqual(len(seeds), len(config['targets']) * len(config['algorithms']) * config['runs'])
                    prior_seeds |= seeds
            explicit = engine.load_config(name, overrides={'seed': 19})
            self.assertIsNone(explicit['seed_batch'])
            self.assertEqual(explicit['fit_seed'], baseline['fit_seed'])

    def test_unsupported_required_backend_fails_without_python_fallback(self):
        config = engine.load_config('eight-schools')
        target = Target(config['targets'][0])
        with patch.object(samplers, 'run_numpy_metropolis') as fallback:
            with self.assertRaisesRegex(ValueError, 'Required Numba'):
                samplers.run_metropolis(target, 'unsupported', BallMap(target.d), config, 9)
            fallback.assert_not_called()
        for changes in ({'metropolis_backend': 'numpy'}, {'metropolis_backend': 'auto'},
                        {'seed_batch': -1}, {'seed_batch': 1000}, {'seed_batch': True},
                        {'seed_batch': .5}, {'seed_batch': 1, 'seed': 7}, {'fit_seed': 17},
                        {'seed': -1}, {'seed': True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                engine.load_config('eight-schools', overrides=changes)
        for changes in ({'algorithms': ['DCS-unknown']},
                        {'targets': [{'kind': 'unimplemented', 'dimension': 1}]}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, 'Required Numba'):
                validate_backend_configuration(dict(config, **changes))

    def test_engine_passes_fixed_fitting_seed_for_different_repeats(self):
        fit_seeds, production = [], []
        def fit(target, config, family, seed):
            fit_seeds.append((family, seed))
            mapping = BallMap(target.d) if family == 'ball' else SphereMap(target.d)
            return mapping, {'seed': seed}
        def launch(job, chains, directory, plan):
            production.append(tuple(chain['seed'] for chain in chains))
        for batch in (0, 7):
            config = engine.load_config('skew-t-50d-nu-0.5', overrides={'seed_batch': batch})
            with tempfile.TemporaryDirectory() as temporary, \
                    patch.object(engine, 'fit_map', side_effect=fit), \
                    patch.object(engine, 'make_reference'), \
                    patch.object(engine, 'environment', return_value={}), \
                    patch.object(engine, 'source_hashes', return_value={}), \
                    patch.object(engine, 'launch_chains', side_effect=launch), \
                    patch.object(engine, 'compilation_summary', return_value={'workers': 0}), \
                    patch('dcs_bench.plotting.rebuild'), redirect_stdout(StringIO()):
                engine.run(config, Path(temporary) / 'run')
        self.assertEqual(fit_seeds[:2], fit_seeds[2:])
        self.assertNotEqual(production[:3], production[3:])

    def test_public_cli_has_only_current_comparisons_and_all_standard_repeats(self):
        self.assertEqual(set(run_experiment.named_experiments()),
                         set(STANDARD) | {'reported-zero-centre', 'reported-zero-centre-numba'})
        for experiment in STANDARD:
            output = StringIO()
            with self.subTest(experiment=experiment), redirect_stdout(output):
                self.assertEqual(run_experiment.main([experiment, '--seed-batch', '7', '--dry-run']), 0)
            config = json.loads(output.getvalue())['configuration']
            self.assertEqual(config, engine.load_config(experiment, overrides={'seed_batch': 7}))
        for experiment in ('skew-t-100d-fixed', 'skew-t-100d-boundary', 'sampler-comparison', 'sampler-comparison-numba'):
            with self.subTest(experiment=experiment), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                run_experiment.main([experiment, '--dry-run'])


if __name__ == '__main__':
    unittest.main()
