"""Checks of the single-comparison contract without running production chains."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
from threading import Barrier, BrokenBarrierError
import unittest
from unittest.mock import patch

import numpy as np

import run_experiment
from scripts import reproduce_scs_100d as experiment


def record(chain, means, seconds=1., iterations=300, acceptance=.2):
    discarded = iterations // 3
    used = iterations - discarded
    return {'chain': chain, 'iterations': iterations, 'mean_all_iterations': means,
            'discarded_iterations': discarded, 'estimation_iterations': used,
            'seconds': seconds, 'acceptance': acceptance, 'final_scale': .1,
            'tail_count': round(used * means[0]), 'norm_sum': used * means[1]}


class SCS100DRunnerTests(unittest.TestCase):
    def test_seed_batches_are_disjoint_reproducible_and_keep_the_fit_fixed(self):
        original = experiment.configuration('paper')
        self.assertEqual(original['seed'],901001)
        self.assertEqual(original['seed_batch'],0)
        previous = None
        for batch in (0,1,2,999):
            config = experiment.configuration('paper',overrides={'seed_batch':batch})
            self.assertEqual(config,experiment.configuration('paper',overrides={'seed_batch':batch}))
            self.assertEqual(config['seed'],901001+1_000_000*batch)
            self.assertEqual(config['fit_seed'],42)
            seeds = set(range(config['seed'],config['seed']+config['runs']))
            if previous is not None:
                self.assertFalse(seeds & previous)
            previous = seeds
        self.assertIsNone(experiment.configuration(overrides={'seed':123})['seed_batch'])
        for invalid in ({'seed_batch':-1},{'seed_batch':1000},{'seed_batch':True},
                        {'seed_batch':1.5},{'seed_batch':0,'seed':901001},
                        {'seed':2**32},{'seed':2**32-1,'runs':2},
                        {'seed_batch':1,'runs':1_000_001}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                experiment.configuration(overrides=invalid)

    def test_paper_prescribes_fifty_timed_chains_separate_tuning_and_one_target(self):
        config = experiment.configuration('paper', workers=1)
        self.assertEqual(config['runs'], 50)
        self.assertNotIn('iterations', config)
        self.assertEqual(config['time_budget'], 20.)
        self.assertEqual(config['warmup'], 1500)
        self.assertEqual(config['mode'], 'time')
        self.assertEqual(config['protocol'], 'timed-matched-dcs-scs-v3')
        self.assertEqual(config['burnin_fraction'], 1/3)
        self.assertEqual(config['step_policy'], 'shared_indicator_dual_averaging_0.234_then_frozen')
        self.assertEqual(config['thin'], 50)
        self.assertEqual(config['algorithms'], ['DCS-RWM', 'DCS-BW', 'SCS'])
        self.assertEqual(config['target']['dimension'], 100)
        self.assertEqual(config['target']['nu'], 3.)
        self.assertEqual(config['target']['alpha'], {'0': 100., '1': -100.})
        self.assertEqual(config['initial_scale'], .1)
        self.assertEqual(config['optimizer_samples'], 2000)
        self.assertEqual(config['optimizer_steps'], 1500)
        self.assertEqual(config['vi_batch_policy'], 'fresh')
        self.assertNotIn('scs_step', config)
        for preset, warmup, seconds in [('quick', 100, .05), ('reproduce', 1000, 2.), ('paper', 1500, 20.)]:
            selected = experiment.configuration(preset, workers=1)
            self.assertEqual(selected['warmup'], warmup)
            self.assertEqual(selected['time_budget'], seconds)
        for overrides in ({'runs': 0}, {'runs': True}, {'thin': 3}, {'iterations': 500},
                          {'warmup': -1}, {'time_budget': 0}, {'time_budget': np.inf},
                          {'scs_optimizer_steps': 1000}, {'scs_step': .1}, {'fit_seed': -1}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                experiment.configuration('quick', workers=1, overrides=overrides)

    def test_separate_fitting_resources_use_original_author_scs_budget(self):
        config = experiment.configuration('paper', workers=1)
        calls = []

        def fake_fit(target, settings, family, seed):
            calls.append((target, settings.copy(), family, seed))
            class Mapping:
                def parameters(self):
                    return {'family': family}
            return Mapping(), {'steps': settings['optimizer_steps'],
                'training_draws': settings['optimizer_samples'], 'seconds': 1.}

        author_result = ({'family': 'sphere'}, {'steps': 1000, 'training_draws': 2000,
                         'seed': 0, 'batch_policy': 'fixed', 'seconds': 1.})
        with patch.object(experiment, 'fit_map', side_effect=fake_fit), \
                patch('dcs_bench.scs_paper.fit', return_value=author_result) as author, \
                redirect_stdout(StringIO()):
            maps, fitting = experiment.prepare(config)
        self.assertEqual([call[2] for call in calls], ['ball'])
        self.assertEqual(calls[0][3], 42)
        author.assert_called_once_with({'target': config['target'],
                                       'optimizer_steps': 1000, 'optimizer_samples': 2000})
        self.assertEqual(set(maps), {'ball', 'sphere'})
        self.assertEqual(fitting['ball']['steps'], 1500)
        self.assertEqual(fitting['sphere']['steps'], 1000)
        self.assertFalse(fitting['ball']['coordinate_clipping'])
        self.assertTrue(fitting['sphere']['coordinate_clipping'])
        self.assertFalse(fitting['sphere']['cdf_clipping'])

    def test_summary_uses_postburn_means_and_variable_actual_iteration_counts(self):
        config = experiment.configuration('quick', workers=1, overrides={'runs': 3, 'time_budget': 1.})
        means = [[.3, 16.], [.1, 20.], [.2, 24.]]
        records = {method: [record(i, value, iterations=(i+1)*300*(method_index+1),
                    acceptance=(i+1)/10.) for i, value in enumerate(means)]
                   for method_index, method in enumerate(experiment.METHODS)}
        with patch.object(experiment, 'radial_truth', return_value=(['tail_probability', 'mean_norm'], np.array([.1, 20.]), None)):
            summary = experiment.summarize(config, {}, {}, records)
        for method_index, (method, result) in enumerate(summary['cases'].items()):
            with self.subTest(method=method):
                np.testing.assert_allclose(result['mse'], [.05/3., 32./3.], atol=1e-14)
                np.testing.assert_allclose(result['bias_squared'], [.01, 0.], atol=1e-14)
                np.testing.assert_allclose(result['variance_ddof0'], [.02/3., 32./3.], atol=1e-14)
                np.testing.assert_allclose(result['mse'], np.add(result['bias_squared'], result['variance_ddof0']), atol=1e-14)
                self.assertEqual(result['chain_means'], means)
                # Time-matched chains and methods may produce unequal counts.
                # Independent-chain errors have equal weight; rejected states
                # stay in the post-burn sample denominator.
                factor = method_index+1
                self.assertEqual(result['total_iterations'], 1800*factor)
                self.assertEqual(result['mean_iterations'], 600*factor)
                self.assertEqual(result['total_estimation_iterations'], 1200*factor)
                self.assertEqual(result['mean_estimation_iterations'], 400*factor)
                self.assertAlmostEqual(result['mean_acceptance'], .2)
                self.assertEqual(sum(row['tail_count'] for row in result['records']), 220*factor)
                self.assertEqual(result['mean_seconds'], 1.)
                self.assertEqual(result['iterations_per_second'], 600*factor)

    def test_incomplete_runs_cannot_be_reported_as_complete(self):
        config = experiment.configuration('quick', workers=1)
        records = {method: [record(0, [0., 10.], iterations=300)]
                   for method in experiment.METHODS}
        with self.assertRaisesRegex(ValueError, 'Incomplete chains'):
            experiment.summarize(config, {}, {}, records)

    def test_summary_rejects_inconsistent_burn_counts_and_short_timed_runs(self):
        config = experiment.configuration('quick', workers=1)
        for changes, message in [({'iterations': 301}, 'burn-in counts'),
                                 ({'discarded_iterations': 99}, 'burn-in counts'),
                                 ({'estimation_iterations': 199}, 'burn-in counts'),
                                 ({'seconds': .049}, 'sampling budget')]:
            records = {method: [record(i, [.1, 20.]) for i in range(2)]
                       for method in experiment.METHODS}
            records['DCS-BW'][0].update(changes)
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, message):
                experiment.summarize(config, {}, {}, records)

    def test_nonempty_output_is_rejected_before_setup_or_sampling(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'already used'
            output.mkdir()
            marker = output / 'chain.txt'
            marker.write_text('keep this existing result', encoding='utf-8')
            with patch.object(experiment, 'prepare') as prepare, patch.object(experiment, 'run_chains') as chains:
                with self.assertRaisesRegex(FileExistsError, 'fresh output folder'):
                    experiment.run(experiment.configuration('quick', workers=1), output)
                prepare.assert_not_called()
                chains.assert_not_called()
            self.assertEqual(marker.read_text(encoding='utf-8'), 'keep this existing result')

    def test_public_cli_routes_to_single_runner_with_overrides(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'path with spaces'
            with patch.object(experiment, 'main', return_value=73) as single, patch.object(run_experiment, 'run') as old:
                status = run_experiment.main(['skew-t-100d-fixed', '--preset', 'paper', '--workers', '1',
                    '--runs', '4', '--time-budget', '1.25', '--warmup', '1000', '--thin', '10',
                    '--seed', '91', '--optimizer-steps', '17', '--optimizer-samples', '42',
                    '--output-dir', str(output), '--dry-run'])
            self.assertEqual(status, 73)
            old.assert_not_called()
            self.assertEqual(single.call_args.args[0], [
                '--preset', 'paper', '--workers', '1', '--output-dir', str(output), '--dry-run',
                '--runs', '4', '--time-budget', '1.25', '--warmup', '1000', '--thin', '10',
                '--optimizer-steps', '17', '--optimizer-samples', '42', '--seed', '91'])
        for options in (['--resume'], ['--ball-map-policy', 'vanilla'], ['--iterations', '1000'],
                        ['--nuts-samples', '500'], ['--reference-samples', '500']):
            with self.subTest(options=options), redirect_stderr(StringIO()), patch.object(experiment, 'main') as single:
                with self.assertRaises(SystemExit) as caught:
                    run_experiment.main(['skew-t-100d-fixed'] + options)
                self.assertEqual(caught.exception.code, 2)
                single.assert_not_called()

    def test_user_configuration_cannot_skip_step_tuning(self):
        with self.assertRaisesRegex(ValueError, 'warmup must be a positive integer'):
            experiment.configuration('quick', overrides={'warmup': 0})

    def test_single_runner_dry_run_has_no_files_or_fitting(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'new results'
            stdout = StringIO()
            with patch.object(experiment, 'run') as run, redirect_stdout(stdout):
                self.assertEqual(experiment.main(['--preset', 'paper', '--workers', '1',
                    '--output-dir', str(output), '--dry-run']), 0)
            run.assert_not_called()
            self.assertFalse(output.exists())
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload['configuration']['runs'], 50)
            self.assertEqual(payload['configuration']['warmup'], 1500)
            self.assertEqual(payload['configuration']['time_budget'], 20.)
            self.assertNotIn('iterations', payload['configuration'])

    def test_worker_failure_aborts_barrier_and_releases_waiting_peer(self):
        from dcs_bench import timed_radial
        barrier = Barrier(2)

        def fail_before_sampling(mapping, config, seed, before_sampling, after_sampling=None):
            # Simulate compilation or warmup failure before reaching the
            # production barrier. A healthy peer must not wait for its timeout.
            raise ValueError('simulated invalid map')

        job = {'algorithm': 'SCS', 'chain': 2, 'wave': 0, 'barrier': barrier,
               'mapping': {}, 'config': {}, 'seed': 123}
        with patch.object(timed_radial, 'run_timed', side_effect=fail_before_sampling):
            with ThreadPoolExecutor(max_workers=1) as pool:
                peer = pool.submit(barrier.wait, 2.)
                with self.assertRaisesRegex(ValueError, 'simulated invalid map') as caught:
                    experiment.chain_job(job)
                with self.assertRaises(BrokenBarrierError):
                    peer.result(timeout=1.)
        self.assertTrue(barrier.broken)
        self.assertIn('SCS, chain 3, wave 0', caught.exception.__notes__)


if __name__ == '__main__':
    unittest.main()
