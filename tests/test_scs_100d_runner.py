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


def record(chain, means, seconds=1., iterations=10, acceptance=.2):
    return {'chain': chain, 'iterations': iterations, 'mean_all_iterations': means,
            'seconds': seconds, 'acceptance': acceptance, 'final_scale': .1,
            'tail_count': round(iterations * means[0]), 'norm_sum': iterations * means[1]}


class SCS100DRunnerTests(unittest.TestCase):
    def test_paper_prescribes_fifty_chains_original_warmup_and_one_target(self):
        config = experiment.configuration('paper', workers=1)
        self.assertEqual(config['runs'], 50)
        self.assertEqual(config['iterations'], 500000)
        self.assertEqual(config['warmup'], 100)
        self.assertEqual(config['thin'], 50)
        self.assertEqual(config['algorithms'], ['DCS-RWM', 'DCS-BW', 'SCS'])
        self.assertEqual(config['target']['dimension'], 100)
        self.assertEqual(config['target']['nu'], 3.)
        self.assertEqual(config['target']['alpha'], {'0': 100., '1': -100.})
        self.assertEqual(config['scs_step'], .1)
        self.assertEqual(config['scs_optimizer_samples'], 2000)
        self.assertEqual(config['scs_optimizer_steps'], 1000)
        for preset in experiment.PRESETS:
            self.assertEqual(experiment.configuration(preset, workers=1)['warmup'], 100)
        for overrides in ({'runs': 0}, {'runs': True}, {'thin': 501, 'iterations': 500},
                          {'warmup': -1}, {'time_budget': 5}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                experiment.configuration('quick', workers=1, overrides=overrides)

    def test_summary_mse_bias_variance_uses_all_iterations_including_rejections(self):
        config = experiment.configuration('quick', workers=1, overrides={'runs': 3, 'iterations': 10})
        means = [[.3, 16.], [.1, 20.], [.2, 24.]]
        rows = [record(i, value, seconds=i+1., acceptance=(i+1)/10.)
                for i, value in enumerate(means)]
        records = {method: [dict(row) for row in rows] for method in experiment.METHODS}
        with patch.object(experiment, 'radial_truth', return_value=(None, np.array([.1, 20.]), None)):
            summary = experiment.summarize(config, {}, {}, records)
        for method, result in summary['cases'].items():
            with self.subTest(method=method):
                np.testing.assert_allclose(result['mse'], [.05/3., 32./3.], atol=1e-14)
                np.testing.assert_allclose(result['bias_squared'], [.01, 0.], atol=1e-14)
                np.testing.assert_allclose(result['variance_ddof0'], [.02/3., 32./3.], atol=1e-14)
                np.testing.assert_allclose(result['mse'], np.add(result['bias_squared'], result['variance_ddof0']), atol=1e-14)
                self.assertEqual(result['chain_means'], means)
                # Acceptance does not remove repeated states or change the
                # sample denominator: ten iterations per chain, thirty total.
                self.assertEqual(result['total_iterations'], 30)
                self.assertAlmostEqual(result['mean_acceptance'], .2)
                self.assertEqual(sum(row['tail_count'] for row in result['records']), 6)
                self.assertEqual(result['mean_seconds'], 2.)
                self.assertEqual(result['iterations_per_second'], 5.)

    def test_incomplete_runs_cannot_be_reported_as_complete(self):
        config = experiment.configuration('quick', workers=1)
        records = {method: [record(0, [0., 10.], iterations=500)]
                   for method in experiment.METHODS}
        with self.assertRaisesRegex(ValueError, 'Incomplete chains'):
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
                    '--runs', '4', '--iterations', '321', '--warmup', '100', '--thin', '3',
                    '--seed', '91', '--optimizer-steps', '17', '--optimizer-samples', '42',
                    '--output-dir', str(output), '--dry-run'])
            self.assertEqual(status, 73)
            old.assert_not_called()
            self.assertEqual(single.call_args.args[0], [
                '--preset', 'paper', '--workers', '1', '--output-dir', str(output), '--dry-run',
                '--runs', '4', '--iterations', '321', '--warmup', '100', '--thin', '3',
                '--optimizer-steps', '17', '--optimizer-samples', '42', '--seed', '91'])
        for options in (['--resume'], ['--ball-map-policy', 'vanilla'], ['--time-budget', '1']):
            with self.subTest(options=options), redirect_stderr(StringIO()), patch.object(experiment, 'main') as single:
                with self.assertRaises(SystemExit) as caught:
                    run_experiment.main(['skew-t-100d-fixed'] + options)
                self.assertEqual(caught.exception.code, 2)
                single.assert_not_called()

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
            self.assertEqual(payload['configuration']['warmup'], 100)

    def test_worker_failure_aborts_barrier_and_releases_waiting_peer(self):
        from dcs_bench import scs_paper
        barrier = Barrier(2)

        def fail_before_sampling(mapping, config, seed, before_sampling):
            # Simulate compilation or warmup failure before reaching the
            # production barrier. A healthy peer must not wait for its timeout.
            raise ValueError('simulated invalid map')

        job = {'algorithm': 'SCS', 'chain': 2, 'wave': 0, 'barrier': barrier,
               'mapping': {}, 'config': {}, 'seed': 123}
        with patch.object(scs_paper, 'run_chain', side_effect=fail_before_sampling):
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
