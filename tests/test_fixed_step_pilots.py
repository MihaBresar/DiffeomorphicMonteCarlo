"""Failure handling for the independent fixed-step pilot helper."""
import tempfile
import unittest
import multiprocessing
from pathlib import Path
from unittest.mock import patch

from scripts import fixed_step_pilots as pilots
from dcs_bench.transforms import BallMap, SphereMap


class FixedStepPilotTests(unittest.TestCase):
    def test_existing_output_is_not_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / 'saved-result.txt'
            artifact.write_text('preserve')
            with patch.object(pilots, 'cpu_info', return_value={'capacity': 1}), \
                    self.assertRaises(FileExistsError):
                pilots.run_pilots({}, root, 'quick', 1)
            self.assertEqual(artifact.read_text(), 'preserve')

    def test_failed_chain_restores_scs_instrumentation(self):
        target = {'kind': 'skew_t', 'dimension': 100, 'nu': 3,
                  'alpha': {'0': 100, '1': -100}}
        with tempfile.TemporaryDirectory() as directory:
            job = {'output_dir': directory, 'target': target, 'algorithm': 'SCS',
                   'map': SphereMap(100).parameters(), 'seed': 18,
                   'initial': [1.] * 100, 'burn': 1, 'scale': .1}
            original = pilots.samplers.cap_stepout
            with patch.object(pilots.samplers.MetropolisKernel, 'step',
                              side_effect=RuntimeError('forced chain failure')), \
                    self.assertRaisesRegex(RuntimeError, 'forced chain failure'):
                pilots.pilot_chain(job)
            self.assertIs(pilots.samplers.cap_stepout, original)

    def test_failed_cpu_allocation_is_reported_by_job(self):
        with patch.object(pilots, '_WORKER_ALLOCATION_ERROR', 'forced affinity failure'), \
                self.assertRaisesRegex(RuntimeError, 'forced affinity failure'):
            pilots.pilot_chain({})

    def test_failed_spawned_job_reaps_workers(self):
        target = {'kind': 'skew_t', 'dimension': 100, 'nu': 3,
                  'alpha': {'0': 100, '1': -100}}
        bad_ball = BallMap(100).parameters()
        bad_ball['scale'] = -1.
        fits = {'config': {'targets': [target]}, 'maps': {
            'current': {'ball': bad_ball, 'sphere': SphereMap(100).parameters()},
            'vanilla': {'ball': BallMap(100).parameters()}}}
        existing = {process.pid for process in multiprocessing.active_children()}
        with tempfile.TemporaryDirectory() as directory, \
                self.assertRaisesRegex(ValueError, 'beta and scale must be positive'):
            pilots.run_pilots(fits, directory, 'quick', 1)
        self.assertFalse([process for process in multiprocessing.active_children()
                          if process.pid not in existing and process.is_alive()])


if __name__ == '__main__':
    unittest.main()
