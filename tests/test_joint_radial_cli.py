"""Pilot-time overrides reach every method without changing study defaults."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dcs_bench.transforms import BallMap, SphereMap
from scripts import audit_joint_radial as audit


class SetupCaptured(Exception):
    """Stop before numerical fitting or sampling in configuration tests."""


class JointRadialCliTests(unittest.TestCase):
    def test_invalid_pilot_seconds_fail_before_hardware_or_output_setup(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'unused'
            for value in ('0', '-1', 'nan', 'inf', '-inf', 'not-a-number'):
                with self.subTest(value=value), redirect_stderr(StringIO()), \
                        patch.object(audit, 'cpu_info') as hardware:
                    with self.assertRaises(SystemExit) as caught:
                        audit.main(['--pilot-seconds='+value, '--output-dir', str(output)])
                    self.assertEqual(caught.exception.code, 2)
                    hardware.assert_not_called()
                    self.assertFalse(output.exists())

    def test_omitted_override_preserves_every_preset_in_recorded_config(self):
        original = deepcopy(audit.PRESETS)
        for preset in audit.PRESETS:
            captured = {}

            def save(path, payload):
                self.assertEqual(path.name, 'protocol.json')
                captured.update(deepcopy(payload))
                raise SetupCaptured

            with self.subTest(preset=preset), tempfile.TemporaryDirectory() as temporary, \
                    patch.object(audit, 'environment', return_value={}), \
                    patch.object(audit, 'source_hashes', return_value={}), \
                    patch.object(audit, 'write_json', side_effect=save), \
                    patch.object(audit, 'fit_map') as fit, \
                    patch.object(audit, 'run_waves') as sampling:
                with self.assertRaises(SetupCaptured):
                    audit.main(['--preset', preset, '--workers', '1', '--output-dir', temporary])
                self.assertEqual(captured['config'], {**original[preset], 'preset': preset, 'workers': 1,
                    'scs_vi': {'optimizer_steps': 1000 if preset == 'paper' else original[preset]['optimizer_steps'],
                               'optimizer_samples': original[preset]['optimizer_samples']}})
                fit.assert_not_called()
                sampling.assert_not_called()
        self.assertEqual(audit.PRESETS, original)

    def test_override_is_recorded_and_applied_to_all_pilot_jobs(self):
        original = deepcopy(audit.PRESETS)
        saved = {}

        def fake_fit(target, config, family, seed):
            self.assertEqual(config['scs_vi'], {'optimizer_steps': 20, 'optimizer_samples': 128})
            cls = BallMap if family == 'ball' else SphereMap
            return cls(target.d, scale=1.463), {'seconds': 0.}

        def save(path, payload):
            saved[path.name] = deepcopy(payload)
            if path.name == 'pilot_jobs.json':
                raise SetupCaptured

        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(StringIO()), \
                patch.object(audit, 'environment', return_value={}), \
                patch.object(audit, 'source_hashes', return_value={}), \
                patch.object(audit, 'fit_map', side_effect=fake_fit), \
                patch.object(audit, 'write_json', side_effect=save), \
                patch.object(audit, 'run_waves') as sampling:
            with self.assertRaises(SetupCaptured):
                audit.main(['--preset', 'quick', '--workers', '1', '--pilot-seconds', '1.0',
                            '--output-dir', temporary])
            sampling.assert_not_called()

        expected_config = {**original['quick'], 'preset': 'quick', 'workers': 1, 'pilot_seconds': 1.,
                           'scs_vi': {'optimizer_steps': 20, 'optimizer_samples': 128}}
        self.assertEqual(saved['protocol.json']['config'], expected_config)
        jobs = saved['pilot_jobs.json']
        self.assertEqual({job['target_id'] for job in jobs}, set(audit.TARGETS))
        self.assertEqual({job['algorithm'] for job in jobs}, set(audit.GRIDS))
        for target in audit.TARGETS:
            for method, steps in audit.GRIDS.items():
                selected = [job for job in jobs if job['target_id'] == target and job['algorithm'] == method]
                self.assertEqual(len(selected), 8 * len(steps) * original['quick']['pilot_chains'])
                self.assertEqual({job['seconds'] for job in selected}, {1.})
                self.assertEqual({job['warmup'] for job in selected}, {original['quick']['pilot_burn']})
        self.assertEqual(audit.PRESETS, original)


if __name__ == '__main__':
    unittest.main()
