"""Keep billiard production coordinates and all seven reported methods explicit."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import run_experiment
from dcs_bench.transforms import BallMap, SphereMap
from scripts import reproduce_comparison as suite
from scripts import reproduce_scs_100d as runner
from scripts.scs_100d_figures import render_report
from tests.test_rwm_rotation_reports import rwm_rotation_summary


def billiard_summary(vi='on'):
    summary = rwm_rotation_summary()
    summary['config'].update(protocol='timed-sampler-comparison-v6',
        implementation_revision='shared-jax-billiard-v6', vi=vi)
    summary['config']['algorithms'].append('DCS-Billiard')
    summary['cases']['DCS-Billiard'] = deepcopy(summary['cases']['DCS-BW-reflect'])
    for record in summary['cases']['DCS-Billiard']['records']:
        record.update(calibration_algorithm='DCS-BW-reflect',
            proposal_coordinates='post_mobius_ball', calibration_coordinates='post_mobius_ball')
    dimension = summary['config']['target']['dimension']
    delta = np.zeros(dimension)
    if vi == 'on':
        delta[:3] = [.1, -.2, .05]
    summary['maps'] = {'ball': BallMap(dimension, delta=delta).parameters(),
                       'sphere': SphereMap(dimension).parameters()}
    summary['method_maps'] = runner.effective_method_maps(summary['config'], summary['maps'])
    return summary


class BilliardComparisonTests(unittest.TestCase):
    def test_effective_map_provenance_preserves_joint_fit_and_all_other_methods(self):
        for vi in ('on', 'off'):
            config = suite.configuration('paper', target='skew-t-100d', vi=vi)
            if vi == 'on':
                ball = BallMap(100, beta=2., scale=3., mu=np.arange(100)/100.,
                               delta=np.full(100, .01)).parameters()
            else:
                ball = BallMap(100, beta=2.).parameters()
            maps = {'ball': ball, 'sphere': SphereMap(100).parameters()}
            before = deepcopy(maps)
            effective = runner.effective_method_maps(config, maps)
            self.assertEqual(maps, before)
            self.assertEqual(set(effective), set(config['algorithms']))
            self.assertEqual(effective['SCS'], maps['sphere'])
            for method in config['algorithms']:
                if method not in ('DCS-Billiard', 'SCS'):
                    self.assertEqual(effective[method], maps['ball'])
            for key in ('mu', 'scale', 'beta', 'dimension'):
                self.assertEqual(effective['DCS-Billiard'][key], maps['ball'][key])
            self.assertEqual(effective['DCS-Billiard']['delta'], [0.]*100)
            effective['DCS-Billiard']['mu'][0] = 999.
            self.assertEqual(maps, before)

    def test_report_preserves_all_eight_metrics_and_exports_actual_billiard_map(self):
        # Render both VI modes from known records; no timing or ranking test.
        for vi in ('on', 'off'):
            summary = billiard_summary(vi)
            before = deepcopy(summary)
            with self.subTest(vi=vi), tempfile.TemporaryDirectory() as directory:
                destination = Path(directory)
                result = render_report(destination, summary)
                self.assertEqual(len(result['figures']), 10)
                def rows(filename):
                    with (destination/filename).open(newline='', encoding='utf-8') as stream:
                        return list(csv.DictReader(stream))
                cases = rows('paper_values.csv')
                self.assertEqual([r['method'] for r in cases], summary['config']['algorithms'])
                self.assertEqual(len(cases), 7)
                for row in cases:
                    for metric, expected in zip(summary['functionals'], summary['cases'][row['method']]['mse']):
                        self.assertEqual(float(row[metric+'_mse']), expected)
                candidate, = [r for r in cases if r['method'] == 'DCS-Billiard']
                self.assertEqual(candidate['execution_backend'], 'JAX')
                self.assertIn('post-Mobius', candidate['production_step_policy'])
                parameters = {r['method']: r for r in rows('effective_maps.csv')}
                self.assertEqual(json.loads(parameters['DCS-Billiard']['delta']),
                                 [0.]*summary['config']['target']['dimension'])
                self.assertEqual(json.loads(parameters['DCS-BW-reflect']['delta']),
                                 summary['maps']['ball']['delta'])
                pilots = [r for r in rows('reflected_tuning.csv') if r['method'] == 'DCS-Billiard']
                self.assertTrue(pilots)
                self.assertTrue(all(r['pilot_kernel'] == 'Reflected BW' for r in pilots))
                self.assertNotIn('DCS-Billiard', {r['method'] for r in rows('angular_tuning.csv')})
                page = (destination/'paper_values.html').read_text(encoding='utf-8')
                for wording in ('seven samplers, JAX', 'Effective maps used by each sampler',
                                'pilot kernel', 'same eight quantities', 'only for a radial transformed target',
                                'production does not retune'):
                    self.assertIn(wording, page)
                if vi == 'on':
                    self.assertIn('not a new delta=0 VI fit', page)
                self.assertEqual(summary, before)

    def test_v6_replot_only_routes_without_sampling(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            (destination/'summary.json').write_text(json.dumps(billiard_summary()), encoding='utf-8')
            with patch('scripts.scs_100d_figures.render_report') as render, \
                    patch.object(run_experiment, 'run') as run:
                self.assertEqual(run_experiment.main(['--plot-only', str(destination)]), 0)
                render.assert_called_once()
                run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
