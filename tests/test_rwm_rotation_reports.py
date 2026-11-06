"""Version-five reports preserve controls and expose Gaussian proposal units."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import run_experiment
from scripts.scs_100d_figures import COLORS, render_report
from tests.test_rotation_reports import rotation_summary


def rwm_rotation_summary(optimized=False):
    result = rotation_summary(optimized)
    result['config']['protocol'] = ('timed-numba-comparison-v5' if optimized else 'timed-sampler-comparison-v5')
    method = 'DCS-RWM-rotate'
    case = deepcopy(result['cases']['DCS-BW-rotate'])
    sigma = .1/np.sqrt(result['config']['target']['dimension']+2.)
    for record in case['records']:
        record.update(scale=sigma, final_scale=sigma, frozen_scale=sigma,
            initial_scale=sigma, initial_gaussian_sigma=sigma,
            initial_scale_reference_bw_radius=.1, gaussian_equivalent_bw_radius=.1,
            scale_parameter='Gaussian coordinate standard deviation',
            gaussian_warmup_iterations=50, reflected_warmup_iterations=50)
        for trial in record['tuning_history']:
            trial['scale'] = sigma
    case['final_scales'] = [sigma]*case['chains']
    result['cases'][method] = case
    result['config']['algorithms'].append(method)
    return result


class RWMRotationReportTests(unittest.TestCase):
    def test_both_v5_modes_preserve_metrics_and_disclose_gaussian_units(self):
        for optimized in (False, True):
            data = rwm_rotation_summary(optimized)
            before = deepcopy(data)
            with self.subTest(optimized=optimized), tempfile.TemporaryDirectory() as directory:
                destination = Path(directory)
                report = render_report(destination, data)
                self.assertEqual(len(report['figures']), 10)
                for figure in report['figures']:
                    self.assertTrue((destination/'figures'/f'{figure}.png').is_file())
                    self.assertTrue((destination/'figures'/f'{figure}.pdf').is_file())
                def rows(filename):
                    with (destination/filename).open(newline='', encoding='utf-8') as source:
                        return list(csv.DictReader(source))
                cases = rows('paper_values.csv')
                self.assertEqual(len(cases), 5 if optimized else 6)
                self.assertEqual([r['method'] for r in cases], data['config']['algorithms'])
                gaussian, = [r for r in cases if r['method'] == 'DCS-RWM-rotate']
                self.assertEqual(gaussian['scale_parameter'], 'Gaussian coordinate standard deviation')
                self.assertEqual(gaussian['execution_backend'], 'Numba' if optimized else 'JAX')
                self.assertIn('reflected Gaussian RWM', gaussian['production_step_policy'])
                self.assertNotIn('0.234', gaussian['production_step_policy'])
                for row in cases:
                    for j,name in enumerate(data['functionals']):
                        self.assertEqual(float(row[name+'_mse']), data['cases'][row['method']]['mse'][j])
                chains = [r for r in rows('scs_100d_chains.csv') if r['method'] == 'DCS-RWM-rotate']
                for record in chains:
                    self.assertAlmostEqual(float(record['final_scale'])*np.sqrt(data['config']['target']['dimension']+2.), .1)
                    self.assertEqual(float(record['gaussian_equivalent_bw_radius']), .1)
                    self.assertEqual(int(record['reflected_warmup_iterations'])+int(record['angular_warmup_iterations']), 100)
                pilots = rows('reflected_tuning.csv')
                self.assertEqual({r['method'] for r in pilots}, {'DCS-BW-reflect', 'DCS-BW-rotate', 'DCS-RWM-rotate'})
                gaussian_pilots = [r for r in pilots if r['method'] == 'DCS-RWM-rotate']
                self.assertTrue(gaussian_pilots)
                self.assertTrue(all(r['scale_parameter'] == 'Gaussian coordinate standard deviation' for r in gaussian_pilots))
                self.assertAlmostEqual(float(gaussian_pilots[0]['scale'])*np.sqrt(data['config']['target']['dimension']+2.), .1)
                self.assertEqual({r['method'] for r in rows('angular_tuning.csv')}, {'DCS-BW-rotate','DCS-RWM-rotate'})
                page = (destination/'paper_values.html').read_text(encoding='utf-8')
                for wording in ('Reflected-step pilot tuning and frozen steps', 'Gaussian coordinate standard deviation',
                                'sigma*sqrt(d+2)', 'matching initial expected squared increment length',
                                'frozen angle degrees', 'same eight quantities'):
                    self.assertIn(wording, page)
                if not optimized:
                    self.assertIn('six samplers, JAX', page)
            self.assertEqual(data, before)
        self.assertEqual(len(COLORS), len(set(COLORS.values())))

    def test_v5_replots_are_read_only_and_launcher_preserves_previous_controls(self):
        for optimized in (False, True):
            with self.subTest(optimized=optimized), tempfile.TemporaryDirectory() as directory:
                destination = Path(directory)
                (destination/'summary.json').write_text(json.dumps(rwm_rotation_summary(optimized)), encoding='utf-8')
                with patch('scripts.scs_100d_figures.render_report') as render, patch.object(run_experiment,'run') as run:
                    self.assertEqual(run_experiment.main(['--plot-only',str(destination)]),0)
                    render.assert_called_once()
                    run.assert_not_called()
        page = (Path(__file__).resolve().parents[1]/'launcher/index.html').read_text(encoding='utf-8')
        for text in ('DCS vs original SCS',
                     'Numba extension',
                     'DCS-RWM', 'DCS-BW', 'DCS-Billiard', 'functional ESS'):
            self.assertIn(text,page)


if __name__ == '__main__':
    unittest.main()
