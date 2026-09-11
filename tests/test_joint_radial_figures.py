"""Reporting checks for joint calibration, invalid pilots and boundary rates."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import joint_radial_figures as figures


def fixture_target(nu=2, alpha=None):
    grid = []
    cases = {}
    boundary = {}
    for index, algorithm in enumerate(('DCS-BW', 'SCS')):
        for radius in (1., 10.04987562112089):
            for scale in (.1, .3):
                grid.append({'algorithm': algorithm, 'R': radius, 'scale': scale,
                             'score': (index+1)*radius/scale, 'valid_selection': True,
                             'total_sampling_seconds': 2.123456789012345,
                             'bulk_ess': {'tail_probability': 101.23456789012345, 'log1p_norm': 123.45},
                             'rank_rhat': {'tail_probability': 1.001, 'log1p_norm': 1.002}})
        name = algorithm+'-joint'
        cases[name] = {'algorithm': algorithm, 'control': 'joint', 'label': algorithm+' joint choice',
                       'R': 10.04987562112089, 'scale': .1, 'map_kind': 'calibrated',
                       'map_group': name, 'policy': 'frozen', 'records': [{'chain': 0}, {'chain': 1}],
                       'total_iterations': 200001, 'mean_iterations': 100000.5,
                       'mean_sampling_seconds': 1.00000123456789,
                       'empirical_mse': {'tail_probability': .0000123456789012345,
                                         'mean_norm': .02345678901234567}}
        boundary[name] = {'iterations': 1000, 'initial_boundary_crossings': 400,
                          'stepout_calls': 400 if algorithm == 'SCS' else 0,
                          'final_infeasible': 50, 'nonfinite_density': 2, 'density_rejections': 148,
                          'accepted': 800, 'accepted_after_stepout': 320 if algorithm == 'SCS' else 0}
    return {'target': {'dimension': 100, 'nu': nu, 'alpha': alpha or {}},
            'config': {'preset': 'quick', 'runs': 2, 'time_budget': 1.},
            'truth': {'tail_probability': .1, 'mean_norm': 12.},
            'cases': cases, 'pilots': {'grid': grid, 'selected': {}},
            'contrasts': [], 'boundary': boundary}


class JointRadialFigureTests(unittest.TestCase):
    def test_profiles_exclude_invalid_winners_and_preserve_exact_candidates(self):
        target = fixture_target()
        target['pilots']['grid'][0]['score'] = 1.e10
        target['pilots']['grid'][0]['valid_selection'] = False
        before = copy.deepcopy(target)
        candidates = figures._candidate_rows('skew', target)
        profile = figures._profile_rows('skew', target)
        self.assertEqual(target, before)
        self.assertEqual(len(candidates), 8)
        self.assertEqual(candidates[0]['score'], 1.e10)
        self.assertIsNone(candidates[0]['eligible_score'])
        dcs_unit = next(row for row in profile if row['algorithm'] == 'DCS-BW' and row['R'] == 1.)
        self.assertEqual(dcs_unit['best_tested_scale'], .3)
        self.assertEqual(dcs_unit['valid_candidates'], 1)
        selected = [row for row in candidates if row['selected_production_cases']]
        self.assertEqual(len(selected), 2)
        self.assertEqual(selected[0]['R'], 10.04987562112089)
        self.assertEqual(selected[0]['bulk_ess_tail_probability'], 101.23456789012345)

    def test_profile_reports_missing_valid_candidates_without_inventing_choice(self):
        target = fixture_target()
        for row in target['pilots']['grid']:
            if row['R'] == 1.:
                row['score'] = float('nan')
        profile = figures._profile_rows('skew', target)
        for row in profile:
            if row['R'] == 1.:
                self.assertIsNone(row['best_tested_scale'])
                self.assertIsNone(row['best_eligible_score'])
                self.assertEqual(row['valid_candidates'], 0)

    def test_boundary_rates_use_probe_iterations_and_allow_overlapping_events(self):
        target = fixture_target()
        rows = figures._boundary_rows('skew', target)
        scs = next(row for row in rows if row['algorithm'] == 'SCS')
        self.assertEqual(scs['initial_boundary_crossings_rate'], .4)
        self.assertEqual(scs['stepout_calls_rate'], .4)
        self.assertEqual(scs['accepted_after_stepout_rate'], .32)
        self.assertEqual(scs['accepted_rate'], .8)
        self.assertGreater(sum(scs[key+'_rate'] for key in figures.BOUNDARY_COUNTS), 1.)
        target['boundary']['SCS-joint']['iterations'] = 0
        self.assertIsNone(figures._boundary_rows('skew', target)[1]['accepted_rate'])
        target['boundary']['SCS-joint']['iterations'] = -1
        with self.assertRaises(ValueError):
            figures._boundary_rows('skew', target)

    def test_report_exports_full_grid_and_uses_correct_target_subreport_labels(self):
        targets = {'symmetric-nu3': fixture_target(3), 'skew-nu2': fixture_target(2, {'0': 3.})}
        targets['skew-nu2']['pilots']['grid'][0]['valid_selection'] = False
        choice = copy.deepcopy(targets['skew-nu2']['pilots']['grid'][2])
        choice.update(step_endpoint=True, radius_endpoint=False)
        targets['skew-nu2']['pilots']['selected']['DCS-BW-joint'] = choice
        summary = {'config': {'preset': 'quick', 'runs': 2, 'time_budget': 1.},
                   'targets': targets, 'notes': []}
        before = copy.deepcopy(summary)
        recorded = []

        def target_report(root, value):
            root.mkdir(parents=True, exist_ok=True)
            (root/'paper_values.html').write_text(value['target_label'], encoding='utf-8')
            recorded.append(value)
            return {'figures': [], 'csv_files': [], 'html': 'paper_values.html'}

        with tempfile.TemporaryDirectory() as tmp, patch.object(figures, 'render_timed_report', side_effect=target_report):
            root = Path(tmp)
            result = figures.render_report(root, summary)
            self.assertEqual(summary, before)
            self.assertEqual(len(result['figures']), 6)
            for name in result['figures']:
                self.assertTrue((root/'figures'/(name+'.png')).is_file())
                self.assertTrue((root/'figures'/(name+'.pdf')).is_file())
            self.assertIn('symmetric', recorded[0]['target_label'])
            self.assertIn('skewed', recorded[1]['target_label'])
            for value in recorded:
                self.assertEqual(value['pilots'], {'grid': [], 'selected': {}})
            with (root/'joint_radial_candidates.csv').open(newline='', encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 16)
            self.assertEqual(float(rows[0]['total_sampling_seconds']), 2.123456789012345)
            self.assertEqual(json.loads(rows[2]['selected_production_cases']), ['DCS-BW-joint'])
            self.assertEqual(json.loads(rows[10]['selected_step_endpoint_cases']), ['DCS-BW-joint'])
            with (root/'joint_radial_selections.csv').open(newline='', encoding='utf-8') as stream:
                selections = list(csv.DictReader(stream))
            self.assertEqual(len(selections), 1)
            self.assertEqual(selections[0]['step_endpoint'], 'True')
            self.assertEqual(selections[0]['radius_endpoint'], 'False')
            with (root/'joint_radial_boundary.csv').open(newline='', encoding='utf-8') as stream:
                boundaries = list(csv.DictReader(stream))
            self.assertEqual(boundaries[1]['probe_iterations'], '1000')
            self.assertEqual(boundaries[1]['accepted_after_stepout_rate'], '0.32')
            report = (root/'paper_values.html').read_text(encoding='utf-8')
            self.assertIn('symmetric-nu3/paper_values.html', report)
            self.assertIn('skew-nu2/paper_values.html', report)
            self.assertIn('Quick execution check only', report)
            self.assertIn('raw norm has infinite variance', report)
            self.assertIn('must not be added as a partition', report)
            self.assertIn('10.05', report)
            self.assertNotIn('10.04987562112089', report)


if __name__ == '__main__':
    unittest.main()
