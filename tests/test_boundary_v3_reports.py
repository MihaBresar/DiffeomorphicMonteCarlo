"""Keep current pilot reports distinct from historical acceptance calibration."""
from copy import deepcopy
import csv
from pathlib import Path
import tempfile
import unittest

from scripts import reproduce_boundary_100d as boundary
from scripts import reproduce_scs_100d as original
from scripts.scs_100d_figures import render_report
from tests.test_scs_100d_figures import example_summary


def boundary_summary(protocol='timed-boundary-dcs-scs-v3'):
    summary = example_summary()
    summary['config'].pop('iterations')
    summary['config'].update(protocol=protocol, mode='time', time_budget=1.,
        implementation_revision='jax-dcs-chord-reflection-v1')
    summary['cases']['DCS-BW-reflect'] = summary['cases'].pop('DCS-RWM')
    for name, case in summary['cases'].items():
        case['total_iterations'] = 3600
        for row in case['records']:
            row.update(iterations=1200, discarded_iterations=400, estimation_iterations=800,
                retained_draws=80, warmup_iterations=100, target_acceptance=.234,
                scale_ceiling_reached=False)
            row.pop('checkpoints', None)
            if name == 'DCS-BW-reflect' and protocol.endswith('v3'):
                history = []
                for index in range(24):
                    round_index, candidate = divmod(index, 3)
                    history.append({'round': round_index+1, 'candidate': candidate,
                        'scale': .1*2**round_index*2**(candidate-1),
                        'iterations': 4+int(index < 4), 'ball_esjd': .0123456789012345+candidate*.002,
                        'acceptance': .8, 'selected': candidate == 2})
                row.update(tuning_policy='ball_esjd_pilot_v1', target_acceptance=None,
                    step_policy='Ball-movement pilot selection, then frozen',
                    tuning_history=history, tuning_selected_esjd=history[-1]['ball_esjd'],
                    tuning_search_status='upper_candidate_won_final_round',
                    tuning_pilot_iterations=100, tuning_rounds=8,
                    scale=25.6, frozen_scale=25.6, final_scale=25.6)
        if name == 'DCS-BW-reflect' and protocol.endswith('v3'):
            case['final_scales'] = [25.6]*case['chains']
    if protocol.endswith('v1'):
        summary['config']['step_scale_ceiling'] = 2.
    elif protocol.endswith('v2'):
        summary['config']['step_scale_ceilings'] = {'DCS-BW-reflect': 2.}
    else:
        summary['config']['step_tuning_policies'] = {'DCS-BW-reflect': 'ball_esjd_pilot_v1'}
    return summary


class BoundaryV3ReportsTests(unittest.TestCase):
    def test_v3_configuration_removes_geometric_limit_and_preserves_resource_budgets(self):
        main, revised = original.configuration('paper'), boundary.configuration('paper')
        self.assertEqual(revised['protocol'], 'timed-boundary-dcs-scs-v3')
        self.assertEqual(revised['step_tuning_policies'], {'DCS-BW-reflect': 'ball_esjd_pilot_v1'})
        self.assertNotIn('step_scale_ceiling', revised)
        self.assertNotIn('step_scale_ceilings', revised)
        self.assertNotEqual(main['step_policy'], revised['step_policy'])
        for key in ('target', 'runs', 'time_budget', 'warmup', 'optimizer_steps', 'optimizer_samples',
                    'initial_scale', 'burnin_fraction', 'seed', 'fit_seed', 'implementation_revision'):
            if key != 'implementation_revision':
                self.assertEqual(main[key], revised[key])
        self.assertEqual(revised['implementation_revision'], 'jax-dcs-chord-reflection-v1')
        self.assertIn('up to eight rounds', revised['tuning_note'])
        self.assertIn('not a claim of optimal tuning', revised['tuning_note'])

    def test_v3_exports_all_pilots_without_relabeling_controls_or_changing_estimates(self):
        summary = boundary_summary()
        before = deepcopy(summary)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = render_report(root, summary)
            self.assertIn('reflected_tuning.csv', result['csv_files'])
            self.assertEqual(result['figures'], ['VsSCS', 'scs_100d_estimates',
                                                 'scs_100d_runtime', 'scs_100d_bias_variance'])
            with (root/'paper_values.csv').open(newline='', encoding='utf-8') as stream:
                cases = {row['method']: row for row in csv.DictReader(stream)}
            self.assertIn('Ball-movement pilot', cases['DCS-BW-reflect']['production_step_policy'])
            self.assertNotIn('0.234', cases['DCS-BW-reflect']['production_step_policy'])
            for name in ('DCS-BW', 'SCS'):
                self.assertIn('0.234', cases[name]['production_step_policy'])
            for name, row in cases.items():
                self.assertEqual(float(row['mean_norm_mse']), summary['cases'][name]['mse'][1])
                self.assertEqual(int(row['total_iterations']), 3600)
                self.assertEqual(int(row['total_estimation_iterations']), 2400)
            with (root/'reflected_tuning.csv').open(newline='', encoding='utf-8') as stream:
                pilots = list(csv.DictReader(stream))
            self.assertEqual(len(pilots), 72)
            self.assertEqual({row['method'] for row in pilots}, {'DCS-BW-reflect'})
            for chain in range(3):
                rows = [row for row in pilots if int(row['chain']) == chain]
                self.assertEqual(sum(int(row['iterations']) for row in rows), 100)
                self.assertEqual(sum(row['selected'] == 'True' for row in rows), 8)
                for row, expected in zip(rows, summary['cases']['DCS-BW-reflect']['records'][chain]['tuning_history']):
                    for key in ('scale', 'ball_esjd', 'acceptance'):
                        self.assertEqual(float(row[key]), expected[key])
            with (root/'scs_100d_chains.csv').open(newline='', encoding='utf-8') as stream:
                reader = csv.DictReader(stream)
                self.assertNotIn('tuning_history', reader.fieldnames)
                chains = list(reader)
            for row in chains:
                if row['method'] == 'DCS-BW-reflect':
                    self.assertEqual(row['target_acceptance'], '')
                    self.assertEqual(float(row['frozen_scale']), 25.6)
                    self.assertEqual(row['tuning_search_status'], 'upper_candidate_won_final_round')
            page = (root/'paper_values.html').read_text(encoding='utf-8')
            self.assertIn('timed-boundary-dcs-scs-v3', page)
            self.assertIn('Reflected BW pilot tuning and frozen steps', page)
            self.assertIn('reflected_tuning.csv', page)
            self.assertIn('25.60', page)
            self.assertIn('different tuning objective', page)
            self.assertIn('not production MSE', page)
            self.assertNotIn('step ceiling of 2', page)
            self.assertNotIn('All three methods use the same separate dual-averaging', page)
            self.assertNotIn('tuning history', page)
        self.assertEqual(summary, before)

    def test_historical_reports_retain_their_declared_calibration(self):
        for protocol in ('timed-boundary-dcs-scs-v1', 'timed-boundary-dcs-scs-v2'):
            summary = boundary_summary(protocol)
            for row in summary['cases']['DCS-BW-reflect']['records']:
                row['scale_ceiling_reached'] = True
            with self.subTest(protocol=protocol), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                result = render_report(root, summary)
                page = (root/'paper_values.html').read_text(encoding='utf-8')
                self.assertIn('same separate dual-averaging calibration toward acceptance 0.234', page)
                self.assertIn('step-size ceiling', page)
                if protocol.endswith('v1'):
                    self.assertIn('common upper step limit of 2.00', page)
                    self.assertNotIn('method-specific step-size ceiling', page)
                else:
                    self.assertIn('Only reflected BW has a step ceiling of 2', page)
                    self.assertIn('its method-specific step-size ceiling', page)
                self.assertNotIn('Reflected BW pilot tuning', page)
                self.assertNotIn('reflected_tuning.csv', result['csv_files'])


if __name__ == '__main__':
    unittest.main()
