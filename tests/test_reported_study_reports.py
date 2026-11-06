"""Reported-study figures retain exact steps and distinguish frozen fit provenance."""
from copy import deepcopy
import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from dcs_bench import reported_study as study
from dcs_bench.references import radial_truth
from dcs_bench.targets import Target
from dcs_bench.timed_radial import _initial_proposal_scale
from scripts import reproduce_scs_100d as runner
from scripts.scs_100d_figures import render_report


def reported_summary(dimension=100, backend='jax', legacy=False, version=None):
    configure = (study.legacy_configuration if legacy else
                 study.calibrated_configuration if version == 'v2' else
                 study.zero_configuration if version == 'v3' else study.configuration)
    config = configure(target=f'skew-t-{dimension}d', comparison=backend)
    prepared = study.load_prepared(config)
    _, truth, _ = radial_truth(Target(config['target']), config['threshold'])
    records = {}
    for method in config['algorithms']:
        method_config = runner.method_execution_config(config, method)
        scale = _initial_proposal_scale(method_config)
        records[method] = []
        for chain in range(config['runs']):
            row = dict(chain=chain, seed=config['seed']+chain, iterations=1200,
                discarded_iterations=400, estimation_iterations=800, retained_draws=16,
                seconds=.1, acceptance=.5, final_scale=scale, initial_scale=scale,
                frozen_scale=scale, warmup_iterations=1500, target_acceptance=None,
                adaptation_during_sampling=False, step_size_adapted=False,
                tuning_policy=config['step_tuning_policies'][method],
                mean_all_iterations=(truth+np.array([.001, .1])*(-1 if chain else 1)).tolist(),
                functional_ess_seconds=.02,
                functional_ess={name: dict(ess=100., ess_per_sampling_second=1000.,
                    draws=800, draw_stride=1, unavailable_reason=None)
                    for name in ('tail_probability', 'mean_norm')})
            if method.startswith('DCS-'):
                row.update(fixed_move_radius=config['dcs_fixed_move'], fixed_step=scale,
                    proposal_coordinate_variance=config['dcs_fixed_move']**2/(dimension+2.))
            records[method].append(row)
    summary = runner.summarize(config, prepared['maps'], prepared['fitting'], records)
    summary['prepared_input_provenance'] = prepared['provenance']
    return summary


class ReportedStudyReportTests(unittest.TestCase):
    def test_reports_preserve_five_figures_exact_steps_and_frozen_provenance(self):
        for dimension, backend in ((100, 'jax'), (500, 'numba')):
            summary = reported_summary(dimension, backend)
            before = deepcopy(summary)
            with self.subTest(dimension=dimension, backend=backend), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                result = render_report(root, summary)
                self.assertEqual(len(result['figures']), 5)
                page = (root/'paper_values.html').read_text(encoding='utf-8')
                self.assertIn('<p>Seed 1;', page)
                self.assertNotIn('Seed batch:', page)
                self.assertIn('Current zero-centre comparison', page)
                self.assertIn('Equal pilot counts do not imply equal seconds', page)
                self.assertIn('maps reused without refitting', page)
                self.assertIn('All four DCS methods use the same frozen zero-centre', page)
                self.assertNotIn('other DCS methods retain the fitted ball map', page)
                if backend == 'numba':
                    self.assertIn('Numba extension', page)
                with (root/'paper_values.csv').open(encoding='utf-8', newline='') as stream:
                    rows = {r['method']: r for r in csv.DictReader(stream)}
                self.assertEqual(set(rows), set(study.METHODS))
                self.assertEqual(float(rows['DCS-BW']['dcs_fixed_move']), study.MATCHED[dimension]['DCS-BW']['h'])
                self.assertIn('fixed h = 0.1', rows['SCS']['production_step_policy'])
                calibrated = summary['config']['reported_calibrated_scs_parameters']
                self.assertIn('SCS calibrated', page)
                self.assertIn('All six methods' if backend == 'jax' else 'Numba extension', page)
                self.assertIn('Final-map joint radius/step calibration', page)
                self.assertIn('Equal pilot counts do not imply equal seconds', page)
                self.assertIn('SCS calibrated (zero)', page)
                self.assertIn('final zero-centre, zero-observer map', rows['SCS-calibrated']['production_step_policy'])
                self.assertIn('no map change after selection', rows['SCS-calibrated']['radius_step_preparation'])
                self.assertEqual(float(rows['SCS-calibrated']['centre_norm']), 0.)
                self.assertEqual(float(rows['SCS-calibrated']['observer_norm']), 0.)
                self.assertEqual(float(rows['SCS-calibrated']['fixed_step']), calibrated['h'])
                self.assertEqual(float(rows['SCS-calibrated']['sphere_radius']), calibrated['R'])
                self.assertEqual(rows['SCS-calibrated']['execution_backend'], 'JAX')
                self.assertEqual(float(rows['SCS']['fixed_step']), .1)
                self.assertEqual(float(rows['SCS']['sphere_radius']), summary['maps']['sphere']['scale'])
                for filename in ('sample_counts.csv', 'effective_maps.csv', 'functional_ess.csv'):
                    with (root/filename).open(encoding='utf-8', newline='') as stream:
                        method_rows = list(csv.DictReader(stream))
                    self.assertEqual({r['method'] for r in method_rows}, set(study.METHODS))
                    if filename == 'effective_maps.csv':
                        zero = next(r for r in method_rows if r['method'] == 'SCS-calibrated')
                        original = next(r for r in method_rows if r['method'] == 'SCS')
                        import json
                        self.assertEqual(json.loads(zero['mu']), [0.]*dimension)
                        self.assertEqual(json.loads(zero['observer']), [0.]*dimension)
                        self.assertEqual(json.loads(original['mu']), summary['maps']['sphere']['mu'])
                        self.assertEqual(json.loads(original['observer']), summary['maps']['sphere']['observer'])
                with (root/'joint_calibration.csv').open(encoding='utf-8', newline='') as stream:
                    pilot_rows = list(csv.DictReader(stream))
                self.assertEqual(len(pilot_rows), 2)
                self.assertEqual({int(r['total_pilot_transitions']) for r in pilot_rows}, {524288})
                self.assertEqual({int(r['shared_exact_target_starts']) for r in pilot_rows}, {4096})
                self.assertTrue(all(float(r['preparation_transition_evaluation_seconds']) > 0 for r in pilot_rows))
                self.assertTrue(all(r['parameters_reselected_after_map_change'] == 'True' for r in pilot_rows))
                scs_pilot = next(r for r in pilot_rows if r['method'] == 'SCS calibrated')
                self.assertEqual(scs_pilot['pilot_map_policy'], 'zero_mu_zero_observer')
                self.assertEqual(scs_pilot['production_map_change_after_pilot'], 'none')
                self.assertEqual(summary, before)

    def test_historical_five_method_reports_remain_readable_without_new_control(self):
        summary = reported_summary(legacy=True)
        before = deepcopy(summary)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = render_report(root, summary)
            self.assertEqual(len(result['figures']), 5)
            page = (root/'paper_values.html').read_text(encoding='utf-8')
            self.assertIn('All five methods', page)
            self.assertNotIn('SCS calibrated', page)
            self.assertFalse((root/'joint_calibration.csv').exists())
            self.assertEqual(summary, before)

    def test_historical_v2_keeps_fitted_calibrated_scs_map_and_label(self):
        summary = reported_summary(500, 'numba', version='v2')
        before = deepcopy(summary)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = render_report(root, summary)
            self.assertEqual(len(result['figures']), 5)
            page = (root/'paper_values.html').read_text(encoding='utf-8')
            self.assertIn('SCS calibrated retains the same author centre and observer', page)
            self.assertNotIn('SCS calibrated (zero)', page)
            self.assertNotIn('production centre and observer now zero', page)
            with (root/'paper_values.csv').open(encoding='utf-8', newline='') as stream:
                rows = {r['method']: r for r in csv.DictReader(stream)}
            self.assertIn('original author centre', rows['SCS-calibrated']['production_step_policy'])
            self.assertEqual(float(rows['SCS-calibrated']['fixed_step']), .025)
            self.assertEqual(rows['SCS-calibrated']['execution_backend'], 'JAX')
            self.assertEqual(summary['method_maps']['SCS-calibrated']['mu'], summary['maps']['sphere']['mu'])
            self.assertEqual(summary['method_maps']['SCS-calibrated']['observer'], summary['maps']['sphere']['observer'])
            self.assertEqual(summary, before)

    def test_calibrated_step_records_cannot_silently_revert_to_original_scs(self):
        summary = reported_summary()
        summary['cases']['SCS-calibrated']['records'][0]['final_scale'] = 123.
        with tempfile.TemporaryDirectory() as temp, self.assertRaisesRegex(ValueError, 'Calibrated SCS step records'):
            render_report(Path(temp), summary)

    def test_changed_maps_or_fit_history_cannot_be_labelled_reported_study(self):
        for mutate in (
            lambda d: d['maps']['ball']['mu'].__setitem__(0, .1),
            lambda d: d['maps']['sphere'].__setitem__('scale', 1.),
            lambda d: d['maps']['sphere_calibrated'].__setitem__('scale', 1.),
            lambda d: d['maps']['sphere_calibrated']['mu'].__setitem__(0, .1),
            lambda d: d['maps']['sphere_calibrated']['observer'].__setitem__(0, .1),
            lambda d: d['fitting']['ball']['loss'].__setitem__(0, 0.),
            lambda d: d['prepared_input_provenance']['joint_calibration']['budget_per_family_per_dimension'].__setitem__('transitions_total', 1),
        ):
            summary = reported_summary()
            mutate(summary)
            with tempfile.TemporaryDirectory() as temp, self.assertRaisesRegex(ValueError, 'provenance changed'):
                render_report(Path(temp), summary)


if __name__ == '__main__':
    unittest.main()
