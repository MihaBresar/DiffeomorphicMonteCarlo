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


def reported_summary(dimension=100, backend='jax'):
    config = study.configuration(target=f'skew-t-{dimension}d', comparison=backend)
    prepared = study.load_prepared(config)
    _, truth, _ = radial_truth(Target(config['target']), config['threshold'])
    records = {}
    for method in config['algorithms']:
        scale = _initial_proposal_scale(dict(config, algorithm=method))
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
            if method != 'SCS':
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
                self.assertIn('Reported zero-centre study', page)
                self.assertIn('not equal total', page)
                self.assertIn('maps reused without refitting', page)
                self.assertIn('All four DCS methods use the same frozen zero-centre', page)
                self.assertNotIn('other DCS methods retain the fitted ball map', page)
                if backend == 'numba':
                    self.assertIn('Numba extension', page)
                with (root/'paper_values.csv').open(encoding='utf-8', newline='') as stream:
                    rows = {r['method']: r for r in csv.DictReader(stream)}
                self.assertEqual(set(rows), set(study.METHODS))
                self.assertEqual(float(rows['DCS-BW']['dcs_fixed_move']), study.FROZEN[dimension]['h'])
                self.assertIn('fixed h = 0.1', rows['SCS']['production_step_policy'])
                self.assertEqual(summary, before)

    def test_changed_maps_or_fit_history_cannot_be_labelled_reported_study(self):
        for mutate in (
            lambda d: d['maps']['ball']['mu'].__setitem__(0, .1),
            lambda d: d['maps']['sphere'].__setitem__('scale', 1.),
            lambda d: d['fitting']['ball']['loss'].__setitem__(0, 0.),
        ):
            summary = reported_summary()
            mutate(summary)
            with tempfile.TemporaryDirectory() as temp, self.assertRaisesRegex(ValueError, 'provenance changed'):
                render_report(Path(temp), summary)


if __name__ == '__main__':
    unittest.main()
