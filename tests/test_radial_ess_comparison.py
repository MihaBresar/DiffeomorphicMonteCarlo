"""Current five-method radial reports preserve counts and disclose functional ESS."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dcs_bench.functional_ess import aggregate_functional_ess
from scripts import reproduce_comparison as suite
from scripts import reproduce_scs_100d as runner
from scripts.scs_100d_figures import render_report
from tests.test_billiard_comparison import billiard_summary
import run_experiment


METHODS = ['DCS-RWM', 'DCS-BW', 'SCS', 'DCS-BW-reflect', 'DCS-Billiard']
FUNCTIONALS = ['tail_probability', 'mean_norm']


def radial_summary(backend='jax', vi='on', unavailable=False):
    data = billiard_summary(vi)
    data['config'].update(protocol='timed-sampler-comparison-v7' if backend == 'jax' else 'timed-numba-comparison-v7',
        algorithms=METHODS.copy(), comparison=backend, dcs_backend=backend,
        coordinate_metrics=False, ess_enabled=True, vi=vi)
    data['config'].pop('coordinate_threshold', None)
    data['functionals'] = FUNCTIONALS.copy()
    data['truth'] = data['truth'][:2]
    for name in ('truth','reference_mcse','exact','functionals'):
        data['reference'][name] = data['reference'][name][:2]
    data['cases'] = {name: data['cases'][name] for name in METHODS}
    for method, case in data['cases'].items():
        for field in ('mse','bias_squared','variance_ddof0'):
            case[field] = case[field][:2]
        case['chain_means'] = [row[:2] for row in case['chain_means']]
        for index, record in enumerate(case['records']):
            record['mean_all_iterations'] = record.get('mean_all_iterations', case['chain_means'][index])[:2]
            record['functional_ess_seconds'] = .02
            record['functional_ess'] = {
                key: dict(ess=float(100+index*10+metric*50), ess_per_sampling_second=10.,
                          draws=record['estimation_iterations'], draw_stride=1,
                          estimator='arviz_mean_split_geyer_initial_positive_monotone', unavailable_reason=None)
                for metric,key in enumerate(FUNCTIONALS)}
            if unavailable and method == 'DCS-Billiard' and index == 0:
                record['functional_ess']['tail_probability'].update(ess=None, ess_per_sampling_second=None,
                    unavailable_reason='Constant post-burn-in functional trace; autocorrelation ESS is unidentified')
        case['functional_ess'] = aggregate_functional_ess(case['records'], FUNCTIONALS)
        case['functional_ess_seconds'] = .02*case['chains']
    data['method_maps'] = runner.effective_method_maps(data['config'], data['maps'])
    return data


class RadialEssComparisonTests(unittest.TestCase):
    def test_current_runner_skips_coordinate_reference_and_preserves_radial_contract(self):
        for backend in ('jax','numba'):
            for vi in ('on',):
                config = suite.configuration('paper', comparison=backend, vi=vi)
                self.assertEqual(config['algorithms'], METHODS)
                self.assertFalse(config['coordinate_metrics'])
                self.assertTrue(config['ess_enabled'])
                self.assertNotIn('coordinate_threshold', config)
                self.assertEqual(config['step_tuning_policies'], {method: 'author_scs_fixed_01' if method == 'SCS' else 'fixed_dcs_move_v1' for method in METHODS})
                self.assertIn('Analytic',config['reference_policy'])
                self.assertEqual(config['warmup'],1500)

    def test_both_reports_have_only_radial_figures_and_complete_ess_exports(self):
        for backend,vi in (('jax','on'),('numba','off')):
            data=radial_summary(backend,vi,unavailable=True)
            before=deepcopy(data)
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as directory:
                root=Path(directory);report=render_report(root,data)
                self.assertEqual(report['figures'], ['VsSCS','scs_100d_estimates','scs_100d_runtime',
                    'scs_100d_bias_variance','scs_100d_ess'])
                def rows(name):
                    with (root/name).open(newline='',encoding='utf-8') as source:
                        return list(csv.DictReader(source))
                cases=rows('paper_values.csv')
                self.assertEqual([row['method'] for row in cases], METHODS)
                self.assertTrue(all(not any('mean_x' in key or 'x1_tail' in key for key in row) for row in cases))
                diagnostics=rows('functional_ess.csv')
                self.assertEqual(len(diagnostics),10)
                unavailable_row=next(row for row in diagnostics if row['method']=='DCS-Billiard' and row['functional']=='tail_probability')
                self.assertEqual(unavailable_row['total_ess'],'')
                self.assertEqual(int(unavailable_row['valid_chains']),data['cases']['DCS-Billiard']['chains']-1)
                self.assertIn('Constant',unavailable_row['unavailable_reasons'])
                self.assertFalse((root/'angular_tuning.csv').exists())
                self.assertEqual({row['method'] for row in rows('reflected_tuning.csv')},{'DCS-BW-reflect','DCS-Billiard'})
                page=(root/'paper_values.html').read_text()
                self.assertIn('five samplers',page)
                self.assertIn('Functional ESS and ESS per second',page)
                self.assertTrue('ESS alone does not establish convergence' in page)
                self.assertNotIn('coordinate_x1',page)
                self.assertEqual(data,before)

    def test_v7_replots_route_without_sampling(self):
        for backend in ('jax','numba'):
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                (root/'summary.json').write_text(json.dumps(radial_summary(backend)))
                with patch('scripts.scs_100d_figures.render_report') as render, patch.object(run_experiment,'run') as run:
                    self.assertEqual(run_experiment.main(['--plot-only',str(root)]),0)
                    render.assert_called_once();run.assert_not_called()


if __name__=='__main__':unittest.main()
