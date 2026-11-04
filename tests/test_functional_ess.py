"""ESS targets the actual estimator's series and never invents missing values."""
from contextlib import contextmanager
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import arviz as az
import numpy as np

from dcs_bench.functional_ess import estimate_ess, radial_ess, aggregate_functional_ess
from dcs_bench import timed_radial as timed
from dcs_bench.radial_reporting import summarize_radial, process_radial_chains
from dcs_bench.engine import save_chain
from dcs_bench.targets import Target


class FunctionalESSTests(unittest.TestCase):
    def test_matches_raw_arviz_mean_and_ar1_effective_count(self):
        rng = np.random.default_rng(19137)
        for rho in (0., .8, -.5):
            n = 80000
            noise = rng.normal(size=n)
            values = np.empty(n);values[0] = noise[0]/np.sqrt(1-rho*rho)
            for i in range(1, n):values[i] = rho*values[i-1]+noise[i]
            result = estimate_ess(values, 4.)
            expected = float(az.ess(values[None, :], method='mean'))
            self.assertAlmostEqual(result['ess']/expected, 1., places=12)
            self.assertLess(abs(result['ess']/(n*(1-rho)/(1+rho))-1.), .17)
            self.assertEqual(result['ess_per_sampling_second'], result['ess']/4.)
            self.assertEqual(result['draw_stride'], 1)
            if rho < 0:self.assertGreater(result['ess'], n)

    def test_unavailable_series_and_moments_never_receive_fake_ess(self):
        for values in (np.ones(100), np.zeros(100), np.arange(7.), np.array([np.nan]*10)):
            row = estimate_ess(values, 1.)
            self.assertIsNone(row['ess'])
            self.assertIsNone(row['ess_per_sampling_second'])
            self.assertTrue(row['unavailable_reason'])
        norms = np.arange(1., 101.)
        rows, _ = radial_ess(norms, 50., 2., nu=1.5)
        self.assertIsNone(rows['mean_norm']['ess'])
        self.assertIn('infinite target variance', rows['mean_norm']['unavailable_reason'])
        rows, _ = radial_ess(norms, 50., 2., nu=.5)
        self.assertIn('mean_log10_one_plus_norm', rows)
        self.assertIsNotNone(rows['mean_log10_one_plus_norm']['ess'])
        row = estimate_ess(norms, None)
        self.assertIsNotNone(row['ess'])
        self.assertIsNone(row['ess_per_sampling_second'])

    def test_no_thinning_multiplier_or_partial_success_total(self):
        values = np.random.default_rng(310).normal(size=3000)
        repeated = np.repeat(values, 5)
        full = estimate_ess(repeated, 2., draw_stride=1)
        saved = estimate_ess(repeated[4::5], 2., draw_stride=5)
        self.assertAlmostEqual(saved['ess'], float(az.ess(values[None, :], method='mean')), places=9)
        self.assertLess(full['ess'], 2.*saved['ess'])
        records = [dict(functional_ess={'f': full}, sampling_seconds=2.),
                   dict(functional_ess={'f': saved}, sampling_seconds=3.)]
        summary = aggregate_functional_ess(records, ['f'])['f']
        self.assertEqual(summary['total_ess'], full['ess']+saved['ess'])
        self.assertEqual(summary['ess_per_sampling_second'], summary['total_ess']/5.)
        self.assertEqual(summary['draw_strides'], [1, 5])
        records.append(dict(functional_ess={'f': estimate_ess(np.zeros(20), 2.)}, sampling_seconds=2.))
        summary = aggregate_functional_ess(records, ['f'])['f']
        self.assertEqual((summary['valid_chains'], summary['chains']), (2, 3))
        self.assertIsNone(summary['total_ess'])
        self.assertIsNone(summary['mean_ess'])
        self.assertIsNone(summary['ess_per_sampling_second'])
        self.assertTrue(summary['unavailable_reasons'])

    def test_timed_capture_uses_every_postburn_iteration_and_cleans_temporary_file(self):
        norms = np.tile([1., 8., 2., 3., 9., 4.], 50)
        accepts = np.tile([0., 1.], 150)
        prefix = np.cumsum(np.column_stack((norms>=5., norms, accepts)).reshape(3,100,3).sum(axis=1), axis=0)
        class Fake:
            compile_seconds=0.;warmup_seconds=0.;warmup_acceptance=.5
            scale=.1;implementation='controlled';tuning_metadata={}
            def batch(self):return prefix,norms
        events=[]
        @contextmanager
        def lock():
            events.append('diagnostic_lock')
            yield
        config=dict(algorithm='DCS-BW', dimension=3, nu=3., alpha={}, thin=50,
                    time_budget=1e-12,warmup=100,threshold=5.,ess_enabled=True)
        with tempfile.TemporaryDirectory() as root:
            original = tempfile.TemporaryDirectory
            def temporary(*args, **kwargs):return original(*args, dir=root, **kwargs)
            with patch.object(timed,'_backend',return_value=Fake()) as backend, \
                 patch.object(timed.tempfile,'TemporaryDirectory',side_effect=temporary):
                traces, record=timed.run_timed({},config,1,
                    before_sampling=lambda:events.append('before'),
                    after_sampling=lambda:events.append('after'), diagnostic_lock=lock())
            self.assertEqual(list(Path(root).iterdir()), [])
        self.assertEqual(events,['before','after','diagnostic_lock'])
        self.assertEqual(backend.call_args.args[1]['thin'],1)
        self.assertEqual(len(traces),4)
        np.testing.assert_array_equal(traces[:,1],norms[149::50])
        self.assertEqual(record['discarded_iterations'],100)
        self.assertEqual(record['functional_ess']['mean_norm']['draws'],200)
        self.assertEqual(record['functional_ess']['mean_norm']['draw_stride'],1)
        expected,_=radial_ess(norms[100:],5.,record['seconds'],nu=3.)
        self.assertEqual(record['functional_ess'],expected)
        self.assertEqual(record['functional_ess_saved_trace_stride'],50)
        np.testing.assert_allclose(record['mean_all_iterations'],[(norms[100:]>=5.).mean(),norms[100:].mean()])

    def test_compact_cache_records_ess_for_actual_retained_series(self):
        target=Target(dict(kind='skew_t',dimension=3,nu=3.,alpha={}))
        draws=np.random.default_rng(171).standard_t(3.,size=(301,3))
        cache=summarize_radial(draws,target,3.,sampling_seconds=2.,draw_stride=10,ess_enabled=True)
        self.assertEqual(cache['functional_ess']['mean_norm']['draws'],301)
        self.assertEqual(cache['functional_ess']['mean_norm']['draw_stride'],10)
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'c.npz'
            save_chain(path,draws,dict(retained_draws=301,thin=10,sampling_seconds=2.,radial_summary=cache))
            with patch('dcs_bench.engine.read_chain',side_effect=AssertionError('Use cached ESS')):
                result=process_radial_chains([path],target,3.,ess_enabled=True,progress=None)
            self.assertEqual(result['records'][0]['functional_ess'],cache['functional_ess'])
            self.assertTrue(result['processing']['functional_ess_computed'])


if __name__=='__main__':unittest.main()
