"""Compact reports use every retained draw without convergence diagnostics."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from dcs_bench.burnin import discard_burnin
from dcs_bench.engine import save_chain
from dcs_bench.radial_reporting import compact_radial_report, process_radial_chains, summarize_radial
from dcs_bench.references import functional_draws
from dcs_bench.targets import Target


class RadialReportingTests(unittest.TestCase):
    def setUp(self):
        self.target = Target(dict(kind='skew_t', dimension=4, nu=3, alpha={'0': 20, '1': -30}))

    def test_compact_policy_only_defaults_for_requested_experiment(self):
        self.assertTrue(compact_radial_report({'experiment': 'skew-t-200d-baselines'}))
        self.assertFalse(compact_radial_report({'experiment': 'skew-t-200d-nuts'}))
        self.assertFalse(compact_radial_report({'experiment': 'eight-schools'}))
        self.assertFalse(compact_radial_report({'experiment': 'skew-t-200d-baselines', 'reporting': 'full'}))
        self.assertTrue(compact_radial_report({'reporting': 'radial'}))

    def test_all_retained_draws_and_burnin_boundary(self):
        target = self.target
        for nu in (3, .5):
            target = Target(dict(self.target.spec, nu=nu))
            draws = np.random.default_rng(230).standard_t(nu, size=(10, target.d))
            retained, record = discard_burnin(draws, dict(iterations=101, thin=10),
                                              {'burnin_fraction': 1/3})
            self.assertEqual(record['burnin_iterations'], 34)
            self.assertEqual(record['first_retained_iteration'], 40)
            actual = summarize_radial(retained, target, 5.)
            expected = functional_draws(target, retained, 5.)[:, :2].mean(axis=0)
            np.testing.assert_allclose(actual['estimates'], expected, rtol=2e-15, atol=0)
            self.assertEqual(actual['retained_draws'], 7)
            self.assertEqual(actual['target'], target.spec)

    def test_sampling_time_does_not_implicitly_enable_ess(self):
        draws = np.random.default_rng(44).normal(size=(53, self.target.d))
        with patch('dcs_bench.functional_ess.radial_ess', side_effect=AssertionError('No ESS requested')):
            result = summarize_radial(draws, self.target, 5., sampling_seconds=2., ess_enabled=False)
        self.assertNotIn('functional_ess', result)
        np.testing.assert_allclose(result['estimates'],
            functional_draws(self.target, draws, 5.)[:, :2].mean(axis=0), rtol=2e-15)

    def test_norm_work_is_bounded_and_all_draws_are_used(self):
        target = Target(dict(kind='skew_t', dimension=200, nu=3))
        draws = np.random.default_rng(815).normal(size=(11000, 200))
        original = np.linalg.norm
        observed = []

        def bounded_norm(values, *args, **kwargs):
            observed.append(values.nbytes)
            self.assertLessEqual(values.nbytes, 8*1024*1024)
            return original(values, *args, **kwargs)

        expected = np.linalg.norm(draws, axis=1)
        with patch('dcs_bench.radial_reporting.np.linalg.norm', side_effect=bounded_norm):
            actual = summarize_radial(draws, target, 15.)
        self.assertGreater(len(observed), 1)
        np.testing.assert_allclose(actual['estimates'], [(expected >= 15.).mean(), expected.mean()],
                                   rtol=2e-15, atol=0)

    def test_cached_metadata_skips_draw_reads_quantiles_and_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, expected = [], []
            for i, length in enumerate((51, 67, 59)):
                draws = np.random.default_rng(i).standard_t(3., size=(length, self.target.d))
                cached = summarize_radial(draws, self.target, 5.)
                path = Path(tmp)/f'chain-{i}.npz'
                save_chain(path, draws, dict(fingerprint='radial', retained_draws=length, radial_summary=cached))
                paths.append(path)
                expected.append(functional_draws(self.target, draws, 5.)[:, :2].mean(axis=0))
            before = {path: path.read_bytes() for path in paths}
            with patch('dcs_bench.engine.read_chain', side_effect=AssertionError('No physical chain reads')), \
                 patch('dcs_bench.references.mcmc_diagnostics', side_effect=AssertionError('No diagnostics')), \
                 patch('numpy.quantile', side_effect=AssertionError('No quantiles')):
                actual = process_radial_chains(paths, self.target, 5., expected='radial', progress=None)
            np.testing.assert_allclose(actual['estimates'], expected, rtol=2e-15, atol=0)
            self.assertEqual(actual['length'], 51)
            self.assertEqual(actual['processing']['retained_draws'], 177)
            self.assertEqual(actual['processing']['cached_chains'], 3)
            self.assertEqual(actual['processing']['recomputed_chains'], 0)
            self.assertFalse(actual['processing']['diagnostics_computed'])
            self.assertEqual(actual['processing']['temporary_bytes_peak_upper_bound'], 0)
            self.assertIsNone(actual['diagnostics'])
            self.assertEqual(actual['indiv'], [])
            self.assertEqual(actual['quantiles'].shape, (3, 0, 0))
            self.assertEqual(before, {path: path.read_bytes() for path in paths})

    def test_readonly_mapping_streams_correct_nested_burnin_offset(self):
        original = np.random.default_rng(120).normal(size=(12, self.target.d))
        norm = np.linalg.norm
        read = np.fromfile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'mapped.bin'
            with path.open('wb') as stream:
                stream.write(bytes(128))
                original.tofile(stream)
            mapping = np.memmap(path, mode='r', dtype=original.dtype, offset=128, shape=original.shape)
            try:
                raw = np.asarray(mapping)[2:]
                retained, _ = discard_burnin(raw, dict(iterations=101, thin=10), {'burnin_fraction': 1/3})

                def unmapped_norm(block, *args, **kwargs):
                    self.assertFalse(np.shares_memory(block, mapping))
                    return norm(block, *args, **kwargs)

                with patch('dcs_bench.radial_reporting.np.linalg.norm', side_effect=unmapped_norm), \
                     patch('dcs_bench.radial_reporting.np.fromfile', wraps=read) as reader:
                    actual = summarize_radial(retained, self.target, 2.)
                reader.assert_called_once()
                np.testing.assert_allclose(actual['estimates'],
                    functional_draws(self.target, original[5:], 2.)[:, :2].mean(axis=0), rtol=2e-15)
                self.assertEqual(actual['retained_draws'], 7)
            finally:
                mapping._mmap.close()

    def test_copy_on_write_mapping_uses_modified_values(self):
        original = np.ones((10, self.target.d))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'private.bin'
            original.tofile(path)
            mapping = np.memmap(path, mode='c', dtype=original.dtype, shape=original.shape)
            try:
                mapping[3] = 20.
                expected = np.array(mapping)
                with patch('dcs_bench.radial_reporting.np.fromfile', side_effect=AssertionError('Use modified values')):
                    actual = summarize_radial(mapping, self.target, 5.)
                np.testing.assert_allclose(actual['estimates'],
                    functional_draws(self.target, expected, 5.)[:, :2].mean(axis=0), rtol=2e-15)
            finally:
                mapping._mmap.close()

    def test_legacy_and_invalid_caches_recompute_from_original_draws(self):
        draws = np.random.default_rng(19).standard_t(3., size=(101, self.target.d))
        correct = summarize_radial(draws, self.target, 5.)
        bad = []
        for key, value in [('threshold', 4.), ('target', dict(self.target.spec, alpha={})),
                           ('retained_draws', 100), ('schema_version', 2),
                           ('functional_names', ['tail_probability', 'mean_norm_squared']),
                           ('estimates', [float('nan'), 2.]), ('estimates', [2., 2.])]:
            entry = copy.deepcopy(correct)
            entry[key] = value
            bad.append(entry)
        with tempfile.TemporaryDirectory() as tmp:
            # Nonfinite JSON metadata is disallowed by save_chain. Mock that
            # malformed metadata only; the valid physical checkpoint is read.
            path = Path(tmp)/'legacy.npz'
            save_chain(path, draws, dict(fingerprint='legacy', retained_draws=len(draws)))
            from dcs_bench.engine import read_chain
            for cache in [None, *bad]:
                with self.subTest(cache=cache), \
                     patch('dcs_bench.engine.read_chain_metadata', return_value=dict(
                         fingerprint='legacy', retained_draws=len(draws), radial_summary=cache)), \
                     patch('dcs_bench.engine.read_chain', wraps=read_chain) as reader, \
                     patch('dcs_bench.references.mcmc_diagnostics', side_effect=AssertionError('No diagnostics')):
                    actual = process_radial_chains([path], self.target, 5., expected='legacy', progress=None)
                reader.assert_called_once_with(path, 'legacy')
                np.testing.assert_array_equal(actual['estimates'][0], correct['estimates'])
                self.assertEqual(actual['processing']['recomputed_chains'], 1)

    def test_wrong_cohort_is_rejected_even_with_valid_cached_estimates(self):
        draws = np.ones((10, self.target.d))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'wrong.npz'
            save_chain(path, draws, dict(fingerprint='original', retained_draws=10,
                       radial_summary=summarize_radial(draws, self.target, 5.)))
            with self.assertRaisesRegex(ValueError, 'different configuration or source'):
                process_radial_chains([path], self.target, 5., expected='other', progress=None)

    def test_invalid_physical_draws_and_metadata_count_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'target dimension'):
            summarize_radial(np.ones((10, 3)), self.target, 5.)
        with self.assertRaisesRegex(ValueError, 'finite physical'):
            summarize_radial(np.full((10, 4), np.inf), self.target, 5.)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'count.npz'
            save_chain(path, np.ones((10, 4)), dict(retained_draws=11))
            with self.assertRaisesRegex(ValueError, 'Retained draw count'):
                process_radial_chains([path], self.target, 5., progress=None)

    def test_worker_caches_only_post_burnin_and_preserves_sampling_time(self):
        from dcs_bench.engine import read_chain
        from dcs_bench.worker import _run_and_save_chain

        raw = np.arange(40, dtype=float).reshape(10, 4)

        class Runner:
            def run(self, seed, seconds, collector):
                collector.append(raw)
                return collector.finalize(), dict(seed=seed, iterations=101, thin=10,
                    sampling_seconds=.75, warmup_iterations=200)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'chain.npz'
            config = dict(experiment='skew-t-200d-baselines', threshold=5., burnin_fraction=1/3,
                          workers=1)
            job = dict(worker_slot=0, fingerprint='worker-test', scenario=0, seconds=.75)
            chain = dict(index=0, seed=4, path=str(path))
            with patch('builtins.print'), patch('dcs_bench.functional_ess.radial_ess', side_effect=AssertionError('No worker ESS')):
                _run_and_save_chain(job, chain, self.target, config, 'RWM-Gaussian', Runner(), None, {})
            saved, record = read_chain(path, 'worker-test')
            np.testing.assert_array_equal(saved, raw[3:])
            np.testing.assert_allclose(record['radial_summary']['estimates'],
                functional_draws(self.target, saved, 5.)[:, :2].mean(axis=0), rtol=2e-15)
            self.assertEqual(record['radial_summary']['retained_draws'], 7)
            self.assertNotIn('functional_ess', record['radial_summary'])
            self.assertEqual(record['sampling_seconds'], .75)
            self.assertEqual(record['burnin_iterations'], 34)


if __name__ == '__main__':
    unittest.main()
