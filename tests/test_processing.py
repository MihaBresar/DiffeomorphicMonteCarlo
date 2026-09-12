"""Full-sample statistical parity and bounded checkpoint processing."""
import gc
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import numpy as np

from dcs_bench.engine import read_chain, save_chain
from dcs_bench.processing import process_chains
from dcs_bench.references import functional_draws, mcmc_diagnostics


class ProcessingTests(unittest.TestCase):
    def assert_diagnostics_equal(self, actual, expected):
        self.assertEqual(actual.keys(), expected.keys())
        self.assertEqual(actual['unavailable_reason'], expected['unavailable_reason'])
        for key in actual.keys()-{'unavailable_reason'}:
            self.assertEqual(actual[key].keys(), expected[key].keys())
            for name, value in expected[key].items():
                if value is None:
                    self.assertIsNone(actual[key][name])
                else:
                    np.testing.assert_allclose(actual[key][name], value, rtol=2e-13, atol=1e-13)

    def test_exact_estimates_quantiles_and_full_sample_diagnostics(self):
        target = SimpleNamespace(kind='eight_schools', d=3, names=['a', 'b', 'c'])
        names = target.names+[f'{n}^2' for n in target.names]
        rng = np.random.default_rng(7621)
        chains = [rng.normal(size=(length, target.d)) for length in [97, 113, 101]]
        chains[1][:, 0] = 3.  # Invalid bulk/tail ESS and R-hat must stay invalid.
        functions = [functional_draws(target, x, 5.) for x in chains]
        expected = mcmc_diagnostics(np.stack([x[:97] for x in functions]), names)
        individual = [mcmc_diagnostics(x[None, ...], names) for x in functions]
        probabilities = [.05, .25, .5, .75, .95]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [root/f'chain-{i}.npz' for i in range(len(chains))]
            records = []
            for i, (path, chain) in enumerate(zip(paths, chains)):
                record = dict(seed=i, retained_draws=len(chain), fingerprint='parity')
                save_chain(path, chain, record)
                records.append(read_chain(path)[1])
            before = {path: path.read_bytes() for path in paths}
            messages = []
            actual = process_chains(paths, target, 5., probabilities, names,
                expected='parity', temporary_parent=root, label='DCS-HnR', progress=messages.append)
            np.testing.assert_array_equal(actual['estimates'], np.array([x.mean(axis=0) for x in functions]))
            np.testing.assert_array_equal(actual['quantiles'], np.array([
                np.quantile(x[:, :4], probabilities, axis=0) for x in chains]))
            self.assertEqual(actual['records'], records)
            self.assertEqual(actual['length'], 97)
            self.assertEqual(actual['coordinate_ess'], [])
            self.assert_diagnostics_equal(actual['diagnostics'], expected)
            for result, old in zip(actual['indiv'], individual):
                self.assert_diagnostics_equal(result, old)
            self.assertEqual(len(messages), len(paths)+len(names))
            self.assertIn('processing saved chain 1/3', messages[0])
            self.assertFalse(actual['processing']['diagnostic_subsampling'])
            self.assertEqual(actual['processing']['retained_draws'], sum(map(len, chains)))
            self.assertEqual(actual['processing']['largest_cross_chain_array_bytes'], 3*97*8)
            for workers in (2, 4, 16):
                threaded = process_chains(paths, target, 5., probabilities, names,
                    expected='parity', temporary_parent=root, progress=None,
                    diagnostic_workers=workers)
                np.testing.assert_array_equal(threaded['estimates'], actual['estimates'])
                np.testing.assert_array_equal(threaded['quantiles'], actual['quantiles'])
                self.assert_diagnostics_equal(threaded['diagnostics'], actual['diagnostics'])
                for new, old in zip(threaded['indiv'], actual['indiv']):
                    self.assert_diagnostics_equal(new, old)
                self.assertEqual(threaded['processing']['diagnostic_workers'], min(workers, 4))
            self.assertEqual(list(root.glob('.processing-*')), [])
            for path, content in before.items():
                self.assertEqual(path.read_bytes(), content)

    def test_skew_coordinate_bulk_ess_uses_all_samples(self):
        target = SimpleNamespace(kind='skew_t', d=3, names=['a', 'b', 'c'],
                                 alpha=np.array([1., 0., 0.]), nu=3.)
        names = ['tail', 'norm', 'direction', 'tanh']
        rng = np.random.default_rng(884)
        chains = [rng.standard_t(3., size=(length, 3)) for length in [103, 99]]
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp)/f'chain-{i}.npz' for i in range(len(chains))]
            for path, chain in zip(paths, chains):
                save_chain(path, chain, {})
            actual = process_chains(paths, target, 2., [.25, .5, .75], names,
                                    temporary_parent=tmp, progress=None, diagnostic_workers=4)
            expected = [mcmc_diagnostics(x[None, ...], target.names)['ess_bulk'] for x in chains]
            for new, old in zip(actual['coordinate_ess'], expected):
                for name in target.names:
                    np.testing.assert_allclose(new[name], old[name], rtol=2e-13)

    def test_diagnostic_worker_count_rejects_invalid_values(self):
        for workers in (0, -1, True, 1.5):
            with self.subTest(workers=workers), self.assertRaisesRegex(ValueError, 'positive integer'):
                process_chains(['unused'], SimpleNamespace(kind='logistic'), 1., [.5], ['x'],
                               diagnostic_workers=workers, progress=None)

    def test_previous_chain_is_released_before_next_load(self):
        target = SimpleNamespace(kind='logistic')
        seen = []
        rng = np.random.default_rng(99)

        def tracked_read(path, expected):
            gc.collect()
            self.assertTrue(all(reference() is None for reference in seen))
            draws = rng.normal(size=(40, 2))
            seen.append(weakref.ref(draws))
            return draws, {'retained_draws': len(draws)}

        with tempfile.TemporaryDirectory() as tmp, \
             patch('dcs_bench.engine.read_chain', side_effect=tracked_read):
            result = process_chains(['first', 'second', 'third'], target, 1., [.5], ['x'],
                                    temporary_parent=tmp, progress=None)
            self.assertEqual(len(result['records']), 3)
            gc.collect()
            self.assertTrue(all(reference() is None for reference in seen))

    def test_requested_diagnostics_skip_unused_work_and_validate_names(self):
        import arviz as az

        draws = np.random.default_rng(72).normal(size=(2, 101, 1))
        with patch('arviz.ess', wraps=az.ess) as ess, \
             patch('arviz.mcse', side_effect=AssertionError('MCSE was not requested')), \
             patch('arviz.rhat', side_effect=AssertionError('R-hat was not requested')):
            result = mcmc_diagnostics(draws, ['x'], metrics=('ess_bulk',))
        self.assertEqual(set(result), {'unavailable_reason', 'ess_bulk'})
        self.assertIsInstance(result['ess_bulk']['x'], float)
        ess.assert_called_once()
        self.assertEqual(ess.call_args.kwargs, {'method': 'bulk'})
        with self.assertRaisesRegex(ValueError, 'Unknown MCMC diagnostic requested'):
            mcmc_diagnostics(draws, ['x'], metrics=('not_a_diagnostic',))

    def test_insufficient_scratch_space_is_reported_and_cleaned(self):
        target = SimpleNamespace(kind='logistic')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'chain.npz'
            save_chain(path, np.random.default_rng(61).normal(size=(50, 2)), {})
            with patch('dcs_bench.processing.shutil.disk_usage',
                       return_value=SimpleNamespace(free=0)):
                with self.assertRaisesRegex(OSError, 'Cross-chain diagnostics need'):
                    process_chains([path], target, 1., [.5], ['x'],
                                   temporary_parent=tmp, progress=None)
            self.assertEqual(list(Path(tmp).glob('.processing-*')), [])

    def test_scratch_is_removed_after_diagnostic_failure(self):
        target = SimpleNamespace(kind='logistic')
        rng = np.random.default_rng(17)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [root/f'chain-{i}.npz' for i in range(2)]
            for path in paths:
                save_chain(path, rng.normal(size=(50, 2)), {})

            def fail_cross(chains, names, **kwargs):
                if len(chains) > 1:
                    raise RuntimeError('intentional cross-chain failure')
                return mcmc_diagnostics(chains, names, **kwargs)

            with patch('dcs_bench.processing.mcmc_diagnostics', side_effect=fail_cross):
                with self.assertRaisesRegex(RuntimeError, 'intentional cross-chain failure'):
                    process_chains(paths, target, 1., [.5], ['x'],
                                   temporary_parent=root, progress=None)
            self.assertEqual(list(root.glob('.processing-*')), [])
            for path in paths:
                self.assertEqual(read_chain(path)[0].shape, (50, 2))


if __name__ == '__main__':
    unittest.main()
