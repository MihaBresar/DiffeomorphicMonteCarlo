"""Verify restored outputs and exact numerical exports from stored checkpoints."""
import run_experiment
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from dcs_bench.engine import load_config, save_chain, write_json
from dcs_bench.references import make_reference, functional_draws
from dcs_bench.targets import Target
from dcs_bench.plotting import rebuild, number, method_labels, ball_map_description


class FigureTests(unittest.TestCase):
    def test_variant_labels_and_historical_scale_description(self):
        config = load_config('skew-t-100d-fixed', 'quick', {'ball_map_policy': 'vanilla'})
        labels = method_labels(['DCS-BW', 'SCS'], config)
        self.assertIn('vanilla', labels[0])
        self.assertNotIn('vanilla', labels[1])
        self.assertIn('R=1, centre=0, Möbius=0', ball_map_description(config))
        legacy = dict(config, ball_scale_policy='sqrt-d')
        legacy.pop('ball_map_policy')
        self.assertIn('R=√d', method_labels(['DCS-BW'], legacy)[0])
        self.assertIn('historical', ball_map_description(legacy))
        posterior = load_config('eight-schools', 'quick', {'ball_map_policy': 'vanilla'})
        self.assertEqual(ball_map_description(posterior), 'posterior identity map')
        self.assertNotIn('vanilla', method_labels(['DCS-BW'], posterior)[0])

    def test_original_figure_set_and_values_from_unchanged_checkpoints(self):
        config = load_config('skew-t-50d-nu-0.5', 'quick', {'ball_map_policy': 'vanilla'})
        config['targets'][0]['dimension'] = 4
        config['reference_samples'] = 400
        config['reference_batch_size'] = 200
        config['thin'] = 10
        target = Target(config['targets'][0])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root/'configuration.json', config)
            write_json(root/'manifest.json', {'fingerprint': 'saved-chain-test'})
            scenario = root/'scenario-0'
            ref = make_reference(target, config, scenario)
            write_json(scenario/'setup.json', {})
            rng = np.random.default_rng(721)
            expected = {}
            for a in config['algorithms']:
                estimates = []
                for i, n in enumerate([51, 53]):
                    x = rng.standard_t(.5, size=(n, 4))
                    estimates.append(functional_draws(target, x, config['threshold']).mean(axis=0))
                    save_chain(scenario/'chains'/f'{a}-{i:03d}.npz', x,
                        dict(fingerprint='saved-chain-test', seed=i, iterations=n*10,
                             retained_draws=n, sampling_seconds=1.0))
                expected[a] = np.mean((np.array(estimates)-ref['functional_truth'])**2, axis=0)
            before = {p:hashlib.sha256(p.read_bytes()).hexdigest() for p in scenario.rglob('*.npz')}
            with patch('dcs_bench.samplers.run_metropolis', side_effect=AssertionError('No MCMC during replot')):
                rebuild(root)
            for p, digest in before.items():
                self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), digest)
            for suffix in ('', '_marginal_qq', '_ergodic_averages', '_sample_counts'):
                self.assertTrue((root/'figures'/f'SuperInfiniteFirstTest{suffix}.png').exists())
            self.assertTrue((root/'figures/supplementary/SuperInfiniteFirstTest_directional.png').exists())
            with (root/'paper_values.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            for row in rows:
                self.assertEqual(row['ball_map_policy'], 'vanilla')
                k = ref['functional_names'].index(row['functional'])
                self.assertEqual(float(row['mse']), expected[row['algorithm']][k])
                self.assertEqual(int(row['total_iterations']), 1040)
                self.assertEqual(int(row['total_retained_samples']), 104)
                self.assertEqual(float(row['mean_retained_samples']), 52)
            self.assertIn(number(expected['DCS-BW'][0]), (root/'paper_values.html').read_text())
            self.assertIn('vanilla: R=1, centre=0, Möbius=0', (root/'paper_values.html').read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
