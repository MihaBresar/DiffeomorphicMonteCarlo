"""Disk collection preserves every draw and has an explicit owned lifetime."""
import errno
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from dcs_bench.burnin import discard_burnin
from dcs_bench.draw_storage import DiskDrawCollector
from dcs_bench.engine import read_chain, save_chain


class DrawStorageTests(unittest.TestCase):
    def test_batches_roundtrip_without_concatenating_and_cleanup_with_live_views(self):
        original = np.arange(300*8, dtype=np.float64).reshape(300, 8)[:, ::2]
        original[0, 0] = -0.0
        with tempfile.TemporaryDirectory() as tmp:
            with DiskDrawCollector(4, directory=tmp, buffer_bytes=257) as collector:
                path = collector.path
                for start, stop in [(0, 0), (0, 71), (71, 215), (215, 300)]:
                    collector.append(original[start:stop])
                collector.flush()
                self.assertEqual(path.stat().st_size, original.nbytes)
                with patch('numpy.concatenate', side_effect=AssertionError('No full-chain concatenation')):
                    actual = collector.finalize()
                self.assertIsInstance(actual, np.memmap)
                self.assertFalse(actual.flags.writeable)
                self.assertEqual(actual.tobytes(), original.tobytes())
                self.assertIs(collector.finalize(), actual)
                view = np.asarray(actual)[100:]
                self.assertTrue(np.shares_memory(view, actual))
                with self.assertRaisesRegex(RuntimeError, 'finalized'):
                    collector.append(original[:1])
                metadata = collector.metadata()
                self.assertEqual(metadata['collection_rows'], 300)
                self.assertEqual(metadata['collection_bytes'], original.nbytes)
                self.assertEqual(metadata['collection_buffer_bytes'], 257)
            # Keep the Python array/view references alive while verifying that
            # their OS mapping/file handles were released (required on Windows).
            self.assertFalse(path.exists())
            self.assertEqual(list(Path(tmp).iterdir()), [])
            with self.assertRaisesRegex(RuntimeError, 'closed'):
                collector.finalize()

    def test_burnin_and_archive_match_original_iteration_grid_exactly(self):
        original = np.arange(303*3, dtype=np.float64).reshape(303, 3)
        record = {'fingerprint': 'fixture', 'iterations': 3030, 'thin': 10,
                  'warmup_iterations': 100}
        config = {'burnin_fraction': 1/3}
        expected, expected_record = discard_burnin(original, record, config)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'chain.npz'
            with DiskDrawCollector(3, directory=tmp, buffer_bytes=128) as collector:
                for start in range(0, len(original), 17):
                    collector.append(original[start:start+17])
                collector.flush()
                draws, actual_record = discard_burnin(collector.finalize(), record, config)
                self.assertEqual(actual_record, expected_record)
                self.assertTrue(np.shares_memory(draws, collector.finalize()))
                save_chain(path, draws, actual_record)
            actual, metadata = read_chain(path, 'fixture')
            self.assertEqual(actual.tobytes(), expected.tobytes())
            self.assertEqual(metadata['retained_draws'], expected_record['retained_draws'])
            self.assertEqual(metadata['first_retained_iteration'], 1020)

    def test_empty_output_and_shape_dtype_validation(self):
        with DiskDrawCollector(2, dtype=np.float32) as collector:
            with self.assertRaisesRegex(ValueError, 'shape'):
                collector.append(np.ones((4, 3), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, 'dtype'):
                collector.append(np.ones((4, 2), dtype=np.float64))
            collector.append(np.empty((0, 2), dtype=np.float32))
            self.assertEqual(collector.finalize().shape, (0, 2))
            self.assertEqual(collector.metadata()['collection_bytes'], 0)

    def test_nonfinite_values_and_disk_failure_release_temporary_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FloatingPointError):
                with DiskDrawCollector(2, directory=tmp, buffer_bytes=32) as collector:
                    collector.append(np.array([[1., 2.], [3., np.inf]]))
            self.assertEqual(list(Path(tmp).iterdir()), [])
            with self.assertRaisesRegex(OSError, 'No space left'):
                with DiskDrawCollector(2, directory=tmp, buffer_bytes=32) as collector:
                    with patch.object(collector._stream, 'write',
                                      side_effect=OSError(errno.ENOSPC, 'No space left on device')):
                        collector.append(np.ones((4, 2)))
            self.assertEqual(list(Path(tmp).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
