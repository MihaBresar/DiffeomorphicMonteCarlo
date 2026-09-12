"""Lossless, backward-compatible chain archives with bounded compression work."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from dcs_bench.engine import read_chain, read_chain_metadata, save_chain


class ChainStorageTests(unittest.TestCase):
    def test_roundtrip_preserves_values_for_both_storage_branches(self):
        rng = np.random.default_rng(902)
        random = rng.normal(size=(40000, 8))
        random[0, 0] = -0.0
        # A strided input also exercises burn-in/view serialization without a
        # requirement that callers allocate a contiguous copy.
        arrays = [(random[:, ::2], zipfile.ZIP_STORED),
                  (np.ones((5000, 4)), zipfile.ZIP_DEFLATED),
                  (np.asfortranarray(random), zipfile.ZIP_STORED)]
        record = {'fingerprint': 'fixture', 'sampling_seconds': 1.5,
                  'nested': {'seed': 902, 'label': '\u03bc', 'optional': None}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'chain.npz'
            for original, compression in arrays:
                with self.subTest(compression=compression, strides=original.strides):
                    receipt = save_chain(path, original, record)
                    actual, metadata = read_chain(path, 'fixture')
                    self.assertEqual(actual.dtype, original.dtype)
                    self.assertEqual(actual.tobytes(), original.tobytes())
                    self.assertEqual({k: v for k, v in metadata.items() if k != 'storage'}, record)
                    self.assertNotIn('storage', record)
                    self.assertEqual(receipt['archive_bytes'], path.stat().st_size)
                    self.assertEqual(receipt['draw_array_bytes'], original.nbytes)
                    self.assertLessEqual(receipt['compression_sample_bytes'], 1024*1024)
                    self.assertEqual(receipt['compression_sample_bytes'], min(original.nbytes, 1024*1024))
                    self.assertGreaterEqual(receipt['save_seconds'], receipt['draw_serialization_seconds'])
                    with zipfile.ZipFile(path) as archive:
                        self.assertEqual(archive.getinfo('draws.npy').compress_type, compression)
                        self.assertGreaterEqual(archive.getinfo('draws.npy').extract_version, 45)
                    # Standard NumPy readers can consume the new archives.
                    with np.load(path, allow_pickle=False) as archive:
                        np.testing.assert_array_equal(archive['draws'], original)

    def test_original_numpy_archives_remain_readable(self):
        draws = np.arange(40, dtype=np.float64).reshape(10, 4)
        record = {'fingerprint': 'original', 'sampling_seconds': 2.0}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'old.npz'
            np.savez_compressed(path, draws=draws, metadata=np.array(json.dumps(record)))
            actual, metadata = read_chain(path, 'original')
            np.testing.assert_array_equal(actual, draws)
            self.assertEqual(metadata, record)
            self.assertEqual(read_chain_metadata(path, 'original'), record)

    def test_mapped_burnin_views_stream_exact_backing_bytes_with_nonzero_file_offset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for dtype in (np.float64, np.int64, np.bool_):
                with self.subTest(dtype=dtype):
                    original = np.arange(150).reshape(50, 3).astype(dtype)
                    raw_path = root/'raw.bin'
                    prefix = b'x'*137
                    raw_path.write_bytes(prefix+original.tobytes())
                    mapping = np.memmap(raw_path, mode='r', dtype=dtype, offset=len(prefix),
                                        shape=original.shape)
                    try:
                        # ndarray wrappers and multiple slices can retain an
                        # inherited memmap.offset: resolve the deepest mapping.
                        draws = np.asarray(mapping[3:])[5:41]
                        writer = np.lib.format.write_array

                        def metadata_only(member, array, **kwargs):
                            if array.ndim == 2:
                                raise AssertionError('Mapped draws must stream from their backing file')
                            return writer(member, array, **kwargs)

                        path = root/'mapped.npz'
                        with patch('numpy.lib.format.write_array', side_effect=metadata_only):
                            receipt = save_chain(path, draws, {'fingerprint': 'mapped'})
                        self.assertIn('Backing file', receipt['draw_serialization_strategy'])
                        actual, _ = read_chain(path, 'mapped')
                        self.assertEqual(actual.tobytes(), original[8:44].tobytes())
                        with zipfile.ZipFile(path) as archive:
                            with archive.open('draws.npy') as member:
                                self.assertEqual(member.read(8), b'\x93NUMPY\x02\x00')
                        # A column stride is not a contiguous span of file bytes.
                        path = root/'strided.npz'
                        receipt = save_chain(path, np.asarray(mapping)[8:44, ::2],
                                             {'fingerprint': 'strided'})
                        self.assertIn('NumPy', receipt['draw_serialization_strategy'])
                        actual, _ = read_chain(path, 'strided')
                        self.assertEqual(actual.tobytes(), original[8:44, ::2].tobytes())
                    finally:
                        mapping._mmap.close()

    def test_copy_on_write_mapping_saves_modified_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = np.arange(40, dtype=np.float64).reshape(10, 4)
            raw_path = root/'raw.bin'
            raw_path.write_bytes(original.tobytes())
            mapping = np.memmap(raw_path, mode='c', dtype=np.float64, shape=original.shape)
            try:
                mapping[2, 1] = -123.0
                path = root/'copy.npz'
                receipt = save_chain(path, mapping, {'fingerprint': 'copy'})
                self.assertIn('NumPy', receipt['draw_serialization_strategy'])
                actual, _ = read_chain(path, 'copy')
                self.assertEqual(actual[2, 1], -123.0)
                self.assertEqual(raw_path.read_bytes(), original.tobytes())
            finally:
                mapping._mmap.close()

    def test_metadata_only_read_never_loads_draws(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'chain.npz'
            save_chain(path, np.ones((10, 4)), {'fingerprint': 'fixture'})
            original = np.lib.npyio.NpzFile.__getitem__
            accesses = []

            def metadata_only(archive, key):
                accesses.append(key)
                if key != 'metadata':
                    raise AssertionError('Metadata inspection must not read draw values')
                return original(archive, key)

            with patch.object(np.lib.npyio.NpzFile, '__getitem__', metadata_only):
                self.assertEqual(read_chain_metadata(path, 'fixture')['fingerprint'], 'fixture')
                with self.assertRaisesRegex(ValueError, 'different configuration'):
                    read_chain_metadata(path, 'wrong')
            self.assertEqual(accesses, ['metadata', 'metadata'])

    def test_failed_write_preserves_previous_archive_and_cleans_temporary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'chain.npz'
            save_chain(path, np.ones((10, 4)), {'fingerprint': 'fixture'})
            before = path.read_bytes()
            with patch('numpy.lib.format.write_array', side_effect=OSError('disk write failed')):
                with self.assertRaisesRegex(OSError, 'disk write failed'):
                    save_chain(path, np.zeros((10, 4)), {'fingerprint': 'fixture'})
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(Path(tmp).iterdir()), [path])


if __name__ == '__main__':
    unittest.main()
