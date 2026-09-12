"""Bounded-memory collection of complete sampler output on local disk.

The owning context must outlive every array/view returned by ``finalize``:
sample, discard burn-in, and save the result inside the context. Closing the
context invalidates these views and releases the mapping before deleting its
temporary directory, including on Windows.
"""
from pathlib import Path
import tempfile
import time

import numpy as np


class DiskDrawCollector:
    """Append draw batches without retaining them or concatenating a full chain.

    A fixed-size write buffer amortizes the small native sampler batches. Data
    is actually written while appending rather than reserving a sparse mapping,
    so exhausted disk space raises a normal write error. ``flush`` belongs
    inside a sampling timer when output collection is part of that timer.
    ``finalize`` then exposes those exact bytes as a read-only NumPy array.
    """

    def __init__(self, dimension, directory=None, buffer_bytes=8*1024*1024,
                 dtype=np.float64):
        self.dimension = int(dimension)
        self.buffer_bytes = int(buffer_bytes)
        self.dtype = np.dtype(dtype)
        if self.dimension != dimension or self.dimension < 1:
            raise ValueError('Draw dimension must be a positive integer')
        if self.buffer_bytes != buffer_bytes or self.buffer_bytes < 1:
            raise ValueError('Draw write buffer must be a positive integer')
        if self.dtype.kind not in 'bfiu':
            raise ValueError('Draw storage requires a real numeric or Boolean dtype')
        if directory is not None:
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
        self._temporary = tempfile.TemporaryDirectory(prefix='.draw-spool-', dir=directory)
        self.directory = Path(self._temporary.name)
        self.path = self.directory/'draws.bin'
        self._stream = None
        self._mapping = None
        self._closed = False
        self._finalized = False
        self._failed = False
        self.rows = 0
        self.finalization_seconds = 0.0
        try:
            self._stream = self.path.open('wb', buffering=self.buffer_bytes)
        except BaseException:
            self._temporary.cleanup()
            raise

    def __enter__(self):
        if self._closed:
            raise RuntimeError('Draw collector is closed')
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # A disk-full failure can recur while closing a buffered stream. Do not
        # mask the original error, but still release the file and directory.
        try:
            self.close()
        except OSError:
            if exc_type is None:
                raise
        return False

    def _check_writable(self):
        if self._closed:
            raise RuntimeError('Draw collector is closed')
        if self._failed:
            raise RuntimeError('Draw collector cannot continue after a failed write')
        if self._finalized:
            raise RuntimeError('Cannot append to finalized draw storage')

    def append(self, batch):
        """Store all rows, validating values in bounded blocks.

        Native batches are already contiguous and incur no contiguous copy.
        Strided inputs copy at most one buffer-sized block at a time.
        """
        self._check_writable()
        batch = np.asarray(batch)
        if batch.ndim != 2 or batch.shape[1] != self.dimension:
            raise ValueError(f'Draw batches must have shape (rows, {self.dimension})')
        if batch.dtype != self.dtype:
            raise ValueError(f'Draw dtype {batch.dtype} differs from storage dtype {self.dtype}')
        rows_per_block = max(1, self.buffer_bytes//(self.dimension*self.dtype.itemsize))
        try:
            for start in range(0, len(batch), rows_per_block):
                block = batch[start:start+rows_per_block]
                if not np.all(np.isfinite(block)):
                    raise FloatingPointError('Nonfinite retained draws cannot be stored')
                block = np.ascontiguousarray(block)
                data = memoryview(block).cast('B')
                written = self._stream.write(data)
                if written != len(data):
                    raise OSError('Incomplete write to temporary draw storage')
                self.rows += len(block)
        except BaseException:
            self._failed = True
            raise

    def flush(self):
        """Write the final partial buffer; call before stopping a sampling timer."""
        if self._closed:
            raise RuntimeError('Draw collector is closed')
        if self._failed:
            raise RuntimeError('Draw collector cannot continue after a failed write')
        if not self._finalized:
            try:
                self._stream.flush()
            except BaseException:
                self._failed = True
                raise

    def finalize(self):
        """Return an ndarray-compatible mapping without a full-chain allocation."""
        if self._closed:
            raise RuntimeError('Draw collector is closed')
        if self._failed:
            raise RuntimeError('Draw collector cannot finalize after a failed write')
        if self._finalized:
            return self._mapping
        started = time.perf_counter()
        self.flush()
        self._stream.close()
        self._stream = None
        expected_bytes = self.rows*self.dimension*self.dtype.itemsize
        if self.path.stat().st_size != expected_bytes:
            self._failed = True
            raise OSError('Temporary draw storage has an incomplete byte count')
        if self.rows:
            self._mapping = np.memmap(self.path, dtype=self.dtype, mode='r',
                                      shape=(self.rows, self.dimension), order='C')
        else:
            self._mapping = np.empty((0, self.dimension), dtype=self.dtype)
            self._mapping.flags.writeable = False
        self._finalized = True
        self.finalization_seconds = time.perf_counter()-started
        return self._mapping

    def metadata(self):
        """Describe collection separately from sampling and archive serialization."""
        return {'collection_strategy': 'buffered temporary binary file; read-only memory map',
                'collection_buffer_bytes': self.buffer_bytes,
                'collection_rows': self.rows,
                'collection_bytes': self.rows*self.dimension*self.dtype.itemsize,
                'collection_finalization_seconds': self.finalization_seconds,
                'collection_timing_policy': 'Appends and final buffer flush belong inside the '
                                            'sampling timer; mapping finalization is separate',
                'collection_thinning_changed': False}

    def close(self):
        """Release mappings before removing files, even after an interrupted write."""
        if self._closed:
            return
        self._closed = True
        try:
            if self._stream is not None:
                self._stream.close()
        finally:
            self._stream = None
            try:
                if isinstance(self._mapping, np.memmap):
                    self._mapping._mmap.close()
            finally:
                self._mapping = None
                self._temporary.cleanup()
