"""Check bounded draw collection and exact archive bytes without running MCMC.

The default is a small synthetic check. To exercise the reported Windows
allocation shape, pass --rows 594918 --dimension 200; this requires about1.6GB
of temporary disk space, checked before starting. All generated arrays/archives
are temporary and removed after validation. Only the JSON receipt is retained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time
import tracemalloc
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np
from benchmark_result_processing import peak_rss_bytes
from dcs_bench.burnin import discard_burnin
from dcs_bench.draw_storage import DiskDrawCollector
from dcs_bench.engine import save_chain


def run(args):
    rows, dimension, thin = args.rows, args.dimension, 10
    batch_rows = min(257, max(1, (1024*1024)//(dimension*8)))
    iterations = rows*thin
    burnin = (iterations+2)//3
    removed, retained = burnin//thin, rows-burnin//thin
    raw_bytes, retained_bytes = rows*dimension*8, retained*dimension*8
    required = raw_bytes+retained_bytes+64*1024*1024
    scratch = args.temporary_directory or Path(tempfile.gettempdir())
    free = shutil.disk_usage(scratch).free
    if free < required:
        raise RuntimeError(f'This check requires {required} free temporary bytes; available {free}')
    source_paths = [Path(__file__), ROOT/'scripts/benchmark_result_processing.py',
                    *(ROOT/'dcs_bench'/name for name in ('draw_storage.py', 'engine.py', 'burnin.py'))]
    source_hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in source_paths}
    record = dict(iterations=iterations, thin=thin, retained_draws=rows,
                  warmup_iterations=100, fingerprint='synthetic-draw-storage-check')
    tile = np.random.default_rng(7418).normal(size=(batch_rows, dimension))
    tile[0, 0] = -0.0
    expected_raw, expected_kept = hashlib.sha256(), hashlib.sha256()
    phases = []
    initial_rss = peak_rss_bytes()
    tracemalloc.start()

    def phase(name, started):
        current, peak = tracemalloc.get_traced_memory()
        phases.append(dict(phase=name, seconds=time.perf_counter()-started,
            process_peak_rss_bytes=peak_rss_bytes(), traced_current_bytes=current,
            traced_peak_bytes=peak))
        print(f'{name}: traced peak {peak/1024**2:.2f} MiB', flush=True)

    with tempfile.TemporaryDirectory(prefix='draw-storage-check-', dir=scratch) as temporary:
        directory = Path(temporary)
        archive_path = directory/'chain.npz'
        with DiskDrawCollector(dimension, directory=directory) as collector:
            with patch('numpy.concatenate', side_effect=AssertionError('Full-chain concatenate forbidden')):
                print(f'Collecting {rows:,}x{dimension} synthetic values ({raw_bytes:,} bytes)', flush=True)
                started = time.perf_counter()
                for start in range(0, rows, batch_rows):
                    count = min(batch_rows, rows-start)
                    batch = tile[:count]
                    collector.append(batch)
                    expected_raw.update(memoryview(batch).cast('B'))
                    keep_start = max(0, removed-start)
                    if keep_start < count:
                        expected_kept.update(memoryview(batch[keep_start:]).cast('B'))
                collector.flush()
                assert collector.path.stat().st_size == raw_bytes
                phase('append_and_flush', started)
                started = time.perf_counter()
                raw = collector.finalize()
                draws, metadata = discard_burnin(raw, record, {'burnin_fraction': 1/3})
                assert raw.shape == (rows, dimension)
                assert draws.shape == (retained, dimension) and draws.nbytes == retained_bytes
                assert metadata['retained_burnin_draws'] == removed
                assert np.shares_memory(raw, draws)
                phase('finalize_and_burnin_views', started)
                started = time.perf_counter()
                storage = save_chain(archive_path, draws, metadata,
                                     temporary_directory=collector.directory)
                assert storage['draw_serialization_strategy'] == 'Backing file streamed in at most 8 MiB blocks'
                phase('save_retained_archive', started)
                started = time.perf_counter()
                actual_raw = hashlib.sha256()
                with collector.path.open('rb', buffering=0) as stream:
                    while True:
                        block = stream.read(1024*1024)
                        if not block:
                            break
                        actual_raw.update(block)
                        del block
                assert actual_raw.hexdigest() == expected_raw.hexdigest()
                actual_kept, data_bytes = hashlib.sha256(), 0
                with zipfile.ZipFile(archive_path) as archive:
                    with archive.open('draws.npy') as member:
                        assert np.lib.format.read_magic(member) == (2, 0)
                        shape, fortran, dtype = np.lib.format.read_array_header_2_0(member)
                        assert shape == (retained, dimension) and not fortran and dtype == np.float64
                        while True:
                            block = member.read(1024*1024)
                            if not block:
                                break
                            values = np.frombuffer(block, dtype=dtype)
                            assert np.all(np.isfinite(values))
                            data_bytes += len(block)
                            actual_kept.update(block)
                            del values, block
                    with archive.open('metadata.npy') as member:
                        saved_metadata = json.loads(str(np.lib.format.read_array(member, allow_pickle=False)))
                assert data_bytes == retained_bytes and actual_kept.hexdigest() == expected_kept.hexdigest()
                assert saved_metadata['retained_draws'] == retained
                phase('stream_validate_raw_and_npz', started)
                archive_bytes, collection = archive_path.stat().st_size, collector.metadata()
                del draws, raw, batch
        assert not collector.directory.exists()
    assert not directory.exists()
    _, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert traced_peak < 64*1024*1024, f'Unexpected large owned allocation: {traced_peak}'
    assert source_hashes == {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in source_paths}, 'Source changed during check'
    return dict(schema_version=1, raw_shape=[rows,dimension], raw_bytes=raw_bytes,
        retained_shape=[retained,dimension], retained_bytes=retained_bytes,
        synthetic_iterations=iterations, burnin_iterations=burnin,
        discarded_saved_rows=removed, archive_bytes=archive_bytes,
        free_temporary_bytes_before=free, required_temporary_bytes=required,
        initial_process_peak_rss_bytes=initial_rss, final_traced_peak_bytes=traced_peak,
        phases=phases, collection=collection, storage=storage,
        exact_raw_sha256=expected_raw.hexdigest(), exact_retained_sha256=expected_kept.hexdigest(),
        all_checks_passed=True, all_large_temporary_files_removed=True, source_sha256=source_hashes,
        platform=platform.platform(), python=sys.version, numpy=np.__version__,
        note='Synthetic repeated finite batches, not an MCMC experiment or sampler timing comparison. '
             'Collected on disk; first-third burn-in and NPZ payload verified blockwise. No full-chain '
             'heap array or boolean mask is created. RSS includes imports and may be unavailable on '
             'some systems. Temporary files are removed; only this receipt is retained.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=int, default=10000)
    parser.add_argument('--dimension', type=int, default=200)
    parser.add_argument('--temporary-directory', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT/'results/draw-storage-check.json')
    args = parser.parse_args()
    if args.rows < 6 or args.dimension < 1:
        parser.error('--rows must be at least6 and --dimension positive')
    if args.output.exists():
        parser.error('Choose a fresh --output path; existing receipts are preserved')
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(f'PASS: {args.output}', flush=True)
