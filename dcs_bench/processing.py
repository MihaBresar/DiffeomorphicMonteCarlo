"""Exact checkpoint summaries with bounded working memory.

Only one physical chain is loaded at a time. Full functional traces are staged
on disk, and cross-chain diagnostics open only one functional at a time. The
saved samples, common-prefix rule, means, quantiles and ArviZ definitions are
unchanged; no extra thinning or diagnostic subsampling is performed.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from pathlib import Path
import errno
import shutil
import tempfile
from time import perf_counter

import numpy as np

from .references import functional_draws, mcmc_diagnostics


@contextmanager
def _mapped_prefix(path, shape, dtype):
    """Close mappings explicitly before temporary-file cleanup on Windows."""
    required = int(np.prod(shape))*np.dtype(dtype).itemsize
    available = shutil.disk_usage(Path(path).parent).free
    if available < required:
        raise OSError(errno.ENOSPC,
            f'Cross-chain diagnostics need {required:,} temporary bytes; '
            f'only {available:,} are available in the results directory', str(path))
    values = np.memmap(path, mode='w+', dtype=dtype, shape=shape)
    try:
        yield values
    finally:
        values._mmap.close()


class _FunctionalStore:
    def __init__(self, directory, count):
        self.directory = Path(directory)
        self.paths = [self.directory/f'functional-{i:03d}.bin' for i in range(count)]
        self.lengths = []
        self.dtype = None

    def append(self, values):
        if values.ndim != 2 or values.shape[1] != len(self.paths):
            raise ValueError('Functional values and diagnostic names do not match')
        if self.dtype is None:
            self.dtype = values.dtype
        elif self.dtype != values.dtype:
            raise ValueError('All saved chains must use the same functional dtype')
        for column, path in enumerate(self.paths):
            # One temporary contiguous column, never a copy of all chains.
            with path.open('ab') as stream:
                np.ascontiguousarray(values[:, column]).tofile(stream)
        self.lengths.append(len(values))

    def diagnostic(self, index, name, length):
        shape = (len(self.lengths), length)
        path = self.paths[index]
        prefix_path = self.directory/'common-prefix.bin'
        with _mapped_prefix(prefix_path, shape, self.dtype) as prefix:
            # Copy bounded blocks from variable-length traces into the common
            # prefix. A single contiguous (chain, draw) mapping lets ArviZ use
            # its ordinary full-sample calculations without a 3-D stack.
            block_size = max(1, (1024*1024)//self.dtype.itemsize)
            with path.open('rb') as source:
                offset = 0
                for chain, chain_length in enumerate(self.lengths):
                    source.seek(offset*self.dtype.itemsize)
                    for start in range(0, length, block_size):
                        count = min(block_size, length-start)
                        block = np.fromfile(source, dtype=self.dtype, count=count)
                        if len(block) != count:
                            raise ValueError('Incomplete temporary functional trace')
                        prefix[chain, start:start+count] = block
                    offset += chain_length
            prefix.flush()
            result = mcmc_diagnostics(prefix[:, :, None], [name])
        prefix_path.unlink()
        # This variable will not be needed again, so release its disk space now.
        path.unlink()
        return result


def _diagnostic_column(item):
    import arviz as az

    chains, index, name, metrics = item
    # Copy inside the worker: at most four contiguous columns exist at once,
    # regardless of the number of functionals or chains.
    values = np.ascontiguousarray(chains[:, :, index])
    result = {'unavailable_reason': {}}
    if not np.all(np.ptp(values, axis=1) > 0):
        result['unavailable_reason'][name] = 'At least one retained chain is constant for this functional'
        result.update({key: {name: None} for key in metrics})
        return result
    operations = {
        'ess_bulk': lambda: az.ess(values, method='bulk'),
        'ess_tail': lambda: az.ess(values, method='tail'),
        'mcse_mean': lambda: az.mcse(values, method='mean'),
        'rhat': lambda: az.rhat(values, method='rank'),
    }
    for key in metrics:
        if key == 'rhat' and len(values) < 2:
            value = None
        else:
            raw = np.asarray(operations[key]()).item()
            value = float(raw) if np.isfinite(raw) else None
        result[key] = {name: value}
    return result


def _independent_diagnostics(chains, names, executor, *,
                             metrics=('ess_bulk', 'ess_tail', 'mcse_mean', 'rhat')):
    if executor is None:
        return mcmc_diagnostics(chains, names, metrics=metrics)
    result = {'unavailable_reason': {}, **{key: {} for key in metrics}}
    items = ((chains, index, name, metrics) for index, name in enumerate(names))
    for entry in executor.map(_diagnostic_column, items):
        for key, values in entry.items():
            result[key].update(values)
    return result


def _summarize_chain(path, target, threshold, q_levels, names, expected, store, executor):
    from .engine import read_chain

    started = perf_counter()
    draws, record = read_chain(path, expected)
    read_seconds = perf_counter()-started
    started = perf_counter()
    values = functional_draws(target, draws, threshold)
    # Preserve the original axis and reduction order, including every retained
    # draw of unequal-length timed chains in their own estimates.
    estimate = values.mean(axis=0)
    quantiles = np.quantile(draws[:, :4], q_levels, axis=0)
    summary_seconds = perf_counter()-started
    started = perf_counter()
    individual = _independent_diagnostics(values[None, ...], names, executor)
    coordinate_ess = None
    if target.kind == 'skew_t':
        coordinate_ess = _independent_diagnostics(
            draws[None, :, :10], target.names[:10], executor, metrics=('ess_bulk',))['ess_bulk']
    independent_seconds = perf_counter()-started
    started = perf_counter()
    store.append(values)
    stage_seconds = perf_counter()-started
    return dict(estimate=estimate, quantiles=quantiles, record=record,
                individual=individual, coordinate_ess=coordinate_ess,
                chain_bytes=draws.nbytes, functional_bytes=values.nbytes,
                read_seconds=read_seconds, summary_seconds=summary_seconds,
                independent_seconds=independent_seconds, stage_seconds=stage_seconds)


def process_chains(paths, target, threshold, q_levels, functional_names, *,
                   expected=None, temporary_parent=None, label='', progress=print, diagnostic_workers=1):
    """Summarize complete saved chains without keeping every chain in RAM.

    ``progress`` accepts one text argument; pass ``None`` to suppress messages.
    Temporary traces live under ``temporary_parent`` when provided and are
    removed on success or exception. ``diagnostic_workers`` permits up to four
    independent functional diagnostics concurrently within the current chain;
    cross-chain calculations remain sequential to bound their working memory.
    Returned arrays contain only estimates and quantiles, not full samples or
    open memory mappings.
    """
    paths, names = list(paths), list(functional_names)
    if not paths or not names:
        raise ValueError('At least one chain and one functional are required')
    if isinstance(diagnostic_workers, bool) or not isinstance(diagnostic_workers, int) or diagnostic_workers < 1:
        raise ValueError('diagnostic_workers must be a positive integer')
    diagnostic_workers = min(4, diagnostic_workers, len(names))
    started = perf_counter()
    timing = dict(read_seconds=0., summary_seconds=0., independent_seconds=0., stage_seconds=0.)
    estimates, quantiles, records, indiv, coordinate_ess = [], [], [], [], []
    largest_chain_bytes = largest_functional_bytes = 0
    prefix = f'{label}: ' if label else ''
    pool = (ThreadPoolExecutor(max_workers=diagnostic_workers)
            if diagnostic_workers > 1 else nullcontext(None))
    with tempfile.TemporaryDirectory(prefix='.processing-', dir=temporary_parent) as temporary, pool as executor:
        store = _FunctionalStore(temporary, len(names))
        for index, path in enumerate(paths):
            if progress is not None:
                progress(f'{prefix}processing saved chain {index+1}/{len(paths)} '
                         '(loading, exact estimates and individual diagnostics)')
            part = _summarize_chain(path, target, threshold, q_levels, names, expected, store, executor)
            estimates.append(part['estimate'])
            quantiles.append(part['quantiles'])
            records.append(part['record'])
            indiv.append(part['individual'])
            if part['coordinate_ess'] is not None:
                coordinate_ess.append(part['coordinate_ess'])
            largest_chain_bytes = max(largest_chain_bytes, part['chain_bytes'])
            largest_functional_bytes = max(largest_functional_bytes, part['functional_bytes'])
            for key in timing:
                timing[key] += part[key]
        length = min(store.lengths)
        staged_bytes = sum(store.lengths)*len(names)*store.dtype.itemsize
        cross_bytes = len(paths)*length*store.dtype.itemsize
        diagnostics = None
        cross_started = perf_counter()
        for index, name in enumerate(names):
            if progress is not None:
                progress(f'{prefix}cross-chain diagnostics {index+1}/{len(names)}: '
                         f'{name} ({length:,} retained draws per chain)')
            entry = store.diagnostic(index, name, length)
            if diagnostics is None:
                diagnostics = {key: {} for key in entry}
            for key, values in entry.items():
                diagnostics[key].update(values)
        cross_seconds = perf_counter()-cross_started
    processing = dict(
        strategy='One saved chain at a time; disk-staged full traces; one cross-chain functional at a time',
        diagnostic_subsampling=False,
        diagnostic_prefix_policy='Only cross-chain diagnostics use the shortest retained chain length',
        chains=len(paths), functionals=len(names), retained_draws=sum(store.lengths),
        diagnostic_workers=diagnostic_workers,
        largest_chain_array_bytes=largest_chain_bytes,
        largest_functional_array_bytes=largest_functional_bytes,
        largest_cross_chain_array_bytes=cross_bytes,
        staged_functional_bytes=staged_bytes,
        temporary_bytes_peak_upper_bound=staged_bytes+cross_bytes,
        cross_chain_seconds=cross_seconds, total_seconds=perf_counter()-started,
        **timing)
    return dict(estimates=np.asarray(estimates), quantiles=np.asarray(quantiles),
                records=records, diagnostics=diagnostics, indiv=indiv,
                coordinate_ess=coordinate_ess, length=length, processing=processing)
