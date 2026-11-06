"""Small radial summaries for the 200D random-walk comparison.

Every retained draw contributes to the two estimates. Optional functional ESS
uses these same retained scalar series, without quantiles or rank diagnostics.
New workers cache the estimates with each checkpoint; historical checkpoints
remain readable and are summarized one chain at a time.
"""
import math
from time import perf_counter

import numpy as np


_BLOCK_BYTES = 8 * 1024 * 1024


def compact_radial_report(config):
    """Keep the smaller report specific to the requested experiment by default."""
    return config.get('reporting', 'radial' if config.get('experiment') ==
                      'skew-t-200d-baselines' else 'full') == 'radial'


def _context(target, threshold):
    if target.kind != 'skew_t':
        raise ValueError('Radial reports require a skew-t target')
    if not np.isfinite(threshold):
        raise ValueError('The radial threshold must be finite')
    return dict(schema_version=1, target=target.spec, dimension=target.d,
                nu=target.nu, threshold=float(threshold),
                functional_names=['tail_probability', 'mean_norm' if target.nu > 1
                                  else 'mean_log10_one_plus_norm'])


def _draw_blocks(values, rows):
    # Worker output is a read-only mapping. Reading its backing file keeps the
    # mapped pages out of this process's RSS, including post-burn-in subviews.
    from .engine import _mapped_draw_source

    source = _mapped_draw_source(values)
    if source is None:
        for start in range(0, len(values), rows):
            yield values[start:start+rows]
        return
    path, offset = source
    with path.open('rb', buffering=0) as stream:
        stream.seek(offset)
        for start in range(0, len(values), rows):
            count = min(rows, len(values)-start) * values.shape[1]
            block = np.fromfile(stream, dtype=values.dtype, count=count)
            if block.size != count:
                raise OSError('Temporary mapped draw file ended before all radial values were read')
            yield block.reshape(-1, values.shape[1])


def summarize_radial(draws, target, threshold, *, sampling_seconds=None, draw_stride=1, ess_enabled=False):
    """Estimate the tail probability and radial moment in bounded blocks.

The input is already post-burn-in. Block reductions can differ from an older
whole-array mean at floating-point roundoff only; there is no extra thinning.
The worker calls this after its production clock has stopped.
"""
    started = perf_counter()
    context = _context(target, threshold)
    values = np.asarray(draws)
    if values.ndim != 2 or values.shape[1] != target.d or len(values) < 1:
        raise ValueError('Radial summaries require nonempty physical draws with the target dimension')
    rows = max(1, _BLOCK_BYTES // (target.d * values.dtype.itemsize))
    tail_count = 0
    norms = np.empty(len(values)) if ess_enabled else None
    offset = 0

    def moment_sums():
        nonlocal tail_count, offset
        for block in _draw_blocks(values, rows):
            if not np.all(np.isfinite(block)):
                raise ValueError('Radial summaries require finite physical draws')
            norm = np.linalg.norm(block, axis=1)
            if not np.all(np.isfinite(norm)):
                raise ValueError('A radial norm exceeded the finite numerical range')
            tail_count += int(np.count_nonzero(norm >= threshold))
            if norms is not None:
                norms[offset:offset+len(norm)] = norm
                offset += len(norm)
            moment = norm if target.nu > 1 else np.log10(1+norm)
            yield float(np.sum(moment, dtype=np.float64))

    moment_total = math.fsum(moment_sums())
    result = dict(context, estimates=[tail_count / len(values), moment_total / len(values)],
                retained_draws=len(values), seconds=perf_counter()-started,
                reduction='All retained draws; norms in blocks of at most 8 MiB of physical values; '
                          'read-only contiguous mappings streamed from their backing files')
    if norms is not None:
        from .functional_ess import radial_ess, DEFINITION
        ess, seconds = radial_ess(norms, threshold, sampling_seconds, draw_stride=draw_stride, nu=target.nu)
        result.update(functional_ess=ess, functional_ess_seconds=seconds,
            functional_ess_definition=DEFINITION,
            functional_ess_series_policy='Every retained post-burn-in draw used by the reported estimate; actual sampler thinning stride is recorded, never multiplied into ESS')
        result['seconds'] = perf_counter()-started
    return result


def _valid_cache(cached, record, target, threshold):
    if not isinstance(cached, dict):
        return False
    if any(cached.get(key) != value for key, value in _context(target, threshold).items()):
        return False
    count = cached.get('retained_draws')
    if isinstance(count, bool) or not isinstance(count, int) or count < 1 or count != record.get('retained_draws'):
        return False
    try:
        estimates = np.asarray(cached['estimates'], dtype=float)
        seconds = float(cached['seconds'])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return (estimates.shape == (2,) and np.all(np.isfinite(estimates))
            and 0 <= estimates[0] <= 1 and estimates[1] >= 0
            and np.isfinite(seconds) and seconds >= 0)


def process_radial_chains(paths, target, threshold, *, expected=None, label='', progress=print, ess_enabled=False):
    """Read cached estimates, falling back to one historical chain at a time."""
    from .engine import read_chain, read_chain_metadata

    paths = list(paths)
    if not paths:
        raise ValueError('At least one chain is required')
    _context(target, threshold)
    started = perf_counter()
    estimates, records, lengths = [], [], []
    cached_chains, largest_chain_bytes = 0, 0
    read_seconds = summary_seconds = 0.
    prefix = f'{label}: ' if label else ''
    for index, path in enumerate(paths):
        tick = perf_counter()
        record = read_chain_metadata(path, expected)
        read_seconds += perf_counter()-tick
        cached = record.get('radial_summary')
        if _valid_cache(cached, record, target, threshold) and (not ess_enabled or 'functional_ess' in cached):
            cached_chains += 1
            if progress is not None:
                progress(f'{prefix}radial summaries {index+1}/{len(paths)} (cached)')
        else:
            if progress is not None:
                progress(f'{prefix}radial summaries {index+1}/{len(paths)} (reading saved draws)')
            tick = perf_counter()
            draws, record = read_chain(path, expected)
            read_seconds += perf_counter()-tick
            largest_chain_bytes = max(largest_chain_bytes, draws.nbytes)
            options = (dict(sampling_seconds=record.get('sampling_seconds'), draw_stride=record.get('thin', 1), ess_enabled=True)
                       if ess_enabled else {})
            cached = summarize_radial(draws, target, threshold, **options)
            summary_seconds += cached['seconds']
            if cached['retained_draws'] != record.get('retained_draws'):
                raise ValueError(f'Retained draw count does not match checkpoint metadata: {path}')
            del draws
        if ess_enabled:
            record = dict(record, **{key: cached[key] for key in
                ('functional_ess', 'functional_ess_seconds', 'functional_ess_definition', 'functional_ess_series_policy')})
        estimates.append(cached['estimates'])
        records.append(record)
        lengths.append(cached['retained_draws'])
    processing = dict(
        strategy='Cached radial estimates; older checkpoints read one chain at a time',
        diagnostics_computed=ess_enabled, functional_ess_computed=ess_enabled, quantiles_computed=False,
        diagnostic_subsampling=False, diagnostic_workers=0,
        chains=len(paths), functionals=2, retained_draws=sum(lengths),
        cached_chains=cached_chains, recomputed_chains=len(paths)-cached_chains,
        largest_chain_array_bytes=largest_chain_bytes,
        largest_cross_chain_array_bytes=0, staged_functional_bytes=0,
        temporary_bytes_peak_upper_bound=0, cross_chain_seconds=0.,
        independent_seconds=0., stage_seconds=0.,
        read_seconds=read_seconds, summary_seconds=summary_seconds,
        total_seconds=perf_counter()-started)
    return dict(estimates=np.asarray(estimates), quantiles=np.empty((len(paths), 0, 0)),
                records=records, diagnostics=None, indiv=[], coordinate_ess=[],
                length=min(lengths), processing=processing)
