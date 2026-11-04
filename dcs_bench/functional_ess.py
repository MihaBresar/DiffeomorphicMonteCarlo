"""ESS for the actual reported functions, with explicit trace/time provenance.

This is raw-mean ESS, not rank-normalized bulk ESS or the minimum quantile ESS
called ``tail ESS`` by ArviZ. In particular, ESS of the radial-tail probability
is calculated from its binary event indicator. Independent chains are handled
separately; their sum is not a cross-chain convergence diagnostic.
"""
from time import perf_counter

import numpy as np

ESTIMATOR = 'arviz_mean_split_geyer_initial_positive_monotone'
DEFINITION = ('ArviZ method=mean on each raw functional trace, split into two halves; '
    'FFT autocovariances and Geyer initial-positive/monotone sequence. No rank normalization, '
    'thinning multiplier or removal of repeated states. ESS can exceed draw count for anticorrelation.')


def prepare_ess():
    """Initialize the existing diagnostics dependency before production clocks."""
    import arviz  # noqa: F401


def estimate_ess(values, sampling_seconds, *, draw_stride=1, unavailable_reason=None):
    """Estimate one functional's ESS; degenerate traces never earn full ESS."""
    import arviz as az

    series = np.asarray(values, dtype=float)
    if series.ndim != 1:
        raise ValueError('Functional ESS requires one scalar time series')
    if isinstance(draw_stride, bool) or not isinstance(draw_stride, (int, np.integer)) or draw_stride < 1:
        raise ValueError('ESS draw_stride must be a positive integer')
    if sampling_seconds is not None and (not np.isfinite(sampling_seconds) or sampling_seconds <= 0):
        raise ValueError('ESS requires the full positive sampling time')
    result = dict(ess=None, ess_per_sampling_second=None, draws=int(len(series)),
        estimator_draws=int(2*(len(series)//2)), draw_stride=int(draw_stride),
        estimator=ESTIMATOR, unavailable_reason=unavailable_reason)
    if unavailable_reason is not None:
        return result
    if len(series) < 8:
        result['unavailable_reason'] = 'Fewer than eight post-burn-in functional draws'
    elif not np.all(np.isfinite(series)):
        result['unavailable_reason'] = 'Nonfinite post-burn-in functional values'
    elif np.max(series) == np.min(series):
        result['unavailable_reason'] = 'Constant post-burn-in functional trace; autocorrelation ESS is unidentified'
    else:
        # Scale/shift invariance avoids overflow in covariance calculations for
        # otherwise finite heavy-tailed values. The observable is not clipped.
        magnitude = float(np.max(np.abs(series)))
        scaled = series / magnitude if magnitude else series
        value = float(np.asarray(az.ess(scaled[None, :], method='mean')))
        if np.isfinite(value) and value > 0.:
            result['ess'] = value
            result['ess_per_sampling_second'] = value / float(sampling_seconds) if sampling_seconds is not None else None
        else:
            result['unavailable_reason'] = 'The split-chain autocorrelation estimator returned no finite positive ESS'
    return result


def functional_ess(values, names, sampling_seconds, *, draw_stride=1, target=None):
    """Process scalar columns sequentially so FFT scratch is not multiplied."""
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] != len(names):
        raise ValueError('Functional ESS names and columns must agree')
    output = {}
    for index, name in enumerate(names):
        reason = None
        if name == 'mean_norm' and target is not None and target.get('kind') == 'skew_t' \
                and float(target['nu']) <= 2.:
            reason = 'The raw norm has infinite target variance for Student-t nu <= 2; mean ESS is not defined'
        output[name] = estimate_ess(values[:, index], sampling_seconds,
            draw_stride=draw_stride, unavailable_reason=reason)
    return output


def radial_ess(norms, threshold, sampling_seconds, *, draw_stride=1, nu=None):
    """Compute the event and moment sequentially without an N-by-2 copy."""
    radii = np.asarray(norms)
    started = perf_counter()
    tail = estimate_ess(radii >= threshold, sampling_seconds, draw_stride=draw_stride)
    name = 'mean_log10_one_plus_norm' if nu is not None and nu <= 1. else 'mean_norm'
    moment = np.log10(1.+radii) if name != 'mean_norm' else radii
    reason = ('The raw norm has infinite target variance for Student-t nu <= 2; mean ESS is not defined'
              if name == 'mean_norm' and nu is not None and nu <= 2. else None)
    norm = estimate_ess(moment, sampling_seconds, draw_stride=draw_stride, unavailable_reason=reason)
    return {'tail_probability': tail, name: norm}, perf_counter()-started


def aggregate_functional_ess(records, names):
    """Never present a sum over only successful chains as an all-chain ESS."""
    records = list(records)
    output = {}
    for name in names:
        entries = [record.get('functional_ess', {}).get(name) for record in records]
        valid = [entry for entry in entries if entry is not None and entry.get('ess') is not None
                 and np.isfinite(entry['ess']) and entry['ess'] > 0]
        reasons = sorted({entry.get('unavailable_reason') or 'ESS missing or invalid'
                          if entry is not None else 'ESS was not recorded for this chain'
                          for entry in entries if entry is None or entry.get('ess') is None})
        result = dict(total_ess=None, mean_ess=None, median_ess=None, ess_per_sampling_second=None,
            valid_chains=len(valid), chains=len(records), unavailable_reasons=reasons,
            draw_strides=sorted({entry['draw_stride'] for entry in entries if entry is not None}),
            estimator=ESTIMATOR,
            aggregation='Sum of separate within-chain split ESS estimates; not pooled-estimator ESS or a cross-chain convergence diagnostic')
        if records and len(valid) == len(records):
            values = [entry['ess'] for entry in valid]
            result.update(total_ess=float(sum(values)), mean_ess=float(np.mean(values)), median_ess=float(np.median(values)))
            times = [record.get('sampling_seconds', record.get('seconds')) for record in records]
            if all(t is not None and np.isfinite(t) and t > 0 for t in times):
                result['ess_per_sampling_second'] = float(sum(values)/sum(times))
            else:
                result['unavailable_reasons'] = ['Full production sampling times are missing or invalid; ESS/time unavailable']
        output[name] = result
    return output
