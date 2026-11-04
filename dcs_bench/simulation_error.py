"""Cheap, descriptive uncertainty of MSE across independent chain replicates.

These summaries condition on the fitted map. They are not convergence checks
or confidence intervals; rare heavy-tail excursions can invalidate ordinary
variance-based inference about mean-norm MSE.
"""
import numpy as np


def mse_variability(means, truth):
    errors = (np.asarray(means, dtype=float) - np.asarray(truth, dtype=float))**2
    if errors.ndim != 2 or not len(errors) or not np.all(np.isfinite(errors)):
        raise ValueError('Expected finite squared errors for one or more chains')
    totals = errors.sum(axis=0)
    shares = np.divide(errors.max(axis=0), totals, out=np.zeros_like(totals), where=totals > 0)
    mcse = (errors.std(axis=0, ddof=1) / np.sqrt(len(errors))).tolist() if len(errors) > 1 else [None]*errors.shape[1]
    return dict(mse_empirical_mcse=mcse, largest_chain_squared_error_share=shares.tolist())


def paired_mse_differences(means, records, truth, reference='SCS', functionals=None):
    """Align complete, unique seed pairs; never pair by accidental row order."""
    functionals = ['tail_probability', 'mean_norm'] if functionals is None else list(functionals)
    truth = np.asarray(truth, dtype=float)
    if truth.shape != (len(functionals),):
        raise ValueError('One reference value is required per functional')
    def by_seed(name):
        rows = records.get(name, [])
        values = np.asarray(means[name], dtype=float)
        seeds = [row.get('seed') for row in rows]
        if len(rows) != len(values) or any(seed is None for seed in seeds) or len(set(seeds)) != len(seeds):
            return None
        return dict(zip(seeds, (values - truth)**2))
    if reference not in means:
        return []
    baseline = by_seed(reference)
    if not baseline:
        return []
    rows = []
    for method in means:
        if method == reference:
            continue
        candidate = by_seed(method)
        if candidate is None or candidate.keys() != baseline.keys():
            continue
        differences = np.asarray([candidate[seed]-baseline[seed] for seed in sorted(baseline)])
        count = len(differences)
        empirical_mcse = (differences.std(axis=0, ddof=1)/np.sqrt(count)).tolist() if count > 1 else [None]*differences.shape[1]
        for index, metric in enumerate(functionals):
            rows.append(dict(method=method, reference=reference, metric=metric, paired_chains=count,
                             mse_difference=float(differences[:, index].mean()),
                             difference_empirical_mcse=empirical_mcse[index]))
    return rows
