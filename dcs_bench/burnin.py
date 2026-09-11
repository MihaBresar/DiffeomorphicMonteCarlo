"""Discard the first third of a sampling run, separately from proposal tuning."""
import numpy as np


def discard_burnin(draws, record, config):
    """Select saved states by their original iteration numbers, exactly once.

    Timed runs use the final actual transition count. Thinning does not change
    the burn-in boundary; a saved state is retained only if its iteration is
    strictly beyond that boundary. Tuning transitions are outside this run.
    """
    fraction = float(config.get('burnin_fraction', 0.))
    if fraction not in (0., 1/3):
        raise ValueError('Supported burn-in fractions are zero and one third')
    if record.get('burnin_applied'):
        if record.get('burnin_fraction') != fraction:
            raise ValueError('A different burn-in policy was already applied')
        return draws, record
    count, thin = int(record['iterations']), int(record['thin'])
    discard = (count+2)//3 if fraction else 0
    saved = np.asarray(draws)
    if len(saved) != count//thin:
        raise ValueError('Saved draws do not match their original iteration indices')
    removed = discard//thin
    result = saved[removed:]
    if len(result) < 4:
        raise RuntimeError('Fewer than four saved draws remain after burn-in; increase the sampling budget')
    metadata = dict(record, burnin_applied=True, burnin_fraction=fraction,
        tuning_iterations=record.get('warmup_iterations', 0),
        burnin_iterations=discard, estimation_iterations=count-discard,
        retained_before_burnin=len(saved), retained_burnin_draws=removed,
        retained_draws=len(result), first_retained_iteration=(removed+1)*thin,
        burnin_policy='First third of generated sampling iterations discarded after separate tuning'
                      if fraction else 'No additional sampling-run burn-in')
    return result, metadata
