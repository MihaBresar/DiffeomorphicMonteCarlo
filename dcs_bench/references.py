"""Independent references and exact radial expectations."""
import hashlib
import json
from functools import lru_cache
import numpy as np
from scipy import special, stats, integrate
from .targets import ROOT, Target, read_archive


def skew_t_iid(target, rng, n):
    # First draw the symmetric t vector, THEN apply the normal-CDF skewing.
    # Since a(-y)=-a(y), sign selection gives density 2 t(y) Phi(a(y)).
    y = rng.normal(size=(n, target.d))/np.sqrt(rng.chisquare(target.nu, size=(n, 1))/target.nu)
    arg = (y@target.alpha)*np.sqrt((target.nu+target.d)/(target.nu+np.sum(y*y, axis=1)))
    keep = rng.random(n) < special.ndtr(arg)
    return np.where(keep[:, None], y, -y)


def radial_truth(target, threshold):
    d, nu = target.d, target.nu
    tail = float(stats.f.sf(threshold**2/d, d, nu))
    if nu > 1:
        expected = float(np.exp(.5*np.log(nu)+special.gammaln((d+1)/2)+special.gammaln((nu-1)/2)
                                -special.gammaln(d/2)-special.gammaln(nu/2)))
        return ['tail_probability', 'mean_norm'], np.array([tail, expected]), [0., 0.]
    def integrand(logr):
        lp = (np.log(2)-special.betaln(d/2, nu/2)+d*logr-d/2*np.log(nu)
              -(d+nu)/2*np.logaddexp(0., 2*logr-np.log(nu)))
        return np.logaddexp(0., logr)/np.log(10)*np.exp(lp)
    value, error = integrate.quad(integrand, -50, 150, epsabs=1e-11, limit=300)
    return ['tail_probability', 'mean_log10_one_plus_norm'], np.array([tail, value]), [0., error]


COMPARISON_REFERENCE_VERSION = 'skew-t-three-coordinate-sign-conditioning-fixed-controls-v2'
COMPARISON_FUNCTIONALS = ['tail_probability', 'mean_norm', 'mean_x1', 'x1_tail_probability',
                         'mean_x2', 'x2_left_tail_probability', 'mean_x3', 'x3_tail_probability']


def comparison_reference(config):
    """Independent reference expectations for the eight current comparison metrics.

    Normal-CDF sign selection is integrated out exactly conditional on each
    independent symmetric Student-t draw. The fixed controls below have known
    expectations; they reduce Monte Carlo error without changing the target.
    Their coefficients never depend on sampler outputs or the reference data.
    The returned MCSE measures reference simulation error, not chain error.
    """
    presets = {'quick': 50_000, 'reproduce': 1_000_000, 'paper': 5_000_000}
    settings = dict(target=config['target'], threshold=float(config.get('threshold', 50.)),
                    coordinate_threshold=float(config.get('coordinate_threshold', 3.)),
                    draws=config.get('reference_samples', presets.get(config.get('preset', 'quick'), 50_000)),
                    seed=config.get('reference_seed', 941_873),
                    batch_size=config.get('reference_batch_size', 8192))
    # Copy via JSON: callers cannot mutate a cached reference. Cache is local to
    # this process and versioned; source hashes are also recorded by the runner.
    return json.loads(_comparison_reference_cached(json.dumps(settings, sort_keys=True),
                                                   COMPARISON_REFERENCE_VERSION))


@lru_cache(maxsize=12)
def _comparison_reference_cached(settings_json, version):
    settings = json.loads(settings_json)
    target = Target(settings['target'])
    if target.kind != 'skew_t' or target.nu <= 2.:
        raise ValueError('Comparison mean references require a skew-t target with finite variance')
    # These extra references exploit actual target symmetries, not an assumed
    # resemblance between coordinates. Reject unsupported custom targets rather
    # than silently assigning them the current suite's exact marginal laws.
    if target.d < 3 or target.alpha[2] != 0.:
        raise ValueError('Three-coordinate references require dimension >= 3 and zero skew coefficient for X3')
    x2_symmetric = target.alpha[1] == 0.
    x2_mirrors_x1 = target.alpha[1] == -target.alpha[0]
    if not (x2_symmetric or x2_mirrors_x1):
        raise ValueError('X2 references require zero X2 skew or opposite X1/X2 skew coefficients')
    n_total, batch_size, seed = settings['draws'], settings['batch_size'], settings['seed']
    for name, value, minimum in [('reference_samples', n_total, 2),
                                  ('reference_batch_size', batch_size, 1), ('reference_seed', seed, 0)]:
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f'{name} must be an integer at least {minimum}')
    threshold, coordinate_threshold = settings['threshold'], settings['coordinate_threshold']
    if not np.all(np.isfinite([threshold, coordinate_threshold])) or threshold <= 0.:
        raise ValueError('Reference thresholds must be finite, with positive radial threshold')
    _, radial_values, _ = radial_truth(target, threshold)
    alpha_norm = float(np.linalg.norm(target.alpha))
    direction = target.alpha / alpha_norm if alpha_norm else np.zeros(target.d)
    absolute_t_mean = float(np.exp(.5*np.log(target.nu) + special.gammaln((target.nu-1.)/2.)
                                  - .5*np.log(np.pi) - special.gammaln(target.nu/2.)))
    offsets = np.array([radial_values[0], radial_values[1], direction[0]*absolute_t_mean, 0.])
    coordinate_base_threshold = coordinate_threshold
    coordinate_tail_control = bool(direction[0] > 0. and coordinate_base_threshold > 0.)
    if coordinate_tail_control:
        offsets[3] = float(2.*stats.t.sf(coordinate_base_threshold, target.nu))
    exact = [True, True, False, False]
    if not alpha_norm:
        offsets[2] = 0.
        exact[2] = True
    rng = np.random.default_rng(seed)
    count, mean, squared = 0, np.zeros(4), np.zeros(4)
    compact = target.d >= 2 and not np.any(target.alpha[2:])
    while count < n_total:
        n = min(batch_size, n_total-count)
        if compact:
            normal = rng.standard_normal((n, 2))
            rest = rng.chisquare(target.d-2, size=n) if target.d > 2 else np.zeros(n)
            normal_norm2 = np.sum(normal*normal, axis=1)+rest
            chisquare = rng.chisquare(target.nu, size=n)
            divisor = np.sqrt(chisquare/target.nu)
            y_first = normal[:, 0]/divisor
            projection = (normal@direction[:2])/divisor
            argument = (normal@target.alpha[:2])*np.sqrt((target.nu+target.d)/(chisquare+normal_norm2))
        else:
            normal = rng.standard_normal((n, target.d))
            chisquare = rng.chisquare(target.nu, size=n)
            divisor = np.sqrt(chisquare/target.nu)
            normal_norm2 = np.sum(normal*normal, axis=1)
            y = normal/divisor[:, None]
            y_first = y[:, 0]
            projection = np.einsum('ij,j->i', y, direction)
            argument = np.einsum('ij,j->i', normal, target.alpha)*np.sqrt((target.nu+target.d)/(chisquare+normal_norm2))
        probability = special.ndtr(argument)
        values = np.zeros((n, 4))
        if alpha_norm:
            # Spherical base symmetry makes E[Y] parallel to alpha. The
            # projection onto this direction is a univariate Student t.
            values[:, 2] = direction[0]*(projection*(2.*probability-1.)-np.abs(projection))
        plus_first, minus_first = y_first, -y_first
        values[:, 3] = probability*(plus_first > coordinate_threshold)+(1.-probability)*(minus_first > coordinate_threshold)
        if coordinate_tail_control:
            # The sign-conditioned tail is zero off this absolute-value event.
            # Positive first-coordinate skew gives E[tail] >= P[event]/2, so
            # this fixed coefficient-one control cannot increase its variance.
            values[:, 3] -= np.abs(y_first) > coordinate_base_threshold
        batch_mean = values.mean(axis=0)
        difference = batch_mean-mean
        squared += np.sum((values-batch_mean)**2, axis=0)+difference*difference*count*n/(count+n)
        mean += difference*n/(count+n)
        count += n
    truth = mean+offsets
    mcse = np.sqrt(squared/(count-1)/count)
    univariate_tail = float(stats.t.sf(coordinate_threshold, target.nu))
    if x2_symmetric:
        x2_truth, x2_mcse, x2_exact = [0., univariate_tail], [0., 0.], [True, True]
        x2_relation = 'Zero X2 skew coefficient gives an ordinary univariate Student-t marginal.'
    else:
        # (-X2,-X1,X3,...) preserves both radius and the skewing argument.
        # Therefore X2 has the reflected X1 law, including its left tail.
        x2_truth, x2_mcse, x2_exact = [-truth[2], truth[3]], mcse[2:4], exact[2:4]
        x2_relation = 'X2 has the law of -X1: E[X2]=-E[X1] and P(X2<-c)=P(X1>c).'
    # With zero X3 skew coefficient, opposite signs of the other integrated
    # coordinates pair Phi(a)+Phi(-a)=1, leaving the ordinary t marginal.
    truth = np.r_[truth, x2_truth, 0., univariate_tail]
    mcse = np.r_[mcse, x2_mcse, 0., 0.]
    exact += x2_exact + [True, True]
    if not np.all(np.isfinite(truth)) or not np.all(np.isfinite(mcse)):
        raise FloatingPointError('Nonfinite independent comparison references')
    metadata = dict(functionals=list(COMPARISON_FUNCTIONALS), truth=truth.tolist(), reference_mcse=mcse.tolist(),
                    exact=exact, target=target.spec, threshold=threshold, coordinate_threshold=coordinate_threshold,
                    seed=seed, draws=count, batch_size=batch_size, version=version,
                    source='Independent symmetric Student-t draws with exact conditional sign weighting and fixed known-mean controls',
                    conditioning='Sign selection is integrated out, with no reference MCMC or sampler-dependent tuning.',
                    coordinate_reference_relations={
                        'x2': x2_relation,
                        'x3': 'X3 has an ordinary univariate Student-t marginal: zero mean and exact t survival probability.',
                        'shared_reference_error': 'For mirrored X2, mean and tail references reuse X1 estimates; their reference errors are dependent.'},
                    control_variates={
                        'mean_x1': 'Use rotational symmetry and subtract absolute skew-direction t projection; add analytic absolute-t mean.',
                        'x1_tail_probability': ('Subtract absolute first-coordinate t tail; add exact two-sided t probability.'
                                                if coordinate_tail_control else 'No coordinate tail control applied.'),
                        'x1_tail_control_applied': coordinate_tail_control,
                        'standardized_coordinate_threshold': coordinate_base_threshold,
                        'coefficients': 'Fixed by target parameters before drawing reference samples.'},
                    radial_truth_exact_except_numerical_evaluation=True,
                    mcse_note='MCSE of independent reference estimates; does not include MCMC error. Close MSE rankings require reference error to be negligible.')
    return json.dumps(metadata, allow_nan=False)


def directional_draws(target, draws):
    direction = target.alpha/np.linalg.norm(target.alpha) if np.any(target.alpha) else np.eye(1, target.d)[0]
    projected = draws@direction
    return np.column_stack([projected > 0, np.tanh(projected/np.sqrt(target.nu))])


def functional_draws(target, draws, threshold):
    if target.kind == 'skew_t':
        norm = np.linalg.norm(draws, axis=-1)
        second = norm if target.nu > 1 else np.log10(1+norm)
        return np.column_stack([norm >= threshold, second, directional_draws(target, draws)])
    if target.kind == 'logistic':
        return draws[:, :1]
    return np.column_stack([draws, draws**2])


def posterior_reference(target):
    name = 'garch-garch11' if target.kind == 'garch' else 'eight_schools-eight_schools_noncentered'
    path = ROOT/f'data/posteriordb/reference_posteriors/draws/draws/{name}.json.zip'
    records = read_archive(path)
    chains = np.array([np.array([chain[k] for k in target.names]).T for chain in records])
    return chains, {'source': 'PosteriorDB', 'record': name,
        'revision': json.loads((ROOT/'data/posteriordb/manifest.json').read_text())['revision'],
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'exact': False, 'chains': len(chains), 'draws_per_chain': chains.shape[1]}


def mcmc_diagnostics(chains, names, *, metrics=('ess_bulk', 'ess_tail', 'mcse_mean', 'rhat')):
    import arviz as az
    available = {'ess_bulk', 'ess_tail', 'mcse_mean', 'rhat'}
    if not set(metrics) <= available:
        raise ValueError('Unknown MCMC diagnostic requested')
    # ArviZ expects (chain,draw) for each variable, not one 3-D ndarray.
    valid = [i for i in range(len(names)) if np.all(np.ptp(chains[:, :, i], axis=1) > 0)]
    payload = {names[i]: chains[:, :, i] for i in valid}
    idata = az.from_dict(posterior=payload) if payload else None
    result = {'unavailable_reason': {n: 'At least one retained chain is constant for this functional' for i, n in enumerate(names) if i not in valid}}
    for key, fn in [('ess_bulk', lambda: az.ess(idata, method='bulk')),
                    ('ess_tail', lambda: az.ess(idata, method='tail')),
                    ('mcse_mean', lambda: az.mcse(idata, method='mean')),
                    ('rhat', lambda: az.rhat(idata, method='rank'))]:
        if key not in metrics:
            continue
        if idata is None or (key == 'rhat' and len(chains) < 2):
            result[key] = dict.fromkeys(names)
            continue
        values = fn()
        result[key] = {n: float(values[n]) if i in valid and np.isfinite(values[n]) else None for i, n in enumerate(names)}
    return result


def make_reference(target, config, directory, logistic_reference=None):
    from .radial_reporting import compact_radial_report

    directory.mkdir(parents=True, exist_ok=True)
    q = np.asarray(config['q_levels'])
    if target.kind == 'skew_t' and compact_radial_report(config):
        # Skewing changes signs but not the radial law. These two reference
        # values need neither simulated reference draws nor coordinate quantiles.
        names, truth, errors = radial_truth(target, config['threshold'])
        metadata = {'source': 'Analytic Student-t radial law (unchanged by sign skewing)',
            'seed': None, 'draws': 0, 'exact': target.nu > 1,
            'functional_names': names, 'functional_truth': truth.tolist(),
            'functional_numerical_error': errors, 'functional_reference_mcse': [0., 0.],
            'radial_truth_exact_except_numerical_evaluation': True,
            'quantiles': [], 'reporting': 'radial'}
    elif target.kind == 'skew_t':
        rng = np.random.default_rng(config['reference_seed'])
        selected, directional = [], []
        remaining = config['reference_samples']
        while remaining:
            n = min(remaining, config['reference_batch_size'])
            batch = skew_t_iid(target, rng, n)
            selected.append(batch[:, :2])
            directional.append(directional_draws(target, batch))
            remaining -= n
        values = np.concatenate(selected)
        names, truth, errors = radial_truth(target, config['threshold'])
        directional = np.concatenate(directional)
        if np.any(target.alpha):
            direction_truth = directional.mean(axis=0)
            direction_mcse = directional.std(axis=0, ddof=1)/np.sqrt(len(directional))
        else:
            direction_truth, direction_mcse = np.array([.5, 0.]), np.zeros(2)
        names += ['direction_positive_probability', 'mean_tanh_direction']
        truth = np.r_[truth, direction_truth]
        blocks = np.array_split(values, 20)
        quantiles = np.quantile(values, q, axis=0)
        quantile_mcse = np.std([np.quantile(b, q, axis=0) for b in blocks], axis=0, ddof=1)/np.sqrt(len(blocks))
        metadata = {'source': 'iid normal-CDF-skewed Student t plus analytic F radial law',
            'seed': config['reference_seed'], 'draws': len(values), 'exact': False,
            'functional_names': names, 'functional_truth': truth.tolist(), 'functional_numerical_error': errors,
            'functional_reference_mcse': np.r_[0., 0., direction_mcse].tolist(),
            'radial_truth_exact_except_numerical_evaluation': True,
            'quantiles': quantiles.tolist(), 'quantile_mcse_batch_approximation': quantile_mcse.tolist()}
        np.savez_compressed(directory/'reference.npz', selected_draws=values, directional_draws=directional, quantiles=quantiles, quantile_mcse=quantile_mcse)
    else:
        if target.kind == 'logistic':
            if logistic_reference is None:
                raise FileNotFoundError('Prepare independent logistic references with prepare_reference.py first')
            path = logistic_reference/f'nu-{target.nu:g}.npz'
            meta_path = path.with_suffix('.json')
            metadata = json.loads(meta_path.read_text())
            if metadata['target'] != target.spec:
                raise ValueError('Logistic reference target does not match experiment')
            if config['preset'] == 'paper' and not metadata['quality_passed']:
                raise ValueError('Paper runs require an independent reference that passed the recorded diagnostic thresholds')
            with np.load(path, allow_pickle=False) as archive:
                chains = archive['draws']
            metadata = dict(metadata, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), path=str(path.resolve()))
        else:
            chains, metadata = posterior_reference(target)
        flat = chains.reshape(-1, target.d)
        diagnostics = mcmc_diagnostics(chains, target.names)
        fnames = [target.names[0]] if target.kind == 'logistic' else target.names+[f'{n}^2' for n in target.names]
        fchains = np.stack([functional_draws(target, c, config['threshold']) for c in chains])
        fdiag = mcmc_diagnostics(fchains, fnames)
        metadata.update(diagnostics=diagnostics, functional_names=fnames,
            functional_truth=fchains.mean(axis=(0, 1)).tolist(),
            functional_reference_mcse=[fdiag['mcse_mean'][n] for n in fnames],
            quantiles=np.quantile(flat[:, :2], q, axis=0).tolist(),
            quantile_mcse_across_reference_chains_approximation=(np.std([np.quantile(c[:, :2], q, axis=0) for c in chains], axis=0, ddof=1)/np.sqrt(len(chains))).tolist())
        # Bundle the exact reference used, so plot-only needs no external files.
        np.savez_compressed(directory/'reference.npz', draws=chains)
    metadata.update(target=target.spec, q_levels=q.tolist())
    (directory/'reference.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    return metadata
