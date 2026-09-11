"""Independent references and exact radial expectations."""
import hashlib
import json
import numpy as np
from scipy import special, stats, integrate
from .targets import ROOT, read_archive


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


def mcmc_diagnostics(chains, names):
    import arviz as az
    # ArviZ expects (chain,draw) for each variable, not one 3-D ndarray.
    valid = [i for i in range(len(names)) if np.all(np.ptp(chains[:, :, i], axis=1) > 0)]
    payload = {names[i]: chains[:, :, i] for i in valid}
    idata = az.from_dict(posterior=payload) if payload else None
    result = {'unavailable_reason': {n: 'At least one retained chain is constant for this functional' for i, n in enumerate(names) if i not in valid}}
    for key, fn in [('ess_bulk', lambda: az.ess(idata, method='bulk')),
                    ('ess_tail', lambda: az.ess(idata, method='tail')),
                    ('mcse_mean', lambda: az.mcse(idata, method='mean')),
                    ('rhat', lambda: az.rhat(idata, method='rank'))]:
        if idata is None or (key == 'rhat' and len(chains) < 2):
            result[key] = dict.fromkeys(names)
            continue
        values = fn()
        result[key] = {n: float(values[n]) if i in valid and np.isfinite(values[n]) else None for i, n in enumerate(names)}
    return result


def make_reference(target, config, directory, logistic_reference=None):
    directory.mkdir(parents=True, exist_ok=True)
    q = np.asarray(config['q_levels'])
    if target.kind == 'skew_t':
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
