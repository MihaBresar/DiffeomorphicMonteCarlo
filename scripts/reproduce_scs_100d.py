"""One 100D skewed Student-t(3) comparison: original DCS-RWM/BW and author SCS.

Fresh runs only. Exact radial estimates use every production iteration, including
rejections. Each method runs alone in synchronized waves at the same concurrency.
"""
import argparse
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dcs_bench.cpu import cpu_info, pin_current_process, validate_workers, worker_environment
os.environ.update(worker_environment())
import numpy as np
from dcs_bench.engine import environment, write_json
from dcs_bench.references import radial_truth
from dcs_bench.targets import Target
from dcs_bench.transforms import BallMap

METHODS = ('DCS-RWM', 'DCS-BW', 'SCS')
TARGET = {'kind': 'skew_t', 'dimension': 100, 'nu': 3., 'alpha': {'0': 100., '1': -100.}}
PRESETS = {
    'quick': dict(runs=2, iterations=500, thin=1, optimizer_steps=20, optimizer_samples=128,
                  scs_optimizer_steps=20, scs_optimizer_samples=128),
    'reproduce': dict(runs=20, iterations=25000, thin=10, optimizer_steps=400, optimizer_samples=500,
                      scs_optimizer_steps=400, scs_optimizer_samples=500),
    'paper': dict(runs=50, iterations=500000, thin=50, optimizer_steps=1500, optimizer_samples=2000,
                  scs_optimizer_steps=1000, scs_optimizer_samples=2000),
}
TIME_SCOPE = ('Elapsed production time per chain includes proposals, acceptance, continued DCS step adaptation, '
              'and all-iteration radial estimation. Excludes imports, compilation, map fitting, warmup, '
              'barrier waiting, file writing and plots. Methods run separately at the same wave size; '
              'method order rotates between waves. This is fixed iterations, not a fixed-time run.')
_ALLOCATION = {}


def configuration(preset='quick', workers=1, overrides=None):
    config = dict(PRESETS[preset], preset=preset, workers=workers, warmup=100,
                  experiment='skew-t-100d-fixed', protocol='original-dcs-author-scs-v1',
                  dimension=100, nu=3., alpha=TARGET['alpha'], target=TARGET,
                  beta=2., latitude=1.1, threshold=50., seed=901001, algorithms=list(METHODS),
                  dcs_step_policy='original_smoothed_continuing_0.234', scs_step=.1)
    allowed = {'runs', 'iterations', 'warmup', 'thin', 'optimizer_steps', 'optimizer_samples',
               'scs_optimizer_steps', 'scs_optimizer_samples', 'seed', 'workers'}
    for key, value in (overrides or {}).items():
        if value is None:
            continue
        if key not in allowed:
            raise ValueError(f'{key} is not an override for the single SCS comparison')
        config[key] = value
    for key in allowed-{'seed', 'warmup'}:
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('seed', 'warmup'):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 0:
            raise ValueError(f'{key} must be a nonnegative integer')
    if config['thin'] > config['iterations']:
        raise ValueError('thin cannot exceed iterations')
    validate_workers(config['workers'])
    return config


def source_hashes():
    paths = list((ROOT/'dcs_bench').glob('*.py')) + list((ROOT/'third_party/scs_paper').rglob('*.py'))
    paths += [Path(__file__), ROOT/'scripts/legacy_fixed_step_loader.py', ROOT/'validation/legacy_fixed_step.py',
              ROOT/'scripts/scs_100d_figures.py', ROOT/'requirements-cpu.txt']
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def prepare(config):
    from scripts.legacy_fixed_step_loader import load_legacy
    from dcs_bench.scs_paper import fit
    print('Fitting the original DCS map (shared by RWM and BW)', flush=True)
    started = time.perf_counter()
    legacy = load_legacy(nu=3., dimension=100, optimizer_steps=config['optimizer_steps'],
                         optimizer_samples=config['optimizer_samples'])
    mu, delta, scale = legacy['optimize_prwm_hyperparameters']()
    ball = BallMap(100, beta=2., mu=mu, delta=delta, scale=scale).parameters()
    ball_metadata = dict(seconds=time.perf_counter()-started, seed=42,
                         steps=config['optimizer_steps']+1, samples_per_step=config['optimizer_samples'],
                         training_batch_policy='fresh', source='validation/legacy_fixed_step.py, VI function only',
                         note='Original VI clips its CDF argument to [-35,35]. Production density is unclipped.')
    print('Fitting SCS with the original author VI on the shared target', flush=True)
    sphere, sphere_metadata = fit({**config, 'optimizer_steps': config['scs_optimizer_steps'],
                                  'optimizer_samples': config['scs_optimizer_samples']})
    return {'ball': ball, 'sphere': sphere}, {'ball': ball_metadata, 'sphere': sphere_metadata}


def initialize_worker(slots):
    global _ALLOCATION
    _ALLOCATION = pin_current_process(slots.get())


def chain_job(job):
    barrier = job['barrier']
    try:
        algorithm = job['algorithm']
        if algorithm == 'SCS':
            from dcs_bench.scs_paper import run_chain
        else:
            from dcs_bench.fast_dcs import run_chain
        print(f'{algorithm}, chain {job["chain"]+1} starting', flush=True)
        traces, record = run_chain(job['mapping'], {**job['config'], 'algorithm': algorithm}, job['seed'],
                                   before_sampling=lambda: barrier.wait(timeout=900))
        # File compression by an earlier finisher must not compete with peers
        # whose production clocks are still running.
        barrier.wait(timeout=900)
        record.update(algorithm=algorithm, chain=job['chain'], seed=job['seed'],
                      wave=job['wave'], wave_size=job['wave_size'],
                      worker={'pid': os.getpid(), **_ALLOCATION}, time_scope=TIME_SCOPE)
        if not np.all(np.isfinite(traces)) or not np.all(np.isfinite(record['mean_all_iterations'])):
            raise FloatingPointError('Nonfinite chain estimates')
        output = Path(job['output']); output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(output)
        with output.open('xb') as stream:
            np.savez_compressed(stream, functionals=traces,
                                metadata=np.array(json.dumps(record, allow_nan=False)))
        print(f'{algorithm}, chain {job["chain"]+1} saved ({record["seconds"]:.3f} sampling seconds)', flush=True)
        return record
    except BaseException as error:
        try:
            barrier.abort()
        except Exception:
            pass
        if hasattr(error, 'add_note'):
            error.add_note(f'{job["algorithm"]}, chain {job["chain"]+1}, wave {job["wave"]}')
        raise


def run_chains(root, config, maps):
    count = min(config['workers'], config['runs'])
    context = multiprocessing.get_context('spawn')
    records = {name: [] for name in METHODS}
    with context.Manager() as manager:
        slots = manager.Queue()
        for cpu in cpu_info()['cpu_ids'][:count]:
            slots.put(cpu)
        with concurrent.futures.ProcessPoolExecutor(max_workers=count, mp_context=context,
                initializer=initialize_worker, initargs=(slots,)) as pool:
            for wave, start in enumerate(range(0, config['runs'], count)):
                chains = list(range(start, min(start+count, config['runs'])))
                order = METHODS[wave % 3:]+METHODS[:wave % 3]
                for algorithm in order:
                    barrier = manager.Barrier(len(chains))
                    jobs = [dict(algorithm=algorithm, chain=chain, seed=config['seed']+chain,
                        wave=wave, wave_size=len(chains), barrier=barrier, config=config,
                        mapping=maps['sphere' if algorithm == 'SCS' else 'ball'],
                        output=str(root/'chains'/algorithm/f'{chain:03d}.npz')) for chain in chains]
                    print(f'Wave {wave+1}, {algorithm}: {len(chains)} concurrent chains', flush=True)
                    futures = [pool.submit(chain_job, job) for job in jobs]
                    errors = []
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            records[algorithm].append(future.result())
                        except BaseException as error:
                            errors.append(error)
                    if errors:
                        from threading import BrokenBarrierError
                        raise next((e for e in errors if not isinstance(e, BrokenBarrierError)), errors[0])
    return {key: sorted(values, key=lambda x: x['chain']) for key, values in records.items()}


def summarize(config, maps, fitting, records):
    truth = radial_truth(Target(TARGET), 50.)[1]
    cases = {}
    for name in METHODS:
        rows = records[name]
        if len(rows) != config['runs'] or any(r['iterations'] != config['iterations'] for r in rows):
            raise ValueError('Incomplete chains cannot be summarized as a completed comparison')
        means = np.asarray([r['mean_all_iterations'] for r in rows])
        mse = np.mean((means-truth)**2, axis=0)
        bias_squared = (means.mean(axis=0)-truth)**2
        variance = np.mean((means-means.mean(axis=0))**2, axis=0)
        np.testing.assert_allclose(mse, bias_squared+variance, rtol=1e-11, atol=1e-15)
        seconds = np.array([r['seconds'] for r in rows])
        if np.any(seconds <= 0):
            raise ValueError('Sampling times must be positive')
        cases[name] = dict(chains=len(rows), chain_means=means.tolist(), mse=mse.tolist(),
            bias_squared=bias_squared.tolist(), variance_ddof0=variance.tolist(),
            mean_seconds=float(seconds.mean()), median_seconds=float(np.median(seconds)),
            iterations_per_second=float(config['iterations']/seconds.mean()),
            total_iterations=config['iterations']*len(rows),
            mean_acceptance=float(np.mean([r['acceptance'] for r in rows])),
            final_scales=[r['final_scale'] for r in rows], records=rows)
    return dict(config=config, truth=truth.tolist(), functionals=['tail_probability', 'mean_norm'],
                maps=maps, fitting=fitting, cases=cases, time_scope=TIME_SCOPE,
                target_note='Shared normal-CDF-skewed t target from the DCS experiment. The SCS paper uses a '
                            'Student-t CDF and nu=1 or2; this applies its author algorithm/settings to our nu=3 target.',
                mse_note='All production iterations, including repeats after rejections; exact radial truth. '
                         'Empirical bias squared plus between-chain variance with divisor number of chains equals MSE. '
                         'More chains estimate error more precisely but cannot repair insufficient chain mixing.')


def run(config, output):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Choose a fresh output folder: {output}')
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    manifest = dict(status='running', protocol=config['protocol'], config=config,
                    source_hashes=source_hashes(), environment=environment(), hardware=cpu_info(),
                    started_utc=datetime.now(timezone.utc).isoformat(), time_scope=TIME_SCOPE)
    write_json(output/'configuration.json', config)
    write_json(output/'manifest.json', manifest)
    try:
        maps, fitting = prepare(config)
        write_json(output/'maps.json', {'maps': maps, 'fitting': fitting})
        records = run_chains(output, config, maps)
        summary = summarize(config, maps, fitting, records)
        write_json(output/'summary.json', summary)
        from scripts.scs_100d_figures import render_report
        render_report(output, summary)
        manifest.update(status='complete', invocation_wall_seconds=time.perf_counter()-started,
                        completed_utc=datetime.now(timezone.utc).isoformat())
        write_json(output/'manifest.json', manifest)
        return output
    except BaseException as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}',
                        invocation_wall_seconds=time.perf_counter()-started)
        write_json(output/'manifest.json', manifest)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preset', choices=list(PRESETS), default='quick')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    for name in ('runs', 'iterations', 'warmup', 'thin', 'optimizer-steps', 'optimizer-samples',
                 'scs-optimizer-steps', 'scs-optimizer-samples', 'seed'):
        parser.add_argument('--'+name, type=int)
    args = parser.parse_args(argv)
    overrides = {k: v for k, v in vars(args).items() if k not in ('preset', 'workers', 'output_dir', 'dry_run')}
    config = configuration(args.preset, args.workers, overrides)
    output = args.output_dir or ROOT/'results'/'skew-t-100d-fixed'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    if args.dry_run:
        print(json.dumps({'output_dir': str(output), 'configuration': config}, indent=2))
        return 0
    print(f'Results: {run(config, output)}', flush=True)
    return 0


if __name__ == '__main__':
    if os.name == 'nt':
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, 'reconfigure'):
                stream.reconfigure(encoding='utf-8', errors='backslashreplace')
    raise SystemExit(main())
