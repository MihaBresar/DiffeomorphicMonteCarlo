"""Internal comparison execution and saved-protocol configuration helpers.

The published 100D/500D defaults are defined in dcs_bench.reported_study.
Direct execution of this module delegates to that current experiment's CLI.
The shared runner uses all post-burn-in iterations, including rejection repeats,
and executes one method at a time in synchronized groups at equal concurrency.
"""
import argparse
import concurrent.futures
import copy
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
from dcs_bench.transforms import BallMap, SphereMap
from dcs_bench.tuning import fit_map
from dcs_bench.simulation_error import mse_variability, paired_mse_differences

METHODS = ('DCS-RWM', 'DCS-BW', 'SCS')
SEED_BATCH_STRIDE = 1_000_000
MAX_SEED_BATCH = 999
EXECUTION_REVISION = 'matched-wave-cpu-v2'
TARGET = {'kind': 'skew_t', 'dimension': 100, 'nu': 3., 'alpha': {'0': 100., '1': -100.}}
PRESETS = {
    'quick': dict(runs=2, time_budget=.05, warmup=100, thin=1, optimizer_steps=20, optimizer_samples=128),
    'reproduce': dict(runs=20, time_budget=2., warmup=1000, thin=10, optimizer_steps=400, optimizer_samples=500),
    'paper': dict(runs=50, time_budget=20., warmup=1500, thin=50, optimizer_steps=1000, optimizer_samples=2000),
}
TIME_SCOPE = ('Equal per-chain sampling time includes first-third burn-in and radial accumulation. '
              'Fitting, compilation and step tuning are outside the clock. All methods use the same '
              'JAX scan framework and their recorded separate step-calibration rules, then freeze their steps. Only post-burn-in '
              'states contribute estimates. All methods use the same 300-transition timing batches, '
              'with actual elapsed time and overshoot recorded. Methods run in equal-size synchronized waves.')

_ALLOCATION = {}


def configuration(preset='quick', workers=1, overrides=None):
    config = dict(PRESETS[preset], preset=preset, workers=workers, mode='time', burnin_fraction=1/3,
                  experiment='skew-t-100d-fixed', protocol='timed-matched-dcs-scs-v3',
                  implementation_revision='jax-dcs-exact-ball-v1',
                  dimension=100, nu=3., alpha=TARGET['alpha'], target=TARGET,
                  beta=2., latitude=1.1, threshold=50., seed=901001, seed_batch=0, algorithms=list(METHODS),
                  step_policy='shared_indicator_dual_averaging_0.234_then_frozen', initial_scale=.1,
                  vi_batch_policy='fixed', ball_map_policy='mobius', map_scale_initialization='unit',
                  vi_budget_policy='equal_updates_and_fixed_reference_clouds_v1',
                  fit_seed=42, sphere_map_policy='author-paper-vi',
                  scs_fitting_revision='author-vi-v1',
                  fitting_policy='DCS fixed-cloud reverse KL with gradient-norm cap 10; '
                                 'SCS original author VI, fixed base draws with seed 0, '
                                 'Adam 0.01, physical-coordinate clip 1000; no gradient clipping')
    supplied = {key: value for key, value in (overrides or {}).items() if value is not None}
    batch = supplied.pop('seed_batch', None)
    if batch is not None:
        if isinstance(batch, bool) or not isinstance(batch, int) or not 0 <= batch <= MAX_SEED_BATCH:
            raise ValueError('seed_batch must be an integer from 0 to 999')
        if 'seed' in supplied:
            raise ValueError('Choose either seed or seed_batch, not both')
        config.update(seed_batch=batch, seed=901001 + batch * SEED_BATCH_STRIDE)
    elif 'seed' in supplied:
        config['seed_batch'] = None
    allowed = {'runs', 'time_budget', 'warmup', 'thin', 'optimizer_steps', 'optimizer_samples',
               'fit_seed', 'seed', 'workers'}
    for key, value in supplied.items():
        if key not in allowed:
            raise ValueError(f'{key} is not an override for the single SCS comparison')
        config[key] = value
    for key in allowed-{'seed', 'fit_seed', 'time_budget'}:
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('seed', 'fit_seed'):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or not 0 <= config[key] <= 2**32-1:
            raise ValueError(f'{key} must be an integer from 0 to 2**32-1')
    if config['seed'] + config['runs'] - 1 > 2**32-1:
        raise ValueError('The chain seed range must fit within 32 bits')
    if config['seed_batch'] is not None and config['runs'] > SEED_BATCH_STRIDE:
        raise ValueError('Seed batches allow at most 1000000 chains to keep batches disjoint')
    if not np.isfinite(config['time_budget']) or config['time_budget'] <= 0:
        raise ValueError('time_budget must be finite and positive')
    if 100 % config['thin']:
        raise ValueError('thin must divide 100 for exact timed burn-in boundaries')
    config['scs_vi'] = {field: config[field] for field in ('optimizer_steps', 'optimizer_samples')}
    hardware = cpu_info()
    validate_workers(config['workers'], hardware)
    config['execution_revision'] = EXECUTION_REVISION
    config['wave_cpu_ids'] = _wave_cpu_ids(config, hardware)
    config['affinity_policy'] = ('fixed_cpu_prefix_per_wave' if all(cpu is not None for cpu in hardware['cpu_ids'])
                                 else 'os_managed_no_strict_pinning')
    return config


def _wave_cpu_ids(config, hardware):
    """Use the same ordered CPU prefix for every method, including partial waves."""
    width = min(config['workers'], config['runs'])
    slots = list(hardware['cpu_ids'])
    if len(slots) < width:
        raise ValueError('Not enough CPU slots for the requested wave size')
    if any(cpu is None for cpu in slots):
        if not all(cpu is None for cpu in slots):
            raise ValueError('CPU slots must be either all pinned or all OS-managed')
    elif len(slots) != len(set(slots)):
        raise ValueError('Pinned CPU slots must be unique')
    return [slots[:min(width, config['runs']-start)] for start in range(0, config['runs'], width)]


def source_hashes():
    paths = list((ROOT/'dcs_bench').glob('*.py')) + list((ROOT/'third_party/scs_paper').rglob('*.py'))
    paths += [Path(__file__), ROOT/'scripts/reproduce_boundary_100d.py', ROOT/'scripts/reproduce_comparison.py',
              ROOT/'scripts/reproduce_reported_study.py', ROOT/'scripts/scs_100d_figures.py', ROOT/'requirements-cpu.txt']
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def prepare(config):
    target = Target(config['target'])
    maps, fitting = {}, {}
    author_vi = config.get('vi', 'on') != 'off'
    if author_vi:
        from dcs_bench.scs_paper import author_vi_budget
        budget = author_vi_budget(config)
        if config.get('vi_budget_policy') == 'equal_updates_and_fixed_reference_clouds_v1':
            from dcs_bench.tuning import validate_matched_vi
            validate_matched_vi(config)
        author_config = {'target': config['target'], **budget}
    for family in ('ball', 'sphere'):
        if config.get('vi', 'on') == 'off':
            mapping = (BallMap(target.d, beta=config['beta']) if family == 'ball'
                       else SphereMap(target.d, latitude=config['latitude']))
            maps[family] = mapping.parameters()
            fitting[family] = dict(seconds=0., steps=0, training_draws=0, total_training_draws=0,
                loss=[], seed=None, vi_enabled=False, initial_scale=1., scale_policy='unit',
                map_policy='untuned', coordinate_clipping=False, cdf_clipping=False,
                fitting_policy='No VI for either family: R=1, centre=0, deformation=0')
            print(f'Using untuned {family} map (VI off)', flush=True)
            continue
        if family == 'sphere' and author_vi:
            from dcs_bench.scs_paper import fit as author_fit
            print('Fitting sphere map with the original SCS author VI routine', flush=True)
            maps[family], metadata = author_fit(author_config)
            fitting[family] = dict(metadata, vi_enabled=True,
                coordinate_clipping=True, cdf_clipping=False,
                fitting_policy=('Original SCS author reverse KL, fixed base draws with seed 0, '
                                'Adam 0.01, physical-coordinate clip 1000; no gradient clipping'))
            continue
        print(f'Fitting {family} map with the DCS VI protocol', flush=True)
        mapping, metadata = fit_map(target, config, family, config['fit_seed'])
        maps[family] = mapping.parameters()
        policy = (f'DCS {config.get("vi_batch_policy", "fresh")}-batch reverse KL, Adam 0.01, '
                  'gradient-norm cap 10; no coordinate/CDF clipping')
        fitting[family] = dict(metadata, learning_rate=.01, gradient_norm_cap=10.,
            coordinate_clipping=False, cdf_clipping=False,
            fitting_policy=policy)
    return maps, fitting


def method_mapping(maps, algorithm):
    """Select a distinct prepared map for each SCS control."""
    key = 'sphere_calibrated' if algorithm == 'SCS-calibrated' else 'sphere' if algorithm == 'SCS' else 'ball'
    return maps[key]


def method_execution_config(config, algorithm):
    """Dispatch both controls through the unchanged author SCS kernel.

    Public method identifiers remain distinct in saved outputs. The alias is
    translated only at the execution boundary, where the existing fixed-scale
    SCS policy supplies the frozen calibrated step.
    """
    result = dict(config, algorithm=algorithm)
    if 'SCS-calibrated' in config.get('algorithms', []):
        result['step_tuning_policies'] = {k: v for k, v in config['step_tuning_policies'].items()
                                           if k != 'SCS-calibrated'}
    if algorithm == 'SCS-calibrated':
        result.update(algorithm='SCS', initial_scale=config['scs_calibrated_move'],
                      scs_fixed_move=config['scs_calibrated_move'])
        result['step_tuning_policies']['SCS'] = 'fixed_scs_move_v1'
    return result


def effective_method_maps(config, maps):
    """Record each production coordinate map separately from the shared fits."""
    if not maps:
        return {}
    from dcs_bench.timed_radial import effective_mapping
    return {name: effective_mapping(method_mapping(maps, name), 'SCS' if name == 'SCS-calibrated' else name)
            for name in config['algorithms']}


def initialize_worker(slots):
    global _ALLOCATION
    _ALLOCATION = pin_current_process(slots.get())


def chain_job(job):
    barrier = job['barrier']
    try:
        algorithm = job['algorithm']
        from dcs_bench.timed_radial import run_timed
        print(f'{algorithm}, chain {job["chain"]+1} starting', flush=True)
        diagnostics = ({'diagnostic_lock': job.get('diagnostic_lock')}
                       if job['config'].get('ess_enabled', False) else {})
        execution_config = method_execution_config(job['config'], algorithm)
        traces, record = run_timed(job['mapping'], execution_config, job['seed'],
                                   before_sampling=lambda: barrier.wait(timeout=900),
                                   after_sampling=lambda: barrier.wait(timeout=900), **diagnostics)
        # File compression by an earlier finisher must not compete with peers
        # whose production clocks are still running.
        barrier.wait(timeout=900)
        record.update(algorithm=algorithm, chain=job['chain'], seed=job['seed'],
                      wave=job['wave'], wave_size=job['wave_size'],
                      wave_cpu_ids=list(job['wave_cpu_ids']), execution_revision=EXECUTION_REVISION,
                      affinity_policy=job['config']['affinity_policy'],
                      worker={'pid': os.getpid(), **_ALLOCATION}, time_scope=job['config'].get('time_scope', TIME_SCOPE))
        if algorithm == 'SCS-calibrated':
            record['kernel_algorithm'] = 'SCS'
        if not np.all(np.isfinite(traces)) or not np.all(np.isfinite(record['mean_all_iterations'])):
            raise FloatingPointError('Nonfinite chain estimates')
        output = Path(job['output']); output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(output)
        with output.open('xb') as stream:
            np.savez_compressed(stream, functionals=traces,
                                metadata=np.array(json.dumps(record, allow_nan=False)))
        print(f'{algorithm}, chain {job["chain"]+1} saved ({record["seconds"]:.2f} sampling seconds)', flush=True)
        return record
    except BaseException as error:
        try:
            barrier.abort()
        except Exception:
            pass
        if hasattr(error, 'add_note'):
            error.add_note(f'{job["algorithm"]}, chain {job["chain"]+1}, wave {job["wave"]}')
        raise


def _validate_wave_allocation(rows, expected_cpus):
    """Fail closed if a completed method did not use its entire assigned pool."""
    if len(rows) != len(expected_cpus):
        raise RuntimeError('Completed method has the wrong number of wave workers')
    workers = [row.get('worker', {}) for row in rows]
    pids = [worker.get('pid') for worker in workers]
    if None in pids or len(set(pids)) != len(expected_cpus):
        raise RuntimeError('Each chain in a wave must use a distinct worker process')
    actual = [worker.get('cpu_id') for worker in workers]
    if all(cpu is None for cpu in expected_cpus):
        if any(cpu is not None for cpu in actual) or any(worker.get('pinned') is not False for worker in workers):
            raise RuntimeError('OS-managed wave unexpectedly claims strict CPU pinning')
    elif not all(worker.get('pinned') is True for worker in workers) or \
            set(actual) != set(expected_cpus) or len(set(actual)) != len(expected_cpus):
        raise RuntimeError('Completed method did not use the expected pinned CPU set')


def run_chains(root, config, maps):
    methods = tuple(config['algorithms'])
    count = min(config['workers'], config['runs'])
    wave_cpus = _wave_cpu_ids(config, cpu_info())
    if config.get('execution_revision') != EXECUTION_REVISION or config.get('wave_cpu_ids') != wave_cpus:
        raise ValueError('CPU allocation changed after configuration; create a fresh configuration')
    context = multiprocessing.get_context('spawn')
    records = {name: [] for name in methods}
    with context.Manager() as manager:
        diagnostic_lock = manager.BoundedSemaphore(1) if config.get('ess_enabled', False) else None
        pool, pool_width = None, 0
        try:
            for wave, start in enumerate(range(0, config['runs'], count)):
                chains = list(range(start, min(start+count, config['runs'])))
                if len(chains) != pool_width:
                    # Stop the larger pool before allocating the partial-wave
                    # pool. Every method then occupies every worker, whose CPU
                    # affinity is established before its first JAX import.
                    if pool is not None:
                        pool.shutdown(wait=True, cancel_futures=True)
                        pool = None
                    slots = manager.Queue()
                    for cpu in wave_cpus[wave]:
                        slots.put(cpu)
                    pool = concurrent.futures.ProcessPoolExecutor(max_workers=len(chains), mp_context=context,
                        initializer=initialize_worker, initargs=(slots,))
                    pool_width = len(chains)
                order = methods[wave % len(methods):]+methods[:wave % len(methods)]
                for algorithm in order:
                    barrier = manager.Barrier(len(chains))
                    jobs = [dict(algorithm=algorithm, chain=chain, seed=config['seed']+chain,
                        wave=wave, wave_size=len(chains), wave_cpu_ids=wave_cpus[wave], barrier=barrier, config=config,
                        mapping=method_mapping(maps, algorithm),
                        output=str(root/'chains'/algorithm/f'{chain:03d}.npz')) for chain in chains]
                    if diagnostic_lock is not None:
                        for job in jobs:
                            job['diagnostic_lock'] = diagnostic_lock
                    print(f'Wave {wave+1}, {algorithm}: {len(chains)} concurrent chains', flush=True)
                    futures, completed, errors = [], [], []
                    try:
                        for job in jobs:
                            futures.append(pool.submit(chain_job, job))
                        for future in concurrent.futures.as_completed(futures):
                            try:
                                completed.append(future.result())
                            except BaseException as error:
                                errors.append(error)
                                # A dead worker or failed initializer cannot
                                # abort its own barrier; release its peers here.
                                try:
                                    barrier.abort()
                                except Exception:
                                    pass
                                for pending in futures:
                                    pending.cancel()
                        if errors:
                            from threading import BrokenBarrierError
                            secondary = (BrokenBarrierError, concurrent.futures.CancelledError)
                            raise next((e for e in errors if not isinstance(e, secondary)), errors[0])
                        _validate_wave_allocation(completed, wave_cpus[wave])
                        records[algorithm].extend(completed)
                    except BaseException:
                        try:
                            barrier.abort()
                        except Exception:
                            pass
                        for future in futures:
                            future.cancel()
                        raise
        finally:
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)
    return {key: sorted(values, key=lambda x: x['chain']) for key, values in records.items()}


def summarize(config, maps, fitting, records, reference=None):
    target = Target(config.get('target', TARGET))
    if config.get('coordinate_metrics', False):
        if reference is None:
            from dcs_bench.references import comparison_reference
            reference = comparison_reference(config)
        functionals = list(reference['functionals'])
        truth = np.asarray(reference['truth'], dtype=float)
    else:
        functionals, truth, _ = radial_truth(target, config.get('threshold', 50.))
        reference = dict(functionals=functionals, truth=truth.tolist(), reference_mcse=[0., 0.],
                         exact=[True, True], source='Analytic Student-t radial law')
    expected = (['tail_probability', 'mean_norm', 'mean_x1', 'x1_tail_probability',
                 'mean_x2', 'x2_left_tail_probability', 'mean_x3', 'x3_tail_probability']
                if config.get('coordinate_metrics', False) else ['tail_probability', 'mean_norm'])
    reference_mcse = np.asarray(reference['reference_mcse'], dtype=float)
    if functionals != expected or truth.shape != (len(expected),) or reference_mcse.shape != truth.shape \
            or not np.all(np.isfinite(truth)) or not np.all(np.isfinite(reference_mcse)) or np.any(reference_mcse < 0):
        raise ValueError('Invalid comparison reference or functional order')
    cases = {}
    for name in config['algorithms']:
        rows = records[name]
        if len(rows) != config['runs'] or any(r['iterations'] <= 0 for r in rows):
            raise ValueError('Incomplete chains cannot be summarized as a completed comparison')
        for row in rows:
            count = row['iterations']
            discarded = row.get('burnin_iterations', row.get('discarded_iterations'))
            if count % 300 or discarded != count//3 or row['estimation_iterations'] != count-discarded:
                raise ValueError('Inconsistent first-third burn-in counts')
            if row['seconds'] < config['time_budget']:
                raise ValueError('A chain stopped before its requested sampling budget')
        means = np.asarray([r['mean_all_iterations'] for r in rows])
        if means.shape != (config['runs'], len(functionals)) or not np.all(np.isfinite(means)):
            raise ValueError('Nonfinite or malformed chain estimates')
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
            iterations_per_second=float(sum(r['iterations'] for r in rows)/seconds.sum()),
            total_iterations=sum(r['iterations'] for r in rows),
            mean_iterations=float(np.mean([r['iterations'] for r in rows])),
            total_estimation_iterations=sum(r['estimation_iterations'] for r in rows),
            mean_estimation_iterations=float(np.mean([r['estimation_iterations'] for r in rows])),
            mean_acceptance=float(np.mean([r['acceptance'] for r in rows])),
            final_scales=[r['final_scale'] for r in rows], records=rows)
        if config.get('ess_enabled', False):
            from dcs_bench.functional_ess import aggregate_functional_ess
            cases[name]['functional_ess'] = aggregate_functional_ess(rows, functionals)
            cases[name]['functional_ess_seconds'] = float(sum(r.get('functional_ess_seconds', 0.) for r in rows))
        cases[name].update(mse_variability(means, truth))
        cases[name]['reference_error_material'] = (reference_mcse > .1*np.sqrt(mse)).tolist()
    return dict(config=config, truth=truth.tolist(), functionals=functionals, reference=reference,
                maps=maps, method_maps=effective_method_maps(config, maps), fitting=fitting,
                cases=cases, time_scope=config.get('time_scope', TIME_SCOPE),
                paired_mse_differences=paired_mse_differences(
                    {name: case['chain_means'] for name, case in cases.items()}, records, truth,
                    functionals=functionals),
                target_note=config.get('target_note', 'Shared normal-CDF-skewed t target from the DCS experiment. '
                            'The SCS paper uses a Student-t CDF and nu=1 or2; this applies the SCS kernel '
                            'with recorded benchmark tuning to our nu=3 target.'),
                mse_note='All post-burn-in iterations, including repeats after rejections. '
                         'Reference values and their uncertainty are recorded separately; MSE and its empirical '
                         'variability condition on the common reference. A reference-error flag marks reference MCSE '
                         'above 10% of that method\'s observed RMSE. '
                         'Empirical bias squared plus between-chain variance with divisor number of chains equals MSE. '
                         'More chains estimate error more precisely but cannot repair insufficient chain mixing.')


def run(config, output, *, prepared_inputs=None, prepared_input_load_seconds=None):
    input_validation_started = time.perf_counter()
    if prepared_input_load_seconds is not None:
        if (prepared_inputs is None or isinstance(prepared_input_load_seconds, bool)
                or not np.isfinite(prepared_input_load_seconds) or prepared_input_load_seconds < 0):
            raise ValueError('Prepared-input loading time must be finite and nonnegative')
    from dcs_bench.reported_study import SUPPORTED_PROTOCOLS, load_prepared
    if config.get('protocol') in SUPPORTED_PROTOCOLS:
        if prepared_inputs != load_prepared(config):
            raise ValueError('Reported-study runs require the exact bundled prepared inputs')
    if config.get('experiment') in ('sampler-comparison', 'sampler-comparison-numba'):
        if (config.get('vi') != 'on' or config.get('vi_enabled') is not True
                or config.get('ball_map_policy') != 'mobius'
                or config.get('sphere_map_policy') != 'author-paper-vi'):
            raise ValueError('New sampler comparisons require VI for both map families.')
        if config.get('protocol') in ('timed-sampler-comparison-v10', 'timed-numba-comparison-v10', 'timed-sampler-comparison-v11', 'timed-numba-comparison-v11'):
            from dcs_bench.tuning import validate_matched_vi
            validate_matched_vi(config)
    if prepared_inputs is not None:
        if set(prepared_inputs) != {'maps', 'fitting', 'provenance'}:
            raise ValueError('Prepared inputs require maps, fitting, and provenance')
        prepared_inputs = copy.deepcopy(prepared_inputs)
        prepared_input_load_seconds = ((prepared_input_load_seconds or 0.)
            + time.perf_counter() - input_validation_started)
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Choose a fresh output folder: {output}')
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    manifest = dict(status='running', protocol=config['protocol'], config=config,
                    source_hashes=source_hashes(), environment=environment(), hardware=cpu_info(),
                    started_utc=datetime.now(timezone.utc).isoformat(), time_scope=config.get('time_scope', TIME_SCOPE))
    if prepared_inputs is not None:
        manifest['prepared_input_provenance'] = prepared_inputs['provenance']
        manifest['prepared_input_load_seconds'] = prepared_input_load_seconds
    write_json(output/'configuration.json', config)
    write_json(output/'manifest.json', manifest)
    try:
        reference = None
        if config.get('coordinate_metrics', False):
            from dcs_bench.references import comparison_reference
            print('Preparing reference values for radial and X1/X2/X3 measurements', flush=True)
            reference_started = time.perf_counter()
            reference = comparison_reference(config)
            manifest['reference_seconds'] = time.perf_counter()-reference_started
            write_json(output/'reference.json', reference)
        if prepared_inputs is None:
            maps, fitting = prepare(config)
        else:
            maps, fitting = prepared_inputs['maps'], prepared_inputs['fitting']
            write_json(output/'prepared_input_provenance.json', prepared_inputs['provenance'])
        method_maps = effective_method_maps(config, maps)
        write_json(output/'maps.json', {'maps': maps, 'method_maps': method_maps, 'fitting': fitting})
        write_json(output/'method_maps.json', method_maps)
        manifest['method_maps'] = method_maps
        write_json(output/'manifest.json', manifest)
        records = run_chains(output, config, maps)
        summary = summarize(config, maps, fitting, records, reference=reference)
        if prepared_inputs is not None:
            summary['prepared_input_provenance'] = prepared_inputs['provenance']
            summary['prepared_input_load_seconds'] = prepared_input_load_seconds
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


def main(argv=None, *, configuration_factory=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preset', choices=list(PRESETS), default='quick')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--time-budget', type=float)
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument('--seed', type=int)
    seeds.add_argument('--seed-batch', type=int,
                       help='0 repeats original seeds; 1..999 select disjoint sampling seed batches. Map-fitting seed stays fixed.')
    for name in ('runs', 'warmup', 'thin', 'optimizer-steps', 'optimizer-samples',
                 'fit-seed'):
        parser.add_argument('--'+name, type=int)
    args = parser.parse_args(argv)
    overrides = {k: v for k, v in vars(args).items() if k not in ('preset', 'workers', 'output_dir', 'dry_run')}
    factory = configuration_factory or configuration
    config = factory(args.preset, args.workers, overrides)
    output = args.output_dir or ROOT/'results'/config['experiment']/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
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
    from scripts.reproduce_reported_study import main as current_main
    raise SystemExit(current_main())
