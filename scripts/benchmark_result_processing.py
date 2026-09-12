"""Benchmark exact posterior-result processing on existing saved chains.

Example (from the repository):
  python scripts/benchmark_result_processing.py --input results/my-eight-schools

Use --repeat 6 only with a small result to add a larger repeated-input workload.

Each implementation runs in a fresh process with identical imports and a warm
diagnostic. Timings exclude imports; peak RSS includes them. The repeated-input
workload measures processing scalability only: duplicated saved chains are not
new independent MCMC chains. No input checkpoint or existing report is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'dcs-processing-benchmark-matplotlib'))


def peak_rss_bytes():
    """Peak resident/working-set memory, or None on unsupported systems."""
    try:
        if sys.platform == 'win32':
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
                    (name, ctypes.c_size_t) for name in (
                        'PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
                        'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                        'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]

            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            query = ctypes.WinDLL('psapi', use_last_error=True).GetProcessMemoryInfo
            query.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
            query.restype = wintypes.BOOL
            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            if query(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                return int(counters.PeakWorkingSetSize)
            return None
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == 'darwin' else value*1024)
    except (ImportError, OSError, AttributeError):
        return None


def old_diagnostics(chains, names):
    """Frozen full-metric ArviZ calls from the original rebuild implementation."""
    import arviz as az
    import numpy as np
    valid = [i for i in range(len(names)) if np.all(np.ptp(chains[:, :, i], axis=1) > 0)]
    payload = {names[i]: chains[:, :, i] for i in valid}
    idata = az.from_dict(posterior=payload) if payload else None
    result = {'unavailable_reason': {
        n: 'At least one retained chain is constant for this functional'
        for i, n in enumerate(names) if i not in valid}}
    for key, fn in [('ess_bulk', lambda: az.ess(idata, method='bulk')),
                    ('ess_tail', lambda: az.ess(idata, method='tail')),
                    ('mcse_mean', lambda: az.mcse(idata, method='mean')),
                    ('rhat', lambda: az.rhat(idata, method='rank'))]:
        if idata is None or (key == 'rhat' and len(chains) < 2):
            result[key] = dict.fromkeys(names)
        else:
            values = fn()
            result[key] = {n: float(values[n]) if i in valid and np.isfinite(values[n]) else None
                           for i, n in enumerate(names)}
    return result


def old_process(paths, target, config, reference, expected):
    """Original all-chains/all-functionals allocation and computation order."""
    import numpy as np
    from dcs_bench.engine import read_chain
    pairs = [read_chain(path, expected) for path in paths]
    chains, records = list(zip(*pairs))
    functions = [np.column_stack([x, x**2]) for x in chains]
    estimates = np.array([f.mean(axis=0) for f in functions])
    quantiles = np.array([np.quantile(x[:, :4], config['q_levels'], axis=0) for x in chains])
    length = min(map(len, functions))
    diagnostics = old_diagnostics(np.stack([f[:length] for f in functions]), reference['functional_names'])
    indiv = [old_diagnostics(f[None, ...], reference['functional_names']) for f in functions]
    return dict(estimates=estimates, quantiles=quantiles, length=length,
                diagnostics=diagnostics, indiv=indiv,
                processing=dict(physical_array_bytes=sum(x.nbytes for x in chains),
                    functional_array_bytes=sum(x.nbytes for x in functions),
                    cross_chain_stack_bytes=len(paths)*length*len(reference['functional_names'])*8))


def input_label(path):
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.name  # Do not publish the user's absolute home directory.


def worker(args):
    startup = time.perf_counter()
    import numpy as np
    import arviz as az
    # The real report path imports these dependencies; keep both arms identical.
    from dcs_bench import engine, plotting, processing
    from dcs_bench.targets import Target
    old_diagnostics(np.random.default_rng(884).normal(size=(2, 100, 1)), ['warmup'])
    warm_seconds, warmed_rss = time.perf_counter()-startup, peak_rss_bytes()
    config = json.loads((args.input/'configuration.json').read_text())
    manifest = json.loads((args.input/'manifest.json').read_text())
    scenario = args.input/f'scenario-{args.scenario}'
    reference = json.loads((scenario/'reference.json').read_text())
    target = Target(config['targets'][args.scenario])
    if target.kind not in ('eight_schools', 'garch'):
        raise ValueError('This benchmark requires an Eight Schools or GARCH saved result.')
    if args.algorithm not in config['algorithms']:
        raise ValueError('The selected method is absent from this saved experiment.')
    paths = [scenario/'chains'/f'{args.algorithm}-{i:03d}.npz'
             for i in range(config['runs'])]*args.repeat
    started = time.perf_counter()
    if args.worker == 'baseline':
        part = old_process(paths, target, config, reference, manifest['fingerprint'])
    else:
        part = processing.process_chains(paths, target, config['threshold'], config['q_levels'],
            reference['functional_names'], expected=manifest['fingerprint'], progress=None,
            diagnostic_workers=args.diagnostic_workers)
    estimates = part['estimates']
    statistics = dict(plotting.mse_summary(estimates, reference['functional_truth']),
        chain_functional_estimates=estimates.tolist(), chain_quantiles=part['quantiles'].tolist(),
        cross_chain_diagnostics=part['diagnostics'], independent_chain_diagnostics=part['indiv'],
        diagnostic_draws_per_chain=part['length'])
    for moment, name in enumerate(('first', 'second')):
        section = slice(moment*target.d, (moment+1)*target.d)
        statistics[f'{name}_moment_rmse_across_chains_and_coordinates'] = float(np.sqrt(np.mean(
            (estimates[:, section]-reference['functional_truth'][section])**2)))
    elapsed, peak_rss = time.perf_counter()-started, peak_rss_bytes()
    paths_to_hash = [Path(__file__), *(ROOT/'dcs_bench'/name for name in (
        'processing.py', 'references.py', 'engine.py', 'plotting.py'))]
    result = dict(branch=args.worker, input=input_label(args.input), algorithm=args.algorithm,
        scenario=args.scenario, repetitions=args.repeat, chain_inputs=len(paths),
        diagnostic_workers=1 if args.worker == 'baseline' else args.diagnostic_workers,
        unique_chain_files=len(set(paths)), processing_seconds=elapsed,
        imports_and_diagnostic_warmup_seconds=warm_seconds,
        process_peak_rss_after_warmup_bytes=warmed_rss, process_peak_rss_bytes=peak_rss,
        processing_peak_rss_increase_bytes=(None if peak_rss is None or warmed_rss is None
                                           else peak_rss-warmed_rss),
        processing_details=part['processing'], statistics=statistics,
        platform=platform.platform(), python=sys.version, numpy=np.__version__, arviz=az.__version__,
        source_sha256={p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in paths_to_hash},
        original_input_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in set(paths)})
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: result[key] for key in ('branch', 'chain_inputs', 'processing_seconds',
                                                  'process_peak_rss_bytes')}), flush=True)


def assert_close(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys(), 'Statistic names changed'
        for key in a:
            assert_close(a[key], b[key])
    elif isinstance(a, list):
        assert len(a) == len(b), 'Statistic lengths changed'
        for left, right in zip(a, b):
            assert_close(left, right)
    elif isinstance(a, (float, int)):
        assert isinstance(b, (float, int)) and math.isclose(a, b, rel_tol=2e-13, abs_tol=2e-13), (a, b)
    else:
        assert a == b, (a, b)


def parent(args):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(input=input_label(args.input), comparisons=[],
        note='Sequential fresh processes; identical imports and diagnostic warmup. RSS includes imports; '
             'processing times exclude imports. Each branch is timed once. Repeated saved inputs are '
             'a processing stress workload, not new independent MCMC chains.',
        baseline='Original all-chain report computation from commit d43b6a6cfabe220b32c4e3a49ad21a9a95b71773')
    for repeat in dict.fromkeys((1, args.repeat)):
        branches = {}
        for branch in ('baseline', 'optimized'):
            path = args.output.with_name(f'{args.output.stem}-{branch}-{repeat}.json')
            if path.exists():
                raise FileExistsError(f'Choose a fresh benchmark output name: {path}')
            print(f'Benchmark {branch}, saved chain set repeated {repeat} time(s)', flush=True)
            subprocess.run([sys.executable, str(Path(__file__)), '--worker', branch,
                '--input', str(args.input), '--output', str(path), '--repeat', str(repeat),
                '--algorithm', args.algorithm, '--scenario', str(args.scenario),
                '--diagnostic-workers', str(args.diagnostic_workers)], check=True)
            branches[branch] = json.loads(path.read_text())
        before, after = branches['baseline'], branches['optimized']
        assert_close(before['statistics'], after['statistics'])
        assert before['original_input_sha256'] == after['original_input_sha256'], 'Input chains changed'
        assert before['source_sha256'] == after['source_sha256'], 'Source changed during benchmark'
        old_rss, new_rss = before['process_peak_rss_bytes'], after['process_peak_rss_bytes']
        report['comparisons'].append(dict(chain_inputs=before['chain_inputs'], repetitions=repeat,
            baseline_diagnostic_workers=before['diagnostic_workers'],
            optimized_diagnostic_workers=after['diagnostic_workers'],
            all_statistics_equal_exactly=before['statistics'] == after['statistics'],
            all_statistics_agree_rtol_atol_2e_13=True, input_hashes_unchanged=True,
            baseline_seconds=before['processing_seconds'], optimized_seconds=after['processing_seconds'],
            speedup=before['processing_seconds']/after['processing_seconds'],
            baseline_peak_rss_bytes=old_rss, optimized_peak_rss_bytes=new_rss,
            peak_rss_reduction_fraction=None if old_rss is None or new_rss is None else 1-new_rss/old_rss,
            baseline_rss_increase_bytes=before['processing_peak_rss_increase_bytes'],
            optimized_rss_increase_bytes=after['processing_peak_rss_increase_bytes'],
            source_sha256=after['source_sha256']))
        report.update(platform=after['platform'], python=after['python'], numpy=after['numpy'], arviz=after['arviz'])
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'results/result-processing-benchmark.json')
    parser.add_argument('--algorithm', default='DCS-HnR')
    parser.add_argument('--scenario', type=int, default=0)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--diagnostic-workers', type=int, choices=range(1, 5), default=4,
                        help='Parallel individual-functional diagnostics in the optimized branch (default: 4).')
    parser.add_argument('--worker', choices=('baseline', 'optimized'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeat < 1 or args.scenario < 0:
        parser.error('--repeat must be positive and --scenario nonnegative')
    if not args.worker and args.output.exists():
        parser.error('Choose a fresh --output file; existing reports are preserved')
    worker(args) if args.worker else parent(args)
