"""CPU discovery and worker policy, using only the standard library."""
import ctypes
import os
from pathlib import Path
import shlex
import subprocess
import sys

THREAD_VARIABLES = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS',
                    'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS', 'PJRT_NPROC')
CPU_THREAD_POLICY = 'pjrt-single-compute-thread-v1'


def _windows_api():
    api = ctypes.WinDLL('kernel32', use_last_error=True)
    api.GetCurrentProcess.restype = ctypes.c_void_p
    api.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    api.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    return api


def _ordered_slots(cores):
    """Use one thread on every core before assigning SMT siblings."""
    return [core[layer] for layer in range(max(map(len, cores)))
            for core in cores if layer < len(core)]


def _windows_cores():
    api = _windows_api()
    allowed, system = ctypes.c_size_t(), ctypes.c_size_t()
    if not api.GetProcessAffinityMask(api.GetCurrentProcess(), ctypes.byref(allowed), ctypes.byref(system)) or not allowed.value:
        raise OSError('Cannot obtain the Windows process affinity mask')
    # The legacy API describes the current processor group (at most 64 CPUs).
    # Keep workers in that group rather than claim unsupported cross-group pinning.
    class Entry(ctypes.Structure):
        _fields_ = [('mask', ctypes.c_size_t), ('relationship', ctypes.c_int),
                    ('reserved_union', ctypes.c_ulonglong * 2)]
    length = ctypes.c_ulong()
    api.GetLogicalProcessorInformation(None, ctypes.byref(length))
    if not length.value:
        raise OSError('Cannot discover Windows physical cores')
    buffer = ctypes.create_string_buffer(length.value)
    if not api.GetLogicalProcessorInformation(buffer, ctypes.byref(length)):
        raise OSError('Cannot discover Windows physical cores')
    cores = []
    for offset in range(0, length.value, ctypes.sizeof(Entry)):
        entry = Entry.from_buffer(buffer, offset)
        mask = entry.mask & allowed.value
        if entry.relationship == 0 and mask:  # RelationProcessorCore
            cores.append([cpu for cpu in range(ctypes.sizeof(ctypes.c_size_t)*8) if mask & (1 << cpu)])
    if not cores:
        raise OSError('No usable Windows physical cores found')
    return sorted(cores)


def cpu_info():
    logical = os.cpu_count() or 1
    if hasattr(os, 'sched_getaffinity'):
        allowed = sorted(os.sched_getaffinity(0))
        physical = {}
        try:
            for cpu in allowed:
                topology = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
                key = ((topology/'physical_package_id').read_text().strip(),
                       (topology/'core_id').read_text().strip())
                physical.setdefault(key, []).append(cpu)
            cores = sorted(physical.values())
            slots, basis = _ordered_slots(cores), 'all allowed CPU threads, physical cores first, then SMT siblings'
            physical_count = len(cores)
        except OSError:
            slots, basis = allowed, 'allowed logical CPUs; physical topology unavailable'
            physical_count = None
        return {'logical_cpus': logical, 'capacity': len(slots), 'cpu_ids': slots,
                'physical_cores': physical_count, 'pinning': 'linux', 'basis': basis}
    if os.name == 'nt':
        try:
            cores = _windows_cores()
            slots = _ordered_slots(cores)
            return {'logical_cpus': logical, 'capacity': len(slots), 'cpu_ids': slots,
                    'physical_cores': len(cores), 'pinning': 'windows',
                    'basis': 'all allowed threads in the current Windows processor group, physical cores first'}
        except OSError as error:
            # Fail closed instead of promising independent CPU allocations.
            return {'logical_cpus': logical, 'capacity': 1, 'cpu_ids': [None],
                    'physical_cores': None, 'pinning': 'unavailable', 'basis': str(error)}
    physical = logical
    basis = 'logical CPU count; physical topology unavailable; OS-managed scheduling'
    if sys.platform == 'darwin':
        try:
            physical = int(subprocess.check_output(['/usr/sbin/sysctl', '-n', 'hw.physicalcpu'], text=True, stderr=subprocess.DEVNULL).strip())
            basis = 'physical core count; the OS schedules workers without CPU pinning'
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return {'logical_cpus': logical, 'physical_cores': physical, 'capacity': logical, 'cpu_ids': [None]*logical,
            'pinning': 'os-managed', 'basis': basis}


def validate_workers(workers, info=None):
    info = info or cpu_info()
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= info['capacity']:
        raise ValueError(f'Workers must be an integer between 1 and {info["capacity"]} on this computer')
    return workers


def execution_plan(workers, chains):
    info = cpu_info()
    validate_workers(workers, info)
    count = min(workers, chains)
    return {'requested_workers': workers, 'max_concurrent_chains': count,
            'cpu_ids': info['cpu_ids'][:count], 'hardware': info,
            'threads_per_worker': 1, 'start_method': 'fresh Python subprocess',
            'cpu_thread_policy': CPU_THREAD_POLICY,
            'jax_threads_per_compute_pool': 1,
            'thread_limit_scope': 'Numerical compute pools; runtime helper threads can remain',
            'schedule': 'one method at a time; workers claim the next pending chain from a shared queue'}


def worker_environment():
    env = dict(os.environ)
    # In pinned JAX/XLA 0.4.35, PJRT_NPROC controls both CPU runtime pools
    # and the compiler's intra-op parallelism budget. Eigen-only flags do
    # not constrain all generated kernels. This overrides inherited PJRT
    # settings and takes precedence over NPROC without changing that variable.
    # Apply before backend creation; helper threads may still exist.
    env.update({key: '1' for key in THREAD_VARIABLES})
    env.update(JAX_PLATFORMS='cpu', JAX_ENABLE_X64='true', PYTHONUNBUFFERED='1',
               PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
    flags = shlex.split(env.get('XLA_FLAGS', ''))
    controlled = ('--xla_cpu_multi_thread_eigen=', '--xla_force_host_platform_device_count=', 'intra_op_parallelism_threads=')
    flags = [flag for flag in flags if not flag.startswith(controlled)]
    flags += ['--xla_cpu_multi_thread_eigen=false', '--xla_force_host_platform_device_count=1', 'intra_op_parallelism_threads=1']
    env['XLA_FLAGS'] = ' '.join(flags)
    return env


def pin_current_process(cpu):
    if cpu is None:
        return {'cpu_id': None, 'pinned': False}
    if hasattr(os, 'sched_setaffinity'):
        os.sched_setaffinity(0, {cpu})
        if os.sched_getaffinity(0) != {cpu}:
            raise RuntimeError('CPU affinity verification failed')
    elif os.name == 'nt':
        api = _windows_api()
        if not api.SetProcessAffinityMask(api.GetCurrentProcess(), 1 << cpu):
            raise OSError('Setting Windows CPU affinity failed')
        actual, system = ctypes.c_size_t(), ctypes.c_size_t()
        if not api.GetProcessAffinityMask(api.GetCurrentProcess(), ctypes.byref(actual), ctypes.byref(system)) or actual.value != 1 << cpu:
            raise OSError('Windows CPU affinity verification failed')
    else:
        raise RuntimeError('CPU pinning is not supported on this OS')
    return {'cpu_id': cpu, 'pinned': True}
