"""Bounded subprocess workers; no fork of an initialized JAX runtime."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

from .cpu import worker_environment

ROOT = Path(__file__).resolve().parents[1]


def launch_chains(job, chains, directory, plan):
    """Wait for all pending chains of one method, preserving finished checkpoints.

    Workers write large draw arrays directly to their distinct atomic checkpoint
    files. Only small JSON job descriptions pass between processes.
    """
    if not chains:
        return
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    count = min(plan['max_concurrent_chains'], len(chains))
    batch = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:8]
    queue = directory/f'{job["algorithm"]}-{batch}-queue'
    queue.mkdir()
    for chain in chains:
        (queue/f'{chain["index"]:06d}.pending.json').write_text(json.dumps(chain)+'\n', encoding='utf-8')
    processes = []
    try:
        for slot in range(count):
            request = dict(job, queue_directory=str(queue.resolve()), worker_slot=slot,
                           cpu_id=plan['cpu_ids'][slot], execution_plan=plan)
            path = directory/f'{job["algorithm"]}-{batch}-worker-{slot}.json'
            path.write_text(json.dumps(request, allow_nan=False)+'\n', encoding='utf-8')
            process = subprocess.Popen([sys.executable, '-u', '-m', 'dcs_bench.worker', str(path.resolve())],
                                       cwd=ROOT, env=worker_environment())
            processes.append(process)
        while any(process.poll() is None for process in processes):
            for process in processes:
                if process.poll() not in (None, 0):
                    raise RuntimeError(f'{job["algorithm"]} worker {process.pid} failed (exit {process.returncode}); completed chains are retained')
            if not (queue/'start.ready').exists() and len(list(queue.glob('worker-*.ready'))) == count:
                # All imports/compilation finish before any timed chain starts.
                (queue/'start.ready').touch()
            time.sleep(.1)
        if any(process.returncode != 0 for process in processes):
            raise RuntimeError(f'{job["algorithm"]} worker failed; see its traceback above. Completed chains are retained.')
    finally:
        # Also handles Stop/KeyboardInterrupt and a sibling worker failure.
        for process in processes:
            if process.poll() is None:
                process.terminate()
        deadline = time.monotonic()+3
        for process in processes:
            try:
                process.wait(timeout=max(.01, deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def compilation_summary(directory):
    entries = [json.loads(path.read_text(encoding='utf-8'))
               for path in sorted(Path(directory).glob('NUTS-*.compilation.json'))]
    return {'seconds': sum(entry['seconds'] for entry in entries), 'workers': entries,
            'accounting': 'Sum of worker compilation durations, including resumed launches; not elapsed wall time'}
