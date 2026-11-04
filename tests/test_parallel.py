"""Exercise real worker processes, CPU allocation and reproducible chain seeds."""
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import textwrap
import time
import unittest
from unittest.mock import patch

import numpy as np
from dcs_bench.cpu import CPU_THREAD_POLICY, THREAD_VARIABLES, cpu_info, execution_plan, validate_workers, worker_environment, _ordered_slots
from dcs_bench.engine import chain_seed, environment, load_config, read_chain
from dcs_bench.parallel import launch_chains
from dcs_bench.samplers import run_metropolis
from dcs_bench.targets import Target
from dcs_bench.transforms import BallMap


class ParallelTests(unittest.TestCase):
    def test_worker_runtime_limit_overrides_inherited_environment(self):
        inherited = {'PJRT_NPROC': '12', 'NPROC': '8', 'OMP_NUM_THREADS': '6',
                     'XLA_FLAGS': '--xla_cpu_multi_thread_eigen=true --xla_force_host_platform_device_count=4 intra_op_parallelism_threads=9'}
        with patch.dict(os.environ, inherited):
            settings = worker_environment()
            self.assertEqual(os.environ['PJRT_NPROC'], '12')
        self.assertEqual(settings['NPROC'], '8')
        for key in THREAD_VARIABLES:
            self.assertEqual(settings[key], '1', key)
        self.assertIn('--xla_cpu_multi_thread_eigen=false', settings['XLA_FLAGS'])
        self.assertIn('--xla_force_host_platform_device_count=1', settings['XLA_FLAGS'])
        self.assertNotIn('intra_op_parallelism_threads=9', settings['XLA_FLAGS'])

    def test_fresh_jax_child_records_limit_and_compiles_without_parallel_tasks(self):
        # Compile only: no noisy speed or CPU-time thresholds in the test.
        child = textwrap.dedent('''
            import json
            import os
            from dcs_bench.cpu import worker_environment
            os.environ.update(worker_environment())
            import jax
            import jax.numpy as jnp
            from dcs_bench.engine import environment
            kernel = jax.jit(lambda x: jnp.sin(x) + x*x).lower(
                jnp.ones((300, 500), dtype=jnp.float64)).compile()
            record = environment()
            print(json.dumps({'threads': record['threads'],
                'policy': record['cpu_thread_policy'],
                'devices': len(jax.devices('cpu')),
                'parallel_tasks': 'outer_dimension_partitions' in kernel.as_text(),
                'inherited_nproc': os.environ['NPROC']}))
        ''')
        inherited = dict(os.environ, PJRT_NPROC='12', NPROC='8')
        result = subprocess.run([sys.executable, '-c', child],
            cwd=Path(__file__).resolve().parents[1], env=inherited,
            capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        record = json.loads(result.stdout)
        self.assertEqual(record['threads']['PJRT_NPROC'], '1')
        self.assertEqual(record['policy'], CPU_THREAD_POLICY)
        self.assertEqual(record['devices'], 1)
        self.assertEqual(record['inherited_nproc'], '8')
        self.assertFalse(record['parallel_tasks'])

    def test_ryzen_16_core_32_thread_allocation(self):
        cores = [[2*i, 2*i+1] for i in range(16)]
        slots = _ordered_slots(cores)
        self.assertEqual(slots[:16], list(range(0, 32, 2)))
        self.assertEqual(slots[16:], list(range(1, 32, 2)))
        info = {'capacity': 32, 'cpu_ids': slots, 'physical_cores': 16}
        with patch('dcs_bench.cpu.cpu_info', return_value=info):
            plan = execution_plan(32, 50)
        self.assertEqual(len(set(plan['cpu_ids'])), 32)
        self.assertEqual(plan['max_concurrent_chains'], 32)

    def test_worker_failure_stops_and_reaps_siblings(self):
        if cpu_info()['capacity'] < 2:
            self.skipTest('Two CPU slots are required')
        config = load_config('skew-t-100d-fixed', 'quick',
                             {'workers': 2, 'iterations': 100000000, 'warmup': 30, 'runs': 2})
        job = {'target': {'kind': 'skew_t', 'dimension': 3, 'nu': 3, 'alpha': {}},
               'config': config, 'algorithm': 'RWM-Gaussian', 'seconds': None,
               'fingerprint': 'failure-test', 'scenario': 0}
        original_popen = subprocess.Popen
        children = []
        def tracked(*args, **kwargs):
            process = original_popen(*args, **kwargs, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            children.append(process)
            return process
        with tempfile.TemporaryDirectory() as tmp:
            chains = [{'index': 0, 'seed': 1, 'path': str(Path(tmp)/'0.npz')},
                      {'index': 1, 'seed': 'invalid seed', 'path': str(Path(tmp)/'1.npz')}]
            plan = execution_plan(2, 2)
            with patch('dcs_bench.parallel.subprocess.Popen', side_effect=tracked):
                with self.assertRaisesRegex(RuntimeError, 'worker.*failed'):
                    launch_chains(job, chains, Path(tmp)/'jobs', plan)
            self.assertEqual(list(Path(tmp).glob('.draws-*')), [])
        self.assertEqual(len(children), 2)
        self.assertTrue(all(child.poll() is not None for child in children))

    def test_worker_limits_and_effective_count(self):
        info = {'capacity': 4, 'cpu_ids': [1, 3, 5, 7]}
        for value in (0, -1, 5, 1.5, True, '2'):
            with self.assertRaises(ValueError):
                validate_workers(value, info)
        with patch('dcs_bench.cpu.cpu_info', return_value=info):
            plan = execution_plan(4, 2)
        self.assertEqual(plan['max_concurrent_chains'], 2)
        self.assertEqual(plan['cpu_ids'], [1, 3])
        self.assertEqual(plan['cpu_thread_policy'], CPU_THREAD_POLICY)
        self.assertEqual(plan['jax_threads_per_compute_pool'], 1)

    def test_two_processes_preserve_seeds_draws_and_checkpoints(self):
        workers = min(2, cpu_info()['capacity'])
        config = load_config('skew-t-100d-fixed', 'quick',
                             {'workers': workers, 'iterations': 12000, 'warmup': 30, 'runs': 5, 'thin': 1})
        spec = {'kind': 'skew_t', 'dimension': 3, 'nu': 3, 'alpha': {}}
        target = Target(spec)
        plan = execution_plan(workers, 5)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chains = [{'index': i, 'seed': chain_seed(config, 0, 0, i), 'path': str(root/f'{i}.npz')} for i in range(5)]
            # Even when a fitted map exists, an ordinary RWM baseline must ignore it.
            job = {'target': spec, 'config': config, 'algorithm': 'RWM-Gaussian',
                   'maps': {'ball': BallMap(3, beta=2).parameters()},
                   'seconds': None, 'fingerprint': 'parallel-test', 'scenario': 0}
            launch_chains(job, chains, root/'jobs', plan)
            records = []
            for chain in chains:
                draws, record = read_chain(Path(chain['path']), 'parallel-test')
                expected, _ = run_metropolis(target, 'RWM-Gaussian', None, config, chain['seed'])
                np.testing.assert_array_equal(draws, expected)
                self.assertEqual(record['seed'], chain['seed'])
                self.assertIn(record['execution']['worker_slot'], range(workers))
                self.assertEqual(record['execution']['threads']['OMP_NUM_THREADS'], '1')
                self.assertEqual(record['execution']['threads']['PJRT_NPROC'], '1')
                self.assertEqual(record['execution']['cpu_thread_policy'], CPU_THREAD_POLICY)
                self.assertEqual(record['draw_collection']['collection_buffer_bytes'], 8*1024*1024)
                self.assertEqual(record['draw_collection']['collection_rows'], config['iterations']//config['thin'])
                if plan['hardware']['pinning'] in ('linux', 'windows'):
                    self.assertTrue(record['execution']['pinned'])
                    self.assertEqual(record['execution']['cpu_id'], plan['cpu_ids'][record['execution']['worker_slot']])
                records.append(record)
            self.assertEqual(list(root.glob('.draws-*')), [])
            # Very short chains can all finish on one worker before another
            # leaves the start-barrier poll, especially on Windows. Verify
            # every worker actually started without imposing queue fairness.
            compiled = [json.loads(p.read_text(encoding='utf-8'))
                        for p in (root/'jobs').glob('RWM-Gaussian-*-worker-*.compilation.json')]
            self.assertEqual(len(compiled), workers)
            self.assertEqual(len({entry['pid'] for entry in compiled}), workers)
            self.assertEqual({entry['worker_slot'] for entry in compiled}, set(range(workers)))
            self.assertEqual(len(list((root/'jobs').glob('*-queue/worker-*.ready'))), workers)
            worker_pids = {entry['worker_slot']: entry['pid'] for entry in compiled}
            for record in records:
                self.assertEqual(record['execution']['pid'], worker_pids[record['execution']['worker_slot']])
            # Every submitted chain was claimed exactly once, including a final
            # unbalanced batch and a resume containing sparse chain indices.
            claims = [json.loads(p.read_text(encoding='utf-8'))
                      for p in (root/'jobs').glob('*-queue/*.claim/owner.json')]
            self.assertEqual(sorted(owner['chain'] for owner in claims), list(range(5)))
            claimed_slots = {owner['chain']: owner['worker_slot'] for owner in claims}
            self.assertEqual(claimed_slots, {r['chain']: r['execution']['worker_slot'] for r in records})
            pending = [json.loads(p.read_text(encoding='utf-8'))
                       for p in (root/'jobs').glob('*-queue/*.pending.json')]
            self.assertEqual(sorted(pending, key=lambda chain: chain['index']), chains)
            # No pending chains must not spawn processes or modify checkpoints.
            before = {p.name: p.read_bytes() for p in root.glob('*.npz')}
            launch_chains(job, [], root/'jobs', plan)
            self.assertEqual(before, {p.name: p.read_bytes() for p in root.glob('*.npz')})

    def test_eight_processes_claim_immutable_jobs_once(self):
        # Queue-only children do not import a numerical runtime or pin CPUs, so
        # this exercises eight contenders even on small Windows CI machines.
        # It tests ownership under contention, without mocking a Windows race.
        child_code = textwrap.dedent('''
            import json
            from pathlib import Path
            import sys
            import time
            from unittest.mock import patch
            from dcs_bench.worker import pending_chains

            root, slot, rounds = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
            original_read = Path.read_text
            def delayed_read(path, *args, **kwargs):
                if path.name.endswith('.pending.json'):
                    # Let competing processes run after ownership is acquired
                    # and before the immutable job description is read.
                    time.sleep(.003)
                return original_read(path, *args, **kwargs)

            results = []
            with patch.object(Path, 'read_text', delayed_read):
                for round_index in range(rounds):
                    queue = root / str(round_index)
                    (queue / f'{slot}.ready').touch()
                    deadline = time.monotonic() + 45
                    while not (queue / 'start.ready').exists():
                        if time.monotonic() > deadline:
                            raise TimeoutError('Queue test start barrier timed out')
                        time.sleep(.005)
                    job = {'queue_directory': str(queue), 'worker_slot': slot}
                    results.append(list(pending_chains(job)))
            assert 'numpy' not in sys.modules and 'jax' not in sys.modules
            print(json.dumps(results))
        ''')
        workers, rounds = 8, 3
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected, initial_bytes = [], {}
            for round_index in range(rounds):
                queue = root/str(round_index)
                queue.mkdir()
                # Sparse indices model a resumed launch; every round has a
                # final unbalanced batch rather than a multiple of workers.
                chains = [{'index': 1000*round_index + 3*i, 'seed': 701 + i,
                           'path': f'chain-{round_index}-{i}.npz'} for i in range(73)]
                expected.append(chains)
                for chain in chains:
                    path = queue/f'{chain["index"]:06d}.pending.json'
                    path.write_text(json.dumps(chain)+'\n', encoding='utf-8')
                    initial_bytes[path] = path.read_bytes()
            children = []
            try:
                for slot in range(workers):
                    children.append(subprocess.Popen(
                        [sys.executable, '-c', child_code, str(root), str(slot), str(rounds)],
                        cwd=Path(__file__).resolve().parents[1],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
                for round_index in range(rounds):
                    queue = root/str(round_index)
                    deadline = time.monotonic()+45
                    while not all((queue/f'{slot}.ready').exists() for slot in range(workers)):
                        for process in children:
                            if process.poll() is not None:
                                stdout, stderr = process.communicate()
                                self.fail(f'Queue worker exited before the start barrier: {stdout}\n{stderr}')
                        if time.monotonic() > deadline:
                            self.fail('Queue workers did not reach the start barrier')
                        time.sleep(.01)
                    (queue/'start.ready').touch()
                observed = []
                for process in children:
                    stdout, stderr = process.communicate(timeout=45)
                    self.assertEqual(process.returncode, 0, stderr)
                    observed.append(json.loads(stdout))
            finally:
                for process in children:
                    if process.poll() is None:
                        process.kill()
                for process in children:
                    process.wait(timeout=10)
                    process.stdout.close()
                    process.stderr.close()
            for round_index, chains in enumerate(expected):
                with self.subTest(round=round_index):
                    received = [chain for worker in observed for chain in worker[round_index]]
                    self.assertEqual(sorted(received, key=lambda chain: chain['index']), chains)
                    ownership = {chain['index']: slot for slot, worker in enumerate(observed)
                                 for chain in worker[round_index]}
                    claims = list((root/str(round_index)).glob('*.claim'))
                    self.assertEqual(len(claims), len(chains))
                    for claim in claims:
                        owner = json.loads((claim/'owner.json').read_text(encoding='utf-8'))
                        self.assertEqual(int(claim.stem), owner['chain'])
                        self.assertEqual(ownership[owner['chain']], owner['worker_slot'])
            self.assertEqual({path: path.read_bytes() for path in initial_bytes}, initial_bytes)

    def test_zip_install_does_not_require_git(self):
        with patch('dcs_bench.engine.subprocess.run', side_effect=FileNotFoundError('git')):
            result = environment()
        self.assertIsNone(result['git_commit'])
        self.assertIsNone(result['git_status'])


if __name__ == '__main__':
    unittest.main()
