"""Check matched CPU allocation and cleanup without compiling or sampling."""
from concurrent.futures import Future
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from queue import Queue
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from scripts import reproduce_scs_100d as experiment


class FakeBarrier:
    def __init__(self, parties):
        self.parties = parties
        self.aborted = False

    def abort(self):
        self.aborted = True


class FakeManager:
    def __init__(self):
        self.barriers = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def Queue(self):
        return Queue()

    def Barrier(self, parties):
        barrier = FakeBarrier(parties)
        self.barriers.append(barrier)
        return barrier


class PoolHarness:
    """Model worker assignments and reject overlapping old/new pools."""
    def __init__(self, *, mutate_record=None, submit_failure=False, worker_failure=False):
        self.pools = []
        self.jobs = []
        self.events = []
        self.mutate_record = mutate_record
        self.submit_failure = submit_failure
        self.worker_failure = worker_failure

    def make_pool(self, max_workers, mp_context, initializer, initargs):
        if any(not pool.closed for pool in self.pools):
            raise AssertionError('Old pool must stop before a new pool is constructed')
        if initializer is not experiment.initialize_worker:
            raise AssertionError('Worker CPU affinity must be initialized before jobs')
        slots, = initargs
        cpus = [slots.get_nowait() for _ in range(max_workers)]
        if not slots.empty():
            raise AssertionError('Initializer queue must contain exactly one slot per worker')
        harness = self

        class Pool:
            closed = False
            submitted = 0
            width = max_workers
            cpu_ids = cpus

            def submit(self, function, job):
                if self.closed:
                    raise AssertionError('Cannot submit to a stopped pool')
                if function is not experiment.chain_job:
                    raise AssertionError('Unexpected job function')
                worker = self.submitted % self.width
                self.submitted += 1
                harness.jobs.append((self, job))
                if harness.submit_failure and self.submitted == 2:
                    raise RuntimeError('simulated submission failure')
                future = Future()
                if harness.worker_failure and self.submitted == 1:
                    future.set_exception(ValueError('simulated worker failure'))
                    return future
                # Reverse worker order in alternating methods: the chain-to-CPU
                # assignment can vary while the method's CPU set must not.
                if job['algorithm'] == 'SCS':
                    worker = self.width-1-worker
                cpu = cpus[worker]
                row = dict(algorithm=job['algorithm'], chain=job['chain'], seed=job['seed'],
                           wave=job['wave'], wave_size=job['wave_size'], wave_cpu_ids=job['wave_cpu_ids'],
                           execution_revision=job['config']['execution_revision'],
                           affinity_policy=job['config']['affinity_policy'],
                           worker={'pid': 1000+worker, 'cpu_id': cpu, 'pinned': cpu is not None})
                if harness.mutate_record:
                    harness.mutate_record(row)
                future.set_result(row)
                return future

            def shutdown(self, wait=True, cancel_futures=False):
                if not wait or not cancel_futures:
                    raise AssertionError('Shutdown must wait and cancel pending jobs')
                self.closed = True
                harness.events.append(('shutdown', self.width))

        pool = Pool()
        self.pools.append(pool)
        self.events.append(('create', max_workers))
        return pool


class WaveAllocationTests(unittest.TestCase):
    def run_fake(self, runs, workers, *, cpus=None, harness=None):
        # Physical threads first, followed by their SMT siblings.
        cpus = list(range(0, 64, 2))+list(range(1, 64, 2)) if cpus is None else cpus
        hardware = {'capacity': len(cpus), 'cpu_ids': cpus}
        manager = FakeManager()
        context = SimpleNamespace(Manager=lambda: manager)
        harness = harness or PoolHarness()
        with patch.object(experiment, 'cpu_info', return_value=hardware), \
                patch.object(experiment.multiprocessing, 'get_context', return_value=context), \
                patch.object(experiment.concurrent.futures, 'ProcessPoolExecutor', side_effect=harness.make_pool), \
                redirect_stdout(StringIO()):
            config = experiment.configuration('quick', workers, {'runs': runs})
            records = experiment.run_chains(Path('unused'), config, {'sphere': {}, 'ball': {}})
        return config, records, harness, manager

    def test_partial_waves_match_cpu_prefix_for_every_method_and_reuse_full_pool(self):
        cases = ((50, 32, [32, 18]), (65, 32, [32, 1]), (17, 8, [8, 1]),
                 (16, 8, [8]), (3, 2, [2, 1]), (2, 32, [2]), (1, 1, [1]))
        for runs, workers, pool_sizes in cases:
            with self.subTest(runs=runs, workers=workers):
                config, records, harness, manager = self.run_fake(runs, workers)
                self.assertEqual(config['execution_revision'], 'matched-wave-cpu-v2')
                self.assertEqual(config['affinity_policy'], 'fixed_cpu_prefix_per_wave')
                self.assertEqual([pool.width for pool in harness.pools], pool_sizes)
                self.assertEqual(harness.events, [event for width in pool_sizes
                                                 for event in (('create', width), ('shutdown', width))])
                expected_slots = list(range(0, 64, 2))+list(range(1, 64, 2))
                for wave, cpu_ids in enumerate(config['wave_cpu_ids']):
                    width = min(workers, runs-wave*min(workers, runs))
                    self.assertEqual(cpu_ids, expected_slots[:width])
                    pools = set()
                    for method in config['algorithms']:
                        jobs = [(pool, job) for pool, job in harness.jobs
                                if job['wave'] == wave and job['algorithm'] == method]
                        self.assertEqual(len(jobs), width)
                        pools.update(pool for pool, _ in jobs)
                        for pool, job in jobs:
                            self.assertEqual(pool.width, width)
                            self.assertEqual(pool.cpu_ids, cpu_ids)
                            self.assertEqual(job['wave_cpu_ids'], cpu_ids)
                            self.assertEqual(job['seed'], config['seed']+job['chain'])
                    self.assertEqual(len(pools), 1)
                for rows in records.values():
                    self.assertEqual([row['chain'] for row in rows], list(range(runs)))
                self.assertFalse(any(barrier.aborted for barrier in manager.barriers))

    def test_os_managed_workers_record_no_strict_affinity_promise(self):
        config, records, harness, _ = self.run_fake(3, 2, cpus=[None]*4)
        self.assertEqual(config['wave_cpu_ids'], [[None, None], [None]])
        self.assertEqual(config['affinity_policy'], 'os_managed_no_strict_pinning')
        self.assertEqual([pool.width for pool in harness.pools], [2, 1])
        for rows in records.values():
            self.assertTrue(all(row['worker']['cpu_id'] is None and not row['worker']['pinned'] for row in rows))

    def test_wrong_cpu_set_unpinned_worker_or_duplicate_worker_fails_closed(self):
        for mutation in (lambda row: row['worker'].update(cpu_id=999),
                         lambda row: row['worker'].update(pinned=False),
                         lambda row: row['worker'].update(pid=42)):
            harness = PoolHarness(mutate_record=mutation)
            with self.subTest(mutation=mutation), self.assertRaises(RuntimeError):
                self.run_fake(3, 2, harness=harness)
            self.assertTrue(all(pool.closed for pool in harness.pools))
            self.assertEqual(len(harness.pools), 1)

    def test_submit_and_worker_failures_abort_peers_and_close_pool(self):
        for failure in ('submit_failure', 'worker_failure'):
            harness = PoolHarness(**{failure: True})
            error = RuntimeError if failure == 'submit_failure' else ValueError
            with self.subTest(failure=failure), self.assertRaisesRegex(error, 'simulated'):
                self.run_fake(3, 2, harness=harness)
            self.assertTrue(harness.jobs[0][1]['barrier'].aborted)
            self.assertTrue(all(pool.closed for pool in harness.pools))
            self.assertEqual(len(harness.pools), 1)

    def test_changed_affinity_after_configuration_fails_before_workers_start(self):
        hardware = {'capacity': 4, 'cpu_ids': [0, 2, 1, 3]}
        with patch.object(experiment, 'cpu_info', return_value=hardware):
            config = experiment.configuration('quick', 2, {'runs': 3})
        with patch.object(experiment, 'cpu_info', return_value={**hardware, 'cpu_ids': [1, 3, 0, 2]}), \
                patch.object(experiment.multiprocessing, 'get_context') as context:
            with self.assertRaisesRegex(ValueError, 'CPU allocation changed'):
                experiment.run_chains(Path('unused'), config, {})
            context.assert_not_called()

    def test_invalid_duplicate_or_mixed_cpu_inventory_is_rejected(self):
        for cpus in ([0, 0], [0, None]):
            with self.subTest(cpus=cpus), \
                    patch.object(experiment, 'cpu_info', return_value={'capacity': 2, 'cpu_ids': cpus}), \
                    self.assertRaises(ValueError):
                experiment.configuration('quick', 2)

    def test_chain_archive_records_expected_cpu_set_and_execution_revision(self):
        from dcs_bench import timed_radial
        barrier = SimpleNamespace(wait=Mock(), abort=Mock())

        def sample(mapping, config, seed, before_sampling, after_sampling):
            before_sampling()
            after_sampling()
            return np.array([[.1, 2.]]), {'mean_all_iterations': [.1, 2.], 'seconds': .01}

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'chain.npz'
            job = dict(algorithm='SCS', chain=0, seed=901001, wave=1, wave_size=2,
                       wave_cpu_ids=[0, 2], config={'affinity_policy': 'fixed_cpu_prefix_per_wave'},
                       mapping={}, barrier=barrier, output=str(path))
            with patch.object(timed_radial, 'run_timed', side_effect=sample), \
                    patch.object(experiment, '_ALLOCATION', {'cpu_id': 2, 'pinned': True}), \
                    redirect_stdout(StringIO()):
                row = experiment.chain_job(job)
            self.assertEqual(row['wave_cpu_ids'], [0, 2])
            self.assertEqual(row['execution_revision'], 'matched-wave-cpu-v2')
            self.assertEqual(row['affinity_policy'], 'fixed_cpu_prefix_per_wave')
            self.assertEqual(row['worker']['cpu_id'], 2)
            self.assertEqual(barrier.wait.call_count, 3)
            barrier.abort.assert_not_called()
            with np.load(path, allow_pickle=False) as archive:
                self.assertEqual(json.loads(str(archive['metadata'])), row)


if __name__ == '__main__':
    unittest.main()
