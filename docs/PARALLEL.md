# Parallel chains on CPU cores

The launcher has a **CPU workers (parallel chains)** selector. A setting of 4
allows up to four independent chains to run concurrently. The initial selection
is at most four workers and leaves one detected slot free when possible. Select
**1 — sequential** for a single worker. More workers also require more RAM.

Click **Use all CPU threads** to select the logical CPU capacity available to
the process, including simultaneous multithreading (SMT) where enabled. The
launcher detects this capacity instead of assuming a processor model or core
count. Available threads can be used while sufficient independent chains
remain. Setup, fitting and reporting may use fewer threads; additional workers
do not guarantee proportional speedups.

Run **Quick check** first. Its production configuration has two chains per
method, so it uses at most two workers even when a larger count is selected. The
four-chain logistic reference can use up to four. Full paper experiments have
20 or 50 production chains per method and can use the larger selected count.
The generic smaller reproductions have eight chains; the current sampler
comparisons have 20. They use at most eight or 20 simultaneous workers,
respectively. Selecting more workers than available chains does not increase
concurrency. Fitting, plotting, reference preparation and the final incomplete
batch can use fewer threads. Chain counts remain fixed by the experiment
configuration. Fewer workers can improve per-chain throughput when SMT, memory
bandwidth or cache contention dominates; assess the worker setting on the
measurement hardware.

The terminal equivalent is `--workers 4`, accepted by `run_experiment.py`,
`prepare_reference.py` and `scripts/reproduce_all.py`. Their default remains 1.
The launcher passes the selected count to both reference preparation and
production experiments. Worker counts above the detected capacity are rejected.

## What runs concurrently

Each chain runs in a separate Python process. The sampler comparisons use
synchronized waves: methods run separately with the same ordered CPU allocation
and number of simultaneous chains. In the other experiments, workers claim the
next pending chain from a shared queue as soon as they finish. An exclusively created
directory grants ownership of each chain; its job JSON stays immutable.
This prevents competing workers from moving one another's job files on Windows.
Faster workers can process more chains. Seeds depend on experiment seed,
scenario, method index and chain index; worker count does not alter them.
Fixed-iteration Metropolis chains have been checked against direct sequential
execution for identical saved draws. Timed chains can retain different numbers
of draws as CPU speed and load vary.

Methods and scenarios run in sequence. For Eight Schools and GARCH, **all NUTS
chains finish before any DCS chains start**. The DCS budget is then the mean
measured NUTS production time under the same worker setting. The map is fitted
once in the coordinator and its frozen parameters are shared with all workers.
The four independent logistic reference chains also run in parallel.

Workers start fresh Python interpreters instead of forking a live JAX runtime.
CPU affinity is established before a worker's sampling backend initializes. A worker compiles its
NUTS functions once and reuses them for subsequent chains. All workers finish
imports and compilation before warmup/sampling starts, so one chain's production
does not compete with another worker's initial compilation. Draw arrays are
written directly to separate atomic checkpoint files instead of being copied
back through a process pool.

## CPU allocation and timing

- **Linux:** workers are pinned to distinct allowed logical CPUs, first selecting
  one thread on every physical core, then SMT siblings. If physical
  topology is unavailable, the fallback uses allowed logical CPUs and records
  that limitation.
- **Windows:** workers are pinned to distinct logical CPUs in the current
  processor group. On machines with more than 64 logical CPUs, this implementation
  uses only that group. If the native affinity APIs fail, parallel capacity is
  restricted to one rather than silently claiming pinned execution.
  Assignment uses one thread per physical core first, then SMT siblings.
- **macOS:** all reported logical CPU slots are available to workers.
  Core assignment remains under the OS scheduler; strict CPU pinning is not
  provided. Performance and efficiency cores may have different speeds.

All workers set BLAS/OpenMP and TensorFlow thread limits to one. JAX is configured
for CPU, one host device, and disabled multithreaded Eigen operations. They also
set **`PJRT_NPROC=1` before backend creation**, limiting the pinned JAX 0.4.35 CPU
compute pools and the compiler's intra-operation parallelism budget. This
overrides an inherited `PJRT_NPROC`; it takes precedence over `NPROC`, which is
left unchanged. The common entry point applies the same setting to the
coordinator before fitting and reference preparation. Runtime helper threads
can remain; this is not a claim that the process contains only one thread.

The recorded policy is **`pjrt-single-compute-thread-v1`**. An audit on
4 November 2026 found that the earlier Eigen-only flags left other XLA kernels
free to consume multiple CPU cores per worker. Which method was affected
changed with dimension, distorting wall-time throughput comparisons on the
unbound macOS workers. The correction applies to every method, including SCS,
without changing the target, proposal or tuning rules. **Rerun all methods
together for new wall-time, sample-count and MSE figures**: changing throughput
also changes how many iterations finish in a fixed-time experiment. Historical
reports remain readable, but their rates must not be mixed with corrected runs.

The pinned XLA source reads the setting in
[`DefaultThreadPoolSize`](https://github.com/openxla/xla/blob/76da730179313b3bebad6dea6861768421b7358c/xla/pjrt/utils.cc#L814)
and uses it for the
[CPU runtime pools and compilation budget](https://github.com/openxla/xla/blob/76da730179313b3bebad6dea6861768421b7358c/xla/pjrt/cpu/cpu_client.cc#L385).
The Eigen flag alone controls Eigen calls, and the LLVM parallel-codegen flag
controls compilation rather than sampling. Process CPU time divided by measured
wall time is useful evidence of actual consumption; a worker count or environment
setting by itself does not prove equal resource use.

The native mechanisms are documented in Python's
[CPU affinity API](https://docs.python.org/3.12/library/os.html#os.sched_setaffinity)
and Microsoft's [process affinity API](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setprocessaffinitymask).

For comparable figures, keep **worker count, hardware, and other machine load
the same across methods**. CPU pinning does not isolate memory bandwidth, caches,
thermal limits, or background processes. Final batches and partially resumed
runs can have fewer active workers. Parallel wall-clock results describe that
execution policy; they should not be presented as identical to isolated
single-worker timings. On macOS, a sequential run is the more controlled option
for isolated timing comparisons.

Each experiment's `manifest.json` records the execution plan, CPU topology,
environment, and total wall time of each invocation. Each chain records worker
slot, PID, assigned CPU, verified pinning, thread settings (including `PJRT_NPROC`), and start/finish
timestamps, in addition to its existing warmup and production timings.
`nuts_compilation.json` sums compilation durations across worker launches,
including resumed launches. That sum is an amortized resource cost, not elapsed
parallel wall time. Use the manifest's invocation wall time to assess total
turnaround, which also includes setup, process startup, saving and plotting.

## Stop and update

The **Stop** button interrupts the coordinator and workers. A failed worker
stops its siblings and fails the experiment. Completed checkpoint files remain
valid. The launcher's **Start run** button starts from scratch in a new dated
results folder, including after a stopped or failed run. Download updates into
a fresh project folder before starting again; see [the Windows queue fix](WINDOWS_QUEUE_FIX.md).

Advanced command-line workflows still support `--resume` and `--plot-only`.
Resume checks the original code, configuration, worker count and execution
environment. To change worker count or scientific implementation, use a new
results folder. See [the command-line rerun guide](RERUN.md).

Do not update the code or move the project while an experiment is running.
