# Parallel chains on CPU cores

The launcher has a **CPU workers (parallel chains)** selector. A setting of 4
allows up to four independent chains to run concurrently. The initial selection
is at most four workers and leaves one detected slot free when possible. Select
**1 — sequential** for a single worker. More workers also require more RAM.

Run **Quick check** first. Its production configuration has two chains per
method, so it uses at most two workers even if you select a larger count. The
four-chain logistic reference can use up to four. Full paper experiments have
20 or 50 production chains per method and can use the larger selected count.

The terminal equivalent is `--workers 4`, accepted by `run_experiment.py`,
`prepare_reference.py` and `scripts/reproduce_all.py`. Their default remains 1.
The launcher passes the selected count to both reference preparation and
production experiments. Worker counts above the detected capacity are rejected.

## What runs concurrently

Each chain runs in a separate Python process. Within a method, chain `i` is
assigned to worker `i % min(workers, chains)`. Seeds depend on experiment seed,
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
NumPy/JAX imports occur after each worker's CPU assignment. A worker compiles its
NUTS functions once and reuses them for its assigned chains. Draw arrays are
written directly to separate atomic checkpoint files instead of being copied
back through a process pool.

## CPU allocation and timing

- **Linux:** workers are pinned to distinct available physical cores, selecting
  one logical CPU per core within the coordinator's allowed CPU set. If physical
  topology is unavailable, the fallback uses allowed logical CPUs and records
  that limitation.
- **Windows:** workers are pinned to distinct physical cores in the current
  processor group. On machines with more than 64 logical CPUs, this implementation
  uses only that group. If the native affinity APIs fail, parallel capacity is
  restricted to one rather than silently claiming pinned execution.
- **macOS:** worker count is limited using the physical core count when available.
  Core assignment remains under the OS scheduler; strict CPU pinning is not
  provided. Performance and efficiency cores may have different speeds.

All workers set BLAS/OpenMP and TensorFlow thread limits to one. JAX is configured
for CPU, one host device, and disabled multithreaded Eigen operations. On systems
without CPU pinning, these settings do not guarantee that every runtime operation
uses exactly one physical core.

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
slot, PID, assigned CPU, verified pinning, thread settings, and start/finish
timestamps, in addition to its existing warmup and production timings.
`nuts_compilation.json` sums compilation durations across worker launches,
including resumed launches. That sum is an amortized resource cost, not elapsed
parallel wall time. Use the manifest's invocation wall time to assess total
turnaround, which also includes setup, process startup, saving and plotting.

## Stop, resume and update

The **Stop** button interrupts the coordinator and workers. A failed worker
stops its siblings and fails the experiment. Completed checkpoint files remain
valid; interrupted chains restart from their recorded seed.

Resume uses the **original saved worker count**, even if you change the selector.
It checks the original code, configuration and execution environment. To change
worker count, start a new run. After updating from the earlier sequential-only
release, start a **new results folder**; existing results can still be replotted,
but cannot be resumed with changed scientific source files.

Do not update the code or move the project while an experiment is running.
