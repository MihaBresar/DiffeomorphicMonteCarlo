# Parallel chains on CPU cores

The launcher has a **CPU workers (parallel chains)** selector. A setting of 4
allows up to four independent chains to run concurrently. The initial selection
is at most four workers and leaves one detected slot free when possible. Select
**1 — sequential** for a single worker. More workers also require more RAM.

Click **Use all CPU threads** to select every thread available to the process.
The AMD Ryzen 9 9950X3D has [16 cores and 32 threads](https://www.amd.com/en/products/processors/desktops/ryzen/9000-series/amd-ryzen-9-9950x3d.html).
With SMT enabled and all threads available to Python, the launcher permits **32
workers**. The old cap at one worker per physical core has been removed. This
option fills available threads while sufficient independent chains remain;
it does not promise 100% CPU activity during every stage or a 2× speedup over 16.

Run **Quick check** first. Its production configuration has two chains per
method, so it uses at most two workers even if you select a larger count. The
four-chain logistic reference can use up to four. Full paper experiments have
20 or 50 production chains per method and can use the larger selected count.
The smaller reproduction has eight chains and therefore uses at most eight
workers. This can account for low CPU utilization even with 16 or 32 workers
selected. Fitting, plotting, reference preparation and the final incomplete
batch can use fewer threads. We do not add chains merely to fill the CPU, since
that would change the specified experiment. Sixteen workers may give better
per-chain throughput than 32 if SMT contention dominates; measure on your CPU.

The terminal equivalent is `--workers 4`, accepted by `run_experiment.py`,
`prepare_reference.py` and `scripts/reproduce_all.py`. Their default remains 1.
The launcher passes the selected count to both reference preparation and
production experiments. Worker counts above the detected capacity are rejected.

## What runs concurrently

Each chain runs in a separate Python process. Workers claim the next pending
chain from a shared queue as soon as they finish. An exclusively created
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
NumPy/JAX imports occur after each worker's CPU assignment. A worker compiles its
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
