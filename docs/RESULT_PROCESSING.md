# Saving and processing large runs

Current comparisons (version 7 onward) include functional ESS for the radial-tail indicator
and norm. They capture unthinned scalar histories into temporary files inside
the sampling clock, then compute ESS after the wave finishes. A shared lock
limits FFT work to one chain at a time, and diagnostic time is recorded
separately. The temporary histories are discarded after ESS; only small
configured-thinning traces and diagnostic records are saved. Current generic
and compact baseline reports also include functional ESS, explicitly on their
actual retained-draw basis. See [ESS definitions](ESS.md).

Sampling, saving, convergence diagnostics and figure rendering are separate
stages. The launcher prints when each chain finishes sampling, how many retained
draws it is saving, and the saving duration. Reporting prints its chain and
diagnostic progress. `processing.json` records reporting and rendering times;
each algorithm's summary records its processing stages and logical array sizes.
These sizes describe the stored arrays, not the process's peak RAM usage.

Production time now consistently includes thinning, physical-coordinate
conversion, streaming retained draws and flushing the final write buffer.
NUTS also records its scalar diagnostic traces within that clock. HnR still
receives mean NUTS production time. Final NPZ serialization and report
construction take place outside that clock. The existing
`seconds_including_setup` measure charges sampling, warmup and shared setup;
it is not the elapsed time until the entire report is ready. The run manifest
also records total invocation wall time.

## Compact 200D random-walk report

The `skew-t-200d-baselines` experiment uses `reporting: radial` in all presets.
Workers calculate mean norm and radial-tail probability in bounded blocks
from all retained post-burn-in states, then cache these estimates and their
functional ESS in `radial_summary`. The compact report reads cached metadata
without reopening large draw arrays. It shows estimates, MSE, counts, timing,
ESS and ESS/s. References are analytic. It omits coordinate quantiles,
directional checks, cross-chain diagnostics, MCSE and R-hat; the original
radial plots remain.

For older checkpoints missing ESS, report processing reads one saved physical
chain at a time, constructs scalar norms in bounded blocks and computes scalar
ESS. The checkpoint is never rewritten. The actual retained draw stride and
diagnostic cost are recorded. A completed summary regenerates figures without
repeating diagnostics. See [ESS definitions](ESS.md).

## Why Eight Schools exposed the bottleneck

The default 100D BW/RWM/SCS comparison calculates its two radial means online
and saves compact radial traces. Its report uses those saved means. The generic
Eight Schools experiment saves all ten physical coordinates and calculates
diagnostics for twenty functionals: each parameter and its square. Its full
preset also has more chains and a larger NUTS sampling count than the smaller
preset, giving HnR a longer matched sampling budget.

A fast HnR chain can produce many more draws than NUTS in the same sampling
time. The old generic report loaded every physical chain, constructed all the
functional arrays, and stacked their common prefixes before calculating
diagnostics. Several large copies could therefore remain in RAM together.
Generic BW/RWM experiments used this same reporting code and could encounter
the same problem with sufficiently large outputs.

## The 200D RWM allocation failure

A separate failure occurred before report construction: the worker held every
sampling batch in a list, then `np.concatenate` requested a second complete
array. The reported shape, 594,918 by 200 float64 values, requires 907.77 MiB
per copy. Thirty-two similarly sized workers could hold 28.37 GiB of raw
chunks before allocating the extra copies, numerical runtimes and workspaces.
The trace shows all 50 BW chains finished before RWM failed in its first wave;
different draw counts within a time budget produce different memory loads.

Every generic experiment worker now streams retained states into a temporary
file using an 8 MiB write buffer. It preserves the original thinning grid and
does not concatenate the complete trace. Burn-in selects a read-only view;
the NPZ writer copies the required bytes directly from the backing file in
bounded blocks. Each worker releases a completed chain before taking its next
job. The parent cleans each launch's temporary draws and incomplete archives
after workers exit, including on a worker failure or Stop.

This applies to DCS, ordinary RWM, Python fallback kernels and both NUTS
implementations. NumPyro production uses compiled batches of at most 1,024
transitions plus an exact remainder. Its carried sampler state preserves the
random sequence. NUTS scalar traces use separate small buffers. These limits
bound output buffers; numerical runtimes, current batches and reporting have
additional memory requirements.

## Full diagnostic reports

- Reporting reads one physical chain at a time. Exact means and quantiles and
  individual-chain diagnostics use all retained draws, as before.
- Individual-chain diagnostics use up to four threads, bounded by the run's
  selected CPU-worker count. Each thread processes one functional at a time;
  cross-chain diagnostics remain sequential to bound their larger working
  arrays. The summary records the diagnostic worker count.
- Full functional traces are staged in temporary files under the result
  directory. Cross-chain diagnostics open one functional at a time, using
  the original common prefix of the shortest chain. No extra thinning or
  diagnostic subsampling is introduced.
- Temporary mappings close explicitly before deletion on Windows. Completed
  files are released as each functional finishes; temporary files are cleaned
  on normal completion and handled errors. Scratch capacity is checked before
  allocating a cross-chain mapping.
- Coordinate ESS plots request only bulk ESS, removing calculations of unused
  tail ESS, MCSE and R-hat for those coordinate traces.
- Saving uses a standard, lossless NPZ archive. The same deterministic rule
  applies to every sampler: compress at level 1 if at most 1 MiB sampled from
  the beginning, middle and end compresses below 90% of its original size;
  otherwise store the draws directly. Metadata records this decision and the
  saving measurements. Both formats load with ordinary `numpy.load`, and
  existing compressed archives remain readable.
- Reading the NUTS timing metadata to set HnR's budget no longer loads the
  NUTS draw arrays.

The processing still sorts and computes autocorrelations of the full retained
data. Large runs therefore require time and temporary disk space. The change
removes the requirement to hold every parameter of every chain in RAM
simultaneously; it does not make diagnostics constant-cost.

## Existing results

This update does not change the transition kernels, tuning, target, burn-in or
statistical definitions. Completed chains remain usable, and refreshing their
figures does not require new MCMC. For new timed comparisons, rerun all methods
together: streaming output is now included consistently in their production
clocks. Do not combine old timings with new timings. The launcher keeps
its existing fresh-run workflow; install the update and start it as usual for
new experiments.

Advanced command-line use can refresh figures from a completed summary:

```bash
python run_experiment.py --plot-only results/my-run
```

To recalculate statistics from saved chains using the bounded-memory pipeline:

```bash
python run_experiment.py --plot-only results/my-run --recompute-diagnostics
```

If no completed summary exists, `--plot-only` rebuilds it automatically from
the saved chains. Rebuilding leaves the chain files unchanged and uses their
original run fingerprint, configuration and references.

The exact array size from the 200D failure has also passed a storage-only
check, including all 396,612 post-burn-in rows and exact byte hashes. The
check uses synthetic repeated blocks to exercise storage, not a new MCMC
experiment. See [the memory check](../validation/draw_storage_memory_check.json).

## Validation

A local macOS comparison processed eight saved Eight Schools HnR chains and
then repeated those inputs six times to test 48-chain processing load. These
repetitions measure implementation scaling, not new independent MCMC results.
Both implementations ran in separate fresh processes with the same imports
and a warmed diagnostic library.

| Saved-chain inputs | Original seconds | Updated seconds | Original peak RAM (MB) | Updated peak RAM (MB) |
| --- | ---: | ---: | ---: | ---: |
| 8 | 4.02 | 2.99 | 490.47 | 342.79 |
| 48 | 25.42 | 18.64 | 1288.09 | 633.90 |

The updated runs used four individual-diagnostic threads. All means, quantiles,
MSE values and diagnostics matched exactly, and all input file hashes were
unchanged. These are individual local timing observations, not estimates of
Windows speed or measurements of the user's completed full run. See the
[receipt](../validation/result_processing_benchmark.json). RAM figures include
imports and diagnostic working arrays; MB here means 1,000,000 bytes.

To repeat the comparison, use a saved **Smaller reproduction** posterior run:

```bash
python scripts/benchmark_result_processing.py --input results/my-eight-schools --repeat 6 --diagnostic-workers 4
```

The baseline deliberately recreates the previous allocation of all chains in
RAM. The benchmark's default is one copy of the input chain set.
