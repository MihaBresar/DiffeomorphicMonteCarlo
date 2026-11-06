# Saving and processing results

Sampling, archive writing, diagnostics and figure rendering have separate progress messages and recorded times. Reports can take time after chains finish. Logical array sizes describe stored data, not peak process memory.

## Main comparison

Workers accumulate mean norm and radial-tail counts from every production iteration, including rejected repeats. Temporary scalar histories support unthinned functional ESS after exact first-third discard. A shared lock bounds concurrent FFT work.

Scalar capture and buffer flushes are inside production time. ESS calculation, waiting and final reports are outside. Temporary unthinned histories are removed after diagnostics; small thinned traces and metadata remain. See [ESS definitions](ESS.md).

## General experiments

Workers stream retained physical states into temporary files with bounded write buffers. Burn-in selects the appropriate view; archive writing copies data in bounded blocks. Completed chains release their storage before the worker takes another job.

Streaming, coordinate conversion, thinning and buffer flushes are included in production time. Final archive serialization and report construction are separate. NUTS also records required scalar diagnostics during sampling.

The compact 200D baseline report reads cached radial estimates and ESS metadata. It omits coordinate quantiles and cross-chain diagnostics. References are analytic.

Full posterior reports process one chain and one functional at a time, staging scalar traces on disk. Cross-chain diagnostics use the common prefix of the shortest chain. Individual diagnostic concurrency is bounded by the selected worker count. Large runs still require temporary disk space and autocorrelation/sorting work.

Archives use a common lossless storage/compression policy for all methods. Metadata records saving choices and costs. Temporary mappings close before deletion, including on Windows.

## Recreating reports

```sh
python run_experiment.py --plot-only results/example-run
```

A completed summary regenerates figures without repeating diagnostics. To explicitly recompute diagnostics from saved chains:

```sh
python run_experiment.py --plot-only results/example-run --recompute-diagnostics
```

Saved-chain files and their original configuration remain unchanged. Rebuilding a report does not recover transitions that were not retained. A new scientific configuration or timing protocol requires a fresh complete run.
