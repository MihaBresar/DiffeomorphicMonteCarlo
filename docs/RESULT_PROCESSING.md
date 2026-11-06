# Saving and processing results

Sampling, archive writing, estimation and figure rendering have separate progress messages and recorded times. ESS is restricted to the main DCS–SCS comparison and its Numba extension. Logical array sizes describe stored data, not peak process memory.

## Main comparison

Workers accumulate mean norm and radial-tail counts from every production iteration, including rejected repeats. Temporary scalar histories support unthinned functional ESS after exact first-third discard. A shared lock bounds concurrent FFT work.

Scalar capture and buffer flushes are inside production time. ESS calculation, waiting and final reports are outside. Temporary unthinned histories are removed after diagnostics; small thinned traces and metadata remain. See [ESS definitions](ESS.md).

## General experiments

Workers stream retained physical states into temporary files with bounded write buffers. Burn-in selects the appropriate view; archive writing copies data in bounded blocks. Completed chains release their storage before the worker takes another job.

Streaming, coordinate conversion, thinning and buffer flushes are included in production time. Final archive serialization and report construction are separate. Standard NUTS runs retain physical states without writing separate per-iteration diagnostic files. Internal adaptation and transitions are unchanged.

The compact 200D baseline report reads cached radial estimates. It omits coordinate quantiles and chain diagnostics. References are analytic.

Posterior reports process one saved chain at a time to calculate estimates, moment errors and the retained accuracy summaries. They do not stage traces for cross-chain analysis or calculate ESS, R-hat or autocorrelation-based MCSE. Eight Schools and GARCH retain their first- and second-moment RMSE tables and skip unused coordinate-quantile sorting. Saving and reading large physical chains still takes time.

Independent reference preparation has separate quality checks and reference-error estimates. These validate the reference used for error calculations; they are not repeated diagnostics of the benchmark chains.

Archives use a common lossless storage/compression policy for all methods. Metadata records saving choices and costs. Temporary mappings close before deletion, including on Windows.

## Recreating reports

```sh
python run_experiment.py --plot-only results/example-run
```

A completed summary regenerates figures without rereading chains. To explicitly recompute estimates from saved chains:

```sh
python run_experiment.py --plot-only results/example-run --recompute-estimates
```

Saved-chain files and their original configuration remain unchanged. Rebuilding a report does not recover transitions that were not retained. A new scientific configuration or timing protocol requires a fresh complete run.
