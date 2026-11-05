# Historical version-8 own-kernel calibration results

Recorded on 4 November 2026. These are prespecified exploratory runs, not full-paper evidence. The tables and archived numerical results below are unchanged. Current version 9 uses a different SCS VI and defaults to the authors' fixed h=0.1, with acceptance adaptation available; see [the active author-baseline study](AUTHOR_SCS_CALIBRATION.md).

The version-8 calibration used the actual billiard and SCS production kernels,
the same bounded physical radial-movement score and the same pilot transition budget.
RWM, BW and reflected BW keep their previous calibration. Targets, maps, VI,
production kernels, radial estimators and ESS definitions are unchanged.

Both 100D skewed Student-t(3) cases use 20 chains per method, 2.00 production seconds
per chain, 1,000 pilot transitions, four workers and fresh sampling seed batch 6.
VI on uses 400 updates with 500 draws each; VI off retains R=1, zero centre/deformation.
The first third of production iterations is discarded. Each pilot is separate.

| VI | Method | Tail MSE | Mean-norm MSE | Samples/s | Tail ESS/s | Norm ESS/s |
|---|---|---:|---:|---:|---:|---:|
| on | DCS-RWM | 7.02e-07 | 7.78e-03 | 154,323.03 | 8,978.38 | 8,593.32 |
| on | DCS-BW | 8.83e-07 | 3.46e-03 | 147,611.10 | 7,580.56 | 7,668.85 |
| on | SCS | 3.42e-07 | 1.04e-02 | 104,900.79 | 12,957.77 | 7,137.72 |
| on | DCS-BW-reflect | 2.01e-07 | 6.68e-03 | 107,872.25 | 37,865.95 | 15,956.30 |
| on | DCS-Billiard | 1.05e-07 | 1.92e-03 | 108,008.82 | 34,887.32 | 21,830.57 |
| off | DCS-RWM | 8.41e-07 | 1.24e-02 | 85,537.51 | 5,641.50 | 5,853.83 |
| off | DCS-BW | 1.76e-06 | 1.25e-02 | 83,460.87 | 6,002.80 | 6,133.47 |
| off | SCS | 4.41e-07 | 6.67e-03 | 60,584.81 | 12,272.82 | 9,610.69 |
| off | DCS-BW-reflect | 1.01e-07 | 2.12e-03 | 62,991.60 | 25,282.00 | 13,511.29 |
| off | DCS-Billiard | 2.16e-07 | 2.43e-03 | 60,851.59 | 22,570.08 | 15,558.55 |

Samples/s uses every production iteration, including burn-in and rejected repeats.
ESS/s is the descriptive sum of separate chain functional ESS divided by total production time;
see [ESS definitions](ESS.md). It is not a pooled-chain convergence diagnostic.

## Matched historical-calibration controls

An additional 80 chains use the published version-7 policies: SCS acceptance calibration
and billiards inheriting the reflected-BW pilot. Each control reuses the exact primary maps,
chain seeds, pilot budget, worker count and production time. All cases and the new pilot
search were fixed before observing results. No outcome-based retuning was performed.

| VI | Method | New/old tail MSE | New/old norm MSE | New/old samples/s |
|---|---|---:|---:|---:|
| on | SCS | 1.00 | 1.46 | 0.99 |
| on | DCS-Billiard | 0.30 | 0.27 | 1.05 |
| off | SCS | 0.54 | 0.66 | 1.00 |
| off | DCS-Billiard | 0.38 | 0.24 | 0.98 |

Ratios below one mean smaller observed MSE; speed ratios above one mean more iterations per second.
Billiards had lower observed MSE than SCS on both radial quantities with and without VI.
Reflected BW had lower observed MSE than billiards on both quantities without VI.
For SCS, the new pilot improved both no-VI MSEs, but its VI-on norm MSE was 45.60% higher
than the historical-pilot control; its VI-on tail MSE was essentially unchanged.
Billiard and SCS sampling throughput was close. These are observations from this batch.

The pilots produce different initial production states and random trajectories, so individual
old/new chains are not trajectory-matched. This single batch does not establish a stable ranking.
The t(3) norm has finite variance but no fourth moment; empirical MSE can vary substantially.

## Historical reproduction and current runs

These results belong to version 8. Running the current launcher or command
line creates version 9 and does **not** reproduce these numerical settings.
The version-8 primary implementation was published at
[`d89b0e6bdbad09991ec5a5910ab354951bca76dd`](https://github.com/MihaBresar/DiffeomorphicMonteCarlo/tree/d89b0e6bdbad09991ec5a5910ab354951bca76dd).
The archived validation records below retain the primary and historical-control
budgets, source/map hashes and numerical outcomes. Reproducing these historical
controls requires the recorded version-8 scientific sources and the old
SCS acceptance/billiard reflected-pilot policies; do not substitute current
version-9 defaults or bypass source-provenance checks.

For current reproducible primary and matched-map sensitivity runs, use the
[version-9 commands](AUTHOR_SCS_CALIBRATION.md#reproduction). They compare
acceptance-tuned SCS with fixed h=0.1 under its authors' fitting routine.
They answer a different question from the historical version-8 controls.

Full paper settings use 50 chains, 20.00 seconds and 1,500 separate pilot
transitions per method. Wall-clock budgets mean exact sample counts and
errors depend on the computer and scheduling, even with identical random
seeds. Freeze the fitting and calibration protocol before running a fresh
full-paper seed batch and report the prespecified sensitivity outcomes.

The shared pilot rule is a defined benchmark, not evidence that every method
has equally optimal tuning. These results concern two radial quantities only.
Earlier [coordinate results](COMPARISON_RESULTS.md) include cases favoring SCS
and remain part of the public record; radial results do not establish general
or marginal superiority. Twenty-chain results here are exploratory, and fresh
full-paper runs assess the frozen implementation rather than guaranteeing its
ordering in advance.
The present DCS-Billiard proposal and calibration differ from the original
paper's BW implementation. These results evaluate the current variant;
the original paper's figures require their own reproduction and correction.
The documented map-family, effective-coordinate and tuning differences remain
part of the experimental protocol.

[Full-precision CSV](../validation/own_kernel_calibration_results.csv),
[JSON results](../validation/own_kernel_calibration_results.json),
[validation record](../validation/own_kernel_calibration_checks.json),
[calibration protocol](COMPARISONS.md).
