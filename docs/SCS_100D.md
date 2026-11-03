# One 100D skewed-t comparison

The default launcher experiment compares **DCS-RWM, DCS-BW and SCS at the same
sampling time per chain** on the existing 100-dimensional normal-CDF-skewed
Student-t target, with ν=3 and α=(100,−100,0,…). It estimates P(‖X‖≥50) and E‖X‖
against exact radial truth. All three methods start at the physical vector of
ones, using the correct inverse map. The protocol identifier is
`timed-matched-dcs-scs-v3`.

| Preset | Independent chains per method | Timed sampling seconds per chain | Separate warmup iterations | Stored radial trace interval |
|---|---:|---:|---:|---:|
| Quick check | 2 | 0.05 | 100 | 1 |
| Smaller reproduction | 20 | 2.00 | 1,000 | 10 |
| Full paper settings | 50 | 20.00 | 1,500 | 50 |

After separate fitting and warmup, each method samples for its time budget.
The first third of **its actual timed iterations** is discarded. Every remaining
iteration contributes to the estimates, including repeated states after
rejection. The stored trace interval only reduces saved instantaneous radial
traces; it does not change the estimates or sample denominator. Different
methods and chains can produce different iteration counts.

Quick check uses abbreviated fitting and cannot support a performance ranking.
Fifty independent chains improve precision of the empirical error assessment;
they do not remove bias caused by insufficient exploration within a chain.

## Matched map fitting and separate step calibration

Both the DCS ball map and the SCS sphere map use the shared reverse-KL fitting
framework in `dcs_bench/tuning.py`. They receive the same optimizer budget and
fresh base draws at every update, seed 42, Adam learning rate 0.01, and gradient
norm clipping at 10. They start from the same parameter convention: unit scale,
zero centre and zero Möbius shift. Their fitted scales, centres and Möbius
parameters may differ because the map geometries differ. DCS uses β=2 and SCS
uses latitude 1.10. DCS-RWM and DCS-BW share the fitted ball map.

| Preset | Optimizer updates per map | Fresh draws per update | Total training draws per map |
|---|---:|---:|---:|
| Quick check | 20 | 128 | 2,560 |
| Smaller reproduction | 400 | 500 | 200,000 |
| Full paper settings | 1,500 | 2,000 | 3,000,000 |

Fitting uses the same target as production, without the old clipped-CDF or
physical-coordinate-clipping objectives. Both fitted maps remain fixed during
warmup and sampling. Parameters, optimizer settings, fitting costs and training
histories are saved, so the comparison is conditional on the recorded fits.

Every chain starts with proposal scale 0.10 and calibrates it with the same raw
dual-averaging rule toward acceptance 0.234 during the separate warmup listed
above. Both implementations feed the observed accept/reject indicator to that
rule. **Each resulting scale is frozen before timed sampling starts.** The map
and step remain fixed during the timed first-third burn-in and all subsequent
estimation. Matching this rule does not assume that 0.234 is optimal for each
method, or that equal numerical step sizes have equal geometric meanings.

DCS-RWM proposes isotropic Gaussian increments; DCS-BW proposes uniformly in a
Euclidean ball. Both reject proposals outside the open unit ball. SCS uses its
tangent-Gaussian proposal, normalization, cap step-out, projection, Jacobian and
Metropolis transition. All three kernels run in compiled JAX CPU scans with the
same batch size and state-output policy. Each transition evaluates current and
proposed densities; DCS skips target evaluation for proposals outside its ball,
which are direct rejections by definition. Physical projection and radial estimation run after each
batch with the same recording policy for all methods. Compilation finishes
before timing; completed JAX work is synchronized at each timing boundary.

## Unmodified author SCS transition

The primary comparison directly calls the authors' unmodified
`rwm_bright_side_stepout` transition under the common map-fitting,
step-calibration and JAX CPU timing protocol above. DCS uses the same scan and
recording structure. This is **not a literal execution of the authors' original
experiment runner**: the author sampler is retained, while fitting and proposal
calibration are matched to the other methods.

The MIT-licensed author source remains vendored without modification at
[`ea6f27a6cb1d0ca90d8c1cd92c32fc3dcf8ea565`](https://github.com/liusf15/sub-cauchy-sampler/tree/ea6f27a6cb1d0ca90d8c1cd92c32fc3dcf8ea565),
the paper-date revision. Source hashes and licence are in `third_party/scs_paper`.
The current SCS runs invoke that source directly; it also remains the reference
for transition validation and historical reports.
The authors' [`run_skewt.py`](https://github.com/liusf15/sub-cauchy-sampler/blob/ea6f27a6cb1d0ca90d8c1cd92c32fc3dcf8ea565/experiments/run_skewt.py)
uses a fixed proposal step of 0.10; that fixed-step policy is not used by the
current matched comparison.

**Target distinction:** the [SCS paper §4.1](https://arxiv.org/html/2601.11066v1#S4.SS1)
uses a Student-t CDF skewing factor and ν=1 or 2. All three methods here evaluate
the same existing normal-CDF-skewed ν=3 target; its radial truth is unchanged by
skewing. These runs do not reproduce the SCS paper's target or plotted values.

## Timing, burn-in and outputs

Fitting, compilation and separate warmup finish before the sampling clock
starts. The timed phase includes its first-third burn-in, transitions, radial
accumulation and compact in-memory recording. Setup, inter-worker
synchronization waits, disk writes and plotting are outside that clock; setup
costs and total invocation wall time are recorded separately.

All methods check elapsed time after the same **300-transition batches**. A
chain stops at the first completed batch meeting its budget; actual elapsed
time and overshoot are recorded. Its iteration count N is a multiple of 300.
Cumulative tail counts and norm sums are stored every 100 transitions, so N/3
is always an available boundary. Subtracting that boundary's cumulative values
gives the estimates over **all 2N/3 post-burn-in iterations**, without storing
every 100-dimensional state. Tests cover a burn boundary inside a timed batch,
variable counts and repeated states. A timing batch can be noticeable relative
to a 0.05-second smoke check; use the longer presets for comparisons.

Parallel chains run in synchronized waves, with only one method sampling at a
time. Method order rotates between waves. Each method receives identical wave
sizes and CPU allocation; Windows/Linux affinity fills physical cores before
SMT siblings. A 50-chain run can use 32 workers on the Ryzen 9 9950X3D. Sequential
runs provide the clearest isolated timing comparison; parallel timings describe
the recorded concurrent CPU workload. Keep the machine otherwise idle when
measuring speed.

`VsSCS.png`/PDF preserve the manuscript's three linear panels: tail MSE, norm
MSE and, for the equal-time design, average generated iteration count.
`scs_100d_estimates` adds radial estimate boxplots and MSE panels. Every chain
estimate is shown, so a small spread around an incorrect value remains visible.
Other plots show timing, actual iteration counts, throughput and empirical bias
squared plus variance. The timed protocol uses the final post-burn estimates;
it does not infer equal-time MSE by scaling errors from fixed-count runs.
Historical fixed-count reports retain runtime in the third `VsSCS` panel.

`paper_values.html` displays two decimals, with scientific notation for MSE and
small values; counts remain exact integers. CSV and JSON retain full stored
precision. Reports include actual total iterations, discarded and estimation
counts, per-chain errors, elapsed times and tuned parameters. Each independent
chain has equal weight in MSE, even when iteration counts differ. Variance uses
divisor n, so **MSE = empirical bias² + variance**, up to floating-point rounding.

Use **Start.bat → Set up environment → 100D skewed t (ν=3) → Start run** on
Windows, or **Start.command** on macOS. Run environment setup again after an
update. A command-line run must use a fresh output folder:

```bash
python run_experiment.py skew-t-100d-fixed --preset paper --workers 32 --output-dir results/scs-100d-timed-paper
```

The historical `skew-t-100d-fixed` identifier is retained for compatibility;
its default protocol now fixes **time**. `--time-budget` overrides seconds per
chain. The launcher and compatibility entry points route this identifier to
the same runner. Rerun this one experiment for the current matched comparison;
existing sampling results cannot be converted by replotting.

## Short matched checks, 3 November 2026

Two prespecified local checks used 20 chains per method, one worker, 1.00 second
of timed sampling per chain, 1,500 separate warmup steps and 1,500 × 2,000 VI
draws per map. Run A used fit seed 42 and chain seed 901001; run B used fit seed
43 and chain seed 902001. Actual mean sampling times were 1.00 seconds to two
decimals for every row. These short macOS ARM64 checks are conditional on two
map fits and do not replace the longer Full paper run.

| Run | Method | Mean generated iterations | Tail MSE | Mean-norm MSE |
|---|---|---:|---:|---:|
| A | DCS-RWM | 148,035 | 1.29e-06 | 1.59e-02 |
| A | DCS-BW | 106,335 | 2.08e-06 | 2.47e-02 |
| A | SCS | 115,740 | 7.19e-07 | 9.58e-03 |
| B | DCS-RWM | 146,895 | 8.92e-07 | 1.26e-02 |
| B | DCS-BW | 106,845 | 1.58e-06 | 2.21e-02 |
| B | SCS | 117,045 | 1.37e-06 | 1.18e-02 |

SCS had lower mean-norm MSE in both checks. The lowest tail MSE switched from
SCS to DCS-RWM. DCS-BW had higher MSE than SCS for both functions in both checks.
The observed DCS-RWM throughput advantage was about 1.26–1.28×; DCS-BW generated
about 0.91–0.92× as many iterations as SCS. Mean production acceptances were
0.22–0.23 for DCS and 0.18 for SCS despite the shared 0.234 warmup target.
Equal calibration rules therefore should not be read as a guarantee of equally
optimal proposal scales.

An archived before-update run used the same chain count and one-second budget
at commit `879297a0df348828f31b36a647d6ade9d98ac46c`; it generated about 14.46×
and 13.20× as many DCS-RWM and DCS-BW iterations as SCS, respectively. Fitting,
tuning and execution all changed together, so this before/after comparison
does not isolate each change's effect. The old ratio cannot be treated as an
algorithm-only speed advantage.

Full-precision settings, fitted maps, per-chain estimates, counts, times,
acceptances, source hashes and the before-update results are in
[`validation/scs_100d_matched_comparison.json`](../validation/scs_100d_matched_comparison.json).
The first third is removed from every count before estimation. To repeat the
two current checks in fresh folders:

```bash
python scripts/reproduce_scs_100d.py --preset reproduce --workers 1 --runs 20 --time-budget 1 --warmup 1500 --optimizer-steps 1500 --optimizer-samples 2000 --output-dir results/matched-check-a
python scripts/reproduce_scs_100d.py --preset reproduce --workers 1 --runs 20 --time-budget 1 --warmup 1500 --optimizer-steps 1500 --optimizer-samples 2000 --fit-seed 43 --seed 902001 --output-dir results/matched-check-b
```

## Superseded comparison protocols

The v2 protocol (`timed-dcs-author-scs-v2`) used separate original map-fitting
recipes, DCS-only proposal calibration, and different compiled timing drivers
for DCS and SCS. V3 matches these three parts. Old reports retain their recorded
policies when replotted; their values do not describe the current protocol.
The earlier fixed-count evidence below is also historical, including its
continuous DCS step adaptation. Neither historical set establishes the outcome
of a new v3 run.

## Historical fixed-count evidence, 11 September 2026

The following archived results predate the timed protocol. They used one worker
on macOS ARM64, 50 chains per method and **500,000 production iterations per
chain**, after 100 warmup iterations. DCS continued adapting its step during
production. These values describe those recorded maps, settings and hardware;
they are not the results of the current 20-second preset.

| Method | Mean sampling seconds per chain | Iterations/s | Tail MSE | Norm MSE |
|---|---:|---:|---:|---:|
| DCS-RWM | 0.31 | 1,595,420.79 | 2.81e-07 | 2.00e-03 |
| DCS-BW | 0.34 | 1,483,033.64 | 2.10e-07 | 2.02e-03 |
| SCS | 5.34 | 93,697.44 | 8.92e-08 | 1.27e-03 |

DCS-RWM and DCS-BW achieved 17.03× and 15.83× the SCS iteration throughput.
SCS had lower error per fixed number of iterations for both functions. No
equal-time MSE is inferred from these ratios. Squared empirical norm bias was
less than 1% of MSE for every method; the recorded error mostly reflected
variation across chains. More chains alone cannot establish that an earlier
biased run was repaired when fitting, adaptation or run length also changed.

Full precision and individual chain results are archived in
`validation/scs_100d_comparison.json`, with environment/source hashes in
`validation/scs_100d_protocol.json`. Every method contributed exactly
25,000,000 production iterations to that historical comparison.

Historical implementation checks also remain available:

- Three 25,000-iteration repeats with the same continuous adaptation found
  compiled DCS-RWM 24.00–24.70× and DCS-BW 25.60–25.90× faster than the NumPy
  reference on macOS ARM64. These are implementation speedups, with no promised
  Ryzen speedup. See `validation/dcs_compilation_benchmark.json` and
  `scripts/benchmark_dcs_compilation.py`.
- Three alternating 50,000-iteration repeats gave median 0.41 seconds for the
  fixed-count SCS adapter and 0.41 seconds for a direct author scan, including
  the same all-iteration radial reporting. The unrounded timing difference was
  0.26%. See `validation/scs_adapter_timing.json` and
  `scripts/benchmark_scs_adapter.py`. This checks the older fixed-count adapter;
  it is not a measurement of the current 300-transition timed batches.
- An earlier constant-caching check improved BW by 1.30× and SCS by 1.29×.
  That change alone cannot explain a large change in their speed ratio.
  Proposal scale, rejection frequency, fitted geometry, implementation and
  timing boundaries all affect transition cost.
