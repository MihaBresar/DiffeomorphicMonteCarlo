# Equal-time skewed Student-t comparison

This experiment uses d=100, Student-t degrees of freedom ν=2,
α=(100,−100,0,…), DCS radial exponent β=2, and SCS latitude 1.1.
The measured quantities are P(‖X‖≥50) and E[‖X‖], both with analytic references.
This is a new fixed-wall-time experiment; elapsed times from earlier mixed
diagnostic queues are not used to rescale their MSEs.

## Run with clicks

With Python 3.12 installed, download and extract the latest repository ZIP.
Double-click **Start.bat** on Windows, click **Set up environment**, and wait
for **Environment ready**. Select **100D skewed t: equal-time DCS/SCS** and run
**Quick check**, followed by **Full paper settings**. Choose the same CPU-worker
setting for comparisons. The eight cases are included automatically.

| Preset | Chains per case | Production time per chain | Discarded warmup |
| --- | ---: | ---: | ---: |
| Quick check | 2 | 1 second | 100 steps |
| Smaller reproduction | 8 | 5 seconds | 1,000 steps |
| Full paper settings | 20 | 20 seconds | 1,500 steps |

Advanced equivalent:

```bash
python scripts/audit_skew_t_timed.py --preset paper --workers 8 --output-dir results/skew-t-nu2-timed
```

Use a fresh output folder. Timing-based counts and estimates will vary by
computer and system load, even with identical random seeds.

## Map and proposal controls

The four map groups are Möbius-fitted DCS-BW, fitted SCS, fully untuned DCS-BW,
and fully untuned SCS. Untuned maps have R=1 and zero centre/deformation; fitted
maps optimize R, centre and deformation by the existing VI procedure. Maps
remain fixed throughout production.

Every group receives an independent proposal pilot, after map fitting. The
full pilot uses eight candidate steps, each with four chains of 2,000 discarded
and 8,000 retained steps. Starts are shared exact target draws, an explicitly
oracle-assisted synthetic diagnostic. All candidates are saved. The criterion
is the same five-function minimum bulk ESS per iteration used in the earlier
closer skewed comparison: log radius, a radial 95% indicator, asinh of the skew
contrast, and asinh of coordinates 3 and 4. It is an efficiency proxy, not a
proven optimizer of radial MSE per second. Endpoint choices are reported.

Each map group has two production policies:

- **Frozen:** use its pilot-selected step for discarded warmup and production.
- **Adaptive:** start at the pilot-selected step and use the historical DCS
  smoothed dual-averaging rule through warmup and production, for both DCS and
  SCS with the same 0.234 acceptance target. The next proposal uses the averaged
  log step. Initial/final steps, ranges and ceiling fractions are recorded.

This yields eight cases. Both policies start at the physical point (1,…,1),
but their warmup trajectories can differ. This evaluates the complete tuning
policies. Adaptive transitions depend on their history; fixed-kernel theory
does not automatically justify that empirical adaptive arm.

## What the clock and estimates include

Production runs in synchronized waves containing only one case. Every case
has the same worker allocation and chain counts; case order rotates across
waves according to a schedule saved before fitting. Workers complete warmup
before meeting the production barrier. Windows/Linux pin distinct CPU slots;
macOS uses its operating-system scheduler.

The production clock includes transitions, any adaptive updates, accumulation
of both radial estimates and collection of thinned diagnostic values. Fitting,
independent pilots, warmup, worker startup and output saving are outside that
clock and are reported separately. The deadline is checked before each step;
the last completed step can slightly overrun it. Actual wall time, CPU time,
overshoot and iteration count are saved for every chain.

Every completed iteration contributes to the reported estimates, including
repeated states after rejection. Counts are not forced to match, chains are
not truncated to equal counts, and errors are not multiplied by a timing ratio.
Each chain has equal weight in the across-chain average squared error. Wall-time
stopping makes the sample count random and potentially state-dependent: the
results describe these actual fixed-budget algorithm outputs.

At ν=2 the norm has a finite mean but infinite second moment. Individual norm
estimates, squared errors, their empirical mean and median absolute error are
reported descriptively; no finite-variance norm MSE interval is claimed. Only
the bounded tail-probability error receives descriptive paired bootstrap
intervals, conditional on the fitted maps and pilot choices, without adjustment
for multiple comparisons. Every chain remains in the results.

Original-style estimator boxplots, empirical error plots and sample-count plots
are accompanied by full-precision CSV tables and `paper_values.html`. The
historical and symmetric-target figures remain available separately.

## Completed full run

The saved [comparison](../validation/skew_t_timed_comparison.json) and
[protocol](../validation/skew_t_timed_protocol.json) record all eight cases,
160 production chains and 128 independent pilot chains. The local run used
eight workers, with four workers for each case's final partial wave. Every
case received 20 chains with a 20-second production budget per chain.
All 196,705,462 completed production iterations contributed to the estimates.
The largest deadline overshoot was 0.0004449160005606245 seconds; mean
per-chain CPU/wall-time ratio was 0.9921322620854811.

Exact references are P(‖X‖≥50)=0.03919519628762017 and
E[‖X‖]=17.68028324242731. Mean N below is the arithmetic mean of 20 integer
iteration counts, including rejection repeats. The raw-norm error column is
the descriptive empirical average squared error described above.

| Map/method | Production policy | Mean N per chain | Tail MSE | Empirical norm squared error |
| --- | --- | ---: | ---: | ---: |
| Möbius DCS-BW | Frozen | 2060198.35 | 6.066169366942345e-7 | 0.015557868505980338 |
| Möbius DCS-BW | Adaptive | 982364.15 | 5.635140483243241e-7 | 0.026477696827442266 |
| Fitted SCS | Frozen | 1218266.25 | 3.0932173600717105e-7 | 0.0051514547152630595 |
| Fitted SCS | Adaptive | 910304.15 | 5.495328347800383e-7 | 0.009948500309416209 |
| Untuned DCS-BW | Frozen | 1714887.65 | 4.4004565731482557e-7 | 0.02265089673945723 |
| Untuned DCS-BW | Adaptive | 1004287.45 | 6.818313423437658e-7 | 0.018279211017069436 |
| Untuned SCS | Frozen | 1094958.6 | 1.9868222730744465e-7 | 0.009154418070592621 |
| Untuned SCS | Adaptive | 850006.5 | 2.768056472443097e-7 | 0.0027300156180938557 |

With fitted maps and frozen steps, DCS generated 1.6910903917760178 times as
many iterations as SCS. Its observed errors were larger. The tail-MSE
difference DCS−SCS was 2.9729520068706336e-7, with descriptive paired 95%
interval [-6.841341279156332e-8, 7.312295244608565e-7]. Thus this fitted-map
tail comparison does not establish a resolved advantage for either method.

With continued adaptation, fitted DCS and SCS had closely matched tail errors:
difference 1.3981213544285768e-8, interval
[-4.2335080204621585e-7, 3.5344328368788606e-7]. For each map group, the
adaptive-minus-frozen tail-error interval also included zero. The adaptive
policies generated fewer iterations and did not establish a tail-accuracy
improvement. Differences in throughput include both adaptation calculations
and the changed proposals/acceptance rates, not just update overhead.

The independent pilots selected DCS step 0.3 with a fitted map and 0.2 untuned;
SCS selected 0.1 fitted and 0.4 untuned. All choices were valid and strictly
inside the declared grids. The untuned adaptive SCS case reached the numerical
step ceiling exp(15) throughout production, with mean acceptance about 0.2935,
above the 0.234 target. It is a ceiling-limited adaptive control, not evidence
of optimal proposal calibration. The fitted adaptive cases did not hit the
ceiling and had acceptance close to 0.234.

These results describe this target, one set of fitted maps/pilots and this
hardware allocation. Changing ν, the budget, the step policy and the estimator
denominator prevents treating them as a literal replication of an earlier
ν=3 fixed-iteration figure.

## Super-heavy tails use equal iterations

The separate **50D t: infinite first moment, equal samples** experiment now runs the same
500,000 production iterations for every method at full settings (50 chains
per method). Its original figure filenames remain. Its norm-related estimand
is E[log10(1+‖X‖)], which is finite; E[‖X‖] does not exist there. Run it afresh
to regenerate figures under the equal-count design. See [the rerun guide](RERUN.md).
