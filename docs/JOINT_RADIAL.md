# Joint calibration of R and proposal step

This separate audit tests whether choosing the map scale R for radial sampling
efficiency changes the DCS-BW/SCS comparison. Both methods receive independent
proposal pilots with the same time budgets and the same number of candidates.
Every production step is pretuned and then frozen. The original experiments and
the earlier fixed-step, symmetric and adaptive equal-time audits remain available.

## Run with clicks

Open **Start.bat** on Windows or **Start.command** on macOS. After environment
setup, select the joint R-and-step audit, choose **Quick check**, and start a run.
Use **Full paper settings** for the scientific comparison. The launcher identifier
is `joint-r-step-audit`; it runs both targets and all 16 comparison cases.
Choose CPU workers for the allocation available on your computer.

The equivalent terminal command is:

```bash
python scripts/audit_joint_radial.py --preset paper --workers 8 --output-dir results/joint-r-step-paper
```

Use `--preset quick` or `--preset reproduce` for smaller runs and a fresh output
directory for each run. Quick settings check execution; their short pilot and
production budgets do not support scientific rankings.

## Targets and map parameters

Both targets have dimension 100, and DCS uses radial exponent β=2:

| Target | Degrees of freedom | Skewness vector |
|---|---:|---|
| Symmetric Student-t | ν=3 | Zero |
| Skewed Student-t | ν=2 | (100, −100, 0, …) |

The symmetric target directly revisits the earlier symmetric comparison. The
skewed target revisits the latest equal-time comparison. Radial truths are known
analytically; the skew factor leaves the distribution of the norm unchanged.

Each method receives a fresh VI map fit. On the symmetric target, centre and
deformation are exactly zero. For the skewed target, the method's fitted centre
and deformation are held fixed while R and the proposal step are searched.
Consequently, this is calibration **conditional on those fitted parameters**;
it does not refit a centre or deformation at every R. SCS retains its cap
latitude. No pilot or production outcome changes the other map parameters.

There are eight R candidates per method:

```text
0.5, 1, sqrt(nu), 2, 4, sqrt(101), 20, the method's VI-fitted R
```

The seven specified values are shared by the methods, while the eighth records
each method's own VI fit. If that fit duplicates a specified value, the
deterministic replacement is R=40. The requested R=√(d+1) is therefore tested
explicitly, with values on both sides.

At every R, the eight proposal candidates are:

| Method | Proposal steps |
|---|---|
| DCS-BW | 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.45 |
| SCS | 0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 1, 4 |

These step parameters have different geometric meanings. DCS-BW uses the radius
of a uniform ball proposal; SCS uses the per-coordinate tangent Gaussian scale
before normalization and cap step-out. Equal numerical values are not an equal
movement constraint. A winner at a grid endpoint is only the best tested value,
not an established optimum; the saved selection diagnostics identify endpoints.

## Equal pilot resources and selection

Independent exact target draws initialize the pilots, shared across methods and
candidates. This is an oracle-assisted synthetic diagnostic: it isolates local
sampling efficiency and does not measure convergence from a poor initial state.
The transition random streams and held-out production streams use disjoint seed
blocks. Pilot waves contain a single method at a time; method order alternates
and reverses every pair of waves to reduce timing drift. Both methods receive
the same worker allocation, including the last partial wave.

| Preset | Pilot chains per (R, step) | Discarded steps per chain | Timed pilot sampling per chain |
|---|---:|---:|---:|
| Quick check | 2 | 50 | 0.1 s |
| Smaller reproduction | 4 | 1,000 | 0.5 s |
| Full paper settings | 4 | 2,000 | 0.5 s |

Each method has 8×8 candidates on each target. Thus full settings allocate 256
pilot chains and 128 seconds of requested aggregate pilot sampling to each
method on each target, plus the same discarded-step budget. Methods can perform
different numbers of iterations during that time. Actual times and deadline
overshoots are recorded; map fitting and discarded sampling have separate
costs. Equal sampling time does not imply exactly equal total preparation time.

Smaller reproduction retains the full pilot sampling duration because shorter
pilots can miss all tail visits at large R. It remains smaller through fewer VI
updates, fewer discarded steps, fewer production chains and shorter production.

The selection score is the smaller of the rank-based bulk ESS values for
**I(||X||≥50)** and **log(1+||X||)**, per second. More precisely, each functional's
ESS is computed separately for each pilot chain, then summed across chains;
the smaller functional total is divided by summed actual pilot sampling time.
Different chain lengths are kept, not truncated to the slowest method's count.
Invalid diagnostics do not create an artificial winning score.

This criterion is a finite radial mixing proxy. It does not directly minimize
the error of the raw mean-norm estimator. In particular, ν=2 has a finite mean
norm but infinite norm variance. It cannot support a finite population MSE
objective for the raw sample mean. Tail-probability errors remain well defined.

For each method and target, the audit records the best step at every R, then
selects four controls:

1. **Joint:** the best (R, step) pair over all candidates.
2. **R=1:** the best step restricted to R=1.
3. **VI R:** the best step restricted to the VI-derived R candidate.
4. **R=√(d+1):** the best step restricted to R=√101.

The profile of best score versus R shows the effect of selecting R after tuning
steps at each candidate. Neither low boundary rejection nor high acceptance is
used as the optimization objective. Controls can select the same parameters;
shared choices do not constitute independent evidence for a tuning method.

## Held-out equal-time production

The two targets, two methods and four controls give 16 cases. Production uses
fresh random streams and freezes each selected map and proposal step before
sampling. The separate adaptive equal-time audit remains available for continued
step adaptation; this audit isolates joint calibration.

| Preset | Chains per case | Discarded steps | Sampling per chain | Trace thinning |
|---|---:|---:|---:|---:|
| Quick check | 2 | 100 | 1 s | 1 |
| Smaller reproduction | 8 | 1,000 | 5 s | 10 |
| Full paper settings | 20 | 1,500 | 20 s | 50 |

Chains run in synchronized waves containing one case at a time, with the same
worker allocation for both methods. Barriers before and after timed sampling
keep another worker's result-file writes outside the wave's sampling interval.
Fitting, pilots, production warmup and disk writes are separate from the
production time budget. Per-chain elapsed times, iterations, retained values
and estimation errors expose the realized allocation.

Both requested estimates use **every production iteration, including repeated
states after rejection**. Thinning controls stored traces only. Compare the
tail estimate and mean-norm estimate using the same production time; counts show
the throughput each method actually achieved.

Tail-error comparisons have paired bootstrap intervals across chain replicates.
These intervals are conditional on the single map fit and pilot selection; they
do not include uncertainty from repeating the calibration study and have no
multiple-comparison correction. On ν=2, realized norm errors and their squared
averages are descriptive finite-run values, not estimates with a finite-MSE
guarantee. Inspect exact per-chain estimates as well as any aggregate ranking.

## What a boundary crossing means

DCS-BW proposes a local Euclidean displacement in the unit ball. A proposed
point outside that ball receives zero target density and is rejected. SCS also
has a boundary: its valid spherical cap. Its initial tangent proposal can cross
that boundary, but the existing cap step-out operation moves the proposal along
its great circle before the Metropolis decision. An initial SCS cap crossing is
therefore not necessarily a rejected transition.

After calibration, separate probes count initial crossings, SCS step-outs,
final infeasible proposals, finite-density Metropolis rejections and accepted
transitions. They use the canonical kernels and exact target starts. They are
separate from production timings, so detailed rejection instrumentation does
not determine the equal-time ranking.

| Preset | Probe chains per case | Steps per chain |
|---|---:|---:|
| Quick check | 2 | 100 |
| Smaller reproduction | 2 | 2,000 |
| Full paper settings | 4 | 10,000 |

For a symmetric Student-t with zero centre/deformation and β=2, the transformed
ball density at radius r is proportional to

```text
(1-r^2)^(nu/2-1) * [nu + (R^2-nu)*r^2]^(-(nu+d)/2).
```

When ν=2 and R=√2 this is exactly uniform. Even a uniform high-dimensional ball
places much of its volume near the boundary. Larger R moves typical points
inward but changes the transformed density, so fewer boundary crossings can be
offset by more density rejections or poorer radial movement. This explains why
R=√(d+1) is a useful test, rather than a guaranteed improvement. The symmetric
arm here has ν=3; the uniform-ball statement is an explanatory special case.

The saved results include all candidate scores, selected settings, exact values,
chain counts, timing records and the separate rejection probes. Preserve the
protocol and source fingerprints with any reported figures. A result favouring
either method applies to these targets, grids, pilot resources and recorded
hardware; it does not establish a universal sampler ranking.

## Completed full comparison

The saved [comparison](../validation/joint_radial_comparison.json), [protocol](../validation/joint_radial_protocol.json) and [verification](../validation/joint_radial_checks.json) retain all pilot cells and final outcomes. The run used 1,024 independent pilot chains, 320 production chains (20 × 20 seconds per case), and 640,000 separate boundary-probe transitions. All 363039077 completed production iterations contributed to the estimates.

| Target | Method/control | R | Step | Mean N | Tail MSE | Observed mean-norm squared error |
|---|---|---:|---:|---:|---:|---:|
| symmetric-nu3 | DCS-BW: joint R + step | 1.0 | 0.1 | 1172902.35 | 8.425288046700396e-08 | 0.0006006323205349393 |
| symmetric-nu3 | DCS-BW: R = 1 | 1.0 | 0.1 | 1179967.7 | 8.502373740310524e-08 | 0.0005298973173035611 |
| symmetric-nu3 | DCS-BW: VI R | 1.4185464422342862 | 0.15 | 1103595.7 | 1.1252249531671336e-07 | 0.000921723895366253 |
| symmetric-nu3 | DCS-BW: R = sqrt(d+1) | 10.04987562112089 | 0.2 | 1271978.25 | 9.100883975864893e-07 | 0.07375504343594545 |
| symmetric-nu3 | SCS: joint R + step | 1.0 | 4.0 | 1095105.65 | 1.2590621786560944e-08 | 0.0001722048651947513 |
| symmetric-nu3 | SCS: R = 1 | 1.0 | 4.0 | 1111681.85 | 1.1524455127599926e-08 | 0.00018726413205847557 |
| symmetric-nu3 | SCS: VI R | 1.3608038983286952 | 4.0 | 1052165.55 | 1.6254377546192658e-08 | 0.000205526489849922 |
| symmetric-nu3 | SCS: R = sqrt(d+1) | 10.04987562112089 | 0.025 | 1369924.3 | 1.464409004334586e-06 | 0.08463479625315314 |
| skewed-nu2 | DCS-BW: joint R + step | 1.4142135623730951 | 0.15 | 1078426.75 | 3.925392500422927e-07 | 0.020829406574303298 |
| skewed-nu2 | DCS-BW: R = 1 | 1.0 | 0.075 | 933122.65 | 2.9577717898350933e-07 | 0.013377492073139177 |
| skewed-nu2 | DCS-BW: VI R | 1.3568255026484344 | 0.15 | 1084081.7 | 3.109849560953634e-07 | 0.021449347279292985 |
| skewed-nu2 | DCS-BW: R = sqrt(d+1) | 10.04987562112089 | 0.15 | 896803.5 | 2.5447105093874706e-05 | 0.8529517491645666 |
| skewed-nu2 | SCS: joint R + step | 0.5 | 0.05 | 1118021.45 | 1.698456480820526e-07 | 0.008093914180992135 |
| skewed-nu2 | SCS: R = 1 | 1.0 | 0.1 | 1186799.15 | 2.507098378009417e-07 | 0.0039297754603440545 |
| skewed-nu2 | SCS: VI R | 1.3368320733432195 | 0.1 | 1169889.25 | 3.5352581273609997e-07 | 0.004570223415019771 |
| skewed-nu2 | SCS: R = sqrt(d+1) | 10.04987562112089 | 0.05 | 1327488.05 | 3.420298082468003e-05 | 2.664734654533435 |

Joint selection retained R=1 for both methods on the symmetric target. For the skewed target it selected DCS R=√2 and SCS R=0.5. Both methods received independent step selection at every tested R. The symmetric SCS step=4 and skewed SCS R=0.5 are grid endpoints; neither is claimed to be globally optimal.

For symmetric-nu3, the joint-control tail-MSE difference DCS−SCS is 7.1662258680443e-08, with descriptive paired 95% interval [3.082906623031077e-08, 1.2627575388836468e-07]. This conditional comparison favours SCS.

For skewed-nu2, the joint-control tail-MSE difference DCS−SCS is 2.2269360196024007e-07, with descriptive paired 95% interval [-4.1026405101164734e-09, 4.789714209710886e-07]. The interval includes zero, so the observed difference remains uncertain.

The joint pilot rule did not improve every held-out error. In the skewed DCS run, R=1 and the VI-R control had lower observed tail error than the joint pilot winner. This is a finite pilot proxy with selection noise, not an oracle for production MSE.

R=√101 performed poorly for both methods. In the symmetric DCS probes it reduced outside-ball rejection from 19935/40000 (49.8375%) to 582/40000 (1.455%), but density rejection increased from 5652/40000 (14.13%) to 33440/40000 (83.6%). Symmetric SCS had 6402/40000 initial cap crossings, zero final infeasible proposals, and 5807 accepted transitions after step-out. These categories explain why the first boundary crossing cannot be treated as the same event in the two methods.

At ν=2, the mean-norm squared-error column is descriptive because the target norm has infinite variance. These results do not establish a universal ranking, and equal-time counts depend on the recorded hardware. All prior audit results remain separate and unchanged.

The full run used [source revision e0164cf](https://github.com/MihaBresar/DiffeomorphicMonteCarlo/commit/e0164cfd6904ab8a7e9a34d580c5164f2c7d4856). Later runner changes increased the smaller preset’s pilot time from 0.15 to 0.5 seconds, increased the execution-only quick pilot from 0.02 to 0.1 seconds, and made a failed worker release barrier-waiting peers immediately while preserving the original error. Full-paper settings and sampling transitions are unchanged. The recorded source fingerprints identify the exact generating code.
