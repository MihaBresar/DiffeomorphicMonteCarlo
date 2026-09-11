# Symmetric Student-t: map fitting and proposal tuning

Every reported production step is **pretuned and then frozen**. An untuned
map does not mean an untuned proposal. This experiment disables map fitting
for both DCS and SCS in its baseline and compares them on exactly symmetric
100-dimensional Student-t with 3 degrees of freedom.

## What the experiment isolates

The baseline maps have R=1, zero centres, and zero Möbius/observer shifts.
DCS keeps beta=2 and SCS keeps cap latitude=1.1. Separate VI controls fit R
only; the existing symmetric-target constraints keep centres and shifts exactly
zero. Thus this experiment can measure the effect of R fitting, but cannot
by itself establish whether a nonzero Möbius deformation helps a skewed target.

Map fitting precedes proposal tuning in both the main engine and this audit.
Each map/method pair receives three proposal-tuning rules:

- Ordinary dual-averaging warmup, followed by its frozen averaged step.
- A separate pilot search maximizing the smallest bulk ESS across five radial
  and coordinate functions.
- A separate evaluation using a pilot choice focused on the paper's two radial
  functions: the tail indicator and mean norm.

Both pilot rules use the same eight candidates and four pilot chains per
candidate for each map/method pair: 2,000 discarded plus 8,000 retained steps.
All candidates, diagnostics and choices are saved. Fixed grid values are
calibration candidates, not untuned production comparisons. These rank-based
ESS criteria are efficiency proxies; neither guarantees minimal MSE.

Full settings evaluate all 12 cases with **20 chains, 500,000 production
iterations, 1,500 discarded warmup steps, and 10,000 retained values per chain**.
The pilot starts are shared exact Student-t draws, an explicitly oracle-assisted
synthetic diagnostic. Production starts at physical (1,...,1) for every method.
Radial references are analytic; asinh-coordinate means are exactly zero by
symmetry. Rejected iterations and repeated retained states count equally.

The radial-focused rule was motivated by an exploratory batch. The final
comparison uses fresh production seeds **5700001–5700020**, separate from that
batch, the pilots, and map fitting. Paired bootstrap intervals are descriptive,
conditional on fitted maps, and not adjusted for multiple comparisons. At nu=3
the norm has finite second moment but infinite fourth moment, so uncertainty in
20-replicate mean-norm MSE estimates can be substantial. Every case and chain
is retained.

## Run without commands

With Python 3.12 installed, download the latest repository ZIP, extract it,
and double-click **Start.bat** on Windows. Click **Set up environment** and
wait for **Environment ready**. Select **100D symmetric t: untuned maps and step tuning**. Run
**Quick check** first, then **Full paper settings**. The audit includes every
map and proposal-tuning control automatically. Use the CPU-workers selector
for parallel chains. The main seven experiments and their original plots are
unchanged.

The new plots and `paper_values.html` display exact MSEs, counts, map scales,
proposal steps, all pilot candidates and paired uncertainty. Full-precision
CSV files are saved alongside them.

The original radial plots use the 10,000 retained values per chain. A separate
`symmetric_t_all_iterations` plot uses all 500,000 production iterations,
including repeated states after rejection. The latter addresses efficiency per
generated iteration; thinning by 50 can hide differences when radial mixing is
already fast. Production ESS still refers to the saved thinned traces, while
the pilot ESS uses unthinned pilot traces. The two denominators are reported
explicitly.

Advanced equivalent:

```bash
python scripts/audit_symmetric_t.py --preset paper --workers 8 --output-dir results/symmetric-t-paper
```

Use a fresh output folder. The script saves source/environment hashes, fitted
maps, initial points, pilot and production job specifications, functional
traces, summaries, and PNG/PDF figures. Its mixed worker pool is for scientific
diagnostics; the recorded times are not throughput comparisons.

## Completed comparison

The saved [summary](../validation/symmetric_t_comparison.json) and
[protocol](../validation/symmetric_t_protocol.json) contain all 12 cases,
240 production chains, 128 pilot chains, and the exact settings. The following
MSEs use **every one of the 500,000 production iterations in each of 20 chains**,
with the radial-focused pilot criterion applied to both methods:

| Map | Method | Fitted R | Pretuned step | Tail-probability MSE | Mean-norm MSE |
| --- | --- | ---: | ---: | ---: | ---: |
| Fully untuned | DCS-BW | 1 | 0.075 | 1.0740152463951053e-7 | 0.0010203698311554796 |
| Fully untuned | SCS | 1 | 4 | 3.6170534901506315e-8 | 0.00023975006289360845 |
| VI scale only | DCS-BW | 1.4037165796455882 | 0.1 | 9.10285320763039e-8 | 0.001993012647892318 |
| VI scale only | SCS | 1.3559551758568449 | 4 | 2.8943651908901023e-8 | 0.00015848592477065847 |

The tuning objective matters. For VI-scale DCS, switching from the five-function
pilot criterion (step 0.3) to the radial criterion (step 0.1) reduces mean-norm
MSE from 0.007229100276137783 to 0.001993012647892318, a 3.63-fold improvement.
The paired MSE difference is -0.005236087628245466, with a descriptive 95%
bootstrap interval [-0.012055204338150499, -0.0007232284003617539]. Tail MSE
improves 6.23-fold, while coordinate MSE worsens. This establishes a tradeoff
between pilot objectives. It is not evidence of improvement over ordinary
dual-averaging warmup: that VI-scale DCS case has mean-norm MSE
0.0018702384363669423, similar to the radial-focused case.

SCS still performs better with both maps fully untuned and both proposal steps
pretuned. Its mean-norm MSE is 4.26 times smaller. The DCS-minus-SCS difference
is 0.0007806197682618713, with interval
[0.00016546161488396153, 0.001510896500634991]. The analogous tail-MSE
difference is 7.123098973800424e-8, with interval
[3.040090465687046e-9, 1.631518787043512e-7]. These are results for this target,
budget, initial state, and selected pilot choices, not a universal ranking.
The symmetric control does not establish a failure of Möbius tuning because
its Möbius parameters are zero throughout.

Both SCS pilot criteria selected 4, the largest tested candidate, for both maps.
This is the best tested value, not an established optimum. SCS uses per-coordinate
tangent standard deviation, so its numerical step is not directly comparable
to the DCS ball radius. In the separate ordinary-warmup controls, SCS reaches
the step ceiling exp(15)=3269017.3724721107 while acceptance remains above the
0.234 target; these are ceiling-limited, nearly orthogonal proposals, rather
than successful calibration to 0.234. The finite pilot-selected steps provide
a separate control. SCS's two pilot labels have identical selected steps and
production seeds, hence identical traces; they are not independent replications.

The original plots based on 10,000 retained values per chain remain available.
Their radial MSE comparisons are less decisive: thinning changes the estimator
and can obscure per-iteration differences. The separate all-iteration plot and
CSV files make this explicit. All 240 retained production traces and 128 pilot
traces were checked to be bitwise unchanged by the added streaming estimates;
all 322 paired intervals were independently recomputed.

## Why a better VI fit need not improve every MSE

The map fit minimizes reverse KL against the target. It does not optimize the
transition's autocorrelation or any particular estimator's asymptotic variance.
For the stored skewed-target maps, independent base-draw integration found a
large reduction in reverse KL for both fitted maps relative to the fully
untuned maps. This demonstrates improvement in that fitting objective, without
establishing a mixing improvement.

There is also measurable optimizer noise: projecting the fitted centre and
shift onto the synthetic target's known skew direction reduces the estimated
objective for both methods. This is a separately evaluated symmetry ablation,
not a generic posterior-map correction. Held-out production comparisons would
be needed before claiming improved MCMC efficiency from that projection.

The portable calculations and exact values are in
[map_geometry.json](../validation/map_geometry.json). They use 100,000 fresh
base draws per method with shared randomness across its map variants:

```bash
python scripts/check_map_geometry.py --output-file results/map_geometry.json
```

For the symmetric ball map, deterministic radial integration gives the
reverse-KL-optimal scalar R=1.4073944761355872. With R=1 the median transformed
radius is 0.9960531463648383, close to the ball boundary even though the target
is perfectly symmetric. The fitted scale can improve the density approximation
without removing the local proposal's boundary rejections or its small angular
moves.
