# Historical boundary relocation comparison

**This page preserves the earlier separate boundary experiment.** It is no
longer a launcher choice. Use the current [configurable comparisons](COMPARISONS.md):
four samplers in JAX (DCS-RWM, DCS-BW, SCS and DCS-BW-reflect), or optimized
Numba BW/reflected BW versus unchanged author SCS. Both offer 100D and 500D skewed t targets, each with VI on or off for both
map families. All seven includes the four-method JAX suite on 100D skewed t
with the selected VI policy; the optimized and additional-target runs are
separate choices.

The historical `skew-t-100d-boundary` CLI identifier and
`scripts/reproduce_boundary_100d.py` remain available, as does
`run_experiment.py --plot-only SAVED_RUN` for archived results. The description
below concerns that historical protocol and its revisions. It compares
**DCS-BW**, **DCS-BW-reflect** and the **unmodified author SCS** on the same
100D normal-CDF-skewed Student-t target with ν=3. References below to an
unchanged main comparison describe the earlier release.

Both BW variants share one fitted ball map. They draw the same uniform-ball
increments at a given proposal scale. Standard BW rejects outside proposals;
the new variant reflects their travel along the proposal's chord, then applies
the usual transformed-density Metropolis test. SCS keeps its spherical cap
step-out. Ball chord reflection is a new reversible kernel, not the literal
spherical step-out or specular billiard reflection.

All methods start at the physical vector of ones. Map-fitting resources,
separate step-calibration budgets, CPU allocation, 300-transition timing
batches and full-state radial recording match the main comparison. Scales are
frozen during sampling. Every run discards the first third of its actual
generated iterations; every remaining state, including rejection repeats,
contributes to tail-probability and mean-norm estimates. Only stored traces are
thinned. Counts and full-precision MSEs are exported; figures use two decimals.

Execution revision `matched-wave-cpu-v2` gives each method the same CPU set
even in a partial final wave, using a smaller pool initialized before JAX.
For 50 chains on 32 workers, the final 18 use the same first 18 CPU slots.
Assignments are recorded and checked on Windows/Linux; macOS records
OS-managed allocation. Earlier revisions could use different subsets in a
partial wave. Rerun all three methods together for corrected timings.

Both estimates are radial and have the same truth for the symmetric and skewed
t target. They assess radial estimation, not exploration of the skewed
direction or overall mixing. Equal VI and step-calibration budgets also do
not establish equally good fits or optimal tuning; the main protocol explains
[these limits](SCS_100D.md#matched-map-fitting-and-separate-step-calibration).

| Preset | Chains per method | Sampling seconds per chain | Separate warmup | VI updates × draws per map |
|---|---:|---:|---:|---:|
| Quick check | 2 | 0.05 | 100 | 20 × 128 |
| Smaller reproduction | 20 | 2.00 | 1,000 | 400 × 500 |
| Full paper settings | 50 | 20.00 | 1,500 | 1,500 × 2,000 |

## Current calibration: protocol v3

**Reflected BW now selects its step using observed squared movement inside the
ball, with no special upper bound of 2.** Ordinary BW and author SCS retain
their original separate acceptance-target 0.234 calibration. All methods use
the same total warmup-transition budget and freeze their scales before timed
sampling. The main 100D comparison is unchanged.

For reflected BW, a pilot window with actual post-Metropolis states Z has score

    J(h) = mean ||Z_next - Z_current||².

Rejected moves contribute zero. The score is at most 4 inside the unit ball
and remains meaningful at 100% acceptance. It measures movement per
transition in ball coordinates; it does not use the paper's radial truth,
production errors or MSE. Expected squared jumping distance is an established
tuning objective; this finite local search is our implementation, not a claim
to reproduce a published optimizer. See
[Pasarica and Gelman (2010)](https://www3.stat.sinica.edu.tw/sstest/j20n1/j20n113/j20n113.html).

The complete rule is fixed before sampling:

1. Start at h=0.10, as in the other methods.
2. Compare h/2, h and 2h in each of eight rounds. Split the existing warmup
   budget across the 24 trial windows, assigning remainder transitions in
   chronological order. For custom budgets below 24, reduce the number of
   rounds; at least three transitions are required.
3. Run a continuous pilot path and rotate candidate order each round. Include
   the first move of every window in its score. Select the largest observed
   score, with exact ties favoring the smaller step, and center the next
   triplet at that step.
4. Retain the final pilot state, and freeze the last round's selected step.
   The normal first-third sampling burn-in remains additional to this pilot.

Steps above 2 are permitted. Only the shared numerical safeguards
exp(−30) and exp(15) remain; they protect finite-precision calculations and are
not claimed to be optimal search limits. The reflected proposal and its
Metropolis correction are unchanged. Increasing the raw step does not make
the final displacement unbounded: it can cause repeated chord reflections.

Every candidate's step, transition count, observed acceptance, jump score and
selection flag is saved in `reflected_tuning.csv` and chain metadata. The HTML
report shows selected steps and scores. An edge winner in the final round is
reported as such, not certified as an interior optimum. Scores from short,
possibly transient windows are noisy; reflection can also make the objective
nonmonotone. This is a local finite-budget search, not a guarantee of optimal
mixing or improved radial MSE. Warmup acceptance is descriptive, not a target
for reflected BW.

Both reflected BW and SCS accept every valid proposal for an exactly uniform
transformed target, so either can fail to attain acceptance 0.234. This change
addresses that issue for the reflected variant requested here; the unchanged
SCS calibration is not claimed immune. Matching total tuning resources does
not make their tuning criteria identical. Compare the specified procedures
and disclose the difference.

In a calibration-only check with the full 1,500 × 2,000 fit (seed 42), 1,500
warmup transitions and all 50 default chain seeds 901001–901050, the selected
steps were 0.20 in 7 chains, 0.40 in 42 chains and 0.80 in 1 chain. All were
below 2 without an enforced cutoff. Six final-round winners were the smaller
candidate and six the larger candidate; this remains a finite local search.
These checks do not establish an optimal scale or a new MSE ranking. See the
[v3 validation record](../validation/boundary_v3_checks.json).

## Historical calibration: v1 and v2

The historical v1 checks below imposed step ceiling 2 on all methods. Protocol
v2 restored ordinary BW/SCS calibration and retained that extra ceiling only
for reflected BW, following a pilot with high acceptance at exp(15). Those
were exploratory settings, not evidence that 2 was optimal or necessary for
validity. Their archives and report labels remain unchanged. Rerun all three
methods together for v3; neither earlier protocol can be updated by replotting.

## Why the reflection proposal is valid

Write a displacement as r u with r≥0 and uniform direction u. The proposal
line through x intersects the unit ball in a chord. On that line, write
x=q+t u, q perpendicular to u, and a=sqrt(1−‖q‖²). The chord coordinate lies
in [−a,a]. Set L=2a and reflect t+r into the chord with

    t_new = L − abs(((t + a + r) mod (2L)) − L) − a.

The proposed point is q+t_new u. Inside proposals are returned unchanged.
The formula handles any number of chord-endpoint reflections without a loop.

For a fixed unoriented line direction, the two equally likely orientations
produce the symmetric signed-step density
g(s)=d|s|^(d−1)/(2h^d) for |s|≤h, where h is the BW step radius. On interval
coordinates z=t+a and z'=t_new+a, the reflected transition density is

    Q(z,z') = sum over integers k of
              [g(z'−z+2kL) + g(z'+z+2kL)].

Since g is even, Q(z,z')=Q(z',z). Along a fixed direction the transverse
coordinate q is unchanged and volume decomposes as dq dt. Averaging these
reversible line kernels over directions therefore gives a proposal reversible
with respect to uniform volume on the ball. Multiplication by the usual
Metropolis acceptance ratio of transformed target densities preserves that
target. No additional proposal-density ratio is needed.

This argument is in exact arithmetic. Numerical endpoints that are nonfinite
or fail the open-ball check become rejected self-loops; they are never clipped
or normalized onto the boundary. Tests check interval symmetry, multiple
reflections, uniform-ball radial and coordinate distributions, large steps,
and finite behavior near the boundary.

## Running and reproducing

The former **100D boundary comparison** launcher entry has been replaced.
For runs without commands, follow [the current launcher guide](CLICK_TO_RUN.md)
and select the four-sampler JAX or optimized Numba comparison. Both include
reflected BW and offer the target and VI selectors. To run this historical
three-method boundary protocol, use the command-line entry point below with
a fresh output folder. Keep the same worker setting when comparing runs.

The default `seed_batch=0` uses sampling seed 901001. Batch b adds
1,000,000 × b to that seed; map-fitting seed 42 remains fixed. The launcher
offers batches 0–5, and the command line accepts 0–999. Repeating a batch
reuses its sampling and warmup streams. Changing batches provides independent
sampling and warmup replications conditional on the same fitted maps; it does
not test variability over new map fits. Within a batch, methods continue to
use matched chain seed IDs. Timed runs can generate different iteration counts
on repetition, changing both the endpoint and the first-third burn-in cutoff.
Consequently the same batch does not promise identical MSE or counts.

```bash
python run_experiment.py skew-t-100d-boundary --preset paper --workers 32 --seed-batch 0 --output-dir results/boundary-paper
python run_experiment.py --plot-only results/boundary-paper
# Fresh sampling and warmup streams with the same fitted maps:
python run_experiment.py skew-t-100d-boundary --preset paper --workers 32 --seed-batch 1 --output-dir results/boundary-paper-batch-1
```

`--seed-batch` and a custom `--seed` are mutually exclusive.

The saved protocol is `timed-boundary-dcs-scs-v3`. Original figure families are
retained in the new result folder, with the reflected variant clearly labelled.
This is an exploratory comparison, not a claim of dominance or a convergence
theorem for the new reflected kernel.

The [preceding v2 comparison audit](../validation/current_comparison_fairness_audit.json)
records independent transformation and boundary checks, the CPU-allocation
correction, author-code provenance and inference limitations. Its description
of the reflected-only ceiling applies to historical v2, not current v3.

The tables export empirical MSE uncertainty summaries computed directly from
the per-chain squared errors, alongside MSE, exact counts and time. These are
inexpensive descriptive checks, not formal 95% confidence intervals. In
particular, the t(3) norm has an infinite fourth moment, so a few large errors
can dominate its MSE and the usual variance-based MSE uncertainty need not be
well defined. Fresh batches and longer runs help assess stability; a single
observed ordering does not establish a ranking.

## Historical v1 validation runs, 3 November 2026

After the initial exp(15)-ceiling pilot, two fresh checks used 20 chains per method,
1.00 sampling second per chain, one worker, 1,500 separate calibration steps,
and 1,500 VI updates with 2,000 draws per update. Fit/chain seeds were
42/903001 and 43/904001. The common step ceiling was 2 in both checks.
All three methods used the same budgets within each check. **These are
historical v1 results with a common cap of 2, not current v3 results.** They
also predate the seed-batch selector; their custom seed values are retained.

| Check | Method | Mean generated iterations | Tail-probability MSE | Mean-norm MSE | Acceptance |
|---|---|---:|---:|---:|---:|
| A | DCS-BW | 159,300 | 1.37e-06 | 1.15e-02 | 23.08% |
| A | DCS-BW-reflect | 118,905 | 2.60e-07 | 1.05e-02 | 80.84% |
| A | SCS | 118,110 | 5.90e-07 | 1.53e-02 | 17.96% |
| B | DCS-BW | 160,065 | 1.33e-06 | 3.05e-02 | 22.29% |
| B | DCS-BW-reflect | 119,505 | 1.01e-07 | 3.89e-03 | 79.59% |
| B | SCS | 118,500 | 2.50e-06 | 1.87e-02 | 16.87% |

Reflected BW had lower observed MSE for both functionals in both checks, at
approximately SCS's iteration throughput. Ordinary BW generated more iterations.
These short checks are conditional on two fitted map pairs; they do not establish
a stable ranking or universal advantage. All 40 reflected chains reached the
step ceiling. The common ceiling also applied during ordinary BW and SCS
calibration; a final step below 2 does not establish that it was inactive earlier.
Compare methods within each new run, rather than attributing differences from
older runs solely to reflection.

The historical [v1 validation archive](../validation/boundary_comparison_2026-11-03.json)
includes the initial exp(15)-ceiling pilot as well as both subsequent checks,
with full-precision values, fitted maps, per-chain records, source hashes and
environment details. The cap was selected after that pilot; none of these
results is presented as a preregistered confirmatory comparison or evidence
for the v2 or v3 protocols.
