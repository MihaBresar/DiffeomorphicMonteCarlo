# Boundary relocation comparison

Select **100D boundary comparison** in the launcher. This separate experiment
compares **DCS-BW**, **DCS-BW-reflect** and the **unmodified author SCS** on the
same 100D normal-CDF-skewed Student-t target with ν=3. The main experiment and
All seven sequence remain unchanged.

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

| Preset | Chains per method | Sampling seconds per chain | Separate warmup | VI updates × draws per map |
|---|---:|---:|---:|---:|
| Quick check | 2 | 0.05 | 100 | 20 × 128 |
| Smaller reproduction | 20 | 2.00 | 1,000 | 400 × 500 |
| Full paper settings | 50 | 20.00 | 1,500 | 1,500 × 2,000 |

## Current calibration: protocol v2

Each variant tunes its step separately toward acceptance 0.234 using the same
dual-averaging rule and warmup budget, then freezes the scale. **Ordinary DCS-BW
and author SCS retain their original calibration**, including the numerical
log-scale ceiling of 15, equivalent to a step ceiling of exp(15). **Only
DCS-BW-reflect has the additional step ceiling of 2.** The main 100D comparison
is unchanged.

This is an explicit method-specific constraint. Reflection can make 0.234
unattainable: for an exactly uniform transformed target, every valid reflected
proposal is accepted at every scale. An initial pilot sent reflected BW to
exp(15), with acceptance around 0.8. The bound of 2 was chosen after that pilot;
it is the unit ball's diameter, not a claim that larger reflected steps are
redundant or that 2 is optimal. Saturation is recorded and flagged rather than
described as successful calibration. Interpret the fitted steps and observed
acceptances together with MSE and time.

The historical v1 checks below imposed the ceiling of 2 on **all** methods.
That can change the warmup path of ordinary BW and SCS even when their final
steps lie below 2. Protocol v2 removes that additional constraint from both
baselines. It preserves their original calibration while studying the new
reflected kernel under its stated bound. This is not a fixed-identical-step
comparison; both boundary handling and the reflected method's calibration
constraint can affect its result. Rerun all three methods together to evaluate
v2; the historical v1 table is not a result for the current protocol.

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

On Windows, without commands:

1. Download the latest GitHub ZIP and extract it into a new folder.
2. Open **Start.bat** in that folder and click **Set up environment** if needed.
3. Select **100D boundary comparison** and **Full paper settings**.
4. Choose **32 CPU workers** on the Ryzen 9 9950X3D with SMT enabled, or click
   **Use all CPU threads**. This permits up to 32 simultaneous chains; the
   preset still runs 50 chains per method.
5. Select **Original seeds (batch 0)**, or **Fresh batch 1** through **Fresh batch 5**
   for a new set of sampling and warmup streams, then click **Start run**.

Each launch creates a fresh dated result folder. Open `paper_values.html`
under **Exact numbers and tables** after it finishes. Use the same worker
setting when comparing runs.

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

The saved protocol is `timed-boundary-dcs-scs-v2`. Original figure families are
retained in the new result folder, with the reflected variant clearly labelled.
This is an exploratory comparison, not a claim of dominance or a convergence
theorem for the new reflected kernel.

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
historical v1 results with a common cap of 2, not current v2 results.** They
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
for the v2 protocol.
