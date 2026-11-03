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

Each variant tunes its step separately toward acceptance 0.234 using the same
dual-averaging rule, with an explicit common upper step limit of 2 in this
new experiment. Two is the unit ball's diameter; this numerical bound is also
applied to SCS, without claiming its step has the same geometric meaning.
The standard main comparison retains its existing settings.
This bound was selected after an initial pilot sent reflected BW to the old
numerical ceiling exp(15), with acceptance around 0.8. The pilot is retained;
subsequent checks use fresh sampling seeds. The diameter choice is an explicit
experimental constraint, not a claim that larger reflected steps are redundant
or that 2 is optimal.
Thus the comparison includes the effect of boundary
handling on the calibrated scale; it is not a fixed-identical-step experiment.
The new kernel need not attain 0.234: for an exactly uniform transformed target,
every valid reflected proposal is accepted at every scale. Scale-ceiling
saturation is recorded and flagged in the report, rather than described as
successful calibration. Interpret the fitted steps and observed acceptances
together with MSE and time.

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

Windows: download the updated ZIP, extract it, open **Start.bat**, and select
**100D boundary comparison**. Use a fresh result folder for this experiment.

```bash
python run_experiment.py skew-t-100d-boundary --preset paper --workers 32 --output-dir results/boundary-paper
python run_experiment.py --plot-only results/boundary-paper
```

The saved protocol is `timed-boundary-dcs-scs-v1`. Original figure families are
retained in the new result folder, with the reflected variant clearly labelled.
This is an exploratory comparison, not a claim of dominance or a convergence
theorem for the new reflected kernel.

## Short validation runs, 3 November 2026

After the initial uncapped pilot, two fresh checks used 20 chains per method,
1.00 sampling second per chain, one worker, 1,500 separate calibration steps,
and 1,500 VI updates with 2,000 draws per update. Fit/chain seeds were
42/903001 and 43/904001. The common step ceiling was 2 in both checks.
All three methods used the same budgets within each check.

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
step ceiling. The common ceiling also applies during ordinary BW and SCS
calibration; a final step below 2 does not establish that it was inactive earlier.
Compare methods within each new run, rather than attributing differences from
older runs solely to reflection.

The [validation archive](../validation/boundary_comparison_2026-11-03.json)
includes the initial exp(15)-ceiling pilot as well as both subsequent checks,
with full-precision values, fitted maps, per-chain records, source hashes and
environment details. The cap was selected after that pilot; none of these
results is presented as a preregistered confirmatory comparison.
