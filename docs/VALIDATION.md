# Validation

## Current version-11 scope

Version 11 uses one shared DCS move: fixed BW/reflected-BW/billiard increment
radius h=0.07 and Gaussian RWM coordinate standard deviation h/√(d+2).
All four keep these steps during discarded equilibration. SCS retains h=0.1.
The declared h=0.15 sensitivity setting and historical mixed-policy control
do not imply optimality. The latter uses acceptance-target 0.234 for ordinary
RWM/BW, ball-movement pilots for reflected BW and radial-movement pilots for
billiards. Version-10 fixed-cloud VI budgets and
maps remain unchanged. Fresh sampling is needed to evaluate the new step
policy. Experimental fresh-cloud VI studies do not change this public default.

Local validation covered **386 regression checks**: the full suite plus a
successful targeted rerun after updating a historical calibration fixture.
Four Quick comparisons completed in JAX/Numba at 100D/500D, covering **40
production chains**. All four DCS variants kept the shared fixed move,
including during their own discarded warmup. Cross-backend fitted maps,
warmup endpoints and chain seeds agreed; exact burn-in, MSE recomputation,
author SCS hashes and fixed-step report provenance were checked. These
execution checks do not establish convergence or comparative performance.

Inspect the exact revision's GitHub Actions results separately. The version-10
results below retain their original validation scope and are not v11
sampling-performance evidence.

## Historical version-10 validation

Version-10 comparisons required VI for both map families and matched fixed-cloud
sizes and optimizer update counts. Full fits use 1,000 updates on 2,000 fixed
points per family; smaller presets reduced both counts equally. DCS's cloud
reuse policy changed at every preset, so preceding comparison outcomes did not
validate version-10 performance.

The automated suite includes new-request policy enforcement, fitting-budget
provenance and saved-report compatibility. Inspect GitHub Actions for the exact
source revision's test results.

Local validation on 5 November 2026 passed **371 regression tests**. Four Quick
comparisons completed in JAX/Numba at 100D/500D, covering **40 production
chains**, with reports, figures and matched fixed-cloud metadata checked.
A further two-chain SCS step-policy control completed using the primary run's
saved maps without refitting.

Separate full-budget fits at both dimensions produced finite maps and all
1,000 finite losses for each family. Each fit recorded 2,000 unique fixed draws
and 2,000,000 objective-point evaluations. SCS's fitted parameters and every
loss matched a direct call to the pinned author optimizer exactly. These
checks validate execution and fitting provenance, not sampler rankings or
convergence of a full sampling run.

## Preceding author-fitting validation

The preceding 5 November 2026 author-fitting update routed every VI-enabled SCS fit through the pinned
author routine. **360 local regression tests passed** after this change.
Checks compare logistic and symmetric Student-t fits directly with the author
optimizer and prevent old configuration labels from selecting another fitter.
The author source files and all production transition kernels are unchanged.

At that preceding release, the existing version-9 100D comparison's ball and
sphere maps and complete VI loss histories reproduced exactly. This check
does not imply that the new version-10 DCS fits are unchanged. Both logistic-prior scenarios and the 50D
super-heavy-tail experiment completed their Quick runs and reports, covering
14 production chains. Separate full-author-budget fits (1,000 updates on 2,000
fixed draws) produced finite maps and losses for all three targets. These
checks validate fitting and execution; they are not new performance rankings.

The preceding version-9 comparisons used the SCS authors' VI implementation and
fixed h=0.1, with an acceptance-tuned sensitivity control. Billiards calibrated
their own production kernel in that historical release; version 11 instead
uses the shared fixed DCS move by default. The associated development study completed
**280 study chains** and **48 Quick-check chains**, historically covering VI on/off, both
SCS step policies and a separate check with the full author VI budget. See the
[protocol](COMPARISONS.md), [calibration study](AUTHOR_SCS_CALIBRATION.md) and
[reviewer guide](REVIEW_GUIDE.md).

Quick checks establish execution and bookkeeping, not comparative performance.
The development study informed the default step policy; assess that choice
with fresh sampling seeds under the frozen protocol. The available checks do
not establish optimal tuning, convergence or universal sampler rankings.

## Automated coverage

The regression suite checks:

- Target densities, independent references, inverse transformations and
  Jacobians, including stability near the ball boundary.
- Proposal geometry, invariant-transition checks, billiard reversibility,
  and agreement between corresponding JAX and Numba implementations.
- Pinned author SCS source hashes, author VI settings, separate pilot budgets,
  frozen production steps and map/seed provenance.
- Exact first-third burn-in, repeated rejected states, radial estimates,
  MSE aggregation, and functional ESS against the specified ArviZ calculation.
- Worker scheduling, checkpoint compatibility, bounded result-processing
  memory, figure generation and saved-report regeneration.

The public experiment checks cover both comparison implementations,
100D/500D targets. New comparisons require VI; saved-report and lower-level
checks retain coverage of earlier untuned maps. They also exercise the 200D NUTS comparison,
200D random-walk baselines, the 50D super-heavy-tail target, Eight Schools,
GARCH and both logistic-prior settings. Their distinct fitting, diagnostic
and timing policies are documented in [the experiment guide](EXPERIMENTS.md)
and [reproducibility notes](REPRODUCIBILITY.md).

Constant or invalid functional traces receive unavailable ESS with a reason;
they are not treated as successful diagnostics. Logistic reference generation
has separate diagnostic screens, and finite reference uncertainty remains
explicit. See [ESS definitions](ESS.md) and [target references](TARGETS.md).

## Run the checks

After installing the repository's CPU requirements, run from the repository
root:

```sh
python -m pip check
python -m unittest discover -s tests -v
```

For an end-to-end experiment check, select **Quick check** in the launcher,
or follow the [command-line instructions](RERUN.md). Use fresh output folders
for new comparisons and run every compared method under the same settings.

GitHub Actions runs the automated suite and execution checks on Linux and
Windows. Inspect the workflow for the exact commit being evaluated; a local
pass does not imply that every remote workflow has finished. Current statuses
and downloadable execution records are in
[GitHub Actions](https://github.com/MihaBresar/DiffeomorphicMonteCarlo/actions).
