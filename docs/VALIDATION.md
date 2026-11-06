# Validation

## Reported zero-centre study

On 6 November 2026, the complete local regression suite passed **426 tests**.
The [release validation record](../validation/reported_study_release.json)
summarizes its scope and the execution checks below.

The default launcher experiment now reproduces the [reported study](REPORTED_STUDY.md),
using its frozen full-budget map assets, zero DCS centre and deformation, and
dimension-specific preset radii and steps. It does not refit maps or select steps
during a new run. The earlier fitted-map version-11 recipe remains separately named.

Independent checks compared **36 preset configurations** with the original study:
both dimensions and implementations, all three run sizes, and both recorded seed
batches plus the fresh default batch. Map parameters, proposal scales, target,
warmup, burn-in and reporting rules matched the declared study settings. All four
Quick executions (JAX/Numba, 100D/500D) completed, covering **40 production chains**
and 20 generated figure families. Checks verified frozen-map provenance, unchanged
original SCS, exact first-third burn-in and unthinned functional ESS. The Numba
option is explicitly labelled an extension of the original JAX study.

These are execution and configuration checks, not new performance benchmarks.
Some Quick executions overlapped in time; their timings must not be used to rank
samplers. The bundled reported results preserve both original sampling batches,
all 400 active-method chain records and the pooled analysis. Additional DCS
development and conditional-on-fit limitations are documented with the study.

## Identity-map and fixed-SCS follow-up

The JAX ball transform now skips Möbius arithmetic only when every supplied
deformation coordinate is exactly zero. The nonzero-deformation path remains
unchanged. Billiards already use zero deformation in their effective map;
this optimization does not refit or change any method's map or proposals.

Validation passed **44 focused core regression tests**, **3 standalone public
identity-transform regressions** and **8 additional transformation parity
checks**. Coverage includes the constant transformed density of the
uniform-ball Student-t case, an independent Jacobian calculation, finite
interior points near the boundary, the existing rejection guards and all
supported low-level target families. Common-input checks in 100D/500D retained
exact acceptance decisions and states for all four DCS methods and the author
SCS controls, including 3,072 transitions from independent stationary starts.
The original fixed-0.1 SCS driver also retained its warmup endpoint, metadata
and subsequent recorded batch. Compiled identity-map projections contained
no Möbius dot operation and fewer divisions. These checks establish the
intended algebraic equivalence; they do not promise bitwise agreement for
every possible floating-point trajectory or comparative sampler superiority.

For controlled studies, low-level configurations may set
`step_tuning_policies['SCS']='fixed_scs_move_v1'` and a positive finite
`scs_fixed_move`. The supplied value remains frozen through discarded warmup
and production using the unchanged author transition. Tests cover 0.025,
0.05, 0.1 and 0.2 in both dimensions and verify that metadata does not label
this policy as the author's fixed-0.1 experiment. This is an execution API,
not a step-selection rule. It remains available for archived results and
separate review studies; current five-method entry points allow only the
original author SCS baseline at 0.1.

A serial paired throughput check retained complete state projection and
recording, with unchanged-path controls. It showed modest or negligible
changes depending on the method and dimension, so no portable speed guarantee
is made. Compare fresh complete cohorts when assessing timed performance
after an implementation update. The larger historical validation counts
below retain their original release scope.

## Earlier fitted-map version-11 scope

Version 11 uses one shared DCS move: fixed BW/reflected-BW/billiard increment
radius h=0.07 and Gaussian RWM coordinate standard deviation h/√(d+2).
All four keep these steps during discarded equilibration. SCS retains h=0.1.
Public entry points select these preset steps automatically and reject manual
step overrides. Historical mixed-policy tests remain separate; they cover
acceptance-target 0.234 for ordinary RWM/BW, ball-movement pilots for reflected
BW and radial-movement pilots for billiards. Version-10 fixed-cloud VI budgets and
maps remain unchanged. Fresh sampling is needed to evaluate the new step
policy. This recipe remains available separately from the default reported study.

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
checks retain coverage of earlier untuned maps. The reported-study entry points
load verified frozen VI assets. They also exercise the 200D NUTS comparison,
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
