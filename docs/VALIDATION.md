# Validation

## Current release

The 5 November 2026 update routes every VI-enabled SCS fit through the pinned
author routine. **360 local regression tests passed** after this change.
Checks compare logistic and symmetric Student-t fits directly with the author
optimizer and prevent old configuration labels from selecting another fitter.
The author source files and all production transition kernels are unchanged.

The existing version-9 100D comparison's ball and sphere maps and complete VI
loss histories reproduced exactly. Both logistic-prior scenarios and the 50D
super-heavy-tail experiment completed their Quick runs and reports, covering
14 production chains. Separate full-author-budget fits (1,000 updates on 2,000
fixed draws) produced finite maps and losses for all three targets. These
checks validate fitting and execution; they are not new performance rankings.

The current version-9 comparisons use the SCS authors' VI implementation and
fixed h=0.1, with an acceptance-tuned sensitivity control. Billiards calibrate
their own production kernel. The associated development study completed
**280 study chains** and **48 Quick-check chains**, covering VI on/off, both
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
100D/500D targets and VI on/off. They also exercise the 200D NUTS comparison,
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
