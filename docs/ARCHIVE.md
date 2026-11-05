# Historical protocols and development evidence

The repository retains earlier experiments so that changes to the comparison
can be inspected. These records are not interchangeable with the
[current version-9 protocol](COMPARISONS.md). Each saved run records its
configuration, source hashes, fitting parameters, seeds and timing policy.
Use the generating revision when reproducing a historical run; a current
entry point can have different defaults.

| Record | Scope |
|---|---|
| [Author SCS calibration](AUTHOR_SCS_CALIBRATION.md) | Version-9 development study, both SCS step rules and the full author VI check; includes the rule used to select the current default. |
| [Own-kernel calibration](CALIBRATION_RESULTS.md) | Version-8 radial pilots for both SCS and billiards, including historical calibration controls. |
| [Earlier comparison results](COMPARISON_RESULTS.md) | Earlier target/VI/backend studies, including coordinate measurements and cases favoring SCS. |
| [Three-method SCS comparison](SCS_100D.md) | Earlier DCS-RWM/BW versus SCS implementations, timing and tuning revisions. |
| [Boundary comparison](BOUNDARY_100D.md) | Earlier reflected-BW comparison and boundary-treatment study. |
| [Fixed-step audit](FIXED_STEP_AUDIT.md) | Transition parity, fixed-count experiments and tuning differences. |
| [Symmetric Student-t audit](SYMMETRIC_T_AUDIT.md) | Symmetric-target map and step-size checks. |
| [Timed skew-t audit](SKEW_T_TIMED.md) | The earlier ν=2 equal-time radial comparison. |
| [Joint radial calibration](JOINT_RADIAL.md) | Earlier joint map-scale and proposal-scale pilots. |
| [Implementation audit](IMPLEMENTATION_AUDIT.md) | Findings about the original implementation and later corrections. |
| [Validation history](VALIDATION.md) | Dated checks with their original revision and experiment scope. |

Historical reports retain their measured values, including unfavorable and
mixed rankings. A later implementation improvement does not retroactively
change them. Replotting preserves the saved protocol; it cannot add an unsaved
sampler, recover unthinned ESS, or turn a timed run into an equal-count run.

Generated runs belong in `results/`, which is excluded from source control.
Selected numerical evidence in `validation/` retains full precision and
provenance. Hardware and package versions in measured records are experimental
metadata; they are not hardware requirements or performance promises.

The original manuscript's ordinary BW sampler and the current reflected/Billiard
variants are distinct. See [manuscript correspondence](MANUSCRIPT_CHANGES.md)
for the interpretation of corrected and extended experiments.
