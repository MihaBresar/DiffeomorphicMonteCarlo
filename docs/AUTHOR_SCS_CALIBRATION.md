# Author SCS fitting and proposal-step sensitivity

Version 9 uses the SCS authors' unchanged VI implementation and production
transition, applied to the same target as DCS. DCS fitting, proposals and
calibration remain unchanged. SCS has two explicit step policies: adaptation
toward acceptance 0.234 in a discarded pilot followed by a frozen production
step, or the authors' skew-t h=0.1 throughout discarded pilot and production.
The global default is **fixed h=0.1**, following the smaller development
study's fallback rule, which was fixed before inspecting its results.
The rule does not choose by DCS ranking. Acceptance adaptation remains
available explicitly for sensitivity checks.

Both step policies use exactly the same fitted maps. The control therefore
isolates step choice within the original author fitting recipe; it does not
measure the effect of changing VI.

## What comes from the authors

The pinned revision is
[`ea6f27a6cb1d0ca90d8c1cd92c32fc3dcf8ea565`](https://github.com/liusf15/sub-cauchy-sampler/tree/ea6f27a6cb1d0ca90d8c1cd92c32fc3dcf8ea565).
Its [skew-t experiment](https://github.com/liusf15/sub-cauchy-sampler/blob/ea6f27a6cb1d0ca90d8c1cd92c32fc3dcf8ea565/experiments/run_skewt.py)
calls VI with seed 0, 2,000 reference draws and learning rate 0.01. The
[VI routine](https://github.com/liusf15/sub-cauchy-sampler/blob/ea6f27a6cb1d0ca90d8c1cd92c32fc3dcf8ea565/src/scp_core.py)
defaults to 1,000 updates and reuses that one reference cloud. It clips each
physical coordinate to ±1,000 inside the VI objective before evaluating both
the target and the Jacobian. The [optimizer](https://github.com/liusf15/sub-cauchy-sampler/blob/ea6f27a6cb1d0ca90d8c1cd92c32fc3dcf8ea565/src/train.py)
is Optax Adam without gradient clipping. The vendored files are unchanged
and checked against their recorded hashes.

The authors' skew-t experiment has fixed step h=0.1, not acceptance
adaptation. Therefore the acceptance-tuned version is an explicitly
modified step-size policy, while the fixed-step control retains that aspect
of their experiment. Neither is an exact reproduction of their experiment:
we use the manuscript's normal-CDF-skewed t(3), radial functions, equal-time
budgets and first-third burn-in. Their skew-t target uses a Student-t CDF.

## Fitting and pilot budgets

| Preset | DCS VI updates × fresh draws/update | SCS VI updates × fixed cloud size | Separate transitions/method/chain |
|---|---:|---:|---:|
| Quick check | 20 × 128 | 20 × 128 | 100 |
| Smaller reproduction | 400 × 500 | 400 × 500 | 1,000 |
| Full paper settings | 1,500 × 2,000 | 1,000 × 2,000 | 1,500 |

DCS retains fit seed 42, Adam 0.01, a gradient-norm cap of 10 and no
coordinate clipping. SCS retains author fit seed 0 and the fitting choices
above. The optimizers and unique-draw budgets are intentionally different.
For example, full SCS VI reuses 2,000 draws over 1,000 updates, giving
2,000,000 target evaluations in the optimization objective, rather than
2,000 evaluations total. DCS uses 3,000,000 fresh draws over 1,500 updates.
Quick and Smaller presets reduce the authors' full fitting budget.
The smaller study's 400 updates on 500 reference draws need not produce the
same map or step-policy ranking as the full 1,000-update/2,000-draw fit.

VI off uses unit scale, zero centre/deformation and no fitting in both
families. Production target densities are unclipped in every method. All
methods start their separate pilots from the same physical all-ones point;
their pilot endpoints and initial production states need not coincide.
SCS's acceptance-tuned and fixed-step controls receive the same number of
discarded pilot transitions. Fixed h=0.1 is not described as a tuned step.

Equal production time measures sampling efficiency after preparation.
Compilation, fitting, pilots and ESS computation are recorded separately
and excluded from that clock. Different preparation costs therefore remain
relevant to an end-to-end application. Fitting losses omit different base
normalizers and must not be compared as normalized KL divergences.

## Smaller JAX study

The study uses the 100D skewed t(3) target, 20 chains per method, 2.00 seconds
per chain, 1,000 separate pilot transitions, four workers and fresh sampling
seed batch 7. Both VI settings and both SCS step choices are reported. The
first third of actual production iterations is discarded. The five-method
primary uses acceptance-tuned SCS; the separate control reuses its exact map,
seeds and budgets while keeping SCS h=0.1. It does not refit either family.

The seed-batch number only selects sampling seeds: batch `b` starts at
`901001 + 1000000*b`, with successive seeds assigned to successive chains.
Batch 7 therefore starts at 7901001; it uses the same random-number generator
as batch 0. Each chain's seed determines its pilot and production streams.
Changing this batch does not change the fitting seeds (DCS 42, author SCS 0)
or refit maps with new VI randomness. "Fresh" refers to a new set of sampling
seeds, not a different sampling or fitting algorithm.

A pre-results addendum also checks SCS with its **full author VI budget**,
1,000 updates on 2,000 fixed reference draws, while keeping the same small
production budget of 20 chains × 2.00 seconds and 1,000 pilot transitions.
This additional VI-on pair compares acceptance tuning with fixed h=0.1 on
the exact same full-budget SCS map. It contains SCS only; it is reported
separately from the five-method smaller-fit comparison. The VI-off pair
needs no additional fit.

The selection rule was fixed before these results: retain acceptance tuning
as the global default only if its observed MSE is lower than fixed h=0.1 for
**both radial functions in the full-author-VI on pair and in the no-VI pair**.
The pre-results full-fit addendum supersedes the reduced-VI-on part of the
original selection plan; all reduced-VI outcomes are still reported.
Otherwise use the paper's fixed h=0.1 globally. This development rule does not
establish optimality or a statistically reliable ranking, and it does not
select different settings for individual reported cases. All outcomes remain
reported. A fresh full-paper seed batch 8 evaluates the frozen choice.
Full-paper runs additionally use longer chains and 1,500 pilot transitions;
their ordering can still differ from this development check.

Validated on 5 November 2026: 304 chains completed across execution checks,
the smaller five-method comparisons, their SCS controls and the full-VI SCS
pair. Tables use two displayed decimals; the [CSV](../validation/author_scs_calibration_results.csv)
and [JSON](../validation/author_scs_calibration_results.json) retain full
precision, exact counts, times, fitting parameters and provenance.

### Smaller fitting budgets: all five methods and SCS control

| VI | Method / step policy | Tail MSE | Mean-norm MSE | Samples/s | Tail ESS/s | Norm ESS/s |
|---|---|---:|---:|---:|---:|---:|
| on | DCS-RWM | 4.31e-07 | 8.85e-03 | 156,067.54 | 8,805.80 | 8,819.46 |
| on | DCS-BW | 4.68e-07 | 7.23e-03 | 147,025.39 | 8,655.00 | 8,200.13 |
| on | SCS (acceptance) | 7.80e-07 | 9.17e-03 | 104,538.31 | 10,772.53 | 7,726.69 |
| on | DCS-BW-reflect | 1.97e-07 | 6.13e-03 | 108,238.35 | 38,494.84 | 15,536.47 |
| on | DCS-Billiard | 1.50e-07 | 2.34e-03 | 108,275.51 | 35,118.33 | 21,591.69 |
| on | SCS (h=0.1) | 3.85e-07 | 5.57e-03 | 104,936.37 | 13,962.03 | 9,911.02 |
| off | DCS-RWM | 2.92e-07 | 3.81e-03 | 150,776.00 | 10,621.19 | 11,113.83 |
| off | DCS-BW | 6.15e-07 | 4.93e-03 | 143,522.45 | 10,494.44 | 10,939.34 |
| off | SCS (acceptance) | 5.45e-07 | 4.83e-03 | 107,961.67 | 15,385.00 | 14,684.56 |
| off | DCS-BW-reflect | 1.07e-07 | 1.33e-03 | 106,243.64 | 42,884.11 | 23,586.06 |
| off | DCS-Billiard | 8.08e-08 | 1.71e-03 | 107,352.61 | 39,685.79 | 29,452.88 |
| off | SCS (h=0.1) | 2.85e-07 | 3.60e-03 | 110,771.07 | 25,336.00 | 19,577.45 |

Both VI-on fits use 400 updates with 500 reference draws per update, fresh
for DCS and fixed for SCS. The fixed-step SCS rows reuse the corresponding
primary sphere map. They are additional controls, not a sixth sampler.

Billiards had lower observed MSE than either SCS step policy for both radial
quantities in these smaller-fit runs. Rankings still depend on the quantity:
with VI, fixed-step SCS had lower mean-norm MSE than reflected BW and ordinary
RWM/BW; without VI, reflected BW had lower mean-norm MSE than billiards.
These are batch observations, not general rankings.

### Full author VI: SCS step-policy check only

| VI | Method / step policy | Tail MSE | Mean-norm MSE | Samples/s | Tail ESS/s | Norm ESS/s |
|---|---|---:|---:|---:|---:|---:|
| on | SCS (acceptance) | 5.71e-07 | 6.99e-03 | 105,181.65 | 10,528.48 | 7,403.12 |
| on | SCS (h=0.1) | 5.60e-07 | 5.87e-03 | 104,738.86 | 14,954.81 | 10,552.32 |

This pair uses the full author VI budget but small production runs. It must
not be presented as a full-paper five-method comparison or substituted
silently for the smaller-fit SCS rows above. Fixed h=0.1 gave lower observed
MSE and higher functional ESS/s for both functions in this pair.

### The default-selection comparisons

| Map setting used for selection | Acceptance / fixed tail MSE | Acceptance / fixed norm MSE |
|---|---:|---:|
| Full author VI | 1.02 | 1.19 |
| No VI | 1.91 | 1.34 |

Ratios above one favor fixed h=0.1 in this batch. Acceptance tuning did not
meet the pre-result rule, so **fixed h=0.1 is the global default**; acceptance
remains explicitly selectable. The full-VI norm-MSE ratio of 1.19 is a small-run
observation, not statistical proof that fixed h=0.1 is optimal.

### SCS fitted maps and selected steps

| Map setting | R | Centre norm | Observer norm | Recorded fitting seconds |
|---|---:|---:|---:|---:|
| Smaller VI | 0.84 | 0.60 | 0.28 | 0.67 |
| No VI | 1.00 | 0.00 | 0.00 | 0.00 |
| Full author VI | 1.10 | 0.42 | 0.29 | 2.54 |

The coordinate vectors remain available in the JSON. Latitude is 1.1 in
all cases. Each acceptance/fixed pair uses exactly the same sphere map;
its control's new fitting time is zero because the primary fit is reused.
Fitting seconds above are the original measured costs, not comparable
end-to-end run times or costs charged again to the control.

| Map setting | SCS policy | Minimum step | Median step | Maximum step | Production acceptance |
|---|---|---:|---:|---:|---:|
| Smaller VI | Acceptance 0.234 | 0.07 | 0.13 | 0.51 | 18.56% |
| Smaller VI | Fixed h=0.1 | 0.10 | 0.10 | 0.10 | 22.23% |
| No VI | Acceptance 0.234 | 133.30 | 3,269,017.37 | 3,269,017.37 | 25.85% |
| No VI | Fixed h=0.1 | 0.10 | 0.10 | 0.10 | 47.84% |
| Full author VI | Acceptance 0.234 | 0.09 | 0.14 | 0.78 | 17.86% |
| Full author VI | Fixed h=0.1 | 0.10 | 0.10 | 0.10 | 25.62% |

Without VI, 14 of 20 acceptance-tuned chains reached the numerical ceiling
exp(15), approximately 3,269,017.37. The resulting mean production acceptance
was 25.85%, versus 47.84% for fixed h=0.1. This shows a limitation of this
acceptance-target rule for SCS; it does not show a defective SCS transition
or that a larger raw step improves exploration. The fixed-step alternative
avoids relying on that calibration behavior.

Samples/s counts all generated transitions, including rejected repeats and
sampling burn-in. Functional ESS/s uses unthinned post-burn-in histories and
divides by total production time. It is not a convergence guarantee.
MSEs are conditional on fixed maps and only 20 chains: the largest squared
error contributed 58.18% of the smaller-VI billiard norm MSE and 48.63% of
the full-VI acceptance-tuned SCS norm MSE. The t(3) norm lacks a fourth
moment, so rare excursions can substantially change empirical MSE.
No confidence claim or universal ordering follows from these tables.


## Reproduction

Run both primary comparisons before their fixed-step controls:

```sh
python run_experiment.py sampler-comparison --target skew-t-100d --vi on --preset reproduce --seed-batch 7 --workers 4 --scs-step-rule acceptance --output-dir results/author-scs-vi-on
python run_experiment.py sampler-comparison --target skew-t-100d --vi off --preset reproduce --seed-batch 7 --workers 4 --scs-step-rule acceptance --output-dir results/author-scs-vi-off
python scripts/reproduce_scs_step_control.py results/author-scs-vi-on --output-dir results/author-scs-fixed-step-vi-on
python scripts/reproduce_scs_step_control.py results/author-scs-vi-off --output-dir results/author-scs-fixed-step-vi-off
```

The explicit `--scs-step-rule acceptance` reproduces the study's primary
policy despite the fixed-step global default. Use `--scs-step-rule fixed`
to run a five-method primary with SCS h=0.1. The control helper automatically
runs the **opposite** SCS policy: fixed h=0.1 after an acceptance-tuned primary,
or acceptance tuning after a fixed-step primary. Thus either primary policy
can supply the same fitted maps and resources for its sensitivity check.

Use fresh output folders. The control requires a complete version-9 primary
and checks its scientific sources, maps and allocation before reuse. It
records original fitting costs separately from zero new fitting work.
These controls assess the step policy conditional on the primary fitted
maps; they do not average over refitting uncertainty.

To reproduce the additional full-author-VI check without running DCS
production, save the following as `run_full_scs_vi_check.py` in the repository
root, then run that file with the repository's configured Python environment.
The main guard is required for Windows worker startup. It uses public
repository modules and new result folders; no external review script or saved
fit is needed. It fits the full-preset maps once, runs only SCS, then reuses
those maps for its control.

```python
from pathlib import Path
from scripts import reproduce_comparison as suite
from scripts import reproduce_scs_100d as runner
from scripts import reproduce_scs_step_control as control

def main():
    config = suite.configuration(
        'paper', workers=4,
        overrides=dict(runs=20, time_budget=2.0, warmup=1000, thin=10,
                       seed_batch=7, scs_step_rule='acceptance'),
        comparison='jax', target='skew-t-100d', vi='on',
    )
    config['algorithms'] = ['SCS']
    root = Path(__file__).resolve().parent
    primary = root / 'results/full-author-vi-small-sampling'
    alternative = root / 'results/full-author-vi-fixed-step'
    runner.run(config, primary)
    control.run(primary, alternative)


if __name__ == '__main__':
    main()
```

The smaller study is development evidence. Report all its outcomes, and
freeze the selected protocol before a fresh full-paper seed batch, such as
batch 8. A decision informed by smaller-study outcomes is not an independent
validation result; the fresh batch supplies that assessment. In particular,
do not select the SCS setting according to which makes DCS look strongest.

Only radial-tail probability and mean norm are measured here, with their
MSEs, counts, production time and functional ESS. The current billiard proposal differs from the
original paper's BW sampler. These results cannot establish general
dominance, marginal superiority or correctness of the original figures.
