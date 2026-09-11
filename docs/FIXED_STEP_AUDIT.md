# What changed in the DCS–SCS fixed-step comparison

The controlled checks found no weakened DCS ball-walk transition or extra SCS
move in the rewritten kernels. With the same maps, states, proposal sizes and
random proposals, both algorithms matched their original implementation's
2,000 acceptance decisions. State differences were below 4×10⁻¹⁶. This check
does not cover the historical artificial boundary and CDF truncation regions.

The experiment protocol did change. The supplied old fixed-step script uses
ν=2, fits maps with different VI rules, adapts DCS throughout sampling using a
smoothed proposal size, and fixes SCS at h=0.1. The current main comparison uses
ν=3, separate warmup, and frozen production tuning for both methods. The old
DCS inverse used for initialization also omits a radial factor and does not
place the chain at its stated physical starting point. These are different
comparisons; old PNGs have no run-level provenance proving their exact settings.

## Full-length controlled study

All diagnostic production cases use the same 100D target with ν=3 and skew
coefficients (100,−100,0,…), the physical starting point (1,…,1), 1,500 discarded
warmup steps, **20 chains of 500,000 production iterations**, and thinning 50.
Every case therefore retains **10,000 values per chain**. Analytic radial
references are shared. No chains are removed because of their result.

An eleven-case bridge exchanges old/current VI fits and proposal schedules,
and includes fully untuned vanilla DCS. That is 220 full-length chains. With
the same current warmup/freeze rule, replacing the old DCS fit with the current
fit changed mean-norm MSE from **0.01582078118074839** to
**0.01754202047935494**. The paired 95% bootstrap interval for the difference
was **[−0.013452401017565526, 0.017776551397712007]**. This does not identify a
large, statistically resolved DCS fitting regression.

SCS's point estimates improved with the current fit. At its original fixed
h=0.1, mean-norm MSE changed from **0.01238335865878731** to
**0.006710295221197556**. However, individual differences are uncertain and
conditional on a single fit. Simply undoing SCS step tuning is not supported:
its fixed-h result already did well in this batch.

The current warmup/freeze comparison itself changed ranking across two
independent 20-chain batches: DCS/SCS mean-norm MSE was
**0.01754202047935494 / 0.009472886809457908** in the first and
**0.010089795812193767 / 0.010407030595857635** in the second. Thus the paper's
replicate count does not make close MSE rankings stable automatically.

## Independent step-size pilots and held-out evaluation

The fair tuning diagnostic freezes the maps, gives every method eight candidate
step sizes and the same pilot budget, then selects using pilot chains only.
It maximizes the smallest bulk ESS per recorded value across log radius, a
95%-radial indicator, a skew contrast, and coordinates 3 and 4. The pilots use
four exact target draws as shared starting points: this is an explicitly
oracle-assisted synthetic diagnostic, not a general posterior tuning rule.

The full pilot uses 2,000 discarded and 8,000 retained steps per chain, four
chains per candidate, with separate seeds from both VI and production. It
chose ball radius **0.15** for Möbius DCS, tangent coordinate standard deviation
**0.1** for SCS, and ball radius **0.2** for vanilla DCS. These parameters have
different geometric meanings. No chosen value was a grid endpoint. The full
grid and all diagnostics are saved; these are selected candidates, not proven
optimal values. Some DCS pilot coordinate R-hat values reached 1.077, so the
pilot ranking has uncertainty.

The following results use a new set of 20 full-length production seeds, never
used to choose the scales:

| Method with pilot-selected step | Tail-probability MSE | Mean-norm MSE |
|---|---:|---:|
| Möbius DCS-BW | 0.0000011521982093886906 | 0.008997801680116855 |
| SCS | 0.0000012670443624586907 | 0.009686881334540084 |
| Vanilla DCS-BW | 0.000001215736749318758 | 0.007127539275843314 |

Möbius DCS has slightly smaller point estimates than SCS for both radial
metrics. **The paired uncertainty interval includes zero.** The paired mean-norm MSE
difference DCS−SCS is −0.0006890796544232307 with descriptive 95% bootstrap
interval **[−0.0063290714136581975, 0.0049483220134272235]**. These results do
not establish decisive or universal superiority.

SCS has substantially faster coordinate mixing in this study. On the held-out
thinned production traces, coordinate-3 bulk ESS was about 183,192 for SCS,
21,919 for Möbius DCS, and 15,598 for vanilla DCS. Each is computed from the
same total of 200,000 retained values. Radial performance and coordinate mixing
are different questions. At the selected Möbius ball-walk radius, the pilots
rejected about 55.75% of proposals outside the ball. Local ball proposals can
therefore be competitive for radial observables while changing direction much
more slowly than spherical proposals.

The appropriate response is to report that distinction, tune on independent
pilots, and retain all comparators and uncertainty. Global proposals or a
local/global mixture would be a separately named DCS kernel and a further
experiment; they must not be silently substituted for the ball walk.

## Run this comparison without commands

In the launcher select **100D: DCS/SCS map and tuning audit**. Run **Quick check**
first, then **Full paper settings** for the full diagnostic. This separate
option compares both map versions automatically. It does not change the main
seven experiments. It runs independent pilots and 14 production cases covering
the bridge and pilot-selected versions, then displays figures and exact-value
tables. All cases use equal iterations, warmup and retention rules.

For the main experiments, **DCS map → Möbius** fits R, centre and Möbius
parameters by VI. **Vanilla** keeps R=1, centre=0 and Möbius=0, with the configured
β unchanged and no VI. Both retain proposal-scale warmup. The removed fixed
R=√d setting is available only when reading historical saved results.

Advanced command-line equivalent:

```bash
python scripts/audit_fixed_steps.py --preset paper --workers 8 --output-dir results/fixed-step-audit
```

To reproduce the additional held-out seed batch in the table above, use a
second fresh folder and the recorded production seed base:

```bash
python scripts/audit_fixed_steps.py --preset paper --workers 8 --production-seed-base 9901001 --output-dir results/fixed-step-audit-heldout
```

Both commands rerun the independently seeded pilots. Production seeds do not
affect their choices. The second command includes the six held-out cases in
the recorded study and the other eight audit cases for completeness.

The script needs a fresh output folder. It saves the predeclared protocol,
source/environment hashes, old/current fitted maps, all pilot candidates,
functional traces, per-chain estimates, paired uncertainty, PNG/PDF figures,
and full-precision CSV/HTML values. Diagnostic cases share a worker pool:
**their elapsed times are not a throughput comparison**. The main fixed-time
experiments continue to compare methods in separate batches.

The historical VI functions are loaded as audit input from
`validation/legacy_fixed_step.py`; the main sampler never executes that file.
The supplied old ν=2 setting has infinite second moment for the norm, so its
finite-replicate mean-norm squared-error averages are particularly unstable and
do not estimate a finite stationary population MSE. This study uses ν=3.

Full records: [fixed_step_comparison.json](../validation/fixed_step_comparison.json)
and [fixed_step_maps.json](../validation/fixed_step_maps.json). The recorded
study used source release `2be8cc30e0da0c6c77acb48b386ba6677d11bfea`, before the
map-selector presentation update; source hashes are included. The public audit
uses the same numerical definitions and records its own source version. Its
default production seeds reproduce the first bridge batch; the additional
held-out batch above used seeds 9901001–9901020.

Optimal-acceptance heuristics depend on the target class and efficiency
criterion; 0.234 is not a universal optimum. The SCS paper also leaves optimal
scaling in this setting open. See [Yang, Roberts and Rosenthal](https://arxiv.org/abs/1904.12157),
[Neal, Roberts and Yuen](https://arxiv.org/abs/1210.5090), and
[the SCS paper](https://arxiv.org/html/2601.11066v1).
