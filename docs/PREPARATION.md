# Reproducing preparation

The public preparation driver reproduces the source VI fits and radius/step calibration for the current 100D and 500D comparisons. It does not read production MSE, saved comparison results or files outside this repository. Original SCS remains the authors' fitted-map, h=0.1 baseline.

## Complete replay

Use the installed Python 3.12 environment and a fresh output directory:

```sh
python scripts/prepare_comparison.py --stage all --verify-bundled --output-dir results/preparation-replay
```

This fits both families in both dimensions, checks the fitted maps and all loss values against the supplied reference fits, and then calibrates using the regenerated maps. Verification uses numerical tolerances because floating-point results can differ across platforms; elapsed times and metadata are not expected to reproduce byte for byte. A mismatch is reported, not silently replaced with a supplied result.

Ordinary launcher runs load checksum-verified bundled inputs. Preparation replay is a reproduce-and-verify path: it does not overwrite those inputs or silently change future experiments. After verification, use the normal launcher to sample with the recorded protocol. Record the software commit and retain the preparation output alongside sampling results. Timed sampling counts still depend on hardware.

The stages may also be run separately:

```sh
python scripts/prepare_comparison.py --stage fit --verify-bundled --output-dir results/refit
python scripts/prepare_comparison.py --stage pilot --fit-dir results/refit/fits --output-dir results/recalibration
python scripts/prepare_comparison.py --stage pilot --output-dir results/bundled-map-calibration
```

The last command calibrates directly from the supplied source fits. None of these commands overwrites an existing preparation directory. Settings are defined in the script, not selected interactively.

## Fitting

Both families use 1,000 optimizer updates and 2,000 fixed reference points: 2,000,000 objective-point evaluations per family and dimension. The full, unconstrained source fits determine the radius grid's scale; their fitted centres and deformation parameters are retained for provenance. The original SCS baseline uses its complete source fit.

| Setting | DCS | SCS |
|---|---|---|
| Seed | 42 | 0 |
| Initial scale | 1 | 1 |
| Reference cloud | Fixed uniform-ball draws | Original author's fixed cap draws |
| Optimizer | Adam, learning rate 0.01 | Original author Adam, learning rate 0.01 |
| Clipping | Gradient norm capped at 10 | Physical coordinates clipped at ±1,000 within VI |
| Production evaluations | Unclipped | Unclipped |

These are the native fitting routines, with equal evaluation budgets. They are not identical objectives in identical coordinates. The SCS optimizer, projection and transition remain pinned author code. Source checksums and the numerical environment are recorded.

## Matched calibration on final maps

Before evaluating any candidate, set the DCS physical centre and Möbius deformation to zero, and the calibrated SCS physical centre and observer to zero. No centre choice is made using the pilot or production outcomes. DCS β stays 2; SCS latitude stays 1.1.

| Resource or rule | Both optimized families |
|---|---|
| Radius factors | 0.5, 1, 2, 4 times the respective source VI radius |
| Step candidates | Four in the method's native units |
| Candidate count | 16 |
| Exact physical starting points | The same 4,096 draws for every candidate and family |
| Transitions per start | 8, including rejected repeats |
| Total selection transitions | 524,288 per family per dimension |
| Selection objective | Same radial CDF and radial-tail movement score |
| Diagnostic starts after selection | 20,000 from a separate random stream |

DCS tests h=(0.25, 0.5, 1, 2)/√d; SCS tests h=(0.025, 0.05, 0.1, 0.2). Literal h values are not equivalent distances across the ball and sphere geometries. Equal transition counts also do not imply equal computation time.

For radial CDF U and tail indicator I with exact tail probability p, the score is

```text
min( mean((U8 − U0)^2) / (2/12),
     mean((I8 − I0)^2) / (2*p*(1−p)) ).
```

Maximize this score. Exact ties prefer the radius factor closest to 1 in absolute log distance, then the DCS scaled step closest to 1 or SCS h closest to 0.1, then the lower radius factor and lower step. If all scores are zero, retain the tie-rule winner and disclose that fact; do not enlarge the grid. Both selections are written before the separate diagnostic draws are generated, and that diagnostic cannot alter them. Its stream is separate from selection, not a new confirmatory study of choices made during development.

The selected DCS-BW R and h apply to ordinary BW, reflected BW, billiard and RWM. Gaussian RWM uses coordinate standard deviation h/√(d+2). There is no additional optimization for the other DCS variants. Calibrated SCS uses its own selected R and h; original SCS retains its complete author fit and h=0.1.

## Random streams and records

Calibration seed: **842901174**. Physical starts use NumPy `SeedSequence([seed, dimension, stage, 0])`, with stage 0 for selection and stage 1 for the separate diagnostic. JAX transition keys fold in dimension, stage, family (1 for BW, 2 for SCS), then starting-point index. Within a family, every candidate reuses the transition-key array. The batch size is 256 and is part of the recipe. Sampling uses separate recorded streams.

The output contains fitted maps and loss histories, numerical verification, a plan written before evaluation, every candidate's score and acceptance, frozen selections, scalar pilot/diagnostic arrays, source hashes, environment and separate cost records. The compact public `matched_calibration.json` records the current choices; historical performance comparisons are not required or distributed.

Exact target sampling and analytic radial references provide equal oracle assistance to both families in this synthetic study. The lag-eight score is a bounded radial proxy, not integrated ESS, production MSE or a claim of globally optimal parameters. Production timings exclude preparation; use the recorded preparation costs when discussing total cost.
