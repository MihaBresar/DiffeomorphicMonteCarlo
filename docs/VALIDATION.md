# Scientific validation record

This document distinguishes cleanup completed in the codebase from validation
that must be completed before numerical comparisons are presented as evidence
for the paper.  It is intentionally prominent: reproducible execution alone
does not establish that a comparison targets the intended distribution.

The project provenance is PosteriorDB: all benchmark models and
ground-truth/reference quantities originate there. The remaining work is to
pin the exact PosteriorDB identifiers and revision and to verify that each
local implementation reproduces its corresponding record.

## Required before reporting synthetic skew-t results

### 1. Prove or test the target/reference correspondence

The skew-t modules locally reconstruct a reference by selecting the sign of a
Gaussian vector with an auxiliary Gaussian draw and then applying a Gamma
scale mixture.  Their primary target log densities use a multivariate
Student-t kernel multiplied by a **normal-CDF** skewing term.  Depending on
the skew-t parameterization, the selection-plus-mixture construction may
instead induce a Student-t-CDF skewing factor.  The current files do not
provide a derivation or unit test that resolves this question.

Before using a locally reconstructed reference value, add a cited derivation
and at least one numerical check against the pinned PosteriorDB reference
showing that the generator and every target implementation have the same
parameterization. The 50-dimensional module
also contains an unused helper with a Student-t-CDF skewing term, which makes
this review especially important.  Until resolved, describe these quantities
only as *fixed-seed Monte Carlo diagnostics*, not exact ground truth.

### 2. Quantify reference Monte Carlo error

The 10-million-draw default references are finite estimates.  Report their
Monte Carlo uncertainty, or use independent reference replicates/analytic
values, for every metric used in a table or figure.  A single fixed seed makes
a run repeatable but does not quantify uncertainty.

### 3. Align implementations and metric definitions

Verify that NumPy, JAX, and BlackJAX target implementations agree at selected
points; compare log-density differences, gradients when applicable, and
parameter constraints.  Make the Euclidean norm explicit in figures and
tables where the original source uses `|x|`, and state all burn-in/thinning and
storage choices with each metric.

## Required before reporting posterior benchmark results

### 4. Pin the PosteriorDB provenance

The Eight Schools and GARCH experiments contain fixed posterior means and
second moments embedded in source code.  Record the exact PosteriorDB model,
data, reference draw file, revision/commit, parameter ordering, and any
transformations used to produce those values.  Then regenerate or independently
verify the moments against that pinned reference.  The GARCH observations are
now bundled with a documented source; the posterior summaries still need this
separate provenance record.

### 5. Use an independent logistic-regression diagnostic

The logistic-regression Q–Q panels use pooled samples from the evaluated
samplers on the horizontal axis.  They are useful for visual disagreement but
cannot establish correctness or convergence.  Add an independent high-quality
reference, replicated chains with convergence diagnostics, or replace the
figure with a diagnostic that does not pool the methods being compared.

## Benchmark protocol to freeze for the manuscript

Document the canonical algorithm names, versioned implementation, tuning
rules, initialization, adaptation/warmup policy, number of chains, random seed
schedule, device/backend, and which stages count toward runtime.  The code
records many of these values now; the paper should report the same protocol.

## Cleanup changes already made

- GARCH no longer assumes a developer-specific working directory; it loads the
  bundled, validated data or an explicit override.
- The 200-dimensional BlackJAX NUTS worker now continues from the state
  returned by window adaptation rather than discarding it and reinitializing.
- Synthetic finite references are named as Monte Carlo references in outputs.
- The logistic Q–Q plot is labelled as a pooled-sampler diagnostic.
- Every public run writes figures, aggregate summaries, configuration, and
  runtime provenance to a dedicated output directory.

These implementation improvements do not replace the validation tasks above.
