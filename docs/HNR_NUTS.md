# HnR and NUTS: changes affecting the comparison

The HnR experiments are **Eight Schools (10 dimensions)** and **GARCH(1,1)
(4 dimensions)**. The 200D comparison uses ball walk, not HnR. Current HnR
uses the full feasible chord and slice-bracket shrinkage; it has no proposal
step size or 0.234 acceptance target.

The current native implementation compiles the HnR loop, ball transformation
and posterior density. This removes Python dispatch from proposals and, for
GARCH, the 200-observation variance recursion. It preserves the canonical
posterior and slice transition. Compilation is outside the sampling clock and
reported separately. Repeat the paired implementation benchmark with
`python scripts/benchmark_canonical_compilation.py`.

The historical posterior scripts differ from the current comparison in several
scientifically relevant ways:

- **Timing:** the old NUTS time included construction, compilation and warmup,
  and HnR received that entire time budget. Current HnR receives mean NUTS
  sampling time; each method's setup costs are separately reported. The old
  setup overhead is especially substantial in short runs.
- **Eight Schools coordinates:** the old HnR sampler used a signed τ and
  evaluated/reported its absolute value. The current sampler uses log τ,
  its Jacobian, and exp(log τ) for physical output. They can represent the
  same physical posterior but yield different HnR transitions. Current HnR
  therefore does not reproduce the historical signed-τ geometry.
- **GARCH target:** the old NUTS model's conditional uniform β₁ prior added
  a factor 1/(1−α₁), whereas HnR and PosteriorDB use constant joint density
  on the stationary triangle. The current methods share the correct
  PosteriorDB target. Restoring the old discrepancy would invalidate the
  comparison against that reference.
- **Error statistic:** the old table averaged estimates across chains before
  squaring errors. The current per-chain RMSE squares errors before averaging.
  These statistics differ because opposite chain errors can cancel when pooled.
- **Burn-in:** the old posterior HnR discarded 20% of its timed trace. The
  current experiments discard the first third of generated sampling iterations
  after separate tuning, for both HnR and NUTS.

The 10,000-attempt slice guard does not imply 10,000 evaluations per iteration:
the loop stops on acceptance. A short probe found 3.95 evaluations/iteration
for Eight Schools and 6.93 for GARCH, with maxima 17 and 18, respectively.
The probe cannot bound rare events or establish mixing.

These differences explain why the old table and a new run need not have the
same ranking. Faster execution addresses one source of the discrepancy. It
does not establish that either sampler must have the smaller error. The next
geometry-specific control, if needed, is signed-τ versus log-τ HnR with the
same physical posterior, starting point and sampling time.

Use fresh **Eight Schools** and **GARCH(1,1)** runs after installing this version.
Inspect per-chain first/second-moment error, counts, setup and sampling time,
and NUTS divergences/depth saturation. The pinned reference draws retain their
source processing; they are not truncated again by the new sampling burn-in.
