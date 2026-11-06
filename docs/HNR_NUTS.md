# HnR and NUTS posterior comparisons

Eight Schools has ten parameters; GARCH(1,1) has four. Both compare DCS-HnR with NUTS. The 200D synthetic NUTS comparison uses DCS-BW instead.

HnR uses the full feasible chord and slice-bracket shrinkage, without a proposal step or acceptance target. Numba compiles its transition, ball map and posterior density, including the GARCH variance recursion. Compilation is recorded outside sampling time.

HnR receives the mean NUTS production-time budget. Construction, compilation and separate warmup are recorded separately. Both methods discard the first third of production iterations. Reference draws keep their source processing.

Eight Schools uses centred coordinates (θ,μ,log τ), including the log-scale Jacobian; outputs use physical τ. GARCH uses the same unconstrained potential and constant physical prior on its stationary triangle for both samplers. See [target definitions](TARGETS.md).

Posterior errors square individual chains' errors before averaging. Inspect moment errors together with counts, preparation/sampling costs, functional ESS, NUTS divergences and depth saturation. Faster execution does not guarantee smaller error.

The slice-attempt guard limits an individual transition's work; most transitions may finish earlier. A guarded failure must retain its declared rejection/error behavior and must not silently change the target.

Large HnR output can make diagnostics take additional time after sampling. The [result pipeline](RESULT_PROCESSING.md) stages data in bounded buffers and reports its separate processing costs.
