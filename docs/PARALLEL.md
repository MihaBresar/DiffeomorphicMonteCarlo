# Parallel execution

The worker selector controls independent sampling processes. Each worker uses a bounded CPU thread allocation. Choose a count within the detected CPU allocation and available memory. More workers can reduce elapsed runtime but need not increase per-chain throughput.

## Main comparison

The six methods run one method at a time in synchronized waves of chains. Each method receives the same worker count and wave allocation. Workers compile and warm up before the sampling barrier; method order rotates between waves.

The final wave may contain fewer chains. Its actual size and allocation are recorded. Keep workers fixed across a comparison and avoid unrelated intensive CPU work. Operating systems without supported affinity control manage placement themselves; the run records that policy.

## Other experiments

Workers take pending chains from the experiment queue. Independent chain randomness does not depend on which worker receives a job. Numba production avoids Python interpreter contention within a chain. NUTS retains its compiled backend.

Fitting, reference preparation, reporting and individual diagnostics may use different concurrency. A low CPU percentage during one of these stages does not imply that production chains are serial.

## Memory and processing

Each active worker needs target state, numerical runtime and output buffers. The comparison captures scalar histories for ESS; full posterior runs retain physical draws. More workers therefore increase memory and disk activity.

The result pipeline streams output and processes diagnostics in bounded workspaces. ESS work may be serialized after a wave, and plotting can take time after sampling ends. Check log messages and recorded processing durations; see [result processing](RESULT_PROCESSING.md).

Use **Stop** to interrupt a run. Completed files are retained; new launcher runs use fresh folders. Extract updates into a fresh project folder and set up its environment before running them.
