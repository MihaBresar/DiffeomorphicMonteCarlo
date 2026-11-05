# Windows worker queue fix

The September 11 queue fix addresses `FileNotFoundError` while a worker reads
`000000.worker-3.json` (or another worker's claimed job filename). This happens
before that worker starts its NUTS chain. The old queue moved each job to a
worker-specific filename and then opened it. The reported failure shows that
the destination disappeared between those operations. Competing Windows
renames are the likely cause; the precise operating-system interleaving has
not been reproduced locally.

Workers now exclusively create one shared claim directory per chain, then read
the unchanged job JSON. Losing workers skip the existing claim. Job inputs are
never moved, and each resumed launch creates a fresh queue from missing chain
checkpoints. The sampling kernels, tuning, seeds, budgets and CPU assignments
are unchanged by this fix.

## Update and start fresh without commands

1. Click **Close launcher**, then close its console window if it remains open.
2. Open the [GitHub repository](https://github.com/MihaBresar/DiffeomorphicMonteCarlo),
   choose **Code → Download ZIP**, and extract it into a new folder.
3. Double-click **Start.bat** in the extracted project and click
   **Set up environment**. Wait for **Environment ready**.
4. Choose the experiment, run size and CPU workers, then click **Start run**.
   The experiment starts from scratch, and figure links appear when it finishes.

Keep the previous project folder to retain its output files. There is
no need to copy its environment or interrupted run into the new project.

## Advanced command-line compatibility

The exception below documents the earlier queue-only release. The subsequent
map-choice/scientific source update requires a fresh run; it does not extend
that exception. The current launcher starts fresh runs.

The compatibility exception applies only to the exact registered queue fix
from release `0a87f3d20045c19d6cd866a2b4de1dc37dc6764b`. Resume still checks the
configuration, scientific source files, Python/packages and hardware settings.
Other source or environment changes remain grounds for a new run. This does
not bypass the separate provenance check for logistic reference preparation.

The original run fingerprint and saved checkpoint bytes are retained. The
manifest records the approved source transition and the source hashes used by
each invocation; new production checkpoints record their producing source.
The exact allowed before/after file hashes are in
[`validation/resume_compatibility.json`](../validation/resume_compatibility.json).

Command-line resume requires new sampling only for unfinished chains. Figures
are rebuilt when the experiment finishes. Completed runs do not need to be
rerun for this fix.
