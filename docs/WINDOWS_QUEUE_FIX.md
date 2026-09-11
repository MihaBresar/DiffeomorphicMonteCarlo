# Recovering the Windows worker queue failure

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

## Update and resume without commands

1. Close the launcher with **Quit launcher** and wait for its terminal to close.
2. Open the [GitHub repository](https://github.com/MihaBresar/DiffeomorphicMonteCarlo),
   choose **Code → Download ZIP**, and extract it into a temporary folder.
3. Copy the extracted **dcs_bench** and **validation** folders into your existing
   project folder, alongside `START_HERE.py`. Choose **Replace the files in the
   destination**. Keep your existing `results` and `.venv` folders in place.
4. Double-click **Start.bat** in the existing project. Under **Saved launcher
   runs**, select the interrupted run and click **Resume selected run**.

For the reported 200D DCS-versus-NUTS run, the eight completed DCS checkpoints
are reused; missing NUTS chains start again with their original seeds. Saved
NUTS checkpoints, if any completed before the failure, are also reused. The
launcher restores the original worker count and experiment settings.

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

New sampling is needed only for unfinished chains. Figures are rebuilt when
the experiment finishes. Completed runs do not need to be rerun for this fix.
