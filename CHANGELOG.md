# Changes

## Parallel CPU execution

- Added a CPU-worker selector and bounded independent chain processes, including logistic references.
- Added native Linux/Windows CPU affinity, explicit thread limits, worker provenance and compilation accounting; macOS scheduling remains OS-managed.
- Preserved deterministic chain seeds, matched NUTS/DCS budget ordering, atomic checkpoints and saved-worker-count resume.
- Added cross-platform worker checks and made Git optional for running downloaded ZIP copies.
- Earlier sequential-only checkpoints remain available for plotting; start a new run after updating the scientific source.

## 0.2.0 — corrected experimental implementation

- Centralized targets, ball/cap transforms, tuning and kernels; retained thin compatibility entry points.
- Corrected GARCH prior, shared constrained-coordinate handling and posterior reference provenance.
- Matched paper settings: d=200 RWM comparison, ν=3 fixed-step comparison, symmetric ν=0.5 target, 30-second logistic budget.
- Replaced the skew-t coordinate reference generator, used analytic radial references, and added bounded directional checks.
- Replaced pooled logistic references with independently generated NUTS references and diagnostic gates for paper use.
- Froze adaptation after warmup, removed artificial density clipping/cutoffs, synchronized timing and reported complete setup costs.
- Saved per-chain draws, seeds, tuning, work, failures and diagnostics; corrected RMSE aggregation; added plot-only and resume workflows.
- Pinned the full Python 3.12 CPU dependency closure, added mathematical tests and end-to-end CI, finalized MIT licensing and citation metadata.
- Added Windows/macOS launchers and a local browser interface for setup, running, resuming and viewing results without typing commands.

Previous manuscript figures remain archival. Version 0.2.0 requires new production simulations; it does not preserve old performance numbers.
