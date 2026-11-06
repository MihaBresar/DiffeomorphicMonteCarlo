# Run without typing commands

## Download and install

Open [the repository](https://github.com/MihaBresar/DiffeomorphicMonteCarlo), choose **Code → Download ZIP**, and extract the whole folder. Keep all project files together. Install 64-bit Python 3.12; on Windows include Python in PATH and install IDLE.

Move the extracted folder to its intended location before setup. If moving it later, recreate its generated `.venv` environment with the setup button. Keep `results/` to retain experiments.

## Open the launcher

- Windows: double-click **Start.bat** and keep its console open.
- macOS: double-click **Start.command** and keep its Terminal window open.
- Any system with IDLE: open **START_HERE.py**, then choose **Run → Run Module** or F5. Keep IDLE open.

Open the Python launcher rather than opening `launcher/index.html` directly.

1. Click **Set up environment** and wait for **Environment ready**.
2. Choose **DCS vs original SCS**, **Quick check**, target dimension and one or two CPU workers.
3. Click **Start run**. Quick checks verify execution rather than performance.
4. After a successful check, choose **Smaller reproduction** or **Full paper settings**.
5. Select **Seed 1–Seed 8** and workers. Click **Use all CPU threads** if appropriate for available memory.
6. Start the run, keep the launcher open and prevent computer sleep.

The main comparison loads frozen maps and steps. Smaller uses 20 chains × 3 seconds per method and Full uses 50 × 20 seconds. Both have 1,500 separate warmup transitions, followed by first-third production discard. No manual fitting or step selection is required.

Original SCS keeps author VI and h=0.1. DCS and calibrated zero-SCS use radius and step values selected on their final zero maps with the same pilot policy and budgets. These settings load automatically; see [the exact protocol](REPORTED_STUDY.md). The optional [preparation replay](PREPARATION.md) is available for inspecting and reproducing the fitting and selection.

**Numba extension** runs the same comparison with Numba DCS production against original JAX SCS. Other choices are listed in [the experiment catalogue](EXPERIMENTS.md). The additional experiments use Numba DCS/random-walk production and retain their compiled NUTS backend.

Seed choices alter sampling streams without altering maps or fitting seeds. Exact seeds are recorded automatically. Repeating them on another machine need not reproduce timed counts.

## Results and stopping

The launcher prints sampling and processing progress. Reports may require extra time after all chains finish; ESS and plots have separate costs.

When complete, open **paper_values.html** from the result links. Figures are available as PNG/PDF, and CSV/JSON preserve full precision. Runs are stored under **results → click-runs → dated folder**.

**Stop** interrupts the current run; completed files remain saved. For a new comparison, start a fresh complete run. Keep the same worker choice across methods and avoid unrelated intensive activity. More workers use more memory and do not guarantee faster execution.
