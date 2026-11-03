# Run experiments without typing commands

The **200D skew-t: random-walk baselines** choice produces a compact report:
mean norm, tail probability, MSE, sample counts and time. It skips the lengthy
individual and cross-chain diagnostics automatically for every run size.

The launcher runs the experiments on **your computer**. GitHub stores the code;
it does not run the full paper experiments for you. An internet connection is
needed for the initial package installation. The reference data are bundled.

## Get the project onto another computer

1. Sign into GitHub and open
   [MihaBresar/DiffeomorphicMonteCarlo](https://github.com/MihaBresar/DiffeomorphicMonteCarlo).
   While the repository is private, use an account that has access.
2. Click the green **Code** button, then **Download ZIP**.
3. Extract the ZIP into a normal local folder, such as Documents. On Windows,
   right-click the ZIP and choose **Extract All**. Open the extracted folder
   containing `START_HERE.py`. Do not run files from inside the ZIP.

For later updates, [GitHub Desktop](https://desktop.github.com/) is convenient:
**File → Clone repository → GitHub.com**, select this repository and click
**Clone**. Subsequent updates use **Fetch origin / Pull origin**. These are
alternatives; you only need one copy. Avoid updating the code during a run.

## Install Python once

Install **Python 3.12, 64-bit** if it is not already installed. The
[Python 3.12.10 download page](https://www.python.org/downloads/release/python-31210/)
provides the Windows and macOS installers. This is the last 3.12 release with
official binary installers; later 3.12 security releases are source releases.
On Windows, select **Add python.exe to PATH** and keep **IDLE** selected in the
installer. The benchmark dependencies are pinned for Python 3.12.

## Open the launcher

- **Windows:** double-click **Start.bat** (it may appear as just “Start” with
  type “Windows Batch File”). A browser window opens. Keep the console open.
- **macOS:** double-click **Start.command**. A browser window opens. Keep the
  Terminal window open. If macOS will not open the downloaded script, use the
  IDLE method below; no Terminal commands are required.
- **Either system, or an IDLE-equipped Linux desktop:** open **IDLE** from the
  Python installation, choose **File → Open**, select **START_HERE.py**, then
  choose **Run → Run Module** (or press **F5**). Keep IDLE open while it runs.

You can drag the extracted project folder to any convenient location **before
setup**. Keep all its files together. If you move it after setup, delete only
its generated `.venv` folder and repeat setup; Python environments do not move
reliably between paths or computers. Keep the `results` folder to retain runs.

## Use the buttons

1. Click **Set up environment** and wait for **Environment ready**. This installs
   the pinned packages inside this project and can take several minutes.
2. Leave **100D skewed t (ν=3): DCS-RWM, DCS-BW and SCS** and **Quick
   check** selected, choose **CPU workers** (start with 2–4), and click **Start
   run**. Quick-check graphs verify execution; they are not publication results.
3. After the quick check succeeds, choose **Smaller reproduction** or **Full
   paper settings** and click **Use all CPU threads**, then **Start run** again.
   On the Ryzen 9 9950X3D, this allows up to 32 workers with SMT enabled.
4. After completion, links to figures and tables appear automatically below the
   progress log. Click a PNG or PDF link to view a graph. All files remain in
   **results → click-runs → dated folder**. Under **Exact numbers and tables**,
   open `paper_values.html` to read estimates, MSE, sample counts and timing
   values. The page uses two decimals; its CSV downloads retain full precision.

See [the full protocol](SCS_100D.md) for matched fitting and step calibration, the unmodified SCS author transition, and separate timing clocks.

The default comparison contains exactly **DCS-RWM, DCS-BW and SCS** on one
100D skewed Student-t target with **3 degrees of freedom**. Each method runs
for the same production sampling time:

| Run size | Chains per method | Sampling seconds per chain |
| --- | ---: | ---: |
| Quick check | 2 | 0.05 |
| Smaller reproduction | 20 | 2.00 |
| Full paper settings | 50 | 20.00 |

The figures report generated iteration counts, mean-norm and tail-probability
estimates, MSE, and the observed bias/variance decomposition across chains.
The comparison fits both map families with the same training budget and tunes
all three proposal steps automatically, then freezes them for sampling. The
**DCS map** selector is disabled for this matched protocol.
The three methods run separately with the same worker setting. Selecting
32 workers allows 2, 20 or 32 simultaneous chains in Quick check, Smaller
reproduction or Full paper settings respectively; it does not multiply the
chain count by the number of methods.

The other original experiments remain in the Experiment menu. **All seven
experiments + posterior table** runs the full collection. The former map and
step-size diagnostic audits are no longer launcher choices; their existing
scripts, documentation and saved results remain available for reference.

For large posterior runs, the log distinguishes sampling, saving and
convergence diagnostics. A chain can finish sampling before its file and
report are ready. These stages now use bounded memory and report their
progress; see [result processing](RESULT_PROCESSING.md).

For the other experiments, the **DCS map** selector offers **Möbius map**, which
fits R, centre and Möbius parameters by VI, and **Vanilla map**, which uses
untuned R=1, centre=0 and Möbius=0. Both keep the experiment's β and the same
proposal-scale warmup. The identity maps in Eight Schools and GARCH are
unchanged. To compare map choices, start a fresh run with each choice and keep
the experiment, run size and worker count the same.

The **50D t: infinite first moment, equal samples** experiment uses 500,000
production iterations per chain in Full paper settings. Its mean norm is
infinite, so it reports tail probability, the finite mean log norm and marginal
quantiles.

Worker limits depend on the number of available chains. The other experiments
use at most two production workers for a quick check, eight for a smaller
reproduction, and 20 or 50 for full paper settings. Logistic reference
preparation uses at most four. Hardware capacity and the selected worker count
also apply. The launcher displays the production limit for the selection.

Full paper runs can take many hours and several gigabytes. Keep the computer
plugged in, awake, and available for the benchmark. Independent chains can run on
several CPU cores; methods are compared using the same worker setting. More
workers need more memory. Linux/Windows workers are pinned to detected cores;
macOS assigns cores through its scheduler. See [parallel execution](PARALLEL.md).
GPU acceleration is not part of this reproduction configuration. Timing-based
graphs will differ across machines and worker settings.

## Stop or finish

- Click **Stop** to interrupt a run. Click **Start run** when you are ready to
  start again from scratch; every launch uses a new dated results folder.
- Use **Close launcher** when finished. Closing only the browser tab does not
  terminate the process. Closing the console/IDLE can interrupt running work.

If the page says **Needs attention**, read the progress log. The same run log
is saved as `launcher.log` in its dated results folder. Setup failures are
displayed on the page. Existing command-line workflows remain available in
[README](../README.md) and the [figure-by-figure rerun guide](RERUN.md).

The launcher was exercised on macOS. Automated CI covers Linux and Windows
worker execution; the Windows desktop launch button still requires a desktop
check. Open `START_HERE.py` through the launcher files or IDLE, not `launcher/index.html`
directly: the HTML interface needs the local Python runner.
