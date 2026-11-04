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
2. Leave **Five samplers: DCS-RWM, DCS-BW, SCS, reflected BW and BW + rotation** and
   **Quick check** selected. Choose a **Target distribution** and whether to
   use **Variational map fitting — both DCS and SCS**. Choose **CPU workers**
   (start with 2–4), then click **Start run**. Quick-check graphs verify
   execution; they are not publication results.
3. After the quick check succeeds, choose **Smaller reproduction** or **Full
   paper settings** and click **Use all CPU threads**, then **Start run** again.
   On the Ryzen 9 9950X3D, this allows up to 32 workers with SMT enabled.
4. After completion, links to figures and tables appear automatically below the
   progress log. Click a PNG or PDF link to view a graph. All files remain in
   **results → click-runs → dated folder**. Under **Exact numbers and tables**,
   open `paper_values.html` to read estimates, MSE, sample counts and timing
   values. The page uses two decimals; its CSV downloads retain full precision.

The two current comparison choices are:

- **Five samplers: DCS-RWM, DCS-BW, SCS, reflected BW and BW + rotation** runs all five in
  the same JAX CPU framework, with full-state recording and common radial and three-coordinate
  processing.
- **Optimized Numba BW, reflected BW and BW + rotation versus SCS** runs the three ball-walk
  methods in optimized Numba loops against the unchanged author SCS JAX
  transition. DCS reuses the same JAX step calibration, then caches densities
  and coordinate values and accumulates all eight estimates inside the sampling loop. This measures
  the selected implementations, including their different processing costs.

Each choice offers **100D skewed Student-t, ν=3** and **500D skewed Student-t,
ν=3**. Both use DCS β=2.
The original radial plots remain. Six coordinate figures show means of X₁,
X₂ and X₃ and the tails X₁ > 3, X₂ < −3 and X₃ > 3. X₂ is the mirrored
skewed coordinate; X₃ has a symmetric Student-t marginal. Independent and
analytic reference values, with their precision, are shown in the report.
Start fresh runs of every method for the version-4 comparison. Earlier reports
do not contain the rotation mixture, and pre-version-3 reports lack the
all-iteration X₂/X₃ values.
See [the current comparison protocol](COMPARISONS.md) for target definitions,
thresholds, tuning and timing policies.

Use **With VI** to fit both map families with the same training resources.
**Without VI** uses unit scale, zero centre and zero
deformation for both families. In either case every method still receives
separate proposal-step tuning: original methods target acceptance 0.234;
reflected BW selects by squared movement in the ball. Sampling steps stay
fixed. BW + rotation splits this same tuning budget between reflected steps and
angles; each iteration then chooses one move with equal probability. Its angular
candidates are in `angular_tuning.csv`. The reflected candidates and scores
appear in `reflected_tuning.csv`;
there is no special upper step bound of 2.

Both current comparisons offer a seed-batch selector. **Original seeds (batch 0)** uses
sampling seed 901001; **Fresh batch 1–5** use new sampling and warmup streams
while keeping map-fitting seed 42 fixed when VI is on. Choose a different batch to check
whether an observed MSE ordering persists. Repeating the same batch reuses
its streams, but timing can change how many iterations finish and where
first-third burn-in ends, so counts and estimates can still differ. These
batches assess sampling variability conditional on the selected maps, not
variability from fitting new maps. Keep target, VI choice, run size and worker
count the same when comparing methods. Each method runs for the same
production sampling time:

| Run size | Chains per method | Sampling seconds per chain |
| --- | ---: | ---: |
| Quick check | 2 | 0.05 |
| Smaller reproduction | 20 | 2.00 |
| Full paper settings | 50 | 20.00 |

The figures report generated iteration counts, mean-norm and tail-probability
estimates, MSE, and the observed bias/variance decomposition across chains.
The exact-value exports also include inexpensive empirical MSE uncertainty
summaries from the per-chain errors. Treat these as descriptive; they are not
formal 95% confidence intervals, and the heavy tail of the norm statistic
makes rare large errors especially important.
Methods run separately with the same worker setting. Selecting
32 workers allows 2, 20 or 32 simultaneous chains in Quick check, Smaller
reproduction or Full paper settings respectively; it does not multiply the
chain count by the number of methods.

The other original experiments remain in the Experiment menu. **All seven
experiments + posterior table** includes the five-sampler JAX comparison on
the 100D skewed-t target and the other six experiments. Its VI selector controls
both maps in that comparison. Run the optimized Numba and 500D
comparisons individually. The former three-method 100D and boundary choices
have been removed from the launcher; their command-line scripts and saved-report
regeneration remain available for historical compatibility.

For large posterior runs, the log distinguishes sampling, saving and
convergence diagnostics. A chain can finish sampling before its file and
report are ready. These stages now use bounded memory and report their
progress; see [result processing](RESULT_PROCESSING.md).

For the other experiments, the **DCS map for the other experiments** selector offers **Möbius map**, which
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
