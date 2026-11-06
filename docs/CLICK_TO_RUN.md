# Run experiments without typing commands

The **200D skew-t: random-walk baselines** choice produces a compact report:
mean norm, tail probability, MSE, sample counts and time. It skips the lengthy
individual and cross-chain diagnostics automatically for every run size.

The launcher runs experiments on the **local computer**. GitHub hosts the code;
benchmark computation takes place locally. An internet connection is
needed for the initial package installation. The reference data are bundled.

## Download the project

1. Open
   [MihaBresar/DiffeomorphicMonteCarlo](https://github.com/MihaBresar/DiffeomorphicMonteCarlo).
2. Click the green **Code** button, then **Download ZIP**.
3. Extract the ZIP into a normal local folder, such as Documents. On Windows,
   right-click the ZIP and choose **Extract All**. Open the extracted folder
   containing `START_HERE.py`. Do not run files from inside the ZIP.

For later updates, [GitHub Desktop](https://desktop.github.com/) is convenient:
**File → Clone repository → GitHub.com**, select this repository and click
**Clone**. Subsequent updates use **Fetch origin / Pull origin**. These are
alternative download methods; only one project copy is needed. Avoid updating
the code during a run.

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

Drag the extracted project folder to a convenient location **before setup**.
Keep all its files together. If the folder is moved after setup, delete only
its generated `.venv` folder and repeat setup; Python environments do not move
reliably between paths or computers. Keep the `results` folder to retain runs.

## Use the buttons

1. Click **Set up environment** and wait for **Environment ready**. This installs
   the pinned packages inside this project and can take several minutes.
2. Leave **Reported study: zero-centre DCS vs original SCS (JAX)** and
   **Quick check** selected. Choose **100D** or **500D** and **CPU workers**
   (start with one or two), then click **Start run**. The study loads its exact
   bundled maps and preset parameters; no manual step-size choice is needed.
   Quick-check graphs verify execution, not sampler performance.
3. After the check succeeds, choose **Smaller reproduction** or **Full paper
   settings**. Select the desired worker count or click **Use all CPU threads**,
   then **Start run**. More workers use more memory and do not guarantee faster runs.
4. When the run finishes, links appear below the log. Open **paper_values.html**
   for figures, MSE, exact sample counts, measured time and functional ESS.
   PNG and PDF figures are available; CSV and JSON retain full precision.
   Files stay in **results → click-runs → dated folder**.

## Reported study and previous comparison

The reported study contains DCS-RWM, DCS-BW, SCS, DCS-BW-reflect and
DCS-Billiard. All four DCS methods use zero centre and zero deformation,
with frozen radius and step values from the reported experiment. Their common
h is 0.025 in 100D and 1/√500 in 500D. Gaussian RWM uses coordinate standard
deviation h/√(d+2). SCS retains the original author-fitted sphere map and h=0.1.
Every preset reuses the same full-budget maps without refitting or retuning.
See [the exact protocol and preparation history](REPORTED_STUDY.md).

**Smaller reproduction** runs 20 chains per method for 3.00 seconds each,
matching one reported batch's budget. **Full paper settings** runs 50 chains
for 20.00 seconds each. Both use 1,500 separate discarded warmup transitions
and additionally discard the first third of production iterations. The original
pooled result combined two 20-chain batches, using two workers and JAX.
The separately labelled **Numba extension** keeps the same study maps and steps
but changes DCS's production implementation; it is not the backend used for
those reported results.

The study uses additional DCS calibration from its development stage. It is
not equal total tuning effort against original SCS, and its radial results
are not a guarantee of overall mixing or a guaranteed ranking on another run.

The **Previous fitted-map comparison** options retain the older version-11
protocol: maps are fitted anew at the selected VI budget, physical centres
remain fitted, and the shared DCS h is 0.07. Billiards use zero effective
Möbius deformation; other DCS variants retain the fitted map. These are
different experiments from the reported zero-centre study. **All seven**
continues to run this previous comparison plus the manuscript experiments;
choose the reported study explicitly to reproduce its configuration.

All comparisons show mean norm, radial-tail probability, MSE, exact counts,
time, functional ESS and ESS per second. They use analytic radial references.
Coordinate measurements are omitted. Original radial figures and ESS figures
remain available. Saved reports retain their original parameters when rebuilt.

## Repeated runs and diagnostics

The reported study defaults to fresh seed batch **22**. The repeat selector
also labels recorded batches **20** and **21** explicitly. Reusing one of these
batches matches its random-stream inputs; different hardware and time stopping
can still change the number of samples and resulting errors. Other fresh
batches keep the same maps and parameters. The previous fitted-map recipe
retains its earlier sampling-batch choices.

Run each target separately and compare all five methods together. Keep the
worker count and computer load comparable. Full paper runs extend sampling
under the same reported-study maps; they do not refit those maps.

ESS is computed after sampling from unthinned scalar histories, including
rejected states. The log reports this stage. FFT work is limited to one chain
at a time to bound memory; diagnostic time is separate. Temporary unthinned
histories are discarded afterward. An unavailable ESS has a reason and is not
replaced by the iteration count. See [ESS interpretation](ESS.md).

Other manuscript experiments retain their own map and step policies. SCS uses
its authors' VI routine where fitting is enabled, including logistic regression
and the 50D infinite-first-moment example; their checked Numba production and
acceptance-target 0.234 calibration remain separately documented.

## Stop or finish

- Click **Stop** to interrupt a run. Click **Start run** to restart from scratch;
  every launch uses a new dated results folder.
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
