# Run experiments without typing commands

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
2. Select **All seven experiments + posterior table**, leave **Quick check**
   selected, choose **CPU workers** (start with 2–4), and click **Start run**. This checks that the computer can run
   the complete pipeline. Quick-check graphs are not publication results.
3. After the quick check succeeds, select **Full paper settings** and click
   **Use all CPU threads**, then **Start run** again. On the Ryzen 9 9950X3D
   this allows up to 32 workers with SMT enabled. This regenerates all original
   figures, supplementary plots, and the combined Eight Schools/GARCH table. You can also
   select one experiment at a time.
4. After completion, links to figures and tables appear automatically below the
   progress log. Click a PNG or PDF link to view a graph. All files remain in
   **results → click-runs → dated folder**.
   Under **Exact numbers and tables**, open `paper_values.html` to read or copy
   the full-precision MSE and sample-count values. Supplementary diagnostics
   appear in a separate collapsed section.

The **DCS map** selector offers **Möbius map**, which fits R, centre and Möbius
parameters by VI, and **Vanilla map**, which uses untuned R=1, centre=0 and
Möbius=0. Both keep the experiment's β and the same proposal-scale warmup.
This selector changes DCS only; the main experiments still fit the SCS map.

The additional **100D: DCS/SCS map and tuning audit** experiment compares both
versions automatically, including step sizes chosen on separate pilot chains.
Use Quick check first, then Full paper settings for 20 chains of 500,000
production iterations in each diagnostic case. Its exact-value tables and
all-case plots appear automatically. See [what this audit measures](FIXED_STEP_AUDIT.md).

To switch off map fitting for both methods, select **100D symmetric t: untuned
maps and step tuning**. This separate audit includes both fully untuned maps
and scale-only VI controls, with step sizes pretuned by ordinary warmup or pilots
targeting either five functions or the paper's two radial functions. Full
settings run 20 chains of 500,000 production iterations in each of 12 cases.
Choose Quick check first, then Full paper settings. No map selection or commands
are needed. See [the symmetric audit](SYMMETRIC_T_AUDIT.md).

For the main experiments, select one DCS map choice and start a fresh run; repeat with the other choice and the
same experiment, run size and worker count for an ablation. The figures and
exact-value tables identify the chosen map. SCS fitting and the identity maps
in Eight Schools and GARCH are unchanged. Fixed R=√d is no longer offered.

Selecting 32 workers does not change the number of chains. In the main
experiments, a smaller reproduction uses at most eight workers, a quick check
two, and logistic reference preparation four. Full paper experiments use at
most 20 or 50 chains depending on the design. The two diagnostic audits queue
cases together and can use more workers: the symmetric quick check has 24
production jobs, and its full run has 240. Hardware capacity and the selected
worker count still apply. The launcher displays the applicable production limit.

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
