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
   selected, and click **Start new run**. This checks that the computer can run
   the complete pipeline. Quick-check graphs are not publication results.
3. After the quick check succeeds, select **Full paper settings** and click
   **Start new run** again. This regenerates all five main figures, the extra
   directional plots, and the combined Eight Schools/GARCH table. You can also
   select one experiment at a time.
4. After completion, click **Show figures and tables**. Click a PNG or PDF link
   to view a graph. All files remain in **results → click-runs → dated folder**.

Full paper runs can take many hours and several gigabytes. Keep the computer
plugged in, awake, and available for the benchmark. The supplied comparisons use
one worker to avoid competing chains distorting the wall-clock comparisons;
GPU acceleration is not part of this reproduction configuration. Timing-based
graphs will differ across machines.

## Stop, resume, or redraw

- Click **Stop** to interrupt a run. Completed chains are retained; the chain
  interrupted before its checkpoint will be repeated.
- To continue later, reopen the launcher, choose the dated folder under
  **Saved launcher runs**, and click **Resume selected run**. This also works
  after a restart. Resume with the same code, Python environment and computer;
  if you change computers, start a new run for comparable timings.
- Click **Rebuild figures** to redraw an existing completed run without
  sampling again. It requires all the relevant saved results.
- Use **Close launcher** when finished. Closing only the browser tab does not
  terminate the process. Closing the console/IDLE can interrupt running work.

If the page says **Needs attention**, read the progress log. The same run log
is saved as `launcher.log` in its dated results folder. Setup failures are
displayed on the page. Existing command-line workflows remain available in
[README](../README.md) and the [figure-by-figure rerun guide](RERUN.md).

The launcher was exercised on macOS. Windows launch scripts and paths are
provided, but have not been executed on a Windows computer in this validation.
