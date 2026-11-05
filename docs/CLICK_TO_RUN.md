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
2. Leave **Five samplers: DCS-RWM, DCS-BW, SCS, reflected BW and DCS-Billiard (JAX)** and
   **Quick check** selected. Choose a **Target distribution** and whether to
   use **Variational map fitting — both DCS and SCS**. Choose **CPU workers**
   (start with 2–4), then click **Start run**. Quick-check graphs verify
   execution; they are not publication results.
3. After the quick check succeeds, choose **Smaller reproduction** or **Full
   paper settings** and click **Use all CPU threads**, then **Start run** again.
   This selects the logical CPU capacity available to the process. Reduce the
   worker count if memory is limited; more workers do not guarantee faster runs.
4. After completion, links to figures and tables appear automatically below the
   progress log. Click a PNG or PDF link to view a graph. All files remain in
   **results → click-runs → dated folder**. Under **Exact numbers and tables**,
   open `paper_values.html` to read estimates, MSE, sample counts and timing
   values. The page uses two decimals; its CSV downloads retain full precision.

The two current comparisons contain the same five methods: DCS-RWM, DCS-BW,
SCS, DCS-BW-reflect and DCS-Billiard. The JAX option runs all five in the same framework;
the Numba option uses optimized native DCS loops against the unchanged author
SCS JAX transition. Both offer 100D/500D and VI on/off. Rotation mixtures
are no longer active choices.

For these two comparisons, **Full paper settings** selects the full benchmark
budget: 50 chains per method, 20.00 seconds per chain and 1,500 separate
discarded warmup transitions. The preset name does not imply an exact reproduction
of previously published figures; the saved protocol version defines the
experiment. Both modes report mean norm, radial-tail
probability, MSE, exact counts, time, functional ESS and ESS per second.
Coordinate measurements and their long reference simulation are omitted;
radial references are analytic. The original radial figures remain, with a new
ESS figure and `functional_ess.csv`. Displayed numbers use two decimals.

VI off keeps unit scale, zero centre and zero deformation. VI on fits both
map families: SCS uses its authors' fixed-cloud VI and DCS retains its
fresh-batch fit. These are different fitting recipes and full fitting budgets;
the recorded fitting costs are outside production time. Billiards retain the fitted
centre/R/β and use zero production deformation, as explained in
[the construction](BILLIARD.md). Billiards tune their own moves using a
bounded radial-movement score; RWM/BW retain acceptance-target 0.234 tuning,
and reflected BW retains ball-movement pilots with no geometric ceiling of 2.
SCS defaults to the authors' fixed h=0.1; acceptance tuning toward 0.234
remains available as a sensitivity option.
Both SCS step rules receive the same discarded warmup-transition budget; the fixed setting
does not adapt. The [SCS study](AUTHOR_SCS_CALIBRATION.md) documents the
default-selection rule and sensitivity results. All production steps are
frozen. Billiard trials and scores are saved in `radial_tuning.csv`.
The first third of production is discarded in every case.

SCS uses its authors' original map-fitting routine in the other experiments
too, including logistic regression and the 50D infinite-first-moment example.
Their SCS sampling implementation and acceptance-target 0.234 step tuning
remain unchanged. Run those experiments again in fresh folders to obtain
figures with the updated SCS fits. The current 100D/500D comparisons already
used author fitting. See [the rerun guide](RERUN.md#scs-fitting-update).

ESS is computed after sampling from unthinned scalar histories, including
rejections. The log reports this stage. FFT work is limited to one chain at a
time to keep memory bounded; diagnostic time is separate. Temporary unthinned
histories are discarded after diagnostics. An unavailable ESS has a reason;
it is not replaced by the iteration count. See [ESS interpretation](ESS.md).

Use a fresh seed batch to check repeatability conditional on the fitted maps.
The same batch reuses random streams, but time stopping can change counts.
Run each target/VI combination separately and rerun all five methods together.
New version-9 runs do not change historical saved reports. Smaller runs use
abbreviated VI; full VI fitting can change the performance ordering.

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
