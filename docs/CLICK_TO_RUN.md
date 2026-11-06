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
   **Quick check** selected. Choose a **Target distribution**; both map families
   are fitted by VI automatically. Leave **Shared DCS move h** at its default
   0.07 for the initial check. Choose **CPU workers**
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
SCS JAX transition. Both offer 100D/500D and always fit both maps by VI. Rotation mixtures
are no longer active choices.

To compare another fixed move without commands, change **Shared DCS move h**
before clicking **Start run**. Enter **0.15** for the larger-move comparison;
the field also accepts values such as **0.10** and **0.025**. Every DCS method
receives the same chosen h: BW, reflected BW and billiard use that increment
radius, while Gaussian RWM uses coordinate standard deviation h/√(d+2).
SCS stays at h=0.1. These are sensitivity choices, not demonstrated optima.
The chosen value is saved in `launcher_request.json` and the resolved run
configuration. For **All seven**, it changes only the five-method comparison;
the other six experiments retain their specified step rules. Changing this
field starts a new experiment; rebuilding an older report uses its saved
settings and cannot change its samples.

For these two comparisons, **Full paper settings** selects the full benchmark
budget: 50 chains per method, 20.00 seconds per chain and 1,500 separate
discarded warmup transitions. The preset name does not imply an exact reproduction
of previously published figures; the saved protocol version defines the
experiment. Both modes report mean norm, radial-tail
probability, MSE, exact counts, time, functional ESS and ESS per second.
Coordinate measurements and their long reference simulation are omitted;
radial references are analytic. The original radial figures remain, with a new
ESS figure and `functional_ess.csv`. Displayed numbers use two decimals.

Every new comparison fits both map families: SCS uses its authors' fixed-cloud VI and DCS retains its
fixed-cloud fit. Both full fits use 1,000 updates on their own 2,000-point fixed clouds. Update counts, cloud size and evaluation counts match; reference distributions, parameterizations and clipping rules differ;
the recorded fitting costs are outside production time. Billiards retain the fitted
centre/R/β and use zero production deformation, as explained in
[the construction](BILLIARD.md). All four DCS variants share the selected h
(default 0.07):
BW/reflected-BW/billiard use a uniform-ball increment of that radius and
RWM uses coordinate standard deviation h/√(d+2). Each uses its own kernel
during discarded equilibration, without separate step adaptation.
The single SCS baseline uses the authors' fixed h=0.1 with no adaptation
and the same discarded warmup-transition budget as DCS. New five-method
comparisons do not offer alternative SCS tuning or recentering. The [SCS study](AUTHOR_SCS_CALIBRATION.md) documents the
default-selection rule and sensitivity results. All production steps are
frozen. Historical adaptive controls retain their recorded pilot histories.
The first third of production is discarded in every case.

SCS uses its authors' original map-fitting routine in the other experiments
too, including logistic regression and the 50D infinite-first-moment example.
Their SCS sampling implementation and acceptance-target 0.234 step tuning
remain unchanged. Run those experiments again in fresh folders to obtain
figures with the updated SCS fits. The current 100D/500D comparisons already
used author fitting. See [the rerun guide](RERUN.md#earlier-version-10-fitting-update).

ESS is computed after sampling from unthinned scalar histories, including
rejections. The log reports this stage. FFT work is limited to one chain at a
time to keep memory bounded; diagnostic time is separate. Temporary unthinned
histories are discarded after diagnostics. An unavailable ESS has a reason;
it is not replaced by the iteration count. See [ESS interpretation](ESS.md).

Use **Repeat run** to choose **Run 1 (default)** through **Run 9**. The same
choice reuses the sampling random streams, but time stopping can change counts.
A different choice changes only those sampling streams; the algorithms, fitting
rules and fitting seeds stay the same.
Run each target separately and rerun all five methods together. VI fitting is mandatory.
Version 11 changes all four DCS step policies and requires new sampling
of all five methods together at every preset. It keeps version 10's matched
fixed-cloud VI budgets and fitted maps. Saved reports retain their original
policy. Smaller runs use abbreviated VI; full fitting can change the performance
ordering. **Shared DCS move h** provides the same setting as the command-line
`--dcs-fixed-move` option. A value of 0.15 is the declared larger-move
comparison; neither it nor the default is claimed optimal. Fresh-cloud
fitting experiments do not change the public defaults.

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
