# DCS versus original SCS

Six methods are compared on 100D and 500D normal-CDF-skewed Student-t targets with ν=3: DCS-RWM, DCS-BW, original SCS, SCS calibrated (zero), DCS-BW-reflect and DCS-Billiard.

The launcher offers **DCS vs original SCS** and **Numba extension**. Their command identifiers are `reported-zero-centre` and `reported-zero-centre-numba`. Scientific protocol identifiers remain `timed-reported-zero-centre-v3` and `timed-reported-zero-centre-numba-v3`.

## Running

Open `Start.bat` on Windows, `Start.command` on macOS, or `START_HERE.py` in IDLE. Set up the environment, choose comparison, target, preset, seed and workers, then start.

| Preset | Chains per method | Production seconds per chain | Separate warmup |
|---|---:|---:|---:|
| Quick check | 2 | 0.05 | 1,500 |
| Smaller reproduction | 20 | 3.00 | 1,500 |
| Full paper settings | 50 | 20.00 | 1,500 |

Quick checks verify execution only. Every preset loads the same full-budget fits and production overrides; none refits VI or chooses a new step. Seed choices change sampling streams only. Exact seeds are saved.

```sh
python run_experiment.py reported-zero-centre --target skew-t-100d --preset reproduce --workers 2 --output-dir results/dcs-scs-100d
python run_experiment.py reported-zero-centre --target skew-t-500d --preset paper --workers 2 --output-dir results/dcs-scs-500d
python run_experiment.py reported-zero-centre-numba --target skew-t-100d --preset reproduce --workers 2 --output-dir results/numba-100d
```

## Target and parameters

The skew vector is α=(100,−100,0,…). Radial thresholds are 50 in 100D and 50√5 in 500D. The skew preserves the Student-t radial law, giving analytic mean-norm and radial-tail references. See [targets](TARGETS.md).

| Parameter | 100D | 500D |
|---|---:|---:|
| Shared DCS R | 0.6941299233184104 | 1.3882232419648068 |
| Shared DCS h | 0.025 | 0.044721359549995794 |
| Original SCS R | 1.1048295589441204 | 2.2313627251649697 |
| Original SCS h | 0.1 | 0.1 |
| SCS calibrated (zero) R | 0.5524147794720602 | 1.1156813625824848 |
| SCS calibrated (zero) h | 0.05 | 0.025 |

All DCS methods use β=2, centre μ=0 and Möbius deformation δ=0. BW, reflected BW and billiard use uniform-ball increments of radius h. RWM uses Gaussian coordinate standard deviation h/√(d+2), matching the increment's coordinate variance.

Original SCS retains its exact author-VI centre and observer. The additional control sets both to zero. Its latitude remains 1.1 and geometric sphere centre remains (0,…,0,1); its observer is not a DCS Möbius parameter. Both SCS variants use unchanged author projection and step-out transitions.

## Preparation provenance

Immutable assets in `studies/zero_centre/maps/` preserve fits, loss histories, checksums and parameter-selection records. Each family used 1,000 VI updates on its own fixed 2,000-point cloud: 2,000,000 objective-point evaluations. SCS uses the original author optimizer; DCS retains its own parameterization and gradient clipping. Equal counts do not make the fitting procedures identical.

R/h selection used 16 candidates × 4,096 common exact target starts × 8 transitions: **524,288 selection transitions per family and dimension**. Radius factors were (0.5,1,2,4) times the fitted radius. DCS tested h=(0.25,0.5,1,2)/√d; SCS tested h=(0.025,0.05,0.1,0.2). Selection maximized the smaller normalized lag-eight squared movement of radial CDF and tail indicator. Rejections remained included. Candidate records and tie rules are in `joint_calibration.json`.

These selections preceded production zero-centre choices. In particular, SCS R/h were selected with the **author-fitted centre and observer**, then retained after both were zeroed. There has been no constrained VI refit or R/h recalibration for that zero map. Original loss histories describe the source fits, not the overridden maps. Preset values are not claimed optimal.

Exact target draws and analytic radial information are available for this synthetic example. Selection-transition counts exclude VI, holdout assessment and other preparation. Equal counts do not imply equal elapsed cost or identical complete procedures. Original SCS is a separate author-default baseline without the selected R/h.

## Sampling and timing

Every method starts from the physical all-ones point and receives 1,500 discarded nonadaptive warmup transitions. Steps remain fixed. Exactly the first third of generated production iterations is excluded from estimation; rejected repeats remain observations.

Methods run in synchronized groups with the same worker allocation; method order rotates. Compilation, warmup, synchronization waits, ESS calculation and reporting are outside production time. Transitions, all-state radial recording and scalar capture for ESS are inside. Final 300-transition blocks can slightly exceed requested time; actual seconds and counts are exported.

JAX uses a common scan/recording framework. The Numba extension uses the same JAX warmup state and DCS step followed by fused Numba production. Both SCS variants remain author JAX implementations. Separate generators mean cross-backend trajectories need not match bitwise.

Reports contain estimates, MSE, bias/variance, counts, time, acceptance and functional ESS. Every post-burn-in scalar observation enters estimates and ESS; thinning controls only saved traces. See [ESS](ESS.md), [figures](FIGURES.md) and [timing](REPRODUCIBILITY.md).

## Interpretation

Results concern the specified target, preparations, functions and budgets. Different map preparation means the comparison does not isolate proposal geometry. Radial performance does not establish angular or coordinate-marginal mixing.

The pilot score uses one lag and bounded radial functions, not integrated autocorrelation, wall-clock MSE or raw norm directly. For ν=3, norm fourth moments do not exist; a few chains can dominate empirical MSE. Retain complete method sets and all chains.

This common normal-CDF-skewed target is not the SCS paper's Student-t-CDF target or a reproduction of its figures. See [source attribution](../third_party/scs_paper/README.md).

The supplied inputs make fresh sampling reproducible; complete preparation
replay is not yet packaged. The pilot source and centre diagnostic are
identified by hashes rather than standalone public executables. See the
[reproducibility scope](REPRODUCIBILITY.md) for this limitation and the
implementation and method-order qualifications on timing comparisons.
