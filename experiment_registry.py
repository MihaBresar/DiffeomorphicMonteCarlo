"""Canonical public names and lightweight metadata for the experiment suite."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Experiment:
    identifier: str
    module: str
    title: str
    description: str
    paper_scale: str
    notes: str = ""


EXPERIMENTS: tuple[Experiment, ...] = (
    Experiment(
        "eight-schools",
        "experiments.eight_schools",
        "Eight Schools centered funnel",
        "NumPyro NUTS versus the projected slice/HnR implementation.",
        "50 chains; 150,000 post-warmup NUTS draws per chain.",
        "Uses hard-coded PosteriorDB summary statistics.",
    ),
    Experiment(
        "garch11",
        "experiments.garch11",
        "PosteriorDB GARCH(1,1)",
        "NumPyro NUTS versus the projected slice/HnR implementation.",
        "50 chains; 150,000 post-warmup NUTS draws per chain.",
        "The bundled data are sourced from PosteriorDB; see data/README.md.",
    ),
    Experiment(
        "logistic-regression",
        "experiments.logistic_regression_student_t",
        "Robust logistic regression",
        "Student-t-prior logistic regression at ν=4 and ν=6; DCS-BW/PBW versus SCS.",
        "20 chains per sampler and ν; 20-second per-chain budget.",
        "The plotted pooled-sampler Q–Q diagnostic is not an independent reference.",
    ),
    Experiment(
        "skew-t-50d-nu-0.5",
        "experiments.skew_t_50d_nu_0_5",
        "Skew-t, 50 dimensions, ν=0.5",
        "DCS-RWM, DCS-BW/PBW, and SCS at ν=0.5.",
        "50 chains; 10-second per-chain budget; 10M-draw reference estimate.",
    ),
    Experiment(
        "skew-t-100d-fixed",
        "experiments.skew_t_100d_fixed_iterations",
        "Skew-t, 100 dimensions, fixed iterations",
        "DCS-RWM, DCS-BW/PBW, and SCS at ν=2.",
        "20 chains; 500,000 iterations per chain; 10M-draw reference estimate.",
    ),
    Experiment(
        "skew-t-200d-nuts",
        "experiments.skew_t_200d_pbw_vs_nuts",
        "Skew-t, 200 dimensions: DCS-BW/PBW versus NUTS",
        "DCS-BW/PBW compared with BlackJAX NUTS at ν=3.",
        "50 chains; 20-second per-chain budget; 10M-draw reference estimate.",
    ),
    Experiment(
        "skew-t-100d-baselines",
        "experiments.skew_t_100d_pbw_rwm_nuts",
        "Skew-t, 100 dimensions: projection sampler and baselines",
        "DCS-BW/PBW, Gaussian RWM, symmetric stable RWM, and BlackJAX NUTS at ν=3.",
        "50 chains; 10-second per-chain budget; 10M-draw reference estimate.",
    ),
)


BY_IDENTIFIER = {experiment.identifier: experiment for experiment in EXPERIMENTS}
