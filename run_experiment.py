#!/usr/bin/env python3
"""Run an experiment by its stable, descriptive public identifier.

Examples
--------
``python run_experiment.py --list``
``python run_experiment.py skew-t-100d-baselines --preset quick``
``python run_experiment.py eight-schools --preset paper --output-dir results/eight-schools/paper``
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from experiment_registry import BY_IDENTIFIER, EXPERIMENTS


ROOT = Path(__file__).resolve().parent
PRESETS_PATH = ROOT / "configs" / "presets.json"

SYNTHETIC_REFERENCE_EXPERIMENTS = frozenset(
    {
        "skew-t-50d-nu-0.5",
        "skew-t-100d-fixed",
        "skew-t-200d-nuts",
        "skew-t-100d-baselines",
    }
)

OPTION_SUPPORT = {
    "--time-budget": frozenset(
        {
            "logistic-regression",
            "skew-t-50d-nu-0.5",
            "skew-t-200d-nuts",
            "skew-t-100d-baselines",
        }
    ),
    "--iterations": frozenset({"skew-t-100d-fixed"}),
    "--reference-samples": SYNTHETIC_REFERENCE_EXPERIMENTS,
    "--reference-batch-size": SYNTHETIC_REFERENCE_EXPERIMENTS,
    "--optimizer-steps": frozenset(
        {
            "logistic-regression",
            "skew-t-50d-nu-0.5",
            "skew-t-100d-fixed",
            "skew-t-200d-nuts",
            "skew-t-100d-baselines",
        }
    ),
}


def positive_int(value: str) -> int:
    """Parse a strictly positive integer CLI value."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {parsed}")
    return parsed


def positive_float(value: str) -> float:
    """Parse a strictly positive floating-point CLI value."""
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a number, got {value!r}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive number, got {parsed}")
    return parsed


def load_presets() -> dict[str, dict[str, str]]:
    with PRESETS_PATH.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return {name: {key: str(value) for key, value in values.items()} for name, values in payload.items()}


def list_experiments() -> None:
    print("Available experiments:\n")
    for experiment in EXPERIMENTS:
        print(f"  {experiment.identifier:<25} {experiment.title}")
        print(f"    {experiment.description}")
        print(f"    Paper profile: {experiment.paper_scale}")
        if experiment.notes:
            print(f"    Note: {experiment.notes}")
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", nargs="?", choices=sorted(BY_IDENTIFIER))
    parser.add_argument("--list", action="store_true", help="List experiment identifiers and exit.")
    parser.add_argument(
        "--preset",
        choices=("quick", "paper"),
        default="quick",
        help="Quick is a smoke profile; paper preserves each script's original defaults.",
    )
    parser.add_argument("--output-dir", type=Path, help="Directory for figures, summaries, metadata, and run.log.")
    parser.add_argument("--workers", type=positive_int, help="Maximum parallel worker processes.")
    parser.add_argument("--runs", type=positive_int, help="Override the number of chains per sampler.")
    parser.add_argument("--time-budget", type=positive_float, help="Override the per-chain timed sampling budget in seconds.")
    parser.add_argument("--iterations", type=positive_int, help="Override a fixed-iteration chain length (fixed-iteration experiment only).")
    parser.add_argument("--reference-samples", type=positive_int, help="Override the synthetic Monte Carlo reference size (synthetic skew-t experiments only).")
    parser.add_argument("--reference-batch-size", type=positive_int, help="Batch size for --reference-samples; it must divide that value.")
    parser.add_argument("--optimizer-steps", type=positive_int, help="Override variational/adaptation optimization steps where supported.")
    parser.add_argument("--show", action="store_true", help="Display figures after the run as well as saving them.")
    parser.add_argument("--dry-run", action="store_true", help="Print the derived command and output location without creating files.")
    return parser.parse_args()


def default_output_directory(experiment_id: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return ROOT / "results" / experiment_id / stamp


def validate_overrides(args: argparse.Namespace, experiment_id: str) -> None:
    """Reject flags that an experiment would otherwise silently ignore."""
    supplied = {
        "--time-budget": args.time_budget,
        "--iterations": args.iterations,
        "--reference-samples": args.reference_samples,
        "--reference-batch-size": args.reference_batch_size,
        "--optimizer-steps": args.optimizer_steps,
    }
    unsupported = [
        flag
        for flag, value in supplied.items()
        if value is not None and experiment_id not in OPTION_SUPPORT[flag]
    ]
    if unsupported:
        choices = ", ".join(sorted(unsupported))
        raise SystemExit(f"{choices} is not supported by experiment {experiment_id!r}.")
    if args.reference_batch_size is not None and args.reference_samples is None:
        raise SystemExit("--reference-batch-size must be supplied with --reference-samples.")
    if (
        args.reference_samples is not None
        and args.reference_batch_size is not None
        and args.reference_samples % args.reference_batch_size
    ):
        raise SystemExit("--reference-batch-size must evenly divide --reference-samples.")


def resolved_reference_batch_size(
    reference_samples: int,
    configured_batch_size: str | None,
) -> int:
    """Pick a compatible batch size when only the total reference size changes."""
    if configured_batch_size is None:
        return reference_samples
    return math.gcd(reference_samples, positive_int(configured_batch_size))


def main() -> int:
    args = parse_args()
    if args.list:
        list_experiments()
        return 0
    if args.experiment is None:
        raise SystemExit("Specify an experiment identifier or use --list.")

    presets = load_presets()
    experiment = BY_IDENTIFIER[args.experiment]
    validate_overrides(args, experiment.identifier)
    environment = os.environ.copy()
    environment.update(presets[args.preset])

    outer_command = shlex.join([sys.executable, str(ROOT / "run_experiment.py"), *sys.argv[1:]])
    environment["HEAVY_TAILS_RUNNER_COMMAND"] = outer_command
    environment["HEAVY_TAILS_RUNNER_PRESET"] = args.preset

    output_directory = (args.output_dir or default_output_directory(experiment.identifier)).resolve()
    environment["HEAVY_TAILS_OUTPUT_DIR"] = str(output_directory)
    environment["HEAVY_TAILS_SHOW"] = "1" if args.show else "0"
    if not args.show:
        environment.setdefault("MPLBACKEND", "Agg")

    overrides = {
        "HEAVY_TAILS_WORKERS": args.workers,
        "HEAVY_TAILS_NUM_RUNS": args.runs,
        "HEAVY_TAILS_TIME_BUDGET": args.time_budget,
        "HEAVY_TAILS_NUM_ITERS": args.iterations,
        "HEAVY_TAILS_REFERENCE_SAMPLES": args.reference_samples,
        "HEAVY_TAILS_OPTIMIZATION_STEPS": args.optimizer_steps,
    }
    for key, value in overrides.items():
        if value is not None:
            environment[key] = str(value)

    if args.reference_samples is not None:
        reference_batch_size = args.reference_batch_size or resolved_reference_batch_size(
            args.reference_samples,
            environment.get("HEAVY_TAILS_REFERENCE_BATCH_SIZE"),
        )
        environment["HEAVY_TAILS_REFERENCE_BATCH_SIZE"] = str(reference_batch_size)

    command = [sys.executable, "-m", experiment.module]
    print(f"Experiment: {experiment.title}")
    print(f"Preset: {args.preset}")
    print(f"Output: {output_directory}")
    print("Command:", " ".join(command))
    if args.reference_samples is not None:
        print(f"Reference draws/batch: {args.reference_samples:,}/{reference_batch_size:,}")
    if args.dry_run:
        return 0

    output_directory.mkdir(parents=True, exist_ok=True)
    # Sandboxed/cluster environments often have no writable home-directory
    # Matplotlib cache.  Keep any cache local to the ignored run directory.
    environment.setdefault("MPLCONFIGDIR", str(output_directory / ".matplotlib"))

    with (output_directory / "run.log").open("w", encoding="utf-8") as log:
        log.write(f"Runner command: {outer_command}\n")
        log.write(f"Preset: {args.preset}\n")
        log.write(f"Module command: {shlex.join(command)}\n\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        return_code = process.wait()

    if return_code:
        print(f"Experiment failed; inspect {output_directory / 'run.log'}", file=sys.stderr)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
