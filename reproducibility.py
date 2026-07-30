"""Small, dependency-light helpers shared by the experiment scripts.

The research implementations deliberately remain readable, standalone modules.
This file supplies the pieces that make a run inspectable outside the original
development machine: validated environment overrides, non-interactive figure
output, JSON summaries, runtime metadata, and a multiprocessing-safe entry
point.

Configuration is passed through ``HEAVY_TAILS_*`` environment variables so a
worker created with ``spawn`` observes exactly the same settings as its parent.
The public ``run_experiment.py`` command is the preferred interface.
"""

from __future__ import annotations

import json
import os
import platform
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping


ENV_PREFIX = "HEAVY_TAILS_"


def env_int(name: str, default: int, *, minimum: int | None = 1) -> int:
    """Read a positive integer override, with a useful error on bad input."""
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}.") from exc
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {parsed}.")
    return parsed


def env_float(name: str, default: float, *, minimum: float | None = 0.0) -> float:
    """Read a floating-point override, with a useful error on bad input."""
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {value!r}.") from exc
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {parsed}.")
    return parsed


def env_flag(name: str, default: bool = False) -> bool:
    """Read a conventional true/false environment flag."""
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean flag, got {value!r}.")


def worker_count(num_runs: int) -> int:
    """Return the requested worker count, capped at the number of chains."""
    available = os.cpu_count() or 1
    requested = env_int(f"{ENV_PREFIX}WORKERS", available)
    return max(1, min(requested, available, num_runs))


def output_directory() -> Path | None:
    """Return the run directory requested by the public runner, if any."""
    value = os.environ.get(f"{ENV_PREFIX}OUTPUT_DIR")
    return Path(value).expanduser().resolve() if value else None


def _json_default(value: Any) -> Any:
    """Convert common numerical objects without making NumPy a dependency here."""
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _jax_metadata() -> dict[str, Any]:
    """Record the active JAX backend when it is installed and importable."""
    try:
        import jax
        import jaxlib

        return {
            "jax_version": getattr(jax, "__version__", "unknown"),
            "jaxlib_version": getattr(jaxlib, "__version__", "unknown"),
            "devices": [str(device) for device in jax.devices()],
            "default_backend": jax.default_backend(),
        }
    except Exception as exc:  # Metadata must not make a successful run fail.
        return {"unavailable": f"{type(exc).__name__}: {exc}"}


def _package_versions() -> dict[str, str]:
    """Record the scientific stack without requiring every package to import."""
    packages = ("numpy", "scipy", "matplotlib", "jax", "jaxlib", "numpyro", "blackjax", "arviz")
    discovered: dict[str, str] = {}
    for package in packages:
        try:
            discovered[package] = version(package)
        except PackageNotFoundError:
            discovered[package] = "not installed"
    return discovered


def _environment_overrides() -> dict[str, str]:
    return {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith(ENV_PREFIX)
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def finalize_experiment(
    experiment_id: str,
    figures: Mapping[str, Any],
    summary: Mapping[str, Any],
    configuration: Mapping[str, Any],
) -> Path | None:
    """Save figures and run metadata, then optionally display Matplotlib figures.

    The helper intentionally stores only aggregate summaries.  Full chains can
    be very large and should be added by a benchmark only when their release
    policy and storage requirements are explicit.
    """
    target = output_directory()
    if target is not None:
        figures_directory = target / "figures"
        figures_directory.mkdir(parents=True, exist_ok=True)
        for stem, figure in figures.items():
            figure.savefig(
                figures_directory / f"{stem}.png",
                dpi=300,
                bbox_inches="tight",
            )
            figure.savefig(
                figures_directory / f"{stem}.pdf",
                bbox_inches="tight",
            )

        write_json(target / "summary.json", dict(summary))
        write_json(target / "configuration.json", dict(configuration))
        write_json(
            target / "manifest.json",
            {
                "experiment_id": experiment_id,
                "created_at_utc": datetime.now(UTC).isoformat(),
                "module_command": sys.argv,
                "runner_command": os.environ.get(f"{ENV_PREFIX}RUNNER_COMMAND"),
                "runner_preset": os.environ.get(f"{ENV_PREFIX}RUNNER_PRESET"),
                "python": sys.version,
                "platform": platform.platform(),
                "processor": platform.processor(),
                "environment_overrides": _environment_overrides(),
                "package_versions": _package_versions(),
                "jax": _jax_metadata(),
            },
        )
        print(f"Saved figures and run metadata to {target}")

    if env_flag(f"{ENV_PREFIX}SHOW", default=target is None):
        import matplotlib.pyplot as plt

        plt.show()
    else:
        import matplotlib.pyplot as plt

        plt.close("all")

    return target


def configure_multiprocessing() -> None:
    """Use spawn, which is safer than fork with JAX accelerator runtimes."""
    import multiprocessing

    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        # Another library or an interactive interpreter has already selected it.
        pass
