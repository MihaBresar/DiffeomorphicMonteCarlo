"""Fast, dependency-light checks for the public repository contract.

These tests intentionally do not execute a scientific experiment.  Full runs
need JAX and may consume substantial CPU time; their numerical validation is
tracked separately in docs/VALIDATION.md.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiment_registry import EXPERIMENTS
import run_experiment
from run_experiment import resolved_reference_batch_size


ROOT = Path(__file__).resolve().parents[1]
GARCH_DATA_SHA256 = "0690040d698496babaa4e5727b08fb978339fb87d91c4a2d9cfe76ab363872fc"


class RepositoryContractTests(unittest.TestCase):
    def test_experiment_modules_are_syntax_valid(self) -> None:
        for path in sorted((ROOT / "experiments").glob("*.py")):
            with self.subTest(path=path.name):
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_registry_modules_and_convenience_entry_points_exist(self) -> None:
        expected_entry_points = {
            "eight-schools": "run_eight_schools.py",
            "garch11": "run_garch11.py",
            "logistic-regression": "run_logistic_regression.py",
            "skew-t-50d-nu-0.5": "run_skew_t_50d_nu_0_5.py",
            "skew-t-100d-fixed": "run_skew_t_100d_fixed_iterations.py",
            "skew-t-200d-nuts": "run_skew_t_200d_pbw_vs_nuts.py",
            "skew-t-100d-baselines": "run_skew_t_100d_baselines.py",
        }
        self.assertEqual(
            {entry.identifier for entry in EXPERIMENTS},
            set(expected_entry_points),
        )
        for entry in EXPERIMENTS:
            with self.subTest(identifier=entry.identifier):
                module_path = ROOT / (entry.module.replace(".", "/") + ".py")
                self.assertTrue(module_path.is_file())
                self.assertTrue(
                    (ROOT / expected_entry_points[entry.identifier]).is_file()
                )

                module_tree = ast.parse(module_path.read_text(encoding="utf-8"))
                identifiers = [
                    node.value.value
                    for node in module_tree.body
                    if isinstance(node, ast.Assign)
                    and any(
                        isinstance(target, ast.Name) and target.id == "EXPERIMENT_ID"
                        for target in node.targets
                    )
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)
                ]
                self.assertEqual(identifiers, [entry.identifier])

    def test_presets_are_well_formed(self) -> None:
        payload = json.loads((ROOT / "configs" / "presets.json").read_text(encoding="utf-8"))
        self.assertEqual(set(payload), {"quick", "paper"})
        self.assertEqual(payload["paper"], {})
        self.assertGreater(payload["quick"]["HEAVY_TAILS_NUM_RUNS"], 0)
        self.assertGreater(payload["quick"]["HEAVY_TAILS_REFERENCE_SAMPLES"], 0)

    def test_bundled_garch_data_are_well_formed(self) -> None:
        data_path = ROOT / "data" / "raw" / "garch11.json"
        payload = json.loads(data_path.read_text(encoding="utf-8"))
        self.assertEqual(set(payload), {"T", "sigma1", "y"})
        self.assertEqual(payload["T"], len(payload["y"]))
        self.assertEqual(payload["T"], 200)
        self.assertGreater(payload["sigma1"], 0)
        self.assertEqual(hashlib.sha256(data_path.read_bytes()).hexdigest(), GARCH_DATA_SHA256)

    def test_public_cli_lists_registered_experiments(self) -> None:
        completed = subprocess.run(
            [sys.executable, "run_experiment.py", "--list"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        for entry in EXPERIMENTS:
            self.assertIn(entry.identifier, completed.stdout)

    def test_dry_run_does_not_create_an_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory) / "would-be-output"
            subprocess.run(
                [
                    sys.executable,
                    "run_experiment.py",
                    "eight-schools",
                    "--preset",
                    "quick",
                    "--dry-run",
                    "--output-dir",
                    str(output_directory),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertFalse(output_directory.exists())

    def test_reference_override_selects_a_compatible_batch_size(self) -> None:
        self.assertEqual(resolved_reference_batch_size(12_000, "10_000"), 2_000)
        self.assertEqual(resolved_reference_batch_size(10_000, None), 10_000)
        completed = subprocess.run(
            [
                sys.executable,
                "run_experiment.py",
                "skew-t-50d-nu-0.5",
                "--preset",
                "quick",
                "--reference-samples",
                "12000",
                "--dry-run",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("Reference draws/batch: 12,000/2,000", completed.stdout)

    def test_unsupported_overrides_are_rejected(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "run_experiment.py",
                "eight-schools",
                "--time-budget",
                "1",
                "--dry-run",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("not supported", completed.stderr)

    def test_show_does_not_force_a_headless_matplotlib_backend(self) -> None:
        captured_environment: dict[str, str] = {}

        class FakeProcess:
            def __init__(self, *_args, **kwargs) -> None:
                captured_environment.update(kwargs["env"])
                self.stdout: list[str] = []

            def wait(self) -> int:
                return 0

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory) / "output"
            with (
                patch.dict(os.environ, {}, clear=True),
                patch.object(
                    sys,
                    "argv",
                    [
                        "run_experiment.py",
                        "eight-schools",
                        "--show",
                        "--output-dir",
                        str(output_directory),
                    ],
                ),
                patch.object(run_experiment.subprocess, "Popen", FakeProcess),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(run_experiment.main(), 0)

        self.assertNotIn("MPLBACKEND", captured_environment)


if __name__ == "__main__":
    unittest.main()
