"""Tests for Hydra CLI integration."""

import unittest
import tempfile
import os
import subprocess
import sys
import pandas as pd
import numpy as np
from pathlib import Path


class TestCLI(unittest.TestCase):
    def setUp(self):
        """Create test data."""
        self.temp_dir = tempfile.mkdtemp()

        # Create test CSV
        np.random.seed(42)
        n_samples = 20
        trials = 5

        data = []
        for i in range(n_samples):
            for t in range(trials):
                data.append(
                    {
                        "chamfer_l1": np.random.exponential(0.05),
                        "chamfer_l2": np.random.exponential(0.02),
                        "iou": np.random.beta(8, 2),
                        "f1": np.random.beta(10, 3),
                    }
                )

        self.df = pd.DataFrame(data)
        self.test_file = os.path.join(self.temp_dir, "test_metrics.csv")
        self.df.to_csv(self.test_file, index=False)

    def tearDown(self):
        """Clean up."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_cli_basic_usage(self):
        """Test basic CLI usage with required arguments."""
        cmd = [
            sys.executable,
            "-m",
            "lidra.metrics.analysis.v3.cli",
            f"input_file={self.test_file}",
            "metrics=[chamfer_l1,iou]",
            "trials_per_sample=5",
            "hydra.run.dir=" + self.temp_dir,  # Prevent hydra from creating outputs
            "hydra.job.chdir=false",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Check successful execution
        self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")

        # Check output contains expected elements
        self.assertIn("chamfer_l1", result.stdout)
        self.assertIn("iou", result.stdout)
        self.assertIn("Mean", result.stdout)

    def test_cli_with_thresholds(self):
        """Test CLI with threshold configuration."""
        cmd = [
            sys.executable,
            "-m",
            "lidra.metrics.analysis.v3.cli",
            f"input_file={self.test_file}",
            "metrics=[iou]",
            "+thresholds.iou=[0.5,0.7,0.9]",  # Need + prefix to add new key
            "trials_per_sample=5",
            "hydra.run.dir=" + self.temp_dir,
            "hydra.job.chdir=false",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        self.assertEqual(result.returncode, 0)
        self.assertIn("iou_at_0.5", result.stdout)
        self.assertIn("iou_at_0.7", result.stdout)
        self.assertIn("iou_at_0.9", result.stdout)

    def test_cli_csv_output(self):
        """Test CLI with CSV output to file."""
        output_file = os.path.join(self.temp_dir, "output.csv")

        cmd = [
            sys.executable,
            "-m",
            "lidra.metrics.analysis.v3.cli",
            f"input_file={self.test_file}",
            "metrics=[chamfer_l1,chamfer_l2]",
            "trials_per_sample=5",
            "output_format=csv",
            f"output_file={output_file}",
            "hydra.run.dir=" + self.temp_dir,
            "hydra.job.chdir=false",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        self.assertEqual(result.returncode, 0)
        self.assertTrue(os.path.exists(output_file))

        # Verify CSV content
        df = pd.read_csv(output_file)
        self.assertIn("metric", df.columns)
        self.assertIn("mean", df.columns)
        metrics = df["metric"].tolist()
        self.assertIn("chamfer_l1", metrics)
        self.assertIn("chamfer_l2", metrics)

    def test_cli_preset_config(self):
        """Test CLI with preset configuration."""
        # Only test if we have the shape metrics in our test data
        if "f1" in self.df.columns and "chamfer_distance" in self.df.columns:
            cmd = [
                sys.executable,
                "-m",
                "lidra.metrics.analysis.v3.cli",
                "--config-name",
                "presets/shape_analysis",
                f"input_file={self.test_file}",
                "trials_per_sample=5",
                "hydra.run.dir=" + self.temp_dir,
                "hydra.job.chdir=false",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            self.assertEqual(result.returncode, 0)
            self.assertIn("f1", result.stdout)
            self.assertIn("chamfer_distance", result.stdout)

    def test_cli_error_handling(self):
        """Test CLI error handling for missing required args."""
        cmd = [
            sys.executable,
            "-m",
            "lidra.metrics.analysis.v3.cli",
            # Missing required input_file
            "metrics=[chamfer_l1]",
            "hydra.run.dir=" + self.temp_dir,
            "hydra.job.chdir=false",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Should fail due to missing required field
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Error", result.stderr)

    def test_cli_different_modes(self):
        """Test different reduction modes."""
        for mode in ["mean", "min", "max"]:
            cmd = [
                sys.executable,
                "-m",
                "lidra.metrics.analysis.v3.cli",
                f"input_file={self.test_file}",
                "metrics=[chamfer_l1]",
                f"mode={mode}",
                "trials_per_sample=5",
                "hydra.run.dir=" + self.temp_dir,
                "hydra.job.chdir=false",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            self.assertEqual(result.returncode, 0, f"Failed for mode {mode}")
            self.assertIn(f"mode: {mode}", result.stdout)


if __name__ == "__main__":
    unittest.main()
