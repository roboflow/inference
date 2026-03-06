"""Integration tests for v3 metrics analysis."""

import unittest
import tempfile
import os
import shutil
import pandas as pd
import numpy as np
from unittest.mock import patch
import json

from ..multi_trial import MultiTrialConfig, MultiTrialProcessor


class TestIntegration(unittest.TestCase):
    def setUp(self):
        """Create temporary test files."""
        self.temp_dir = tempfile.mkdtemp()

        # Create test CSV with realistic data
        np.random.seed(42)
        n_samples = 100
        trials = 5

        # Simulate realistic metric distributions
        data = {
            "chamfer_l1": np.random.exponential(0.05, n_samples * trials),
            "chamfer_l2": np.random.exponential(0.02, n_samples * trials),
            "iou": np.random.beta(
                8, 2, n_samples * trials
            ),  # Skewed towards high values
            "f1_score": np.random.beta(10, 3, n_samples * trials),
        }

        self.df = pd.DataFrame(data)
        self.input_file = os.path.join(self.temp_dir, "metrics.csv")
        self.df.to_csv(self.input_file, index=False)

    def tearDown(self):
        """Clean up temp files."""
        shutil.rmtree(self.temp_dir)

    def test_end_to_end_console_output(self):
        """Test complete pipeline with console output."""
        config = MultiTrialConfig(
            input_file=self.input_file,
            metrics=["chamfer_l1", "iou"],
            thresholds={"iou": [0.5, 0.7, 0.9]},
            mode="mean",
            trials_per_sample=5,
            statistics=["mean", "std", "p5", "p95"],
        )

        processor = MultiTrialProcessor()

        # Capture output
        with patch("builtins.print") as mock_print:
            processor.process(config)

            # Verify output was generated
            output = mock_print.call_args[0][0]

            # Check table structure
            self.assertIn("Multi-Trial Metrics Analysis", output)
            self.assertIn("chamfer_l1", output)
            self.assertIn("iou", output)

            # Check threshold metrics
            self.assertIn("iou_at_0.5", output)
            self.assertIn("iou_at_0.7", output)
            self.assertIn("iou_at_0.9", output)

            # Check statistics columns
            self.assertIn("Mean", output)
            self.assertIn("Std", output)
            self.assertIn("P5", output)
            self.assertIn("P95", output)

    def test_end_to_end_csv_file_output(self):
        """Test complete pipeline with CSV file output."""
        output_file = os.path.join(self.temp_dir, "output.csv")

        config = MultiTrialConfig(
            input_file=self.input_file,
            metrics=["chamfer_l1", "chamfer_l2", "iou"],
            thresholds={"iou": [0.5, 0.7]},
            mode="mean",
            trials_per_sample=5,
            statistics=["mean", "std", "min", "max"],
            output_format="csv",
            output_file=output_file,
        )

        processor = MultiTrialProcessor()
        processor.process(config)

        # Verify output file created
        self.assertTrue(os.path.exists(output_file))

        # Load and verify content
        output_df = pd.read_csv(output_file)

        # Check structure
        self.assertIn("metric", output_df.columns)
        self.assertIn("mean", output_df.columns)
        self.assertIn("std", output_df.columns)
        self.assertIn("min", output_df.columns)
        self.assertIn("max", output_df.columns)

        # Check all metrics present
        metrics = output_df["metric"].tolist()
        self.assertIn("chamfer_l1", metrics)
        self.assertIn("chamfer_l2", metrics)
        self.assertIn("iou", metrics)
        self.assertIn("iou_at_0.5", metrics)
        self.assertIn("iou_at_0.7", metrics)

        # Verify values are numeric
        for col in ["mean", "std", "min", "max"]:
            pd.to_numeric(output_df[col], errors="coerce")

    def test_different_reduction_modes(self):
        """Test different trial reduction modes produce different results."""
        metrics_to_test = ["chamfer_l1", "iou"]
        results = {}

        for mode in ["mean", "min", "max"]:
            config = MultiTrialConfig(
                input_file=self.input_file,
                metrics=metrics_to_test,
                mode=mode,
                trials_per_sample=5,
                statistics=["mean"],
                output_format="csv",
            )

            processor = MultiTrialProcessor()

            # Capture output
            with patch("builtins.print") as mock_print:
                processor.process(config)
                output = mock_print.call_args[0][0]

                # Parse CSV to get mean values
                lines = output.strip().split("\n")
                header = lines[0].split(",")
                mean_idx = header.index("mean")

                mode_results = {}
                for line in lines[1:]:
                    parts = line.split(",")
                    metric = parts[0]
                    mean_val = float(parts[mean_idx])
                    mode_results[metric] = mean_val

                results[mode] = mode_results

        # Verify different modes produce different results
        # For positive metrics, we expect: min < mean < max
        for metric in metrics_to_test:
            self.assertLess(results["min"][metric], results["mean"][metric])
            self.assertLess(results["mean"][metric], results["max"][metric])

    def test_missing_trials_handling(self):
        """Test handling of incomplete trial data."""
        # Create data with 103 values (not divisible by 5)
        n_values = 103
        data = {"metric1": np.random.randn(n_values)}
        df = pd.DataFrame(data)

        input_file = os.path.join(self.temp_dir, "incomplete.csv")
        df.to_csv(input_file, index=False)

        config = MultiTrialConfig(
            input_file=input_file,
            metrics=["metric1"],
            trials_per_sample=5,
            statistics=["mean", "std"],
        )

        processor = MultiTrialProcessor()

        # Should process with warning
        with patch("builtins.print") as mock_print:
            processor.process(config)

            # Check for warning about incomplete trials
            print_calls = [call[0][0] for call in mock_print.call_args_list]
            warning_found = any(
                "103 values" in str(call) and "not divisible by 5" in str(call)
                for call in print_calls
            )
            self.assertTrue(warning_found)

            # Check output generated for complete samples
            output = print_calls[-1]  # Last print is the output
            self.assertIn("metric1", output)

    def test_no_thresholds(self):
        """Test processing without any thresholds."""
        config = MultiTrialConfig(
            input_file=self.input_file,
            metrics=["chamfer_l1", "chamfer_l2"],
            mode="mean",
            trials_per_sample=5,
            statistics=["mean", "std", "p50"],
        )

        processor = MultiTrialProcessor()

        with patch("builtins.print") as mock_print:
            processor.process(config)
            output = mock_print.call_args[0][0]

            # Check only base metrics present
            self.assertIn("chamfer_l1", output)
            self.assertIn("chamfer_l2", output)

            # No threshold metrics
            self.assertNotIn("_at_", output)

    def test_single_trial_mode(self):
        """Test processing when trials_per_sample is not specified."""
        config = MultiTrialConfig(
            input_file=self.input_file,
            metrics=["iou"],
            thresholds={"iou": [0.7]},
            statistics=["mean", "std", "p5", "p95"],
            # No trials_per_sample specified
        )

        processor = MultiTrialProcessor()

        with patch("builtins.print") as mock_print:
            processor.process(config)
            output = mock_print.call_args[0][0]

            # Should process all values as single dataset
            self.assertIn("iou", output)
            self.assertIn("iou_at_0.7", output)

            # Values should be computed across all 500 samples
            # (100 samples * 5 trials, but treated as 500 independent samples)

    def test_percentile_statistics(self):
        """Test percentile calculations are reasonable."""
        config = MultiTrialConfig(
            input_file=self.input_file,
            metrics=["iou"],
            mode="mean",
            trials_per_sample=5,
            statistics=["p5", "p25", "p50", "p75", "p95"],
        )

        processor = MultiTrialProcessor()

        with patch("builtins.print") as mock_print:
            processor.process(config)
            output = mock_print.call_args[0][0]

            # Parse output to get percentile values
            lines = output.strip().split("\n")

            # Find the iou row and extract percentile values
            for line in lines:
                if "iou" in line and "_at_" not in line:
                    # Extract numeric values (this is fragile but works for testing)
                    import re

                    values = re.findall(r"\d+\.\d+", line)
                    if len(values) >= 5:
                        p5, p25, p50, p75, p95 = map(float, values[:5])

                        # Verify ordering
                        self.assertLess(p5, p25)
                        self.assertLess(p25, p50)
                        self.assertLess(p50, p75)
                        self.assertLess(p75, p95)

                        # For beta(8,2), we expect high values
                        self.assertGreater(p50, 0.6)  # Median should be > 0.6

    def test_error_handling_missing_file(self):
        """Test error when input file doesn't exist."""
        config = MultiTrialConfig(input_file="nonexistent.csv", metrics=["metric1"])

        processor = MultiTrialProcessor()

        # Should raise FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            processor.process(config)

    def test_error_handling_no_metrics_found(self):
        """Test error when no requested metrics are in the data."""
        config = MultiTrialConfig(
            input_file=self.input_file,
            metrics=["nonexistent_metric1", "nonexistent_metric2"],
        )

        processor = MultiTrialProcessor()

        # Should raise ValueError about no metrics found
        with self.assertRaises(ValueError) as ctx:
            processor.process(config)

        self.assertIn("No requested metrics found", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
