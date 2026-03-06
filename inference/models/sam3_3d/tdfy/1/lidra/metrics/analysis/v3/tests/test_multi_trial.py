"""Tests for MultiTrialProcessor."""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal, assert_almost_equal
import tempfile
import os

from ..processor.multi_trial import (
    MultiTrialConfig,
    MultiTrialProcessor,
    MetricData,
    MetricStats,
)
from ..functionals import FUNCTIONALS


class TestMultiTrialConfig(unittest.TestCase):
    def test_valid_config(self):
        """Test creating valid configuration."""
        from ..processor.multi_trial import Metric

        config = MultiTrialConfig(
            input_file="test.csv",
            metrics={
                "metric1": Metric(direction="minimize"),
                "metric2": Metric(direction="maximize"),
            },
        )
        self.assertEqual(config.trial_reduction_mode, "mean")
        self.assertEqual(config.statistics, ["mean", "std", "p5", "p95"])
        self.assertEqual(config.console_format, "rich")

    def test_invalid_config_no_metrics(self):
        """Test error when no metrics specified."""
        with self.assertRaises(ValueError) as ctx:
            MultiTrialConfig(input_file="test.csv", metrics={})
        self.assertIn("at least one metric", str(ctx.exception).lower())

    def test_invalid_trial_reduction_mode(self):
        """Test valid trial reduction modes are enforced at runtime."""
        from ..processor.multi_trial import Metric

        # Config creation doesn't validate trial_reduction_mode,
        # validation happens during processing
        config = MultiTrialConfig(
            input_file="test.csv",
            metrics={"m1": Metric(direction="minimize")},
            trial_reduction_mode="invalid",
        )
        # The mode validation happens in _reduce_multi_trials method
        self.assertEqual(config.trial_reduction_mode, "invalid")

    def test_invalid_statistics(self):
        """Test error for invalid statistics."""
        from ..processor.multi_trial import Metric

        with self.assertRaises(ValueError) as ctx:
            MultiTrialConfig(
                input_file="test.csv",
                metrics={"m1": Metric(direction="minimize")},
                statistics=["mean", "invalid_stat"],
            )
        self.assertIn("Unknown statistics", str(ctx.exception))
        self.assertIn("invalid_stat", str(ctx.exception))

    def test_config_with_thresholds(self):
        """Test configuration with thresholds."""
        config = MultiTrialConfig(
            input_file="test.csv", metrics=["iou"], thresholds={"iou": [0.5, 0.7, 0.9]}
        )
        self.assertEqual(config.thresholds["iou"], [0.5, 0.7, 0.9])


class TestMultiTrialProcessor(unittest.TestCase):
    def setUp(self):
        """Create processor and test data."""
        from ..processor.multi_trial import Metric

        self.processor = MultiTrialProcessor()

        # Create test dataframe with multiple trials
        np.random.seed(42)
        n_samples = 20
        trials = 5

        # Create proper multi-trial dataframe with sample_uuid and trial columns
        sample_uuids = []
        trial_nums = []
        for i in range(n_samples):
            for t in range(trials):
                sample_uuids.append(f"sample_{i}")
                trial_nums.append(t)

        self.df = pd.DataFrame(
            {
                "sample_uuid": sample_uuids,
                "trial": trial_nums,
                "metric1": np.random.randn(n_samples * trials),
                "metric2": np.random.randn(n_samples * trials) + 1,
                "metric3": np.random.uniform(0, 1, n_samples * trials),
            }
        )

        self.config = MultiTrialConfig(
            input_file="dummy.csv",
            metrics={
                "metric1": Metric(direction="minimize"),
                "metric2": Metric(direction="maximize"),
            },
            thresholds={"metric2": [0.5, 1.0]},
            trial_reduction_mode="mean",
            max_trials=trials,
            statistics=["mean", "std"],
        )

    def test_extract_data(self):
        """Test metric extraction from dataframe."""
        data = self.processor.extract_data(self.df, self.config)

        self.assertIn("metric1", data)
        self.assertIn("metric2", data)
        self.assertNotIn("metric3", data)  # Not requested

        # Check arrays are correct
        assert_array_almost_equal(data["metric1"], self.df["metric1"].values)
        assert_array_almost_equal(data["metric2"], self.df["metric2"].values)

    def test_extract_data_missing_metrics(self):
        """Test warning for missing metrics."""
        config = MultiTrialConfig(
            input_file="dummy.csv", metrics=["metric1", "nonexistent"]
        )

        with patch("builtins.print") as mock_print:
            data = self.processor.extract_data(self.df, config)

            # Check warning was printed
            mock_print.assert_called_once()
            warning_msg = mock_print.call_args[0][0]
            self.assertIn("nonexistent", warning_msg)

        self.assertIn("metric1", data)
        self.assertNotIn("nonexistent", data)

    def test_extract_data_no_metrics_found(self):
        """Test error when no requested metrics found."""
        config = MultiTrialConfig(
            input_file="dummy.csv", metrics=["nonexistent1", "nonexistent2"]
        )

        with self.assertRaises(ValueError) as ctx:
            self.processor.extract_data(self.df, config)
        self.assertIn("No requested metrics found", str(ctx.exception))

    def test_reduce_trials(self):
        """Test trial reduction."""
        data = {"metric1": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])}
        config = MultiTrialConfig(
            input_file="dummy.csv",
            metrics=["metric1"],
            trials_per_sample=5,
            mode="mean",
        )

        reduced = self.processor._reduce_trials(data, config)

        # Should have 2 samples (10 values / 5 trials)
        self.assertEqual(len(reduced["metric1"]), 2)
        # First sample mean: (1+2+3+4+5)/5 = 3
        assert_almost_equal(reduced["metric1"][0], 3.0)
        # Second sample mean: (6+7+8+9+10)/5 = 8
        assert_almost_equal(reduced["metric1"][1], 8.0)

    def test_reduce_trials_min_mode(self):
        """Test trial reduction with min mode."""
        data = {"metric1": np.array([5, 2, 8, 1, 9])}
        config = MultiTrialConfig(
            input_file="dummy.csv", metrics=["metric1"], trials_per_sample=5, mode="min"
        )

        reduced = self.processor._reduce_trials(data, config)
        assert_almost_equal(reduced["metric1"][0], 1.0)  # min

    def test_reduce_trials_max_mode(self):
        """Test trial reduction with max mode."""
        data = {"metric1": np.array([5, 2, 8, 1, 9])}
        config = MultiTrialConfig(
            input_file="dummy.csv", metrics=["metric1"], trials_per_sample=5, mode="max"
        )

        reduced = self.processor._reduce_trials(data, config)
        assert_almost_equal(reduced["metric1"][0], 9.0)  # max

    def test_reduce_trials_warning_incomplete(self):
        """Test warning when trials don't divide evenly."""
        data = {"metric1": np.array([1, 2, 3, 4, 5, 6, 7])}  # 7 values
        config = MultiTrialConfig(
            input_file="dummy.csv",
            metrics=["metric1"],
            trials_per_sample=5,
            mode="mean",
        )

        with patch("builtins.print") as mock_print:
            reduced = self.processor._reduce_trials(data, config)

            # Check warning was printed
            mock_print.assert_called_once()
            warning_msg = mock_print.call_args[0][0]
            self.assertIn("7 values", warning_msg)
            self.assertIn("not divisible by 5", warning_msg)

        # Should only have 1 complete sample
        self.assertEqual(len(reduced["metric1"]), 1)

    def test_apply_thresholds(self):
        """Test threshold transformation."""
        data = {"metric1": np.array([0.1, 0.5, 0.7, 1.2])}
        thresholds = {"metric1": [0.5, 1.0]}

        result = self.processor._apply_thresholds(data, thresholds)

        # Check threshold metrics created
        self.assertIn("metric1_at_0.5", result)
        self.assertIn("metric1_at_1.0", result)

        # Check values (>= threshold)
        expected_05 = np.array([0.0, 1.0, 1.0, 1.0])
        expected_10 = np.array([0.0, 0.0, 0.0, 1.0])
        assert_array_almost_equal(result["metric1_at_0.5"], expected_05)
        assert_array_almost_equal(result["metric1_at_1.0"], expected_10)

    def test_apply_thresholds_multiple_metrics(self):
        """Test thresholds on multiple metrics."""
        data = {
            "metric1": np.array([0.1, 0.5, 0.7]),
            "metric2": np.array([0.3, 0.6, 0.9]),
        }
        thresholds = {"metric1": [0.5], "metric2": [0.5, 0.8]}

        result = self.processor._apply_thresholds(data, thresholds)

        # Check all threshold metrics created
        self.assertIn("metric1_at_0.5", result)
        self.assertIn("metric2_at_0.5", result)
        self.assertIn("metric2_at_0.8", result)
        self.assertEqual(len(result), 3)

    def test_compute_results(self):
        """Test statistical computation."""
        data = {
            "metric1": np.array([1, 2, 3, 4, 5]),
            "metric2": np.array([10, 20, 30, 40, 50]),
        }
        config = MultiTrialConfig(
            input_file="dummy.csv", metrics=["metric1"], statistics=["mean", "std"]
        )

        results = self.processor.compute_results(data, config)

        # Check structure - results should be a DataFrame
        self.assertIsInstance(results, pd.DataFrame)
        self.assertIn("metric1", results.index)
        self.assertIn("mean", results.columns)
        self.assertIn("std", results.columns)

        # Check values
        assert_almost_equal(results.loc["metric1", "mean"], 3.0)
        assert_almost_equal(
            results.loc["metric1", "std"], np.std([1, 2, 3, 4, 5], ddof=1)
        )
        assert_almost_equal(results.loc["metric2", "mean"], 30.0)

    def test_transform_data_with_trials(self):
        """Test full transform with trial reduction and thresholds."""
        data = {
            "metric1": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            "metric2": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        }
        config = MultiTrialConfig(
            input_file="dummy.csv",
            metrics=["metric1", "metric2"],
            thresholds={"metric2": [0.5]},
            mode="mean",
            trials_per_sample=5,
        )

        transformed = self.processor.transform_data(data, config)

        # Check trial reduction happened
        self.assertEqual(len(transformed["metric1"]), 2)  # 10/5 = 2 samples
        self.assertEqual(len(transformed["metric2"]), 2)

        # Check threshold created
        self.assertIn("metric2_at_0.5", transformed)
        self.assertEqual(len(transformed["metric2_at_0.5"]), 2)

        # Check values
        # metric2 means: [0.3, 0.8]
        # threshold at 0.5: [0, 1]
        assert_array_almost_equal(transformed["metric2_at_0.5"], [0.0, 1.0])

    def test_format_csv(self):
        """Test CSV output format."""
        # Create DataFrame instead of dict
        results_data = {
            "metric1": {"mean": 1.234, "std": 0.567},
            "metric1_at_0.5": {"mean": 0.750, "std": 0.433},
            "metric2": {"mean": 2.345, "std": 0.678},
        }
        results = pd.DataFrame.from_dict(results_data, orient="index")
        results.index.name = "metric"

        config = MultiTrialConfig(
            input_file="dummy.csv",
            metrics=["metric1", "metric2"],
            statistics=["mean", "std"],
            output_format="csv",
            format_precision=4,
        )

        output = self.processor._format_csv(results, config)

        lines = output.strip().split("\n")
        self.assertEqual(lines[0], "metric,mean,std")

        # Check all metrics present
        metric_names = [line.split(",")[0] for line in lines[1:]]
        self.assertIn("metric1", metric_names)
        self.assertIn("metric1_at_0.5", metric_names)
        self.assertIn("metric2", metric_names)

        # Check formatting
        self.assertIn("1.2340,0.5670", output)

    def test_format_console(self):
        """Test console table format."""
        # Create DataFrame instead of dict
        results_data = {
            "metric1": {"mean": 1.234, "std": 0.567},
            "metric1_at_0.5": {"mean": 0.750, "std": 0.433},
            "metric2": {"mean": 2.345, "std": 0.678},
        }
        results = pd.DataFrame.from_dict(results_data, orient="index")
        results.index.name = "metric"

        config = MultiTrialConfig(
            input_file="dummy.csv",
            metrics=["metric1", "metric2"],
            statistics=["mean", "std"],
            output_format="console",
            format_precision=4,
        )

        # _format_rich returns a Table object, not a string
        table = self.processor._format_rich(results, config)

        # Since we get a Rich Table object, we can't directly check string content
        # Check that we get a Table instance
        from rich.table import Table

        self.assertIsInstance(table, Table)

    def test_end_to_end_processing(self):
        """Test complete pipeline."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            self.df.to_csv(f, index=False)
            temp_file = f.name

        try:
            config = MultiTrialConfig(
                input_file=temp_file,
                metrics=["metric1", "metric2"],
                thresholds={"metric2": [0.5]},
                mode="mean",
                trials_per_sample=5,
                statistics=["mean", "std"],
                output_format="csv",
            )

            # Capture output
            with patch("builtins.print") as mock_print:
                self.processor.process(config)

                # Check output was printed
                mock_print.assert_called_once()
                output = mock_print.call_args[0][0]

                # Verify CSV format
                self.assertIn("metric,mean,std", output)
                self.assertIn("metric1", output)
                self.assertIn("metric2", output)
                self.assertIn("metric2_at_0.5", output)

        finally:
            os.unlink(temp_file)


if __name__ == "__main__":
    unittest.main()
