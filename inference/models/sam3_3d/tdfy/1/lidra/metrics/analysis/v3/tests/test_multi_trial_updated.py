"""Updated tests for MultiTrialProcessor with DataFrame-based MetricStats."""

import unittest
import numpy as np
import pandas as pd
from numpy.testing import assert_almost_equal
import tempfile
import os

from ..processor.multi_trial import (
    MultiTrialConfig,
    MultiTrialProcessor,
    Metric,
    MetricData,
    MetricStats,
)
from ..functionals import FUNCTIONALS


class TestMultiTrialConfigUpdated(unittest.TestCase):
    def test_valid_config(self):
        """Test creating valid configuration."""
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


class TestMultiTrialProcessorUpdated(unittest.TestCase):
    def setUp(self):
        """Create processor and test data."""
        self.processor = MultiTrialProcessor()

        # Create test dataframe with multiple trials
        np.random.seed(42)
        n_samples = 5
        trials = 3

        # Create proper multi-trial dataframe
        data_rows = []
        for i in range(n_samples):
            for t in range(trials):
                data_rows.append(
                    {
                        "sample_uuid": f"sample_{i}",
                        "trial": t,
                        "metric1": np.random.randn()
                        + i,  # Add i to make values different per sample
                        "metric2": np.random.randn() + i * 2,
                    }
                )

        self.df = pd.DataFrame(data_rows)

        self.config = MultiTrialConfig(
            input_file="dummy.csv",
            metrics={
                "metric1": Metric(direction="minimize"),
                "metric2": Metric(direction="maximize"),
            },
            thresholds={"metric2": [0.5, 1.0]},
            trial_reduction_mode="mean",
            statistics=["mean", "std"],
        )

    def test_compute_results_returns_dataframe(self):
        """Test that compute_results returns a DataFrame."""
        # Create simple metric data
        data = {
            "metric1": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            "metric2": np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
        }

        results = self.processor.compute_results(data, self.config)

        # Check that results is a DataFrame
        self.assertIsInstance(results, pd.DataFrame)

        # Check structure
        self.assertIn("metric1", results.index)
        self.assertIn("metric2", results.index)
        self.assertIn("mean", results.columns)
        self.assertIn("std", results.columns)

        # Check values
        assert_almost_equal(results.loc["metric1", "mean"], 3.0)
        assert_almost_equal(results.loc["metric2", "mean"], 30.0)

        # Check that columns are in the order specified in config
        self.assertListEqual(list(results.columns), ["mean", "std"])

    def test_format_csv_with_dataframe(self):
        """Test CSV formatting with DataFrame input."""
        # Create DataFrame results
        results_data = {"mean": [1.234, 2.345], "std": [0.567, 0.678]}
        results = pd.DataFrame(results_data, index=["metric1", "metric2"])
        results.index.name = "metric"

        # Update config with proper precision
        config = MultiTrialConfig(
            input_file="dummy.csv",
            metrics={
                "metric1": Metric(direction="minimize"),
                "metric2": Metric(direction="maximize"),
            },
            statistics=["mean", "std"],
            format_precision=4,
        )

        output = self.processor.get_output(results, config, "csv")

        lines = output.strip().split("\n")
        self.assertEqual(lines[0], "metric,mean,std")
        self.assertIn("metric1,1.2340,0.5670", output)
        self.assertIn("metric2,2.3450,0.6780", output)

    def test_format_rich_with_dataframe(self):
        """Test Rich formatting with DataFrame input."""
        # Create DataFrame results
        results_data = {"mean": [1.234, 0.750, 2.345], "std": [0.567, 0.433, 0.678]}
        results = pd.DataFrame(
            results_data, index=["metric1", "metric1_at_0.5", "metric2"]
        )
        results.index.name = "metric"

        table = self.processor.get_output(results, self.config, "rich")

        # Check that we get a Rich Table object
        from rich.table import Table

        self.assertIsInstance(table, Table)

        # We can't easily check the content of a Rich Table,
        # but we've verified it's the right type

    def test_extract_data_returns_dict(self):
        """Test that extract_data returns MetricData (dict of arrays)."""
        data = self.processor._extract_data(self.df, self.config)

        # Check it returns a dict
        self.assertIsInstance(data, dict)

        # Check keys match metrics
        self.assertIn("metric1", data)
        self.assertIn("metric2", data)

        # Check values are numpy arrays
        self.assertIsInstance(data["metric1"], np.ndarray)
        self.assertIsInstance(data["metric2"], np.ndarray)

        # Check array length matches number of samples
        self.assertEqual(len(data["metric1"]), 5)  # 5 samples after trial reduction


if __name__ == "__main__":
    unittest.main()
