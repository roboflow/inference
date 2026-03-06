"""Tests for formatting modules."""

import unittest
import pandas as pd
import numpy as np
from rich.table import Table

from ..formatting.csv import CSVFormatter
from ..formatting.rich_table import RichTableFormatter


class MockConfig:
    """Mock configuration for testing."""

    def __init__(self, **kwargs):
        self.format_precision = kwargs.get("format_precision", 6)
        self.show_section_lines = kwargs.get("show_section_lines", False)
        self.trial_reduction_mode = kwargs.get("trial_reduction_mode", "mean")
        self.metrics = kwargs.get("metrics", {})
        self.statistics = kwargs.get("statistics", ["mean", "std"])


class TestCSVFormatter(unittest.TestCase):
    def setUp(self):
        """Create test data."""
        self.formatter = CSVFormatter()

        # Create test DataFrame
        self.df = pd.DataFrame(
            {
                "mean": [0.123456789, 0.987654321, 0.555555555],
                "std": [0.012345678, 0.098765432, 0.055555555],
                "p95": [0.234567890, 1.087654321, 0.655555555],
            },
            index=["metric1", "metric2", "metric3"],
        )
        self.df.index.name = "metric"

    def test_format_basic(self):
        """Test basic CSV formatting."""
        config = MockConfig(format_precision=4)
        result = self.formatter.format(self.df, config)

        # Check header
        self.assertIn("metric,mean,std,p95", result)

        # Check formatting
        self.assertIn("metric1,0.1235,0.0123,0.2346", result)
        self.assertIn("metric2,0.9877,0.0988,1.0877", result)

        # Check line endings
        self.assertNotIn("\r\n", result)
        self.assertTrue(result.endswith("\n"))

    def test_format_with_missing_values(self):
        """Test formatting with NaN values."""
        df_with_nan = self.df.copy()
        df_with_nan.loc["metric2", "std"] = np.nan

        config = MockConfig(format_precision=3)
        result = self.formatter.format(df_with_nan, config)

        # Check NaN is formatted as empty string
        lines = result.strip().split("\n")
        metric2_line = [l for l in lines if l.startswith("metric2")][0]
        self.assertIn("metric2,0.988,,1.088", metric2_line)

    def test_precision_settings(self):
        """Test different precision settings."""
        # Test high precision
        config = MockConfig(format_precision=8)
        result = self.formatter.format(self.df, config)
        self.assertIn("0.12345679", result)

        # Test low precision
        config = MockConfig(format_precision=1)
        result = self.formatter.format(self.df, config)
        self.assertIn("0.1,0.0,0.2", result)


class TestRichTableFormatter(unittest.TestCase):
    def setUp(self):
        """Create test data with hierarchical metrics."""
        self.formatter = RichTableFormatter()

        # Create test DataFrame with base metrics and thresholds
        self.df = pd.DataFrame(
            {
                "mean": [0.85, 0.72, 0.91, 0.05, 0.95, 0.12],
                "std": [0.10, 0.15, 0.08, 0.02, 0.03, 0.05],
            },
            index=[
                "f1_score",
                "f1_score_at_0.5",
                "f1_score_at_0.7",
                "chamfer_distance",
                "precision",
                "precision_at_0.8",
            ],
        )
        self.df.index.name = "metric"

    def test_create_rich_table(self):
        """Test Rich table creation."""
        config = MockConfig()
        result = self.formatter.format(self.df, config)

        # Check we get a Rich Table
        self.assertIsInstance(result, Table)

        # Table should have correct number of columns
        # Metric + 2 statistics (mean, std)
        self.assertEqual(len(result.columns), 3)

    def test_hierarchical_sorting(self):
        """Test metrics are sorted hierarchically."""
        sorted_metrics = self.formatter._sort_metrics_hierarchically(self.df.index)

        expected_order = [
            "chamfer_distance",
            "f1_score",
            "f1_score_at_0.5",
            "f1_score_at_0.7",
            "precision",
            "precision_at_0.8",
        ]

        self.assertEqual(sorted_metrics, expected_order)

    def test_metric_name_formatting(self):
        """Test metric name formatting with hierarchy."""
        # Base metric
        self.assertEqual(self.formatter._format_metric_name("f1_score"), "f1_score")

        # Threshold metric
        self.assertEqual(
            self.formatter._format_metric_name("f1_score_at_0.5"),
            "  └─ f1_score_at_0.5",
        )

    def test_with_direction_indicators(self):
        """Test direction indicators when metrics have directions."""
        from ..processor.multi_trial import Metric

        config = MockConfig(
            metrics={
                "f1_score": Metric(direction="maximize"),
                "chamfer_distance": Metric(direction="minimize"),
                "precision": Metric(direction="maximize"),
            }
        )

        # Test direction indicators
        self.assertEqual(
            self.formatter._get_direction_indicator("f1_score", config), "↑"
        )
        self.assertEqual(
            self.formatter._get_direction_indicator("chamfer_distance", config), "↓"
        )

        # Threshold metrics should not have indicators
        self.assertEqual(
            self.formatter._get_direction_indicator("f1_score_at_0.5", config), ""
        )

    def test_table_with_sections(self):
        """Test table creation with section lines."""
        config = MockConfig(show_section_lines=True)
        formatter = RichTableFormatter(show_section_lines=True)

        result = formatter.format(self.df, config)

        # We can't easily test the visual output, but ensure no errors
        self.assertIsInstance(result, Table)


if __name__ == "__main__":
    unittest.main()
