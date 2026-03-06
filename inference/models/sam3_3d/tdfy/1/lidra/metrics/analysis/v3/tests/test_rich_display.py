"""Tests for Rich display improvements in multi-trial processor."""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from io import StringIO

from ..processor.multi_trial import MultiTrialProcessor, MultiTrialConfig, Metric
from ..processor.formatting import RichTableFormatter


class TestRichDisplay(unittest.TestCase):
    """Test Rich display formatting enhancements."""

    def setUp(self):
        """Set up test data."""
        # Create sample data
        self.test_data = pd.DataFrame(
            {
                "sample_uuid": ["s1", "s1", "s2", "s2"],
                "trial": [1, 2, 1, 2],
                "f1_score": [0.85, 0.87, 0.82, 0.84],
                "precision": [0.88, 0.89, 0.86, 0.87],
                "chamfer_distance": [0.12, 0.11, 0.14, 0.13],
            }
        )

        # Create config
        self.config = MultiTrialConfig(
            input_file="test.csv",
            metrics=Metric.make_metrics_dict(
                f1_score={"direction": "maximize"},
                precision={"direction": "maximize"},
                chamfer_distance={"direction": "minimize"},
            ),
            thresholds={"f1_score": [0.8, 0.85], "chamfer_distance": [0.15]},
            statistics=["mean", "std"],
            format_precision=3,
        )

        # Create processor and formatter
        self.processor = MultiTrialProcessor()
        self.formatter = RichTableFormatter()

    def test_calculate_metric_column_width(self):
        """Test adaptive column width calculation."""
        results = {
            "short": {"mean": 0.5},
            "a_very_long_metric_name_that_exceeds_normal_width": {"mean": 0.5},
            "metric_at_0.5": {"mean": 0.5},
        }

        width = self.formatter._calculate_metric_column_width(results)

        # Should be clamped between min and max
        self.assertGreaterEqual(width, 25)
        self.assertLessEqual(width, 50)

    def test_create_enhanced_title(self):
        """Test enhanced title creation."""
        title = self.formatter._create_title(self.config)

        self.assertIn("Multi-Trial Metrics Analysis", title)
        self.assertIn("mean", title)
        self.assertIn("average across trials", title)

    def test_sort_metrics_hierarchically(self):
        """Test hierarchical metric sorting."""
        metrics = [
            "precision",
            "f1_score_at_0.85",
            "chamfer_distance",
            "f1_score",
            "f1_score_at_0.8",
            "chamfer_distance_at_0.15",
        ]

        sorted_metrics = self.formatter._sort_metrics_hierarchically(metrics)

        # Check order: base metrics first, then their thresholds
        expected_order = [
            "chamfer_distance",
            "chamfer_distance_at_0.15",
            "f1_score",
            "f1_score_at_0.8",
            "f1_score_at_0.85",
            "precision",
        ]
        self.assertEqual(sorted_metrics, expected_order)

    def test_format_metric_name_hierarchical(self):
        """Test hierarchical metric name formatting."""
        # Base metric
        base_name = self.formatter._format_metric_name("f1_score")
        self.assertEqual(base_name, "f1_score")

        # Threshold metric
        threshold_name = self.formatter._format_metric_name("f1_score_at_0.8")
        self.assertEqual(threshold_name, "  └─ f1_score_at_0.8")

    def test_get_direction_indicator(self):
        """Test direction indicator retrieval."""
        # Maximize metric
        indicator = self.formatter._get_direction_indicator("f1_score", self.config)
        self.assertEqual(indicator, "↑")

        # Minimize metric
        indicator = self.formatter._get_direction_indicator(
            "chamfer_distance", self.config
        )
        self.assertEqual(indicator, "↓")

        # Unknown metric
        indicator = self.formatter._get_direction_indicator("unknown", self.config)
        self.assertEqual(indicator, "")

    def test_create_metric_row(self):
        """Test metric row creation."""
        # Base metric row
        stats = {"mean": 0.85, "std": 0.02}
        row = self.formatter._create_row("f1_score", stats, self.config)

        self.assertEqual(len(row), 4)  # metric_text, direction, mean, std
        self.assertEqual(row[1], "↑")  # direction indicator
        self.assertEqual(row[2], "0.850")  # mean value
        self.assertEqual(row[3], "0.020")  # std value

        # Threshold metric row
        threshold_stats = {"mean": 0.75, "std": 0.1}
        threshold_row = self.formatter._create_row(
            "f1_score_at_0.8", threshold_stats, self.config
        )

        self.assertEqual(threshold_row[1], "")  # no direction for thresholds

    def test_rich_table_structure(self):
        """Test that Rich table is created with correct structure."""
        # Test the actual table creation without mocking since Table is imported inside the method
        table = self.formatter._create_table(self.config, 35)

        # Verify table has expected attributes
        self.assertIsNotNone(table)
        self.assertIn("Multi-Trial Metrics Analysis", table.title)

    def test_populate_table_rows_spacing(self):
        """Test that spacing is added between metric groups."""
        mock_table = MagicMock()

        results = {
            "f1_score": {"mean": 0.85, "std": 0.02},
            "f1_score_at_0.8": {"mean": 0.75, "std": 0.1},
            "precision": {"mean": 0.88, "std": 0.01},
        }

        self.formatter._populate_rows(mock_table, results, self.config)

        # Check that spacing row was added between f1_score group and precision
        spacing_calls = [
            call
            for call in mock_table.add_row.call_args_list
            if call[0][0] == "" and "end_section" in call[1]
        ]
        self.assertEqual(len(spacing_calls), 1)

    def test_full_rich_output(self):
        """Test complete Rich output generation."""
        # Create sample results
        results = {
            "f1_score": {"mean": 0.85, "std": 0.02},
            "f1_score_at_0.8": {"mean": 0.75, "std": 0.1},
            "chamfer_distance": {"mean": 0.125, "std": 0.015},
        }

        output = self.formatter.format_results(results, self.config)

        # Check that output contains expected elements
        self.assertIn("Multi-Trial Metrics Analysis", output)
        self.assertIn("↑/↓", output)
        self.assertIn("f1_score", output)
        self.assertIn("└─ f1_score_at_0.8", output)
        self.assertIn("chamfer_distance", output)
        self.assertIn("0.850", output)  # formatted mean value


if __name__ == "__main__":
    unittest.main()
