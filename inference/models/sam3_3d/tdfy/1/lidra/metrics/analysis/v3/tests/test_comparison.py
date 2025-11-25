"""Tests for ComparisonProcessor."""

import unittest
from pathlib import Path
import tempfile
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from ..processor.comparison import ComparisonProcessor, ComparisonConfig
from ..formatting.comparison import ComparisonFormatter


class TestComparisonConfig(unittest.TestCase):
    """Test configuration validation."""

    def test_should_require_input_files(self):
        """Test that at least one input file is required."""
        with self.assertRaises(ValueError) as cm:
            ComparisonConfig(
                input_file="dummy.csv",  # Required by BaseConfig
                input_files=[],  # Empty list should fail
            )
        self.assertIn("At least one input file", str(cm.exception))

    def test_should_validate_experiment_names_count(self):
        """Test experiment names must match input files count."""
        with self.assertRaises(ValueError) as cm:
            ComparisonConfig(
                input_file="dummy.csv",
                input_files=["file1.csv", "file2.csv"],
                experiment_names=["exp1"],  # Mismatch: 1 name for 2 files
            )
        self.assertIn("must match number of input files", str(cm.exception))

    def test_should_accept_valid_config(self):
        """Test valid configuration is accepted."""
        config = ComparisonConfig(
            input_file="dummy.csv",
            input_files=["file1.csv", "file2.csv"],
            experiment_names=["exp1", "exp2"],
            statistic="p95",
            metrics_filter=["chamfer_*"],
            sort_by="exp1",
            transpose=True,
        )
        self.assertEqual(config.statistic, "p95")
        self.assertEqual(len(config.input_files), 2)


class TestComparisonProcessor(unittest.TestCase):
    """Test ComparisonProcessor functionality."""

    def setUp(self):
        """Create test data."""
        self.temp_dir = tempfile.mkdtemp()
        self.processor = ComparisonProcessor()

        # Create sample CSV files
        self.sample_data1 = pd.DataFrame(
            {
                "metric": ["chamfer_distance", "f1_score", "rotation_error"],
                "mean": [0.025, 0.85, 12.5],
                "std": [0.005, 0.03, 2.1],
                "p50": [0.024, 0.86, 12.0],
                "p95": [0.035, 0.91, 16.5],
            }
        ).set_index("metric")

        self.sample_data2 = pd.DataFrame(
            {
                "metric": ["chamfer_distance", "f1_score", "rotation_error"],
                "mean": [0.018, 0.89, 10.2],
                "std": [0.004, 0.02, 1.8],
                "p50": [0.017, 0.90, 9.8],
                "p95": [0.025, 0.93, 13.2],
            }
        ).set_index("metric")

        # Save to CSV files
        self.file1 = Path(self.temp_dir) / "exp1" / "results.csv"
        self.file2 = Path(self.temp_dir) / "exp2" / "results.csv"
        self.file1.parent.mkdir(exist_ok=True)
        self.file2.parent.mkdir(exist_ok=True)

        self.sample_data1.to_csv(self.file1)
        self.sample_data2.to_csv(self.file2)

    def tearDown(self):
        """Clean up test files."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_should_load_multiple_csv_files(self):
        """Test loading and validating multiple CSV files."""
        config = ComparisonConfig(
            input_file="dummy.csv",
            input_files=[str(self.file1), str(self.file2)],
            statistic="mean",
        )

        # Extract data
        data = self.processor._extract_data(pd.DataFrame(), config)

        self.assertEqual(len(data), 2)
        self.assertIn("exp1", data)
        self.assertIn("exp2", data)
        self.assertTrue(data["exp1"].equals(self.sample_data1))
        self.assertTrue(data["exp2"].equals(self.sample_data2))

    def test_should_use_custom_experiment_names(self):
        """Test using custom experiment names."""
        config = ComparisonConfig(
            input_file="dummy.csv",
            input_files=[str(self.file1), str(self.file2)],
            experiment_names=["baseline", "improved"],
            statistic="mean",
        )

        data = self.processor._extract_data(pd.DataFrame(), config)

        self.assertIn("baseline", data)
        self.assertIn("improved", data)
        self.assertNotIn("exp1", data)
        self.assertNotIn("exp2", data)

    def test_should_validate_statistic_column_exists(self):
        """Test validation of requested statistic column."""
        config = ComparisonConfig(
            input_file="dummy.csv",
            input_files=[str(self.file1)],
            statistic="nonexistent",
        )

        with self.assertRaises(ValueError) as cm:
            self.processor._extract_data(pd.DataFrame(), config)

        self.assertIn("Statistic 'nonexistent' not found", str(cm.exception))
        self.assertIn("Available: mean, std, p50, p95", str(cm.exception))

    def test_should_transform_data_to_comparison_table(self):
        """Test transforming data into comparison DataFrame."""
        config = ComparisonConfig(
            input_file="dummy.csv",
            input_files=[str(self.file1), str(self.file2)],
            experiment_names=["baseline", "improved"],
            statistic="mean",
        )

        # Load data
        data = self.processor._extract_data(pd.DataFrame(), config)

        # Transform
        result = self.processor.transform_data(data, config)

        # Check structure
        self.assertEqual(list(result.columns), ["baseline", "improved"])
        self.assertEqual(
            list(result.index), ["chamfer_distance", "f1_score", "rotation_error"]
        )

        # Check values
        self.assertAlmostEqual(result.loc["chamfer_distance", "baseline"], 0.025)
        self.assertAlmostEqual(result.loc["chamfer_distance", "improved"], 0.018)
        self.assertAlmostEqual(result.loc["f1_score", "baseline"], 0.85)
        self.assertAlmostEqual(result.loc["f1_score", "improved"], 0.89)

    def test_should_filter_metrics_by_pattern(self):
        """Test metric filtering with wildcards."""
        config = ComparisonConfig(
            input_file="dummy.csv",
            input_files=[str(self.file1), str(self.file2)],
            statistic="mean",
            metrics_filter=["chamfer_*", "f1_score"],
        )

        data = self.processor._extract_data(pd.DataFrame(), config)
        result = self.processor.transform_data(data, config)

        # Should include chamfer_distance and f1_score, but not rotation_error
        self.assertIn("chamfer_distance", result.index)
        self.assertIn("f1_score", result.index)
        self.assertNotIn("rotation_error", result.index)

    def test_should_sort_by_experiment_column(self):
        """Test sorting by a specific experiment column."""
        # Add a fourth metric to make sorting visible
        for df in [self.sample_data1, self.sample_data2]:
            df.loc["accuracy"] = [0.7, 0.1, 0.71, 0.75]

        self.sample_data1.to_csv(self.file1)
        self.sample_data2.to_csv(self.file2)

        config = ComparisonConfig(
            input_file="dummy.csv",
            input_files=[str(self.file1), str(self.file2)],
            experiment_names=["baseline", "improved"],
            statistic="mean",
            sort_by="baseline",
            ascending=True,
        )

        data = self.processor._extract_data(pd.DataFrame(), config)
        result = self.processor.transform_data(data, config)

        # Check sorted order (ascending by baseline values)
        sorted_metrics = list(result.index)
        baseline_values = list(result["baseline"])
        self.assertEqual(baseline_values, sorted(baseline_values))

    def test_should_handle_transpose(self):
        """Test transposing the comparison table."""
        config = ComparisonConfig(
            input_file="dummy.csv",
            input_files=[str(self.file1), str(self.file2)],
            experiment_names=["baseline", "improved"],
            statistic="mean",
            transpose=True,
        )

        data = self.processor._extract_data(pd.DataFrame(), config)
        result = self.processor.transform_data(data, config)

        # Check transposed structure
        self.assertEqual(
            list(result.columns), ["chamfer_distance", "f1_score", "rotation_error"]
        )
        self.assertEqual(list(result.index), ["baseline", "improved"])

    def test_should_handle_missing_values(self):
        """Test handling of missing values."""
        # Create data with missing metric
        incomplete_data = pd.DataFrame(
            {
                "metric": ["chamfer_distance", "f1_score"],  # Missing rotation_error
                "mean": [0.020, 0.87],
                "std": [0.004, 0.025],
            }
        ).set_index("metric")

        file3 = Path(self.temp_dir) / "exp3" / "results.csv"
        file3.parent.mkdir(exist_ok=True)
        incomplete_data.to_csv(file3)

        config = ComparisonConfig(
            input_file="dummy.csv",
            input_files=[str(self.file1), str(file3)],
            statistic="mean",
            show_missing_as="--",
        )

        data = self.processor._extract_data(pd.DataFrame(), config)
        transformed = self.processor.transform_data(data, config)
        result = self.processor.compute_results(transformed, config)

        # Check missing value is filled
        self.assertEqual(result.loc["rotation_error", "exp3"], "--")

    def test_should_handle_file_not_found(self):
        """Test proper error when file doesn't exist."""
        config = ComparisonConfig(
            input_file="dummy.csv",
            input_files=["/nonexistent/path.csv"],
            statistic="mean",
        )

        with self.assertRaises(ValueError) as cm:
            self.processor._extract_data(pd.DataFrame(), config)

        self.assertIn("Results file not found", str(cm.exception))

    def test_should_format_csv_output(self):
        """Test CSV output formatting."""
        config = ComparisonConfig(
            input_file="dummy.csv",
            input_files=[str(self.file1), str(self.file2)],
            experiment_names=["baseline", "improved"],
            statistic="mean",
        )

        # Run full pipeline
        data = self.processor._extract_data(pd.DataFrame(), config)
        transformed = self.processor.transform_data(data, config)
        results = self.processor.compute_results(transformed, config)

        # Get CSV output
        csv_output = self.processor.get_output(results, config, "csv")

        # Check it's valid CSV
        self.assertIsInstance(csv_output, str)
        self.assertIn("metric,baseline,improved", csv_output)
        self.assertIn("chamfer_distance,0.025", csv_output)
        self.assertIn("f1_score,0.85", csv_output)

    @patch("lidra.metrics.analysis.v3.processor.comparison.ComparisonFormatter")
    def test_should_format_rich_output(self, mock_formatter_class):
        """Test Rich table output formatting."""
        config = ComparisonConfig(
            input_file="dummy.csv",
            input_files=[str(self.file1), str(self.file2)],
            statistic="mean",
        )

        # Setup mock
        mock_formatter = MagicMock()
        mock_formatter_class.return_value = mock_formatter
        mock_table = MagicMock()
        mock_formatter.format.return_value = mock_table

        # Run pipeline
        data = self.processor._extract_data(pd.DataFrame(), config)
        transformed = self.processor.transform_data(data, config)
        results = self.processor.compute_results(transformed, config)

        # Get Rich output
        output = self.processor.get_output(results, config, "rich")

        # Verify formatter was called correctly
        mock_formatter_class.assert_called_once_with(transpose=False)
        mock_formatter.format.assert_called_once()
        self.assertEqual(output, mock_table)


class TestComparisonFormatter(unittest.TestCase):
    """Test ComparisonFormatter functionality."""

    def test_should_create_rich_table(self):
        """Test creating a Rich table from comparison results."""
        # Create sample data
        data = pd.DataFrame(
            {"baseline": [0.025, 0.85], "improved": [0.018, 0.89]},
            index=["chamfer_distance", "f1_score"],
        )

        config = MagicMock()
        config.statistic = "mean"
        config.format_precision = 3
        config.show_missing_as = "N/A"

        formatter = ComparisonFormatter(transpose=False)
        table = formatter._create_comparison_table(data, "Test Comparison", config)

        # Verify table structure
        self.assertEqual(len(table.columns), 3)  # Metric + 2 experiments
        self.assertEqual(table.columns[0].header, "Metric")
        self.assertEqual(table.columns[1].header, "baseline")
        self.assertEqual(table.columns[2].header, "improved")


if __name__ == "__main__":
    unittest.main()
