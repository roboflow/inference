"""Unit tests for the TrialMetricsProcessor class."""

import unittest
import pandas as pd
import numpy as np

from .multi_trial import TrialMetricsProcessor
from ..definitions import Report, Table, BestOfNSelection, Threshold


class TestTrialFiltering(unittest.TestCase):
    """Test cases for trial filtering functionality."""

    def setUp(self):
        """Set up test data with 5 trials per sample (1-indexed)."""
        self.test_data = []
        for sample_id in range(2):  # 2 samples
            for trial in range(1, 6):  # 5 trials: 1, 2, 3, 4, 5
                self.test_data.append(
                    {
                        "sample_uuid": f"sample_{sample_id}",
                        "trial": trial,
                        "f1": 0.8 + trial * 0.01,
                        "precision": 0.85,
                        "recall": 0.75,
                        "chamfer_distance": 0.2 - trial * 0.01,
                        "oriented_f1": 0.7,
                        "oriented_precision": 0.75,
                        "oriented_recall": 0.65,
                        "oriented_chamfer_distance": 0.25,
                        "oriented_rot_error_deg": 20.0,
                    }
                )

        self.df = pd.DataFrame(self.test_data)
        # Create a test report configuration instead of loading from file
        self.report = Report(
            tables={
                "shape": Table(
                    columns=["f1", "precision", "recall", "chamfer_distance"],
                    best_of_trials=BestOfNSelection(
                        select_column="chamfer_distance",
                        select_max=False,  # Lower is better for chamfer distance
                    ),
                    thresholds={
                        "f1": Threshold(thresholds=[0.8, 0.85], higher_is_better=True),
                        "chamfer_distance": Threshold(
                            thresholds=[0.15, 0.2], higher_is_better=False
                        ),
                    },
                ),
                "oriented": Table(
                    columns=[
                        "oriented_f1",
                        "oriented_precision",
                        "oriented_recall",
                        "oriented_chamfer_distance",
                        "oriented_rot_error_deg",
                    ],
                    best_of_trials=BestOfNSelection(
                        select_column="oriented_f1",
                        select_max=True,  # Higher is better for F1
                    ),
                ),
            }
        )

    def test_filter_trials_basic(self):
        """Test filtering to first N trials."""
        analyzer = TrialMetricsProcessor(self.df.copy(), self.report)

        # Filter to first 3 trials
        analyzer.filter_trials(3)

        # Check results
        self.assertEqual(len(analyzer.df), 6)  # 2 samples * 3 trials
        self.assertEqual(analyzer.original_row_count, 10)
        self.assertListEqual(sorted(analyzer.df["trial"].unique()), [1, 2, 3])

    def test_filter_preserves_all_samples(self):
        """Test that filtering preserves all unique samples."""
        analyzer = TrialMetricsProcessor(self.df.copy(), self.report)
        original_samples = set(analyzer.df["sample_uuid"].unique())

        analyzer.filter_trials(2)

        filtered_samples = set(analyzer.df["sample_uuid"].unique())
        self.assertEqual(original_samples, filtered_samples)

    def test_metadata_shows_filtering(self):
        """Test that metadata correctly reports filtering."""
        analyzer = TrialMetricsProcessor(self.df.copy(), self.report)
        analyzer.filter_trials(3)

        results = analyzer.create_report("mean")
        metadata = analyzer.generate_run_metadata("mean", results, "test.csv")

        # Check filtering info in metadata
        self.assertEqual(metadata["input_data_summary"]["original_rows"], 10)
        self.assertEqual(metadata["input_data_summary"]["rows_after_filtering"], 6)
        self.assertEqual(metadata["input_data_summary"]["n_trials"], 3)

    def test_best_mode_after_filtering(self):
        """Test that best mode works correctly on filtered data."""
        analyzer = TrialMetricsProcessor(self.df.copy(), self.report)

        # Filter to exclude trials 4 and 5 (which have the best chamfer_distance)
        analyzer.filter_trials(3)

        results = analyzer.create_report("best")

        # Best should come from trial 3 (chamfer_distance = 0.17)
        # The mean chamfer_distance for best trials should be 0.17
        shape_results = results["shape"]
        best_cd = shape_results.loc["chamfer_distance", "value"]
        self.assertAlmostEqual(best_cd, 0.17, places=5)


class TestDropMissingValues(unittest.TestCase):
    """Test cases for handling missing values."""

    def setUp(self):
        """Set up test data with missing values."""
        self.test_data = []
        # Sample 0: Complete data
        for trial in range(1, 4):
            self.test_data.append(
                {
                    "sample_uuid": "sample_0",
                    "trial": trial,
                    "f1": 0.8,
                    "chamfer_distance": 0.2,
                    "oriented_f1": 0.7,
                }
            )

        # Sample 1: Missing some values
        for trial in range(1, 4):
            self.test_data.append(
                {
                    "sample_uuid": "sample_1",
                    "trial": trial,
                    "f1": np.nan if trial == 2 else 0.75,
                    "chamfer_distance": 0.25,
                    "oriented_f1": 0.65,
                }
            )

        # Sample 2: Incomplete trials (only 2 trials)
        for trial in range(1, 3):
            self.test_data.append(
                {
                    "sample_uuid": "sample_2",
                    "trial": trial,
                    "f1": 0.85,
                    "chamfer_distance": 0.15,
                    "oriented_f1": 0.75,
                }
            )

        self.df = pd.DataFrame(self.test_data)
        self.report = Report(
            tables={"metrics": Table(columns=["f1", "chamfer_distance", "oriented_f1"])}
        )

    def test_drop_missing_all(self):
        """Test dropping samples with any missing values."""
        processor = TrialMetricsProcessor(self.df.copy(), self.report)
        processor.drop_missing_values(drop_all=True)

        # Should only keep sample_0 (3 trials)
        self.assertEqual(len(processor.df), 3)
        self.assertEqual(processor.df["sample_uuid"].unique()[0], "sample_0")

    def test_drop_missing_specific_columns(self):
        """Test dropping samples with missing values in specific columns."""
        processor = TrialMetricsProcessor(self.df.copy(), self.report)
        processor.drop_missing_values(columns=["f1"])

        # Should drop sample_1 (has NaN in f1)
        # Should also drop sample_2 (incomplete trials)
        self.assertEqual(len(processor.df), 3)
        self.assertEqual(processor.df["sample_uuid"].unique()[0], "sample_0")

    def test_drop_incomplete_trials_only(self):
        """Test dropping only incomplete trials."""
        processor = TrialMetricsProcessor(self.df.copy(), self.report)
        processor.drop_incomplete_trials(expected_trials=3)

        # Should drop sample_2 (only has 2 trials)
        self.assertEqual(len(processor.df), 6)  # sample_0 and sample_1
        self.assertIn("sample_0", processor.df["sample_uuid"].values)
        self.assertIn("sample_1", processor.df["sample_uuid"].values)
        self.assertNotIn("sample_2", processor.df["sample_uuid"].values)

    def test_no_drop_incomplete_trials(self):
        """Test keeping incomplete trials when requested."""
        processor = TrialMetricsProcessor(self.df.copy(), self.report)
        processor.drop_missing_values(
            columns=["chamfer_distance"], drop_incomplete_trials=False
        )

        # Should keep all samples since chamfer_distance has no missing values
        self.assertEqual(len(processor.df), 8)  # All original rows
        self.assertEqual(processor.df["sample_uuid"].nunique(), 3)


class TestBestOfTrialsCalculations(unittest.TestCase):
    """Test cases for best-of-trials metric calculations."""

    def setUp(self):
        """Set up test data with known patterns."""
        self.test_data = []
        for sample_id in range(3):
            for trial in range(1, 4):
                self.test_data.append(
                    {
                        "sample_uuid": f"sample_{sample_id}",
                        "trial": trial,
                        "f1": 0.6
                        + sample_id * 0.1
                        + trial * 0.05,  # Increases with trial
                        "chamfer_distance": 0.3
                        - sample_id * 0.05
                        - trial * 0.02,  # Decreases with trial
                    }
                )

        self.df = pd.DataFrame(self.test_data)
        self.report = Report(
            tables={
                "best_f1": Table(
                    columns=["f1", "chamfer_distance"],
                    best_of_trials=BestOfNSelection(
                        select_column="f1", select_max=True
                    ),
                    thresholds={
                        "f1": Threshold(thresholds=[0.7, 0.8], higher_is_better=True)
                    },
                ),
                "best_chamfer": Table(
                    columns=["f1", "chamfer_distance"],
                    best_of_trials=BestOfNSelection(
                        select_column="chamfer_distance", select_max=False
                    ),
                    thresholds={
                        "chamfer_distance": Threshold(
                            thresholds=[0.2, 0.25], higher_is_better=False
                        )
                    },
                ),
            }
        )

    def test_best_of_trials_max(self):
        """Test selecting best trial by maximum value."""
        processor = TrialMetricsProcessor(self.df, self.report)
        results = processor.create_table("best_f1", "best")

        # For each sample, trial 3 should be selected (highest f1)
        # Sample 0: f1 = 0.75, Sample 1: f1 = 0.85, Sample 2: f1 = 0.95
        expected_mean_f1 = (0.75 + 0.85 + 0.95) / 3
        self.assertAlmostEqual(results.loc["f1", "value"], expected_mean_f1, places=5)

        # Check threshold metrics
        self.assertAlmostEqual(
            results.loc["f1_acc0.7", "value"], 1.0, places=5
        )  # All > 0.7
        self.assertAlmostEqual(
            results.loc["f1_acc0.8", "value"], 2 / 3, places=5
        )  # 2 out of 3 > 0.8

    def test_best_of_trials_min(self):
        """Test selecting best trial by minimum value."""
        processor = TrialMetricsProcessor(self.df, self.report)
        results = processor.create_table("best_chamfer", "best")

        # For each sample, trial 3 should be selected (lowest chamfer_distance)
        # Sample 0: cd = 0.24, Sample 1: cd = 0.19, Sample 2: cd = 0.14
        expected_mean_cd = (0.24 + 0.19 + 0.14) / 3
        self.assertAlmostEqual(
            results.loc["chamfer_distance", "value"], expected_mean_cd, places=5
        )

        # Check threshold metrics (lower is better)
        self.assertAlmostEqual(
            results.loc["chamfer_distance_acc0.25", "value"], 1.0, places=5
        )  # All < 0.25
        self.assertAlmostEqual(
            results.loc["chamfer_distance_acc0.2", "value"], 2 / 3, places=5
        )  # 2 out of 3 < 0.2


class TestValidation(unittest.TestCase):
    """Test cases for validation logic."""

    def test_missing_required_columns(self):
        """Test error when required columns are missing."""
        df = pd.DataFrame({"some_column": [1, 2, 3]})

        with self.assertRaises(ValueError) as context:
            TrialMetricsProcessor._validate_df(df)

        self.assertIn("Missing required columns", str(context.exception))

    def test_empty_dataframe(self):
        """Test error when DataFrame is empty."""
        df = pd.DataFrame(columns=["sample_uuid", "trial"])

        with self.assertRaises(ValueError) as context:
            TrialMetricsProcessor._validate_df(df)

        self.assertIn("empty", str(context.exception))

    def test_report_validation_missing_columns(self):
        """Test report validation catches missing columns."""
        df = pd.DataFrame({"sample_uuid": ["s1"], "trial": [1], "metric1": [0.5]})

        report = Report(
            tables={
                "test": Table(columns=["metric1", "metric2"])  # metric2 doesn't exist
            }
        )

        with self.assertRaises(ValueError) as context:
            Report.validate(report, set(df.columns))

        self.assertIn("missing columns", str(context.exception))
        self.assertIn("metric2", str(context.exception))


class TestOverallMetrics(unittest.TestCase):
    """Test cases for overall (mean) metric calculations."""

    def test_calculate_overall_metrics(self):
        """Test mean calculation across all samples and trials."""
        test_data = []
        for sample in range(2):
            for trial in range(1, 4):
                test_data.append(
                    {
                        "sample_uuid": f"sample_{sample}",
                        "trial": trial,
                        "metric1": 0.5 + sample * 0.1,
                        "metric2": 0.8 - sample * 0.1,
                    }
                )

        df = pd.DataFrame(test_data)
        # Don't include missing_metric in the report config since it doesn't exist in the data
        report = Report(tables={"overall": Table(columns=["metric1", "metric2"])})

        processor = TrialMetricsProcessor(df, report)
        # But we can still try to calculate it - it just won't appear in results
        results = processor.calculate_overall_metrics(
            ["metric1", "metric2", "missing_metric"]
        )

        # Check calculated means
        self.assertAlmostEqual(results.loc["metric1_mean", "value"], 0.55, places=5)
        self.assertAlmostEqual(results.loc["metric2_mean", "value"], 0.75, places=5)

        # Missing metric should not appear in results
        self.assertNotIn("missing_metric_mean", results.index)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def test_single_sample_single_trial(self):
        """Test with minimal data - one sample, one trial."""
        df = pd.DataFrame([{"sample_uuid": "sample_0", "trial": 1, "metric1": 0.9}])

        report = Report(tables={"test": Table(columns=["metric1"])})

        processor = TrialMetricsProcessor(df, report)
        results = processor.calculate_overall_metrics(["metric1"])

        self.assertAlmostEqual(results.loc["metric1_mean", "value"], 0.9, places=5)

    def test_filter_trials_exceeds_available(self):
        """Test filtering to more trials than available."""
        df = pd.DataFrame([{"sample_uuid": "sample_0", "trial": 1, "metric1": 0.5}])

        report = Report(tables={})
        processor = TrialMetricsProcessor(df, report)

        # Should not raise error - keeps the 1 trial we have
        processor.filter_trials(5)
        self.assertEqual(len(processor.df), 1)

    def test_inconsistent_trial_counts_error(self):
        """Test error when samples have different trial counts in metadata generation."""
        df = pd.DataFrame(
            [
                {"sample_uuid": "sample_0", "trial": 1, "metric1": 0.5},
                {"sample_uuid": "sample_0", "trial": 2, "metric1": 0.6},
                {"sample_uuid": "sample_1", "trial": 1, "metric1": 0.7},
            ]
        )

        report = Report(tables={"test": Table(columns=["metric1"])})
        processor = TrialMetricsProcessor(df, report)

        # This should raise an error during metadata generation
        with self.assertRaises(ValueError) as context:
            processor.generate_run_metadata("mean", {}, "test.csv")

        self.assertIn("Inconsistent trial counts", str(context.exception))

    def test_empty_results_after_filtering(self):
        """Test appropriate error when all data is filtered out."""
        df = pd.DataFrame(
            [{"sample_uuid": "sample_0", "trial": 5, "metric1": 0.5}]  # Only trial 5
        )

        report = Report(tables={})
        processor = TrialMetricsProcessor(df, report)

        # Filter to trials 1-3, which don't exist
        with self.assertRaises(ValueError) as context:
            processor.filter_trials(3)

        self.assertIn("No data remaining", str(context.exception))

    def test_from_csv_validation(self):
        """Test CSV loading with validation."""
        # Create a temporary CSV file for testing
        import tempfile
        import os

        # Valid CSV
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("sample_uuid,trial,metric1\n")
            f.write("sample_0,1,0.5\n")
            f.write("sample_0,2,0.6\n")
            valid_csv = f.name

        try:
            report = Report(tables={"test": Table(columns=["metric1"])})
            processor = TrialMetricsProcessor.from_csv(valid_csv, report)
            self.assertEqual(len(processor.df), 2)
        finally:
            os.unlink(valid_csv)

        # Invalid CSV (missing required columns)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("metric1,metric2\n")
            f.write("0.5,0.6\n")
            invalid_csv = f.name

        try:
            with self.assertRaises(ValueError) as context:
                TrialMetricsProcessor.from_csv(invalid_csv, report)
            self.assertIn("Missing required columns", str(context.exception))
        finally:
            os.unlink(invalid_csv)


class TestMetadataGeneration(unittest.TestCase):
    """Test metadata generation functionality."""

    def test_metadata_basic(self):
        """Test basic metadata generation."""
        df = pd.DataFrame(
            [
                {"sample_uuid": "sample_0", "trial": 1, "metric1": 0.5, "metric2": 0.8},
                {"sample_uuid": "sample_0", "trial": 2, "metric1": 0.6, "metric2": 0.7},
                {"sample_uuid": "sample_1", "trial": 1, "metric1": 0.4, "metric2": 0.9},
                {"sample_uuid": "sample_1", "trial": 2, "metric1": 0.5, "metric2": 0.8},
            ]
        )

        report = Report(tables={"test": Table(columns=["metric1", "metric2"])})

        processor = TrialMetricsProcessor(df, report)
        results = processor.create_report("mean")
        metadata = processor.generate_run_metadata("mean", results, "test.csv")

        # Check metadata structure
        self.assertIn("run_info", metadata)
        self.assertIn("input_data_summary", metadata)
        self.assertIn("report_config", metadata)

        # Check data summary
        summary = metadata["input_data_summary"]
        self.assertEqual(summary["total_rows"], 4)
        self.assertEqual(summary["unique_samples"], 2)
        self.assertEqual(summary["n_trials"], 2)
        self.assertEqual(summary["metrics_count"], 2)

        # Check run info
        run_info = metadata["run_info"]
        self.assertEqual(run_info["input_file"], "test.csv")
        self.assertEqual(run_info["mode"], "mean")
        self.assertIn("timestamp", run_info)


if __name__ == "__main__":
    unittest.main()
