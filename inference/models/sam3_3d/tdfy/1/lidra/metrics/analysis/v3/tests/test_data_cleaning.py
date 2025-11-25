"""Tests for data cleaning functionality in MultiTrialProcessor."""

import unittest
import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

from ..multi_trial import MultiTrialConfig, MultiTrialProcessor


class TestDataCleaning(unittest.TestCase):
    """Test data cleaning operations in MultiTrialProcessor."""

    def setUp(self):
        """Create test processor and sample data."""
        self.processor = MultiTrialProcessor()

        # Create test data with known structure
        np.random.seed(42)
        n_samples = 10
        trials_per_sample = 5

        # Create data with sample_uuid and trial columns
        sample_uuids = []
        trials = []
        for i in range(n_samples):
            sample_uuids.extend([f"sample_{i:03d}"] * trials_per_sample)
            trials.extend(list(range(1, trials_per_sample + 1)))

        self.complete_data = pd.DataFrame(
            {
                "sample_uuid": sample_uuids,
                "trial": trials,
                "metric1": np.random.randn(n_samples * trials_per_sample),
                "metric2": np.random.randn(n_samples * trials_per_sample) + 1,
                "metric3": np.random.uniform(0, 1, n_samples * trials_per_sample),
            }
        )

        # Create data with missing values
        self.data_with_na = self.complete_data.copy()
        self.data_with_na.loc[5:7, "metric1"] = np.nan
        self.data_with_na.loc[12, "metric2"] = np.nan

        # Create data with incomplete trials
        self.incomplete_data = self.complete_data.copy()
        # Remove some trials for sample_002
        self.incomplete_data = self.incomplete_data[
            ~(
                (self.incomplete_data["sample_uuid"] == "sample_002")
                & (self.incomplete_data["trial"].isin([3, 4]))
            )
        ]

    def test_no_cleaning(self):
        """Test that data passes through unchanged when no cleaning requested."""
        config = MultiTrialConfig(
            input_file="dummy.csv",
            metrics=["metric1", "metric2"],
            trials_per_sample=5,
            drop_na=False,
            drop_incomplete_trials=False,
        )

        result = self.processor._clean_data(self.complete_data, config)
        pd.testing.assert_frame_equal(result, self.complete_data)

    def test_filter_trials(self):
        """Test filtering to first N trials."""
        config = MultiTrialConfig(
            input_file="dummy.csv",
            metrics=["metric1"],
            max_trials=3,
            trials_per_sample=5,
        )

        result = self.processor._clean_data(self.complete_data, config)

        # Check that only trials 1-3 remain
        self.assertTrue(all(result["trial"] <= 3))
        self.assertEqual(len(result), 10 * 3)  # 10 samples * 3 trials

        # Verify all samples still present
        unique_samples = result["sample_uuid"].unique()
        self.assertEqual(len(unique_samples), 10)

    def test_filter_trials_error(self):
        """Test error when trial column missing."""
        data_no_trial = self.complete_data.drop(columns=["trial"])
        config = MultiTrialConfig(
            input_file="dummy.csv",
            metrics=["metric1"],
            max_trials=3,
            trials_per_sample=5,
        )

        with self.assertRaises(ValueError) as ctx:
            self.processor._clean_data(data_no_trial, config)
        self.assertIn("'trial' column not found", str(ctx.exception))

    def test_drop_incomplete_trials(self):
        """Test dropping samples with incomplete trials."""
        config = MultiTrialConfig(
            input_file="dummy.csv",
            metrics=["metric1"],
            trials_per_sample=5,
            drop_incomplete_trials=True,
            verbose=True,
        )

        # Capture print output
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            result = self.processor._clean_data(self.incomplete_data, config)

        output = f.getvalue()

        # Check that sample_002 was dropped
        self.assertNotIn("sample_002", result["sample_uuid"].values)
        self.assertEqual(len(result), 9 * 5)  # 9 complete samples * 5 trials

        # Check verbose output
        self.assertIn("Data Cleaning Summary", output)
        self.assertIn("samples with incomplete trials", output)

    def test_drop_incomplete_trials_auto_detect(self):
        """Test auto-detection of expected trials."""
        config = MultiTrialConfig(
            input_file="dummy.csv",
            metrics=["metric1"],
            trials_per_sample=None,  # Will be auto-detected
            drop_incomplete_trials=True,
        )

        # Should auto-detect 5 as the mode
        result = self.processor._clean_data(self.incomplete_data, config)

        # Verify sample_002 was dropped
        self.assertNotIn("sample_002", result["sample_uuid"].values)

    def test_drop_na_all(self):
        """Test dropping all rows with any missing values."""
        config = MultiTrialConfig(
            input_file="dummy.csv",
            metrics=["metric1", "metric2"],
            trials_per_sample=5,
            drop_na=True,
            drop_incomplete_trials=False,
        )

        result = self.processor._clean_data(self.data_with_na, config)

        # Check no NaN values remain
        self.assertFalse(result.isna().any().any())

        # Check correct rows were dropped
        self.assertEqual(len(result), len(self.complete_data) - 4)  # 3 + 1 rows had NaN

    def test_drop_na_specific_columns(self):
        """Test dropping rows with missing values in specific columns."""
        config = MultiTrialConfig(
            input_file="dummy.csv",
            metrics=["metric1", "metric2"],
            trials_per_sample=5,
            drop_na_columns=["metric1"],
            drop_incomplete_trials=False,
        )

        result = self.processor._clean_data(self.data_with_na, config)

        # Check no NaN in metric1
        self.assertFalse(result["metric1"].isna().any())

        # But metric2 can still have NaN
        self.assertTrue(result["metric2"].isna().any())

        # Only 3 rows should be dropped (those with NaN in metric1)
        self.assertEqual(len(result), len(self.complete_data) - 3)

    def test_drop_na_invalid_columns(self):
        """Test error when specified columns don't exist."""
        config = MultiTrialConfig(
            input_file="dummy.csv",
            metrics=["metric1"],
            trials_per_sample=5,
            drop_na_columns=["nonexistent_column"],
        )

        with self.assertRaises(ValueError) as ctx:
            self.processor._clean_data(self.complete_data, config)
        self.assertIn("Columns not found", str(ctx.exception))

    def test_combined_cleaning(self):
        """Test multiple cleaning operations together."""
        # Add more issues to test data
        messy_data = self.incomplete_data.copy()
        messy_data.loc[15:17, "metric1"] = np.nan

        config = MultiTrialConfig(
            input_file="dummy.csv",
            metrics=["metric1", "metric2"],
            trials_per_sample=5,
            max_trials=4,  # Keep only first 4 trials
            drop_incomplete_trials=True,  # Drop sample_002
            drop_na=True,  # Drop rows with NaN
            verbose=True,
        )

        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            result = self.processor._clean_data(messy_data, config)

        output = f.getvalue()

        # Verify all cleaning applied
        self.assertTrue(all(result["trial"] <= 4))  # Max 4 trials
        self.assertNotIn(
            "sample_002", result["sample_uuid"].values
        )  # Incomplete dropped
        self.assertFalse(result.isna().any().any())  # No NaN values

        # Check summary shows all operations
        self.assertIn("Filtered to first 4 trials", output)
        self.assertIn("samples with incomplete trials", output)
        self.assertIn("rows due to any missing value", output)

    def test_empty_after_cleaning_error(self):
        """Test error when all data is cleaned away."""
        # Create data that will be completely filtered
        bad_data = pd.DataFrame(
            {
                "sample_uuid": ["s1"],
                "trial": [10],  # Higher than max_trials
                "metric1": [1.0],
            }
        )

        config = MultiTrialConfig(
            input_file="dummy.csv",
            metrics=["metric1"],
            max_trials=5,
            trials_per_sample=1,
        )

        with self.assertRaises(ValueError) as ctx:
            self.processor._clean_data(bad_data, config)
        self.assertIn("No data remaining after", str(ctx.exception))

    def test_detect_trials_per_sample(self):
        """Test auto-detection of trials per sample."""
        # Test with proper structure
        detected = self.processor._detect_trials_per_sample(self.complete_data)
        self.assertEqual(detected, 5)

        # Test without proper columns
        data_no_structure = self.complete_data.drop(columns=["sample_uuid", "trial"])
        detected = self.processor._detect_trials_per_sample(data_no_structure)
        self.assertIsNone(detected)

    def test_full_extract_with_cleaning(self):
        """Test extract_data method with cleaning enabled."""
        config = MultiTrialConfig(
            input_file="dummy.csv",
            metrics=["metric1", "metric2"],
            trials_per_sample=5,
            drop_na=True,
            drop_incomplete_trials=True,
            verbose=False,
        )

        # Use incomplete data with NaN
        result = self.processor.extract_data(self.data_with_na, config)

        # Check we got metric arrays
        self.assertIn("metric1", result)
        self.assertIn("metric2", result)

        # Arrays should have no NaN (dropna in extract)
        self.assertFalse(np.isnan(result["metric1"]).any())
        self.assertFalse(np.isnan(result["metric2"]).any())

    def test_cleaning_preserves_data_types(self):
        """Test that cleaning preserves column data types."""
        # Add different data types
        mixed_data = self.complete_data.copy()
        mixed_data["category_col"] = pd.Categorical(["A", "B"] * 25)
        mixed_data["int_col"] = range(50)

        config = MultiTrialConfig(
            input_file="dummy.csv",
            metrics=["metric1"],
            trials_per_sample=5,
            max_trials=3,
        )

        result = self.processor._clean_data(mixed_data, config)

        # Check types preserved
        self.assertEqual(result["category_col"].dtype.name, "category")
        self.assertEqual(result["int_col"].dtype, mixed_data["int_col"].dtype)


if __name__ == "__main__":
    unittest.main()
