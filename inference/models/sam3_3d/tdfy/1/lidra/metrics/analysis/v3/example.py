"""Example usage of the v3 MultiTrialProcessor."""

import numpy as np
import pandas as pd
import tempfile
import os

from lidra.metrics.analysis.v3 import MultiTrialConfig, MultiTrialProcessor


def create_example_data():
    """Create example metrics data for demonstration."""
    np.random.seed(42)

    # Simulate 100 samples with 5 trials each
    n_samples = 100
    trials_per_sample = 5
    total_values = n_samples * trials_per_sample

    # Create realistic metric distributions
    data = {
        "chamfer_l1": np.random.exponential(0.05, total_values),
        "chamfer_l2": np.random.exponential(0.02, total_values),
        "iou": np.random.beta(8, 2, total_values),  # Skewed towards high values
        "f1_score": np.random.beta(10, 3, total_values),
    }

    return pd.DataFrame(data)


def main():
    """Demonstrate v3 processor usage."""
    # Create temporary file with example data
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df = create_example_data()
        df.to_csv(f, index=False)
        temp_file = f.name

    try:
        print("=== V3 MultiTrialProcessor Example ===\n")

        # Example 1: Basic usage with console output
        print("Example 1: Basic console output with thresholds")
        print("-" * 50)

        config = MultiTrialConfig(
            input_file=temp_file,
            metrics=["chamfer_l1", "iou"],
            thresholds={"iou": [0.5, 0.7, 0.9]},
            mode="mean",
            trials_per_sample=5,
            statistics=["mean", "std", "p5", "p95"],
        )

        processor = MultiTrialProcessor()
        processor.process(config)

        print("\n" + "=" * 50 + "\n")

        # Example 2: CSV output with different reduction mode
        print("Example 2: CSV output with 'min' reduction mode")
        print("-" * 50)

        output_file = temp_file.replace(".csv", "_output.csv")

        config = MultiTrialConfig(
            input_file=temp_file,
            metrics=["chamfer_l1", "chamfer_l2", "f1_score"],
            mode="min",  # Take minimum value across trials
            trials_per_sample=5,
            statistics=["mean", "std", "min", "max"],
            output_format="csv",
            output_file=output_file,
        )

        processor.process(config)

        # Read and display the CSV output
        print(f"Output saved to: {output_file}")
        print("\nCSV content:")
        with open(output_file, "r") as f:
            print(f.read())

        print("\n" + "=" * 50 + "\n")

        # Example 3: All percentiles
        print("Example 3: Percentile analysis")
        print("-" * 50)

        config = MultiTrialConfig(
            input_file=temp_file,
            metrics=["iou"],
            thresholds={"iou": [0.8]},
            mode="mean",
            trials_per_sample=5,
            statistics=["p5", "p25", "p50", "p75", "p95"],
        )

        processor.process(config)

    finally:
        # Clean up
        os.unlink(temp_file)
        if os.path.exists(output_file):
            os.unlink(output_file)


if __name__ == "__main__":
    main()
